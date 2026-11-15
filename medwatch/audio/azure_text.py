"""Análise do texto transcrito com **Azure AI Language (Text Analytics)**.

Recursos utilizados:
* `analyze_sentiment` (com opinion mining) — sentimento geral e por frase;
* `extract_key_phrases` — principais queixas;
* `begin_analyze_healthcare_entities` (Text Analytics for Health) — sintomas,
  medicamentos, partes do corpo e negação ("não tenho febre");
* `recognize_pii_entities` — texto anonimizado para armazenar/logar (LGPD).

Sobre o resultado do Azure é aplicada uma camada de **termos críticos**
(léxico clínico pt-BR) que mapeia o que foi dito para as condições
monitoradas: fadiga, dificuldade respiratória, sinais neurológicos (ex.:
disartria/AVC), dor torácica, infecção. Sem credenciais, essa mesma camada
e um léxico simples de sentimento funcionam offline.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from ..config import AzureSettings

log = logging.getLogger(__name__)

# termo (sem acento, minúsculo) -> (categoria, peso 0..1)
CRITICAL_TERMS: dict[str, tuple[str, float]] = {
    # respiratório
    "falta de ar": ("respiratorio", 0.9), "dispneia": ("respiratorio", 0.9), "sem ar": ("respiratorio", 0.8),
    "respiracao curta": ("respiratorio", 0.8), "chiado": ("respiratorio", 0.6), "cansaco para respirar": ("respiratorio", 0.8),
    "tosse": ("respiratorio", 0.4), "sufocando": ("respiratorio", 1.0),
    # cardiovascular
    "dor no peito": ("cardiovascular", 1.0), "aperto no peito": ("cardiovascular", 0.9), "palpitacao": ("cardiovascular", 0.6),
    "desmaio": ("cardiovascular", 0.8), "desmaiei": ("cardiovascular", 0.8),
    # neurológico (inclui sinais de AVC / disartria)
    "fala enrolada": ("neurologico", 1.0), "fala arrastada": ("neurologico", 1.0), "dificuldade para falar": ("neurologico", 1.0),
    "dormente": ("neurologico", 0.8), "dormencia": ("neurologico", 0.8), "formigamento": ("neurologico", 0.6),
    "boca torta": ("neurologico", 1.0), "fraqueza de um lado": ("neurologico", 1.0), "confuso": ("neurologico", 0.7),
    "tontura": ("neurologico", 0.5), "dor de cabeca forte": ("neurologico", 0.7), "visao turva": ("neurologico", 0.6),
    # fadiga / estado geral
    "cansado": ("fadiga", 0.6), "cansada": ("fadiga", 0.6), "cansaco": ("fadiga", 0.6), "exausto": ("fadiga", 0.8),
    "sem forca": ("fadiga", 0.7), "fraqueza": ("fadiga", 0.6), "sonolento": ("fadiga", 0.5), "sem energia": ("fadiga", 0.6),
    # infecção
    "febre": ("infeccao", 0.7), "calafrio": ("infeccao", 0.7), "calafrios": ("infeccao", 0.7), "suor frio": ("infeccao", 0.6),
    # dor / outros
    "dor forte": ("dor", 0.6), "dor insuportavel": ("dor", 0.9), "sangramento": ("hemorragico", 0.9),
    "vomito com sangue": ("hemorragico", 1.0), "nao consigo": ("funcional", 0.3),
}
POS_WORDS = {"bem", "melhor", "otimo", "tranquilo", "passou", "melhorou", "consigo"}
NEG_WORDS = {"dor", "mal", "pior", "cansado", "falta", "medo", "fraco", "tontura", "enrolada", "aperto", "febre", "dormente"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


@dataclass
class TextAnalysis:
    source: str
    sentiment: str                       # positive | neutral | negative | mixed
    sentiment_scores: dict
    key_phrases: list[str] = field(default_factory=list)
    health_entities: list[dict] = field(default_factory=list)
    critical_terms: list[dict] = field(default_factory=list)
    category_scores: dict = field(default_factory=dict)
    redacted_text: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def find_critical_terms(text: str) -> list[dict]:
    t = _norm(text)
    found = []
    for term, (cat, w) in CRITICAL_TERMS.items():
        for m in re.finditer(r"\b" + re.escape(term) + r"\b", t):
            window = t[max(0, m.start() - 20): m.start()]
            # negação até 1 palavra antes do termo: "não tenho febre", "sem febre"
            negated = bool(re.search(r"\b(nao|nunca|nem|sem|nenhum|nenhuma)\s+(\w+\s+)?$", window))
            found.append({"term": term, "category": cat, "weight": w, "negated": negated, "pos": m.start()})
    # remove sobreposições (mantém termo mais longo)
    found.sort(key=lambda d: (d["pos"], -len(d["term"])))
    out, last_end = [], -1
    for d in found:
        if d["pos"] >= last_end:
            out.append(d)
            last_end = d["pos"] + len(d["term"])
    return out


def _category_scores(terms: list[dict], entities: list[dict]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for d in terms:
        if d["negated"]:
            continue
        p = scores.get(d["category"], 0.0)
        scores[d["category"]] = 1 - (1 - p) * (1 - d["weight"])  # OR probabilístico
    for e in entities:  # sintomas do TA4H reforçam a categoria correspondente
        if e.get("category") in ("SymptomOrSign", "Diagnosis") and not e.get("negated"):
            for d in find_critical_terms(e["text"]):
                p = scores.get(d["category"], 0.0)
                scores[d["category"]] = 1 - (1 - p) * (1 - 0.3)
    return {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])}


class AzureTextAnalyzer:
    def __init__(self, settings: AzureSettings):
        self.settings = settings
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential

            self._client = TextAnalyticsClient(self.settings.language_endpoint,
                                               AzureKeyCredential(self.settings.language_key))
        return self._client

    def analyze(self, text: str) -> TextAnalysis:
        terms = find_critical_terms(text)
        if not text.strip():
            return TextAnalysis("none", "neutral", {}, critical_terms=terms)
        if self.settings.language_enabled:
            try:
                return self._azure(text, terms)
            except Exception as exc:
                log.warning("Azure Language falhou (%s); usando análise offline.", exc)
        return self._offline(text, terms)

    def _azure(self, text: str, terms: list[dict]) -> TextAnalysis:
        lang = self.settings.language_code
        docs = [text]
        sent = self.client.analyze_sentiment(docs, language=lang, show_opinion_mining=True)[0]
        kp = self.client.extract_key_phrases(docs, language=lang)[0]
        pii = self.client.recognize_pii_entities(docs, language=lang)[0]
        entities = []
        try:
            poller = self.client.begin_analyze_healthcare_entities(docs, language=self.settings.health_language)
            for doc in poller.result():
                if doc.is_error:
                    continue
                for e in doc.entities:
                    neg = bool(e.assertion and getattr(e.assertion, "certainty", None) in ("negative", "negativePossible"))
                    entities.append({"text": e.text, "category": e.category, "confidence": e.confidence_score,
                                     "negated": neg, "normalized": e.normalized_text})
        except Exception as exc:  # TA4H pode não estar disponível na região/idioma
            log.info("Text Analytics for Health indisponível: %s", exc)
        scores = {"positive": sent.confidence_scores.positive, "neutral": sent.confidence_scores.neutral,
                  "negative": sent.confidence_scores.negative}
        return TextAnalysis("azure", sent.sentiment, scores, list(kp.key_phrases) if not kp.is_error else [],
                            entities, terms, _category_scores(terms, entities),
                            pii.redacted_text if not pii.is_error else None)

    @staticmethod
    def _offline(text: str, terms: list[dict]) -> TextAnalysis:
        tokens = re.findall(r"\w+", _norm(text))
        pos = sum(t in POS_WORDS for t in tokens)
        neg = sum(t in NEG_WORDS for t in tokens) + sum(1 for d in terms if not d["negated"])
        tot = pos + neg or 1
        scores = {"positive": round(pos / tot, 3), "negative": round(neg / tot, 3), "neutral": 0.0}
        if pos == neg == 0:
            label, scores["neutral"] = "neutral", 1.0
        elif pos and neg and min(pos, neg) / max(pos, neg) > 0.6:
            label = "mixed"
        else:
            label = "positive" if pos > neg else "negative"
        kp = sorted({d["term"] for d in terms if not d["negated"]})
        return TextAnalysis("offline", label, scores, kp, [], terms, _category_scores(terms, []), None)
