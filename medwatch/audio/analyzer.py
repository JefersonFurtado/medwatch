"""Análise de áudio de consultas: biomarcadores vocais + Azure Speech + Azure Language.

Fluxo:  áudio ─┬─> features acústicas (librosa/Praat) ─┐
               └─> Azure Speech to Text ─> Azure Language ─┴─> escores por condição ─> Findings

Os limiares (`THRESHOLDS`) são pontos de partida baseados na literatura de
análise acústica (ex.: jitter > 1,04 % e HNR < 20 dB como valores
alterados no Praat) e devem ser recalibrados com gravações de referência do
próprio hospital (mesmo microfone/sala), como descrito no relatório.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import AzureSettings
from ..schemas import Finding, Modality, Severity
from .azure_speech import AzureSpeechTranscriber, Transcript
from .azure_text import AzureTextAnalyzer, TextAnalysis
from .features import VoiceFeatures, extract_voice_features

# feature: (valor normal, valor patológico) — o escore sobe linearmente entre os dois
THRESHOLDS = {
    "jitter_local_pct": (1.04, 3.0),
    "shimmer_local_pct": (9.0, 15.0),
    "hnr_db": (20.0, 10.0),
    "syllable_rate": (4.0, 2.5),
    "mean_pause_s": (0.40, 0.90),
    "pause_ratio": (0.15, 0.30),
    "breaths_per_min": (5.0, 20.0),
    "energy_slope_pct": (0.0, -12.0),
    "f0_slope_pct": (0.0, -8.0),
}

CONDITIONS = {
    # condição: (pesos das features, categoria de texto relacionada, rótulo)
    "disartria": ({"jitter_local_pct": .30, "shimmer_local_pct": .15, "syllable_rate": .25, "mean_pause_s": .20, "hnr_db": .10},
                  "neurologico", "Alteração articulatória compatível com disartria"),
    "dificuldade_respiratoria": ({"breaths_per_min": .50, "hnr_db": .30, "pause_ratio": .20},
                                 "respiratorio", "Padrão vocal de esforço respiratório"),
    "fadiga_vocal": ({"energy_slope_pct": .45, "f0_slope_pct": .35, "syllable_rate": .20},
                     "fadiga", "Queda progressiva de energia/F0 compatível com fadiga"),
}


def _ramp(x: float, normal: float, patho: float) -> float:
    if np.isnan(x):
        return 0.0
    return float(np.clip((x - normal) / (patho - normal), 0, 1))


@dataclass
class AudioReport:
    patient_id: str
    audio_path: str
    features: VoiceFeatures
    transcript: Transcript
    text: TextAnalysis
    indicator_scores: dict
    condition_scores: dict

    def to_dict(self) -> dict:
        return {"patient_id": self.patient_id, "audio_path": self.audio_path,
                "features": self.features.to_dict(), "transcript": self.transcript.to_dict(),
                "text_analysis": self.text.to_dict(), "indicator_scores": self.indicator_scores,
                "condition_scores": self.condition_scores}


class AudioAnalyzer:
    def __init__(self, azure: AzureSettings):
        self.transcriber = AzureSpeechTranscriber(azure)
        self.text_analyzer = AzureTextAnalyzer(azure)

    def analyze(self, audio_path: str | Path, patient_id: str, timestamp: datetime | None = None
                ) -> tuple[AudioReport, list[Finding]]:
        timestamp = timestamp or datetime.now()
        feats = extract_voice_features(str(audio_path))
        transcript = self.transcriber.transcribe(audio_path)
        text = self.text_analyzer.analyze(transcript.text)

        fd = feats.to_dict()
        if transcript.words_per_minute:  # taxa de fala real (palavras) do Azure complementa a acústica
            fd["words_per_minute"] = transcript.words_per_minute
        indicators = {k: round(_ramp(float(fd[k]), *v), 3) for k, v in THRESHOLDS.items()}

        cond_scores, findings = {}, []
        for cond, (weights, text_cat, label) in CONDITIONS.items():
            acoustic = sum(w * indicators[f] for f, w in weights.items())
            textual = text.category_scores.get(text_cat, 0.0)
            fused = 1 - (1 - acoustic) * (1 - 0.6 * textual)
            cond_scores[cond] = {"acoustic": round(acoustic, 3), "text": round(textual, 3), "fused": round(fused, 3)}
            if fused < 0.5:
                continue
            sev = Severity.URGENTE if fused >= 0.75 else Severity.ATENCAO
            extra = ""
            if cond == "disartria" and textual >= 0.9 and acoustic >= 0.5:
                sev, extra = Severity.CRITICO, " Relato de sinais neurológicos agudos — considerar protocolo de AVC."
            top = sorted(weights, key=lambda f: -indicators[f] * weights[f])[:3]
            evid = {f: fd[f] for f in top}
            terms = [d["term"] for d in text.critical_terms if d["category"] == text_cat and not d["negated"]]
            msg = (f"{label} (escore {fused:.2f}; acústico {acoustic:.2f}, relato {textual:.2f}). "
                   f"Principais marcadores: " + ", ".join(f"{k}={fd[k]:.2f}" for k in top)
                   + (f". Termos citados: {', '.join(terms)}" if terms else "") + "." + extra)
            findings.append(Finding(patient_id, Modality.AUDIO, cond, sev, fused, msg, timestamp,
                                    evidence={"markers": evid, "terms": terms, "sentiment": text.sentiment}))

        # queixas críticas sem correlato acústico (ex.: dor torácica)
        covered = {c[1] for c in CONDITIONS.values()}
        for cat, sc in text.category_scores.items():
            if cat in covered or sc < 0.6:
                continue
            terms = [d["term"] for d in text.critical_terms if d["category"] == cat and not d["negated"]]
            sev = Severity.URGENTE if cat in ("cardiovascular", "hemorragico") and sc >= 0.85 else Severity.ATENCAO
            findings.append(Finding(patient_id, Modality.AUDIO, f"queixa_{cat}", sev, sc,
                                    f"Queixa relatada na consulta: {', '.join(terms)} (categoria {cat}).",
                                    timestamp, evidence={"terms": terms, "sentiment": text.sentiment}))

        report = AudioReport(patient_id, str(audio_path), feats, transcript, text, indicators, cond_scores)
        return report, findings
