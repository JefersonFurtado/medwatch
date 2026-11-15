"""Fusão multimodal (late fusion) de achados por paciente.

Cada modalidade gera `Finding`s independentes. A fusão:

1. **Risco agregado do paciente** — combinação probabilística ("OR
   ruidoso") dos achados recentes, ponderados por severidade, confiança da
   modalidade e decaimento temporal:
       risco = 1 − Π (1 − w_mod · w_sev · score · e^(−Δt/τ))
2. **Correlação clínica entre modalidades** — regras que reconhecem
   síndromes quando sinais de fontes diferentes coincidem numa janela de
   tempo (ex.: disartria na voz + crise hipertensiva + déficit motor
   unilateral no vídeo ⇒ suspeita de AVC). O achado fundido herda a maior
   severidade e é escalonado um nível.
3. **Concordância genérica** — ≥ 3 modalidades distintas alteradas na janela
   geram "risco_multimodal" mesmo sem regra específica.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from .schemas import Finding, Modality, Severity

MODALITY_WEIGHT = {Modality.VITALS: 1.0, Modality.MOVEMENT: 0.9, Modality.VIDEO: 0.8, Modality.AUDIO: 0.75,
                   Modality.PRESCRIPTION: 0.6, Modality.FUSION: 1.0}
SEVERITY_WEIGHT = {Severity.INFO: 0.1, Severity.ATENCAO: 0.45, Severity.URGENTE: 0.75, Severity.CRITICO: 0.95}

# regra: nome -> (requisitos [(modalidade, {kinds aceitos ou prefixos})], mínimo de requisitos, descrição,
#                  modalidade obrigatória ou None)
CORRELATION_RULES = {
    "suspeita_avc": ([(Modality.AUDIO, {"disartria"}),
                      (Modality.VITALS, {"crise_hipertensiva"}),
                      (Modality.VIDEO, {"assimetria", "amplitude_reduzida"}),
                      (Modality.AUDIO, {"queixa_neurologico"})], 2,
                     "Sinais neurológicos em múltiplas fontes (voz, PA e/ou déficit motor unilateral).", None),
    "suspeita_sepse": ([(Modality.VITALS, {"deterioracao_clinica", "febre", "taquicardia", "hipotensao"}),
                        (Modality.AUDIO, {"fadiga_vocal", "queixa_infeccao"}),
                        (Modality.PRESCRIPTION, {"escalonamento_abrupto", "alteracoes_frequentes", "duplicidade_terapeutica"}),
                        (Modality.MOVEMENT, {"imobilidade_prolongada"})], 2,
                       "Deterioração hemodinâmica/infecciosa corroborada por voz, prescrição ou mobilidade.",
                       Modality.VITALS),
    "insuficiencia_respiratoria": ([(Modality.VITALS, {"dessaturacao", "taquipneia", "deterioracao_clinica"}),
                                    (Modality.AUDIO, {"dificuldade_respiratoria", "queixa_respiratorio"})], 2,
                                   "Dessaturação/taquipneia com padrão vocal de esforço respiratório.", Modality.VITALS),
    "queda_com_instabilidade": ([(Modality.MOVEMENT, {"queda", "saida_leito_noturna"}),
                                 (Modality.VIDEO, {"queda"}),
                                 (Modality.VITALS, {"dessaturacao", "hipotensao", "deterioracao_clinica", "taquicardia"})], 2,
                                "Queda/saída do leito associada a instabilidade de sinais vitais.", Modality.VITALS),
}


def _matches(f: Finding, modality: str, kinds: set[str]) -> bool:
    if f.modality != modality:
        return False
    parts = set(f.kind.split("+"))
    return bool(parts & kinds) or any(f.kind.startswith(k) for k in kinds)


def patient_risk(findings: list[Finding], at: datetime, tau_h: float = 6.0) -> float:
    prod = 1.0
    for f in findings:
        dt = (at - f.timestamp).total_seconds() / 3600
        if dt < 0:
            continue
        end_dt = (at - (f.end_timestamp or f.timestamp)).total_seconds() / 3600
        decay = np.exp(-max(end_dt, 0) / tau_h)
        p = MODALITY_WEIGHT.get(f.modality, 0.5) * SEVERITY_WEIGHT[f.severity] * max(f.score, 0.3) * decay
        prod *= 1 - min(p, 0.99)
    return 1 - prod


def risk_timeline(findings: list[Finding], start: datetime, end: datetime, step_min: int = 15) -> list[dict]:
    out, t = [], start
    by_patient: dict[str, list[Finding]] = {}
    for f in findings:
        by_patient.setdefault(f.patient_id, []).append(f)
    while t <= end:
        for pid, fs in by_patient.items():
            out.append({"timestamp": t, "patient_id": pid, "risk": round(patient_risk(fs, t), 4)})
        t += timedelta(minutes=step_min)
    return out


def fuse(findings: list[Finding], window_h: float = 12.0) -> list[Finding]:
    """Retorna os achados fundidos (modalidade 'fusao')."""
    fused: list[Finding] = []
    window = timedelta(hours=window_h)
    by_patient: dict[str, list[Finding]] = {}
    for f in findings:
        if f.modality != Modality.FUSION and f.severity >= Severity.ATENCAO:
            by_patient.setdefault(f.patient_id, []).append(f)

    for pid, fs in by_patient.items():
        fs.sort(key=lambda f: f.timestamp)
        # avaliação incremental: o achado fundido nasce no PRIMEIRO instante em que a regra é satisfeita
        for name, (reqs, min_req, desc, required) in CORRELATION_RULES.items():
            for i, now in enumerate(fs):
                win = [f for f in fs[: i + 1] if now.timestamp - f.timestamp <= window]
                hits = _rule_hits(win, reqs)
                mods = {h.modality for h in hits}
                if len({id(h) for h in hits}) >= min_req and len(mods) >= 2 and (required is None or required in mods):
                    later = [f for f in fs[i + 1:] if f.timestamp - now.timestamp <= window]
                    fused.append(_make(pid, name, desc, hits, now.timestamp, _rule_hits(later, reqs)))
                    break
        if any(f.patient_id == pid for f in fused):
            continue
        for i, now in enumerate(fs):  # concordância genérica de ≥3 modalidades
            win = [f for f in fs[: i + 1] if now.timestamp - f.timestamp <= window]
            mods = {f.modality for f in win}
            if len(mods) >= 3:
                top = [max([f for f in win if f.modality == m], key=lambda f: (f.severity, f.score)) for m in mods]
                fused.append(_make(pid, "risco_multimodal",
                                   f"{len(mods)} modalidades com achados na mesma janela de {window_h:.0f} h.", top,
                                   now.timestamp, []))
                break
    return fused


def _rule_hits(win: list[Finding], reqs) -> list[Finding]:
    hits = []
    for mod, kinds in reqs:
        m = [f for f in win if _matches(f, mod, kinds)]
        if m:
            hits.append(max(m, key=lambda f: (f.severity, f.score)))
    return hits


def _make(pid: str, kind: str, desc: str, hits: list[Finding], ts: datetime, later: list[Finding]) -> Finding:
    """`ts` = instante em que a evidência ficou suficiente; `later` = evidências que chegaram depois."""
    sev = Severity(min(int(max(h.severity for h in hits)) + 1, int(Severity.CRITICO)))
    score = 1 - float(np.prod([1 - MODALITY_WEIGHT[h.modality] * h.score for h in hits]))
    fmt = lambda hs: "; ".join(f"{h.modality}: {h.kind.replace('_', ' ')} ({h.timestamp:%d/%m %H:%M})"  # noqa: E731
                               for h in sorted(hs, key=lambda h: h.timestamp))
    later = [h for h in later if all(h is not x for x in hits)]
    msg = f"{desc} Evidências — {fmt(hits)}."
    if later:
        msg += f" Corroborado depois por — {fmt(later)}."
    return Finding(pid, Modality.FUSION, kind, sev, score, msg, ts,
                   evidence={"sources": [h.finding_id for h in hits + later],
                             "modalities": sorted({h.modality for h in hits + later}), "_related": hits + later})
