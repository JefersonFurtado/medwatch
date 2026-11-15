"""Detecção de alterações inesperadas na evolução das prescrições.

Abordagem híbrida:

* **Regras farmacológicas** (base de conhecimento simplificada, *não*
  substitui a validação farmacêutica): dose acima da máxima diária,
  interações graves, suspensão abrupta de fármacos que exigem desmame,
  duplicidade terapêutica.
* **Regras de evolução**: escalonamento de dose > 100 % em < 6 h, muitas
  alterações em janela curta (instabilidade do tratamento).
* **Estatística**: z-score robusto da variação relativa de dose frente ao
  histórico de variações do hospital (todas as prescrições) — capta
  mudanças incomuns mesmo sem regra explícita.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ..schemas import Finding, Modality, Severity

# Dose máxima diária (mg/dia) — valores de referência simplificados para demo
MAX_DAILY_DOSE_MG = {
    "dipirona": 4000, "paracetamol": 4000, "captopril": 150, "losartana": 100, "metoprolol": 400,
    "varfarina": 10, "enoxaparina": 200, "omeprazol": 80, "prednisona": 80, "ceftriaxona": 4000,
    "meropenem": 6000, "salbutamol": 32, "acido acetilsalicilico": 4000, "noradrenalina": 30,
    "morfina": 200, "furosemida": 600, "insulina regular": 300,
}
DRUG_CLASS = {
    "ceftriaxona": "antibiotico_betalactamico", "meropenem": "antibiotico_betalactamico",
    "dipirona": "analgesico", "paracetamol": "analgesico", "captopril": "ieca", "enalapril": "ieca",
    "losartana": "bra", "metoprolol": "betabloqueador", "atenolol": "betabloqueador",
    "varfarina": "anticoagulante", "enoxaparina": "anticoagulante", "acido acetilsalicilico": "antiagregante",
    "noradrenalina": "vasopressor", "prednisona": "corticoide", "salbutamol": "broncodilatador",
}
# Interações graves (pares não ordenados)
INTERACTIONS = {
    frozenset({"varfarina", "acido acetilsalicilico"}): "Risco aumentado de sangramento (anticoagulante + antiagregante).",
    frozenset({"varfarina", "enoxaparina"}): "Dupla anticoagulação — risco hemorrágico.",
    frozenset({"captopril", "losartana"}): "Duplo bloqueio do SRAA — hipercalemia/insuficiência renal.",
    frozenset({"metoprolol", "verapamil"}): "Bradicardia/bloqueio AV.",
}
REQUIRES_TAPER = {"metoprolol", "atenolol", "prednisona", "clonidina", "propranolol"}


def _active_regimen(rx: pd.DataFrame, at) -> dict[str, dict]:
    """Prescrição vigente por fármaco no instante `at` (última linha ≤ at com dose > 0)."""
    cur = {}
    for row in rx[rx["timestamp"] <= at].sort_values("timestamp").to_dict("records"):
        if row["dose"] and row["doses_per_day"]:
            cur[row["drug"]] = row
        else:
            cur.pop(row["drug"], None)
    return cur


def detect_prescription_anomalies(rx: pd.DataFrame, change_window_h: int = 6, max_changes: int = 4) -> list[Finding]:
    rx = rx.copy()
    rx["timestamp"] = pd.to_datetime(rx["timestamp"])
    rx["drug"] = rx["drug"].str.lower().str.strip()
    rx["daily_dose"] = rx["dose"].astype(float) * rx["doses_per_day"].astype(float)
    rx = rx.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    findings: list[Finding] = []

    # variação relativa de dose por fármaco (todas as prescrições -> distribuição de referência)
    rx["prev_daily"] = rx.groupby(["patient_id", "drug"])["daily_dose"].shift()
    rel = (rx["daily_dose"] - rx["prev_daily"]) / rx["prev_daily"].replace(0, np.nan)
    rx["rel_change"] = rel
    ref = rel.dropna()
    med = ref.median() if len(ref) else 0.0
    mad = max(1.4826 * (ref - med).abs().median(), 0.25) if len(ref) else 0.25

    for pid, g in rx.groupby("patient_id"):
        first_ts = g["timestamp"].min()
        for row in g.to_dict("records"):
            ts = row["timestamp"].to_pydatetime()
            drug, daily = row["drug"], row["daily_dose"]
            is_new_order = row["timestamp"] > first_ts

            # 1) dose acima da máxima
            mx = MAX_DAILY_DOSE_MG.get(drug)
            if mx and row["unit"] == "mg" and daily > mx:
                findings.append(Finding(pid, Modality.PRESCRIPTION, "dose_acima_maxima", Severity.URGENTE,
                                        min(1.0, daily / mx / 2),
                                        f"{drug}: {daily:g} mg/dia excede a máxima de referência ({mx} mg/dia).",
                                        ts, evidence={"drug": drug, "daily_dose": daily, "max": mx,
                                                      "prescriber": row["prescriber"]}))

            # 2) suspensão abrupta de fármaco que exige desmame
            if daily == 0 and drug in REQUIRES_TAPER and row["prev_daily"] and row["prev_daily"] > 0:
                findings.append(Finding(pid, Modality.PRESCRIPTION, "suspensao_abrupta", Severity.ATENCAO, 0.6,
                                        f"{drug} suspenso abruptamente (de {row['prev_daily']:g} mg/dia para 0) — "
                                        "risco de efeito rebote; avaliar desmame.", ts,
                                        evidence={"drug": drug, "previous_daily": row["prev_daily"]}))

            # 3) escalonamento abrupto / variação estatisticamente incomum
            if row["prev_daily"] and row["prev_daily"] > 0 and daily > 0:
                prev_rows = g[(g["drug"] == drug) & (g["timestamp"] < row["timestamp"])]
                dt_h = (row["timestamp"] - prev_rows["timestamp"].max()).total_seconds() / 3600
                rc = row["rel_change"]
                z = (rc - med) / mad
                if (rc >= 1.0 and dt_h <= change_window_h) or abs(z) >= 4:
                    sev = Severity.URGENTE if rc >= 2 else Severity.ATENCAO
                    findings.append(Finding(pid, Modality.PRESCRIPTION, "escalonamento_abrupto", sev,
                                            min(1.0, abs(rc) / 4),
                                            f"{drug}: dose diária de {row['prev_daily']:g} → {daily:g} "
                                            f"({rc:+.0%}) em {dt_h:.1f} h.", ts,
                                            evidence={"drug": drug, "rel_change": rc, "robust_z": z, "hours": dt_h}))

            if not is_new_order or daily == 0:
                continue
            active = _active_regimen(g, row["timestamp"])
            # 4) interações com o regime vigente
            for other in active:
                if other == drug:
                    continue
                msg = INTERACTIONS.get(frozenset({drug, other}))
                if msg:
                    findings.append(Finding(pid, Modality.PRESCRIPTION, "interacao_medicamentosa", Severity.URGENTE, 0.8,
                                            f"{drug} + {other}: {msg}", ts, evidence={"drugs": [drug, other]}))
            # 5) duplicidade terapêutica
            cls = DRUG_CLASS.get(drug)
            dups = [o for o in active if o != drug and cls and DRUG_CLASS.get(o) == cls]
            if dups:
                findings.append(Finding(pid, Modality.PRESCRIPTION, "duplicidade_terapeutica", Severity.ATENCAO, 0.5,
                                        f"{drug} e {', '.join(dups)} são da mesma classe ({cls}).", ts,
                                        evidence={"drugs": [drug, *dups], "class": cls}))

        # 6) instabilidade: muitas alterações numa janela curta
        times = g.loc[g["timestamp"] > first_ts, "timestamp"].sort_values().tolist()
        flagged_until = None
        for i, t0 in enumerate(times):
            if flagged_until is not None and t0 <= flagged_until:
                continue
            window = [t for t in times[i:] if t - t0 <= timedelta(hours=change_window_h)]
            if len(window) >= max_changes:
                findings.append(Finding(pid, Modality.PRESCRIPTION, "alteracoes_frequentes", Severity.ATENCAO, 0.5,
                                        f"{len(window)} alterações de prescrição em {change_window_h} h — "
                                        "tratamento instável; correlacionar com quadro clínico.",
                                        t0.to_pydatetime(), end_timestamp=window[-1].to_pydatetime(),
                                        evidence={"n_changes": len(window)}))
                flagged_until = window[-1]
    return sorted(findings, key=lambda f: (f.patient_id, f.timestamp))
