"""Padrões de movimentação do paciente durante a internação.

Entrada: série por minuto com `activity` (contagem do acelerômetro/sensor de
leito ou energia de movimento derivada da pose no vídeo), `position`
(supino, lateral_esq, lateral_dir, sentado, fora_do_leito) e `impact_g`
(pico de aceleração). Também aceita a saída de `video.analyzer` convertida
por `pose_to_activity`.

Detecções
---------
* **Imobilidade prolongada** — sem mudança de decúbito por > 2 h (risco de
  lesão por pressão; protocolos recomendam reposicionar a cada 2 h).
* **Agitação** — atividade muito acima do perfil do próprio paciente para
  aquela hora (z-score robusto), sobretudo à noite (possível delirium).
* **Saída do leito noturna** — risco de queda.
* **Queda** — pico de impacto seguido de imobilidade.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from ..schemas import Finding, Modality, Severity


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Intervalos [ini, fim) onde mask é verdadeiro."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def detect_movement_anomalies(df: pd.DataFrame, immobility_h: float = 2.0, agitation_z: float = 4.0,
                              night=(22, 6), impact_threshold_g: float = 3.0) -> list[Finding]:
    findings: list[Finding] = []
    for pid, g in df.sort_values("timestamp").groupby("patient_id"):
        g = g.reset_index(drop=True)
        ts = pd.to_datetime(g["timestamp"])
        act = g["activity"].astype(float).to_numpy()
        hour = ts.dt.hour.to_numpy()
        is_night = (hour >= night[0]) | (hour < night[1])

        # imobilidade: mesma posição e atividade muito baixa
        low = pd.Series(act).rolling(10, min_periods=1).mean().to_numpy() < 1.0
        same_pos = g["position"].eq(g["position"].shift()).to_numpy()
        for a, b in _runs(low & same_pos):
            dur_h = (b - a) / 60
            if dur_h >= immobility_h:
                sev = Severity.URGENTE if dur_h >= 3 else Severity.ATENCAO
                findings.append(Finding(pid, Modality.MOVEMENT, "imobilidade_prolongada", sev, min(1.0, dur_h / 4),
                                        f"Sem mudança de decúbito por {dur_h:.1f} h (posição: {g['position'][a]}). "
                                        "Risco de lesão por pressão — reposicionar.",
                                        ts[a].to_pydatetime(), ts[b - 1].to_pydatetime(),
                                        evidence={"duration_h": dur_h, "position": g["position"][a]}))

        # agitação: comparado ao perfil do próprio paciente no mesmo período (dia/noite)
        z = np.zeros(len(act))
        for mask in (is_night, ~is_night):
            if not mask.any():
                continue
            ref = act[mask]
            med = np.median(ref)
            mad = max(1.4826 * np.median(np.abs(ref - med)), 1.0)
            z[mask] = (act[mask] - med) / mad
        z_smooth = pd.Series(z).rolling(10, min_periods=1).median().to_numpy()
        for a, b in _runs(z_smooth >= agitation_z):
            if b - a < 5:
                continue
            night_ep = bool(is_night[a:b].mean() > 0.5)
            findings.append(Finding(pid, Modality.MOVEMENT, "agitacao_noturna" if night_ep else "agitacao",
                                    Severity.ATENCAO, min(1.0, float(z_smooth[a:b].max()) / 10),
                                    f"Atividade motora {z_smooth[a:b].max():.0f}σ acima do padrão do paciente "
                                    f"{'à noite' if night_ep else 'no período'} por {b - a} min (avaliar delirium/dor).",
                                    ts[a].to_pydatetime(), ts[b - 1].to_pydatetime(),
                                    evidence={"max_z": float(z_smooth[a:b].max()), "minutes": b - a}))

        # saída do leito noturna
        out_bed = (g["position"] == "fora_do_leito").to_numpy() & is_night
        for a, b in _runs(out_bed):
            findings.append(Finding(pid, Modality.MOVEMENT, "saida_leito_noturna", Severity.ATENCAO, 0.6,
                                    f"Paciente fora do leito às {ts[a]:%H:%M} por {b - a} min — risco de queda.",
                                    ts[a].to_pydatetime(), ts[b - 1].to_pydatetime(), evidence={"minutes": b - a}))

        # queda: impacto + imobilidade nos 5 min seguintes
        if "impact_g" in g:
            for i in np.where(g["impact_g"].to_numpy() >= impact_threshold_g)[0]:
                after = act[i + 1: i + 6]
                if len(after) and after.mean() < 1.0:
                    findings.append(Finding(pid, Modality.MOVEMENT, "queda", Severity.CRITICO, 0.95,
                                            f"Provável queda às {ts[i]:%H:%M} (impacto {g['impact_g'][i]:.1f} g "
                                            "seguido de imobilidade). Verificar o paciente imediatamente.",
                                            ts[i].to_pydatetime(), ts[min(i + 5, len(ts) - 1)].to_pydatetime(),
                                            evidence={"impact_g": float(g["impact_g"][i])}))
    return sorted(findings, key=lambda f: (f.patient_id, f.timestamp))


def pose_to_activity(frame_metrics: pd.DataFrame, patient_id: str, start, resample: str = "1min") -> pd.DataFrame:
    """Converte métricas de pose por frame (vídeo de quarto) em série de atividade por minuto."""
    m = frame_metrics.copy()
    m["timestamp"] = pd.to_datetime(start) + pd.to_timedelta(m["t"], unit="s")
    m = m.set_index("timestamp")
    out = pd.DataFrame({
        "activity": m["motion_energy"].resample(resample).sum(),
        "position": m["posture"].resample(resample).agg(lambda s: s.mode().iat[0] if len(s) else "supino"),
        "impact_g": m.get("hip_drop_speed", pd.Series(0, index=m.index)).resample(resample).max() / 100,
    }).reset_index()
    out.insert(1, "patient_id", patient_id)
    return out
