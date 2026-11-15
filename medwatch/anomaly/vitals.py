"""Detecção de anomalias em séries temporais de sinais vitais.

Combina quatro camadas complementares, todas executáveis **amostra a
amostra** (tempo real):

1. **Regras clínicas (NEWS2)** — escore de alerta precoce do Royal College of
   Physicians, mais regra de crise hipertensiva (PAS ≥ 180 ou PAD ≥ 120).
2. **Desvio do basal do próprio paciente** — z-score robusto (mediana/MAD)
   numa janela móvel de 6 h com defasagem de 30 min; detecta mudanças
   *antes* que os limiares absolutos sejam atingidos (sinal precoce).
3. **Tendência (CUSUM)** — acumula pequenos desvios persistentes (ex.: FC
   subindo 2 bpm por hora), típicos de deterioração lenta/sepse.
4. **Filtro de artefatos** — picos de 1–2 amostras fisiologicamente
   implausíveis são marcados como artefato e não disparam alerta.

Adicionalmente, `multivariate_isolation_forest` aplica Isolation Forest às
variáveis normalizadas por paciente para encontrar combinações incomuns.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..schemas import Finding, Modality, Severity

SIGNALS = ["hr", "sbp", "dbp", "spo2", "rr", "temp"]
MAD_FLOOR = {"hr": 3.0, "sbp": 4.0, "dbp": 3.0, "spo2": 0.7, "rr": 1.2, "temp": 0.15}
PLAUSIBLE_JUMP = {"hr": 35, "sbp": 45, "dbp": 30, "spo2": 8, "rr": 12, "temp": 1.5}
SIGNAL_PT = {"hr": "FC", "sbp": "PAS", "dbp": "PAD", "spo2": "SpO₂", "rr": "FR", "temp": "Temp"}


# --------------------------------------------------------------------------- #
# NEWS2
# --------------------------------------------------------------------------- #
def news2_components(hr=None, sbp=None, spo2=None, rr=None, temp=None) -> dict[str, int]:
    def band(x, bands):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        for upper, score in bands:
            if x <= upper:
                return score
        return bands[-1][1]

    return {
        "rr": band(rr, [(8, 3), (11, 1), (20, 0), (24, 2), (np.inf, 3)]),
        "spo2": band(spo2, [(91, 3), (93, 2), (95, 1), (np.inf, 0)]),
        "sbp": band(sbp, [(90, 3), (100, 2), (110, 1), (219, 0), (np.inf, 3)]),
        "hr": band(hr, [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (np.inf, 3)]),
        "temp": band(temp, [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1), (np.inf, 2)]),
    }


def news2_severity(components: dict[str, int], sbp=None, dbp=None) -> Severity:
    total = sum(components.values())
    if total >= 7:
        sev = Severity.CRITICO
    elif total >= 5:
        sev = Severity.URGENTE
    elif max(components.values()) >= 3:
        sev = Severity.ATENCAO
    else:
        sev = Severity.INFO
    if (sbp is not None and sbp >= 180) or (dbp is not None and dbp >= 120):
        sev = max(sev, Severity.URGENTE)
    return sev


def clinical_conditions(s: dict) -> list[str]:
    c = []
    g = lambda k: s.get(k)  # noqa: E731
    if g("spo2") is not None and g("spo2") <= 92: c.append("dessaturacao")
    if g("hr") is not None and g("hr") >= 111: c.append("taquicardia")
    if g("hr") is not None and g("hr") <= 45: c.append("bradicardia")
    if g("sbp") is not None and g("sbp") <= 100: c.append("hipotensao")
    if (g("sbp") is not None and g("sbp") >= 180) or (g("dbp") is not None and g("dbp") >= 120): c.append("crise_hipertensiva")
    if g("rr") is not None and g("rr") >= 22: c.append("taquipneia")
    if g("temp") is not None and g("temp") >= 38.1: c.append("febre")
    return c


# --------------------------------------------------------------------------- #
# Monitor em tempo real
# --------------------------------------------------------------------------- #
@dataclass
class _Episode:
    start: datetime
    last_abnormal: datetime
    severity: Severity
    conditions: set = field(default_factory=set)
    peak: dict = field(default_factory=dict)
    news2_max: int = 0
    max_z: dict = field(default_factory=dict)
    emitted_severity: Severity | None = None
    finding: Finding | None = None
    episode_id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex[:8])


class StreamingVitalsMonitor:
    """Processa uma amostra por vez e devolve findings novos (abertura/escalonamento).

    Uso:
        mon = StreamingVitalsMonitor("P001")
        for row in stream:
            for finding in mon.update(row):
                dispatcher.dispatch(finding)
    """

    def __init__(self, patient_id: str, baseline_window: int = 360, baseline_lag: int = 30,
                 warmup: int = 120, z_threshold: float = 4.0, persistence: tuple[int, int] = (3, 5),
                 close_after_min: int = 15, cusum_k: float = 2.0, cusum_h: float = 30.0):
        self.patient_id = patient_id
        self.window = baseline_window
        self.lag = baseline_lag
        self.warmup = warmup
        self.z_thr = z_threshold
        self.persist_k, self.persist_n = persistence
        self.close_after = timedelta(minutes=close_after_min)
        self.cusum_k, self.cusum_h = cusum_k, cusum_h
        self.hist = {s: deque(maxlen=baseline_window + baseline_lag) for s in SIGNALS}
        self.recent_flags: deque = deque(maxlen=self.persist_n)
        self.recent_raw: deque = deque(maxlen=3)
        self.cusum = {s: 0.0 for s in SIGNALS}
        self.jump_run = {s: 0 for s in SIGNALS}
        self.episode: _Episode | None = None
        self.closed: list[Finding] = []
        self.artifacts: list[Finding] = []
        self.n = 0

    # -- utilidades ---------------------------------------------------------
    def _baseline(self, s: str) -> tuple[float, float] | None:
        h = [v for v in list(self.hist[s])[: -self.lag or None] if v is not None and not np.isnan(v)]
        if len(h) < self.warmup:
            return None
        med = float(np.median(h))
        mad = float(np.median(np.abs(np.array(h) - med)))
        return med, max(1.4826 * mad, MAD_FLOOR[s])

    def _is_artifact(self, s: str, value: float) -> bool:
        """Salto implausível em relação à mediana recente (será confirmado a seguir)."""
        prev = [r.get(s) for r in self.recent_raw if r.get(s) is not None and not np.isnan(r.get(s))]
        if len(prev) < 2:
            return False
        return abs(value - float(np.median(prev))) > PLAUSIBLE_JUMP[s]

    # -- API ----------------------------------------------------------------
    def update(self, sample: dict) -> list[Finding]:
        ts: datetime = pd.Timestamp(sample["timestamp"]).to_pydatetime()
        vals = {s: (None if sample.get(s) is None or pd.isna(sample.get(s)) else float(sample[s])) for s in SIGNALS}
        self.n += 1
        out: list[Finding] = []

        # 1) filtro de artefatos: salto implausível isolado
        suspicious = []
        for s in SIGNALS:
            if vals[s] is not None and self._is_artifact(s, vals[s]):
                self.jump_run[s] += 1
                # salto mantido por ≥3 amostras é mudança real, não artefato
                if self.jump_run[s] < 3:
                    suspicious.append(s)
            else:
                self.jump_run[s] = 0
        clean = dict(vals)
        for s in suspicious:
            clean[s] = None  # não entra no basal nem nas regras desta amostra
        if suspicious:
            self.artifacts.append(Finding(
                self.patient_id, Modality.VITALS, "possivel_artefato", Severity.INFO, 0.2,
                f"Leitura implausível descartada ({', '.join(SIGNAL_PT[s] + '=' + str(vals[s]) for s in suspicious)})",
                ts, evidence={"raw": {s: vals[s] for s in suspicious}}))
        self.recent_raw.append(clean)

        # 2) regras clínicas
        comp = news2_components(**{k: clean[k] for k in ["hr", "sbp", "spo2", "rr", "temp"]})
        sev_rules = news2_severity(comp, clean["sbp"], clean["dbp"])
        conds = set(clinical_conditions(clean))

        # 3) desvio do basal + CUSUM
        zs = {}
        for s in SIGNALS:
            b = self._baseline(s)
            if b and clean[s] is not None:
                med, scale = b
                z = (clean[s] - med) / scale
                zs[s] = z
                # CUSUM bilateral: acumula apenas o excesso além de k desvios
                self.cusum[s] = max(0.0, self.cusum[s] + abs(z) - self.cusum_k)
        z_flags = {s for s, z in zs.items() if abs(z) >= self.z_thr}
        cusum_flags = {s for s in SIGNALS if self.cusum[s] >= self.cusum_h and abs(zs.get(s, 0)) >= self.cusum_k}
        for s in SIGNALS:
            if clean[s] is not None:
                self.hist[s].append(clean[s])

        sev = sev_rules
        if (z_flags or cusum_flags) and sev < Severity.ATENCAO:
            sev = Severity.ATENCAO
        for s in z_flags | cusum_flags:
            zsign = "alto" if zs.get(s, 0) > 0 else "baixo"
            conds.add(f"desvio_basal_{s}_{zsign}")
        abnormal = sev >= Severity.ATENCAO
        self.recent_flags.append(sev if abnormal else Severity.INFO)

        # 4) persistência: k de n amostras alteradas
        persistent = sum(1 for f in self.recent_flags if f >= Severity.ATENCAO) >= self.persist_k
        if persistent and abnormal:
            # severidade efetiva = 2ª maior das últimas n (evita escalar por 1 amostra)
            eff = sorted(self.recent_flags, reverse=True)[min(1, len(self.recent_flags) - 1)]
            eff = max(eff, Severity.ATENCAO)
            total = sum(comp.values())
            if self.episode is None:
                self.episode = _Episode(start=ts, last_abnormal=ts, severity=eff)
            ep = self.episode
            ep.last_abnormal = ts
            ep.severity = max(ep.severity, eff)
            ep.conditions |= conds
            ep.news2_max = max(ep.news2_max, total)
            for s in SIGNALS:
                if clean[s] is not None:
                    lo_hi = ep.peak.setdefault(s, [clean[s], clean[s]])
                    lo_hi[0], lo_hi[1] = min(lo_hi[0], clean[s]), max(lo_hi[1], clean[s])
                if s in zs:
                    ep.max_z[s] = max(ep.max_z.get(s, 0.0), abs(zs[s]))
            if ep.emitted_severity is None or ep.severity > ep.emitted_severity:
                ep.emitted_severity = ep.severity
                ep.finding = self._episode_finding(ep, ts, vals)
                out.append(ep.finding)
        elif self.episode and ts - self.episode.last_abnormal > self.close_after:
            self._close(self.episode.last_abnormal)
        return out

    def flush(self) -> list[Finding]:
        if self.episode:
            self._close(self.episode.last_abnormal)
        return self.closed

    def _close(self, end: datetime) -> None:
        ep = self.episode
        f = self._episode_finding(ep, ep.last_abnormal, {})
        f.end_timestamp = end
        f.finding_id = ep.finding.finding_id if ep.finding else f.finding_id
        self.closed.append(f)
        self.episode = None
        self.cusum = {s: 0.0 for s in SIGNALS}

    def _episode_finding(self, ep: _Episode, ts: datetime, current: dict) -> Finding:
        clinical = sorted(c for c in ep.conditions if not c.startswith("desvio_basal"))
        drift = sorted(c for c in ep.conditions if c.startswith("desvio_basal"))
        if len(clinical) >= 3:
            kind = "deterioracao_clinica"
        elif clinical:
            kind = "+".join(clinical)
        else:
            kind = "desvio_do_basal"
        parts = []
        for s in SIGNALS:
            if s in ep.peak:
                lo, hi = ep.peak[s]
                parts.append(f"{SIGNAL_PT[s]} {lo:g}–{hi:g}")
        msg = (f"{kind.replace('_', ' ').capitalize()} desde {ep.start:%d/%m %H:%M}. NEWS2 máx {ep.news2_max}. "
               + "; ".join(parts))
        if drift and not clinical:
            msg = "Sinal precoce: desvio persistente do basal do paciente (" + ", ".join(
                d.replace("desvio_basal_", "").replace("_", " ") for d in drift) + "). " + msg
        score = min(1.0, 0.25 + 0.1 * ep.news2_max + 0.04 * max(ep.max_z.values(), default=0))
        return Finding(self.patient_id, Modality.VITALS, kind, ep.severity, score, msg, ep.start,
                       end_timestamp=None, evidence={"episode_id": ep.episode_id, "news2_max": ep.news2_max, "conditions": sorted(ep.conditions),
                                                     "range": ep.peak, "max_abs_z": ep.max_z,
                                                     "detected_at": ts.isoformat()})


# --------------------------------------------------------------------------- #
# Execução em lote (reproduz o streaming)
# --------------------------------------------------------------------------- #
def detect_vitals_anomalies(df: pd.DataFrame, **monitor_kwargs) -> tuple[list[Finding], list[Finding], pd.DataFrame]:
    """Retorna (episódios, alertas emitidos em tempo real, df anotado)."""
    episodes, realtime, annotated = [], [], []
    for pid, g in df.sort_values("timestamp").groupby("patient_id"):
        mon = StreamingVitalsMonitor(pid, **monitor_kwargs)
        g = g.copy()
        news, flags = [], []
        for row in g.to_dict("records"):
            new = mon.update(row)
            realtime.extend(new)
            comp = news2_components(**{k: (None if pd.isna(row.get(k)) else row.get(k)) for k in ["hr", "sbp", "spo2", "rr", "temp"]})
            news.append(sum(comp.values()))
            flags.append(int(mon.episode is not None))
        episodes.extend(mon.flush())
        episodes.extend(mon.artifacts)
        g["news2"] = news
        g["in_episode"] = flags
        annotated.append(g)
    ann = pd.concat(annotated, ignore_index=True)
    ann = multivariate_isolation_forest(ann)
    return episodes, realtime, ann


def multivariate_isolation_forest(df: pd.DataFrame, contamination: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """Isolation Forest sobre valores normalizados por paciente + inclinações de 15 min."""
    from sklearn.ensemble import IsolationForest

    feats = []
    for _, g in df.groupby("patient_id"):
        x = g[SIGNALS].astype(float).interpolate(limit_direction="both")
        med = x.iloc[: min(len(x), 360)].median()
        mad = (x.iloc[: min(len(x), 360)] - med).abs().median().clip(lower=pd.Series(MAD_FLOOR))
        z = (x - med) / (1.4826 * mad)
        slope = x.diff(15).fillna(0) / pd.Series(MAD_FLOOR)
        f = pd.concat([z.add_prefix("z_"), slope.add_prefix("d_")], axis=1)
        f.index = g.index
        feats.append(f)
    F = pd.concat(feats).loc[df.index].fillna(0)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    model.fit(F.values)
    out = df.copy()
    out["iforest_score"] = -model.score_samples(F.values)
    out["iforest_anomaly"] = (model.predict(F.values) == -1).astype(int)
    return out


# --------------------------------------------------------------------------- #
# Avaliação contra ground truth
# --------------------------------------------------------------------------- #
def evaluate_event_detection(findings: list[Finding], events: list, tolerance_min: int = 30,
                             min_severity: Severity = Severity.ATENCAO) -> dict:
    """Métricas por evento: recall, precisão (episódios com sobreposição), tempo até detecção."""
    alerts = [f for f in findings if f.severity >= min_severity]
    tol = timedelta(minutes=tolerance_min)
    detected, delays = 0, []
    matched_ids = set()
    for ev in events:
        hits = [f for f in alerts if f.patient_id == ev.patient_id
                and f.timestamp <= ev.end and (f.end_timestamp or f.timestamp) >= ev.start - tol]
        if hits:
            detected += 1
            first = min(hits, key=lambda f: f.timestamp)
            delays.append((first.timestamp - ev.start).total_seconds() / 60)
            matched_ids |= {f.finding_id for f in hits}
    tp_alerts = sum(1 for f in alerts if f.finding_id in matched_ids)
    recall = detected / len(events) if events else 1.0
    precision = tp_alerts / len(alerts) if alerts else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"events": len(events), "detected": detected, "alerts": len(alerts), "true_alerts": tp_alerts,
            "recall": round(recall, 3), "precision": round(precision, 3), "f1": round(f1, 3),
            "mean_detection_delay_min": round(float(np.mean(delays)), 1) if delays else None}
