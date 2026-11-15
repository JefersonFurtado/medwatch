"""Análise de movimento em vídeos clínicos (fisioterapia / cirurgia / quarto).

Pipeline
--------
vídeo ─> pose por frame (YOLOv8-pose | OpenPose) ─> métricas biomecânicas
      ─> segmentação de repetições ─> regras do exercício + Isolation Forest
      ─> eventos (desvios) ─> relatório automático + Findings
      └> YOLOv8 detecção ─> zonas críticas / contagem de instrumentais

Métricas por frame: ângulos de abdução do ombro (E/D), flexão do cotovelo e
do joelho, inclinação lateral do tronco, postura (em pé / sentado /
deitado), energia de movimento, velocidade vertical do quadril
(normalizada pela altura do corpo) e índice de tremor dos punhos.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..schemas import Finding, Modality, Severity
from .pose import KP, SKELETON, PoseEstimator

log = logging.getLogger(__name__)

EXERCISE_PROFILES = {
    "abducao_ombro": {"angle": "shoulder_abd", "min_peak": 140.0, "max_asymmetry": 25.0, "max_trunk_lean": 10.0,
                      "rep_prominence": 40.0, "rep_min_interval_s": 1.5,
                      "descricao": "Abdução bilateral de ombro em pé"},
    "agachamento": {"angle": "knee_flex", "min_peak": 80.0, "max_asymmetry": 20.0, "max_trunk_lean": 15.0,
                    "rep_prominence": 30.0, "rep_min_interval_s": 1.5, "descricao": "Agachamento livre"},
    "flexao_cotovelo": {"angle": "elbow_flex", "min_peak": 110.0, "max_asymmetry": 25.0, "max_trunk_lean": 10.0,
                        "rep_prominence": 40.0, "rep_min_interval_s": 1.2, "descricao": "Flexão de cotovelo"},
}


# --------------------------------------------------------------------------- #
# Geometria
# --------------------------------------------------------------------------- #
def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ângulo ABC (graus) para arrays (T,2)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def clean_keypoints(kp: np.ndarray, conf_thr: float = 0.3) -> np.ndarray:
    """Pontos com baixa confiança viram NaN e são interpolados no tempo."""
    kp = kp.astype(float).copy()
    kp[kp[..., 2] < conf_thr, :2] = np.nan
    T = kp.shape[0]
    for j in range(kp.shape[1]):
        for d in range(2):
            s = pd.Series(kp[:, j, d]).interpolate(limit_direction="both", limit=int(max(T * 0.1, 5)))
            kp[:, j, d] = s.to_numpy()
    return kp


def frame_metrics(kp: np.ndarray, fps: float) -> pd.DataFrame:
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)
    visible = (kp[..., 2] > 0.3).sum(1) >= 6
    kp = clean_keypoints(kp)
    P = lambda n: kp[:, KP[n], :2]  # noqa: E731
    mid_sh = (P("l_shoulder") + P("r_shoulder")) / 2
    mid_hip = (P("l_hip") + P("r_hip")) / 2
    mid_ank = (P("l_ankle") + P("r_ankle")) / 2
    body_h = np.nanmedian(np.linalg.norm(mid_sh - mid_ank, axis=1)[: max(int(fps * 3), 5)])
    body_h = body_h if body_h and not np.isnan(body_h) else 200.0

    df = pd.DataFrame({"t": np.arange(len(kp)) / fps})
    for side in ("l", "r"):
        df[f"shoulder_abd_{side}"] = _angle(P(f"{side}_hip"), P(f"{side}_shoulder"), P(f"{side}_elbow"))
        df[f"elbow_flex_{side}"] = 180 - _angle(P(f"{side}_shoulder"), P(f"{side}_elbow"), P(f"{side}_wrist"))
        df[f"knee_flex_{side}"] = 180 - _angle(P(f"{side}_hip"), P(f"{side}_knee"), P(f"{side}_ankle"))
    # inclinação do tronco: vetor quadril->ombro vs vertical (y para cima = -y da imagem)
    v = mid_sh - mid_hip
    df["trunk_angle"] = np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1]))     # 0 = ereto, 90 = horizontal
    df["trunk_lean"] = np.degrees(np.arctan2(v[:, 0], -v[:, 1]))              # lateral com sinal
    df["hip_y_norm"] = mid_hip[:, 1] / body_h
    dt = 1 / fps
    df["hip_drop_speed"] = pd.Series(df["hip_y_norm"]).diff().fillna(0).rolling(3, min_periods=1).mean() / dt
    vel = np.linalg.norm(np.diff(kp[:, :, :2], axis=0, prepend=kp[:1, :, :2]), axis=-1) / body_h / dt
    df["motion_energy"] = np.nanmean(vel, axis=1)
    # tremor: energia de alta frequência das trajetórias dos punhos — magnitude da 2ª diferença
    # (movimento voluntário suave tem 2ª diferença pequena; tremor/sacudidas, grande)
    trem = []
    for n in ("l_wrist", "r_wrist"):
        x = P(n)
        d2 = np.zeros(len(x))
        d2[1:-1] = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=1)
        trem.append(d2 / body_h)
    df["tremor_index"] = pd.Series(np.nanmax(np.vstack(trem), axis=0)).rolling(int(fps), min_periods=1).mean()

    knee_hip_dy = (((P("l_knee") + P("r_knee")) / 2)[:, 1] - mid_hip[:, 1]) / body_h
    posture = np.where(df["trunk_angle"] > 60, "deitado", np.where(knee_hip_dy < 0.15, "sentado", "em_pe"))
    df["posture"] = posture
    df["body_height_px"] = body_h
    df["visible"] = visible
    df["hip_speed"] = pd.Series(np.linalg.norm(np.diff(mid_hip, axis=0, prepend=mid_hip[:1]), axis=1) / body_h / dt
                                ).rolling(max(int(fps / 2), 1), min_periods=1).mean().fillna(0)
    # troca de identidade/reaparecimento gera saltos espúrios: zera velocidades nesses frames
    hip_dx = pd.Series(mid_hip[:, 0] / body_h).diff().abs().fillna(0) / dt
    unreliable = (~pd.Series(visible)) | (~pd.Series(visible).shift(1, fill_value=False)) | (hip_dx > 3.0)
    unreliable = unreliable.rolling(3, center=True, min_periods=1).max().astype(bool)
    df.loc[unreliable.to_numpy(), ["hip_drop_speed", "motion_energy", "tremor_index"]] = 0.0
    return df


# --------------------------------------------------------------------------- #
# Eventos
# --------------------------------------------------------------------------- #
@dataclass
class VideoEvent:
    kind: str
    start_s: float
    end_s: float
    severity: Severity
    score: float
    description: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "start_s": round(self.start_s, 2), "end_s": round(self.end_s, 2),
                "severity": self.severity.label, "score": round(self.score, 3), "description": self.description,
                "evidence": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.evidence.items()}}


def segment_repetitions(df: pd.DataFrame, profile: dict, fps: float) -> pd.DataFrame:
    from scipy.signal import find_peaks

    a = profile["angle"]
    sig = df[[f"{a}_l", f"{a}_r"]].max(axis=1).rolling(max(int(fps * 0.2), 1), center=True, min_periods=1).mean()
    peaks, _ = find_peaks(sig.fillna(0), prominence=profile["rep_prominence"],
                          distance=int(profile["rep_min_interval_s"] * fps))
    valleys, _ = find_peaks(-sig.fillna(0), distance=int(profile["rep_min_interval_s"] * fps * 0.5))
    reps = []
    for i, p in enumerate(peaks):
        lo = valleys[valleys < p].max() if (valleys < p).any() else max(0, p - int(fps))
        hi = valleys[valleys > p].min() if (valleys > p).any() else min(len(df) - 1, p + int(fps))
        seg = df.iloc[lo:hi + 1]
        reps.append({"rep": i + 1, "start_s": df.t[lo], "peak_s": df.t[p], "end_s": df.t[hi],
                     "peak_l": seg[f"{a}_l"].max(), "peak_r": seg[f"{a}_r"].max(),
                     "max_trunk_lean": seg["trunk_lean"].abs().max(),
                     "duration_s": df.t[hi] - df.t[lo]})
    r = pd.DataFrame(reps)
    if len(r):
        r["asymmetry"] = (r["peak_l"] - r["peak_r"]).abs()
    return r


def _merge(events: list[VideoEvent], gap_s: float = 4.5) -> list[VideoEvent]:
    out: list[VideoEvent] = []
    for e in sorted(events, key=lambda e: (e.kind, e.start_s)):
        if out and out[-1].kind == e.kind and e.start_s - out[-1].end_s <= gap_s:
            o = out[-1]
            o.end_s = max(o.end_s, e.end_s)
            o.severity = max(o.severity, e.severity)
            o.score = max(o.score, e.score)
            o.evidence["n_ocorrencias"] = o.evidence.get("n_ocorrencias", 1) + 1
            if "rep" in e.evidence:
                o.evidence.setdefault("reps", [o.evidence["rep"]]).append(e.evidence["rep"])
                reps = o.evidence["reps"]
                head, _, tail = o.description.partition(": ")
                o.description = f"Repetições {reps[0]}–{reps[-1]} ({len(reps)}x): {tail}"
        else:
            out.append(e)
    return sorted(out, key=lambda e: e.start_s)


def detect_video_events(df: pd.DataFrame, fps: float, exercise: str | None = "abducao_ombro",
                        calibration_frac: float = 0.25) -> tuple[list[VideoEvent], pd.DataFrame]:
    events: list[VideoEvent] = []
    reps = pd.DataFrame()

    # 1) queda: descida rápida do quadril + postura deitada mantida
    for i in np.where(df["hip_drop_speed"] > 0.35)[0]:
        i = int(i)
        before = df.iloc[max(0, i - int(fps)): i]
        after = df.iloc[i: i + int(fps * 3)]
        after_vis = after[after["visible"]]
        was_upright = len(before) and before["visible"].mean() > 0.8 and (before["posture"] != "deitado").mean() > 0.8
        lying = len(after_vis) >= 0.6 * len(after) and (after_vis["trunk_angle"] > 60).mean() > 0.5
        if not (was_upright and lying):
            continue
        events.append(VideoEvent("queda", df.t[i], float(df.t.iloc[-1]), Severity.CRITICO, 0.95,
                                 f"Queda detectada em t={df.t[i]:.1f}s: quadril desceu "
                                 f"{df['hip_drop_speed'].iloc[i:i + int(fps * 2)].max():.1f} alturas/s e o tronco "
                                 f"ficou a {after['trunk_angle'].max():.0f}° da vertical.",
                                 {"hip_drop_speed": float(df["hip_drop_speed"].max())}))
        break

    # 2) regras do exercício, por repetição
    if exercise:
        prof = EXERCISE_PROFILES[exercise]
        reps = segment_repetitions(df, prof, fps)
        fall_t = next((e.start_s for e in events if e.kind == "queda"), np.inf)
        if len(reps):  # descarta repetições durante/após queda ou com paciente deitado
            reps = reps[reps.end_s < fall_t - 1.5].reset_index(drop=True)  # margem: a queda começa antes de ser confirmada
            upright = np.array([(df[(df.t >= r.start_s) & (df.t <= r.end_s)]["posture"] != "deitado").mean() > 0.8
                                for r in reps.itertuples()], dtype=bool)
            reps = reps.loc[upright].reset_index(drop=True)
        for r in reps.to_dict("records"):
            worst = min(r["peak_l"], r["peak_r"])
            if worst < prof["min_peak"]:
                side = "direito" if r["peak_r"] < r["peak_l"] else "esquerdo"
                events.append(VideoEvent("amplitude_reduzida", r["start_s"], r["end_s"], Severity.ATENCAO,
                                         min(1, (prof["min_peak"] - worst) / prof["min_peak"] * 2),
                                         f"Repetição {r['rep']}: amplitude máx. {worst:.0f}° no lado {side} "
                                         f"(esperado ≥ {prof['min_peak']:.0f}°).",
                                         {"peak_l": r["peak_l"], "peak_r": r["peak_r"], "rep": r["rep"]}))
            if r["asymmetry"] > prof["max_asymmetry"]:
                events.append(VideoEvent("assimetria", r["start_s"], r["end_s"], Severity.ATENCAO,
                                         min(1, r["asymmetry"] / 90),
                                         f"Repetição {r['rep']}: assimetria E/D de {r['asymmetry']:.0f}°.",
                                         {"asymmetry": r["asymmetry"], "rep": r["rep"]}))
            if r["max_trunk_lean"] > prof["max_trunk_lean"]:
                events.append(VideoEvent("compensacao_tronco", r["start_s"], r["end_s"], Severity.ATENCAO,
                                         min(1, r["max_trunk_lean"] / 30),
                                         f"Repetição {r['rep']}: inclinação lateral do tronco de "
                                         f"{r['max_trunk_lean']:.0f}° (compensação).",
                                         {"trunk_lean": r["max_trunk_lean"], "rep": r["rep"]}))

    # 3) movimento irregular / tremor (exige fps suficiente para captar 4–12 Hz parcialmente)
    vis = df[df["visible"]]
    base = vis["tremor_index"].iloc[: max(int(len(vis) * calibration_frac), 10)].dropna()
    thr = max(float(base.median() + 6 * 1.4826 * (base - base.median()).abs().median()), 0.01) if len(base) else np.inf
    # avaliado com o corpo parado (exercício no lugar); deslocamento/caminhada é ignorado
    mask = (df["tremor_index"] > thr) & (df["posture"] != "deitado") & df["visible"] & (df["hip_speed"] < 0.3)
    for a, b in (_runs(mask.to_numpy(), min_len=int(fps)) if fps >= 8 else []):
        events.append(VideoEvent("movimento_irregular", df.t[a], df.t[b - 1], Severity.ATENCAO,
                                 float(min(1, df["tremor_index"].iloc[a:b].max() / (3 * thr))),
                                 f"Oscilação de alta frequência dos punhos entre {df.t[a]:.1f}s e {df.t[b - 1]:.1f}s "
                                 f"(índice {df['tremor_index'].iloc[a:b].max():.3f} vs limiar {thr:.3f}).",
                                 {"tremor_max": float(df["tremor_index"].iloc[a:b].max()), "threshold": thr}))

    # 4) paciente fora do campo de visão por > 5 s (sem supervisão visual)
    for a, b in _runs(~df["visible"].to_numpy(), min_len=int(5 * fps)):
        events.append(VideoEvent("paciente_fora_do_campo", df.t[a], df.t[b - 1], Severity.INFO, 0.3,
                                 f"Paciente não visível entre {df.t[a]:.0f}s e {df.t[b - 1]:.0f}s.", {}))

    # 5) não supervisionado: Isolation Forest por janela de 1 s, calibrado nas repetições iniciais
    events += _iforest_windows(df, fps, calibration_frac, already=events)
    return _merge(events), reps


def _iforest_windows(df: pd.DataFrame, fps: float, calibration_frac: float, already: list[VideoEvent]) -> list[VideoEvent]:
    from sklearn.ensemble import IsolationForest

    cols = ["shoulder_abd_l", "shoulder_abd_r", "trunk_lean", "trunk_angle", "motion_energy", "tremor_index", "hip_y_norm"]
    win = max(int(fps), 2)
    g = df[cols].groupby(np.arange(len(df)) // win)
    feats = pd.concat([g.mean().add_suffix("_mean"), g.std().add_suffix("_std"), g.max().add_suffix("_max")], axis=1).fillna(0)
    t0 = df.t.groupby(np.arange(len(df)) // win).first().to_numpy()
    n_cal = max(int(len(feats) * calibration_frac), 8)
    if len(feats) < n_cal + 4:
        return []
    model = IsolationForest(n_estimators=200, random_state=0).fit(feats.iloc[:n_cal])
    score = -model.score_samples(feats)
    thr = np.quantile(score[:n_cal], 0.99) * 1.05
    out = []
    for a, b in _runs(score > thr, min_len=2):
        s, e = float(t0[a]), float(t0[b - 1] + 1)
        covered = any(ev.start_s - 1 <= s and e <= ev.end_s + 1 for ev in already)
        if not covered:
            out.append(VideoEvent("padrao_anomalo", s, e, Severity.INFO, float(min(1, score[a:b].max() / (2 * thr))),
                                  f"Padrão de movimento atípico entre {s:.0f}s e {e:.0f}s (Isolation Forest).",
                                  {"if_score": float(score[a:b].max()), "threshold": float(thr)}))
    return out


def _runs(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


# --------------------------------------------------------------------------- #
# Processamento de vídeo
# --------------------------------------------------------------------------- #
@dataclass
class VideoResult:
    source: str
    fps: float
    keypoints: np.ndarray
    metrics: pd.DataFrame
    events: list[VideoEvent]
    repetitions: pd.DataFrame
    objects: list = field(default_factory=list)
    object_events: list[VideoEvent] = field(default_factory=list)
    annotated_path: str | None = None
    backend: str = ""

    @property
    def all_events(self) -> list[VideoEvent]:
        return sorted(self.events + self.object_events, key=lambda e: e.start_s)

    def to_findings(self, patient_id: str, session_start: datetime) -> list[Finding]:
        out = []
        for e in self.all_events:
            if e.severity < Severity.ATENCAO:
                continue
            out.append(Finding(patient_id, Modality.VIDEO, e.kind, e.severity, e.score,
                               f"[{Path(self.source).name} {e.start_s:.0f}–{e.end_s:.0f}s] {e.description}",
                               session_start + timedelta(seconds=e.start_s),
                               session_start + timedelta(seconds=e.end_s), evidence=e.evidence))
        return out


def extract_keypoints_from_video(path: str | Path, estimator: PoseEstimator, stride: int = 2,
                                 object_monitor=None, max_frames: int | None = None):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kps, objs, frames_idx = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        if i % stride == 0:
            p = estimator(frame)
            kps.append(p if p is not None else np.zeros((17, 3)))
            if object_monitor is not None:
                objs.append(object_monitor.evaluate(i / fps_in, object_monitor.detect(frame)))
            frames_idx.append(i)
        i += 1
    cap.release()
    return np.array(kps), fps_in / stride, objs


def object_events(objs: list) -> list[VideoEvent]:
    from .objects import instrument_count_check

    out = []
    viol = [(o.t, v) for o in objs for v in o.violations]
    by_key: dict = {}
    for t, v in viol:
        by_key.setdefault((v["zone"], v["type"], v.get("cls")), []).append(t)
    for (zone, typ, cls), ts in by_key.items():
        if len(ts) < 3:  # exige persistência (≥3 frames amostrados)
            continue
        desc = (f"Objeto proibido '{cls}' na zona '{zone}'" if typ == "objeto_proibido"
                else f"Excesso de pessoas na zona '{zone}'")
        out.append(VideoEvent(f"zona_critica_{typ}", min(ts), max(ts), Severity.URGENTE, 0.8,
                              f"{desc} entre {min(ts):.0f}s e {max(ts):.0f}s ({len(ts)} frames).",
                              {"zone": zone, "frames": len(ts)}))
    chk = instrument_count_check(objs)
    if chk and chk["mismatch"]:
        out.append(VideoEvent("contagem_instrumentais", objs[0].t, objs[-1].t, Severity.URGENTE, 0.85,
                              f"Contagem de instrumentais divergente: {chk['start']} no início vs {chk['end']} no fim "
                              "— conferir possível item retido/faltante.", chk))
    return out


def analyze_video(source: str | Path | np.ndarray, estimator: PoseEstimator | None = None, fps: float = 15.0,
                  exercise: str | None = "abducao_ombro", stride: int = 2, object_monitor=None,
                  annotate_to: str | Path | None = None, max_frames: int | None = None) -> VideoResult:
    """`source` pode ser um caminho de vídeo ou um array de keypoints (T,17,3) já extraído."""
    objs: list = []
    backend = "keypoints"
    if isinstance(source, np.ndarray):
        kp, src_name = source, "keypoints.npy"
    else:
        src_name = str(source)
        if str(source).endswith(".npy"):
            kp = np.load(source)
        else:
            if estimator is None:
                raise ValueError("Informe um PoseEstimator para processar vídeo.")
            kp, fps, objs = extract_keypoints_from_video(source, estimator, stride, object_monitor, max_frames)
            backend = estimator.name
    metrics = frame_metrics(kp, fps)
    events, reps = detect_video_events(metrics, fps, exercise)
    res = VideoResult(src_name, fps, kp, metrics, events, reps, objs, object_events(objs) if objs else [],
                      backend=backend)
    if annotate_to:
        res.annotated_path = str(render_annotated(res, annotate_to,
                                                  None if isinstance(source, np.ndarray) or src_name.endswith(".npy") else source,
                                                  stride))
    return res


def render_annotated(res: VideoResult, out_path: str | Path, video_path=None, stride: int = 2) -> Path:
    """Vídeo com esqueleto, ângulos e faixa de alerta quando há evento ativo."""
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path)) if video_path else None
    kp = res.keypoints
    if cap is not None:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        w = int(max(640, np.nanmax(kp[..., 0]) + 40))
        h = int(max(480, np.nanmax(kp[..., 1]) + 40))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), res.fps, (w, h))
    evs = res.all_events
    for f in range(len(kp)):
        if cap is not None:
            for _ in range(stride):
                ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = np.full((h, w, 3), 245, np.uint8)
        t = f / res.fps
        active = [e for e in evs if e.start_s <= t <= e.end_s and e.severity >= Severity.ATENCAO]
        color = (0, 0, 220) if any(e.severity >= Severity.URGENTE for e in active) else (0, 160, 255) if active else (40, 170, 40)
        p = kp[f]
        for a, b in SKELETON:
            if p[a, 2] > 0.3 and p[b, 2] > 0.3:
                cv2.line(frame, tuple(p[a, :2].astype(int)), tuple(p[b, :2].astype(int)), color, 3)
        for q in p:
            if q[2] > 0.3:
                cv2.circle(frame, tuple(q[:2].astype(int)), 4, (255, 255, 255), -1)
        m = res.metrics.iloc[f]
        cv2.putText(frame, f"t={t:5.1f}s  ombro E {m.shoulder_abd_l:4.0f}  D {m.shoulder_abd_r:4.0f}  tronco {m.trunk_lean:+4.0f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        if active:
            cv2.rectangle(frame, (0, h - 34), (w, h), color, -1)
            label = " | ".join(sorted({e.kind.replace("_", " ").upper() for e in active}))
            cv2.putText(frame, f"ALERTA: {label}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vw.write(frame)
    vw.release()
    if cap is not None:
        cap.release()
    to_browser_mp4(out_path)
    return out_path


def to_browser_mp4(path: Path) -> bool:
    """Recodifica para H.264/yuv420p (reprodutível em navegadores e no Streamlit), se houver ffmpeg."""
    import shutil
    import subprocess

    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            log.info("ffmpeg indisponível: vídeo anotado mantido em mp4v.")
            return False
    tmp = Path(path).with_suffix(".h264.mp4")
    r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(tmp)], capture_output=True)
    if r.returncode == 0 and tmp.exists():
        tmp.replace(path)
        return True
    log.warning("Falha ao recodificar vídeo: %s", r.stderr.decode()[-300:])
    return False
