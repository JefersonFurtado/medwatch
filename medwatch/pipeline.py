"""Orquestrador do fluxo multimodal.

    ┌ vídeo  ─ pose (YOLOv8/OpenPose) + YOLOv8 objetos ─ desvios ─┐
    ├ áudio  ─ Azure Speech → Azure Language + biomarcadores ─────┤
    ├ sinais vitais ─ NEWS2 + basal + CUSUM + Isolation Forest ───┼─> Findings ─> fusão ─> alertas ─> equipe
    ├ prescrições ─ regras farmacológicas + z-score ──────────────┤
    └ movimentação ─ imobilidade, agitação, saída do leito, queda ┘
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .alerts.dispatcher import AlertManager, build_alert_manager
from .anomaly.movement import detect_movement_anomalies
from .anomaly.prescriptions import detect_prescription_anomalies
from .anomaly.vitals import detect_vitals_anomalies, evaluate_event_detection
from .audio.analyzer import AudioAnalyzer
from .config import ROOT_DIR, Settings, get_settings
from .data import synthetic as syn
from .fusion import fuse, risk_timeline
from .reporting import alerts_html, plot_vitals, plot_voice, video_report
from .schemas import Alert, Finding

log = logging.getLogger(__name__)
DEMO_DIR = ROOT_DIR / "data" / "demo"


@dataclass
class PipelineResult:
    findings: list[Finding] = field(default_factory=list)
    fused: list[Finding] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Dados de demonstração
# --------------------------------------------------------------------------- #
def prepare_demo_data(data_dir: Path = DEMO_DIR, force: bool = False) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {"vitals": data_dir / "vitals.csv", "prescriptions": data_dir / "prescriptions.csv",
             "movement": data_dir / "movement.csv", "audio": data_dir / "audio",
             "video_kp": data_dir / "video" / "fisioterapia_P004_keypoints.npy",
             "video_mp4": data_dir / "video" / "fisioterapia_P004_esqueleto.mp4",
             "ground_truth": data_dir / "ground_truth.json"}
    if paths["ground_truth"].exists() and not force:
        return paths
    vit, ev_v = syn.generate_vitals()
    rx, ev_p = syn.generate_prescriptions()
    mv, ev_m = syn.generate_movement()
    ev_a = syn.generate_audio(paths["audio"])
    kp, ev_vid = syn.generate_physio_keypoints()
    vit.to_csv(paths["vitals"], index=False)
    rx.to_csv(paths["prescriptions"], index=False)
    mv.to_csv(paths["movement"], index=False)
    paths["video_kp"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["video_kp"], kp)
    syn.render_skeleton_video(kp, paths["video_mp4"])
    gt = {"vitals": [e.to_dict() for e in ev_v], "prescriptions": [e.to_dict() for e in ev_p],
          "movement": [e.to_dict() for e in ev_m], "audio": [e.to_dict() for e in ev_a],
          "video": {"patient_id": "P004", "session_start": (syn.START + timedelta(hours=12)).isoformat(),
                    "events": ev_vid}}
    paths["ground_truth"].write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def _events_from_gt(items: list[dict]) -> list[syn.Event]:
    return [syn.Event(d["patient_id"], d["modality"], d["kind"], datetime.fromisoformat(d["start"]),
                      datetime.fromisoformat(d["end"])) for d in items]


# --------------------------------------------------------------------------- #
# Execução por modalidade
# --------------------------------------------------------------------------- #
def run_vitals(csv: Path, out_dir: Path):
    df = pd.read_csv(csv, parse_dates=["timestamp"])
    episodes, realtime, ann = detect_vitals_anomalies(df)
    ann.to_csv(out_dir / "vitals_anotado.csv", index=False)
    return episodes, realtime, ann


def run_audio(audio_dir: Path, settings: Settings, timestamp: datetime) -> tuple[list, list[Finding]]:
    analyzer = AudioAnalyzer(settings.azure)
    reports, findings = [], []
    for wav in sorted(audio_dir.glob("*.wav")):
        pid = wav.stem.split("_")[-1]
        rep, fs = analyzer.analyze(wav, pid, timestamp)
        reports.append(rep)
        findings += fs
    return reports, findings


def run_video(source: Path | str, patient_id: str, session_start: datetime, settings: Settings, out_dir: Path,
              exercise: str | None = "abducao_ombro", backend: str | None = None, detect_objects: bool = False,
              max_frames: int | None = None):
    from .video.analyzer import analyze_video

    source = Path(source)
    est = om = None
    if source.suffix != ".npy":
        from .video.pose import build_pose_estimator

        est = build_pose_estimator(backend or settings.video.pose_backend, settings.video)
        if detect_objects:
            from .video.objects import ObjectZoneMonitor

            zones_file = ROOT_DIR / "configs" / "zones.json"
            zones = json.loads(zones_file.read_text(encoding="utf-8")) if zones_file.exists() else None
            om = ObjectZoneMonitor(settings.video.yolo_det_weights, settings.video.conf_threshold, zones)
    res = analyze_video(source, est, fps=15.0, exercise=exercise, stride=settings.video.frame_stride,
                        object_monitor=om, annotate_to=out_dir / f"{source.stem}_anotado.mp4", max_frames=max_frames)
    rep = video_report(res, patient_id, out_dir, exercise)
    res.metrics.to_csv(out_dir / f"{source.stem}_metricas.csv", index=False)
    return res, res.to_findings(patient_id, session_start), rep


# --------------------------------------------------------------------------- #
# Demo completa
# --------------------------------------------------------------------------- #
def run_demo(settings: Settings | None = None, console: bool = True, extra_video: str | None = None,
             extra_video_patient: str = "P004", extra_video_exercise: str | None = None) -> PipelineResult:
    settings = settings or get_settings()
    out = settings.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if settings.alerts.jsonl_path.exists():
        settings.alerts.jsonl_path.unlink()
    paths = prepare_demo_data()
    gt = json.loads(paths["ground_truth"].read_text(encoding="utf-8"))
    result = PipelineResult()

    log.info("1/6 Sinais vitais (simulação em tempo real)…")
    episodes, realtime, ann = run_vitals(paths["vitals"], out)
    result.metrics["sinais_vitais"] = evaluate_event_detection(episodes, _events_from_gt(gt["vitals"]))
    result.metrics["sinais_vitais"]["artefatos_descartados"] = sum(f.kind == "possivel_artefato" for f in episodes)

    log.info("2/6 Prescrições…")
    rx_f = detect_prescription_anomalies(pd.read_csv(paths["prescriptions"], parse_dates=["timestamp"]))
    result.metrics["prescricao"] = evaluate_event_detection(rx_f, _events_from_gt(gt["prescriptions"]), tolerance_min=5)

    log.info("3/6 Movimentação no leito…")
    mv_f = detect_movement_anomalies(pd.read_csv(paths["movement"], parse_dates=["timestamp"]))
    result.metrics["movimentacao"] = evaluate_event_detection(mv_f, _events_from_gt(gt["movement"]), tolerance_min=5)

    log.info("4/6 Áudio das consultas (Azure Speech + Language)…")
    audio_reports, au_f = run_audio(paths["audio"], settings, syn.START + timedelta(hours=13))
    result.metrics["audio"] = evaluate_event_detection(au_f, _events_from_gt(gt["audio"]), tolerance_min=5)
    result.metrics["audio"]["azure_speech"] = settings.azure.speech_enabled
    result.metrics["audio"]["azure_language"] = settings.azure.language_enabled
    (out / "audio_relatorios.json").write_text(json.dumps([r.to_dict() for r in audio_reports], indent=2,
                                                          ensure_ascii=False, default=str), encoding="utf-8")

    log.info("5/6 Vídeo de fisioterapia…")
    vg = gt["video"]
    v_start = datetime.fromisoformat(vg["session_start"])
    vres, vid_f, vrep = run_video(paths["video_kp"], vg["patient_id"], v_start, settings, out)
    vid_events = [syn.Event(vg["patient_id"], "video", e["kind"], v_start + timedelta(seconds=e["start_s"]),
                            v_start + timedelta(seconds=e["end_s"])) for e in vg["events"]]
    result.metrics["video"] = _evaluate_video(vid_f, vid_events)
    if extra_video:
        _, xf, xrep = run_video(Path(extra_video), extra_video_patient, datetime.now(), settings, out,
                                exercise=extra_video_exercise, detect_objects=True)
        vid_f += xf
        result.outputs["video_extra"] = str(xrep["markdown"])

    result.findings = sorted(episodes + rx_f + mv_f + au_f + vid_f, key=lambda f: f.timestamp)

    log.info("6/6 Fusão multimodal e alertas…")
    result.fused = fuse(result.findings)
    manager: AlertManager = build_alert_manager(settings, console=console)
    # findings de vitais já fechados geram o alerta no instante de detecção (realtime)
    to_alert = [f for f in result.findings if f.modality != "sinais_vitais"] + realtime + result.fused
    result.alerts = manager.dispatch_all(to_alert)
    result.metrics["alertas"] = {"enviados": len(result.alerts), "suprimidos_cooldown": manager.suppressed,
                                 "por_severidade": pd.Series([a.severity.label for a in result.alerts]).value_counts().to_dict()}

    # artefatos
    all_f = [f.to_dict() for f in result.findings + result.fused]
    (out / "findings.json").write_text(json.dumps(all_f, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metricas.json").write_text(json.dumps(result.metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    risk = pd.DataFrame(risk_timeline(result.findings + result.fused, syn.START, syn.START + timedelta(hours=24)))
    risk.to_csv(out / "risco_pacientes.csv", index=False)
    imgs = plot_vitals(ann, episodes, out) + [plot_voice(audio_reports, out), vrep["plot"]]
    result.outputs.update({"html": str(alerts_html(result.alerts, result.metrics, imgs, out / "painel_alertas.html")),
                           "video_report": str(vrep["markdown"]), "alerts_jsonl": str(settings.alerts.jsonl_path)})
    return result


def _evaluate_video(findings: list[Finding], events: list) -> dict:
    hits = 0
    for ev in events:
        if any(f.timestamp <= ev.end and (f.end_timestamp or f.timestamp) >= ev.start and
               (f.kind == ev.kind or (ev.kind == "amplitude_reduzida" and f.kind == "assimetria")) for f in findings):
            hits += 1
    tp = sum(1 for f in findings if any(f.timestamp <= e.end and (f.end_timestamp or f.timestamp) >= e.start for e in events))
    rec = hits / len(events) if events else 1.0
    prec = tp / len(findings) if findings else 1.0
    return {"events": len(events), "detected": hits, "alerts": len(findings), "recall": round(rec, 3),
            "precision": round(prec, 3), "f1": round(2 * rec * prec / (rec + prec), 3) if rec + prec else 0.0,
            "mean_detection_delay_min": None}


# --------------------------------------------------------------------------- #
# Streaming (demonstração em "tempo real")
# --------------------------------------------------------------------------- #
def stream_vitals(csv: Path, settings: Settings | None = None, speed: float = 600.0, patient: str | None = None):
    """Reproduz o CSV como se as amostras chegassem em tempo real (speed× mais rápido)."""
    import time

    from .anomaly.vitals import StreamingVitalsMonitor

    settings = settings or get_settings()
    df = pd.read_csv(csv, parse_dates=["timestamp"]).sort_values("timestamp")
    if patient:
        df = df[df.patient_id == patient]
    monitors: dict[str, StreamingVitalsMonitor] = {}
    manager = build_alert_manager(settings)
    prev = None
    for row in df.to_dict("records"):
        if prev is not None and speed > 0:
            time.sleep(max((row["timestamp"] - prev).total_seconds() / speed, 0) / max(df.patient_id.nunique(), 1))
        prev = row["timestamp"]
        mon = monitors.setdefault(row["patient_id"], StreamingVitalsMonitor(row["patient_id"]))
        for f in mon.update(row):
            manager.dispatch(f)
    return manager.sent
