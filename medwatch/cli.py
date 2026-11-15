"""Interface de linha de comando.

    python -m medwatch demo                       # pipeline completo com dados de demonstração
    python -m medwatch demo --video sessao.mp4    # + um vídeo real
    python -m medwatch video sessao.mp4 --patient P010 --exercise abducao_ombro --objects
    python -m medwatch audio consulta.wav --patient P010
    python -m medwatch vitals sinais.csv
    python -m medwatch stream --speed 1200        # sinais vitais "ao vivo" com alertas
    python -m medwatch physionet --subject 1      # sinais reais do BIDMC (PhysioNet)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="medwatch", description="Monitoramento multimodal de pacientes")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="pipeline completo com dados de demonstração")
    d.add_argument("--video", help="vídeo real adicional para analisar")
    d.add_argument("--video-patient", default="P004")
    d.add_argument("--video-exercise", default=None)
    d.add_argument("--regen", action="store_true", help="regerar dados sintéticos")

    v = sub.add_parser("video", help="analisar um vídeo clínico")
    v.add_argument("path")
    v.add_argument("--patient", default="P000")
    v.add_argument("--exercise", default=None, help="abducao_ombro | agachamento | flexao_cotovelo")
    v.add_argument("--backend", default=None, help="yolov8 | openpose")
    v.add_argument("--objects", action="store_true", help="detecção de objetos/zonas críticas (YOLOv8)")
    v.add_argument("--max-frames", type=int, default=None)

    a = sub.add_parser("audio", help="analisar áudio de consulta")
    a.add_argument("path")
    a.add_argument("--patient", default="P000")

    s = sub.add_parser("vitals", help="detectar anomalias em CSV de sinais vitais")
    s.add_argument("path")

    st = sub.add_parser("stream", help="simular monitoramento em tempo real")
    st.add_argument("--path", default=None)
    st.add_argument("--speed", type=float, default=1200.0)
    st.add_argument("--patient", default=None)

    pn = sub.add_parser("physionet", help="baixar e analisar sinais do BIDMC (PhysioNet)")
    pn.add_argument("--subject", type=int, default=1)
    pn.add_argument("--local-dir", default=None)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    from .alerts.dispatcher import build_alert_manager
    from .config import get_settings
    from .pipeline import prepare_demo_data, run_demo, run_video, stream_vitals

    settings = get_settings()
    out = settings.output_dir

    if args.cmd == "demo":
        if args.regen:
            prepare_demo_data(force=True)
        res = run_demo(settings, extra_video=args.video, extra_video_patient=args.video_patient,
                       extra_video_exercise=args.video_exercise)
        print("\n=== Métricas ===")
        print(json.dumps(res.metrics, indent=2, ensure_ascii=False, default=str))
        print(f"\nPainel: {res.outputs['html']}\nAlertas: {res.outputs['alerts_jsonl']}")
    elif args.cmd == "video":
        res, findings, rep = run_video(Path(args.path), args.patient, datetime.now(), settings, out,
                                       exercise=args.exercise, backend=args.backend, detect_objects=args.objects,
                                       max_frames=args.max_frames)
        build_alert_manager(settings).dispatch_all(findings)
        print(f"\nRelatório: {rep['markdown']}\nVídeo anotado: {res.annotated_path}")
    elif args.cmd == "audio":
        from .audio.analyzer import AudioAnalyzer

        rep, findings = AudioAnalyzer(settings.azure).analyze(args.path, args.patient)
        print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False, default=str))
        build_alert_manager(settings).dispatch_all(findings)
    elif args.cmd == "vitals":
        import pandas as pd

        from .anomaly.vitals import detect_vitals_anomalies

        eps, realtime, ann = detect_vitals_anomalies(pd.read_csv(args.path, parse_dates=["timestamp"]))
        ann.to_csv(out / "vitals_anotado.csv", index=False)
        build_alert_manager(settings).dispatch_all(realtime)
    elif args.cmd == "stream":
        path = Path(args.path) if args.path else prepare_demo_data()["vitals"]
        stream_vitals(path, settings, args.speed, args.patient)
    elif args.cmd == "physionet":
        from .anomaly.vitals import detect_vitals_anomalies
        from .data.physionet import load_bidmc_numerics

        df = load_bidmc_numerics(args.subject, args.local_dir, resample=None)  # 1 Hz, ~8 min
        eps, realtime, ann = detect_vitals_anomalies(df, warmup=60, baseline_window=180, baseline_lag=10)
        ann.to_csv(out / f"bidmc{args.subject:02d}_anotado.csv", index=False)
        build_alert_manager(settings).dispatch_all(realtime)
        print(f"{len(df)} amostras, {len(eps)} episódios.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
