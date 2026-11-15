"""Baixa os modelos usados na análise de vídeo.

    python scripts/download_models.py            # YOLOv8 pose + detecção
    python scripts/download_models.py --openpose # + OpenPose COCO (Caffe, ~200 MB)

O OpenPose é distribuído pela CMU sob licença acadêmica/não comercial.
Se o servidor da CMU estiver fora do ar, baixe `pose_iter_440000.caffemodel`
manualmente e coloque em `models/`.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
OPENPOSE = {
    "pose_deploy_linevec.prototxt": [
        "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"],
    "pose_iter_440000.caffemodel": [
        "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"],
}


def fetch(name: str, urls: list[str]) -> None:
    dst = MODELS / name
    if dst.exists():
        print(f"✔ {name} já existe")
        return
    for url in urls:
        try:
            print(f"↓ {url}")
            urllib.request.urlretrieve(url, dst)
            print(f"✔ {name}")
            return
        except Exception as exc:
            print(f"  falhou: {exc}")
    print(f"✖ não foi possível baixar {name}; baixe manualmente para {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpose", action="store_true")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    from ultralytics import YOLO

    for w in ("yolov8n-pose.pt", "yolov8n.pt"):
        YOLO(str(MODELS / w))  # baixa automaticamente do GitHub da Ultralytics
        print(f"✔ {w}")
    if args.openpose:
        for name, urls in OPENPOSE.items():
            fetch(name, urls)


if __name__ == "__main__":
    main()
