"""Detecção de objetos e áreas críticas com YOLOv8.

Casos de uso
------------
* **Campo estéril (cirurgia)** — objetos proibidos na zona estéril (ex.:
  celular, copo/garrafa) ou pessoas em excesso.
* **Contagem de instrumentais** — compara a contagem de instrumentos
  (classes `knife`/`scissors` no COCO; em produção, um modelo YOLOv8
  re-treinado com instrumentais cirúrgicos) no início e no fim do
  procedimento → possível item retido/faltante.
* **Área do leito/fisioterapia** — paciente sai da área de segurança ou fica
  sem supervisão (nenhuma outra pessoa na sala).

Zonas são polígonos em coordenadas normalizadas (0–1), configuráveis por
sala em `configs/zones.json`.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

DEFAULT_ZONES = {
    "campo_esteril": {"polygon": [[0.25, 0.35], [0.75, 0.35], [0.75, 0.95], [0.25, 0.95]],
                      "forbidden": ["cell phone", "cup", "bottle", "handbag", "backpack"], "max_persons": 4},
    "area_seguranca": {"polygon": [[0.05, 0.05], [0.95, 0.05], [0.95, 0.98], [0.05, 0.98]],
                       "forbidden": [], "max_persons": 99},
}
INSTRUMENT_CLASSES = {"knife", "scissors", "fork", "spoon"}  # proxies COCO para instrumentais


def point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


@dataclass
class FrameObjects:
    t: float
    detections: list[dict]                     # {cls, conf, box:[x1,y1,x2,y2] normalizado}
    zone_counts: dict = field(default_factory=dict)
    violations: list[dict] = field(default_factory=list)


class ObjectZoneMonitor:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, zones: dict | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.zones = zones or DEFAULT_ZONES

    def detect(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        out = []
        for box, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = box
            out.append({"cls": res.names[int(k)], "conf": float(c), "box": [x1 / w, y1 / h, x2 / w, y2 / h]})
        return out

    def evaluate(self, t: float, detections: list[dict]) -> FrameObjects:
        fo = FrameObjects(t, detections)
        for zname, z in self.zones.items():
            inside = [d for d in detections
                      if point_in_polygon((d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2, z["polygon"])]
            counts = Counter(d["cls"] for d in inside)
            fo.zone_counts[zname] = dict(counts)
            for d in inside:
                if d["cls"] in z.get("forbidden", []):
                    fo.violations.append({"zone": zname, "type": "objeto_proibido", "cls": d["cls"], "conf": d["conf"]})
            if counts.get("person", 0) > z.get("max_persons", 99):
                fo.violations.append({"zone": zname, "type": "excesso_pessoas", "count": counts["person"]})
        return fo


def instrument_count_check(frames: list[FrameObjects], window_s: float = 5.0) -> dict | None:
    """Compara a contagem mediana de instrumentais no início e no fim do vídeo."""
    if not frames:
        return None
    t0, t1 = frames[0].t, frames[-1].t
    def count(fs):  # noqa: E306
        vals = [sum(1 for d in f.detections if d["cls"] in INSTRUMENT_CLASSES) for f in fs]
        return int(np.median(vals)) if vals else 0
    start = count([f for f in frames if f.t <= t0 + window_s])
    end = count([f for f in frames if f.t >= t1 - window_s])
    return {"start": start, "end": end, "mismatch": start != end}
