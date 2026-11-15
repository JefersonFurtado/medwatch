"""Estimação de pose humana — backends YOLOv8-pose e OpenPose.

Ambos devolvem keypoints no formato **COCO-17** `(17, 3) = [x, y, conf]`
em pixels, para o paciente principal do quadro (maior caixa / mais próxima
do quadro anterior). Assim o restante do pipeline independe do modelo.

* **YOLOv8-pose** (`ultralytics`): padrão; pesos baixados automaticamente.
* **OpenPose** (CMU, modelo COCO 18 pontos em Caffe) via `cv2.dnn` — baixe
  `pose_deploy_linevec.prototxt` e `pose_iter_440000.caffemodel` para
  `models/` (ver `scripts/download_models.py`).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

COCO17 = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
          "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
KP = {n: i for i, n in enumerate(COCO17)}
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 5), (0, 6)]
# índice OpenPose COCO-18 correspondente a cada ponto COCO-17
OPENPOSE18_TO_COCO17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]


class PoseEstimator(ABC):
    name = "base"

    def __init__(self):
        self._prev_center: np.ndarray | None = None

    @abstractmethod
    def _detect_all(self, frame: np.ndarray) -> list[np.ndarray]:
        """Retorna lista de arrays (17,3) — uma por pessoa."""

    def __call__(self, frame: np.ndarray) -> np.ndarray | None:
        people = self._detect_all(frame)
        if not people:
            return None
        if self._prev_center is None or len(people) == 1:
            areas = [_bbox_area(p) for p in people]
            best = people[int(np.argmax(areas))]
        else:  # rastreio simples: pessoa mais próxima da anterior
            d = [np.linalg.norm(_center(p) - self._prev_center) for p in people]
            best = people[int(np.argmin(d))]
        self._prev_center = _center(best)
        return best


def _visible(p: np.ndarray, thr: float = 0.2) -> np.ndarray:
    return p[p[:, 2] > thr, :2]


def _center(p: np.ndarray) -> np.ndarray:
    v = _visible(p)
    return v.mean(0) if len(v) else np.zeros(2)


def _bbox_area(p: np.ndarray) -> float:
    v = _visible(p)
    if len(v) < 2:
        return 0.0
    return float(np.ptp(v[:, 0]) * np.ptp(v[:, 1]))


class YoloPoseEstimator(PoseEstimator):
    name = "yolov8-pose"

    def __init__(self, weights: str = "yolov8n-pose.pt", conf: float = 0.35, device: str | None = None):
        super().__init__()
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def _detect_all(self, frame):
        res = self.model.predict(frame, conf=self.conf, verbose=False, device=self.device)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return []
        xy = res.keypoints.xy.cpu().numpy()
        cf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(xy.shape[:2])
        return [np.concatenate([xy[i], cf[i][:, None]], axis=1) for i in range(len(xy))]


class OpenPoseEstimator(PoseEstimator):
    """OpenPose (CMU) COCO-18 via OpenCV DNN — pessoa única (máximo de cada heatmap)."""

    name = "openpose"

    def __init__(self, proto: str | Path, weights: str | Path, in_size: int = 368, thr: float = 0.1):
        super().__init__()
        import cv2

        if not Path(proto).exists() or not Path(weights).exists():
            raise FileNotFoundError(
                "Modelo OpenPose não encontrado. Execute `python scripts/download_models.py --openpose` "
                f"ou ajuste OPENPOSE_PROTO/OPENPOSE_WEIGHTS ({proto}, {weights}).")
        if str(weights).endswith(".onnx"):  # OpenPose convertido para ONNX
            self.net = cv2.dnn.readNetFromONNX(str(weights))
        elif hasattr(cv2.dnn, "readNetFromCaffe"):
            self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        else:
            raise RuntimeError("Esta versão do OpenCV (5.x) não lê modelos Caffe. Instale opencv-python<5 "
                               "(ver requirements.txt) ou converta o OpenPose para ONNX.")
        self.in_size, self.thr = in_size, thr

    def _detect_all(self, frame):
        import cv2

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.in_size, self.in_size), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()  # (1, 57, H', W'): 19 heatmaps + 38 PAFs
        hm_h, hm_w = out.shape[2], out.shape[3]
        pts18 = np.zeros((18, 3))
        for i in range(18):
            hm = out[0, i]
            _, conf, _, (x, y) = cv2.minMaxLoc(hm)
            pts18[i] = (x * w / hm_w, y * h / hm_h, conf if conf > self.thr else 0.0)
        if (pts18[:, 2] > 0).sum() < 4:
            return []
        return [pts18[OPENPOSE18_TO_COCO17]]


def build_pose_estimator(backend: str, settings) -> PoseEstimator:
    if backend == "openpose":
        return OpenPoseEstimator(settings.openpose_proto, settings.openpose_weights)
    return YoloPoseEstimator(settings.yolo_pose_weights, settings.conf_threshold)
