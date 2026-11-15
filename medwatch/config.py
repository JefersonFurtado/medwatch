"""Configuração central do MedWatch.

Todas as credenciais vêm de variáveis de ambiente (ou de um arquivo `.env`),
nunca do código-fonte. Quando as chaves do Azure não estão presentes, os
módulos de áudio entram em modo *offline* (fallback local) para que o
pipeline continue executável em desenvolvimento e em testes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv é opcional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


ROOT_DIR = Path(__file__).resolve().parents[1]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    return value if value not in ("", None) else default


@dataclass
class AzureSettings:
    speech_key: str | None = field(default_factory=lambda: _env("AZURE_SPEECH_KEY"))
    speech_region: str | None = field(default_factory=lambda: _env("AZURE_SPEECH_REGION", "brazilsouth"))
    speech_language: str = field(default_factory=lambda: _env("AZURE_SPEECH_LANGUAGE", "pt-BR"))
    language_key: str | None = field(default_factory=lambda: _env("AZURE_LANGUAGE_KEY"))
    language_endpoint: str | None = field(default_factory=lambda: _env("AZURE_LANGUAGE_ENDPOINT"))
    language_code: str = field(default_factory=lambda: _env("AZURE_LANGUAGE_CODE", "pt-BR"))
    # Text Analytics for Health só suporta alguns idiomas; para pt-BR o
    # pipeline pode traduzir antes ou usar o modelo em inglês.
    health_language: str = field(default_factory=lambda: _env("AZURE_HEALTH_LANGUAGE", "pt"))

    @property
    def speech_enabled(self) -> bool:
        return bool(self.speech_key and self.speech_region)

    @property
    def language_enabled(self) -> bool:
        return bool(self.language_key and self.language_endpoint)


@dataclass
class AlertSettings:
    webhook_url: str | None = field(default_factory=lambda: _env("ALERT_WEBHOOK_URL"))
    jsonl_path: Path = field(default_factory=lambda: Path(_env("ALERT_LOG", str(ROOT_DIR / "outputs" / "alerts.jsonl"))))
    cooldown_minutes: int = field(default_factory=lambda: int(_env("ALERT_COOLDOWN_MIN", "15")))
    min_severity_to_notify: str = field(default_factory=lambda: _env("ALERT_MIN_SEVERITY", "ATENCAO"))


@dataclass
class VideoSettings:
    pose_backend: str = field(default_factory=lambda: _env("POSE_BACKEND", "yolov8"))  # yolov8 | openpose
    yolo_pose_weights: str = field(default_factory=lambda: _env("YOLO_POSE_WEIGHTS", str(ROOT_DIR / "models" / "yolov8n-pose.pt")))
    yolo_det_weights: str = field(default_factory=lambda: _env("YOLO_DET_WEIGHTS", str(ROOT_DIR / "models" / "yolov8n.pt")))
    openpose_proto: str = field(default_factory=lambda: _env("OPENPOSE_PROTO", str(ROOT_DIR / "models" / "pose_deploy_linevec.prototxt")))
    openpose_weights: str = field(default_factory=lambda: _env("OPENPOSE_WEIGHTS", str(ROOT_DIR / "models" / "pose_iter_440000.caffemodel")))
    frame_stride: int = field(default_factory=lambda: int(_env("VIDEO_FRAME_STRIDE", "2")))
    conf_threshold: float = field(default_factory=lambda: float(_env("VIDEO_CONF", "0.35")))


@dataclass
class Settings:
    azure: AzureSettings = field(default_factory=AzureSettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    output_dir: Path = field(default_factory=lambda: Path(_env("OUTPUT_DIR", str(ROOT_DIR / "outputs"))))
    random_seed: int = 42

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alerts.jsonl_path.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
