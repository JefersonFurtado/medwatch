"""Estruturas de dados compartilhadas entre as modalidades.

Toda análise (vídeo, áudio, sinais vitais, prescrição, movimentação) produz
`Finding`s com o mesmo formato. Isso permite a fusão multimodal e um único
mecanismo de alertas.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any


class Severity(IntEnum):
    INFO = 0
    ATENCAO = 1
    URGENTE = 2
    CRITICO = 3

    @classmethod
    def parse(cls, value: "str | int | Severity") -> "Severity":
        if isinstance(value, Severity):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value).upper().replace("Ç", "C").replace("Ã", "A")]

    @property
    def label(self) -> str:
        return {0: "INFO", 1: "ATENÇÃO", 2: "URGENTE", 3: "CRÍTICO"}[int(self)]


class Modality:
    VIDEO = "video"
    AUDIO = "audio"
    VITALS = "sinais_vitais"
    PRESCRIPTION = "prescricao"
    MOVEMENT = "movimentacao"
    FUSION = "fusao"


@dataclass
class Finding:
    """Uma anomalia/evento detectado por um analisador."""

    patient_id: str
    modality: str
    kind: str                    # ex.: "dessaturacao", "disartria", "queda"
    severity: Severity
    score: float                 # 0..1 — confiança/intensidade normalizada
    message: str
    timestamp: datetime
    end_timestamp: datetime | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    finding_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def detected_at(self) -> datetime:
        """Momento em que o sistema detectou/escalonou (≠ início do evento, em monitores de streaming)."""
        v = self.evidence.get("detected_at")
        return datetime.fromisoformat(v) if isinstance(v, str) else (v or self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "evidence"}
        d["evidence"] = {k: v for k, v in self.evidence.items() if not k.startswith("_")}
        d["severity"] = self.severity.label
        d["severity_level"] = int(self.severity)
        d["timestamp"] = self.timestamp.isoformat()
        d["end_timestamp"] = self.end_timestamp.isoformat() if self.end_timestamp else None
        d["evidence"] = _jsonable(d["evidence"])
        return d


@dataclass
class Alert:
    """Alerta enviado à equipe médica (pode agregar vários findings)."""

    patient_id: str
    severity: Severity
    title: str
    message: str
    recommended_action: str
    findings: list[Finding]
    risk_score: float
    created_at: datetime = field(default_factory=datetime.now)
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def modalities(self) -> list[str]:
        return sorted({f.modality for f in self.findings})

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "severity": self.severity.label,
            "severity_level": int(self.severity),
            "title": self.title,
            "message": self.message,
            "recommended_action": self.recommended_action,
            "risk_score": round(float(self.risk_score), 3),
            "modalities": self.modalities,
            "created_at": self.created_at.isoformat(),
            "event_time": self.findings[0].detected_at.isoformat() if self.findings else None,
            "event_start": self.findings[0].timestamp.isoformat() if self.findings else None,
            "findings": [f.to_dict() for f in self.findings],
        }


def _jsonable(obj: Any) -> Any:
    """Converte tipos numpy/pandas/datetime para JSON."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, float):
        return None if obj != obj else round(obj, 4)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj
