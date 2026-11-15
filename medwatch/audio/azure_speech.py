"""Transcrição de consultas com **Azure Speech to Text**.

Usa reconhecimento contínuo (áudios longos) com timestamps por palavra, o
que permite medir taxa de fala e pausas a partir da transcrição.

Fallback offline (sem credenciais): lê a transcrição de um arquivo `.txt`
com o mesmo nome do áudio — útil para testes, CI e desenvolvimento.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..config import AzureSettings

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    source: str                                  # "azure" | "sidecar" | "none"
    language: str = "pt-BR"
    segments: list[dict] = field(default_factory=list)   # {text, offset_s, duration_s, confidence}
    words: list[dict] = field(default_factory=list)      # {word, offset_s, duration_s}

    @property
    def words_per_minute(self) -> float | None:
        if not self.words:
            return None
        span = (self.words[-1]["offset_s"] + self.words[-1]["duration_s"]) - self.words[0]["offset_s"]
        return len(self.words) / span * 60 if span > 0 else None

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "language": self.language,
                "segments": self.segments, "words_per_minute": self.words_per_minute}


class AzureSpeechTranscriber:
    def __init__(self, settings: AzureSettings):
        self.settings = settings

    def transcribe(self, audio_path: str | Path, timeout_s: int = 600) -> Transcript:
        audio_path = Path(audio_path)
        if self.settings.speech_enabled:
            try:
                return self._azure(audio_path, timeout_s)
            except Exception as exc:  # rede, cota, formato...
                log.warning("Azure Speech falhou (%s); usando fallback offline.", exc)
        return self._sidecar(audio_path)

    # ------------------------------------------------------------------ #
    def _azure(self, audio_path: Path, timeout_s: int) -> Transcript:
        import azure.cognitiveservices.speech as speechsdk

        cfg = speechsdk.SpeechConfig(subscription=self.settings.speech_key, region=self.settings.speech_region)
        cfg.speech_recognition_language = self.settings.speech_language
        cfg.request_word_level_timestamps()
        cfg.output_format = speechsdk.OutputFormat.Detailed
        cfg.set_profanity(speechsdk.ProfanityOption.Raw)
        audio_cfg = speechsdk.audio.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)

        segments, words, errors = [], [], []
        done = threading.Event()

        def on_recognized(evt):
            res = evt.result
            if res.reason != speechsdk.ResultReason.RecognizedSpeech or not res.text:
                return
            import json

            detail = json.loads(res.json)
            best = (detail.get("NBest") or [{}])[0]
            segments.append({"text": res.text, "offset_s": res.offset / 1e7, "duration_s": res.duration / 1e7,
                             "confidence": best.get("Confidence")})
            for w in best.get("Words", []):
                words.append({"word": w["Word"], "offset_s": w["Offset"] / 1e7, "duration_s": w["Duration"] / 1e7})

        def on_canceled(evt):
            if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
                errors.append(evt.cancellation_details.error_details)
            done.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(lambda _: done.set())
        recognizer.canceled.connect(on_canceled)
        recognizer.start_continuous_recognition()
        done.wait(timeout=timeout_s)
        recognizer.stop_continuous_recognition()
        if errors:
            raise RuntimeError(errors[0])
        text = " ".join(s["text"] for s in segments)
        log.info("Azure Speech: %d segmentos, %d palavras", len(segments), len(words))
        return Transcript(text, "azure", self.settings.speech_language, segments, words)

    @staticmethod
    def _sidecar(audio_path: Path) -> Transcript:
        txt = audio_path.with_suffix(".txt")
        if txt.exists():
            return Transcript(txt.read_text(encoding="utf-8").strip(), "sidecar")
        log.warning("Sem Azure e sem transcrição local para %s.", audio_path.name)
        return Transcript("", "none")
