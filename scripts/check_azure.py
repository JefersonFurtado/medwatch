"""Verifica as credenciais Azure configuradas no `.env`.

    python scripts/check_azure.py [audio.wav]

1. Azure AI Language: sentimento + frases-chave + entidades de saúde num texto curto.
2. Azure Speech: transcreve o WAV informado (ou o áudio de demonstração do P001).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medwatch.audio.azure_speech import AzureSpeechTranscriber  # noqa: E402
from medwatch.audio.azure_text import AzureTextAnalyzer  # noqa: E402
from medwatch.config import get_settings  # noqa: E402


def main() -> int:
    s = get_settings().azure
    ok = True
    print(f"Speech:   {'configurado' if s.speech_enabled else 'NÃO configurado'} (região {s.speech_region})")
    print(f"Language: {'configurado' if s.language_enabled else 'NÃO configurado'} ({s.language_endpoint})")
    if s.language_enabled:
        r = AzureTextAnalyzer(s).analyze("Estou com falta de ar e dor no peito desde ontem, não tenho febre.")
        print(f"  fonte={r.source} sentimento={r.sentiment} frases={r.key_phrases}")
        print(f"  entidades de saúde={[(e['text'], e['category'], e['negated']) for e in r.health_entities]}")
        ok &= r.source == "azure"
    if s.speech_enabled:
        wav = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data/demo/audio/consulta_P001.wav"
        t = AzureSpeechTranscriber(s).transcribe(wav)
        print(f"  fonte={t.source} texto='{t.text[:120]}' wpm={t.words_per_minute}")
        ok &= t.source == "azure"
        if wav.name.startswith("consulta_P0"):
            print("  (o áudio de demonstração é voz sintética sem palavras — use uma gravação real para ver a transcrição)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
