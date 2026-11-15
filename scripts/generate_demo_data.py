"""Gera (ou regenera) os dados de demonstração rotulados em data/demo/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medwatch.pipeline import prepare_demo_data  # noqa: E402

if __name__ == "__main__":
    for k, v in prepare_demo_data(force=True).items():
        print(f"{k:14s} {v}")
