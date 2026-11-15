"""Carregadores de dados reais do PhysioNet (https://physionet.org/).

Requer `wfdb` e acesso à internet (ou os arquivos baixados localmente).

Datasets suportados
-------------------
* BIDMC PPG and Respiration (`bidmc/1.0.0`): registros *numerics* com HR,
  PULSE, RESP e SpO2 a 1 Hz — ideal para a detecção de anomalias em séries
  temporais de sinais vitais.
* MIMIC-III Waveform Database Matched Subset (`mimic3wdb-matched/1.0`):
  registros numéricos com HR, ABPSys/ABPDias/ABPMean, SpO2, RESP.

Exemplo
-------
>>> df = load_bidmc_numerics(1)                         # baixa do PhysioNet
>>> df = load_bidmc_numerics(1, local_dir="data/raw/bidmc")  # arquivos locais
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# mapeamento nomes PhysioNet -> nomes do MedWatch
SIGNAL_MAP = {
    "hr": "hr", "pulse": "hr_pulse", "resp": "rr", "spo2": "spo2", "%spo2": "spo2",
    "abpsys": "sbp", "abp sys": "sbp", "nbpsys": "sbp", "nbp sys": "sbp",
    "abpdias": "dbp", "abp dias": "dbp", "nbpdias": "dbp", "nbp dias": "dbp",
    "temp": "temp", "tblood": "temp",
}


def _record_to_df(record, patient_id: str, start: datetime | None) -> pd.DataFrame:
    start = start or datetime(2026, 10, 15)
    fs = float(record.fs)
    n = record.p_signal.shape[0]
    ts = [start + timedelta(seconds=i / fs) for i in range(n)]
    df = pd.DataFrame({"timestamp": ts, "patient_id": patient_id})
    for i, name in enumerate(record.sig_name):
        key = SIGNAL_MAP.get(name.strip().lower())
        if key and key not in df:
            df[key] = record.p_signal[:, i]
    if "hr" not in df and "hr_pulse" in df:
        df["hr"] = df["hr_pulse"]
    return df.drop(columns=[c for c in ["hr_pulse"] if c in df])


def load_bidmc_numerics(subject: int, local_dir: str | Path | None = None,
                        resample: str | None = "1min") -> pd.DataFrame:
    import wfdb

    rec_name = f"bidmc{subject:02d}n"
    if local_dir:
        record = wfdb.rdrecord(str(Path(local_dir) / rec_name))
    else:
        record = wfdb.rdrecord(rec_name, pn_dir="bidmc/1.0.0")
    df = _record_to_df(record, f"BIDMC{subject:02d}", None)
    return _resample(df, resample)


def load_mimic3_numerics(record_path: str, pn_dir: str = "mimic3wdb-matched/1.0",
                         patient_id: str | None = None, resample: str | None = "1min") -> pd.DataFrame:
    """`record_path` ex.: 'p00/p000020/p000020-2183-04-28-17-47n'."""
    import wfdb

    directory, rec = record_path.rsplit("/", 1)
    record = wfdb.rdrecord(rec, pn_dir=f"{pn_dir}/{directory}")
    df = _record_to_df(record, patient_id or rec.split("-")[0], None)
    return _resample(df, resample)


def _resample(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if not rule:
        return df
    pid = df["patient_id"].iloc[0]
    out = df.set_index("timestamp").drop(columns="patient_id").resample(rule).median().reset_index()
    out.insert(1, "patient_id", pid)
    return out
