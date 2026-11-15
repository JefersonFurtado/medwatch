from datetime import datetime, timedelta

import pandas as pd
import pytest

from medwatch.anomaly.movement import detect_movement_anomalies
from medwatch.anomaly.prescriptions import detect_prescription_anomalies
from medwatch.anomaly.vitals import (StreamingVitalsMonitor, detect_vitals_anomalies, evaluate_event_detection,
                                     news2_components, news2_severity)
from medwatch.data import synthetic as syn
from medwatch.schemas import Severity


# ---------------------------------------------------------------- sinais vitais
def test_news2_bands():
    c = news2_components(hr=135, sbp=88, spo2=90, rr=26, temp=39.5)
    assert c == {"rr": 3, "spo2": 3, "sbp": 3, "hr": 3, "temp": 2}
    assert news2_severity(c) == Severity.CRITICO
    assert news2_severity(news2_components(hr=75, sbp=120, spo2=98, rr=14, temp=36.8)) == Severity.INFO
    assert news2_severity(news2_components(hr=75, sbp=190, spo2=98, rr=14, temp=36.8), sbp=190) == Severity.URGENTE


@pytest.fixture(scope="module")
def vitals_result():
    df, events = syn.generate_vitals()
    return detect_vitals_anomalies(df), events


def test_vitals_detects_all_injected_events(vitals_result):
    (episodes, realtime, ann), events = vitals_result
    m = evaluate_event_detection(episodes, events)
    assert m["recall"] == 1.0
    assert m["precision"] >= 0.8


def test_control_patient_has_no_alerts_and_artifacts_are_filtered(vitals_result):
    (episodes, realtime, ann), _ = vitals_result
    p3 = [f for f in episodes if f.patient_id == "P003"]
    assert all(f.severity == Severity.INFO for f in p3)
    assert any(f.kind == "possivel_artefato" for f in p3)


def test_streaming_emits_on_escalation_only():
    mon = StreamingVitalsMonitor("X", warmup=10, baseline_window=30, baseline_lag=5)
    t0 = datetime(2026, 1, 1)
    out = []
    for i in range(60):
        bad = i >= 40
        out += mon.update({"timestamp": t0 + timedelta(minutes=i), "hr": 125 if bad else 75, "sbp": 120,
                           "dbp": 75, "spo2": 88 if bad else 97, "rr": 26 if bad else 14, "temp": 36.8})
    assert out, "deveria alertar"
    assert len(out) <= 3  # abre + escalonamentos, não um alerta por amostra
    assert out[-1].severity == Severity.CRITICO


# ---------------------------------------------------------------- prescrições
def test_prescription_rules():
    rx, events = syn.generate_prescriptions()
    f = detect_prescription_anomalies(rx)
    kinds = {(x.patient_id, x.kind) for x in f}
    assert ("P004", "interacao_medicamentosa") in kinds
    assert ("P004", "suspensao_abrupta") in kinds
    assert ("P004", "dose_acima_maxima") in kinds
    assert ("P001", "escalonamento_abrupto") in kinds
    assert not [x for x in f if x.patient_id in ("P002", "P003")]
    assert evaluate_event_detection(f, events, tolerance_min=5)["recall"] == 1.0


# ---------------------------------------------------------------- movimentação
def test_movement_detects_fall_and_immobility():
    mv, events = syn.generate_movement()
    f = detect_movement_anomalies(mv)
    kinds = {(x.patient_id, x.kind) for x in f}
    assert ("P002", "queda") in kinds and ("P001", "imobilidade_prolongada") in kinds
    assert next(x for x in f if x.kind == "queda").severity == Severity.CRITICO
    assert evaluate_event_detection(f, events, tolerance_min=5)["recall"] == 1.0
    assert not [x for x in f if x.patient_id == "P003"]


def test_movement_no_false_fall_without_post_impact_immobility():
    t = pd.date_range("2026-01-01 10:00", periods=30, freq="1min")
    df = pd.DataFrame({"timestamp": t, "patient_id": "Z", "activity": 10.0, "position": "sentado", "impact_g": 0.0})
    df.loc[10, "impact_g"] = 5.0  # impacto, mas paciente continua ativo (ex.: sentou bruscamente)
    assert not [f for f in detect_movement_anomalies(df) if f.kind == "queda"]
