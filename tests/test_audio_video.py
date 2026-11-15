from unittest import mock

import numpy as np
import pytest

from medwatch.audio.analyzer import AudioAnalyzer
from medwatch.audio.azure_text import AzureTextAnalyzer, find_critical_terms
from medwatch.config import AzureSettings
from medwatch.data import synthetic as syn
from medwatch.schemas import Severity
from medwatch.video.analyzer import analyze_video, frame_metrics
from medwatch.video.objects import FrameObjects, instrument_count_check, point_in_polygon
from medwatch.video.pose import OPENPOSE18_TO_COCO17, OpenPoseEstimator

OFFLINE = AzureSettings(speech_key=None, language_key=None, language_endpoint=None)


# ---------------------------------------------------------------- texto
def test_critical_terms_and_negation():
    t = {d["term"]: d["negated"] for d in find_critical_terms("Não tenho febre, mas sinto falta de ar. Sem dor no peito.")}
    assert t == {"febre": True, "falta de ar": False, "dor no peito": True}


def test_offline_text_analysis():
    r = AzureTextAnalyzer(OFFLINE).analyze("Estou muito cansado e com falta de ar")
    assert r.source == "offline" and r.sentiment == "negative"
    assert r.category_scores["respiratorio"] >= 0.9 and "fadiga" in r.category_scores


def test_azure_language_client_is_used_when_configured():
    s = AzureSettings(language_key="k", language_endpoint="https://x.cognitiveservices.azure.com/")
    an = AzureTextAnalyzer(s)
    fake = mock.MagicMock()
    sent = mock.MagicMock(sentiment="negative", is_error=False)
    sent.confidence_scores.positive, sent.confidence_scores.neutral, sent.confidence_scores.negative = .05, .1, .85
    fake.analyze_sentiment.return_value = [sent]
    fake.extract_key_phrases.return_value = [mock.MagicMock(key_phrases=["falta de ar"], is_error=False)]
    fake.recognize_pii_entities.return_value = [mock.MagicMock(redacted_text="***", is_error=False)]
    ent = mock.MagicMock(text="falta de ar", category="SymptomOrSign", confidence_score=.9, normalized_text="Dyspnea")
    ent.assertion = None
    fake.begin_analyze_healthcare_entities.return_value.result.return_value = [mock.MagicMock(is_error=False, entities=[ent])]
    an._client = fake
    r = an.analyze("Estou com falta de ar")
    assert r.source == "azure" and r.sentiment == "negative"
    assert r.health_entities[0]["normalized"] == "Dyspnea"
    assert r.redacted_text == "***"


# ---------------------------------------------------------------- voz
@pytest.fixture(scope="module")
def audio_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("audio")
    syn.generate_audio(d)
    return d


@pytest.mark.parametrize("pid,expected", [("P001", "fadiga_vocal"), ("P002", "dificuldade_respiratoria"),
                                          ("P004", "disartria")])
def test_voice_conditions(audio_dir, pid, expected):
    rep, findings = AudioAnalyzer(OFFLINE).analyze(audio_dir / f"consulta_{pid}.wav", pid)
    assert rep.transcript.source == "sidecar"
    best = max(rep.condition_scores, key=lambda c: rep.condition_scores[c]["acoustic"])
    assert best == expected  # a acústica sozinha já aponta a condição certa
    assert any(f.kind == expected for f in findings)


def test_healthy_voice_has_no_findings(audio_dir):
    _, findings = AudioAnalyzer(OFFLINE).analyze(audio_dir / "consulta_P003.wav", "P003")
    assert findings == []


def test_dysarthria_with_neuro_complaint_is_critical(audio_dir):
    _, findings = AudioAnalyzer(OFFLINE).analyze(audio_dir / "consulta_P004.wav", "P004")
    assert next(f for f in findings if f.kind == "disartria").severity == Severity.CRITICO


# ---------------------------------------------------------------- vídeo
@pytest.fixture(scope="module")
def physio():
    kp, events = syn.generate_physio_keypoints()
    return analyze_video(kp, fps=15), events


def test_video_detects_all_injected_deviations(physio):
    res, events = physio
    found = {e.kind for e in res.all_events}
    for ev in events:
        assert ev["kind"] in found, ev
    fall = next(e for e in res.all_events if e.kind == "queda")
    assert 51 <= fall.start_s <= 54 and fall.severity == Severity.CRITICO


def test_video_normal_segment_is_clean(physio):
    res, _ = physio
    assert not [e for e in res.all_events if e.end_s < 23 and e.severity >= Severity.ATENCAO]
    assert len(res.repetitions) >= 12


def test_frame_metrics_angles():
    kp, _ = syn.generate_physio_keypoints()
    m = frame_metrics(kp[:30], 15)
    assert m.loc[0, "shoulder_abd_l"] < 30 and m["posture"].eq("em_pe").all()


def test_openpose_mapping_with_mocked_net(tmp_path):
    (tmp_path / "p.prototxt").write_text("x")
    (tmp_path / "w.caffemodel").write_text("x")
    import cv2

    with mock.patch.object(cv2.dnn, "readNetFromCaffe", create=True) as rd:
        heat = np.zeros((1, 57, 46, 46), np.float32)
        for i in range(18):
            heat[0, i, 10 + i, 20] = 0.9  # pico em y = 10+i
        rd.return_value.forward.return_value = heat
        est = OpenPoseEstimator(tmp_path / "p.prototxt", tmp_path / "w.caffemodel")
        p = est(np.zeros((460, 460, 3), np.uint8))
    assert p.shape == (17, 3)
    # l_shoulder (COCO 5) vem do OpenPose 5; r_shoulder (COCO 6) do OpenPose 2
    assert OPENPOSE18_TO_COCO17[5] == 5 and OPENPOSE18_TO_COCO17[6] == 2
    assert p[6, 1] == pytest.approx((10 + 2) * 10, abs=1)


def test_zones_and_instrument_count():
    assert point_in_polygon(0.5, 0.5, [[0, 0], [1, 0], [1, 1], [0, 1]])
    assert not point_in_polygon(1.5, 0.5, [[0, 0], [1, 0], [1, 1], [0, 1]])
    inst = {"cls": "scissors", "conf": .9, "box": [.4, .4, .5, .5]}
    frames = [FrameObjects(t, [inst, inst] if t < 5 else [inst]) for t in range(20)]
    assert instrument_count_check(frames) == {"start": 2, "end": 1, "mismatch": True}
