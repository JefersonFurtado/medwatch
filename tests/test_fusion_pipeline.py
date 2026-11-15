import json
from datetime import datetime, timedelta

from medwatch.alerts.dispatcher import AlertManager, JsonlChannel, WebhookChannel, finding_to_alert
from medwatch.fusion import fuse, patient_risk
from medwatch.schemas import Finding, Modality, Severity

T0 = datetime(2026, 10, 15, 10)


def F(mod, kind, sev=Severity.URGENTE, h=0.0, pid="P9", score=0.8):
    return Finding(pid, mod, kind, sev, score, kind, T0 + timedelta(hours=h))


def test_stroke_rule_fires_at_first_sufficient_evidence():
    fs = [F(Modality.VITALS, "crise_hipertensiva", h=0), F(Modality.VIDEO, "assimetria", Severity.ATENCAO, h=2),
          F(Modality.AUDIO, "disartria", Severity.CRITICO, h=3)]
    fused = fuse(fs)
    avc = next(f for f in fused if f.kind == "suspeita_avc")
    assert avc.timestamp == T0 + timedelta(hours=2)
    assert avc.severity == Severity.CRITICO
    assert "disartria" in avc.message  # evidência posterior anexada


def test_sepsis_requires_vitals():
    fs = [F(Modality.AUDIO, "fadiga_vocal"), F(Modality.MOVEMENT, "imobilidade_prolongada", h=1)]
    assert not [f for f in fuse(fs) if f.kind == "suspeita_sepse"]
    fs.append(F(Modality.VITALS, "deterioracao_clinica", Severity.CRITICO, h=2))
    assert [f for f in fuse(fs) if f.kind == "suspeita_sepse"]


def test_no_fusion_across_distant_events():
    fs = [F(Modality.VITALS, "crise_hipertensiva"), F(Modality.AUDIO, "disartria", h=30)]
    assert not fuse(fs)


def test_risk_decays():
    fs = [F(Modality.VITALS, "deterioracao_clinica", Severity.CRITICO, score=0.9)]
    assert patient_risk(fs, T0) > patient_risk(fs, T0 + timedelta(hours=12)) > 0


def test_alert_cooldown_and_escalation(tmp_path):
    mgr = AlertManager([JsonlChannel(tmp_path / "a.jsonl")], cooldown_min=15)
    assert mgr.dispatch(F(Modality.VITALS, "dessaturacao", Severity.ATENCAO))
    assert mgr.dispatch(F(Modality.VITALS, "dessaturacao", Severity.ATENCAO, h=0.1)) is None  # duplicado
    assert mgr.dispatch(F(Modality.VITALS, "dessaturacao", Severity.CRITICO, h=0.15))         # escalou
    assert mgr.dispatch(F(Modality.VITALS, "dessaturacao", Severity.INFO, h=2)) is None       # abaixo do mínimo
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["severity"] == "CRÍTICO"


def test_webhook_payload_teams_format():
    a = finding_to_alert(F(Modality.AUDIO, "disartria", Severity.CRITICO))
    p = WebhookChannel("http://x").payload(a)
    assert p["@type"] == "MessageCard" and p["themeColor"] == "C62828"
    assert "NIHSS" in p["text"]


def test_full_demo_pipeline(tmp_path, monkeypatch):
    from medwatch.config import Settings
    from medwatch.pipeline import run_demo

    s = Settings()
    s.output_dir = tmp_path
    s.alerts.jsonl_path = tmp_path / "alerts.jsonl"
    s.azure.speech_key = s.azure.language_key = None
    res = run_demo(s, console=False)
    for mod in ("sinais_vitais", "prescricao", "movimentacao", "audio", "video"):
        assert res.metrics[mod]["recall"] == 1.0, mod
    kinds = {(f.patient_id, f.kind) for f in res.fused}
    assert {("P004", "suspeita_avc"), ("P001", "suspeita_sepse"), ("P002", "insuficiencia_respiratoria")} <= kinds
    assert not [a for a in res.alerts if a.patient_id == "P003"]  # controle negativo
    assert (tmp_path / "painel_alertas.html").exists() and (tmp_path / "alerts.jsonl").exists()
