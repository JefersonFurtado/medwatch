"""Painel da equipe médica — MedWatch.

Execute:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medwatch.config import get_settings  # noqa: E402
from medwatch.pipeline import DEMO_DIR, run_demo  # noqa: E402

st.set_page_config(page_title="MedWatch — Monitoramento Multimodal", page_icon="🩺", layout="wide")
settings = get_settings()
OUT = settings.output_dir
SEV_COLOR = {"INFO": "#9e9e9e", "ATENÇÃO": "#f9a825", "URGENTE": "#ef6c00", "CRÍTICO": "#c62828"}


@st.cache_data(show_spinner=False)
def load_outputs(stamp: float):
    alerts = [json.loads(line) for line in (OUT / "alerts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    findings = json.loads((OUT / "findings.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUT / "metricas.json").read_text(encoding="utf-8"))
    vitals = pd.read_csv(OUT / "vitals_anotado.csv", parse_dates=["timestamp"])
    risk = pd.read_csv(OUT / "risco_pacientes.csv", parse_dates=["timestamp"])
    audio = json.loads((OUT / "audio_relatorios.json").read_text(encoding="utf-8"))
    return alerts, findings, metrics, vitals, risk, audio


# --------------------------------------------------------------------------- #
st.sidebar.title("🩺 MedWatch")
st.sidebar.caption("Monitoramento multimodal de pacientes — vídeo, áudio, sinais vitais, prescrições e movimentação.")
st.sidebar.markdown(
    f"**Azure Speech:** {'🟢 conectado' if settings.azure.speech_enabled else '⚪ offline (fallback)'}  \n"
    f"**Azure Language:** {'🟢 conectado' if settings.azure.language_enabled else '⚪ offline (fallback)'}  \n"
    f"**Webhook de alertas:** {'🟢 configurado' if settings.alerts.webhook_url else '⚪ não configurado'}")
if st.sidebar.button("▶️ Executar pipeline completo", width="stretch"):
    with st.spinner("Processando todas as modalidades…"):
        run_demo(settings, console=False)
    st.cache_data.clear()
    st.sidebar.success("Pipeline concluído.")

if not (OUT / "findings.json").exists():
    st.info("Nenhum resultado ainda. Clique em **Executar pipeline completo** na barra lateral.")
    st.stop()

alerts, findings, metrics, vitals, risk, audio_reports = load_outputs((OUT / "findings.json").stat().st_mtime)
patients = sorted({a["patient_id"] for a in alerts} | set(vitals.patient_id.unique()))
tab_alert, tab_vit, tab_audio, tab_video, tab_rx, tab_flow = st.tabs(
    ["🚨 Central de alertas", "💓 Sinais vitais", "🎙️ Áudio", "🎥 Vídeo", "💊 Prescrições & movimentação", "🔁 Fluxo do alerta"])

# --------------------------------------------------------------------------- #
with tab_alert:
    sev_counts = pd.Series([a["severity"] for a in alerts]).value_counts()
    c = st.columns(5)
    c[0].metric("Alertas", len(alerts))
    for i, s in enumerate(["CRÍTICO", "URGENTE", "ATENÇÃO"]):
        c[i + 1].metric(s, int(sev_counts.get(s, 0)))
    c[4].metric("Suprimidos (cooldown)", metrics.get("alertas", {}).get("suprimidos_cooldown", 0))

    fig = px.line(risk, x="timestamp", y="risk", color="patient_id", title="Risco agregado por paciente (fusão multimodal)",
                  labels={"risk": "risco (0–1)", "timestamp": ""})
    fig.update_layout(height=320, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    f1, f2, f3 = st.columns(3)
    sel_p = f1.multiselect("Paciente", patients, default=patients)
    sel_s = f2.multiselect("Severidade", list(SEV_COLOR), default=["CRÍTICO", "URGENTE", "ATENÇÃO"])
    mods = sorted({m for a in alerts for m in a["modalities"]})
    sel_m = f3.multiselect("Fonte", mods, default=mods)
    shown = [a for a in alerts if a["patient_id"] in sel_p and a["severity"] in sel_s and set(a["modalities"]) & set(sel_m)]
    for a in sorted(shown, key=lambda a: a["event_time"], reverse=True):
        color = SEV_COLOR[a["severity"]]
        st.markdown(
            f"<div style='border-left:6px solid {color};padding:6px 12px;margin:6px 0;background:rgba(127,127,127,.06)'>"
            f"<b>{a['event_time'][5:16].replace('T', ' ')}</b> · <b style='color:{color}'>{a['severity']}</b> · "
            f"{a['title'].split('] ', 1)[-1]}<br><span style='font-size:0.9em'>{a['message']}</span><br>"
            f"<i style='font-size:0.9em'>➜ {a['recommended_action']}</i></div>", unsafe_allow_html=True)

    st.subheader("Desempenho dos detectores (dados de demonstração rotulados)")
    rows = [{"modalidade": k, **{m: v.get(m) for m in ["events", "detected", "alerts", "recall", "precision", "f1",
                                                         "mean_detection_delay_min"]}}
            for k, v in metrics.items() if isinstance(v, dict) and "recall" in v]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
with tab_vit:
    pid = st.selectbox("Paciente", sorted(vitals.patient_id.unique()), key="vit_p")
    g = vitals[vitals.patient_id == pid]
    cols = [("hr", "FC (bpm)"), ("sbp", "PAS (mmHg)"), ("spo2", "SpO₂ (%)"), ("rr", "FR (irpm)"), ("temp", "Temp (°C)"), ("news2", "NEWS2")]
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=[c[1] for c in cols])
    for i, (col, _) in enumerate(cols, 1):
        fig.add_trace(go.Scatter(x=g.timestamp, y=g[col], mode="lines", line=dict(width=1), name=col, showlegend=False), i, 1)
    for f in findings:
        if f["patient_id"] == pid and f["modality"] == "sinais_vitais":
            if f["severity"] == "INFO":
                fig.add_vline(x=f["timestamp"], line=dict(color="gray", dash="dot", width=1))
            else:
                fig.add_vrect(x0=f["timestamp"], x1=f["end_timestamp"] or f["timestamp"],
                              fillcolor=SEV_COLOR[f["severity"]], opacity=0.15, line_width=0)
    fig.update_layout(height=900, margin=dict(t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    ep = [f for f in findings if f["patient_id"] == pid and f["modality"] == "sinais_vitais"]
    st.dataframe(pd.DataFrame([{"início": f["timestamp"], "fim": f["end_timestamp"], "severidade": f["severity"],
                                "tipo": f["kind"], "descrição": f["message"]} for f in ep]), hide_index=True,
                 width="stretch")

# --------------------------------------------------------------------------- #
with tab_audio:
    st.caption("Azure Speech to Text → Azure AI Language (sentimento, frases-chave, entidades de saúde, PII) "
               "+ biomarcadores acústicos (jitter, shimmer, HNR, taxa silábica, pausas, respiração).")
    names = [r["patient_id"] for r in audio_reports]
    choice = st.radio("Consulta", names + ["📤 Enviar novo áudio"], horizontal=True)
    if choice == "📤 Enviar novo áudio":
        up = st.file_uploader("Arquivo WAV (mono, 16 kHz recomendado)", type=["wav"])
        pid_new = st.text_input("Paciente", "P010")
        if up and st.button("Analisar"):
            from medwatch.audio.analyzer import AudioAnalyzer

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(up.read())
            with st.spinner("Transcrevendo e analisando…"):
                rep, fs = AudioAnalyzer(settings.azure).analyze(tmp.name, pid_new)
            st.session_state["audio_live"] = (rep.to_dict(), [f.to_dict() for f in fs])
        r, fs = st.session_state.get("audio_live", (None, []))
    else:
        r = next(x for x in audio_reports if x["patient_id"] == choice)
        fs = [f for f in findings if f["patient_id"] == choice and f["modality"] == "audio"]
        wav = Path(r["audio_path"])
        if wav.exists():
            st.audio(str(wav))
    if r:
        c1, c2 = st.columns([3, 2])
        with c1:
            tr = r["transcript"]
            st.markdown(f"**Transcrição** ({tr['source']}):  \n> {tr['text'] or '—'}")
            ta = r["text_analysis"]
            st.markdown(f"**Sentimento:** `{ta['sentiment']}` {ta['sentiment_scores']}  \n"
                        f"**Frases-chave:** {', '.join(ta['key_phrases']) or '—'}  \n"
                        f"**Termos críticos:** " + (", ".join(f"{d['term']} ({d['category']}{', negado' if d['negated'] else ''})"
                                                             for d in ta["critical_terms"]) or "—"))
            if ta.get("health_entities"):
                st.dataframe(pd.DataFrame(ta["health_entities"]), hide_index=True)
            for f in fs:
                st.markdown(f"<div style='border-left:6px solid {SEV_COLOR[f['severity']]};padding:4px 10px;margin:4px 0'>"
                            f"<b>{f['severity']}</b> — {f['message']}</div>", unsafe_allow_html=True)
        with c2:
            ind = pd.Series(r["indicator_scores"]).rename("escore").reset_index().rename(columns={"index": "marcador"})
            st.plotly_chart(px.bar(ind, x="escore", y="marcador", orientation="h", range_x=[0, 1],
                                   title="Marcadores vocais (0 = normal, 1 = alterado)").update_layout(height=360),
                            width="stretch")
            st.json({k: v for k, v in r["condition_scores"].items()}, expanded=False)
            st.json(r["features"], expanded=False)

# --------------------------------------------------------------------------- #
with tab_video:
    st.caption("Pose com YOLOv8-pose ou OpenPose · objetos/zonas críticas com YOLOv8 · regras do exercício + Isolation Forest.")
    vids = sorted(OUT.glob("*_anotado.mp4"))
    left, right = st.columns([3, 2])
    with left:
        if vids:
            v = st.selectbox("Vídeo processado", vids, format_func=lambda p: p.name)
            st.video(str(v))
            md = OUT / f"video_{v.stem.replace('_anotado', '')}_relatorio.md"
            png = OUT / f"video_{v.stem.replace('_anotado', '')}_timeline.png"
            if png.exists():
                st.image(str(png))
    with right:
        if vids and md.exists():
            st.markdown(md.read_text(encoding="utf-8").split("![")[0])
        st.divider()
        st.markdown("**Analisar novo vídeo**")
        upv = st.file_uploader("MP4/AVI/MOV", type=["mp4", "avi", "mov"])
        ex = st.selectbox("Exercício de referência", ["(nenhum)", "abducao_ombro", "agachamento", "flexao_cotovelo"])
        objs = st.checkbox("Detectar objetos / zonas críticas (YOLOv8)", value=True)
        vp = st.text_input("Paciente", "P010", key="vid_pid")
        if upv and st.button("Processar vídeo"):
            from medwatch.pipeline import run_video

            dst = OUT / "uploads"
            dst.mkdir(exist_ok=True)
            path = dst / upv.name
            path.write_bytes(upv.read())
            with st.spinner("Estimando pose e detectando eventos…"):
                res, fs, rep = run_video(path, vp, datetime.now(), settings, OUT,
                                         exercise=None if ex == "(nenhum)" else ex, detect_objects=objs)
            from medwatch.alerts.dispatcher import build_alert_manager

            build_alert_manager(settings, console=False).dispatch_all(fs)
            st.success(f"{len(fs)} desvios detectados — relatório gerado.")
            st.cache_data.clear()
            st.rerun()

# --------------------------------------------------------------------------- #
with tab_rx:
    c1, c2 = st.columns(2)
    rx = pd.read_csv(DEMO_DIR / "prescriptions.csv") if (DEMO_DIR / "prescriptions.csv").exists() else pd.DataFrame()
    with c1:
        st.subheader("Prescrições — anomalias")
        st.dataframe(pd.DataFrame([{"hora": f["timestamp"][5:16], "paciente": f["patient_id"], "severidade": f["severity"],
                                    "tipo": f["kind"], "descrição": f["message"]}
                                   for f in findings if f["modality"] == "prescricao"]), hide_index=True, width="stretch")
        with st.expander("Histórico de prescrições"):
            st.dataframe(rx, hide_index=True)
    with c2:
        st.subheader("Movimentação no leito — anomalias")
        st.dataframe(pd.DataFrame([{"hora": f["timestamp"][5:16], "paciente": f["patient_id"], "severidade": f["severity"],
                                    "tipo": f["kind"], "descrição": f["message"]}
                                   for f in findings if f["modality"] == "movimentacao"]), hide_index=True, width="stretch")
        mv_path = DEMO_DIR / "movement.csv"
        if mv_path.exists():
            mv = pd.read_csv(mv_path, parse_dates=["timestamp"])
            pm = st.selectbox("Paciente", sorted(mv.patient_id.unique()), key="mv_p")
            st.plotly_chart(px.line(mv[mv.patient_id == pm], x="timestamp", y="activity", color_discrete_sequence=["#455a64"],
                                    title="Atividade motora (contagens/min)").update_layout(height=280), width="stretch")

# --------------------------------------------------------------------------- #
with tab_flow:
    st.markdown("""
**Do dado ao alerta**

1. **Ingestão** — monitor multiparamétrico (stream), prontuário/prescrição, sensor de leito, câmera, áudio da consulta.
2. **Análise por modalidade** — cada analisador gera *Findings* padronizados (paciente, tipo, severidade, escore, evidências).
3. **Fusão multimodal** — regras de correlação clínica (AVC, sepse, insuficiência respiratória, queda com instabilidade) e risco agregado.
4. **Gestão de alertas** — severidade mínima, deduplicação e cooldown por (paciente, tipo) para reduzir fadiga de alarmes.
5. **Entrega** — painel (este), log de auditoria JSONL e webhook (Microsoft Teams / Azure Logic Apps → e-mail/SMS/prontuário).
""")
    crit = [a for a in alerts if a["severity"] == "CRÍTICO"]
    if crit:
        a = st.selectbox("Exemplo de alerta", crit, format_func=lambda a: a["title"])
        from medwatch.alerts.dispatcher import WebhookChannel
        from medwatch.schemas import Alert, Finding, Severity

        st.json(a, expanded=False)
        if settings.alerts.webhook_url and st.button("📨 Enviar este alerta ao webhook (teste)"):
            fake = Alert(a["patient_id"], Severity.parse(a["severity_level"]), a["title"], a["message"],
                         a["recommended_action"],
                         [Finding(a["patient_id"], m, a["title"], Severity.parse(a["severity_level"]), a["risk_score"],
                                  a["message"], datetime.fromisoformat(a["event_time"])) for m in a["modalities"]],
                         a["risk_score"])
            WebhookChannel(settings.alerts.webhook_url).send(fake)
            st.success("Enviado.")
