"""Relatórios automáticos (Markdown + HTML + gráficos PNG)."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .schemas import Alert, Finding, Severity  # noqa: E402

SEV_COLOR = {Severity.INFO: "#9e9e9e", Severity.ATENCAO: "#f9a825", Severity.URGENTE: "#ef6c00", Severity.CRITICO: "#c62828"}


# --------------------------------------------------------------------------- #
# Vídeo
# --------------------------------------------------------------------------- #
def plot_video_timeline(res, path: Path, title: str) -> Path:
    m = res.metrics
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(m.t, m.shoulder_abd_l, label="Ombro E (abdução)", lw=1.2)
    axes[0].plot(m.t, m.shoulder_abd_r, label="Ombro D (abdução)", lw=1.2)
    axes[0].set_ylabel("graus"); axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(m.t, m.trunk_lean, color="#6a1b9a", lw=1.2, label="Inclinação do tronco")
    axes[1].plot(m.t, m.trunk_angle, color="#00838f", lw=1, ls="--", label="Ângulo tronco × vertical")
    axes[1].set_ylabel("graus"); axes[1].legend(loc="upper left", fontsize=8)
    axes[2].plot(m.t, m.tremor_index, color="#455a64", lw=1, label="Índice de tremor")
    axes[2].plot(m.t, m.hip_drop_speed.clip(lower=0), color="#d84315", lw=1, label="Velocidade de queda do quadril")
    axes[2].set_xlabel("tempo (s)"); axes[2].legend(loc="upper left", fontsize=8)
    ymax = axes[0].get_ylim()[1]
    for k, e in enumerate(res.all_events):
        for ax in axes:
            ax.axvspan(e.start_s, e.end_s, color=SEV_COLOR[e.severity], alpha=0.12)
        axes[0].text(e.start_s + 0.3, ymax * (0.97 - 0.09 * (k % 4)), e.kind.replace("_", " "), fontsize=7, va="top",
                     color="#333", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def video_report(res, patient_id: str, out_dir: Path, exercise: str | None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(res.source).stem
    png = plot_video_timeline(res, out_dir / f"video_{stem}_timeline.png", f"{patient_id} — {Path(res.source).name}")
    evs = res.all_events
    status = "DESVIOS DETECTADOS" if any(e.severity >= Severity.ATENCAO for e in evs) else "SEM DESVIOS"
    lines = [f"# Relatório automático de vídeo — {patient_id}", "",
             f"- **Arquivo:** `{Path(res.source).name}`  ",
             f"- **Modelo de pose:** {res.backend}  ",
             f"- **Exercício de referência:** {exercise or '—'}  ",
             f"- **Duração analisada:** {res.metrics.t.iloc[-1]:.1f} s ({len(res.metrics)} frames @ {res.fps:.1f} fps)  ",
             f"- **Gerado em:** {datetime.now():%d/%m/%Y %H:%M}  ", f"- **Resultado:** **{status}**", ""]
    if len(res.repetitions):
        r = res.repetitions
        lines += ["## Repetições", "", "| # | início (s) | pico E (°) | pico D (°) | assimetria (°) | tronco máx (°) |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {x.rep} | {x.start_s:.1f} | {x.peak_l:.0f} | {x.peak_r:.0f} | {x.asymmetry:.0f} | {x.max_trunk_lean:.0f} |"
                  for x in r.itertuples()]
        lines.append("")
    lines += ["## Desvios e eventos", ""]
    if evs:
        lines += ["| severidade | evento | intervalo | descrição |", "|---|---|---|---|"]
        lines += [f"| {e.severity.label} | {e.kind} | {e.start_s:.1f}–{e.end_s:.1f}s | {e.description} |" for e in evs]
    else:
        lines.append("Nenhum desvio detectado.")
    lines += ["", f"![Linha do tempo]({png.name})", ""]
    md = out_dir / f"video_{stem}_relatorio.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return {"markdown": md, "plot": png}


# --------------------------------------------------------------------------- #
# Sinais vitais / áudio
# --------------------------------------------------------------------------- #
def plot_vitals(ann: pd.DataFrame, findings: list[Finding], out_dir: Path) -> list[Path]:
    out = []
    labels = {"hr": "FC (bpm)", "sbp": "PAS (mmHg)", "spo2": "SpO₂ (%)", "rr": "FR (irpm)", "temp": "Temp (°C)"}
    for pid, g in ann.groupby("patient_id"):
        fig, axes = plt.subplots(len(labels) + 1, 1, figsize=(12, 10), sharex=True)
        for ax, (col, lab) in zip(axes, labels.items()):
            ax.plot(g.timestamp, g[col], lw=0.8, color="#37474f")
            ax.set_ylabel(lab, fontsize=8)
        axes[-1].plot(g.timestamp, g.news2, color="#5d4037", lw=0.8, label="NEWS2")
        axes[-1].plot(g.timestamp, g.iforest_score * 10, color="#00897b", lw=0.6, alpha=0.7, label="Isolation Forest (×10)")
        axes[-1].legend(fontsize=7, loc="upper left")
        for f in findings:
            if f.patient_id != pid or f.modality != "sinais_vitais":
                continue
            if f.severity == Severity.INFO:
                axes[2].axvline(f.timestamp, color="#9e9e9e", ls=":", lw=0.8)
                continue
            for ax in axes:
                ax.axvspan(f.timestamp, f.end_timestamp or f.timestamp, color=SEV_COLOR[f.severity], alpha=0.15)
        fig.suptitle(f"{pid} — sinais vitais e anomalias detectadas (faixas = episódios; pontilhado = artefato)")
        fig.tight_layout()
        p = out_dir / f"vitals_{pid}.png"
        fig.savefig(p, dpi=100); plt.close(fig)
        out.append(p)
    return out


def plot_voice(reports: list, out_dir: Path) -> Path:
    rows = {r.patient_id: r.indicator_scores for r in reports}
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(df.values, cmap="OrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(df.columns)), df.columns)
    ax.set_yticks(range(len(df.index)), df.index, fontsize=8)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="escore de alteração (0–1)")
    ax.set_title("Biomarcadores vocais por paciente")
    fig.tight_layout()
    p = out_dir / "audio_biomarcadores.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Painel HTML consolidado
# --------------------------------------------------------------------------- #
def alerts_html(alerts: list[Alert], metrics: dict, images: list[Path], path: Path) -> Path:
    rows = []
    for a in sorted(alerts, key=lambda a: (a.findings[0].detected_at, -int(a.severity))):
        t = a.findings[0].detected_at
        rows.append(f"<tr class='s{int(a.severity)}'><td>{t:%d/%m %H:%M}</td><td>{a.patient_id}</td>"
                    f"<td><b>{a.severity.label}</b></td><td>{html.escape(', '.join(a.modalities))}</td>"
                    f"<td><b>{html.escape(a.title.split('— ', 1)[-1])}</b><br>{html.escape(a.message)}"
                    f"<br><i>➜ {html.escape(a.recommended_action)}</i></td></tr>")
    met = "".join(f"<tr><td>{html.escape(k)}</td><td>{v.get('recall')}</td><td>{v.get('precision')}</td>"
                  f"<td>{v.get('f1')}</td><td>{v.get('mean_detection_delay_min')}</td></tr>"
                  for k, v in metrics.items() if isinstance(v, dict) and "recall" in v)
    imgs = "".join(f"<figure><img src='{p.name}'><figcaption>{p.stem}</figcaption></figure>" for p in images)
    doc = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8"><title>MedWatch — alertas</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border-bottom:1px solid #ddd;padding:6px;vertical-align:top;text-align:left}}tr.s3{{background:#ffebee}}
tr.s2{{background:#fff3e0}}tr.s1{{background:#fffde7}}img{{max-width:100%;border:1px solid #eee}}figure{{margin:16px 0}}</style>
</head><body><h1>MedWatch — monitoramento multimodal</h1><p>Gerado em {datetime.now():%d/%m/%Y %H:%M}. {len(alerts)} alertas.</p>
<h2>Desempenho por modalidade (dados de demonstração com ground truth)</h2>
<table><tr><th>modalidade</th><th>recall</th><th>precisão</th><th>F1</th><th>atraso médio (min)</th></tr>{met}</table>
<h2>Alertas enviados à equipe</h2><table><tr><th>evento</th><th>paciente</th><th>severidade</th><th>fontes</th><th>detalhe</th></tr>
{''.join(rows)}</table><h2>Gráficos</h2>{imgs}</body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path
