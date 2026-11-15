"""Geração e envio de alertas para a equipe médica.

* Converte `Finding`s em `Alert`s com ação recomendada por tipo de evento.
* **Deduplicação/cooldown** por (paciente, tipo): o mesmo evento não é
  reenviado dentro da janela, a menos que a severidade aumente — reduz a
  fadiga de alarmes, problema clássico de UTIs.
* Canais plugáveis: console, arquivo JSONL (auditoria) e webhook
  (Microsoft Teams / Slack / Azure Logic Apps, que pode encaminhar para
  e-mail, SMS via Azure Communication Services ou prontuário).
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

from ..schemas import Alert, Finding, Modality, Severity

log = logging.getLogger(__name__)

RECOMMENDED_ACTIONS = {
    "deterioracao_clinica": "Avaliação médica imediata à beira-leito (NEWS2 ≥ 7: acionar time de resposta rápida); investigar sepse, insuficiência respiratória ou choque.",
    "dessaturacao": "Checar via aérea e posicionamento do oxímetro; oferecer O₂; gasometria se persistir.",
    "crise_hipertensiva": "Confirmar PA manual; avaliar lesão de órgão-alvo; acionar médico plantonista.",
    "taquicardia": "ECG de 12 derivações; avaliar dor, febre, hipovolemia.",
    "hipotensao": "Avaliar perfusão; considerar prova volêmica conforme protocolo.",
    "desvio_do_basal": "Reavaliar o paciente; aumentar frequência de aferição dos sinais vitais.",
    "possivel_artefato": "Verificar posicionamento dos sensores.",
    "dose_acima_maxima": "Farmácia clínica: revisar dose antes da administração.",
    "interacao_medicamentosa": "Farmácia clínica: revisar associação; monitorar sangramento/efeito adverso.",
    "suspensao_abrupta": "Confirmar intenção de suspensão; avaliar desmame gradual.",
    "escalonamento_abrupto": "Confirmar prescrição com o prescritor; checar bomba de infusão.",
    "duplicidade_terapeutica": "Revisar prescrição — possível duplicidade.",
    "alteracoes_frequentes": "Discussão do caso pela equipe; revisar plano terapêutico.",
    "imobilidade_prolongada": "Mudança de decúbito e inspeção de pele (escala de Braden).",
    "agitacao_noturna": "Avaliar dor, delirium (CAM-ICU) e necessidade de acompanhante.",
    "agitacao": "Avaliar dor, ansiedade e delirium.",
    "saida_leito_noturna": "Enfermagem ao quarto; reforçar medidas de prevenção de quedas.",
    "queda": "Atendimento IMEDIATO no local; avaliar trauma e nível de consciência; notificar evento adverso.",
    "disartria": "Avaliação neurológica imediata (escala NIHSS); considerar protocolo de AVC.",
    "dificuldade_respiratoria": "Avaliar FR, SpO₂ e ausculta; considerar gasometria.",
    "fadiga_vocal": "Correlacionar com sinais vitais e exames; reavaliar estado geral.",
    "queixa_cardiovascular": "ECG e troponina conforme protocolo de dor torácica.",
    "queixa_infeccao": "Aferir temperatura; investigar foco infeccioso.",
    "amplitude_reduzida": "Fisioterapeuta: reavaliar dor/força do membro; ajustar plano.",
    "assimetria": "Fisioterapeuta: investigar déficit unilateral (força/sensibilidade).",
    "compensacao_tronco": "Fisioterapeuta: corrigir execução; reduzir carga/amplitude.",
    "movimento_irregular": "Interromper exercício; avaliar tremor, fadiga muscular ou hipoglicemia.",
    "zona_critica_objeto_proibido": "Remover objeto do campo estéril; registrar não conformidade.",
    "contagem_instrumentais": "Recontagem de instrumentais/compressas antes do fechamento (checklist OMS).",
    "suspeita_sepse": "Acionar time de resposta rápida; pacote de sepse da 1ª hora.",
    "suspeita_avc": "Acionar protocolo de AVC (código AVC): TC de crânio, tempo de início dos sintomas.",
    "insuficiencia_respiratoria": "Acionar médico; suporte ventilatório conforme gravidade.",
    "queda_com_instabilidade": "Avaliação médica imediata; investigar causa da queda (hipoxemia/arritmia/hipotensão).",
    "risco_multimodal": "Reavaliação clínica: múltiplas fontes de dados indicam piora.",
}


def _action_for(kind: str) -> str:
    for k, v in RECOMMENDED_ACTIONS.items():
        if kind == k or kind.startswith(k) or k in kind.split("+"):
            return v
    return "Avaliar o paciente e registrar conduta."


def finding_to_alert(f: Finding, related: list[Finding] | None = None) -> Alert:
    findings = [f] + (related or [])
    title = f"[{f.severity.label}] {f.patient_id} — {f.kind.replace('_', ' ').replace('+', ' + ')} ({f.modality})"
    return Alert(f.patient_id, f.severity, title, f.message, _action_for(f.kind), findings, f.score)


# --------------------------------------------------------------------------- #
# Canais
# --------------------------------------------------------------------------- #
class Channel(ABC):
    @abstractmethod
    def send(self, alert: Alert) -> None: ...


class ConsoleChannel(Channel):
    COLORS = {Severity.INFO: "\033[37m", Severity.ATENCAO: "\033[33m", Severity.URGENTE: "\033[91m",
              Severity.CRITICO: "\033[41;97m"}

    def __init__(self, use_color: bool = True):
        self.use_color = use_color

    def send(self, alert: Alert) -> None:
        c = self.COLORS[alert.severity] if self.use_color else ""
        r = "\033[0m" if self.use_color else ""
        when = alert.findings[0].detected_at
        print(f"{c}{when:%d/%m %H:%M} {alert.title}{r}\n    {alert.message}\n    ➜ {alert.recommended_action}")


class JsonlChannel(Channel):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")


class WebhookChannel(Channel):
    """Teams (Incoming Webhook / Workflows), Slack ou Azure Logic Apps (HTTP trigger)."""

    def __init__(self, url: str, timeout: float = 5.0):
        self.url, self.timeout = url, timeout

    def payload(self, alert: Alert) -> dict:
        color = {0: "808080", 1: "FFB000", 2: "E65100", 3: "C62828"}[int(alert.severity)]
        facts = [{"name": "Paciente", "value": alert.patient_id}, {"name": "Severidade", "value": alert.severity.label},
                 {"name": "Fontes", "value": ", ".join(alert.modalities)},
                 {"name": "Risco", "value": f"{alert.risk_score:.2f}"}]
        return {  # MessageCard — aceito por Teams; Slack/Logic Apps usam o campo "text"
            "@type": "MessageCard", "@context": "https://schema.org/extensions", "themeColor": color,
            "summary": alert.title, "title": alert.title, "text": f"{alert.message}\n\n**Conduta sugerida:** {alert.recommended_action}",
            "sections": [{"facts": facts}], "alert": alert.to_dict(),
        }

    def send(self, alert: Alert) -> None:
        import requests

        try:
            requests.post(self.url, json=self.payload(alert), timeout=self.timeout).raise_for_status()
        except Exception as exc:  # alerta não pode derrubar o pipeline
            log.error("Falha ao enviar webhook: %s", exc)


# --------------------------------------------------------------------------- #
# Gerenciador
# --------------------------------------------------------------------------- #
class AlertManager:
    def __init__(self, channels: list[Channel], cooldown_min: int = 15, min_severity: Severity = Severity.ATENCAO):
        self.channels = channels
        self.cooldown = timedelta(minutes=cooldown_min)
        self.min_severity = min_severity
        self._last: dict[tuple[str, str], tuple[datetime, Severity]] = {}
        self.sent: list[Alert] = []
        self.suppressed = 0

    def dispatch(self, finding: Finding, related: list[Finding] | None = None) -> Alert | None:
        if finding.severity < self.min_severity:
            return None
        # episódios de sinais vitais mudam de "tipo" ao evoluir; a chave é o episódio
        key = (finding.patient_id, finding.evidence.get("episode_id") or finding.kind)
        now = finding.detected_at
        last = self._last.get(key)
        if last and now - last[0] < self.cooldown and finding.severity <= last[1]:
            self.suppressed += 1
            return None
        if last and finding.severity <= last[1] and "episode_id" in finding.evidence:
            self.suppressed += 1  # mesmo episódio sem piora: não realertar
            return None
        self._last[key] = (now, max(finding.severity, last[1]) if last else finding.severity)
        alert = finding_to_alert(finding, related)
        for ch in self.channels:
            ch.send(alert)
        self.sent.append(alert)
        return alert

    def dispatch_all(self, findings: list[Finding]) -> list[Alert]:
        out = []
        for f in sorted(findings, key=lambda f: (f.detected_at, -int(f.severity))):
            a = self.dispatch(f, f.evidence.get("_related"))
            if a:
                out.append(a)
        return out


def build_alert_manager(settings, console: bool = True) -> AlertManager:
    channels: list[Channel] = [JsonlChannel(settings.alerts.jsonl_path)]
    if console:
        channels.insert(0, ConsoleChannel())
    if settings.alerts.webhook_url:
        channels.append(WebhookChannel(settings.alerts.webhook_url))
    return AlertManager(channels, settings.alerts.cooldown_minutes, Severity.parse(settings.alerts.min_severity_to_notify))


__all__ = ["AlertManager", "ConsoleChannel", "JsonlChannel", "WebhookChannel", "build_alert_manager",
           "finding_to_alert", "Modality"]
