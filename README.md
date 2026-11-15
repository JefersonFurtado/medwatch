# MedWatch — Monitoramento Multimodal de Pacientes

**Tech Challenge — Fase 4 · Pós-Tech IA para Devs (FIAP)**

Sistema que monitora pacientes continuamente a partir de **vídeo, áudio, sinais vitais, prescrições e
movimentação no leito**, detecta anomalias em cada fonte, **funde as evidências** e envia **alertas
priorizados à equipe médica**, integrando **Azure Speech to Text** e **Azure AI Language (Text Analytics)**.

```
┌ Vídeo ── YOLOv8-pose / OpenPose + YOLOv8 objetos ── desvios do exercício, queda, tremor, zonas críticas ┐
├ Áudio ── Azure Speech → Azure Language + biomarcadores (jitter, shimmer, HNR, pausas, respiração) ──────┤
├ Sinais vitais ── NEWS2 + desvio do basal + CUSUM + Isolation Forest (streaming) ───────────────────────┼─▶ Fusão ─▶ Alertas
├ Prescrições ── dose máxima, interações, suspensão abrupta, escalonamento, instabilidade ────────────────┤   multimodal   (painel, JSONL,
└ Movimentação ── imobilidade, agitação noturna, saída do leito, queda ──────────────────────────────────┘                Teams/Logic Apps)
```

| Requisito do desafio | Onde está |
|---|---|
| Vídeo: OpenPose para análise postural | `medwatch/video/pose.py` (`OpenPoseEstimator`) |
| Vídeo: YOLOv8 para objetos e áreas críticas | `medwatch/video/pose.py` (`YoloPoseEstimator`), `medwatch/video/objects.py` |
| Vídeo: relatório automático de desvios | `medwatch/video/analyzer.py`, `medwatch/reporting.py` → `outputs/video_*_relatorio.md` + vídeo anotado |
| Áudio: Azure Speech to Text | `medwatch/audio/azure_speech.py` |
| Áudio: Azure Text Analytics (termos críticos e sentimento) | `medwatch/audio/azure_text.py` |
| Áudio: alterações vocais (cansaço, dificuldade respiratória, disartria) | `medwatch/audio/features.py`, `medwatch/audio/analyzer.py` |
| Anomalias em séries temporais de sinais vitais | `medwatch/anomaly/vitals.py` |
| Anomalias na evolução de prescrições | `medwatch/anomaly/prescriptions.py` |
| Anomalias na movimentação durante a internação | `medwatch/anomaly/movement.py` |
| Alertas automáticos à equipe médica | `medwatch/alerts/dispatcher.py`, `app/dashboard.py` |
| Fusão de dados multimodais | `medwatch/fusion.py` |
| Datasets sugeridos (PhysioNet) | `medwatch/data/physionet.py`, `python -m medwatch physionet` |
| Relatório técnico | [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) |
| Roteiro do vídeo (≤ 15 min) | [`docs/roteiro_video.md`](docs/roteiro_video.md) |

## Instalação

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py                       # YOLOv8 (pose + detecção)
cp .env.example .env                                    # preencha as chaves Azure (opcional)
```

Opcional — **OpenPose** (modelo Caffe da CMU, uso acadêmico):

```bash
pip install "opencv-python<5"                           # o OpenCV 5 removeu o importador Caffe
python scripts/download_models.py --openpose
# depois use --backend openpose ou POSE_BACKEND=openpose
```

`ffmpeg` no PATH é recomendado para que os vídeos anotados toquem no navegador (H.264).

## Uso rápido

```bash
python -m medwatch demo            # executa TODO o fluxo com os dados de demonstração rotulados
streamlit run app/dashboard.py     # painel da equipe médica (alertas, vitais, áudio, vídeo, prescrições)
pytest -q                          # 27 testes
```

Outros comandos:

```bash
python -m medwatch video sessao.mp4 --patient P010 --exercise abducao_ombro --objects
python -m medwatch video cirurgia.mp4 --patient P011 --objects          # zonas críticas / instrumentais
python -m medwatch audio consulta.wav --patient P010                    # Azure Speech + Language
python -m medwatch vitals meus_sinais.csv                               # colunas: timestamp,patient_id,hr,sbp,dbp,spo2,rr,temp
python -m medwatch stream --speed 1200                                  # sinais vitais "ao vivo" com alertas no console
python -m medwatch physionet --subject 1                                # BIDMC (PhysioNet), dados reais
python scripts/check_azure.py minha_gravacao.wav                        # testa as credenciais Azure
```

## Configuração do Azure

1. No portal Azure crie um recurso **Speech** e um recurso **Language** (região *Brazil South* ou *East US*;
   o plano gratuito F0 é suficiente para a demonstração).
2. Copie chave/região/endpoint para o `.env` (veja `.env.example`).
3. Rode `python scripts/check_azure.py` — deve aparecer `fonte=azure`.

Sem as chaves, o sistema **não quebra**: usa a transcrição em `.txt` ao lado do áudio e uma análise léxica
local (mesmo léxico de termos críticos). O painel mostra na barra lateral qual modo está ativo.

**Alertas por e-mail (Logic App + Gmail):** [`infra/logicapp_alertas.json`](infra/logicapp_alertas.json) cria
um Logic App com gatilho HTTP que responde `202` e envia e-mail para alertas a partir de `minEmailSeverity`
(1=ATENÇÃO, 2=URGENTE, 3=CRÍTICO; padrão 3 — os demais ficam no histórico de execuções).

```bash
az deployment group create -g <rg> --template-file infra/logicapp_alertas.json --parameters recipient=<email>
# 1ª vez: o deploy do workflow falha (GmailConnectorPolicyViolation) até a conexão ser autorizada:
#   portal → API connection "gmail-medwatch" → Edit API connection → Authorize (marque TODAS as permissões) → Save
#   e rode o deploy de novo. Depois copie a URL do gatilho para ALERT_WEBHOOK_URL:
az rest --method post --uri "https://management.azure.com/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Logic/workflows/la-medwatch-alertas/triggers/manual/listCallbackUrl?api-version=2016-06-01" --query value -o tsv
```

**Alertas no Teams:** alternativamente, crie um *Workflow* "Post to a channel when a webhook request is
received" e coloque a URL em `ALERT_WEBHOOK_URL`.

## Dados de demonstração

`data/demo/` contém 24 h de 4 pacientes com eventos **injetados e rotulados** (`ground_truth.json`), o que
permite medir recall/precisão de cada detector:

| Paciente | Cenário |
|---|---|
| P001 | Deterioração séptica progressiva, imobilidade de 4 h, voz com fadiga, escalonamento de vasopressor |
| P002 | 3 episódios de dessaturação, agitação noturna + queda, voz com esforço respiratório |
| P003 | **Controle negativo** — estável, com artefatos de sensor que **não** devem alertar |
| P004 | Crise hipertensiva, fala disártrica, déficit motor no braço direito na fisioterapia, interação medicamentosa |

Para dados reais: sinais vitais do PhysioNet (`python -m medwatch physionet`), áudio de consultas gravadas
(WAV) e vídeos de fisioterapia/cirurgia (MP4) pelos comandos `audio`/`video` ou pelo painel.

## Estrutura

```
medwatch/
  config.py, schemas.py         configuração (.env) e o formato comum Finding/Alert
  data/                         dados sintéticos rotulados + carregador PhysioNet
  video/                        pose (YOLOv8/OpenPose), objetos/zonas (YOLOv8), análise e vídeo anotado
  audio/                        biomarcadores vocais, Azure Speech, Azure Language, analisador
  anomaly/                      sinais vitais (streaming), prescrições, movimentação
  fusion.py                     correlação clínica entre modalidades e risco agregado
  alerts/dispatcher.py          deduplicação, cooldown, console/JSONL/webhook
  pipeline.py, cli.py           orquestração e linha de comando
app/dashboard.py                painel Streamlit
configs/zones.json              zonas críticas por sala (polígonos normalizados)
docs/                           relatório técnico e roteiro do vídeo
tests/                          testes automatizados
```

> ⚠️ Projeto acadêmico. Limiares e bases farmacológicas são simplificados e **não** substituem validação
> clínica, farmacêutica ou regulatória (ANVISA/RDC 657/2022 para software como dispositivo médico).
