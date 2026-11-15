"""Geradores de dados sintéticos com *ground truth* rotulado.

Os dados reais de hospital não podem ser publicados. Para demonstrar e
avaliar o sistema de forma reprodutível geramos dados fisiologicamente
plausíveis e injetamos eventos conhecidos (com início/fim). Com isso
conseguimos medir precisão/recall dos detectores. Para dados reais, ver
`medwatch.data.physionet`.

Pacientes de demonstração
-------------------------
P001  Deterioração progressiva (padrão séptico) + imobilidade prolongada +
      voz com fadiga + escalonamento abrupto de prescrição.
P002  Episódios de dessaturação/taquicardia + saída do leito à noite e queda
      + voz com dificuldade respiratória.
P003  Paciente estável com artefatos de sensor (controle negativo).
P004  Crise hipertensiva + fala com padrão disártrico + interação
      medicamentosa.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

START = datetime(2026, 10, 15, 0, 0)
PATIENTS = ["P001", "P002", "P003", "P004"]

BASELINES = {
    #        hr   sbp  dbp  spo2  rr   temp
    "P001": (78, 124, 78, 97.0, 15, 36.7),
    "P002": (84, 118, 74, 95.5, 17, 36.6),
    "P003": (70, 120, 76, 97.5, 14, 36.5),
    "P004": (76, 138, 86, 96.5, 15, 36.6),
}


@dataclass
class Event:
    patient_id: str
    modality: str
    kind: str
    start: datetime
    end: datetime

    def to_dict(self) -> dict:
        return {"patient_id": self.patient_id, "modality": self.modality, "kind": self.kind,
                "start": self.start.isoformat(), "end": self.end.isoformat()}


def _ramp(n: int, i0: int, i1: int, amp: float, hold: int = 0) -> np.ndarray:
    """Rampa suave de 0 até `amp` entre i0 e i1, mantida por `hold` amostras e depois retorna."""
    x = np.zeros(n)
    i1 = min(i1, n)
    x[i0:i1] = np.linspace(0, amp, i1 - i0)
    h1 = min(i1 + hold, n)
    x[i1:h1] = amp
    back = min(h1 + max((i1 - i0) // 2, 1), n)
    if back > h1:
        x[h1:back] = np.linspace(amp, 0, back - h1)
    return x


# --------------------------------------------------------------------------- #
# Sinais vitais
# --------------------------------------------------------------------------- #
def generate_vitals(hours: int = 24, freq_min: int = 1, seed: int = 42) -> tuple[pd.DataFrame, list[Event]]:
    rng = np.random.default_rng(seed)
    n = hours * 60 // freq_min
    t = np.arange(n)
    ts = [START + timedelta(minutes=int(i * freq_min)) for i in t]
    circ = np.sin(2 * np.pi * (t * freq_min / 60 - 4) / 24)  # mínimo ~04h
    frames, events = [], []

    def idx(h: float) -> int:
        return int(h * 60 / freq_min)

    for pid in PATIENTS:
        hr0, sbp0, dbp0, spo20, rr0, temp0 = BASELINES[pid]
        hr = hr0 + 5 * circ + _ar1(n, rng, 2.0)
        sbp = sbp0 + 6 * circ + _ar1(n, rng, 3.0)
        dbp = dbp0 + 3 * circ + _ar1(n, rng, 2.0)
        spo2 = spo20 + _ar1(n, rng, 0.5)
        rr = rr0 + 1 * circ + _ar1(n, rng, 0.8)
        temp = temp0 + 0.25 * circ + _ar1(n, rng, 0.05, phi=0.98)

        if pid == "P001":  # deterioração séptica a partir das 14h
            a, b = idx(14), idx(20)
            hr += _ramp(n, a, b, 38, hold=n)
            rr += _ramp(n, a, b, 9, hold=n)
            temp += _ramp(n, a, b, 2.0, hold=n)
            sbp -= _ramp(n, idx(17), idx(21), 32, hold=n)
            dbp -= _ramp(n, idx(17), idx(21), 18, hold=n)
            spo2 -= _ramp(n, idx(18), idx(22), 4, hold=n)
            events.append(Event(pid, "sinais_vitais", "deterioracao_septica", ts[a], ts[-1]))
        elif pid == "P002":  # 3 episódios de dessaturação + taquicardia reflexa
            for h, dur in [(3.0, 25), (9.5, 40), (16.2, 30)]:
                a = idx(h)
                b = a + dur // freq_min
                bump = np.zeros(n)
                bump[a:b] = np.hanning(b - a)
                spo2 -= 10 * bump
                hr += 28 * bump
                rr += 8 * bump
                events.append(Event(pid, "sinais_vitais", "dessaturacao", ts[a], ts[min(b, n - 1)]))
        elif pid == "P003":  # artefatos de sensor (pico de 1–2 amostras)
            for h in [5.1, 11.7, 19.3]:
                a = idx(h)
                spo2[a] = 72.0
                hr[a: a + 2] = 168.0
            # sem evento clínico (controle negativo)
        elif pid == "P004":  # crise hipertensiva 10h–13h
            a, b = idx(10), idx(11.5)
            sbp += _ramp(n, a, b, 62, hold=idx(1.5))
            dbp += _ramp(n, a, b, 30, hold=idx(1.5))
            hr += _ramp(n, a, b, 12, hold=idx(1.5))
            events.append(Event(pid, "sinais_vitais", "crise_hipertensiva", ts[a], ts[idx(14.2)]))

        df = pd.DataFrame({
            "timestamp": ts, "patient_id": pid,
            "hr": np.round(hr, 0), "sbp": np.round(sbp, 0), "dbp": np.round(dbp, 0),
            "spo2": np.round(np.clip(spo2, 50, 100), 1), "rr": np.round(np.clip(rr, 5, 60), 0),
            "temp": np.round(temp, 1),
        })
        # alguns valores faltantes (desconexão de sensor)
        miss = rng.choice(n, size=n // 200, replace=False)
        df.loc[miss, ["spo2"]] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True), events


def _ar1(n: int, rng: np.random.Generator, sigma: float, phi: float = 0.9) -> np.ndarray:
    e = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# --------------------------------------------------------------------------- #
# Prescrições
# --------------------------------------------------------------------------- #
def generate_prescriptions() -> tuple[pd.DataFrame, list[Event]]:
    h = lambda x: START + timedelta(hours=x)  # noqa: E731
    rows = [
        # P001 — escalonamento abrupto de noradrenalina e troca de antibiótico em sequência
        ("P001", h(-48), "ceftriaxona", 1000, "mg", "IV", 2, "Dr. Silva"),
        ("P001", h(-48), "dipirona", 1000, "mg", "IV", 4, "Dr. Silva"),
        ("P001", h(-48), "enoxaparina", 40, "mg", "SC", 1, "Dr. Silva"),
        ("P001", h(18.5), "meropenem", 1000, "mg", "IV", 3, "Dr. Costa"),
        ("P001", h(18.6), "ceftriaxona", 0, "mg", "IV", 0, "Dr. Costa"),
        ("P001", h(19.0), "noradrenalina", 8, "mg", "IV", 1, "Dr. Costa"),
        ("P001", h(20.0), "noradrenalina", 24, "mg", "IV", 1, "Dr. Costa"),
        ("P001", h(21.0), "dipirona", 1000, "mg", "IV", 6, "Dr. Costa"),
        # P002 — estável, só ajustes pequenos
        ("P002", h(-24), "salbutamol", 5, "mg", "INAL", 4, "Dra. Lima"),
        ("P002", h(-24), "prednisona", 40, "mg", "VO", 1, "Dra. Lima"),
        ("P002", h(12), "salbutamol", 5, "mg", "INAL", 6, "Dra. Lima"),
        # P003 — controle (sem anomalia)
        ("P003", h(-72), "omeprazol", 20, "mg", "VO", 1, "Dr. Souza"),
        ("P003", h(-72), "losartana", 50, "mg", "VO", 1, "Dr. Souza"),
        ("P003", h(8), "losartana", 50, "mg", "VO", 1, "Dr. Souza"),
        # P004 — dose acima do máximo + interação + suspensão abrupta de betabloqueador
        ("P004", h(-72), "metoprolol", 100, "mg", "VO", 2, "Dr. Rocha"),
        ("P004", h(-72), "varfarina", 5, "mg", "VO", 1, "Dr. Rocha"),
        ("P004", h(9.0), "metoprolol", 0, "mg", "VO", 0, "Dr. Alves"),
        ("P004", h(11.0), "acido acetilsalicilico", 100, "mg", "VO", 1, "Dr. Alves"),
        ("P004", h(11.2), "captopril", 50, "mg", "VO", 4, "Dr. Alves"),
    ]
    df = pd.DataFrame(rows, columns=["patient_id", "timestamp", "drug", "dose", "unit", "route",
                                     "doses_per_day", "prescriber"])
    events = [
        Event("P001", "prescricao", "escalonamento_abrupto", h(20.0), h(20.0)),
        Event("P001", "prescricao", "alteracoes_frequentes", h(18.5), h(21.0)),
        Event("P001", "prescricao", "dose_acima_maxima", h(21.0), h(21.0)),
        Event("P004", "prescricao", "suspensao_abrupta", h(9.0), h(9.0)),
        Event("P004", "prescricao", "interacao_medicamentosa", h(11.0), h(11.0)),
        Event("P004", "prescricao", "dose_acima_maxima", h(11.2), h(11.2)),
    ]
    return df, events


# --------------------------------------------------------------------------- #
# Movimentação no leito (sensor de pressão/acelerômetro, 1 amostra/min)
# --------------------------------------------------------------------------- #
POSITIONS = ["supino", "lateral_esq", "lateral_dir", "sentado", "fora_do_leito"]


def generate_movement(hours: int = 24, seed: int = 7) -> tuple[pd.DataFrame, list[Event]]:
    rng = np.random.default_rng(seed)
    n = hours * 60
    ts = [START + timedelta(minutes=i) for i in range(n)]
    hour = (np.arange(n) / 60) % 24
    frames, events = [], []
    for pid in PATIENTS:
        night = (hour >= 22) | (hour < 6)
        base = np.where(night, 2.0, 12.0)
        activity = rng.gamma(2.0, base / 2.0)
        pos = []
        current = "supino"
        for i in range(n):
            if i % rng.integers(90, 150) == 0:
                current = rng.choice(POSITIONS[:3]) if night[i] else rng.choice(POSITIONS[:4], p=[.4, .2, .2, .2])
            pos.append(current)
        pos = np.array(pos, dtype=object)
        impact = np.zeros(n)

        if pid == "P001":  # imobilidade de 4h (risco de lesão por pressão), 15h–19h
            a, b = 15 * 60, 19 * 60
            activity[a:b] = rng.uniform(0, 0.4, b - a)
            pos[a:b] = "supino"
            events.append(Event(pid, "movimentacao", "imobilidade_prolongada", ts[a], ts[b - 1]))
        elif pid == "P002":  # agitação noturna + saída do leito às 02h40 + queda
            a = int(2.2 * 60)
            activity[a:a + 30] = rng.uniform(25, 45, 30)
            events.append(Event(pid, "movimentacao", "agitacao_noturna", ts[a], ts[a + 29]))
            b = int(2 * 60 + 40)
            pos[b:b + 12] = "fora_do_leito"
            events.append(Event(pid, "movimentacao", "saida_leito_noturna", ts[b], ts[b + 11]))
            impact[b + 8] = 6.5  # g
            activity[b + 9:b + 20] = rng.uniform(0, 0.3, 11)
            events.append(Event(pid, "movimentacao", "queda", ts[b + 8], ts[b + 19]))
        frames.append(pd.DataFrame({"timestamp": ts, "patient_id": pid,
                                    "activity": np.round(activity, 2), "position": pos,
                                    "impact_g": np.round(impact, 2)}))
    return pd.concat(frames, ignore_index=True), events


# --------------------------------------------------------------------------- #
# Áudio (voz sintetizada com fonte glotal + formantes)
# --------------------------------------------------------------------------- #
TRANSCRIPTS = {
    "P001": "Doutor, eu estou muito cansado, sem força nenhuma desde ontem. Sinto calafrios e "
            "um pouco de febre à noite. Não consegui comer direito.",
    "P002": "Estou com falta de ar quando ando até o banheiro. A respiração fica curta e "
            "sinto um aperto no peito. Tenho tosse à noite.",
    "P003": "Estou me sentindo bem melhor hoje, dormi bem e a dor passou. Já consigo "
            "caminhar pelo corredor sem problema.",
    "P004": "Minha fala está enrolada desde cedo, a boca ficou dormente do lado direito e "
            "estou com dor de cabeça forte e tontura.",
}

# perfis: f0, jitter, shimmer, ruído (aspiração), sílabas/s, pausa média, decaimento ao longo do áudio
VOICE_PROFILES = {
    "P001": dict(f0=118, jitter=0.012, shimmer=0.09, noise=0.10, rate=3.4, pause=0.45, decay=0.45, breath=0.0),   # fadiga
    "P002": dict(f0=122, jitter=0.010, shimmer=0.08, noise=0.30, rate=4.0, pause=0.55, decay=0.10, breath=0.9),   # respiratório
    "P003": dict(f0=120, jitter=0.005, shimmer=0.035, noise=0.05, rate=4.6, pause=0.25, decay=0.02, breath=0.0),  # saudável
    "P004": dict(f0=110, jitter=0.032, shimmer=0.18, noise=0.18, rate=2.2, pause=0.85, decay=0.0, breath=0.0),   # disártrico
}


def synth_voice(profile: dict, duration: float = 14.0, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """Sintetiza fala vocálica (sem conteúdo lexical) com características controladas."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    out = np.zeros(n)
    t_pos = 0.25
    vowels = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410)]
    while t_pos < duration - 0.5:
        frac = t_pos / duration
        # frase: 4–8 sílabas, depois pausa
        n_syl = rng.integers(4, 9)
        for _ in range(n_syl):
            syl_dur = max(0.08, rng.normal(1.0 / profile["rate"], 0.03))
            i0, i1 = int(t_pos * sr), int(min(t_pos + syl_dur, duration) * sr)
            if i1 - i0 < 200:
                break
            f0 = profile["f0"] * (1 - 0.18 * profile["decay"] * frac) * (1 + rng.normal(0, 0.03))
            seg = _glottal(i1 - i0, sr, f0, profile["jitter"], profile["shimmer"], rng)
            F = vowels[rng.integers(len(vowels))]
            for fc, bw in zip(F, (80, 100, 120)):
                r = np.exp(-np.pi * bw / sr)
                th = 2 * np.pi * fc / sr
                seg = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], seg)
            seg = seg / (np.abs(seg).max() + 1e-9)
            seg += profile["noise"] * rng.normal(0, 0.3, len(seg))
            from scipy.signal.windows import tukey
            env = tukey(len(seg), 0.5)
            amp = 0.6 * (1 - 0.6 * profile["decay"] * frac)
            out[i0:i1] += amp * env * seg
            t_pos += syl_dur
        pause = max(0.12, rng.normal(profile["pause"], 0.08))
        j0, j1 = int(t_pos * sr), int(min(t_pos + pause * 0.8, duration) * sr)
        if profile["breath"] > 0 and pause > 0.3 and j1 - j0 > 400:  # inspiração audível (ruído filtrado)
            br = rng.normal(0, 1, j1 - j0)
            br = lfilter([1, -0.6], [1], br)
            out[j0:j1] += 0.10 * profile["breath"] * np.hanning(j1 - j0) * br
        t_pos += pause
    out += rng.normal(0, 0.002, n)
    return (out / (np.abs(out).max() + 1e-9) * 0.8).astype(np.float32)


def _glottal(n: int, sr: int, f0: float, jitter: float, shimmer: float, rng) -> np.ndarray:
    sig = np.zeros(n)
    i = 0
    while i < n:
        period = sr / (f0 * (1 + rng.normal(0, jitter)))
        a = 1 + rng.normal(0, shimmer)
        L = int(period)
        if L <= 2:
            break
        k = np.arange(L)
        pulse = np.where(k < 0.6 * L, 0.5 * (1 - np.cos(np.pi * k / (0.6 * L))), np.cos(np.pi * (k - 0.6 * L) / (0.8 * L)))
        end = min(i + L, n)
        sig[i:end] += a * np.diff(np.concatenate([[0], pulse]))[: end - i]
        i += L
    return sig


def generate_audio(out_dir: Path, sr: int = 16000) -> list[Event]:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    events = []
    labels = {"P001": "fadiga_vocal", "P002": "dificuldade_respiratoria", "P004": "disartria"}
    for i, pid in enumerate(PATIENTS):
        y = synth_voice(VOICE_PROFILES[pid], sr=sr, seed=100 + i)
        sf.write(out_dir / f"consulta_{pid}.wav", y, sr)
        (out_dir / f"consulta_{pid}.txt").write_text(TRANSCRIPTS[pid], encoding="utf-8")
        if pid in labels:
            t = START + timedelta(hours=13)
            events.append(Event(pid, "audio", labels[pid], t, t))
    return events


# --------------------------------------------------------------------------- #
# Vídeo de fisioterapia — sequência de keypoints COCO-17 (formato do YOLOv8-pose)
# --------------------------------------------------------------------------- #
COCO_KP = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow",
           "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def generate_physio_keypoints(fps: int = 15, seed: int = 3) -> tuple[np.ndarray, list[dict]]:
    """Sessão de abdução de ombro bilateral (paciente em pé), 60 s.

    0–24 s  : 6 repetições normais (abdução ~160°, simétrica)
    24–40 s : amplitude reduzida no braço direito + inclinação compensatória do tronco
    40–48 s : tremor/movimento brusco
    52–54 s : queda (quadril desce rapidamente e corpo fica horizontal)
    Retorna array (T, 17, 3) [x, y, conf] em pixels (frame 640x480) e eventos rotulados (segundos).
    """
    rng = np.random.default_rng(seed)
    T = 60 * fps
    kp = np.zeros((T, 17, 3))
    kp[..., 2] = 0.9
    cx, shoulder_y, hip_y, ankle_y = 320.0, 170.0, 290.0, 440.0
    arm = 115.0
    for f in range(T):
        s = f / fps
        phase = 2 * np.pi * s / 4.0                     # 1 repetição a cada 4 s
        lift = (1 - np.cos(phase)) / 2 if s < 52 else 0.0  # 0..1 (braços caídos após a queda)
        ang_l = np.deg2rad(10 + 150 * lift)
        ang_r = ang_l
        lean = 0.0
        if 24 <= s < 40:
            ang_r = np.deg2rad(10 + 70 * lift)           # ROM reduzida
            lean = np.deg2rad(14 * lift)                 # compensação de tronco
        trem = rng.normal(0, 9, 2) if 40 <= s < 48 else np.zeros(2)
        fall = np.clip((s - 52) / 1.2, 0, 1) if s >= 52 else 0.0

        sx = cx + np.sin(lean) * 120
        pts = {}
        pts["l_hip"], pts["r_hip"] = (cx + 30, hip_y), (cx - 30, hip_y)
        pts["l_shoulder"], pts["r_shoulder"] = (sx + 45, shoulder_y), (sx - 45, shoulder_y)
        pts["nose"] = (sx, shoulder_y - 55)
        pts["l_eye"], pts["r_eye"] = (sx + 10, shoulder_y - 62), (sx - 10, shoulder_y - 62)
        pts["l_ear"], pts["r_ear"] = (sx + 20, shoulder_y - 58), (sx - 20, shoulder_y - 58)
        for side, ang, sign in (("l", ang_l, 1), ("r", ang_r, -1)):
            shx, shy = pts[f"{side}_shoulder"]
            pts[f"{side}_elbow"] = (shx + sign * np.sin(ang) * arm / 2, shy + np.cos(ang) * arm / 2)
            pts[f"{side}_wrist"] = (shx + sign * np.sin(ang) * arm + trem[0], shy + np.cos(ang) * arm + trem[1])
        pts["l_knee"], pts["r_knee"] = (cx + 30, (hip_y + ankle_y) / 2), (cx - 30, (hip_y + ankle_y) / 2)
        pts["l_ankle"], pts["r_ankle"] = (cx + 30, ankle_y), (cx - 30, ankle_y)

        for j, name in enumerate(COCO_KP):
            x, y = pts[name]
            if fall > 0:  # rotaciona o corpo em torno dos tornozelos até a horizontal
                ax, ay = cx, ankle_y
                th = fall * np.pi / 2 * 0.95
                dx, dy = x - ax, y - ay
                x, y = ax + dx * np.cos(th) - dy * np.sin(th), ay + dx * np.sin(th) + dy * np.cos(th)
            kp[f, j, :2] = (x + rng.normal(0, 1.2), y + rng.normal(0, 1.2))
    events = [
        {"kind": "amplitude_reduzida", "start_s": 24, "end_s": 40},
        {"kind": "compensacao_tronco", "start_s": 24, "end_s": 40},
        {"kind": "movimento_irregular", "start_s": 40, "end_s": 48},
        {"kind": "queda", "start_s": 52, "end_s": 60},
    ]
    return kp, events


def render_skeleton_video(kp: np.ndarray, path: Path, fps: int = 15, size=(640, 480)) -> Path:
    """Renderiza a sequência de keypoints como vídeo (para visualização na demo)."""
    import cv2

    edges = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
             (12, 14), (14, 16), (0, 5), (0, 6)]
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for f in range(len(kp)):
        img = np.full((size[1], size[0], 3), 245, np.uint8)
        cv2.rectangle(img, (0, 445), (size[0], size[1]), (200, 200, 200), -1)
        for a, b in edges:
            pa, pb = kp[f, a, :2].astype(int), kp[f, b, :2].astype(int)
            cv2.line(img, tuple(pa), tuple(pb), (60, 60, 60), 4)
        for p in kp[f, :, :2].astype(int):
            cv2.circle(img, tuple(p), 5, (30, 110, 220), -1)
        cv2.putText(img, f"t={f / fps:5.1f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        vw.write(img)
    vw.release()
    from ..video.analyzer import to_browser_mp4

    to_browser_mp4(path)
    return path
