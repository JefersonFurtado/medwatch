"""Extração de biomarcadores vocais.

Medidas clássicas de qualidade vocal (via Praat/parselmouth, quando
disponível) e de prosódia/temporalidade da fala (via librosa):

| grupo         | medida                         | relação clínica                    |
|---------------|--------------------------------|------------------------------------|
| fonação       | jitter local, shimmer local    | instabilidade (disartria, fadiga)  |
|               | HNR (harmônico/ruído)          | soprosidade, dispneia              |
| prosódia      | F0 médio, variabilidade (st)   | monotonia (disartria hipocinética) |
| temporalidade | taxa silábica, razão de pausas | lentificação (disartria, fadiga)   |
|               | duração média de frase         | frases curtas (dispneia)           |
| respiração    | inspirações audíveis/min       | esforço respiratório               |
| tendência     | queda de energia e F0 no áudio | fadiga vocal                       |
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class VoiceFeatures:
    duration_s: float
    speech_time_s: float
    f0_mean_hz: float
    f0_std_st: float
    jitter_local_pct: float
    shimmer_local_pct: float
    hnr_db: float
    syllable_rate: float          # sílabas por segundo de fala
    pause_ratio: float            # tempo em pausa / tempo total
    mean_pause_s: float
    mean_phrase_s: float
    breaths_per_min: float
    energy_slope_pct: float       # variação da energia (último terço vs primeiro), %
    f0_slope_pct: float           # variação do F0 (último terço vs primeiro), %
    backend: str

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _segments(mask: np.ndarray, hop_s: float, min_len_s: float) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if (i - start) * hop_s >= min_len_s:
                segs.append((start, i))
            start = None
    return segs


def _praat_measures(y: np.ndarray, sr: int) -> tuple[float, float, float] | None:
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return None
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    pp = call(snd, "To PointProcess (periodic, cc)", 60, 400)
    jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    hnr = call(call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0), "Get mean", 0, 0)
    return float(jitter), float(shimmer), float(hnr)


def extract_voice_features(path_or_array, sr: int = 16000) -> VoiceFeatures:
    import librosa
    from scipy.signal import find_peaks

    if isinstance(path_or_array, (str, bytes)) or hasattr(path_or_array, "__fspath__"):
        y, sr = librosa.load(path_or_array, sr=sr, mono=True)
    else:
        y = np.asarray(path_or_array, dtype=np.float32)
    duration = len(y) / sr
    hop = 160  # 10 ms
    hop_s = hop / sr

    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=hop)
    n = min(len(db), len(voiced))
    db, rms, f0, voiced = db[:n], rms[:n], f0[:n], voiced[:n]

    # atividade de fala: energia acima de -30 dB do pico e frame sonoro
    active = (db > -30) & voiced
    active = np.convolve(active.astype(float), np.ones(5) / 5, mode="same") > 0.5
    phrases = _segments(active, hop_s, 0.15)
    speech_time = sum(b - a for a, b in phrases) * hop_s
    pauses = [(phrases[i][1], phrases[i + 1][0]) for i in range(len(phrases) - 1)
              if (phrases[i + 1][0] - phrases[i][1]) * hop_s >= 0.15]
    pause_durs = [(b - a) * hop_s for a, b in pauses]
    total_span = (phrases[-1][1] - phrases[0][0]) * hop_s if phrases else duration
    pause_ratio = sum(pause_durs) / total_span if total_span else 0.0

    # núcleos silábicos: picos do envelope de energia durante a fala
    win = np.hanning(13)
    env = np.convolve(rms, win / win.sum(), mode="same")
    peaks, _ = find_peaks(env, distance=int(0.12 / hop_s), prominence=np.max(env) * 0.12)
    syl = [p for p in peaks if active[p]]
    syllable_rate = len(syl) / speech_time if speech_time else 0.0

    # inspirações audíveis: dentro das pausas, ruído não sonoro com espectro plano
    flat = librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=hop)[0][:n]
    breath_mask = (~voiced) & (db > -40) & (db < -12) & (flat > 0.25)
    breaths = 0
    trim = int(0.06 / hop_s)
    for a, b in pauses:
        inner = breath_mask[a + trim: b - trim]
        if len(inner) and max((e - s_) for s_, e in _segments(inner, hop_s, 0.0) or [(0, 0)]) * hop_s >= 0.10:
            breaths += 1
    breaths_per_min = breaths / (duration / 60)

    # tendências (fadiga): último terço vs primeiro terço
    vf0 = np.where(voiced & active, f0, np.nan)
    thirds = np.array_split(np.arange(n), 3)
    e1, e3 = np.nanmean(np.where(active, rms, np.nan)[thirds[0]]), np.nanmean(np.where(active, rms, np.nan)[thirds[2]])
    f1, f3 = np.nanmean(vf0[thirds[0]]), np.nanmean(vf0[thirds[2]])
    energy_slope = (e3 - e1) / e1 * 100 if e1 else 0.0
    f0_slope = (f3 - f1) / f1 * 100 if f1 else 0.0

    f0v = vf0[~np.isnan(vf0)]
    f0_mean = float(np.mean(f0v)) if len(f0v) else 0.0
    f0_std_st = float(np.std(12 * np.log2(f0v / np.median(f0v)))) if len(f0v) else 0.0

    praat = _praat_measures(y, sr)
    if praat:
        jitter, shimmer, hnr = praat
        backend = "praat+librosa"
    else:  # aproximações por frame
        periods = 1.0 / f0v if len(f0v) > 2 else np.array([1.0, 1.0])
        jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100)
        amps = rms[active]
        shimmer = float(np.mean(np.abs(np.diff(amps))) / (np.mean(amps) + 1e-9) * 100) if len(amps) > 2 else 0.0
        ac = []
        for a, b in phrases:
            seg = y[a * hop: b * hop]
            if len(seg) > 1024:
                r = np.correlate(seg[:2048], seg[:2048], "full")[len(seg[:2048]) - 1:]
                r = r / (r[0] + 1e-9)
                lo, hi = int(sr / 400), int(sr / 60)
                ac.append(np.clip(r[lo:hi].max(), 1e-3, 0.999))
        hnr = float(np.mean([10 * np.log10(r / (1 - r)) for r in ac])) if ac else 0.0
        backend = "librosa"

    return VoiceFeatures(
        duration_s=duration, speech_time_s=speech_time, f0_mean_hz=f0_mean, f0_std_st=f0_std_st,
        jitter_local_pct=jitter, shimmer_local_pct=shimmer, hnr_db=hnr, syllable_rate=syllable_rate,
        pause_ratio=pause_ratio, mean_pause_s=float(np.mean(pause_durs)) if pause_durs else 0.0,
        mean_phrase_s=float(np.mean([(b - a) * hop_s for a, b in phrases])) if phrases else 0.0,
        breaths_per_min=breaths_per_min, energy_slope_pct=float(energy_slope), f0_slope_pct=float(f0_slope),
        backend=backend)
