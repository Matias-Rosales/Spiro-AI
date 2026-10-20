"""Cough features designed to ignore the recording background.

In the COPD-Cough dataset the room/microphone noise alone predicts the label
(AUC ~0.78), so a naive model learns "sounds like a hospital". This module
removes as much of that channel as it can before any feature is computed:

  1. band-limit to 100-4000 Hz: the device bandwidth differs between classes,
     and cough acoustics live mostly below 4 kHz;
  2. spectral gating: estimate each recording's own noise spectrum from its
     quietest frames and subtract it;
  3. keep only detected cough events: silence and background between coughs
     never reach the feature extractor;
  4. per-event loudness normalisation, so recording gain does not matter;
  5. features describe the cough itself (spectral shape, dynamics, duration),
     with no absolute-level or noise-floor features.

`swap_background` builds the robustness test used in training: the same coughs
laid over another recording's background noise. A model that ignores the
background gives (almost) the same answer for both.
"""
from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from .features import SR, load_audio

N_FFT = 512
HOP = 128
BAND = (100.0, 4000.0)
_SOS = butter(4, BAND, btype="bandpass", fs=SR, output="sos")


def bandlimit(y: np.ndarray) -> np.ndarray:
    return sosfiltfilt(_SOS, y).astype(np.float32)


def noise_profile(y: np.ndarray) -> np.ndarray:
    """Mean magnitude spectrum of the quietest 15% of frames."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    energy = S.sum(axis=0)
    quiet = S[:, energy <= np.percentile(energy, 15)]
    return quiet.mean(axis=1, keepdims=True)


def denoise(y: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Spectral gating with the recording's own noise profile."""
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    mag, phase = np.abs(D), np.angle(D)
    noise = noise_profile(y)
    clean = np.maximum(mag - strength * noise, 0.0)
    # Soft mask: bins close to the noise floor are attenuated further.
    mask = clean / (mag + 1e-9)
    mask = median_filter(mask, size=(3, 5))  # smooth over frequency and time to avoid "musical noise"
    out = librosa.istft(mag * mask * np.exp(1j * phase), hop_length=HOP, length=y.size)
    return out.astype(np.float32)


def cough_events(y: np.ndarray, min_dur: float = 0.12, max_events: int = 12) -> list[tuple[int, int]]:
    """Energy-based cough event detection on the denoised signal."""
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    if rms.max() <= 0:
        return []
    db = librosa.amplitude_to_db(rms, ref=rms.max())
    active = db > -25
    events, start = [], None
    for i, a in enumerate(np.append(active, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            s, e = start * HOP, min(y.size, i * HOP + N_FFT)
            if (e - s) / SR >= min_dur:
                events.append((s, e))
            start = None
    # Merge events separated by < 60 ms (one cough with a brief dip).
    merged = []
    for s, e in events:
        if merged and s - merged[-1][1] < 0.06 * SR:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    merged.sort(key=lambda se: -np.sum(y[se[0]:se[1]] ** 2))
    return sorted(merged[:max_events])


def _event_features(seg: np.ndarray) -> dict:
    seg = seg / (np.sqrt(np.mean(seg ** 2)) + 1e-9)  # loudness-normalised
    S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP)) ** 2
    mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=SR, n_mels=40, fmin=BAND[0], fmax=BAND[1]))
    mfcc = librosa.feature.mfcc(S=mel, n_mfcc=20)
    env = librosa.feature.rms(y=seg, frame_length=N_FFT, hop_length=HOP)[0]
    peak = int(np.argmax(env))
    f = {}
    for i in range(1, 20):  # skip c0 (overall level)
        f[f"mfcc{i}_mean"] = float(mfcc[i].mean())
        f[f"mfcc{i}_std"] = float(mfcc[i].std())
    d = librosa.feature.delta(mfcc, width=min(9, mfcc.shape[1] - (1 - mfcc.shape[1] % 2))) \
        if mfcc.shape[1] >= 3 else np.zeros_like(mfcc)
    for i in range(1, 13):
        f[f"dmfcc{i}_std"] = float(d[i].std())
    for name, v in {
        "centroid": librosa.feature.spectral_centroid(S=S, sr=SR)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=SR)[0],
        "rolloff85": librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85)[0],
        "flatness": librosa.feature.spectral_flatness(S=S)[0],
        "zcr": librosa.feature.zero_crossing_rate(seg, frame_length=N_FFT, hop_length=HOP)[0],
    }.items():
        f[f"{name}_mean"] = float(v.mean())
        f[f"{name}_std"] = float(v.std())
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR, fmin=BAND[0], n_bands=4)
    for i in range(contrast.shape[0]):
        f[f"contrast{i}"] = float(contrast[i].mean())
    # Temporal shape: how fast the cough builds up and decays, and its length.
    env_db = librosa.amplitude_to_db(env + 1e-9, ref=env.max())
    after = env_db[peak:]
    f["duration_s"] = seg.size / SR
    f["attack_s"] = peak * HOP / SR
    f["decay_to_-20dB_s"] = float(np.argmax(after < -20) * HOP / SR) if np.any(after < -20) else after.size * HOP / SR
    f["peak_position"] = peak / max(1, env.size - 1)
    # Low/high band energy ratio inside the cough (wet vs dry cough).
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    low, high = S[freqs < 1000].sum(), S[(freqs >= 1000) & (freqs <= BAND[1])].sum()
    f["low_high_ratio_db"] = float(10 * np.log10((low + 1e-9) / (high + 1e-9)))
    return f


def extract_cough(y: np.ndarray) -> dict:
    """Recording-level features: mean/std over detected cough events."""
    y = bandlimit(y)
    y = denoise(y)
    events = cough_events(y)
    if not events:
        raise ValueError("No cough detected. Cough clearly 2-3 times, close to the microphone.")
    per_event = [_event_features(y[s:e]) for s, e in events]
    keys = per_event[0].keys()
    out = {"n_coughs": float(len(events)),
           "cough_rate_per_s": len(events) / (y.size / SR)}
    for k in keys:
        vals = np.array([pe[k] for pe in per_event])
        out[f"{k}__mean"] = float(vals.mean())
        out[f"{k}__std"] = float(vals.std()) if len(vals) > 1 else 0.0
    return out


def extract_cough_file(path) -> dict:
    return extract_cough(load_audio(path))


def background_of(y: np.ndarray, n_samples: int) -> np.ndarray:
    """A noise-only signal of n_samples built from the quietest frames of a recording."""
    frame = 2048
    n = y.size // frame
    if n < 2:
        return np.zeros(n_samples, np.float32)
    frames = y[: n * frame].reshape(n, frame)
    e = (frames ** 2).mean(axis=1)
    quiet = frames[e <= np.percentile(e, 20)]
    reps = int(np.ceil(n_samples / quiet.size)) + 1
    noise = np.tile(quiet.reshape(-1), reps)[:n_samples]
    return noise.astype(np.float32)


def swap_background(cough_rec: np.ndarray, background_rec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Coughs of one recording over the background noise of another.

    The cough recording is denoised first (removing its own room noise), then the
    other recording's background is added at that recording's original level.
    """
    clean = denoise(bandlimit(cough_rec))
    noise = background_of(background_rec, clean.size)
    noise = np.roll(noise, rng.integers(0, max(1, noise.size)))
    return (clean + noise).astype(np.float32)
