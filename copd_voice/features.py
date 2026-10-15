"""Acoustic features for COPD voice screening.

The same code runs at training time (Coswara WAV files) and in the web app
(browser recordings converted to WAV), so both see identical preprocessing.

Feature groups, motivated by the COPD speech literature:
  * phonation / breath support : phonation time, voiced fraction, pause ratio
  * vocal fold stability        : F0 statistics, jitter, shimmer, HNR
  * spectral envelope           : MFCC (+ deltas), centroid, bandwidth, rolloff,
                                  flatness, contrast, zero-crossing rate
  * loudness                    : RMS energy statistics
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

SR = 16000
MAX_SECONDS = 30
TASKS = ("vowel-a", "counting-normal")


def load_audio(path: str | Path) -> np.ndarray:
    """Load any audio file as mono 16 kHz float32 (falls back to ffmpeg for webm/ogg/m4a)."""
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_SECONDS)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                            "-ac", "1", "-ar", str(SR), "-t", str(MAX_SECONDS), tmp.name], check=True)
            y, _ = librosa.load(tmp.name, sr=SR, mono=True)
    return y.astype(np.float32)


def _stats(prefix: str, x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan}
    return {f"{prefix}_mean": float(np.mean(x)), f"{prefix}_std": float(np.std(x))}


def _voice_quality(y: np.ndarray, f0: np.ndarray, voiced: np.ndarray, hop: int) -> dict:
    """Jitter, shimmer and HNR approximated from frame-level pitch and amplitude."""
    out = {"jitter_local": np.nan, "shimmer_local": np.nan, "hnr_db": np.nan}
    periods = 1.0 / f0[voiced & np.isfinite(f0)]
    if periods.size > 3:
        out["jitter_local"] = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
    amp = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0][: voiced.size]
    amp = amp[voiced[: amp.size]]
    if amp.size > 3 and np.mean(amp) > 0:
        out["shimmer_local"] = float(np.mean(np.abs(np.diff(amp))) / np.mean(amp))
    harmonic, percussive = librosa.effects.hpss(y)
    noise = np.sum(percussive ** 2)
    if noise > 0:
        out["hnr_db"] = float(10 * np.log10(np.sum(harmonic ** 2) / noise + 1e-12))
    return out


def extract(y: np.ndarray) -> dict:
    """Feature dictionary for one recording (already 16 kHz mono)."""
    if y.size < SR // 2 or np.max(np.abs(y)) < 1e-4:
        raise ValueError("Recording is too short or silent.")
    y = y / (np.max(np.abs(y)) + 1e-9)
    total_s = y.size / SR

    # Speech/silence segmentation for breath-related timing features.
    intervals = librosa.effects.split(y, top_db=30)
    active_s = float(sum(e - s for s, e in intervals)) / SR
    longest_s = float(max((e - s for s, e in intervals), default=0)) / SR
    trimmed = np.concatenate([y[s:e] for s, e in intervals]) if len(intervals) else y
    if trimmed.size < SR // 4:
        trimmed = y

    feats = {
        "active_seconds": active_s,
        "longest_segment_seconds": longest_s,
        "pause_ratio": 1.0 - active_s / total_s,
        "n_segments_per_s": len(intervals) / total_s,
    }

    hop = 256
    f0, voiced, _ = librosa.pyin(trimmed, fmin=60, fmax=500, sr=SR, frame_length=1024, hop_length=hop)
    feats["voiced_fraction"] = float(np.mean(voiced))
    f0v = f0[voiced & np.isfinite(f0)]
    if f0v.size:
        semitones = 12 * np.log2(f0v / np.median(f0v))
        feats.update({"f0_median": float(np.median(f0v)),
                      "f0_sd_semitones": float(np.std(semitones)),
                      "f0_range_semitones": float(np.percentile(semitones, 95) - np.percentile(semitones, 5))})
    else:
        feats.update({"f0_median": np.nan, "f0_sd_semitones": np.nan, "f0_range_semitones": np.nan})
    feats.update(_voice_quality(trimmed, f0, voiced, hop))

    mfcc = librosa.feature.mfcc(y=trimmed, sr=SR, n_mfcc=20, hop_length=hop)
    delta = librosa.feature.delta(mfcc)
    for i in range(mfcc.shape[0]):
        feats.update(_stats(f"mfcc{i + 1}", mfcc[i]))
        feats.update(_stats(f"dmfcc{i + 1}", delta[i]))

    S = np.abs(librosa.stft(trimmed, n_fft=1024, hop_length=hop))
    feats.update(_stats("centroid", librosa.feature.spectral_centroid(S=S, sr=SR)[0]))
    feats.update(_stats("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=SR)[0]))
    feats.update(_stats("rolloff", librosa.feature.spectral_rolloff(S=S, sr=SR)[0]))
    feats.update(_stats("flatness", librosa.feature.spectral_flatness(S=S)[0]))
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR)
    for i in range(contrast.shape[0]):
        feats.update(_stats(f"contrast{i + 1}", contrast[i]))
    feats.update(_stats("zcr", librosa.feature.zero_crossing_rate(trimmed, hop_length=hop)[0]))
    rms = librosa.feature.rms(y=trimmed, hop_length=hop)[0]
    feats.update(_stats("rms_db", librosa.amplitude_to_db(rms + 1e-9)))
    return feats


def extract_file(path: str | Path) -> dict:
    return extract(load_audio(path))


def participant_vector(per_task: dict[str, dict]) -> dict:
    """Flatten {task: features} into one row with task-prefixed column names."""
    return {f"{task}__{k}": v for task in TASKS for k, v in per_task[task].items()}
