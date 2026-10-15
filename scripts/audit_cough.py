"""Audit the COPD-Cough (CC) dataset for recording-condition shortcuts.

Compares three feature sets with the same classifier:
  * cough      : acoustic features of the cough segments (what we want to learn from)
  * background : MFCC/spectral stats of the *non-cough* frames only (room + device)
  * noise floor: spectrum of the quietest 10% of frames (room/microphone, not the person)
  * nuisance   : duration, peak level, background level
If background or nuisance alone predict COPD well, the label is confounded with
where/how people were recorded, and a cough model may learn that instead.
"""
import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"numba-{os.getpid()}-")

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from copd_voice.features import SR, extract, load_audio  # noqa: E402


def spectral_stats(prefix, y):
    if y.size < SR // 10:
        return {}
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    out = {f"{prefix}_mfcc{i}_mean": float(m.mean()) for i, m in enumerate(mfcc)}
    out.update({f"{prefix}_mfcc{i}_std": float(m.std()) for i, m in enumerate(mfcc)})
    out[f"{prefix}_centroid"] = float(librosa.feature.spectral_centroid(y=y, sr=SR).mean())
    out[f"{prefix}_rms_db"] = float(librosa.amplitude_to_db(librosa.feature.rms(y=y)).mean())
    return out


def work(path):
    y = load_audio(path)
    raw_peak = float(np.max(np.abs(y)))
    intervals = librosa.effects.split(y, top_db=30)
    mask = np.ones(y.size, bool)
    for s, e in intervals:
        mask[s:e] = False
    background = y[mask]
    row = {"file": str(path), "label": int(Path(path).parent.name == "COPD"),
           "split": Path(path).parent.parent.name}
    try:
        row.update({f"cough__{k}": v for k, v in extract(y).items()})
    except ValueError:
        pass
    row.update({f"bg__{k}": v for k, v in spectral_stats("bg", background).items()})
    # Noise floor: log-mel spectrum of the quietest 10% of frames.
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=24) + 1e-12)
    quiet = mel[:, mel.mean(axis=0) <= np.percentile(mel.mean(axis=0), 10)]
    row.update({f"nf__mel{i}": float(v) for i, v in enumerate(quiet.mean(axis=1))})
    row.update({"nz__duration": y.size / SR, "nz__peak": raw_peak,
                "nz__background_fraction": float(mask.mean()),
                "nz__background_rms_db": float(20 * np.log10(np.sqrt(np.mean(background ** 2)) + 1e-9))
                if background.size else np.nan})
    return row


def cv_auc(X, y):
    model = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                 class_weight="balanced_subsample", random_state=0))
    aucs = []
    for rep in range(5):
        p = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=rep),
                              method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", required=True, type=Path, help="extracted Cough-COPD(CC) folder")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(args.cc.glob("*/*/*.wav"))
    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as ex:
        df = pd.DataFrame(list(ex.map(work, files, chunksize=8)))
    df.to_csv(args.out, index=False)
    y = df["label"].to_numpy()
    for name, prefix in [("cough", "cough__"), ("background", "bg__"), ("noise floor", "nf__"), ("nuisance", "nz__")]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        m, s = cv_auc(df[cols], y)
        print(f"{name:11s} ({len(cols):3d} features)  AUC {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
