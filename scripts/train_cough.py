"""Train a COPD cough model that is robust to recording background.

Data: COPD-Cough (CC) dataset, Zenodo 10.5281/zenodo.14881818 (CC BY 4.0),
221 COPD and 632 healthy smartphone cough recordings.

Robustness strategy
  * features come from copd_voice.cough (band-limited, denoised, cough events only);
  * background-swap augmentation: every training recording is added twice more
    with its own noise removed and another random recording's background added,
    so background and label are no longer correlated in the training set;
  * stress test: every test recording is re-scored with the background of a
    recording from the *opposite* class. A background-driven model changes its
    answer; a cough-driven model does not.

A naive baseline (generic voice features on the raw audio, no denoising, no
augmentation) is evaluated the same way for comparison.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"numba-{os.getpid()}-")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from copd_voice.cough import extract_cough, swap_background  # noqa: E402
from copd_voice.features import extract, load_audio  # noqa: E402

N_AUG = 2
SEED = 0


def models():
    return {
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                              class_weight="balanced_subsample", random_state=SEED)),
        "gradient_boosting": make_pipeline(SimpleImputer(strategy="median"),
                                           HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                          max_iter=300, class_weight="balanced",
                                                                          random_state=SEED)),
    }


def job(args):
    """Compute robust (and optionally naive) features for one, possibly swapped, recording."""
    kind, src, bg, seed, naive = args
    y = load_audio(src)
    if bg is not None:
        y = swap_background(y, load_audio(bg), np.random.default_rng(seed))
    row = {"kind": kind, "src": str(src), "bg": str(bg) if bg else ""}
    try:
        row.update({f"r__{k}": v for k, v in extract_cough(y).items()})
    except ValueError:
        pass
    if naive:
        try:
            row.update({f"n__{k}": v for k, v in extract(y).items()})
        except ValueError:
            pass
    return row


def build_table(files: pd.DataFrame, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache)
    rng = np.random.default_rng(SEED)
    jobs = []
    all_paths = files["path"].tolist()
    by_label = {lbl: g["path"].tolist() for lbl, g in files.groupby("label")}
    for i, r in files.iterrows():
        jobs.append(("orig", r.path, None, 0, True))
        for k in range(N_AUG):  # augmentation: background of any random recording
            bg = all_paths[rng.integers(len(all_paths))]
            jobs.append((f"aug{k}", r.path, bg, int(rng.integers(1 << 31)), False))
        other = by_label[1 - r.label]  # stress test: background of the opposite class
        jobs.append(("swap", r.path, other[rng.integers(len(other))], int(rng.integers(1 << 31)), True))
    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(job, jobs, chunksize=8))
    table = pd.DataFrame(rows).merge(files.rename(columns={"path": "src"}), on="src")
    table.to_csv(cache, index=False)
    return table


def evaluate(table, cols, model_name, use_aug, n_repeats=3):
    """Recording-grouped CV: train on train-fold recordings (+ their augmentations),
    test on the held-out originals and on their background-swapped versions."""
    orig = table[table.kind == "orig"].set_index("src")
    swap = table[table.kind == "swap"].set_index("src").reindex(orig.index)
    y = orig["label"].to_numpy()
    p_orig = np.zeros((n_repeats, len(y)))
    p_swap = np.zeros((n_repeats, len(y)))
    for rep in range(n_repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(orig, y):
            train_src = orig.index[tr]
            kinds = ["orig"] + ([f"aug{k}" for k in range(N_AUG)] if use_aug else [])
            train = table[table.src.isin(train_src) & table.kind.isin(kinds)]
            m = models()[model_name].fit(train[cols], train["label"])
            p_orig[rep, te] = m.predict_proba(orig.iloc[te][cols])[:, 1]
            p_swap[rep, te] = m.predict_proba(swap.iloc[te][cols])[:, 1]
    po, ps = p_orig.mean(0), p_swap.mean(0)
    return {
        "auc": float(np.mean([roc_auc_score(y, p) for p in p_orig])),
        "auc_sd": float(np.std([roc_auc_score(y, p) for p in p_orig])),
        "auc_background_swapped": float(roc_auc_score(y, ps)),
        "mean_abs_shift": float(np.mean(np.abs(po - ps))),
        # Fraction of recordings whose decision (at 0.5) flips when only the background changes.
        "decision_flip_rate": float(np.mean((po >= 0.5) != (ps >= 0.5))),
        "balanced_accuracy": float(balanced_accuracy_score(y, po >= 0.5)),
        "sensitivity": float(np.mean(po[y == 1] >= 0.5)),
        "specificity": float(np.mean(po[y == 0] < 0.5)),
    }, po


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", required=True, type=Path, help="extracted Cough-COPD(CC) folder")
    ap.add_argument("--cache", required=True, type=Path, help="feature table CSV (created if missing)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    files = pd.DataFrame([{"path": str(p), "label": int(p.parent.name == "COPD"), "split": p.parent.parent.name}
                          for p in sorted(args.cc.glob("*/*/*.wav"))])
    table = build_table(files, args.cache)
    table = table.dropna(subset=["r__n_coughs"])  # recordings with no detectable cough
    robust_cols = [c for c in table.columns if c.startswith("r__")]
    naive_cols = [c for c in table.columns if c.startswith("n__")]
    print(f"{table[table.kind == 'orig'].shape[0]} recordings with detected coughs "
          f"({table[(table.kind == 'orig') & (table.label == 1)].shape[0]} COPD)")

    report = {"n_recordings": int((table.kind == "orig").sum()),
              "n_copd": int(((table.kind == "orig") & (table.label == 1)).sum()), "models": {}}
    naive_table = table.dropna(subset=["n__active_seconds"])
    r, _ = evaluate(naive_table, naive_cols, "random_forest", use_aug=False)
    report["naive_baseline"] = r
    print(f"[naive ] random_forest       AUC {r['auc']:.3f}  swapped {r['auc_background_swapped']:.3f}  "
          f"shift {r['mean_abs_shift']:.3f}  flips {r['decision_flip_rate']:.1%}")

    best, best_score = None, -1
    for name in models():
        for use_aug in (False, True):
            r, _ = evaluate(table, robust_cols, name, use_aug)
            key = f"{name}{'+aug' if use_aug else ''}"
            report["models"][key] = r
            print(f"[robust] {key:24s} AUC {r['auc']:.3f}  swapped {r['auc_background_swapped']:.3f}  "
                  f"shift {r['mean_abs_shift']:.3f}  flips {r['decision_flip_rate']:.1%}")
            # Selection favours robustness: the worse of normal and swapped AUC.
            score = min(r["auc"], r["auc_background_swapped"])
            if use_aug and score > best_score:
                best, best_score = (name, key), score

    name, key = best
    report["best_model"] = key
    kinds = ["orig"] + [f"aug{k}" for k in range(N_AUG)]
    train = table[table.kind.isin(kinds)]
    final = CalibratedClassifierCV(models()[name], method="sigmoid", cv=5).fit(train[robust_cols], train["label"])

    # Official train/test split of the dataset, for comparability with the paper.
    tr = table[table.kind.isin(kinds) & (table.split == "train")]
    te = table[(table.kind == "orig") & (table.split == "test")]
    te_swap = table[(table.kind == "swap") & (table.split == "test")]
    m = models()[name].fit(tr[robust_cols], tr["label"])
    report["official_split"] = {"auc": float(roc_auc_score(te.label, m.predict_proba(te[robust_cols])[:, 1])),
                                "auc_background_swapped": float(roc_auc_score(
                                    te_swap.label, m.predict_proba(te_swap[robust_cols])[:, 1])),
                                "n_test": int(len(te))}
    print(f"[official split] {report['official_split']}")

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "columns": robust_cols,  # "r__" + extract_cough() keys
                 "train_prevalence": float(table[table.kind == "orig"].label.mean())},
                args.out / "copd_cough_model.joblib")
    (args.out / "metrics_cough.json").write_text(json.dumps(report, indent=2))
    print(f"saved {key} -> {args.out}")


if __name__ == "__main__":
    main()
