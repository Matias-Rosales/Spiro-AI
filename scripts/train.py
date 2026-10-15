"""Train and evaluate COPD-from-voice classifiers; save the best one for the web app.

Evaluation is participant-level, with repeated stratified 5-fold cross-validation
(each participant appears in exactly one test fold per repeat, so no speaker leaks
between train and test). Two sanity baselines are reported next to the voice models:
  * age + sex only      -> should be near chance, since controls are age/sex matched
  * recording year only -> measures how much the label is confounded with
                           *when* (and so on which app/device version) people recorded
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from copd_voice.features import TASKS  # noqa: E402

SEED = 0


def models():
    return {
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000)),
        "svm_rbf": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 SVC(C=1.0, class_weight="balanced", probability=True, random_state=SEED)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                              class_weight="balanced_subsample", random_state=SEED)),
        "gradient_boosting": make_pipeline(SimpleImputer(strategy="median"),
                                           HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                          max_iter=200, class_weight="balanced",
                                                                          random_state=SEED)),
    }


def cross_validate(model, X, y, n_repeats=10):
    """Out-of-fold probabilities averaged over repeats + per-repeat AUCs."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    oof = np.zeros((n_repeats, len(y)))
    for i, (tr, te) in enumerate(cv.split(X, y)):
        m = model.fit(X.iloc[tr], y[tr])
        oof[i // 5, te] = m.predict_proba(X.iloc[te])[:, 1]
    aucs = [roc_auc_score(y, p) for p in oof]
    p = oof.mean(axis=0)
    pred = (p >= 0.5).astype(int)
    tp, tn = np.sum((pred == 1) & (y == 1)), np.sum((pred == 0) & (y == 0))
    return {
        "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / max(1, y.sum())), "specificity": float(tn / max(1, (1 - y).sum())),
    }, p


def wide(features: pd.DataFrame, tasks) -> pd.DataFrame:
    meta = features.groupby("id")[["label", "a", "g", "match_of"]].first()
    feat_cols = [c for c in features.columns if c not in {"id", "task", "label", "a", "g", "match_of"}]
    parts = []
    for t in tasks:
        sub = features[features["task"] == t].set_index("id")[feat_cols]
        parts.append(sub.add_prefix(f"{t}__"))
    X = pd.concat(parts, axis=1, join="inner")
    return X, meta.loc[X.index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--metadata", required=True, help="Coswara combined_data.csv (for record year)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(args.features)
    years = pd.read_csv(args.metadata).set_index("id")["record_date"].str[:4].astype(int)
    report = {"per_task": {}, "baselines": {}, "combined": {}}

    # 1) Which single task carries the most signal?
    for task in sorted(feats["task"].unique()):
        X, meta = wide(feats, [task])
        r, _ = cross_validate(models()["logistic_regression"], X, meta["label"].to_numpy(), n_repeats=5)
        report["per_task"][task] = {**r, "n": int(len(X))}
        print(f"[task] {task:16s} AUC {r['auc_mean']:.3f} ± {r['auc_sd']:.3f}  (n={len(X)})")

    # 2) The deployed input: the tasks the web app records.
    X, meta = wide(feats, TASKS)
    y = meta["label"].to_numpy()
    report["n_participants"] = int(len(y))
    report["n_cases"] = int(y.sum())
    report["n_controls"] = int(len(y) - y.sum())
    report["tasks"] = list(TASKS)

    demo = pd.DataFrame({"age": meta["a"].astype(float), "male": (meta["g"] == "male").astype(float)})
    r, _ = cross_validate(models()["logistic_regression"], demo, y)
    report["baselines"]["age_sex_only"] = r
    yr = pd.DataFrame({"year": years.reindex(meta.index).astype(float)})
    r, _ = cross_validate(models()["logistic_regression"], yr, y)
    report["baselines"]["recording_year_only"] = r
    for k, v in report["baselines"].items():
        print(f"[baseline] {k:22s} AUC {v['auc_mean']:.3f} ± {v['auc_sd']:.3f}")

    best_name, best_auc, best_oof = None, -1, None
    for name, model in models().items():
        r, oof = cross_validate(model, X, y)
        report["combined"][name] = r
        print(f"[voice] {name:20s} AUC {r['auc_mean']:.3f} ± {r['auc_sd']:.3f}  "
              f"bal.acc {r['balanced_accuracy']:.3f}  sens {r['sensitivity']:.2f}  spec {r['specificity']:.2f}")
        if r["auc_mean"] > best_auc:
            best_name, best_auc, best_oof = name, r["auc_mean"], oof

    # Within-year check: does the voice signal survive when the year confound is removed?
    within = {}
    for year, idx in pd.Series(meta.index, index=meta.index).groupby(yr["year"]):
        mask = meta.index.isin(idx)
        if 0 < y[mask].sum() < mask.sum():
            within[str(int(year))] = {"auc": float(roc_auc_score(y[mask], best_oof[mask])),
                                      "n_cases": int(y[mask].sum()), "n_controls": int((1 - y[mask]).sum())}
    report["best_model_within_year_auc"] = within
    print("[within-year]", within)

    # Matched-set concordance: within each case's own sex/age/year-matched set, how often
    # does the case score above its controls? Age, sex and year are held fixed by design.
    oof = pd.Series(best_oof, index=meta.index)
    wins = []
    for case_id, grp in meta.groupby("match_of"):
        cases, ctrls = grp.index[grp["label"] == 1], grp.index[grp["label"] == 0]
        wins += [float(oof[c] > oof[k]) + 0.5 * float(oof[c] == oof[k]) for c in cases for k in ctrls]
    report["matched_pair_concordance"] = {"value": float(np.mean(wins)), "n_pairs": len(wins)}
    print(f"[matched pairs] concordance {np.mean(wins):.3f} over {len(wins)} case-control pairs")

    fpr, tpr, _ = roc_curve(y, best_oof)
    report["best_model"] = best_name
    report["best_model_roc"] = {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()}

    # Reference ranges shown in the web app next to the user's own measurements.
    ref_feats = ["vowel-a__longest_segment_seconds", "vowel-a__jitter_local", "vowel-a__shimmer_local",
                 "vowel-a__hnr_db", "counting-normal__pause_ratio", "counting-normal__active_seconds"]
    reference = {f: {str(lbl): [float(v) for v in X.loc[meta["label"] == lbl, f].quantile([0.25, 0.5, 0.75])]
                     for lbl in (0, 1)} for f in ref_feats}

    final = CalibratedClassifierCV(models()[best_name], method="sigmoid", cv=5).fit(X, y)
    joblib.dump({"model": final, "columns": list(X.columns), "tasks": list(TASKS),
                 "train_prevalence": float(y.mean()), "reference": reference}, args.out / "copd_voice_model.joblib")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"saved {best_name} -> {args.out}")


if __name__ == "__main__":
    main()
