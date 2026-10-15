"""Build a case/control cohort from the Coswara metadata.

Cases    : participants who self-reported Chronic Lung Disease (`cld == True`),
           the Coswara field that covers COPD.
Controls : participants with no respiratory condition or symptom at all,
           matched to each case on sex, recording year and age
           (nearest age, then nearest recording date, without replacement).

Age matching matters: CLD participants are ~22 years older than the Coswara
average, and age alone changes the voice, so an unmatched model would mostly
learn to detect age. Matching on recording year matters too: most CLD cases
recorded in 2021 while most Coswara controls recorded in 2020, and a model can
pick up app/device/codec changes over time instead of the voice itself.
"""
import argparse

import numpy as np
import pandas as pd

RESP_FLAGS = ["cld", "asthma", "others_resp", "bd", "cough", "cold", "st", "fever", "pneumonia"]
HEALTHY_STATUS = {"healthy", "no_resp_illness_exposed"}


def is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "y"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Coswara combined_data.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-per-case", type=int, default=3)
    ap.add_argument("--max-age-gap", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)
    df["a"] = pd.to_numeric(df["a"], errors="coerce")
    df["date"] = pd.to_datetime(df["record_date"])
    cases = df[is_true(df["cld"])].copy()

    no_resp = ~np.logical_or.reduce([is_true(df[c]) for c in RESP_FLAGS])
    pool = df[no_resp & df["covid_status"].isin(HEALTHY_STATUS) & df["a"].between(18, 100)].copy()

    pool = pool.sample(frac=1, random_state=args.seed)  # random tie-breaking
    used, rows = set(), []
    # Match the oldest cases first: old healthy controls are the scarce resource.
    for _, case in cases.sort_values("a", ascending=False).iterrows():
        cand = pool[(pool["g"] == case["g"]) & (pool["date"].dt.year == case["date"].year)
                    & ~pool["id"].isin(used)].copy()
        cand["gap"] = (cand["a"] - case["a"]).abs()
        cand["days"] = (cand["date"] - case["date"]).abs().dt.days
        cand = cand[cand["gap"] <= args.max_age_gap].sort_values(["gap", "days"], kind="stable")
        picked = cand.head(args.controls_per_case)
        used.update(picked["id"])
        for _, c in picked.iterrows():
            rows.append({**c[["id", "a", "g", "record_date"]].to_dict(), "label": 0, "match_of": case["id"]})
        if picked.empty:
            print(f"case {case['id']} (age {case['a']}, {case['g']}): no control within age gap, dropped")
            continue
        if len(picked) < args.controls_per_case:
            print(f"case {case['id']} (age {case['a']}, {case['g']}): only {len(picked)} controls")
        rows.append({**case[["id", "a", "g", "record_date"]].to_dict(), "label": 1, "match_of": case["id"]})

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.groupby("label").agg(n=("id", "size"), age_mean=("a", "mean"), age_sd=("a", "std")))
    print(pd.crosstab(out["label"], out["g"]))
    print(pd.crosstab(out["label"], out["record_date"].str[:4]))


if __name__ == "__main__":
    main()
