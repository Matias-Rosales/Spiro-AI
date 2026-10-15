"""Compute per-recording features for every cohort participant -> features.csv."""
import argparse
import multiprocessing as mp
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

# One numba cache per process: librosa's JIT cache is not safe to write from
# several processes at once, and a corrupted cache segfaults on the next run.
os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"numba-{os.getpid()}-")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from copd_voice.features import extract_file  # noqa: E402

ALL_TASKS = ["vowel-a", "vowel-e", "vowel-o", "counting-normal", "counting-fast"]


def work(job):
    pid, task, path = job
    try:
        return {"id": pid, "task": task, **extract_file(path)}
    except Exception as e:  # empty/corrupt recordings exist in Coswara
        print(f"skip {pid}/{task}: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort)
    jobs = [(pid, t, args.audio / pid / f"{t}.wav") for pid in cohort["id"] for t in ALL_TASKS
            if (args.audio / pid / f"{t}.wav").exists()]
    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as ex:
        rows = [r for r in ex.map(work, jobs, chunksize=4) if r]
    df = pd.DataFrame(rows).merge(cohort[["id", "label", "a", "g", "match_of"]], on="id")
    df.to_csv(args.out, index=False)
    print(f"{len(df)} recordings, {df['id'].nunique()} participants -> {args.out}")


if __name__ == "__main__":
    main()
