"""Extract the voice recordings of the cohort from a Coswara checkout.

Works with either
  * a git clone of github.com/iiscleap/Coswara-Data (archives read straight
    from git objects, so no checkout of the ~13 GB of tar parts is needed), or
  * a folder that already holds the split archives (<date>/<date>.tar.gz.aa ...),
    e.g. a Kaggle mirror of Coswara.
"""
import argparse
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

VOICE_TASKS = ["vowel-a", "vowel-e", "vowel-o", "counting-normal", "counting-fast"]


def archive_stream(repo: Path, date: str):
    parts = sorted((repo / date).glob(f"{date}.tar.gz.*"))
    if parts:
        return subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    names = subprocess.run(["git", "-C", str(repo), "ls-tree", "--name-only", "HEAD", f"{date}/"],
                           capture_output=True, text=True, check=True).stdout.split()
    names = sorted(n for n in names if ".tar.gz." in n)
    if not names:
        raise FileNotFoundError(f"no archive parts for {date}")
    cmd = " ; ".join(f"git -C '{repo}' cat-file blob HEAD:'{n}'" for n in names)
    return subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE)


def read_text(repo: Path, rel: str) -> str:
    f = repo / rel
    if f.exists():
        return f.read_text()
    return subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{rel}"],
                          capture_output=True, text=True, check=True).stdout


def folder_of_each_id(repo: Path) -> dict:
    """Map participant id -> archive folder, using each folder's <date>.csv."""
    folders = sorted({p.name for p in repo.iterdir() if p.name.isdigit()})
    if not folders:  # bare git checkout without working-tree folders
        folders = [n for n in subprocess.run(["git", "-C", str(repo), "ls-tree", "--name-only", "HEAD"],
                   capture_output=True, text=True, check=True).stdout.split() if n.isdigit()]
    mapping = {}
    for d in folders:
        for line in read_text(repo, f"{d}/{d}.csv").splitlines()[1:]:
            mapping[line.split(",", 1)[0]] = d
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coswara", required=True, type=Path)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort)
    cohort["date_folder"] = cohort["id"].map(folder_of_each_id(args.coswara))
    missing = cohort["date_folder"].isna().sum()
    if missing:
        print(f"warning: {missing} participants not found in any folder")
    for date, ids in cohort.groupby("date_folder")["id"]:
        wanted = set(ids)
        proc = archive_stream(args.coswara, date)
        n = 0
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
            for m in tar:
                p = Path(m.name)
                if len(p.parts) == 3 and p.parts[1] in wanted and p.stem in VOICE_TASKS and m.isfile():
                    dst = args.out / p.parts[1] / p.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(tar.extractfile(m).read())
                    n += 1
        proc.wait()
        print(f"{date}: {n} files for {len(wanted)} participants")


if __name__ == "__main__":
    main()
