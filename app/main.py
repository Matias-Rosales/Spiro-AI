"""Spiro AI web app: estimate COPD probability from two short voice recordings."""
import json
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from copd_voice.features import extract_file, participant_vector  # noqa: E402

bundle = joblib.load(ROOT / "models" / "copd_voice_model.joblib")
metrics = json.loads((ROOT / "models" / "metrics.json").read_text())

app = FastAPI(title="Spiro AI")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

DISPLAY = {
    "vowel-a__longest_segment_seconds": ("Sustained phonation time", "s", "Longer is typically better: reflects breath support."),
    "vowel-a__jitter_local": ("Jitter (pitch instability)", "%", "Cycle-to-cycle variation of pitch."),
    "vowel-a__shimmer_local": ("Shimmer (loudness instability)", "%", "Cycle-to-cycle variation of loudness."),
    "vowel-a__hnr_db": ("Harmonics-to-noise ratio", "dB", "Higher means a clearer, less breathy voice."),
    "counting-normal__pause_ratio": ("Pause ratio while counting", "%", "Share of time spent silent, e.g. pausing to breathe."),
    "counting-normal__active_seconds": ("Speaking time while counting", "s", "Total voiced time to count 1 to 20."),
}
PERCENT = {"vowel-a__jitter_local", "vowel-a__shimmer_local", "counting-normal__pause_ratio"}


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/model-info")
def model_info():
    best = metrics["best_model"]
    return {
        "best_model": best,
        "cv": metrics["combined"][best],
        "all_models": metrics["combined"],
        "baselines": metrics["baselines"],
        "n_participants": metrics["n_participants"],
        "n_cases": metrics["n_cases"],
        "n_controls": metrics["n_controls"],
        "roc": metrics["best_model_roc"],
        "per_task": metrics["per_task"],
        "matched_pair_concordance": metrics["matched_pair_concordance"],
    }


async def _features(upload: UploadFile) -> dict:
    data = await upload.read()
    if not data:
        raise HTTPException(400, f"{upload.filename or 'recording'} is empty")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "Recording is larger than 20 MB")
    suffix = Path(upload.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            return extract_file(tmp.name)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception:
            raise HTTPException(422, "Could not decode the audio. Try WAV, MP3, M4A, OGG or WebM.")


@app.post("/api/predict")
async def predict(vowel: UploadFile = File(...), counting: UploadFile = File(...)):
    row = participant_vector({"vowel-a": await _features(vowel), "counting-normal": await _features(counting)})
    X = pd.DataFrame([row]).reindex(columns=bundle["columns"])
    prob = float(bundle["model"].predict_proba(X)[0, 1])

    measures = []
    for key, (label, unit, help_text) in DISPLAY.items():
        scale = 100 if key in PERCENT else 1
        ref = bundle["reference"][key]
        value = row.get(key)
        measures.append({
            "label": label, "unit": unit, "help": help_text,
            "value": None if value is None or pd.isna(value) else round(value * scale, 2),
            "healthy_iqr": [round(v * scale, 2) for v in (ref["0"][0], ref["0"][2])],
            "copd_median": round(ref["1"][1] * scale, 2),
        })
    band = "low" if prob < 0.3 else "moderate" if prob < 0.6 else "elevated"
    return {"probability": prob, "band": band, "measures": measures,
            "train_prevalence": bundle["train_prevalence"]}
