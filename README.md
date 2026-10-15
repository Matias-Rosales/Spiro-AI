# VoiceLung: COPD likelihood from voice

Research prototype that estimates the probability of COPD-like voice patterns from two short
recordings (a sustained vowel and counting 1 to 20), with an English web front end.

**Not a medical device.** Trained on a small, self-reported dataset; it cannot diagnose COPD.

## Data

[Coswara](https://github.com/iiscleap/Coswara-Data) (IISc Bangalore, CC BY 4.0), also mirrored on Kaggle.
Crowdsourced smartphone/browser recordings, 2020 to 2022, with self-reported health metadata.

* **Cases**: `cld == True` ("Chronic Lung Disease", the Coswara field that covers COPD).
* **Controls**: no respiratory condition or symptom, COVID-negative/healthy, matched to each
  case on **sex, age (±10 y) and recording year** (up to 3 per case).

Matching on recording year is essential: without it, the recording year *alone* predicts the
label with AUC 0.85 (cases mostly recorded in 2021, controls in 2020), so a model learns app
and device changes instead of the voice.

## Results (repeated 5-fold CV, participant level, 98 participants: 29 cases / 69 controls)

| Model | ROC AUC | Balanced acc. | Sens. | Spec. |
|---|---|---|---|---|
| Random forest (deployed) | 0.82 ± 0.02 | 0.71 | 0.48 | 0.94 |
| Gradient boosting | 0.80 ± 0.04 | 0.75 | 0.62 | 0.88 |
| SVM (RBF) | 0.78 ± 0.03 | 0.64 | 0.34 | 0.94 |
| Logistic regression | 0.74 ± 0.03 | 0.69 | 0.55 | 0.83 |
| *Baseline: age + sex only* | 0.68 | | | |
| *Baseline: recording year only* | 0.44 | | | |

**Matched-pair concordance: 0.71.** Within each case's own sex/age/year-matched set, the case
scores above its control 71% of the time. This is the fairest number, since demographics and
recording period are held fixed. Full numbers are in `models/metrics.json`.

Counting tasks carry more signal (AUC ≈ 0.79) than sustained vowels (0.60 to 0.75).

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone https://github.com/iiscleap/Coswara-Data ../coswara        # ~13 GB
.venv/bin/python scripts/build_cohort.py   --metadata ../coswara/combined_data.csv --out data/cohort.csv
.venv/bin/python scripts/extract_audio.py  --coswara ../coswara --cohort data/cohort.csv --out data/audio
.venv/bin/python scripts/build_features.py --cohort data/cohort.csv --audio data/audio --out data/features.csv
.venv/bin/python scripts/train.py          --features data/features.csv --metadata ../coswara/combined_data.csv --out models
```

`extract_audio.py` also works on a folder holding the split `<date>/<date>.tar.gz.*` archives
(e.g. a Kaggle download of Coswara). The audio itself is never checked out.

## Run the web app

```bash
.venv/bin/uvicorn app.main:app --port 8000     # needs ffmpeg for browser (WebM/M4A) recordings
```

Open http://localhost:8000. Recording uses the browser microphone (served over `localhost` or HTTPS),
and uploads of WAV/MP3/M4A/OGG/WebM work too. Audio is processed in memory and not stored.

## Layout

```
copd_voice/features.py   shared feature extraction (training + app)
scripts/                 cohort, audio extraction, features, training
models/                  copd_voice_model.joblib + metrics.json
app/                     FastAPI server + static front end
```

## Limitations

* Small cohort; "chronic lung disease" is self-reported and broader than COPD.
* Indian participants recording in English on their own devices; generalization to other
  languages, accents and microphones is untested.
* Probabilities reflect the ~30% case share of the training cohort, not population prevalence.
* For a stronger model, request clinically confirmed COPD voice data such as COPDVD
  (Blekinge Institute of Technology) or the Danish COPD speech dataset (Aalborg University).
