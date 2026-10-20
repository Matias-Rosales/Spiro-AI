const recordings = { vowel: null, counting: null, cough: null };
const analyzeBtn = document.getElementById("analyze");
let active = null; // the recorder currently running

function updateAnalyze() {
  const ready = (recordings.vowel && recordings.counting) || recordings.cough;
  analyzeBtn.disabled = !ready || active !== null;
}

function setRecording(card, blob, name) {
  const task = card.dataset.task;
  recordings[task] = { blob, name };
  const audio = card.querySelector("audio");
  audio.src = URL.createObjectURL(blob);
  audio.hidden = false;
  card.classList.add("done");
  updateAnalyze();
}

function setStatus(card, msg, isErr = false) {
  const el = card.querySelector(".status");
  el.textContent = msg;
  el.classList.toggle("err", isErr);
}

function pickMime() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  return types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || "";
}

function drawMeter(canvas, analyser) {
  const ctx = canvas.getContext("2d");
  const data = new Uint8Array(analyser.fftSize);
  const color = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const track = getComputedStyle(document.documentElement).getPropertyValue("--track").trim();
  return () => {
    analyser.getByteTimeDomainData(data);
    let peak = 0;
    for (const v of data) peak = Math.max(peak, Math.abs(v - 128) / 128);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = track;
    ctx.fillRect(0, 12, canvas.width, 8);
    ctx.fillStyle = color;
    ctx.fillRect(0, 12, canvas.width * Math.min(1, peak * 1.6), 8);
  };
}

async function startRecording(card) {
  const btn = card.querySelector(".rec");
  const timer = card.querySelector(".timer");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (e) {
    setStatus(card, "Microphone access was denied. You can upload a file instead.", true);
    return;
  }
  const mime = pickMime();
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const chunks = [];
  const audioCtx = new AudioContext();
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  const draw = drawMeter(card.querySelector(".meter"), analyser);
  const t0 = performance.now();
  const tick = setInterval(() => {
    const s = (performance.now() - t0) / 1000;
    timer.textContent = `${s.toFixed(1)} s`;
    draw();
    if (s >= 30) stop();
  }, 50);

  function stop() {
    if (rec.state !== "inactive") rec.stop();
  }
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    clearInterval(tick);
    stream.getTracks().forEach((t) => t.stop());
    audioCtx.close();
    btn.classList.remove("on");
    btn.querySelector(".lbl").textContent = "Record again";
    active = null;
    const secs = (performance.now() - t0) / 1000;
    if (secs < 1.5) {
      setStatus(card, "That was very short — please record at least a few seconds.", true);
      updateAnalyze();
      return;
    }
    const type = rec.mimeType || "audio/webm";
    const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
    setRecording(card, new Blob(chunks, { type }), `${card.dataset.task}.${ext}`);
    setStatus(card, `Recorded ${secs.toFixed(1)} s.`);
  };

  rec.start();
  active = { card, stop };
  btn.classList.add("on");
  btn.querySelector(".lbl").textContent = "Stop";
  setStatus(card, "Recording… press Stop when you are done.");
  updateAnalyze();
}

document.querySelectorAll(".task").forEach((card) => {
  card.querySelector(".rec").addEventListener("click", () => {
    if (active && active.card === card) return active.stop();
    if (active) active.stop();
    startRecording(card);
  });
  card.querySelector("input[type=file]").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setRecording(card, f, f.name);
    setStatus(card, `Loaded ${f.name}.`);
  });
});

const BAND_LABEL = { low: "Low likelihood", moderate: "Moderate likelihood", elevated: "Elevated likelihood" };
const BAND_COLOR = { low: "--low", moderate: "--mid", elevated: "--high" };
const BAND_TEXT = {
  voice: {
    low: "Your voice resembles the healthy recordings in our data.",
    moderate: "Some voice features resemble recordings from people with chronic lung disease.",
    elevated: "Your voice shares several patterns with recordings from people with chronic lung disease.",
  },
  cough: {
    low: "Your cough resembles the healthy coughs in our data.",
    moderate: "Some cough features resemble coughs from people with COPD.",
    elevated: "Your cough shares several patterns with coughs from people with COPD.",
  },
};
const ADVICE = " Consider asking a doctor about a spirometry test, especially if you have breathlessness, a lasting cough or a smoking history.";

function renderGauge(card, kind, r) {
  card.hidden = !r;
  if (!r) return;
  card.querySelector(".pct").textContent = `${Math.round(r.probability * 100)}%`;
  const fill = card.querySelector(".gauge .fill");
  const len = fill.getTotalLength();
  const color = `var(${BAND_COLOR[r.band]})`;
  fill.style.strokeDasharray = len;
  fill.style.strokeDashoffset = len * (1 - r.probability);
  fill.style.stroke = color;
  const band = card.querySelector(".band");
  band.textContent = BAND_LABEL[r.band];
  band.style.color = color;
  card.querySelector(".band-text").textContent = BAND_TEXT[kind][r.band] + (r.band === "elevated" ? ADVICE : "");
}

function renderResult(r) {
  renderGauge(document.getElementById("voice-result"), "voice", r.voice);
  renderGauge(document.getElementById("cough-result"), "cough", r.cough);
  const measures = document.getElementById("voice-measures");
  measures.hidden = !r.voice;
  if (r.voice) {
    const tbody = document.querySelector("#measures tbody");
    tbody.innerHTML = "";
    for (const m of r.voice.measures) {
      const tr = document.createElement("tr");
      const fmt = (v) => (v === null ? "–" : `${v} ${m.unit}`);
      tr.innerHTML = `<td>${m.label}<small>${m.help}</small></td>
        <td class="num">${fmt(m.value)}</td>
        <td class="num">${m.healthy_iqr[0]} – ${m.healthy_iqr[1]} ${m.unit}</td>
        <td class="num">${fmt(m.copd_median)}</td>`;
      tbody.appendChild(tr);
    }
  }
  const section = document.getElementById("result");
  section.hidden = false;
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

analyzeBtn.addEventListener("click", async () => {
  const fd = new FormData();
  for (const task of ["vowel", "counting", "cough"]) {
    if (task !== "cough" && !(recordings.vowel && recordings.counting)) continue;
    if (recordings[task]) fd.append(task, recordings[task].blob, recordings[task].name);
  }
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";
  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Analysis failed");
    renderResult(body);
  } catch (e) {
    alert(`Could not analyze the recordings: ${e.message}`);
  } finally {
    analyzeBtn.textContent = "Analyze my recordings";
    updateAnalyze();
  }
});

function renderCoughInfo(c) {
  if (!c) return;
  document.getElementById("cough-model").hidden = false;
  document.getElementById("cough-cohort").textContent =
    `Dataset: ${c.n_recordings} cough recordings (${c.n_copd} COPD, ${c.n_recordings - c.n_copd} healthy).`;
  const cv = c.cv, nb = c.naive_baseline;
  const rows = [
    ["ROC AUC", cv.auc.toFixed(2), nb.auc.toFixed(2)],
    ["ROC AUC, background swapped", cv.auc_background_swapped.toFixed(2), nb.auc_background_swapped.toFixed(2)],
    ["Decisions flipped by background", `${(cv.decision_flip_rate * 100).toFixed(0)}%`, `${(nb.decision_flip_rate * 100).toFixed(0)}%`],
    ["Sensitivity", cv.sensitivity.toFixed(2), nb.sensitivity.toFixed(2)],
    ["Specificity", cv.specificity.toFixed(2), nb.specificity.toFixed(2)],
  ];
  document.querySelector("#cough-perf tbody").innerHTML =
    `<tr><th></th><th>This model</th><th>Naive model</th></tr>` +
    rows.map(([k, a, b]) => `<tr><td>${k}</td><td class="num">${a}</td><td class="num">${b}</td></tr>`).join("");
}

function renderModelInfo(info) {
  document.getElementById("cohort").textContent =
    `Cohort: ${info.n_participants} participants (${info.n_cases} with chronic lung disease, ${info.n_controls} matched controls).`;
  const svg = document.getElementById("roc");
  const P = 30, S = 180; // padding, plot size
  const pts = info.roc.fpr.map((x, i) => `${P + x * S},${P + S - info.roc.tpr[i] * S - 20}`).join(" ");
  svg.innerHTML = `
    <line class="axis" x1="${P}" y1="${P + S - 20}" x2="${P + S}" y2="${P + S - 20}"/>
    <line class="axis" x1="${P}" y1="${P - 20}" x2="${P}" y2="${P + S - 20}"/>
    <line class="diag" x1="${P}" y1="${P + S - 20}" x2="${P + S}" y2="${P - 20}"/>
    <polyline class="curve" points="${pts}"/>
    <text x="${P + S / 2}" y="${P + S}" text-anchor="middle">False positive rate</text>
    <text x="10" y="${P + S / 2 - 20}" text-anchor="middle" transform="rotate(-90 10 ${P + S / 2 - 20})">True positive rate</text>`;
  const cv = info.cv;
  const rows = [
    ["ROC AUC", `${cv.auc_mean.toFixed(2)} ± ${cv.auc_sd.toFixed(2)}`],
    ["Matched-pair concordance", info.matched_pair_concordance.value.toFixed(2)],
    ["Balanced accuracy", cv.balanced_accuracy.toFixed(2)],
    ["Sensitivity", cv.sensitivity.toFixed(2)],
    ["Specificity", cv.specificity.toFixed(2)],
    ["Age + sex only (baseline AUC)", info.baselines.age_sex_only.auc_mean.toFixed(2)],
    ["Recording year only (baseline AUC)", info.baselines.recording_year_only.auc_mean.toFixed(2)],
  ];
  document.querySelector("#perf tbody").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td class="num">${v}</td></tr>`).join("");
  renderCoughInfo(info.cough);
}

fetch("/api/model-info").then((r) => r.json()).then(renderModelInfo).catch(() => {});
