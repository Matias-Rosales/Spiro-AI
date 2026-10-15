const recordings = { vowel: null, counting: null };
const analyzeBtn = document.getElementById("analyze");
let active = null; // the recorder currently running

function updateAnalyze() {
  analyzeBtn.disabled = !(recordings.vowel && recordings.counting) || active !== null;
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

function renderResult(r) {
  const pct = Math.round(r.probability * 100);
  document.getElementById("prob").textContent = `${pct}%`;
  const fill = document.querySelector(".gauge .fill");
  const len = fill.getTotalLength();
  fill.style.strokeDasharray = len;
  fill.style.strokeDashoffset = len * (1 - r.probability);
  const colors = { low: "--low", moderate: "--mid", elevated: "--high" };
  const color = `var(${colors[r.band]})`;
  fill.style.stroke = color;
  const band = document.getElementById("band");
  band.textContent = { low: "Low likelihood", moderate: "Moderate likelihood", elevated: "Elevated likelihood" }[r.band];
  band.style.color = color;
  document.getElementById("band-text").textContent = {
    low: "Your voice resembles the healthy recordings in our data.",
    moderate: "Some voice features resemble recordings from people with chronic lung disease.",
    elevated: "Your voice shares several patterns with recordings from people with chronic lung disease. Consider asking a doctor about a spirometry test, especially if you have cough, breathlessness or a smoking history.",
  }[r.band];

  const tbody = document.querySelector("#measures tbody");
  tbody.innerHTML = "";
  for (const m of r.measures) {
    const tr = document.createElement("tr");
    const fmt = (v) => (v === null ? "–" : `${v} ${m.unit}`);
    tr.innerHTML = `<td>${m.label}<small>${m.help}</small></td>
      <td class="num">${fmt(m.value)}</td>
      <td class="num">${m.healthy_iqr[0]} – ${m.healthy_iqr[1]} ${m.unit}</td>
      <td class="num">${fmt(m.copd_median)}</td>`;
    tbody.appendChild(tr);
  }
  const section = document.getElementById("result");
  section.hidden = false;
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

analyzeBtn.addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("vowel", recordings.vowel.blob, recordings.vowel.name);
  fd.append("counting", recordings.counting.blob, recordings.counting.name);
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
    analyzeBtn.textContent = "Analyze my voice";
    updateAnalyze();
  }
});

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
}

fetch("/api/model-info").then((r) => r.json()).then(renderModelInfo).catch(() => {});
