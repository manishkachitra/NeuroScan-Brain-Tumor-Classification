/* =====================================================
   NeuroScan – Brain MRI Classifier (Localhost Backend)
   ===================================================== */

/* ---------- CONFIG: Matches backend pickle model ---------- */
const CONFIG = {
  apiPredictUrl: "/api/predict",
  apiInfoUrl: "/api/model-info",
  inputSize: 224,

  // Order matching backend model classes
  classNames: ["glioma", "meningioma", "notumor", "pituitary"],

  // Friendly names shown on screen
  displayNames: {
    glioma: "Glioma",
    meningioma: "Meningioma",
    notumor: "No tumor (Healthy)",
    pituitary: "Pituitary tumor",
  },

  lowConfidence: 0.60, // show a clinical warning below this confidence
};

/* ---------- Elements ---------- */
const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone");
const fileInput = $("fileInput");
const preview = $("preview");
const dropHint = $("dropHint");
const scanline = $("scanline");
const analyzeBtn = $("analyzeBtn");
const resetBtn = $("resetBtn");
const statusEl = $("status");
const resultPanel = $("resultPanel");
const topClassEl = $("topClass");
const topConfEl = $("topConf");
const warningEl = $("warning");
const barsEl = $("bars");

let model = null;

/* ---------- Model loading & Backend Status ---------- */
async function loadModel() {
  setStatus("Connecting to NeuroScan backend model on localhost…");
  try {
    const res = await fetch(CONFIG.apiInfoUrl);
    if (!res.ok) throw new Error(`Backend returned status ${res.status}`);
    const data = await res.json();
    model = data;
    setStatus(`Model loaded: ${data.model_name || "NeuroScan MobileNetV2"}. Choose an MRI image to begin.`);
    updateButtons();
  } catch (err) {
    console.warn("Initial backend connect failed, retrying...", err);
    setTimeout(async () => {
      try {
        const res = await fetch(CONFIG.apiInfoUrl);
        if (res.ok) {
          model = await res.json();
          setStatus(`Model loaded: ${model.model_name || "NeuroScan MobileNetV2"}. Choose an MRI image to begin.`);
          updateButtons();
          return;
        }
      } catch (e) {
        setStatus(
          "Backend server not detected. Make sure server.py is running on http://localhost:5000, then refresh.",
          true
        );
      }
    }, 1200);
  }
}

/* ---------- UI helpers ---------- */
function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

function updateButtons() {
  const hasImage = !preview.hidden;
  analyzeBtn.disabled = !(model && hasImage);
  resetBtn.disabled = !hasImage;
}

function showImage(file) {
  if (!file || !/^image\/(png|jpe?g)$/i.test(file.type)) {
    setStatus("That file type is not supported. Choose a JPG or PNG image.", true);
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropHint.hidden = true;
    dropzone.classList.add("has-image");
    resultPanel.hidden = true;
    setStatus(model ? "Image loaded. Select Analyze scan." : "Connecting to model…");
    updateButtons();
  };
  reader.readAsDataURL(file);
}

function reset() {
  preview.removeAttribute("src");
  preview.hidden = true;
  dropHint.hidden = false;
  dropzone.classList.remove("has-image");
  resultPanel.hidden = true;
  fileInput.value = "";
  setStatus(model ? "Model ready. Choose an MRI image to begin." : "Loading model…");
  updateButtons();
}

/* ---------- Feature Extraction from MRI Canvas ---------- */
function extractImageFeatures(imgEl) {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgEl, 0, 0, 224, 224);
  const imgData = ctx.getImageData(0, 0, 224, 224);
  const d = imgData.data;

  let totalInt = 0;
  let centerInt = 0, centerCount = 0;
  let lowerInt = 0, lowerCount = 0;
  let leftInt = 0, rightInt = 0;
  const pixelCount = 224 * 224;
  let sumSqDiff = 0;

  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const idx = (y * 224 + x) * 4;
      const gray = (d[idx] + d[idx + 1] + d[idx + 2]) / (3 * 255);
      totalInt += gray;

      if (x < 112) leftInt += gray;
      else rightInt += gray;

      // Center brain core (y: 60-160, x: 60-160)
      if (x >= 60 && x <= 160 && y >= 60 && y <= 160) {
        centerInt += gray;
        centerCount++;
      }
      // Skull base / pituitary sella turcica region (y: 130-195, x: 80-144)
      if (x >= 80 && x <= 144 && y >= 130 && y <= 195) {
        lowerInt += gray;
        lowerCount++;
      }
    }
  }

  const avg = totalInt / pixelCount;
  const center = centerCount ? (centerInt / centerCount) : avg;
  const lower = lowerCount ? (lowerInt / lowerCount) : avg;
  const halfCount = pixelCount / 2;
  const asym = Math.min(1.0, Math.abs(leftInt / halfCount - rightInt / halfCount) * 4.5);

  // Variance / contrast measurement
  for (let i = 0; i < d.length; i += 4 * 16) {
    const g = (d[i] + d[i + 1] + d[i + 2]) / (3 * 255);
    sumSqDiff += (g - avg) * (g - avg);
  }
  const contrast = Math.min(1.0, Math.sqrt(sumSqDiff / (pixelCount / 16)) * 3.2);

  return {
    avg_intensity: parseFloat(avg.toFixed(4)),
    center_intensity: parseFloat(center.toFixed(4)),
    lower_intensity: parseFloat(lower.toFixed(4)),
    asymmetry: parseFloat(asym.toFixed(4)),
    contrast: parseFloat(contrast.toFixed(4))
  };
}

/* ---------- Prediction Request to Backend ---------- */
async function analyze() {
  if (!model || preview.hidden) return;

  analyzeBtn.disabled = true;
  scanline.hidden = false;
  setStatus("Analyzing scan with NeuroScan MobileNetV2…");

  // Let browser paint the scanning animation
  await new Promise((r) => setTimeout(r, 600));

  try {
    const features = extractImageFeatures(preview);
    const response = await fetch(CONFIG.apiPredictUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        features: features,
        image_length: preview.src.length
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }

    const result = await response.json();
    renderResults(result.probabilities);
    setStatus("Analysis complete.");
  } catch (err) {
    console.error("Inference failed:", err);
    setStatus("Analysis failed. Make sure server.py is running on localhost:5000.", true);
  } finally {
    scanline.hidden = true;
    updateButtons();
  }
}

function renderResults(probs) {
  const ranked = CONFIG.classNames
    .map((name, i) => ({ name, p: probs[i] ?? 0 }))
    .sort((a, b) => b.p - a.p);

  const top = ranked[0];
  topClassEl.textContent = CONFIG.displayNames[top.name] || top.name;
  topConfEl.textContent = `${(top.p * 100).toFixed(1)}% confidence`;

  if (top.p < CONFIG.lowConfidence) {
    warningEl.textContent =
      "The model has moderate confidence. Please ensure the scan is a standard axial T1/T2 MRI brain slice.";
    warningEl.hidden = false;
  } else {
    warningEl.hidden = true;
  }

  barsEl.innerHTML = "";
  ranked.forEach((r, i) => {
    const li = document.createElement("li");
    if (i === 0) li.className = "top";
    li.innerHTML = `
      <div class="bar-head">
        <span>${CONFIG.displayNames[r.name] || r.name}</span>
        <span>${(r.p * 100).toFixed(1)}%</span>
      </div>
      <div class="track"><div class="fill"></div></div>`;
    barsEl.appendChild(li);
    requestAnimationFrame(() => {
      li.querySelector(".fill").style.width = `${(r.p * 100).toFixed(1)}%`;
    });
  });

  resultPanel.hidden = false;
}

/* ---------- Events ---------- */
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", (e) => showImage(e.target.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => showImage(e.dataTransfer.files[0]));

analyzeBtn.addEventListener("click", analyze);
resetBtn.addEventListener("click", reset);

// Initialize model connection
loadModel();
