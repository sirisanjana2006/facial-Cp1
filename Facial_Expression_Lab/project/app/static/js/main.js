// ============================================================
// FACIAL EXPRESSION LAB — front-end logic
// ============================================================

const EMOTION_COLORS = {
  Anger: "var(--e-anger)",
  Contempt: "var(--e-contempt)",
  Disgust: "var(--e-disgust)",
  Fear: "var(--e-fear)",
  Happy: "var(--e-happy)",
  Sadness: "var(--e-sadness)",
  Surprise: "var(--e-surprise)",
  Neutral: "var(--e-neutral)",
};

function colorFor(label) {
  return EMOTION_COLORS[label] || "var(--ink-soft)";
}


// ------------------------------------------------------------
// UPLOAD + PREDICT DEMO
// ------------------------------------------------------------

(function setupDemo() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const emptyState = document.getElementById("dropzone-empty");
  const previewImage = document.getElementById("preview-image");
  const predictBtn = document.getElementById("predict-btn");
  const resultBox = document.getElementById("result");
  const resultEmotion = document.getElementById("result-emotion");
  const resultConfidence = document.getElementById("result-confidence");
  const probList = document.getElementById("prob-list");
  const errorBox = document.getElementById("demo-error");

  let selectedFile = null;

  function showPreview(file) {
    const url = URL.createObjectURL(file);
    previewImage.src = url;
    previewImage.hidden = false;
    emptyState.hidden = true;
    predictBtn.disabled = false;
    resultBox.hidden = true;
    errorBox.hidden = true;
  }

  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    selectedFile = file;
    showPreview(file);
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    handleFile(e.target.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    handleFile(file);
  });

  predictBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    predictBtn.textContent = "Reading...";
    errorBox.hidden = true;

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Something went wrong.");
      }

      renderResult(data);
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.hidden = false;
      resultBox.hidden = true;
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "Read the expression";
    }
  });

  function renderResult(data) {
    resultEmotion.textContent = data.top_emotion;
    resultEmotion.style.color = colorFor(data.top_emotion);
    resultConfidence.textContent =
      Math.round(data.confidence * 100) + "% confident";

    probList.innerHTML = "";

    data.probabilities.forEach(({ label, value }) => {
      const li = document.createElement("li");
      li.className = "prob-row";
      li.innerHTML = `
        <span class="prob-label">${label}</span>
        <span class="prob-track">
          <span class="prob-fill" style="background:${colorFor(label)}"></span>
        </span>
        <span class="prob-value">${Math.round(value * 100)}%</span>
      `;
      probList.appendChild(li);
    });

    resultBox.hidden = false;

    // animate bars in on next frame so the transition triggers
    requestAnimationFrame(() => {
      const fills = probList.querySelectorAll(".prob-fill");
      data.probabilities.forEach(({ value }, i) => {
        fills[i].style.width = Math.round(value * 100) + "%";
      });
    });
  }
})();


// ------------------------------------------------------------
// RESULTS + DATASET SECTIONS (populated from /api/results)
// ------------------------------------------------------------

(async function loadDashboardData() {
  let data;
  try {
    const response = await fetch("/api/results");
    data = await response.json();
  } catch (err) {
    return; // leave the "empty" placeholders in place
  }

  if (!data || !data.available) return;

  renderComparison(data);
  renderPerClass(data);
  renderDataset(data);
})();

function renderComparison(data) {
  document.getElementById("results-content").hidden = false;
  document.getElementById("results-empty").hidden = true;

  const baselinePct = Math.round(data.attention.accuracy * 100);
  const attentionPct = Math.round(data.baseline.accuracy * 100);

  const barBaseline = document.getElementById("bar-baseline");
  const barAttention = document.getElementById("bar-attention");

  document.getElementById("value-baseline").textContent = baselinePct + "%";
  document.getElementById("value-attention").textContent = attentionPct + "%";

  requestAnimationFrame(() => {
    barBaseline.style.width = baselinePct + "%";
    barAttention.style.width = attentionPct + "%";
  });

  const delta = attentionPct - baselinePct;
  const agreementPct = Math.round(data.agreement_rate * 100);
  const deltaText =
    delta > 0
      ? `Attention scored ${delta} point${delta === 1 ? "" : "s"} higher on the test set.`
      : delta < 0
      ? `Attention scored ${Math.abs(delta)} point${Math.abs(delta) === 1 ? "" : "s"} lower on the test set.`
      : `Both models scored the same on the test set.`;

  document.getElementById("agreement-note").textContent =
    `${deltaText} The two models agreed on ${agreementPct}% of the ${data.test_set_size} test images.`;
}

function renderPerClass(data) {
  const list = document.getElementById("per-class-list");
  list.innerHTML = "";

  data.classes.forEach((label, i) => {
    const baseline = data.attention.per_class_accuracy[i];
    const attention = data.baseline.per_class_accuracy[i];

    const li = document.createElement("li");
    li.className = "per-class-row";

    const baselinePct = baseline == null ? "—" : Math.round(baseline * 100) + "%";
    const attentionPct = attention == null ? "—" : Math.round(attention * 100) + "%";
    const attentionWidth = attention == null ? 0 : Math.round(attention * 100);

    li.innerHTML = `
      <span class="emotion-swatch" style="background:${colorFor(label)}"></span>
      <span class="per-class-label">${label}</span>
      <span class="per-class-track">
        <span class="per-class-fill per-class-fill--attention" style="background:${colorFor(label)}"></span>
      </span>
      <span class="per-class-value" title="baseline">${baselinePct}</span>
      <span class="per-class-value" title="attention">${attentionPct}</span>
    `;
    list.appendChild(li);

    requestAnimationFrame(() => {
      li.querySelector(".per-class-fill--attention").style.width = attentionWidth + "%";
    });
  });
}

function renderDataset(data) {
  document.getElementById("dataset-content").hidden = false;
  document.getElementById("dataset-empty").hidden = true;

  const list = document.getElementById("dist-list");
  list.innerHTML = "";

  const trainCounts = data.class_distribution.train;
  const max = Math.max(...trainCounts);

  data.classes.forEach((label, i) => {
    const count = trainCounts[i];
    const width = max ? Math.round((count / max) * 100) : 0;

    const li = document.createElement("li");
    li.className = "dist-row";
    li.innerHTML = `
      <span class="emotion-swatch" style="background:${colorFor(label)}"></span>
      <span class="per-class-label">${label}</span>
      <span class="dist-track">
        <span class="dist-fill" style="background:${colorFor(label)}"></span>
      </span>
      <span class="dist-value">${count}</span>
    `;
    list.appendChild(li);

    requestAnimationFrame(() => {
      li.querySelector(".dist-fill").style.width = width + "%";
    });
  });
}
