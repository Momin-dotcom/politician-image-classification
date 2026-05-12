const API_BASE = "/api";

document.getElementById("imageInput").addEventListener("change", function () {
  const file = this.files[0];
  if (!file) return;

  const preview = document.getElementById("preview");
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;

  document.getElementById("uploadBox").style.display = "none";
  document.getElementById("predictBtn").disabled = false;
  document.getElementById("results").hidden = true;
  document.getElementById("error").hidden = true;
});

async function predict() {
  const file = document.getElementById("imageInput").files[0];
  if (!file) return;

  document.getElementById("loading").hidden = false;
  document.getElementById("results").hidden = true;
  document.getElementById("error").hidden = true;
  document.getElementById("predictBtn").disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction failed");
    }

    const data = await res.json();
    displayResults(data.top3_predictions);

  } catch (err) {
    const errorDiv = document.getElementById("error");
    errorDiv.textContent = "Error: " + err.message;
    errorDiv.hidden = false;
  } finally {
    document.getElementById("loading").hidden = true;
    document.getElementById("predictBtn").disabled = false;
  }
}

function displayResults(predictions) {
  const cards = document.getElementById("predictionCards");
  cards.innerHTML = "";

  predictions.forEach((pred, i) => {
    const card = document.createElement("div");
    card.className = "card" + (i === 0 ? " top" : "");
    card.innerHTML = `
      <span class="rank">#${i + 1}</span>
      <span class="name">${pred.class.replace(/_/g, " ").toUpperCase()}</span>
      <div class="bar-container">
        <div class="bar" style="width: ${pred.confidence}%"></div>
      </div>
      <span class="confidence">${pred.confidence}%</span>
    `;
    cards.appendChild(card);
  });

  document.getElementById("results").hidden = false;
}
