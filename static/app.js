const form = document.querySelector("#prediction-form");
const grid = document.querySelector("#feature-grid");
const count = document.querySelector("#feature-count");
const clearButton = document.querySelector("#clear-button");
const predictButton = document.querySelector("#predict-button");
const resultCard = document.querySelector("#result-card");
const errorMessage = document.querySelector("#error-message");

let featureNames = [];

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

function createField(name) {
  const label = document.createElement("label");
  label.textContent = name;

  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.inputMode = "decimal";
  input.name = name;
  input.placeholder = "Optional";

  label.appendChild(input);
  return label;
}

async function loadFeatures() {
  try {
    const response = await fetch("/api/features");
    if (!response.ok) throw new Error("Unable to load model features.");

    const data = await response.json();
    featureNames = data.features;
    count.textContent = `${featureNames.length} features expected by the trained model`;
    featureNames.forEach((name) => grid.appendChild(createField(name)));
  } catch (error) {
    showError(error.message);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  resultCard.hidden = true;
  predictButton.disabled = true;
  predictButton.textContent = "Predicting…";

  const values = {};
  for (const name of featureNames) {
    const raw = form.elements[name].value.trim();
    values[name] = raw === "" ? null : Number(raw);
  }

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: values })
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Prediction failed.");

    document.querySelector("#logqe-result").textContent = data.predicted_logqe.toFixed(4);
    document.querySelector("#qe-result").textContent = data.predicted_qe.toFixed(4);
    document.querySelector("#result-note").textContent = `Qe was calculated using logarithm base ${data.log_base}.`;
    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showError(error.message);
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Predict logQe";
  }
});

clearButton.addEventListener("click", () => {
  form.reset();
  resultCard.hidden = true;
  clearError();
});

loadFeatures();
