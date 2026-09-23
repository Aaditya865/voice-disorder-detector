const API_BASE = "http://localhost:8000";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelInfo() {
  const res = await fetch(`${API_BASE}/model-info`);
  if (!res.ok) throw new Error(`Failed to fetch model info: ${res.statusText}`);
  return res.json();
}

export async function fetchClasses() {
  const res = await fetch(`${API_BASE}/classes`);
  if (!res.ok) throw new Error(`Failed to fetch classes: ${res.statusText}`);
  return res.json();
}

export async function fetchDatasetStatistics() {
  const res = await fetch(`${API_BASE}/dataset-statistics`);
  if (!res.ok) throw new Error(`Failed to fetch dataset statistics: ${res.statusText}`);
  return res.json();
}

export async function fetchEvaluation() {
  const res = await fetch(`${API_BASE}/evaluation`);
  if (!res.ok) throw new Error(`Failed to fetch evaluation metrics: ${res.statusText}`);
  return res.json();
}

export async function triggerTrain(testSize = 0.20, valSize = 0.10) {
  const res = await fetch(`${API_BASE}/train?test_size=${testSize}&val_size=${valSize}`, {
    method: "POST"
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.message || data.detail || "Training failed");
  }
  return data;
}

export async function extractWaveform(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/extract-waveform`, {
    method: "POST",
    body: formData
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to extract waveform envelope");
  }
  return res.json();
}

export async function predictVoice(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.message || data.detail || "Prediction failed");
  }
  return data;
}
