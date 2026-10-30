// Same limit as MAX_TEXT_LENGTH in api.py.
export const MAX_TEXT_LENGTH = 5000;

async function request(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("ارتباط با سرور برقرار نشد.");
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    // FastAPI's own validation errors carry a list here; only our messages are plain strings.
    const detail = typeof body?.detail === "string" ? body.detail : null;
    throw new Error(detail ?? "ارتباط با سرور برقرار نشد.");
  }
  return body;
}

export function fetchModel(signal) {
  return request("/api/model", { signal });
}

export function predict(text, signal) {
  return request("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal,
  });
}
