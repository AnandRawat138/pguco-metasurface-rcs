// Thin client for the FastAPI backend. Every POST uses multipart form data, matching the Form(...) endpoints.

async function handle(r) {
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg = j.detail || JSON.stringify(j);
    } catch (_) {
      /* not JSON */
    }
    throw new Error(`${r.status}: ${msg}`);
  }
  return r.json();
}

export const get = (path, params) => {
  const q = params ? "?" + new URLSearchParams(params).toString() : "";
  return fetch(`/api${path}${q}`).then(handle);
};

export const post = (path, fields = {}, method = "POST") => {
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) {
    if (v === undefined || v === null) continue;
    fd.append(k, v instanceof Blob ? v : typeof v === "object" ? JSON.stringify(v) : String(v));
  }
  return fetch(`/api${path}`, { method, body: fd }).then(handle);
};

export const fileUrl = (path) => `/api/files?path=${encodeURIComponent(path)}`;
