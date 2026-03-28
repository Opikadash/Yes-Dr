import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 3000;
const RAG_API_BASE = process.env.RAG_API_BASE || "http://localhost:8000";

const app = express();
app.use(express.json({ limit: "2mb" }));

app.get("/health", (_req, res) => res.json({ status: "ok", rag_api: RAG_API_BASE }));

app.get("/api/info", async (_req, res) => {
  const r = await fetch(`${RAG_API_BASE}/api/info`);
  const data = await r.json();
  res.status(r.status).json(data);
});

app.post("/api/query", async (req, res) => {
  const r = await fetch(`${RAG_API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req.body || {}),
  });
  const data = await r.text();
  res.status(r.status).set("Content-Type", r.headers.get("content-type") || "application/json").send(data);
});

// Proxy SSE streaming from FastAPI to browsers. Keeps recruiters happy: Node <-> local AI.
app.post("/api/query-stream", async (req, res) => {
  const r = await fetch(`${RAG_API_BASE}/query-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req.body || {}),
  });

  res.status(r.status);
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  if (!r.ok || !r.body) {
    res.write(`event: error\ndata: ${JSON.stringify(`Upstream error: ${r.status}`)}\n\n`);
    res.end();
    return;
  }

  const reader = r.body.getReader();
  const encoder = new TextEncoder();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) res.write(Buffer.from(value));
    }
  } catch (e) {
    res.write(encoder.encode(`event: error\ndata: ${JSON.stringify(String(e))}\n\n`));
  } finally {
    res.end();
  }
});

// OpenAI-style API passthrough (useful with vLLM + OpenAI SDKs pointing at the gateway)
app.get("/v1/models", async (_req, res) => {
  const r = await fetch(`${RAG_API_BASE}/v1/models`);
  const data = await r.text();
  res.status(r.status).set("Content-Type", r.headers.get("content-type") || "application/json").send(data);
});

app.post("/v1/chat/completions", async (req, res) => {
  const r = await fetch(`${RAG_API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req.body || {}),
  });

  // If streaming, pass through SSE; otherwise just forward JSON
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("text/event-stream")) {
    res.status(r.status);
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    if (!r.ok || !r.body) return res.end();
    const reader = r.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) res.write(Buffer.from(value));
    }
    return res.end();
  }

  const data = await r.text();
  res.status(r.status).set("Content-Type", ct || "application/json").send(data);
});

// Proxy multipart uploads to FastAPI
app.post("/api/upload-doc", async (req, res) => {
  const url = new URL(`${RAG_API_BASE}/upload-doc`);
  for (const [k, v] of Object.entries(req.query || {})) url.searchParams.set(k, String(v));

  const r = await fetch(url.toString(), {
    method: "POST",
    headers: { "content-type": req.headers["content-type"] || "" },
    body: req,
    duplex: "half",
  });
  const data = await r.text();
  res.status(r.status).set("Content-Type", r.headers.get("content-type") || "application/json").send(data);
});

// Serve React build if present (bundled into the gateway image as /app/web/dist)
const webDist = path.join(__dirname, "web", "dist");
app.use(express.static(webDist));
app.get("/", (_req, res) => res.sendFile(path.join(webDist, "index.html")));

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Gateway listening on http://localhost:${PORT} -> ${RAG_API_BASE}`);
});
