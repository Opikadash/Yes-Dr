const statusPill = document.getElementById("statusPill");
const storeStats = document.getElementById("storeStats");
const modelStats = document.getElementById("modelStats");
const ingestLog = document.getElementById("ingestLog");
const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const sourceInput = document.getElementById("sourceInput");
const asyncMode = document.getElementById("asyncMode");
const forceIngest = document.getElementById("forceIngest");

const chat = document.getElementById("chat");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sourcesBox = document.getElementById("sourcesBox");
const modelInput = document.getElementById("modelInput");
const topKInput = document.getElementById("topKInput");

function addBubble(text, who) {
  const div = document.createElement("div");
  div.className = `bubble ${who}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function logLine(line) {
  ingestLog.textContent = (ingestLog.textContent ? ingestLog.textContent + "\n" : "") + line;
  ingestLog.scrollTop = ingestLog.scrollHeight;
}

async function refreshInfo() {
  try {
    const res = await fetch("/api/info");
    const data = await res.json();
    statusPill.textContent = "Online";
    statusPill.style.borderColor = "rgba(32, 201, 151, 0.5)";
    storeStats.textContent = `${data.store.docs} docs • ${data.store.chunks} chunks • ${data.store.vectors} vectors`;
    modelStats.textContent = `LLM: ${data.ollama_model} • Embed: ${data.embedding_model}`;
  } catch {
    statusPill.textContent = "Offline";
    statusPill.style.borderColor = "rgba(255, 255, 255, 0.12)";
  }
}

async function pollJob(jobId) {
  for (let i = 0; i < 120; i++) {
    const res = await fetch(`/jobs/${jobId}`);
    const job = await res.json();
    if (job.status === "done") return job.result;
    if (job.status === "error") throw new Error(job.error || "Job failed");
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("Timed out waiting for background ingest");
}

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = fileInput.files?.[0];
  if (!f) return;

  const fd = new FormData();
  fd.append("file", f);

  const params = new URLSearchParams();
  if (sourceInput.value.trim()) params.set("source", sourceInput.value.trim());
  params.set("async_mode", asyncMode.checked ? "true" : "false");
  params.set("force", forceIngest.checked ? "true" : "false");

  logLine(`Uploading: ${f.name}`);

  const res = await fetch(`/upload-doc?${params.toString()}`, { method: "POST", body: fd });
  if (!res.ok) {
    logLine(`Upload failed: ${res.status}`);
    return;
  }
  const data = await res.json();
  if (asyncMode.checked && data.status === "queued") {
    logLine(`Queued background job: ${data.job_id}`);
    try {
      const done = await pollJob(data.job_id);
      logLine(`Ingested: +${done.added_chunks} chunks (doc_id=${done.doc_id || "?"})`);
    } catch (err) {
      logLine(`Job error: ${err.message}`);
    }
  } else {
    if (data.skipped) logLine(`Skipped (dedup): ${f.name}`);
    else logLine(`Ingested: +${data.added_chunks} chunks (doc_id=${data.doc_id || "?"})`);
  }
  await refreshInfo();
});

function parseSSE(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  for (let i = 0; i < parts.length - 1; i++) {
    const msg = parts[i];
    let event = "message";
    let data = "";
    for (const line of msg.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    onEvent({ event, data });
  }
  return parts[parts.length - 1];
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = questionInput.value.trim();
  if (!q) return;
  questionInput.value = "";

  addBubble(q, "user");
  const assistantBubble = addBubble("", "assistant");
  sourcesBox.textContent = "—";

  const body = {
    question: q,
  };
  const m = modelInput.value.trim();
  const topK = topKInput.value.trim();
  if (m) body.model = m;
  if (topK) body.top_k = Number(topK);

  const res = await fetch("/query-stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    assistantBubble.textContent = `Error: ${res.status}`;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    buf = parseSSE(buf, ({ event, data }) => {
      if (event === "sources") {
        try {
          const sources = JSON.parse(data);
          sourcesBox.textContent = sources
            .map((s, i) => `S${i + 1}: ${s.source || "unknown"} (chunk ${s.id})\n${s.preview}`)
            .join("\n\n---\n\n");
        } catch {}
      } else if (event === "token") {
        try {
          const token = JSON.parse(data);
          assistantBubble.textContent += token;
          chat.scrollTop = chat.scrollHeight;
        } catch {}
      }
    });
  }
});

refreshInfo();
setInterval(refreshInfo, 5000);
