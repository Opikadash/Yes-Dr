import React, { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = ""; // served via gateway: /api/*

function sseParse(buffer, onEvent) {
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

function Bubble({ who, children }) {
  return <div className={`bubble ${who}`}>{children}</div>;
}

export default function App() {
  const [info, setInfo] = useState(null);
  const [question, setQuestion] = useState("");
  const [model, setModel] = useState("");
  const [topK, setTopK] = useState("");
  const [sources, setSources] = useState([]);
  const [log, setLog] = useState([]);
  const [chat, setChat] = useState([]);
  const chatRef = useRef(null);

  const storeLine = useMemo(() => {
    if (!info?.store) return "—";
    return `${info.store.docs} docs • ${info.store.chunks} chunks • ${info.store.vectors} vectors`;
  }, [info]);

  useEffect(() => {
    const run = async () => {
      const res = await fetch(`${API_BASE}/api/info`);
      setInfo(await res.json());
    };
    run();
    const t = setInterval(run, 5000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [chat]);

  async function onUpload(e) {
    e.preventDefault();
    const file = e.target.file.files?.[0];
    if (!file) return;

    const fd = new FormData();
    fd.append("file", file);

    const params = new URLSearchParams();
    const src = e.target.source.value?.trim();
    if (src) params.set("source", src);
    params.set("async_mode", e.target.async_mode.checked ? "true" : "false");

    setLog((l) => [...l, `Uploading: ${file.name}`]);
    const res = await fetch(`${API_BASE}/api/upload-doc?${params.toString()}`, { method: "POST", body: fd });
    const data = await res.json();
    setLog((l) => [...l, `Upload result: ${JSON.stringify(data)}`]);
  }

  async function onAsk(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q) return;
    setQuestion("");
    setSources([]);

    setChat((c) => [...c, { who: "user", text: q }, { who: "assistant", text: "" }]);

    const body = { question: q };
    if (model.trim()) body.model = model.trim();
    if (topK.trim()) body.top_k = Number(topK.trim());

    const res = await fetch(`${API_BASE}/api/query-stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      setChat((c) => {
        const out = [...c];
        out[out.length - 1] = { who: "assistant", text: `Error: ${res.status}` };
        return out;
      });
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      buf = sseParse(buf, ({ event, data }) => {
        if (event === "sources") {
          try {
            setSources(JSON.parse(data));
          } catch {}
        }
        if (event === "token") {
          try {
            const token = JSON.parse(data);
            setChat((c) => {
              const out = [...c];
              out[out.length - 1] = { who: "assistant", text: out[out.length - 1].text + token };
              return out;
            });
          } catch {}
        }
      });
    }
  }

  return (
    <div className="page">
      <header className="top">
        <div className="brand">
          <div className="badge">AI</div>
          <div>
            <div className="title">Yes Doctor - Medical Assistant</div>
            <div className="sub">
              Private local AI ({info?.llm_backend === "openai_compat" ? "vLLM" : "Ollama"}) + RAG over your documents
            </div>
          </div>
        </div>
        <div className="pill">{storeLine}</div>
      </header>

      <main className="grid">
        <section className="card">
          <h2>Ingest</h2>
          <form onSubmit={onUpload} className="form">
            <input name="file" type="file" />
            <input name="source" placeholder="source label (optional)" />
            <label className="row">
              <input name="async_mode" type="checkbox" defaultChecked /> Background ingest
            </label>
            <button className="btn" type="submit">
              Upload
            </button>
          </form>
          <pre className="log">{log.slice(-8).join("\n") || "—"}</pre>
        </section>

        <section className="card">
          <h2>Chat</h2>
          <div className="row">
            <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="model override (optional)" />
            <input value={topK} onChange={(e) => setTopK(e.target.value)} placeholder="top_k (optional)" />
          </div>
          <div className="chat" ref={chatRef}>
            {chat.map((m, i) => (
              <Bubble who={m.who} key={i}>
                {m.text}
              </Bubble>
            ))}
          </div>
          <form className="ask" onSubmit={onAsk}>
            <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question…" />
            <button className="btn" type="submit">
              Send
            </button>
          </form>
          <details className="sources">
            <summary>Sources</summary>
            <pre className="log">
              {sources?.length
                ? sources.map((s, i) => `S${i + 1}: ${s.source || "unknown"} (chunk ${s.id})\n${s.preview}`).join("\n\n---\n\n")
                : "—"}
            </pre>
          </details>
          <div className="note">Demo only. Always verify medical information with a qualified clinician.</div>
        </section>
      </main>
    </div>
  );
}
