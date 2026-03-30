from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass
class Result:
    latency_s: float
    n_chars: int
    streamed: bool = False


async def one(
    client: httpx.AsyncClient,
    url: str,
    question: str,
    top_k: int | None,
    model: str | None,
) -> Result:
    payload: dict = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k
    if model:
        payload["model"] = model

    t0 = time.perf_counter()
    r = await client.post(url, json=payload)
    r.raise_for_status()
    data = r.json()
    answer = str(data.get("answer", ""))
    return Result(latency_s=time.perf_counter() - t0, n_chars=len(answer), streamed=False)


async def one_openai(
    client: httpx.AsyncClient,
    base: str,
    question: str,
    top_k: int | None,
    model: str | None,
    stream: bool,
) -> Result:
    payload: dict = {"messages": [{"role": "user", "content": question}], "stream": stream}
    if top_k is not None:
        payload["top_k"] = top_k
    if model:
        payload["model"] = model

    t0 = time.perf_counter()
    url = f"{base.rstrip('/')}/v1/chat/completions"
    if not stream:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        content = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
        return Result(latency_s=time.perf_counter() - t0, n_chars=len(content), streamed=False)

    n_chars = 0
    async with client.stream("POST", url, json=payload) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                delta = obj.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content")
                if token:
                    n_chars += len(str(token))
            except Exception:
                continue

    return Result(latency_s=time.perf_counter() - t0, n_chars=n_chars, streamed=True)


async def run(args) -> None:
    url = args.url.rstrip("/")
    endpoint = f"{url}/query"
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(connect=10, read=120, write=10, pool=10)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        sem = asyncio.Semaphore(args.concurrency)
        results: list[Result] = []

        async def task():
            async with sem:
                if args.endpoint == "openai":
                    res = await one_openai(
                        client,
                        url,
                        args.question,
                        args.top_k,
                        args.model,
                        stream=bool(args.stream),
                    )
                else:
                    res = await one(client, endpoint, args.question, args.top_k, args.model)
                results.append(res)

        await asyncio.gather(*[task() for _ in range(args.requests)])

    latencies = [r.latency_s for r in results]
    if not latencies:
        print("No results.")
        return

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(0.50 * (len(latencies_sorted) - 1))]
    p95 = latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))]
    avg = statistics.mean(latencies)

    print("Benchmark results")
    print(f"- url: {url}")
    print(f"- endpoint: {args.endpoint}{' (stream)' if args.stream else ''}")
    print(f"- requests: {args.requests}")
    print(f"- concurrency: {args.concurrency}")
    print(f"- avg_latency_s: {avg:.3f}")
    print(f"- p50_latency_s: {p50:.3f}")
    print(f"- p95_latency_s: {p95:.3f}")
    print(f"- avg_answer_chars: {statistics.mean([r.n_chars for r in results]):.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple concurrency benchmark for /query")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the RAG API")
    ap.add_argument("--endpoint", choices=["query", "openai"], default="query")
    ap.add_argument("--stream", action="store_true", help="Only for --endpoint openai")
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--question", default="Summarize the key points from the uploaded documents.")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
