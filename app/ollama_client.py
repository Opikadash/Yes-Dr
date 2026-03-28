from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class OllamaClient:
    base_url: str

    def generate(self, *, model: str, prompt: str, options: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options

        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", ""))

    def generate_stream(self, *, model: str, prompt: str, options: dict[str, Any] | None = None):
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
        if options:
            payload["options"] = options

        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    import json

                    obj = json.loads(line)
                except Exception:
                    continue

                chunk = str(obj.get("response", ""))
                done = bool(obj.get("done", False))

                if chunk:
                    yield chunk
                if done:
                    break
