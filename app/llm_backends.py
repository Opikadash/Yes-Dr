from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import requests

from .ollama_client import OllamaClient


class LLMBackend:
    def generate(self, *, model: str, prompt: str) -> str:  # pragma: no cover (interface)
        raise NotImplementedError

    def generate_stream(self, *, model: str, prompt: str) -> Iterable[str]:  # pragma: no cover (interface)
        raise NotImplementedError


@dataclass(frozen=True)
class OllamaBackend(LLMBackend):
    client: OllamaClient

    def generate(self, *, model: str, prompt: str) -> str:
        return self.client.generate(model=model, prompt=prompt)

    def generate_stream(self, *, model: str, prompt: str):
        return self.client.generate_stream(model=model, prompt=prompt)


@dataclass(frozen=True)
class OpenAICompatBackend(LLMBackend):
    base_url: str  # should include /v1
    api_key: str = ""

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, *, model: str, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except Exception:
            return ""

    def generate_stream(self, *, model: str, prompt: str):
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        with requests.post(url, headers=self._headers(), json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                try:
                    delta = obj["choices"][0]["delta"]
                    token = delta.get("content")
                except Exception:
                    token = None
                if token:
                    yield str(token)

