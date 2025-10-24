# src/llm/client_ollama.py
from __future__ import annotations
import os, requests

class OllamaClient:
    """
    Minimal HTTP client for a local Ollama server (http://localhost:11434).
    Set LLM_MODEL / LLM_BASE_URL via environment or .env.
    """
    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: int = 60):
        self.model = model or os.getenv("LLM_MODEL", "llama3.1")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434")
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            if max_tokens is not None:
                payload["options"] = {"num_predict": max_tokens}
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception:
            # return empty => caller will trigger parser fallback
            return ""