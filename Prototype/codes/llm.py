import os, time, requests
from typing import Optional

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
OPENAI_COMPAT_PATH = "/v1/chat/completions"
TIMEOUT = float(os.getenv("VLLM_HTTP_TIMEOUT", "120"))

def _post_chat(prompt: str, temperature: float, max_new_tokens: int) -> str:
    url = VLLM_BASE_URL.rstrip("/") + OPENAI_COMPAT_PATH
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": max(0.0, min(2.0, temperature)),
        "max_tokens": max(1, int(max_new_tokens)),
    }
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def chat(prompt: str, temperature: float = 0.2, max_new_tokens: int = 512, retries: int = 2, backoff: float = 1.5) -> str:
    last_err: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            txt = _post_chat(prompt, temperature, max_new_tokens)
            return txt.strip()
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(backoff ** i)
            else:
                raise
    raise RuntimeError(f"vLLM chat failed: {last_err}")
