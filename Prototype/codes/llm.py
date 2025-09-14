
import os
import sys
import time
import json
import requests
from typing import Optional

# Backend selection
VALID = {"ollama", "openrouter"}
BACKEND = os.getenv("PEC_BACKEND", "").strip().lower()  # ollama | openrouter


def _choose_backend_interactively() -> str:
    print("[PEC] Select backend [ollama / openrouter]: ", end="", flush=True)
    choice = sys.stdin.readline().strip().lower()
    if choice not in VALID:
        print(f"[PEC] Unrecognized backend '{choice}'. Defaulting to 'ollama'.")
        choice = "ollama"
    return choice


def _get_backend() -> str:
    global BACKEND
    if BACKEND in VALID:
        return BACKEND
    BACKEND = _choose_backend_interactively()
    return BACKEND

def _clip_temperature(t: float) -> float:
    try:
        t = float(t)
    except Exception:
        t = 0.2
    return max(0.0, min(2.0, t))


def _max_tokens(x: int) -> int:
    try:
        x = int(x)
    except Exception:
        x = 512
    return max(1, x)



# Ollama config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_HTTP_TIMEOUT", "120"))


def _chat_ollama(prompt: str, temperature: float, max_new_tokens: int) -> str:   
    url = OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "options": {
            "temperature": _clip_temperature(temperature),
            "num_predict": _max_tokens(max_new_tokens),
        },
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_BASE_URL}. "
            "Start it with `ollama serve` (Windows: ensure the service is running), "
            "or set OLLAMA_BASE_URL to the correct host/port. "
            "If calling from WSL to a Windows Ollama, either run Ollama inside WSL or expose the Windows service."
        ) from e
    except Exception as e:
        raise

    # Typical response shape: {"message": {"content": "..."}}
    msg = (data.get("message") or {}).get("content", "")
    if not msg:
        # Older/alternative responses may return a messages list
        msgs = data.get("messages")
        if isinstance(msgs, list) and msgs:
            msg = msgs[-1].get("content", "") or ""
    return (msg or "").strip()



# OpenRouter config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai")

# Default to OpenAI-compatible path; we also support /responses via fallback or env override.
OPENROUTER_PATH = os.getenv("OPENROUTER_PATH", "/api/v1/chat/completions")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_HTTP_TIMEOUT", "120"))

# Optional headers 
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "https://github.com/hpriyankaa/fossee-python-task")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "PEC Prototype")


def _chat_openrouter(prompt: str, temperature: float, max_new_tokens: int) -> str:
    """
    Calls OpenRouter. Tries /chat/completions first; if 404, falls back to /responses.
    Set OPENROUTER_API_KEY and optionally OPENROUTER_MODEL.
    You may also force a path with OPENROUTER_PATH=/api/v1/responses
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Get one at https://openrouter.ai/ and export it.")

    base = OPENROUTER_BASE_URL.rstrip("/")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": OPENROUTER_TITLE,
    }

    def post_chat_completions() -> str:
        url = f"{base}/api/v1/chat/completions"
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": _clip_temperature(temperature),
            "max_tokens": _max_tokens(max_new_tokens),
        }
        r = requests.post(url, headers=headers, json=payload, timeout=OPENROUTER_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    def post_responses() -> str:
        url = f"{base}/api/v1/responses"
        payload = {
            "model": OPENROUTER_MODEL,
            "input": [{"role": "user", "content": prompt}],
            "temperature": _clip_temperature(temperature),
            "max_output_tokens": _max_tokens(max_new_tokens),
        }
        r = requests.post(url, headers=headers, json=payload, timeout=OPENROUTER_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        
        out = data.get("output")
        if isinstance(out, list) and out:
            for piece in reversed(out):
                content = piece.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text" and "text" in block:
                                return str(block["text"]).strip()
                            if "text" in block:
                                return str(block["text"]).strip()
                            if "content" in block and isinstance(block["content"], str):
                                return block["content"].strip()
                elif isinstance(content, str):
                    return content.strip()
        return json.dumps(data, ensure_ascii=False)

    forced = OPENROUTER_PATH.strip()
    if forced.endswith("/responses"):
        return post_responses()
    if forced.endswith("/chat/completions"):
        return post_chat_completions()

    try:
        return post_chat_completions()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return post_responses()
        raise


# Public API
def chat(
    prompt: str,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    retries: int = 2,
    backoff: float = 1.5,
) -> str:
    """
    Unified chat() for Ollama or OpenRouter.
    Backend chosen via PEC_BACKEND (ollama|openrouter); if unset, asks once.
    Retries transient failures with exponential backoff.
    """
    backend = _get_backend()
    last_err: Optional[Exception] = None

    for i in range(retries + 1):
        try:
            if backend == "ollama":
                return _chat_ollama(prompt, temperature, max_new_tokens)
            elif backend == "openrouter":
                return _chat_openrouter(prompt, temperature, max_new_tokens)
            else:
                raise ValueError(f"Unsupported backend: {backend}")
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(backoff ** i)
            else:
                raise RuntimeError(f"chat() failed for backend '{backend}': {last_err}") from last_err
