"""Thin Ollama HTTP client, stdlib-only (no ``requests`` dependency).

Some models (e.g. deepseek-r1) wrap their reply in a ``<think> ... </think>``
block that is not part of the answer. ``chat()`` returns the raw text; call
:func:`strip_think` before parsing (a no-op for plain instruct models).
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

# llama-server occasionally returns a 500 "Compute error" (transient Metal/GPU
# hiccup). Retrying the same request almost always succeeds -- so a one-off
# failure shouldn't sink a long multi-call run.
_TRANSIENT = ("compute error", "500", "llama runner process has terminated",
              "failed to decode", "cuda error")
_MAX_TRIES = 3
_RETRY_BACKOFF = 4  # seconds, doubled each retry

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

# The custom model built from ./Modelfile. Override with MOM_MODEL, e.g.
# `MOM_MODEL=qwen2.5:7b-instruct-q4_K_M` to run against the base directly.
DEFAULT_MODEL = os.environ.get("MOM_MODEL", "mom-phase1")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaError(RuntimeError):
    pass


def strip_think(text: str) -> str:
    """Remove deepseek-r1 reasoning blocks. Also handles a dangling
    ``<think>`` with no close tag (truncated generation) by dropping
    everything up to the last ``</think>`` if one exists, else leaving the
    text alone."""
    if text is None:
        return ""
    cleaned = _THINK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1]
    return cleaned.strip()


def chat(messages, model=None, num_ctx=6144, num_predict=1536,
         temperature=0.0, show_progress=True, timeout=600):
    """Stream a chat completion from Ollama and return the full text.

    Raises :class:`OllamaError` if Ollama is unreachable or the model is
    missing, with a message the CLI surfaces directly to the user.
    """
    model = model or DEFAULT_MODEL
    last_err = None
    for attempt in range(1, _MAX_TRIES + 1):
        try:
            return _chat_once(messages, model, num_ctx, num_predict,
                              temperature, show_progress, timeout)
        except OllamaError as exc:
            msg = str(exc).lower()
            transient = any(t in msg for t in _TRANSIENT)
            if not transient or attempt == _MAX_TRIES:
                raise
            last_err = exc
            wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
            print(f"\n  (Ollama transient error, retry {attempt}/{_MAX_TRIES - 1} "
                  f"in {wait}s...)")
            time.sleep(wait)
    raise last_err  # unreachable, keeps linters happy


def _chat_once(messages, model, num_ctx, num_predict, temperature,
               show_progress, timeout):
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    try:
        try:
            resp_cm = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as http_err:
            # Ollama returns 404 + a JSON body like {"error":"model 'x' not found"}
            body = http_err.read().decode("utf-8", "replace")
            try:
                msg = json.loads(body).get("error", body)
            except json.JSONDecodeError:
                msg = body or str(http_err)
            raise OllamaError(_explain_error(msg, model)) from http_err
        with resp_cm as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                piece = json.loads(line)
                if piece.get("error"):
                    raise OllamaError(_explain_error(piece["error"], model))
                content = piece.get("message", {}).get("content", "")
                if content:
                    if show_progress:
                        print(content, end="", flush=True)
                    chunks.append(content)
                if piece.get("done"):
                    break
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Could not reach Ollama at {OLLAMA_URL} ({exc.reason}). "
            f"Is it running?  Try:  brew services start ollama"
        ) from exc
    if show_progress:
        print()
    return "".join(chunks).strip()


def _explain_error(err: str, model: str) -> str:
    if "not found" in err.lower() or "no such model" in err.lower():
        return (
            f"Model '{model}' is not available in Ollama.\n"
            f"  Build the Phase 1 model:   ollama create mom-phase1 -f Modelfile\n"
            f"  or run against the base:    MOM_MODEL=qwen2.5:7b-instruct-q4_K_M mom-phase1\n"
            f"  (first pull it once:        ollama pull qwen2.5:7b-instruct-q4_K_M )"
        )
    return f"Ollama error: {err}"
