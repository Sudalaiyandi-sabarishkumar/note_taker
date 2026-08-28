"""Load and chunk a call transcript.

Accepts plain ``.txt`` (one ``Speaker: text`` line per turn) or Teams
``.vtt`` export. For .vtt the cue start-time is kept as a ``[HH:MM:SS.mmm]``
prefix on each line so it can flow into the citation trail -- Phase 1 needs
a real timestamp, not just a speaker name.
"""

import os
import re

_CUE_NUMBER_RE = re.compile(r"^\d+$")
_TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}"
)
_VOICE_TAG_RE = re.compile(r"<v\s+([^>]+)>(.*?)</v>", re.DOTALL)


def _parse_vtt(raw: str) -> str:
    lines = []
    pending_start = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or _CUE_NUMBER_RE.match(line):
            continue
        if _TIMESTAMP_RE.match(line):
            pending_start = line.split("-->")[0].strip()
            continue
        m = _VOICE_TAG_RE.search(line)
        if m:
            speaker, text = m.group(1).strip(), m.group(2).strip()
            prefix = f"[{pending_start}] " if pending_start else ""
            lines.append(f"{prefix}{speaker}: {text}")
        else:
            lines.append(line)
    return "\n".join(lines)


def load_transcript(path: str):
    """Returns ``(text, error)`` -- exactly one is None."""
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError as exc:
        return None, f"Couldn't read {path}: {exc}"

    text = _parse_vtt(raw) if path.lower().endswith(".vtt") else raw.strip()
    if not text:
        return None, f"{path} is empty after parsing."
    return text, None


def chunk_transcript(text: str, max_chars: int = 5000):
    """Split on line boundaries (whole speaker turns) so a chunk never cuts
    a sentence -- the model must be able to quote verbatim from what it
    sees. A real hour-long call is many chunks; that is the normal path."""
    chunks, current, current_len = [], [], 0
    for line in text.splitlines():
        if current and current_len + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks or [text]
