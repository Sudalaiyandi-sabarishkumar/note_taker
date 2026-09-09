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
_VOICE_TAG_RE = re.compile(r"<v\b\s*([^>]*)>(.*?)</v>", re.DOTALL)
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")


def _parse_vtt(raw: str) -> str:
    lines = []
    pending_start = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("WEBVTT") or _CUE_NUMBER_RE.match(line):
            continue
        if _TIMESTAMP_RE.match(line):
            pending_start = line.split("-->")[0].strip()
            continue
        prefix = f"[{pending_start}] " if pending_start else ""
        m = _VOICE_TAG_RE.search(line)
        if m:
            speaker = m.group(1).strip() or "Unidentified speaker"
            text = _STRIP_TAGS_RE.sub("", m.group(2)).strip()
            lines.append(f"{prefix}{speaker}: {text}")
        else:
            text = _STRIP_TAGS_RE.sub("", line).strip()
            if text:
                lines.append(f"{prefix}Unidentified speaker: {text}"
                             if prefix else text)
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


def chunk_transcript(text: str, max_chars: int = 2500, overlap_lines: int = 2):
    """Split on line boundaries (whole speaker turns) so a chunk never cuts a
    sentence -- the model must be able to quote verbatim from what it sees.

    Chunks are kept small (a 7B extracts more completely from a short, dense
    span than from a long one), and each chunk repeats the last
    ``overlap_lines`` lines of the previous one so a requirement stated right
    at a boundary is not lost. Duplicate statements from the overlap are
    removed downstream by normalised-quote de-dup.
    """
    lines = [ln for ln in text.splitlines()]
    if sum(len(ln) + 1 for ln in lines) <= max_chars:
        return ["\n".join(lines)] if lines else [text]

    chunks, current, current_len = [], [], 0
    for line in lines:
        if current and current_len + len(line) + 1 > max_chars:
            chunks.append(current)
            current = current[-overlap_lines:] if overlap_lines else []
            current_len = sum(len(x) + 1 for x in current)
        current.append(line)
        current_len += len(line) + 1
    if current and (not chunks or current[overlap_lines:]):
        chunks.append(current)
    return ["\n".join(c) for c in chunks] or [text]
