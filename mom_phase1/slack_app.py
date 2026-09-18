"""
Slack integration for mom-phase1.

Run this from the `note_taker/` directory (or set MOM_DOCS_DIR /
MOM_KNOWLEDGE_ROOT env vars) so it can see the same `knowledge/` folder
the CLI uses.

Commands (all Socket Mode — no public URL, no exposed Ollama):
  /mom-extract         Opens a modal with a file-upload field. Upload a
                        .txt/.vtt transcript; the bot runs Phase 1 and
                        posts the result back to the channel.
  /mom-features        Lists discovered feature docs.
  /mom-show <feature>  Posts one feature doc as a snippet.
  /mom-ask <question>  Answers a question from the knowledge docs.
  /mom-merge "A" "B" [...]   Combine feature docs into the first.
  /mom-model           Shows which Ollama model is in use.
  /mom-skills          Lists all of the above.

Setup: see slack_integration/README.md in this folder. Each of these needs
its own Slash Command entry created in api.slack.com/apps (Slash Commands
page) before Slack will route it to this bot — see the README for exact
steps.
"""

import contextlib
import io
import os
import re
import sys
import tempfile
import threading

import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Make the mom_phase1 package importable regardless of which subfolder of
# note_taker/ this file lives in (note_taker/mom_phase1/ here).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mom_phase1.cli import run_phase1                       # noqa: E402
from mom_phase1.extract import answer_question, synthesize_user_story  # noqa: E402
from mom_phase1.knowledge_docs import apply_merges, discover_features  # noqa: E402
from mom_phase1.ollama_client import DEFAULT_MODEL, OllamaError  # noqa: E402

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]

app = App(token=SLACK_BOT_TOKEN)

# Matches quoted or bare tokens in "/mom-merge "A" "B" C" -- same parsing
# cli.py's /merge uses.
_QUOTED_RE = re.compile(r'"([^"]+)"|(\S+)')


# ---------------------------------------------------------------------------
# /mom-extract — opens a modal that accepts a transcript file upload
# ---------------------------------------------------------------------------

@app.command("/mom-extract")
def open_extract_modal(ack, body, client):
    ack()
    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "mom_extract_submit",
            # remember which channel to reply in once the model finishes
            "private_metadata": body["channel_id"],
            "title": {"type": "plain_text", "text": "Run Phase 1"},
            "submit": {"type": "plain_text", "text": "Extract"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "transcript_block",
                    "label": {"type": "plain_text", "text": "Transcript (.txt or .vtt)"},
                    "element": {
                        # "vtt" isn't in Slack's file_input filetypes whitelist,
                        # so don't restrict here -- we check the extension
                        # ourselves once the file's downloaded (see
                        # handle_extract_submit below).
                        "type": "file_input",
                        "action_id": "transcript_file",
                        "max_files": 1,
                    },
                }
            ],
        },
    )


@app.view("mom_extract_submit")
def handle_extract_submit(ack, body, client):
    ack()
    channel_id = body["view"]["private_metadata"]
    user_id = body["user"]["id"]
    files = (
        body["view"]["state"]["values"]["transcript_block"]["transcript_file"]
        .get("files", [])
    )
    if not files:
        client.chat_postMessage(channel=channel_id, text=f"<@{user_id}> no file was attached — try `/mom-extract` again.")
        return

    slack_file = files[0]
    ext = os.path.splitext(slack_file["name"])[1].lower()
    if ext not in (".txt", ".vtt"):
        client.chat_postMessage(
            channel=channel_id,
            text=f"<@{user_id}> `{slack_file['name']}` isn't a .txt or .vtt file — try `/mom-extract` again with a transcript.",
        )
        return

    # Run the (possibly slow, multi-LLM-call) extraction off the event loop.
    threading.Thread(
        target=_run_extraction_job,
        args=(slack_file, channel_id, user_id, client),
        daemon=True,
    ).start()
    client.chat_postMessage(
        channel=channel_id,
        text=f"<@{user_id}> got `{slack_file['name']}` — running Phase 1 against `{DEFAULT_MODEL}`, will post here when done.",
    )


def _run_extraction_job(slack_file, channel_id, user_id, client):
    tmp_path = _download_slack_file(slack_file)
    if tmp_path is None:
        client.chat_postMessage(channel=channel_id, text=f"<@{user_id}> couldn't download that file from Slack.")
        return

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run_phase1(tmp_path)
    except OllamaError as exc:
        client.chat_postMessage(channel=channel_id, text=f"<@{user_id}> Ollama error: `{exc}`")
        return
    finally:
        os.unlink(tmp_path)

    output = buf.getvalue().strip() or "(no output)"
    client.chat_postMessage(
        channel=channel_id,
        text=f"<@{user_id}> Phase 1 finished for `{slack_file['name']}`:",
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": f"<@{user_id}> Phase 1 finished for `{slack_file['name']}`:"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"```{output[:2900]}```"}},
        ],
    )


def _download_slack_file(slack_file) -> str | None:
    """Download a Slack-uploaded file to a temp path, using the bot token."""
    url = slack_file.get("url_private_download") or slack_file.get("url_private")
    if not url:
        return None
    resp = requests.get(url, headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}, timeout=30)
    resp.raise_for_status()
    suffix = os.path.splitext(slack_file["name"])[1] or ".txt"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path


# ---------------------------------------------------------------------------
# /mom-features — list discovered feature docs
# ---------------------------------------------------------------------------

@app.command("/mom-features")
def list_features(ack, respond):
    ack()
    feats = discover_features()
    if not feats:
        respond("No feature docs yet.")
        return
    lines = [f"• *{name}*" for name in sorted(feats)]
    respond("\n".join(["Feature docs:"] + lines))


# ---------------------------------------------------------------------------
# /mom-show <feature> — post one feature doc
# ---------------------------------------------------------------------------

@app.command("/mom-show")
def show_feature(ack, respond, command):
    ack()
    query = command["text"].strip().lower()
    if not query:
        respond("Usage: `/mom-show <feature name>`")
        return
    feats = discover_features()
    for name, path in feats.items():
        if query in name.lower():
            with open(path, encoding="utf-8") as f:
                content = f.read()
            respond(f"*{name}*\n```{content[:2900]}```")
            return
    respond(f'No feature doc matching "{query}".')


# ---------------------------------------------------------------------------
# /mom-ask <question> — answer from the knowledge docs
# ---------------------------------------------------------------------------

@app.command("/mom-ask")
def ask_question(ack, respond, command):
    ack()
    question = command["text"].strip()
    if not question:
        respond("Usage: `/mom-ask <question>`")
        return

    def job():
        texts = {}
        for name, path in discover_features().items():
            try:
                with open(path, encoding="utf-8") as f:
                    texts[name] = f.read()
            except OSError:
                continue
        try:
            answer = answer_question(question, texts)
        except OllamaError as exc:
            respond(f"Ollama error: `{exc}`")
            return
        respond(answer)

    threading.Thread(target=job, daemon=True).start()
    respond(f"Thinking about: _{question}_ …")


# ---------------------------------------------------------------------------
# /mom-merge "A" "B" [...] — fold B, C, ... into A (keeps every fact)
# ---------------------------------------------------------------------------

@app.command("/mom-merge")
def merge_docs(ack, respond, command):
    ack()
    argstr = command["text"].strip()
    if not argstr:
        respond('Usage: `/mom-merge "First Doc" "Second Doc" ["Third" ...]`')
        return

    raw = [a or b for a, b in _QUOTED_RE.findall(argstr)]
    feats = discover_features()
    lower = {n.lower(): n for n in feats}
    names, missing = [], []
    for tok in raw:
        hit = lower.get(tok.lower()) or next(
            (n for n in feats if tok.lower() in n.lower()), None)
        (names if hit else missing).append(hit or tok)
    names = list(dict.fromkeys(names))

    if missing:
        respond("No feature doc for: " + ", ".join(f'"{m}"' for m in missing))
    if len(names) < 2:
        respond('Usage: `/mom-merge "First Doc" "Second Doc" ["Third" ...]`')
        return

    def job():
        try:
            results = apply_merges([names], story_fn=synthesize_user_story, explicit=True)
        except OllamaError as exc:
            respond(f"Ollama error: `{exc}`")
            return
        respond("\n".join(results) if results else "Nothing merged.")

    threading.Thread(target=job, daemon=True).start()
    respond(f"Merging {', '.join(names)} …")


# ---------------------------------------------------------------------------
# /mom-model — show which Ollama model is in use
# ---------------------------------------------------------------------------

@app.command("/mom-model")
def show_model(ack, respond):
    ack()
    respond(f"model: `{DEFAULT_MODEL}`  (override with the `MOM_MODEL` env var on the bot's host)")


# ---------------------------------------------------------------------------
# /mom-skills — list all commands
# ---------------------------------------------------------------------------

_SKILLS = [
    ("/mom-extract", "", "Upload a .txt/.vtt transcript, run Phase 1 on it."),
    ("/mom-features", "", "List the feature docs discovered so far."),
    ("/mom-show", "<feature>", "Show one feature doc (partial name match)."),
    ("/mom-ask", "<question>", "Answer a question from the knowledge docs."),
    ("/mom-merge", '"A" "B" [...]', "Combine feature docs into the first."),
    ("/mom-model", "", "Show which Ollama model is in use."),
    ("/mom-skills", "", "Show this list."),
]


@app.command("/mom-skills")
def list_skills(ack, respond):
    ack()
    lines = [f"• `{cmd} {args}`".rstrip() + f" — {desc}" for cmd, args, desc in _SKILLS]
    respond("\n".join(["Available commands:"] + lines))


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()