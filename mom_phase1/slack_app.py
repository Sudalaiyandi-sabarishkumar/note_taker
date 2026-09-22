"""
Slack integration for mom-phase1.

Run this from the `note_taker/` directory (or set MOM_DOCS_DIR /
MOM_KNOWLEDGE_ROOT env vars) so it can see the same `knowledge/` folder
the CLI uses by default.

Each Slack channel can point at its own docs directory -- one bot process
serves many channels, each with an isolated set of feature docs, organized
as named "projects" under `projects/`. Channels that never create/switch
to a project fall back to MOM_DOCS_DIR / `knowledge`. The channel -> dir
mapping is persisted to `channel_docs_dirs.json` next to this file, so it
survives a bot restart.

Commands (all Socket Mode — no public URL, no exposed Ollama). /extract,
/features, /show, /ask, and /open-questions all open a modal that starts
with a project picker, so each run is explicit about which project it
reads/writes instead of relying on whatever the channel happens to be
switched to:
  /extract          Modal: pick a project, upload a .txt/.vtt
                         transcript. The bot runs Phase 1 against that
                         project's dir and posts the result back to the
                         channel.
  /features          Modal: pick a project, lists its discovered
                         feature docs.
  /show              Modal: pick a project and a feature name (partial
                         match ok), posts that feature doc as a snippet.
  /ask               Modal: pick a project and a question, answers it
                         from that project's knowledge docs.
  /open-questions    Modal: pick a project, lists unresolved
                         "Open Questions" from its feature docs.
  /merge "A" "B" [...]   Combine feature docs into the first.
  /model             Shows which Ollama model is in use.
  /create-project <name>   Creates a new project and switches this
                         channel to it.
  /switch-project [name]   Switches this channel to an existing
                         project, or lists available projects if no name
                         is given.
  /help             Lists all of the above.

Setup: see slack_integration/README.md in this folder. Each of these needs
its own Slash Command entry created in api.slack.com/apps (Slash Commands
page) before Slack will route it to this bot — see the README for exact
steps.
"""

import io
import json
import os
import re
import sys
import tempfile
import threading

import requests
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Load SLACK_BOT_TOKEN / SLACK_APP_TOKEN (and anything else you add) from
# note_taker/.env -- no more manual `export` before every run. Safe to call
# even if .env doesn't exist (falls back to whatever's already in the shell
# environment).
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Make the mom_phase1 package importable regardless of which subfolder of
# note_taker/ this file lives in (note_taker/mom_phase1/ here).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mom_phase1 import knowledge_docs                        # noqa: E402
from mom_phase1.cli import run_phase1                       # noqa: E402
from mom_phase1.extract import answer_question, synthesize_user_story  # noqa: E402
from mom_phase1.knowledge_docs import apply_merges, discover_features  # noqa: E402
from mom_phase1.ollama_client import DEFAULT_MODEL, OllamaError  # noqa: E402

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]

app = App(token=SLACK_BOT_TOKEN)

# ---------------------------------------------------------------------------
# Per-channel docs directory, organized as named projects
# ---------------------------------------------------------------------------
# {channel_id: docs_dir}. Channels not in this map use knowledge_docs.DOCS_DIR
# (i.e. MOM_DOCS_DIR / "knowledge") as their default. Persisted to disk so
# the mapping survives a bot restart.
#
# Projects live as subdirectories of PROJECTS_ROOT, one per project name, so
# /create-project and /switch-project can create/list/validate them
# without the caller having to know or type a full path.

_CHANNEL_DIRS_PATH = os.path.join(os.path.dirname(__file__), "..", "channel_docs_dirs.json")
PROJECTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "projects")
_channel_dirs_lock = threading.Lock()


def _load_channel_dirs() -> dict:
    try:
        with open(_CHANNEL_DIRS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[mom] warning: couldn't read {_CHANNEL_DIRS_PATH}: {exc}", file=sys.stderr)
        return {}


_channel_dirs = _load_channel_dirs()


def _save_channel_dirs() -> None:
    tmp_path = _CHANNEL_DIRS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_channel_dirs, f, indent=2, sort_keys=True)
    os.replace(tmp_path, _CHANNEL_DIRS_PATH)


def _docs_dir_for(channel_id: str) -> str:
    """This channel's docs dir if it's been set, else the process default."""
    return _channel_dirs.get(channel_id, knowledge_docs.DOCS_DIR)


def _project_path(name: str) -> str:
    """Resolve a project name to its directory path under PROJECTS_ROOT."""
    return os.path.join(PROJECTS_ROOT, name)


def _list_projects() -> list:
    """Names of existing projects (subdirectories of PROJECTS_ROOT)."""
    if not os.path.isdir(PROJECTS_ROOT):
        return []
    return sorted(
        d for d in os.listdir(PROJECTS_ROOT)
        if os.path.isdir(os.path.join(PROJECTS_ROOT, d))
    )


def _project_select_element(channel_id: str) -> dict:
    """A static_select block element listing every project, pre-selecting
    whichever one this channel is currently switched to (if any)."""
    options = [
        {"text": {"type": "plain_text", "text": name}, "value": name}
        for name in _list_projects()
    ]
    current_name = os.path.basename(_docs_dir_for(channel_id).rstrip(os.sep))
    initial_option = next((o for o in options if o["value"] == current_name), None)

    element = {
        "type": "static_select",
        "action_id": "project_select",
        "placeholder": {"type": "plain_text", "text": "Choose a project"},
        "options": options,
    }
    if initial_option is not None:
        element["initial_option"] = initial_option
    return element


# One lock per docs_dir, created on first use. Two extractions into the same
# project (e.g. two people, or the same person twice) are serialized so
# run_phase1's read-modify-write of feature docs can't race; extractions
# into different projects use different locks and still run fully in
# parallel.
_project_locks_guard = threading.Lock()
_project_locks: dict = {}


def _lock_for(docs_dir: str) -> threading.Lock:
    with _project_locks_guard:
        return _project_locks.setdefault(docs_dir, threading.Lock())


# ---------------------------------------------------------------------------
# Thread-safe stdout capture for /extract
# ---------------------------------------------------------------------------
# run_phase1() (cli.py) reports progress via plain print(). contextlib's
# redirect_stdout swaps out sys.stdout for the WHOLE PROCESS, not just the
# calling thread -- fine for the single-threaded CLI, but /extract runs
# each job on its own background thread, and two jobs can be in flight at
# once (two channels, or two people, extracting at the same time). With
# redirect_stdout, one job's redirect can silently steal another job's
# print() output into the wrong buffer, and Slack ends up showing a channel's
# extraction result mixed with or missing part of another channel's. This
# wrapper keys off the calling thread instead, so each job's prints only
# ever land in that job's own buffer.
class _PerThreadStdout:
    def __init__(self, default):
        self._default = default
        self._local = threading.local()

    def register(self, buf) -> None:
        self._local.buf = buf

    def unregister(self) -> None:
        if hasattr(self._local, "buf"):
            del self._local.buf

    def write(self, s: str) -> None:
        (getattr(self._local, "buf", None) or self._default).write(s)

    def flush(self) -> None:
        (getattr(self._local, "buf", None) or self._default).flush()


sys.stdout = _PerThreadStdout(sys.stdout)


# Matches quoted or bare tokens in "/merge "A" "B" C" -- same parsing
# cli.py's /merge uses.
_QUOTED_RE = re.compile(r'"([^"]+)"|(\S+)')

# Matches a markdown heading that starts an "Open Questions" section, e.g.
# "## Open Questions" or "### Open Question". Assumes feature docs mark
# unresolved items this way -- adjust the wording here if yours differ.
_OPEN_Q_HEADING_RE = re.compile(r'^(#{1,6})\s*open questions?\b.*$', re.IGNORECASE | re.MULTILINE)


def _extract_open_questions(content: str) -> list:
    """Pull the bullet/numbered items out of a doc's "Open Questions"
    section, if it has one. Stops at the next heading of the same or
    higher level (or end of file)."""
    match = _OPEN_Q_HEADING_RE.search(content)
    if not match:
        return []
    level = len(match.group(1))
    start = match.end()
    next_heading_re = re.compile(rf'^#{{1,{level}}}\s+\S', re.MULTILINE)
    end_match = next_heading_re.search(content, start)
    section = content[start: end_match.start() if end_match else len(content)]

    questions = []
    for line in section.splitlines():
        # Only top-level bullets (no leading indent) are real entries --
        # an indented "- *quote*" line under an item is supporting evidence,
        # not a separate question, so skip those.
        if line[:1] in (" ", "\t"):
            continue
        line = line.strip()
        if line.startswith(("-", "*", "•")):
            item = line.lstrip("-*• ").strip()
        elif re.match(r'^\d+[.)]\s+', line):
            item = re.sub(r'^\d+[.)]\s+', '', line).strip()
        else:
            continue
        if item:
            questions.append(_clean_question_text(item))
    return [q for q in questions if q and q.rstrip(".").strip().lower() != "none"]


# Strips the "**ID** [TAG]: " prefix and the "— raised by ..." / "— not yet
# decided; ..." attribution suffix off a raw Open Questions bullet, leaving
# just the plain question/statement text.
_OQ_PREFIX_RE = re.compile(r'^\*{0,2}[A-Z]+-[\w.-]+\*{0,2}\s*(?:\[[^\]]+\]\s*)?:?\s*')


def _clean_question_text(item: str) -> str:
    item = _OQ_PREFIX_RE.sub("", item).strip()
    # Cut off the trailing attribution/reasoning, which is separated from
    # the actual question by an em dash (e.g. "— raised by ...", "— not yet
    # decided; follows from EF-1. (raised by ...)").
    item = re.split(r'\s+—\s+', item, maxsplit=1)[0].strip()
    return item.strip(' *"\'')


# ---------------------------------------------------------------------------
# /extract — opens a modal that accepts a transcript file upload
# ---------------------------------------------------------------------------

@app.command("/extract")
def open_extract_modal(ack, respond, body, client):
    ack()
    channel_id = body["channel_id"]
    projects = _list_projects()

    if not projects:
        respond(
            "No projects exist yet, so there's nothing to extract into.\n"
            "Use `/create-project <name>` first, then run `/extract` again."
        )
        return

    project_options = [
        {"text": {"type": "plain_text", "text": name}, "value": name}
        for name in projects
    ]
    # Pre-select whichever project this channel is currently pointed at, if
    # it's a real project (not the fallback knowledge dir).
    current_dir = _docs_dir_for(channel_id)
    current_name = os.path.basename(current_dir.rstrip(os.sep))
    initial_option = next(
        (o for o in project_options if o["value"] == current_name), None
    )

    project_select_element = {
        "type": "static_select",
        "action_id": "project_select",
        "placeholder": {"type": "plain_text", "text": "Choose a project"},
        "options": project_options,
    }
    if initial_option is not None:
        project_select_element["initial_option"] = initial_option

    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "mom_extract_submit",
            # remember which channel to reply in once the model finishes
            "private_metadata": channel_id,
            "title": {"type": "plain_text", "text": "Run Phase 1"},
            "submit": {"type": "plain_text", "text": "Extract"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "project_block",
                    "label": {"type": "plain_text", "text": "Project"},
                    "element": project_select_element,
                },
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
                },
            ],
        },
    )


@app.view("mom_extract_submit")
def handle_extract_submit(ack, body, client):
    ack()
    channel_id = body["view"]["private_metadata"]
    user_id = body["user"]["id"]
    values = body["view"]["state"]["values"]

    project_name = values["project_block"]["project_select"]["selected_option"]["value"]
    docs_dir = _project_path(project_name)
    if not os.path.isdir(docs_dir):
        # Project could've been deleted/renamed between opening the modal
        # and submitting it.
        client.chat_postMessage(
            channel=channel_id,
            text=f"<@{user_id}> project `{project_name}` no longer exists — try `/extract` again.",
        )
        return

    files = values["transcript_block"]["transcript_file"].get("files", [])
    if not files:
        client.chat_postMessage(channel=channel_id, text=f"<@{user_id}> no file was attached — try `/extract` again.")
        return

    slack_file = files[0]
    ext = os.path.splitext(slack_file["name"])[1].lower()
    if ext not in (".txt", ".vtt"):
        client.chat_postMessage(
            channel=channel_id,
            text=f"<@{user_id}> `{slack_file['name']}` isn't a .txt or .vtt file — try `/extract` again with a transcript.",
        )
        return

    # Run the (possibly slow, multi-LLM-call) extraction off the event loop.
    threading.Thread(
        target=_run_extraction_job,
        args=(slack_file, channel_id, user_id, client, docs_dir),
        daemon=True,
    ).start()
    client.chat_postMessage(
        channel=channel_id,
        text=(
            f"<@{user_id}> got `{slack_file['name']}` — running Phase 1 against "
            f"`{DEFAULT_MODEL}` for project `{project_name}`, writing to "
            f"`{docs_dir}/`, will post here when done."
        ),
    )


def _run_extraction_job(slack_file, channel_id, user_id, client, docs_dir):
    tmp_path = _download_slack_file(slack_file)
    if tmp_path is None:
        client.chat_postMessage(channel=channel_id, text=f"<@{user_id}> couldn't download that file from Slack.")
        return

    # slack_file's own name (e.g. "ex2.txt"), not the random temp path it was
    # downloaded to -- otherwise citations/change-log entries would show
    # something like "tmp60eusw_4" instead of "ex2".
    source_name = os.path.splitext(slack_file["name"])[0]

    buf = io.StringIO()
    sys.stdout.register(buf)
    try:
        with _lock_for(docs_dir):
            run_phase1(tmp_path, source_name=source_name, docs_dir=docs_dir)
    except OllamaError as exc:
        client.chat_postMessage(channel=channel_id, text=f"<@{user_id}> Ollama error: `{exc}`")
        return
    finally:
        sys.stdout.unregister()
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
# /features — opens a modal: pick a project, list its feature docs
# ---------------------------------------------------------------------------

@app.command("/features")
def open_features_modal(ack, respond, body, client):
    ack()
    channel_id = body["channel_id"]
    if not _list_projects():
        respond("No projects exist yet. Use `/create-project <name>` first.")
        return

    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "features_submit",
            "private_metadata": channel_id,
            "title": {"type": "plain_text", "text": "Features"},
            "submit": {"type": "plain_text", "text": "Show"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "project_block",
                    "label": {"type": "plain_text", "text": "Project"},
                    "element": _project_select_element(channel_id),
                },
            ],
        },
    )


@app.view("features_submit")
def handle_features_submit(ack, body, client):
    ack()
    channel_id = body["view"]["private_metadata"]
    project_name = (
        body["view"]["state"]["values"]["project_block"]["project_select"]["selected_option"]["value"]
    )
    docs_dir = _project_path(project_name)

    feats = discover_features(docs_dir=docs_dir)
    if not feats:
        client.chat_postMessage(channel=channel_id, text=f"No feature docs yet in project `{project_name}`.")
        return
    lines = [f"• *{name}*" for name in sorted(feats)]
    client.chat_postMessage(
        channel=channel_id,
        text="\n".join([f"Feature docs in `{project_name}`:"] + lines),
    )


# ---------------------------------------------------------------------------
# /show — opens a modal: pick a project and a feature, posts that doc
# ---------------------------------------------------------------------------

@app.command("/show")
def open_show_modal(ack, respond, body, client):
    ack()
    channel_id = body["channel_id"]
    if not _list_projects():
        respond("No projects exist yet. Use `/create-project <name>` first.")
        return

    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "show_submit",
            "private_metadata": channel_id,
            "title": {"type": "plain_text", "text": "Show Feature"},
            "submit": {"type": "plain_text", "text": "Show"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "project_block",
                    "label": {"type": "plain_text", "text": "Project"},
                    "element": _project_select_element(channel_id),
                },
                {
                    "type": "input",
                    "block_id": "feature_block",
                    "label": {"type": "plain_text", "text": "Feature name (partial match ok)"},
                    "element": {"type": "plain_text_input", "action_id": "feature_query"},
                },
            ],
        },
    )


@app.view("show_submit")
def handle_show_submit(ack, body, client):
    ack()
    channel_id = body["view"]["private_metadata"]
    values = body["view"]["state"]["values"]
    project_name = values["project_block"]["project_select"]["selected_option"]["value"]
    query = values["feature_block"]["feature_query"]["value"].strip().lower()
    docs_dir = _project_path(project_name)

    feats = discover_features(docs_dir=docs_dir)
    for name, path in feats.items():
        if query in name.lower():
            with open(path, encoding="utf-8") as f:
                content = f.read()
            client.chat_postMessage(channel=channel_id, text=f"*{name}* (`{project_name}`)\n```{content[:2900]}```")
            return
    client.chat_postMessage(channel=channel_id, text=f'No feature doc matching "{query}" in project `{project_name}`.')


# ---------------------------------------------------------------------------
# /ask — opens a modal: pick a project and a question, answers from its docs
# ---------------------------------------------------------------------------

@app.command("/ask")
def open_ask_modal(ack, respond, body, client):
    ack()
    channel_id = body["channel_id"]
    if not _list_projects():
        respond("No projects exist yet. Use `/create-project <name>` first.")
        return

    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "ask_submit",
            "private_metadata": channel_id,
            "title": {"type": "plain_text", "text": "Ask"},
            "submit": {"type": "plain_text", "text": "Ask"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "project_block",
                    "label": {"type": "plain_text", "text": "Project"},
                    "element": _project_select_element(channel_id),
                },
                {
                    "type": "input",
                    "block_id": "question_block",
                    "label": {"type": "plain_text", "text": "Question"},
                    "element": {"type": "plain_text_input", "action_id": "question_text"},
                },
            ],
        },
    )


@app.view("ask_submit")
def handle_ask_submit(ack, body, client):
    ack()
    channel_id = body["view"]["private_metadata"]
    values = body["view"]["state"]["values"]
    project_name = values["project_block"]["project_select"]["selected_option"]["value"]
    question = values["question_block"]["question_text"]["value"].strip()
    docs_dir = _project_path(project_name)

    def job():
        texts = {}
        for name, path in discover_features(docs_dir=docs_dir).items():
            try:
                with open(path, encoding="utf-8") as f:
                    texts[name] = f.read()
            except OSError:
                continue
        try:
            answer = answer_question(question, texts)
        except OllamaError as exc:
            client.chat_postMessage(channel=channel_id, text=f"Ollama error: `{exc}`")
            return
        client.chat_postMessage(channel=channel_id, text=f"*[{project_name}]* {answer}")

    threading.Thread(target=job, daemon=True).start()
    client.chat_postMessage(channel=channel_id, text=f"Thinking about: _{question}_ (project `{project_name}`) …")


# ---------------------------------------------------------------------------
# /open-questions [project] — list unresolved "Open Questions" across a
# project's feature docs
# ---------------------------------------------------------------------------

@app.command("/open-questions")
def open_questions_command(ack, respond, body, client):
    ack()
    channel_id = body["channel_id"]

    if not _list_projects():
        respond("No projects exist yet. Use `/create-project <name>` first.")
        return

    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "open_questions_submit",
            "private_metadata": channel_id,
            "title": {"type": "plain_text", "text": "Open Questions"},
            "submit": {"type": "plain_text", "text": "Show"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "project_block",
                    "label": {"type": "plain_text", "text": "Project"},
                    "element": _project_select_element(channel_id),
                },
            ],
        },
    )


@app.view("open_questions_submit")
def handle_open_questions_submit(ack, body, client):
    ack()
    channel_id = body["view"]["private_metadata"]
    project_name = (
        body["view"]["state"]["values"]["project_block"]["project_select"]["selected_option"]["value"]
    )
    docs_dir = _project_path(project_name)

    feats = discover_features(docs_dir=docs_dir)
    if not feats:
        client.chat_postMessage(channel=channel_id, text=f"No feature docs yet in project `{project_name}`.")
        return

    by_feature = {}
    for name, path in feats.items():
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        questions = _extract_open_questions(content)
        if questions:
            by_feature[name] = questions

    if not by_feature:
        client.chat_postMessage(channel=channel_id, text=f"No open questions found in project `{project_name}`.")
        return

    lines = [f"Open questions in `{project_name}`:"]
    for name in sorted(by_feature):
        lines.append(f"\n*{name}*")
        lines.extend(f"  • {q}" for q in by_feature[name])
    client.chat_postMessage(channel=channel_id, text="\n".join(lines))


# ---------------------------------------------------------------------------
# /merge "A" "B" [...] — fold B, C, ... into A (keeps every fact)
# ---------------------------------------------------------------------------

@app.command("/merge")
def merge_docs(ack, respond, command):
    ack()
    argstr = command["text"].strip()
    if not argstr:
        respond('Usage: `/merge "First Doc" "Second Doc" ["Third" ...]`')
        return
    docs_dir = _docs_dir_for(command["channel_id"])

    raw = [a or b for a, b in _QUOTED_RE.findall(argstr)]
    feats = discover_features(docs_dir=docs_dir)
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
        respond('Usage: `/merge "First Doc" "Second Doc" ["Third" ...]`')
        return

    def job():
        try:
            results = apply_merges([names], docs_dir=docs_dir,
                                   story_fn=synthesize_user_story, explicit=True)
        except OllamaError as exc:
            respond(f"Ollama error: `{exc}`")
            return
        respond("\n".join(results) if results else "Nothing merged.")

    threading.Thread(target=job, daemon=True).start()
    respond(f"Merging {', '.join(names)} …")


# ---------------------------------------------------------------------------
# /model — show which Ollama model is in use
# ---------------------------------------------------------------------------

@app.command("/model")
def show_model(ack, respond):
    ack()
    respond(f"model: `{DEFAULT_MODEL}`  (override with the `MOM_MODEL` env var on the bot's host)")


# ---------------------------------------------------------------------------
# /create-project <name> — create a new project and switch to it
# ---------------------------------------------------------------------------

@app.command("/create-project")
def create_project_command(ack, respond, command):
    ack()
    channel_id = command["channel_id"]
    name = command["text"].strip()

    if not name:
        respond("Usage: `/create-project <name>` to create a new project and switch this channel to it.")
        return

    project_dir = _project_path(name)

    if os.path.isdir(project_dir):
        respond(
            f"A project named `{name}` already exists.\n"
            f"Use `/switch-project {name}` to switch to it instead."
        )
        return

    with _channel_dirs_lock:
        try:
            os.makedirs(project_dir, exist_ok=False)
        except OSError as exc:
            respond(f"Couldn't create `{project_dir}`: `{exc}`")
            return

        _channel_dirs[channel_id] = project_dir
        try:
            _save_channel_dirs()
        except OSError as exc:
            respond(
                f"Created and switched to `{name}`, but couldn't save this to disk "
                f"(`{exc}`) — it won't survive a bot restart."
            )
            return

    respond(
        f"Created project `{name}` and switched this channel to it.\n"
        f"`/extract`, `/features`, `/show`, `/ask`, and `/merge` "
        f"run *in this channel* will read/write there from now on — other channels "
        f"are unaffected. This is saved to disk, so it survives a bot restart."
    )


# ---------------------------------------------------------------------------
# /switch-project [name] — switch to an existing project, or list them
# ---------------------------------------------------------------------------

@app.command("/switch-project")
def switch_project_command(ack, respond, command):
    ack()
    channel_id = command["channel_id"]
    name = command["text"].strip()

    if not name:
        projects = _list_projects()
        current = _docs_dir_for(channel_id)
        if projects:
            listing = "\n".join(f"• `{p}`" for p in projects)
            respond(
                f"This channel's current docs directory: `{current}`\n\n"
                f"Existing projects:\n{listing}\n\n"
                f"Usage: `/switch-project <name>`"
            )
        else:
            respond(
                f"This channel's current docs directory: `{current}`\n"
                f"No projects exist yet. Use `/create-project <name>` to create one."
            )
        return

    project_dir = _project_path(name)

    if not os.path.isdir(project_dir):
        respond(
            f"No project named `{name}` found.\n"
            f"Use `/create-project {name}` to create it."
        )
        return

    with _channel_dirs_lock:
        _channel_dirs[channel_id] = project_dir
        try:
            _save_channel_dirs()
        except OSError as exc:
            respond(
                f"Switched to `{name}` for now, but couldn't save this to disk "
                f"(`{exc}`) — it won't survive a bot restart."
            )
            return

    respond(
        f"Switched this channel to project `{name}`.\n"
        f"`/extract`, `/features`, `/show`, `/ask`, and `/merge` "
        f"run *in this channel* will now read/write there — other channels are "
        f"unaffected. This is saved to disk, so it survives a bot restart."
    )


# ---------------------------------------------------------------------------
# /help — list all commands
# ---------------------------------------------------------------------------

_SKILLS = [
    ("/extract", "", "Pick a project, upload a .txt/.vtt transcript, run the extractor."),
    ("/features", "", "Pick a project, list the feature docs discovered so far."),
    ("/show", "", "Pick a project and feature (partial name match), show that doc."),
    ("/ask", "", "Pick a project and a question, answer it from that project's docs."),
    ("/open-questions", "", "Pick a project, list its open questions."),
    ("/create-project", "<name>", "Create a new project and switch this channel to it."),
    ("/help", "", "Show the list of commands."),
]


@app.command("/help")
def list_skills(ack, respond):
    ack()
    lines = [f"• `{cmd} {args}`".rstrip() + f" — {desc}" for cmd, args, desc in _SKILLS]
    respond("\n".join(["Available commands:"] + lines))


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()