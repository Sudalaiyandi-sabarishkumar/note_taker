# mom-phase1 → Slack

Wraps the existing `mom_phase1` pipeline in a Slack bot using **Socket Mode**
(no public URL, no exposing your local Ollama server to the internet — the
bot process talks to `localhost:11434` exactly like the CLI does).

## Architecture

```
Slack workspace
   │  slash command / modal submission (via WebSocket, Socket Mode)
   ▼
slack_app.py  (this machine — same one running Ollama)
   │  imports mom_phase1.cli.run_phase1 / extract.answer_question directly
   ▼
Ollama (localhost:11434) → mom-phase1 model (Modelfile, qwen2.5:7b-instruct base)
   │
   ▼
knowledge/*.md   (same files the CLI writes — bot just reuses them)
```

## 1. Build the model (if you haven't already)

```bash
cd note_taker
./build_model.sh
```

## 2. Create the Slack app

1. Go to https://api.slack.com/apps → **Create New App** → **From scratch**.
2. Name it (e.g. "Mom Phase1") and pick your workspace.
3. **Socket Mode** (left sidebar) → toggle **Enable Socket Mode** on.
   This generates an **App-Level Token** — create one with the `connections:write`
   scope. Save it: this is `SLACK_APP_TOKEN` (starts with `xapp-`).
4. **OAuth & Permissions** → **Scopes** → **Bot Token Scopes**, add:
   - `commands` (to receive slash commands)
   - `chat:write` (to post messages)
   - `files:read` (to download uploaded transcripts)
5. **Slash Commands** (left sidebar) → **Create New Command**, once for each:
   | Command | Short description |
   |---|---|
   | `/mom-extract` | Upload a transcript and run Phase 1 |
   | `/mom-features` | List feature docs |
   | `/mom-show` | Show one feature doc |
   | `/mom-ask` | Ask a question against the knowledge base |
   | `/mom-merge` | Combine two or more feature docs into the first |
   | `/mom-model` | Show which Ollama model is in use |
   | `/mom-docs-dir` | Show, or change, the active output directory |
   | `/mom-skills` | List all of the above, in Slack |

   With Socket Mode there's no "Request URL" to fill in — leave it blank /
   any placeholder; Slack routes it over the socket instead. Each command
   needs its own entry here — Slack won't route a command to your app
   unless it's been registered this way, even if the code already handles
   it.
6. **Install App** (top of OAuth & Permissions) to your workspace. This gives
   you a **Bot User OAuth Token** (`xoxb-...`) — this is `SLACK_BOT_TOKEN`.
   If you add more slash commands (like the ones above) *after* installing,
   reinstall the app once from the same page so Slack picks them up.
7. Invite the bot to the channel(s) you want it in: `/invite @Feature Extractor`
   (use your app's actual name).

## 3. Install & run

```bash
cd note_taker
pip install -e ".[cli]"
pip install -r slack_integration/requirements.txt
```

Put your tokens in a `.env` file in `note_taker/` (already `.gitignore`d) so
you don't have to `export` them every session:

```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

`slack_app.py` loads this automatically via `python-dotenv`. Optional
overrides (same ones the CLI supports) can go in the same file:
```
MOM_MODEL=mom-phase1
MOM_DOCS_DIR=knowledge
```

Then just run:
```bash
python mom_phase1/slack_app.py
```

(If you're on the pipx-managed venv instead, use that venv's python — see
the note at the end of this section.)

You should see the Socket Mode connection open with no errors. Ollama must
already be running (`brew services start ollama`, or `ollama serve`).

**pipx users:** inject the extra deps into the same venv pipx built for
`mom-phase1`, then run with that venv's interpreter:
```bash
pipx inject mom-phase1 slack_bolt requests python-dotenv
"$(pipx environment --value PIPX_LOCAL_VENVS)/mom-phase1/bin/python" mom_phase1/slack_app.py
```

## 4. Use it

- `/mom-extract` → a modal opens with a file-upload field → attach a
  `.txt`/`.vtt` transcript → **Extract**. The bot posts an ack immediately,
  runs Phase 1 in a background thread (this can take a minute or two per
  transcript since it's several sequential LLM calls), then posts the same
  summary the CLI prints (feature docs touched, possible fragmentation, any
  `[NEEDS REVIEW]` / `[UNVERIFIED CITATION]` flags).
- `/mom-features` → lists feature docs.
- `/mom-show notifications` → posts that doc as a code block.
- `/mom-ask when did we switch from email to SMS?` → answers from the docs.
- `/mom-merge "Certificate Format" "Certificate Content"` → folds the second
  doc into the first, keeping every fact and citation from both.
- `/mom-model` → shows which Ollama model the bot is running against.
- `/mom-docs-dir` (no argument) → shows the docs directory the bot is
  currently reading/writing.
- `/mom-docs-dir client-acme` → switches the bot to read/write
  `client-acme/` instead, for every command, from then on. The directory is
  created if it doesn't exist yet. This change is **in-memory only**: it
  applies to the running bot process but reverts to `MOM_DOCS_DIR` (or
  `knowledge`) from `.env` the next time the bot restarts. To make a
  directory change permanent, edit `MOM_DOCS_DIR` in `note_taker/.env`
  instead and restart the bot.
- `/mom-skills` → lists all of the above, inside Slack.

## 5. Run it as a persistent service

Socket Mode needs a long-lived process. Simplest options:

**tmux / screen** (quick, single machine):
```bash
tmux new -s mom-slack 'python mom_phase1/slack_app.py'
```

**systemd** (Linux, survives reboot):
```ini
# /etc/systemd/system/mom-slack.service
[Unit]
Description=mom-phase1 Slack bot
After=network.target

[Service]
WorkingDirectory=/path/to/note_taker
Environment=SLACK_BOT_TOKEN=xoxb-...
Environment=SLACK_APP_TOKEN=xapp-...
ExecStart=/usr/bin/python3 mom_phase1/slack_app.py
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now mom-slack
```

## Notes / things to watch

- **Slack's 3-second ack window**: every handler calls `ack()` immediately
  and does the actual model work in a background thread — don't move the
  `run_phase1(...)` call in front of `ack()`.
- **Multi-user knowledge dir**: everyone hitting `/mom-extract` writes into
  the same `knowledge/` folder, same as multiple people running the CLI
  locally. `/mom-docs-dir <path>` lets anyone switch that folder for the
  whole bot process without a restart, but it's still one shared setting —
  it doesn't scope by user or channel, so two people switching it at once
  will step on each other. For real per-project/per-channel doc sets, run
  one bot process + channel per client project (with its own `MOM_DOCS_DIR`
  in a separate `.env`) rather than trying to make the single bot
  multi-tenant.
- **Long transcripts / slow model**: `num_predict` is capped at 1536 in the
  Modelfile and extraction chunks the transcript, so a long call can mean
  several sequential requests to Ollama. There's no Slack-side timeout to
  worry about since results are pushed via `chat.postMessage`, not returned
  in the original response.
- **Output truncation**: Slack `section` blocks cap out around 3000
  characters; the code truncates at 2900 as a safety margin. For very large
  outputs, consider uploading the changed `knowledge/*.md` file via
  `files_upload_v2` instead of pasting it inline.