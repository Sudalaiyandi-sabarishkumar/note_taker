# Developer setup

Just pulled this repo? Here's how to get it running.

## What you need first

- Python 3.9 or newer
- Ollama, to run the model locally. On a Mac: `brew install ollama && brew services start ollama`
  (or grab it from https://ollama.com). Everything runs on your machine — no API key, nothing sent anywhere.

## 1. Get the code

```bash
git clone https://github.com/Sudalaiyandi-sabarishkumar/note_taker.git
cd note_taker/mom
git checkout main
```

## 2. Install it

The package doesn't actually need any third-party libraries, so you don't
have to fuss with a venv if you don't want to. Two easy ways to install it:

```bash
pipx install -e ".[cli]"
```
This is probably the easiest — pipx quietly sets up its own little
environment for you and puts `mom-phase1` on your PATH. Nothing to activate,
just run `mom-phase1` from anywhere afterwards.

If you'd rather do it the normal way with your own venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[cli]"
```

Just remember to `source .venv/bin/activate` again every time you open a new
terminal. (The `[cli]` part just adds prompt_toolkit for a nicer REPL —
leave it off and plain `pip install -e .` still works fine.)

## 3. Build the model

This project uses its own Ollama model, `mom-phase1`, built on top of
`qwen2.5:7b-instruct-q4_K_M`. There's a script for it:

```bash
./build_model.sh
```

First time you run this it'll download the base model (about 4.7 GB), then
build `mom-phase1` from it. Check it landed:

```bash
ollama list
```

You should see both `mom-phase1` and `qwen2.5:7b-instruct-q4_K_M` in the list.

A quick word on hardware: 4.7 GB isn't huge, and it'll run on an 8 GB Mac —
just expect it to be slow, since a chunk of it spills over to CPU. On 16 GB
or more it's noticeably smoother. You don't need a GPU, but it helps.


## 4. Running it on your own stuff

```bash
mom-phase1 path/to/call1.txt
```

That's the one-shot way — point it at a `.txt` or `.vtt` transcript and
it does its thing. Or drop into the interactive shell instead:

```bash
mom-phase1
mom> /requirements path/to/call1.txt
mom> /features
mom> /show notifications
mom> /exit
```

By default everything gets written to a `knowledge/` folder (also
gitignored — call transcripts and anything derived from them shouldn't
end up in git). If you run it again with a follow-up call, it merges into
what's already there instead of starting over.

## 5. Settings you can tweak

None of these are required, they're just there if you need them:

- `OLLAMA_URL` — talk to a different Ollama server instead of the one on
  your own machine. Defaults to `http://localhost:11434/api/chat`.
- `MOM_MODEL` — use a different model instead of `mom-phase1`, e.g. the
  plain base model, or something like `qwen3:8b` if you're experimenting.
- `MOM_DOCS_DIR` — change where the output docs go. Defaults to `knowledge`.
- `MOM_COVERAGE` — set to `0` or `1` to force the extra "did we miss
  anything" pass off or on. Normally it decides on its own.
- `MOM_SENTENCE_PASS` — set to `0` if you want to turn off the pass that
  tries to catch requirements buried in dense, run-on sentences.
- `MOM_GAPS` — `0` turns off the gap analysis (the stuff that raises
  `[GAP]` questions).
- `MOM_CONTRA` — `0` turns off the check that flags facts contradicting
  each other across docs.
- `MOM_NO_SPLIT` — set this to skip the step that splits a messy doc back
  into separate topics.
- `MOM_DEBUG` — give it a file path and it'll dump the raw prompts and
  model replies there, handy when something looks off.
