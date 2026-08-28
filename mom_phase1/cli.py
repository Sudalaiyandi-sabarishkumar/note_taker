"""Interactive slash-command CLI for Phase 1.

    $ mom-phase1
    mom> /requirements examples/delivery_call1.txt
    mom> /features
    mom> /exit

Slash commands
--------------
  /requirements <file>   Run Phase 1 on a transcript (.txt or .vtt):
                         extract cited statements -> merge into knowledge/ docs.
  /extract <file>        Alias for /requirements.
  /features              List the feature docs discovered under knowledge/.
  /show <feature>        Print a feature doc.
  /model                 Show which Ollama model is in use.
  /skills                List all commands.
  /help                  Show this help.
  /exit                  Quit.

One-shot form (no REPL):  mom-phase1 <file>
"""

import os
import re
import sys

from . import __version__
from .extract import (canonicalize_feature_names, extract_statements,
                      reconcile_statement, synthesize_user_story)
from .knowledge_docs import DOCS_DIR, discover_features, merge_statements
from .ollama_client import DEFAULT_MODEL, OllamaError

_REQ_RE = re.compile(r"^/(?:requirements|extract)\s+(.+)$", re.IGNORECASE)
_SHOW_RE = re.compile(r"^/show\s+(.+)$", re.IGNORECASE)

# Single source of truth for the slash commands: (command, args, description).
# Drives /skills, /help, and tab-completion.
SKILLS = [
    ("/requirements", "<file>", "Run Phase 1 on a transcript (.txt or .vtt): extract "
                                "cited statements and merge them into the per-feature "
                                "knowledge docs."),
    ("/extract", "<file>", "Alias for /requirements."),
    ("/features", "", "List the feature docs discovered under the knowledge dir."),
    ("/show", "<feature>", "Print one feature doc (partial name match)."),
    ("/model", "", "Show which Ollama model is in use."),
    ("/skills", "", "List these commands."),
    ("/help", "", "Show usage help."),
    ("/exit", "", "Quit."),
]

BANNER = (
    f"mom-phase1 {__version__} — client-call -> requirements, Phase 1\n"
    f"model: {DEFAULT_MODEL}   docs: {DOCS_DIR}/\n"
    f'Type /skills for commands, /exit to quit.'
)


def _print_skills() -> None:
    print("Available skills:")
    width = max(len(f"{c} {a}".strip()) for c, a, _ in SKILLS)
    for cmd, args, desc in SKILLS:
        left = f"{cmd} {args}".strip()
        print(f"  {left:<{width}}   {desc}")


def run_phase1(path: str) -> None:
    from .transcript import load_transcript

    text, error = load_transcript(path)
    if error:
        print(error)
        return

    known = list(discover_features().keys())
    try:
        statements = extract_statements(text, known)
        if not statements:
            print("No concrete, citable statements were found in this transcript.")
            return

        source_name = os.path.splitext(os.path.basename(path))[0]
        print("Reconciling against existing facts...")
        summary = merge_statements(statements, source_name,
                                   reconcile=reconcile_statement,
                                   story_fn=synthesize_user_story,
                                   canon_fn=canonicalize_feature_names)
    except OllamaError as exc:
        print(f"\n{exc}")
        return

    print("\nDone. Updated feature docs:")
    print("\n".join(summary))
    print(
        f"\n{len(summary)} feature doc(s) touched. Changes were applied in place; "
        f"review anything tagged [NEEDS REVIEW] / [UNVERIFIED CITATION] in {DOCS_DIR}/."
    )


def _list_features() -> None:
    feats = discover_features()
    if not feats:
        print(f"No feature docs yet under {DOCS_DIR}/.")
        return
    for name, path in sorted(feats.items()):
        print(f"  {name}  ->  {path}")


def _show_feature(query: str) -> None:
    feats = discover_features()
    q = query.strip().lower()
    for name, path in feats.items():
        if q in name.lower():
            with open(path, encoding="utf-8") as f:
                print(f.read())
            return
    print(f'No feature doc matching "{query}".')


def _handle(line: str) -> bool:
    """Return False to exit the REPL."""
    line = line.strip()
    if not line:
        return True
    if line in ("/exit", "/quit", "/q"):
        return False
    if line in ("/help", "/?", "help"):
        print(__doc__)
        return True
    if line in ("/skills", "/commands", "skills"):
        _print_skills()
        return True
    if line in ("/features", "/list"):
        _list_features()
        return True
    if line == "/model":
        print(f"model: {DEFAULT_MODEL}  (override with MOM_MODEL=...)")
        return True

    m = _REQ_RE.match(line)
    if m:
        run_phase1(m.group(1).strip().strip('"').strip("'"))
        return True
    m = _SHOW_RE.match(line)
    if m:
        _show_feature(m.group(1))
        return True

    if line.startswith("/"):
        print(f"Unknown command: {line}. Try /help.")
    else:
        print("This is a slash-command tool. Try: /requirements <file>  or  /help.")
    return True


def _repl() -> None:
    print(BANNER)
    prompt = "mom> "
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter

        session = PromptSession(
            completer=WordCompleter([c for c, _, _ in SKILLS], sentence=True)
        )
        read = lambda: session.prompt(prompt)
    except ImportError:
        read = lambda: input(prompt)

    while True:
        try:
            line = read()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not _handle(line):
            break


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv and argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    if argv:
        # one-shot: treat args as a transcript path
        run_phase1(" ".join(argv))
        return 0
    _repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
