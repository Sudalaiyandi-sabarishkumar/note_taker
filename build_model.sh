#!/bin/bash
# Build the Phase 1 extraction model from ./Modelfile.
# Pulls the qwen2.5:7b-instruct-q4_K_M base first if it isn't present.
set -e

if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama not found. Install it:  brew install ollama && brew services start ollama"
    exit 1
fi

if ! ollama list | grep -q "qwen2.5:7b-instruct-q4_K_M"; then
    echo "Pulling base model qwen2.5:7b-instruct-q4_K_M ..."
    ollama pull qwen2.5:7b-instruct-q4_K_M
fi

echo "Creating model 'mom-phase1' from Modelfile ..."
ollama create mom-phase1 -f "$(dirname "$0")/Modelfile"
echo "Done.  Run:  mom-phase1"
