#!/usr/bin/env bash
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
echo "  ContextFuse UI  ->  http://127.0.0.1:8000"
echo "  API docs        ->  http://127.0.0.1:8000/docs"
uvicorn contextfuse.api:app --port 8000 --host 127.0.0.1
