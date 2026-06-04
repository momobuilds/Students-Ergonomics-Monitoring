#!/usr/bin/env bash
#
# run.sh - Launch the Streamlit GUI.
#
# Runs from the Scripts/GUI folder so Streamlit picks up
# Scripts/GUI/.streamlit/config.toml (the green theme). All project paths are
# resolved inside pipeline.py, so the working directory does not affect imports.
#
# Usage:
#   ./Scripts/GUI/run.sh
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Use the project venv if present (project root is two levels up: Scripts/GUI -> root).
# Any extra arguments are forwarded to streamlit (e.g. --server.port 8600).
if [ -x "$PROJECT_ROOT/.venv/bin/streamlit" ]; then
    exec "$PROJECT_ROOT/.venv/bin/streamlit" run app.py "$@"
else
    exec streamlit run app.py "$@"
fi
