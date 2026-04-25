#!/usr/bin/env bash
set -euo pipefail

# Run the terminal telemetry dashboard with project defaults.
#
# Purpose:
# - Provide a single stable entrypoint for launching the telemetry TUI.
# - Keep invocation consistent across sessions.
#
# Operational boundary:
# - Executes scripts/telemetry_tui.py from repo root.
# - Uses workspace venv interpreter at .venv/bin/python.
# - Passes through user-supplied CLI args unchanged.
#
# Inputs:
# - Optional CLI args supported by telemetry_tui.py.
#
# Outputs:
# - Starts an interactive curses dashboard in the current terminal.
#
# Usage:
#   bash scripts/run_telemetry_tui.sh
#   bash scripts/run_telemetry_tui.sh --gpu-backend auto
#   bash scripts/run_telemetry_tui.sh --hide-processes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter not found at $PYTHON_BIN" >&2
  echo "Create/activate the project venv first." >&2
  exit 1
fi

exec "$PYTHON_BIN" scripts/telemetry_tui.py "$@"
