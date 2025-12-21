#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate

# If launching over SSH to show on Pi's HDMI:
export DISPLAY="${DISPLAY:-:0}"

python main.py