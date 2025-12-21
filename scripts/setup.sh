#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "==> Installing apt dependencies"
if [[ -f apt-packages.txt ]]; then
  sudo apt update
  xargs -a apt-packages.txt sudo apt install -y
else
  echo "apt-packages.txt not found; skipping apt deps"
fi

echo "==> Creating virtualenv"
python3 -m venv .venv --system-site-packages

echo "==> Installing Python dependencies"
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

if [[ -f requirements.lock.txt ]]; then
  pip install -r requirements.lock.txt
else
  pip install -r requirements.txt
fi

echo "==> Done."
echo "Next: ./scripts/run.sh"