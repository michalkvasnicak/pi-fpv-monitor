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

echo "==> Setting up pyenv"
# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
  echo "pyenv not found. Installing pyenv..."
  curl https://pyenv.run | bash
  
  # Add pyenv to PATH for current session
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
  
  # Initialize pyenv
  eval "$(pyenv init -)"
else
  echo "pyenv found. Initializing..."
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"
fi

# Install build dependencies for Python (if not already installed)
echo "==> Ensuring Python build dependencies are installed"
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev || true

# Install Python 3.12.8 if not already installed
PYTHON_VERSION="3.12.8"
if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
  echo "Python ${PYTHON_VERSION} is already installed"
else
  echo "==> Installing Python ${PYTHON_VERSION} with pyenv"
  pyenv install ${PYTHON_VERSION}
fi

# Set Python version for this project
echo "==> Setting Python ${PYTHON_VERSION} as local version"
pyenv local ${PYTHON_VERSION}

echo "==> Creating virtualenv"
python -m venv .venv --system-site-packages

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