#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ──────────────────────────────────────────────
# 1. Check for required system tools
# ──────────────────────────────────────────────
info "Checking system dependencies..."

command -v git >/dev/null 2>&1 || error "git is not installed."
command -v python3 >/dev/null 2>&1 || error "python3 is not installed."
command -v pip >/dev/null 2>&1 && PIP="pip" || PIP="pip3"
command -v "$PIP" >/dev/null 2>&1 || error "pip is not installed."

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_PYTHON="3.10"
if [ "$(printf '%s\n' "$REQUIRED_PYTHON" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_PYTHON" ]; then
    error "Python >= $REQUIRED_PYTHON is required (found $PYTHON_VERSION)."
fi
info "Python $PYTHON_VERSION found."

# ──────────────────────────────────────────────
# 2. Create/activate virtual environment
# ──────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in .venv..."
    python3 -m venv "$VENV_DIR"
else
    info "Virtual environment already exists."
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
info "Virtual environment activated."

# ──────────────────────────────────────────────
# 3. Install Python dependencies
# ──────────────────────────────────────────────
info "Upgrading pip..."
pip install --quiet --upgrade pip

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    info "Installing project dependencies from requirements.txt..."
    pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
fi

if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    info "Installing project from pyproject.toml..."
    pip install --quiet -e "$SCRIPT_DIR"
fi

# ──────────────────────────────────────────────
# 4. Install and configure pre-commit
# ──────────────────────────────────────────────
info "Installing pre-commit..."
pip install --quiet pre-commit

info "Installing pre-commit hooks..."
pre-commit install

info "Running pre-commit on all files (first-time setup)..."
pre-commit run --all-files || warn "Some pre-commit hooks failed. Please fix the issues above."

# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────
echo ""
info "=========================================="
info "  Setup complete!"
info "=========================================="
info "Activate the environment in your shell with:"
echo ""
echo "  source .venv/bin/activate"
echo ""
