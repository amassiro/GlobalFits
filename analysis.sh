#!/usr/bin/env bash
set -euo pipefail

# Sets up an isolated venv (pylhe is not in the stdlib) and runs analysis.py,
# which reads the LHE events produced by create.sh and compares the cHWB
# SM/Lin/Quad dilepton mass distributions, direct vs. reweighted.

ROOT_DIR="$PWD"
VENV_DIR="$ROOT_DIR/.analysis_venv"

# pylhe 2.x requires Python >= 3.10. Search the same way setup.sh searches
# for a working python3 for f2py, since a bare `python3` on macOS is commonly
# an older system/Xcode Python (e.g. 3.9) even when a newer one is installed
# alongside it (e.g. via Homebrew).
echo "==> Looking for a Python >= 3.10 (required by pylhe) ..."
PYTHON3=""
for CAND in python3.13 python3.12 python3.11 python3.10 python3 \
            /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
    command -v "$CAND" >/dev/null 2>&1 || continue
    RESOLVED="$(command -v "$CAND")"
    if "$RESOLVED" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PYTHON3="$RESOLVED"
        break
    fi
done

if [ -z "$PYTHON3" ]; then
    echo "ERROR: no python3 >= 3.10 found on PATH (pylhe requires it)." >&2
    echo "       install one, e.g.: brew install python@3.12" >&2
    exit 1
fi
echo "    using $PYTHON3 ($("$PYTHON3" --version 2>&1))"

if [ -d "$VENV_DIR" ]; then
    echo "[skip] $VENV_DIR already exists -> reusing venv"
else
    "$PYTHON3" -m venv "$VENV_DIR"
    echo "==> Created venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing pylhe/numpy/matplotlib into the venv ..."
pip install --quiet --upgrade pip
# pylhe pinned: analysis.py is written against its 2.0.0 dataclass API
# (LHEFile.fromfile, event.eventinfo.weight, event.weights dict, raw
# px/py/pz/e/m/id/status on particles). Unpinned, a future pylhe release
# could change that API out from under this script.
pip install --quiet "pylhe==2.0.0" numpy matplotlib

echo "==> Running analysis.py ..."
python "$ROOT_DIR/analysis.py" --root "$ROOT_DIR" "$@"
