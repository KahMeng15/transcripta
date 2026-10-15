#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================"
echo "    Transcripta - macOS Launcher        "
echo "========================================"

# Find the best Python 3 interpreter (prefer Homebrew if present, otherwise system/conda)
if [ -x "/opt/homebrew/bin/python3" ]; then
    SYSTEM_PYTHON="/opt/homebrew/bin/python3"
elif command -v python3 &> /dev/null; then
    SYSTEM_PYTHON="$(command -v python3)"
else
    echo "Python3 could not be found. Please install Python3."
    exit 1
fi

# Create virtual environment if it doesn't exist or is broken
if [ ! -f ".venv/bin/python" ]; then
    echo "Creating/repairing virtual environment..."
    rm -rf .venv
    "$SYSTEM_PYTHON" -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements using the virtual environment's pip
echo "Checking/Installing requirements..."
"$DIR/.venv/bin/python" -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[!] Failed to install requirements."
    exit 1
fi

# Run the app directly with the venv python
echo "Starting Transcripta..."
"$DIR/.venv/bin/python" transcripta.py

echo "Done."

