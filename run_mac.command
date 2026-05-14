#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================"
echo "    Transcripta - macOS Launcher        "
echo "========================================"

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 could not be found. Please install Python3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "Checking/Installing requirements..."
pip install -r requirements.txt

# Run the app
echo "Starting Transcripta..."
python transcripta.py

echo "Done."
