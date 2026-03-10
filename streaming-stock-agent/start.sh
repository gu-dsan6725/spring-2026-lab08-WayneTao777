#!/bin/bash

# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"

# Function to kill process using a specific port
kill_port() {
    local port=$1

    echo "Checking for existing process on port $port..."

    # Find PID using the port
    PID=$(lsof -ti:$port 2>/dev/null || true)

    if [ -n "$PID" ]; then
        echo "Found existing process (PID: $PID) on port $port. Killing it..."
        kill -9 $PID 2>/dev/null || true
        sleep 1
        echo "Process killed."
    else
        echo "No existing process on port $port."
    fi
}

echo "Starting Stock Query Agent API..."
echo "=================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please copy .env.example to .env and add your GROQ_API_KEY"
    echo ""
    echo "  cp .env.example .env"
    echo ""
    exit 1
fi

# Check if GROQ_API_KEY is set
source .env
if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" = "your_groq_api_key_here" ]; then
    echo "Error: GROQ_API_KEY not configured in .env file"
    echo "Please edit .env and add your Groq API key"
    echo ""
    exit 1
fi

# Install dependencies if needed
if ! uv pip list | grep -q fastapi; then
    echo "Installing dependencies..."
    uv pip install -r requirements.txt
    echo ""
fi

# Kill any existing process on port 5003
kill_port 5003

# Start the server
echo ""
echo "Starting server on http://${HOST:-127.0.0.1}:${PORT:-5003}"
echo "Press Ctrl+C to stop"
echo ""

uv run python main.py
