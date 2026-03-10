#!/bin/bash

# Start both MCP servers in the background
# Note: This is for testing purposes. In production, you'd want proper process management.

set -e

# Function to kill process on a port
kill_port() {
    local port=$1
    local name=$2

    echo "Checking for existing $name server on port $port..."

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

echo "=========================================="
echo "MCP Servers Startup Script"
echo "=========================================="
echo ""

# Kill existing servers if running
kill_port 5001 "Bank Account"
kill_port 5002 "Credit Card"

echo ""
echo "Starting Bank Account MCP Server on port 5001..."
uv run python bank_server.py > bank_server.log 2>&1 &
BANK_PID=$!
echo "Bank server started (PID: $BANK_PID)"

echo ""
echo "Starting Credit Card MCP Server on port 5002..."
uv run python credit_card_server.py > credit_card_server.log 2>&1 &
CC_PID=$!
echo "Credit card server started (PID: $CC_PID)"

echo ""
echo "=========================================="
echo "Both servers are starting up..."
echo "=========================================="
echo "Logs:"
echo "  Bank: bank_server.log"
echo "  Credit Card: credit_card_server.log"
echo ""
echo "To stop servers:"
echo "  kill $BANK_PID $CC_PID"
echo ""
echo "Wait a few seconds, then test with:"
echo "  uv run python test_mcp_protocol.py"
echo "=========================================="
