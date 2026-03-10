# Streaming Stock Query Agent

A FastAPI-based conversational agent that provides streaming responses for stock market queries using Yahoo Finance API and Groq LLM.

## Features

- Streaming text responses for real-time feedback
- Multi-turn conversation with session management
- In-memory conversation history (circular buffer of 100 messages)
- Yahoo Finance integration for stock data
- Groq API for fast LLM inference

## Prerequisites

- Python 3.10+
- Groq API key ([Get one here](https://console.groq.com/))

## Setup

1. Install dependencies:
```bash
cd streaming-stock-agent
uv pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

3. Start the server:

**Option A: Using start script (recommended)**
```bash
./start.sh
```
The start script automatically:
- Checks for valid configuration
- Kills any existing process on port 5003
- Starts the server cleanly

**Option B: Running directly with Python**
```bash
uv run python main.py
```
If running directly, ensure no other process is using port 5003. If you get an "address already in use" error, kill the existing process first:
```bash
lsof -ti:5003 | xargs kill -9
```

The server will start on `http://127.0.0.1:5003`

## API Endpoints

### Health Check
```bash
curl http://127.0.0.1:5003/ping
```

Response:
```json
{"status": "ok"}
```

### Query Agent (Streaming)
```bash
curl -X POST http://127.0.0.1:5003/invocation \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "message": "What is the current price of AAPL?"
  }'
```

Response: Server-Sent Events (SSE) stream with incremental text chunks

## Example Usage

### Python Client
```python
import requests

session_id = "user123"
url = "http://127.0.0.1:5003/invocation"

response = requests.post(
    url,
    json={
        "session_id": session_id,
        "message": "What's the price of Tesla stock?"
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### Multi-turn Conversation
```bash
# First question
curl -X POST http://127.0.0.1:5003/invocation \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "message": "What is AAPL stock price?"}'

# Follow-up question (uses conversation history)
curl -X POST http://127.0.0.1:5003/invocation \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "message": "How does that compare to yesterday?"}'
```

## Testing

### Test Client
Use the included test client to verify the system:

```bash
# Run full test suite (health check + multi-turn conversation)
uv run python test_client.py

# Single query test
uv run python test_client.py "What is the price of NVDA?"
```

The test client demonstrates:
- Health check (`/ping` endpoint)
- Single query with streaming response
- Multi-turn conversation with context retention
- Proper SSE parsing and display

### Multi-turn Test Script
For testing session management with a unique session ID:

```bash
./test_multiturn.sh
```

This script runs a 3-turn conversation:
1. Ask about AAPL stock price
2. Follow-up question (tests context understanding)
3. Switch to different stock (Tesla)
4. Display session info (message count, timestamps)

## Architecture

- **FastAPI**: Web framework for API endpoints
- **Strands Agent**: Conversational agent with tool calling
- **Yahoo Finance**: Real-time stock data via yfinance library
- **Groq**: Fast LLM inference (qwen/qwen3-32b)
- **In-Memory Store**: Circular buffer for conversation history per session

## Configuration

Environment variables in `.env`:
- `GROQ_API_KEY`: Your Groq API key (required)
- `HOST`: Server host (default: 127.0.0.1)
- `PORT`: Server port (default: 5003)
- `MAX_HISTORY_SIZE`: Max messages per session (default: 100)

## Model Selection

The system uses **Groq's Qwen3-32b model** (`qwen/qwen3-32b`) which provides:
- Reliable streaming function calling (more stable than llama models)
- Fast inference
- Good context understanding for multi-turn conversations

**Note**: Earlier testing showed llama-3.1-8b-instant and llama-3.3-70b-versatile had intermittent function calling errors (~10-20% failure rate). Qwen3-32b resolves these issues.
