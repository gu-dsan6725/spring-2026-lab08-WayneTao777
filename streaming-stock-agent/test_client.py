"""Simple test client for the Stock Query Agent API."""

import requests
import json
import sys


BASE_URL = "http://127.0.0.1:5003"


def test_ping():
    """Test the health check endpoint."""
    print("Testing /ping endpoint...")
    response = requests.get(f"{BASE_URL}/ping")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_invocation(
    session_id: str,
    message: str
):
    """Test the invocation endpoint with streaming."""
    print(f"Session: {session_id}")
    print(f"Query: {message}")
    print("-" * 60)

    response = requests.post(
        f"{BASE_URL}/invocation",
        json={
            "session_id": session_id,
            "message": message
        },
        stream=True
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')

            if line_str.startswith('data: '):
                data_str = line_str[6:]

                try:
                    data = json.loads(data_str)

                    if data['type'] == 'text':
                        print(data['content'], end='', flush=True)

                    elif data['type'] == 'tool_call':
                        print(f"\n[Calling tool: {data['name']} with args {data['args']}]")

                    elif data['type'] == 'tool_calls_start':
                        print("\n[Tool calls starting...]")

                    elif data['type'] == 'tool_calls_end':
                        print("[Tool calls completed]\n")

                    elif data['type'] == 'done':
                        print("\n[Done]")

                    elif data['type'] == 'error':
                        print(f"\n[Error: {data['message']}]")

                except json.JSONDecodeError:
                    print(f"\n[Could not parse: {data_str}]")

    print("\n" + "=" * 60 + "\n")


def test_multi_turn():
    """Test multi-turn conversation."""
    session_id = "test_session_123"

    print("=" * 60)
    print("MULTI-TURN CONVERSATION TEST")
    print("=" * 60 + "\n")

    test_invocation(session_id, "What is the current price of Apple stock?")

    test_invocation(session_id, "How has it changed in the last 30 days?")

    test_invocation(session_id, "What about Tesla?")


def main():
    """Run test scenarios."""
    if len(sys.argv) > 1:
        test_invocation("cli_session", " ".join(sys.argv[1:]))
    else:
        test_ping()
        test_multi_turn()


if __name__ == "__main__":
    main()
