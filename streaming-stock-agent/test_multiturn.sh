#!/bin/bash

SESSION="test_$(date +%s)"

echo "Testing multi-turn conversation with session: $SESSION"
echo ""

echo "===== Turn 1: Ask about AAPL ====="
curl -X POST http://127.0.0.1:5003/invocation \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"What is AAPL stock price?\"}" \
  --no-buffer 2>/dev/null | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
        echo "$line"
    fi
done | head -40

echo ""
echo ""
echo "===== Turn 2: Follow-up question (should understand AAPL context) ====="
curl -X POST http://127.0.0.1:5003/invocation \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"How about yesterday's close?\"}" \
  --no-buffer 2>/dev/null | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
        echo "$line"
    fi
done | head -40

echo ""
echo ""
echo "===== Turn 3: Switch context to different stock ====="
curl -X POST http://127.0.0.1:5003/invocation \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"What about Tesla?\"}" \
  --no-buffer 2>/dev/null | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
        echo "$line"
    fi
done | head -40

echo ""
echo ""
echo "===== Session Info ====="
curl http://127.0.0.1:5003/session/$SESSION/info 2>/dev/null | jq .
