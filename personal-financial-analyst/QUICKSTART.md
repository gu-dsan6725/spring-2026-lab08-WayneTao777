# Quick Start Guide

Get the Personal Financial Analyst multi-agent system running in 5 minutes.

## Prerequisites

- Python 3.10+
- `uv` package manager installed
- Anthropic API key (get one at https://console.anthropic.com/)

## Step 1: Install Dependencies

```bash
cd personal-financial-analyst

# Install MCP server dependencies
cd mcp_servers
uv pip install -r requirements.txt

# Install agent dependencies
cd ../agent
uv pip install -r requirements.txt
```

## Step 2: Set Up API Key

```bash
cd ..
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Step 3: Start MCP Servers

**Option A: Quick Start (Background)**
```bash
cd mcp_servers
./start_servers.sh
```

**Option B: Manual Start (Separate Terminals)**

Open 2 terminal windows:

**Terminal 1 - Bank Server:**
```bash
cd mcp_servers
uv run python bank_server.py
```

**Terminal 2 - Credit Card Server:**
```bash
cd mcp_servers
uv run python credit_card_server.py
```

Wait for the messages:
```
Starting Bank Account MCP Server on port 5001
Starting Credit Card MCP Server on port 5002
```

## Step 4: Test MCP Servers

**Terminal 3:**
```bash
cd mcp_servers
uv run python test_servers.py
```

You should see:
```
✓ Bank server working
✓ Credit card server working
✓ All servers are operational!
```

## Step 5: Run the Orchestrator Agent

```bash
cd agent
uv run python financial_orchestrator.py \
    --username john_doe \
    --start-date 2026-01-01 \
    --end-date 2026-01-31 \
    --query "How can I save $500 per month?"
```

## What You'll Build

The starter code provides:
- ✓ MCP servers with simulated data
- ✓ Basic orchestrator structure
- ✓ Helper functions for data loading

You'll implement:
- [ ] MCP server connections in the orchestrator
- [ ] Sub-agent definitions (Research, Negotiation, Tax)
- [ ] Orchestration logic (when to invoke which agents)
- [ ] Result synthesis and report generation

## Available Test Users

The mock data includes two users:

1. **john_doe**: Average spender with several subscriptions
   - Income: $5,000/month
   - Major expenses: Rent, subscriptions, dining
   - Has suspicious $499 charge (ACME Corp)

2. **jane_smith**: Higher earner with premium subscriptions
   - Income: $6,500/month
   - Major expenses: Mortgage, premium gym, shopping
   - More subscription services

## Sample Queries to Try

1. **Savings Analysis:**
   ```bash
   --query "How can I save $500 per month?"
   ```

2. **Subscription Audit:**
   ```bash
   --query "Analyze all my subscriptions and find better deals"
   ```

3. **Bill Negotiation:**
   ```bash
   --query "Help me negotiate my Comcast bill"
   ```

4. **Tax Optimization:**
   ```bash
   --query "What expenses from this month are tax deductible?"
   ```

## Expected Output Files

After running the orchestrator, check these directories:

```
data/
├── raw_data/
│   ├── bank_transactions.json      # Fetched from MCP
│   └── credit_card_transactions.json
├── agent_outputs/
│   ├── research_results.json       # Sub-agent output
│   ├── negotiation_scripts.json    # Sub-agent output
│   └── tax_analysis.json          # Sub-agent output
└── final_report.txt                # Orchestrator synthesis
```

## Troubleshooting

### MCP Server Won't Start
```bash
# Check if port is already in use
lsof -i :5001
lsof -i :5002

# Kill process if needed
kill -9 <PID>

# Or if you used start_servers.sh, check logs
cat mcp_servers/bank_server.log
cat mcp_servers/credit_card_server.log
```

### Stop Servers
```bash
# If started with start_servers.sh
ps aux | grep "python.*_server.py"
kill <PID>

# Or
pkill -f "python.*_server.py"
```

### Agent Can't Connect to MCP
- Verify servers are running: `uv run python test_servers.py`
- Check URLs in agent configuration match server ports

### API Key Error
- Ensure `.env` file exists with valid `ANTHROPIC_API_KEY`
- Verify key is loaded: `echo $ANTHROPIC_API_KEY`

## Next Steps

1. Review the README.md for full architecture details
2. Study the MCP server implementations
3. Implement the TODOs in `financial_orchestrator.py`
4. Test with different queries and users
5. Add more sub-agents or features

## Getting Help

- Claude Agent SDK: https://github.com/anthropics/claude-agent-sdk
- FastMCP: https://github.com/gofastmcp/fastmcp
- Anthropic Cookbook: https://github.com/anthropics/anthropic-cookbook
