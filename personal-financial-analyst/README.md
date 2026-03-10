# Personal Financial Analyst: Multi-Agent Orchestration Lab

## Overview

Build a hierarchical multi-agent system using Claude Agent SDK that analyzes personal finances and provides actionable recommendations. The system demonstrates the orchestrator-workers pattern where an orchestrator agent coordinates specialized sub-agents to research alternatives, create negotiation strategies, identify tax deductions, and handle disputes.

## Problem Statement

**User Scenario**: "I want to understand my monthly expenses across all my accounts and find ways to save money."

The system should:
1. Pull transaction data from multiple bank accounts and credit cards
2. Identify all active subscriptions
3. Analyze spending patterns
4. Detect anomalies (unusual charges, duplicate subscriptions)
5. Provide actionable recommendations (cancel unused subscriptions, negotiate bills, etc.)

## Architecture: Hierarchical Multi-Agent Orchestration

### Why Multi-Agent?

**Key Design Decision**: The orchestrator agent has access to MCP servers for fetching financial data (bank accounts, credit cards) as tools. We use **sub-agents for tasks that require significant reasoning, research, or complex workflows**, not simple data fetching.

### Agent Hierarchy

```
┌─────────────────────────────────────┐
│   Orchestrator Agent                │
│   (Financial Optimization Manager)  │
│   Tools: Bank MCP, Credit Card MCP  │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┬──────────────┐
       │               │              │
┌──────▼──────┐ ┌──────▼──────┐ ┌────▼─────┐
│  Research   │ │ Negotiation │ │    Tax   │
│   Agent     │ │   Agent     │ │   Agent  │
└─────────────┘ └─────────────┘ └──────────┘
```

### Agent Responsibilities

#### 1. Orchestrator Agent (Main Agent)
**Role**: Fetches financial data, coordinates specialized sub-agents, aggregates results, generates recommendations

**Capabilities**:
- **Fetches raw data** using MCP server tools (bank transactions, credit card statements)
- Performs initial analysis (spending by category, subscription detection, anomaly detection)
- **Delegates complex tasks** to specialized sub-agents
- Reads sub-agent results from files
- Synthesizes all findings into actionable recommendations

**Tools**:
- Bank MCP server (fetch transactions)
- Credit Card MCP server (fetch statements)
- File reading/writing tools
- Sub-agent invocation

#### 2. Research Agent
**Role**: Research better alternatives for current services

**Task**: "Find cheaper alternatives to the following subscriptions: Netflix, Spotify Premium, Planet Fitness"

**Output**: `data/agent_outputs/research_results.json` with detailed alternatives, cost comparisons, pros/cons

**Why separate agent**: Requires extensive research, reasoning about tradeoffs, and domain knowledge

#### 3. Negotiation Agent
**Role**: Research negotiation strategies and draft negotiation scripts

**Task**: "Create negotiation strategy for: Comcast Internet, Car Insurance"

**Output**: `data/agent_outputs/negotiation_scripts.json` with detailed scripts, competitor prices, talking points

**Why separate agent**: Requires strategic reasoning, understanding of negotiation tactics, persuasive writing

#### 4. Tax Optimization Agent
**Role**: Analyze tax implications and identify deductions

**Task**: "Analyze 2026 transactions for tax optimization opportunities"

**Output**: `data/agent_outputs/tax_analysis.json` with deductible expenses, potential savings

**Why separate agent**: Requires specialized tax knowledge and complex regulatory reasoning

## MCP Server Architecture

The lab includes **2 FastMCP servers** that provide simulated financial data from CSV files:

### Server 1: Bank Account Server (Port 5001)
- **Tool**: `get_bank_transactions`
- **Parameters**: `username: str`, `start_date: str`, `end_date: str`
- **Returns**: JSON with bank transactions from `mock_data/bank_transactions.csv`
- **Data includes**: Date, description, category, amount, recurring flag

### Server 2: Credit Card Server (Port 5002)
- **Tool**: `get_credit_card_transactions`
- **Parameters**: `username: str`, `start_date: str`, `end_date: str`
- **Returns**: JSON with credit card transactions from `mock_data/credit_card_transactions.csv`
- **Data includes**: Date, merchant, category, amount, recurring flag

**Note**: There is no separate subscription MCP server. Subscription data is identified from bank and credit card transactions by detecting recurring charges.

## Data Flow

```
User Query: "How can I save $500/month?"
    ↓
┌─────────────────────────────────────────────────────┐
│ Orchestrator Agent                                  │
│ - Fetches bank transactions (MCP tool)             │
│ - Fetches credit card data (MCP tool)              │
│ - Performs initial analysis                        │
│ - Identifies subscriptions (recurring charges)     │
│ - Writes raw data to data/raw_data/               │
└─────────────────────────────────────────────────────┘
    ↓
Identifies opportunities requiring specialized work:
- $250 in subscriptions (can we find cheaper alternatives?)
- $200 high internet bill (can we negotiate?)
- $150 in potentially deductible expenses
    ↓
┌─────────────────────────────────────────────────────┐
│  Parallel Invocation of Specialized Sub-Agents      │
│  (Each does complex reasoning and writes results)   │
└─────────────────────────────────────────────────────┘
    ↓
Research Agent       → data/agent_outputs/research_results.json
Negotiation Agent    → data/agent_outputs/negotiation_scripts.json
Tax Agent            → data/agent_outputs/tax_analysis.json
    ↓
┌─────────────────────────────────────────────────────┐
│ Orchestrator reads all sub-agent results            │
│ - Synthesizes findings                              │
│ - Prioritizes recommendations                       │
│ - Generates action plan                             │
└─────────────────────────────────────────────────────┘
    ↓
Final Report: data/final_report.md
- Save $85/month switching to YouTube TV + Netflix plan
- Save $30/month negotiating Comcast (script provided)
- Save $450 in taxes with home office deduction
Total Potential Savings: $565/month
```

## File System Structure

```
personal-financial-analyst/
├── README.md                           # This file
├── mcp_servers/
│   ├── bank_server.py                  # Bank MCP server (port 5001)
│   ├── credit_card_server.py           # Credit card MCP server (port 5002)
│   ├── requirements.txt
│   └── mock_data/
│       ├── bank_transactions.csv       # Simulated bank data
│       └── credit_card_transactions.csv # Simulated credit card data
├── agent/
│   ├── financial_orchestrator.py       # Main orchestrator agent
│   ├── prompts/                        # Agent prompt templates
│   │   ├── orchestrator_system_prompt.txt
│   │   ├── research_agent_prompt.txt
│   │   ├── negotiation_agent_prompt.txt
│   │   └── tax_agent_prompt.txt
│   ├── requirements.txt
└── data/
    ├── raw_data/                       # Orchestrator writes fetched data here
    │   ├── bank_transactions.json
    │   └── credit_card_transactions.json
    ├── agent_outputs/                  # Sub-agents write results here
    │   ├── research_results.json
    │   ├── negotiation_scripts.json
    │   └── tax_analysis.json
    └── final_report.txt                # Orchestrator's final synthesis
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd personal-financial-analyst

# Install MCP server dependencies
cd mcp_servers
uv pip install -r requirements.txt

# Install agent dependencies
cd ../agent
uv pip install -r requirements.txt
```

### 2. Start MCP Servers

Open 2 terminal windows and run:

```bash
# Terminal 1: Bank Server
cd mcp_servers
uv run python bank_server.py
# Server starts on http://127.0.0.1:5001

# Terminal 2: Credit Card Server
cd mcp_servers
uv run python credit_card_server.py
# Server starts on http://127.0.0.1:5002
```

### 3. Run the Orchestrator Agent

**Important**: The Claude Agent SDK cannot run inside another Claude Code session. If you're developing in Claude Code, you need to:
- Run the orchestrator in a regular terminal (outside Claude Code)
- OR temporarily bypass the check: `unset CLAUDECODE` before running

```bash
cd agent
uv run python financial_orchestrator.py \
    --username john_doe \
    --start-date 2026-01-01 \
    --end-date 2026-01-31 \
    --query "How can I save money?"
```

## Implementation Tasks

### Phase 1: Basic Orchestration (Starter)
- [ ] Review MCP server code and understand mock data format
- [ ] Test MCP servers independently using curl or HTTP client
- [ ] Implement orchestrator agent skeleton with Claude Agent SDK
- [ ] Connect orchestrator to MCP servers as tools
- [ ] Fetch and save raw transaction data

### Phase 2: Sub-Agent Implementation
- [ ] Define Research Agent using `AgentDefinition`
- [ ] Define Negotiation Agent using `AgentDefinition`
- [ ] Define Tax Agent using `AgentDefinition`
- [ ] Implement file-based communication pattern
- [ ] Test each sub-agent independently

### Phase 3: Orchestration Logic
- [ ] Implement subscription detection from transactions
- [ ] Implement logic to decide which sub-agents to invoke
- [ ] Invoke sub-agents in parallel when possible
- [ ] Read and parse sub-agent outputs
- [ ] Generate final synthesized report

### Phase 4: Advanced Features (Optional)
- [ ] Add anomaly detection (duplicate charges, unusual amounts)
- [ ] Implement iterative refinement (orchestrator asks for more data)
- [ ] Add budget comparison and tracking
- [ ] Create visualization of spending patterns

## Example User Queries

### Simple Queries (Orchestrator handles alone)
1. "Show me my spending for the last month"
   - Orchestrator fetches data and analyzes directly

### Complex Queries (Requires Sub-Agents)
2. "How can I save $500 per month?"
   - Triggers all sub-agents for comprehensive analysis

3. "Analyze my subscriptions and find better deals"
   - Research Agent finds alternatives for each subscription

4. "Help me negotiate my cable bill"
   - Negotiation Agent creates detailed negotiation strategy

5. "What expenses are tax deductible?"
   - Tax Agent analyzes transactions for deductions

## Testing the System

### Quick Configuration Test

Run the automated test script to validate your setup:

```bash
./test_solution.sh
```

This script checks:
- API key configuration
- MCP server availability
- Python syntax validation
- Claude Code environment detection

### Test 1: Verify MCP Servers

```bash
# Test bank server
curl -X POST http://127.0.0.1:5001/get_bank_transactions \
  -H "Content-Type: application/json" \
  -d '{"username": "john_doe", "start_date": "2026-01-01", "end_date": "2026-01-31"}'

# Test credit card server
curl -X POST http://127.0.0.1:5002/get_credit_card_transactions \
  -H "Content-Type: application/json" \
  -d '{"username": "john_doe", "start_date": "2026-01-01", "end_date": "2026-01-31"}'
```

### Test 2: Run Orchestrator

```bash
cd agent
uv run python financial_orchestrator.py \
  --username john_doe \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --query "How can I save money?"
```

### Test 3: Verify Outputs

Check that the following files are created:
- `data/raw_data/bank_transactions.json`
- `data/raw_data/credit_card_transactions.json`
- `data/agent_outputs/research_results.json`
- `data/agent_outputs/negotiation_scripts.json`
- `data/agent_outputs/tax_analysis.json`
- `data/final_report.md`

## Model Selection Strategy

| Agent | Model | Rationale |
|-------|-------|-----------|
| **Orchestrator** | Sonnet 4.6 | Balanced reasoning for coordination |
| **Research Agent** | Haiku 4.5 | Fast, cheap for web research |
| **Negotiation Agent** | Sonnet 4.6 | Strategic reasoning needed |
| **Tax Agent** | Sonnet 4.6 | Complex regulatory reasoning |

## Key Learning Objectives

1. **Understand when to use multi-agent vs single agent with tools**
   - Data fetching → Use tools
   - Complex reasoning → Use sub-agents

2. **Implement orchestrator-workers pattern**
   - Central orchestrator coordinates work
   - Specialized workers handle specific tasks
   - File-based communication for results

3. **Use Claude Agent SDK's native subagent support**
   - Define agents with `AgentDefinition`
   - Mix different models per agent (cost optimization)
   - Monitor lifecycle with hooks

4. **Work with MCP servers as tools**
   - Connect to external data sources
   - Parse and process returned data
   - Handle errors gracefully

## Troubleshooting

### MCP Server Not Responding
- Verify server is running: `curl http://127.0.0.1:5001/health`
- Check port is not in use: `lsof -i :5001`
- Review server logs for errors

### Agent Not Finding MCP Servers
- Ensure MCP server URLs are correct in agent configuration
- Verify network connectivity
- Check that tools are properly registered

### Sub-Agents Not Being Invoked
- Review orchestrator logic for when to delegate
- Check that `AgentDefinition` is properly configured
- Verify file paths for communication are correct

## Resources

- **Claude Agent SDK**: https://github.com/anthropics/claude-agent-sdk
- **FastMCP Documentation**: https://github.com/gofastmcp/fastmcp
- **Anthropic Cookbook**: https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents
- **Building Effective Agents**: https://www.anthropic.com/research/building-effective-agents

## Success Criteria

Your implementation should:
- [ ] Successfully fetch data from both MCP servers
- [ ] Detect subscriptions from transaction data
- [ ] Invoke appropriate sub-agents based on user query
- [ ] Generate actionable recommendations with specific savings amounts
- [ ] Produce a well-formatted final report
- [ ] Handle errors gracefully
- [ ] Demonstrate cost optimization (mix of Haiku and Sonnet models)
