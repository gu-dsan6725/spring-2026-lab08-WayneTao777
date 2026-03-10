# Student Exercise: Multi-Agent Financial Orchestrator

## Objective
Implement a hierarchical multi-agent system using Claude Agent SDK that fetches financial data from MCP servers and coordinates specialized sub-agents to provide financial optimization recommendations.

This exercise reinforces understanding of:
- Orchestrator-workers pattern with multiple specialized agents
- Claude Agent SDK's native subagent support
- MCP protocol integration for external data sources
- Parallel agent execution for efficiency
- File-based communication between agents
- Cost optimization through model selection

## Background
You have two MCP servers running that provide simulated financial data:
- **Bank Account Server** (port 5001): Provides bank transaction data
- **Credit Card Server** (port 5002): Provides credit card transaction data

Your task is to implement the orchestrator agent that:
1. Fetches data from both MCP servers
2. Analyzes transactions to detect subscriptions and spending patterns
3. Delegates specialized tasks to three sub-agents (research, negotiation, tax)
4. Synthesizes results into a final comprehensive report

## Prerequisites

**Important**: Make sure both MCP servers are running before starting:
```bash
# Terminal 1: Bank Server
cd mcp_servers
uv run python bank_server.py

# Terminal 2: Credit Card Server
cd mcp_servers
uv run python credit_card_server.py
```

Verify they're working:
```bash
curl http://127.0.0.1:5001/health
curl http://127.0.0.1:5002/health
```

## Your Task

Complete the implementation in [agent/financial_orchestrator.py](agent/financial_orchestrator.py) by filling in all sections marked with `TODO`.

### Part 1: Implement Subscription Detection (15 min)

**Function**: `_detect_subscriptions()`

**Location**: Lines 48-73 in [financial_orchestrator.py](agent/financial_orchestrator.py)

**Requirements**:
- Filter transactions with `recurring=True`
- Extract subscription name, amount, and frequency
- Return list of subscription dictionaries

**Expected Output Format**:
```python
[
    {
        "service": "Netflix",
        "amount": 15.99,
        "frequency": "monthly"
    },
    ...
]
```

**Hints**:
- Look for transactions with `recurring` field set to `True`
- Subscriptions are typically negative amounts (money going out)
- Both bank and credit card transactions can contain subscriptions

### Part 2: Configure MCP Server Connections (20 min)

**Function**: `_fetch_financial_data()`

**Location**: Lines 76-121 in [financial_orchestrator.py](agent/financial_orchestrator.py)

**Requirements**:
- Configure MCP server connections (ports 5001 and 5002)
- Call `get_bank_transactions` tool with username, start_date, end_date
- Call `get_credit_card_transactions` tool with username, start_date, end_date
- Save raw data to files using `_save_json()`

**MCP Configuration Format**:
```python
mcp_servers = {
    "Bank Account Server": {  # MUST match FastMCP server name exactly
        "type": "http",
        "url": "http://127.0.0.1:5001/mcp"
    },
    "Credit Card Server": {
        "type": "http",
        "url": "http://127.0.0.1:5002/mcp"
    }
}
```

**Hints**:
- Server names in config MUST match the FastMCP constructor names exactly
- Type is `"http"` for FastMCP's HTTP transport
- URL path includes `/mcp` endpoint
- See [architecture.md](agent/architecture.md) Section "Initialization & Configuration" for full example

### Part 3: Define Sub-Agents (25 min)

**Function**: `_run_orchestrator()` - Step 3

**Location**: Lines 172-189 in [financial_orchestrator.py](agent/financial_orchestrator.py)

**Requirements**:
Define three specialized agents using `AgentDefinition`:

1. **Research Agent**:
   - Description: "Research cheaper alternatives for subscriptions and services"
   - Prompt: Load from `prompts/research_agent_prompt.txt`
   - Tools: `["write"]`
   - Model: `"haiku"` (cost optimization)

2. **Negotiation Agent**:
   - Description: "Create negotiation strategies and scripts for bills and services"
   - Prompt: Load from `prompts/negotiation_agent_prompt.txt`
   - Tools: `["write"]`
   - Model: `"haiku"`

3. **Tax Agent**:
   - Description: "Identify tax-deductible expenses and optimization opportunities"
   - Prompt: Load from `prompts/tax_agent_prompt.txt`
   - Tools: `["write"]`
   - Model: `"haiku"`

**Helper Function**:
```python
def _load_prompt(filename: str) -> str:
    """Load prompt from prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / filename
    return prompt_path.read_text()
```

**Hints**:
- Create the `_load_prompt()` helper function at the top of the file
- All sub-agents use Haiku for cost optimization (75% cheaper than Sonnet)
- Orchestrator uses Sonnet for complex reasoning
- See [architecture.md](agent/architecture.md) Section "Sub-Agent Definition" for full example

### Part 4: Configure Orchestrator Agent (20 min)

**Function**: `_run_orchestrator()` - Step 4

**Location**: Lines 191-201 in [financial_orchestrator.py](agent/financial_orchestrator.py)

**Requirements**:
Create `ClaudeAgentOptions` with:
- `model="sonnet"` (orchestrator needs complex reasoning)
- `system_prompt` loaded from `prompts/orchestrator_system_prompt.txt`
- `mcp_servers` configuration from Part 2
- `agents` dictionary with the three sub-agents from Part 3
- `can_use_tool` set to auto-approve callback
- `cwd` set to the project working directory

**Auto-Approve Callback**:
```python
async def _auto_approve_all(
    tool_name: str,
    input_data: dict,
    context
):
    """Auto-approve all tools without prompting."""
    logger.debug(f"Auto-approving tool: {tool_name}")
    from claude_agent_sdk import PermissionResultAllow
    return PermissionResultAllow()
```

**Working Directory**:
```python
working_dir = Path(__file__).parent.parent  # personal-financial-analyst/
```

**Hints**:
- Add `_auto_approve_all()` function at the top of the file
- `cwd` should point to project root, not agent/ subdirectory
- See [architecture.md](agent/architecture.md) Section "Complete Agent Configuration" for details

### Part 5: Execute Orchestrator (25 min)

**Function**: `_run_orchestrator()` - Step 5

**Location**: Lines 203-224 in [financial_orchestrator.py](agent/financial_orchestrator.py)

**Requirements**:
- Create `ClaudeSDKClient` with the options from Part 4
- Use `async with` for proper cleanup
- Send initial query with context about transactions and subscriptions
- Stream responses and handle different message types
- Print text output in real-time

**Message Handling**:
```python
async with ClaudeSDKClient(options=options) as client:
    await client.query(prompt)

    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end='', flush=True)
        elif isinstance(message, ResultMessage):
            logger.info(f"Duration: {message.duration_ms}ms")
            logger.info(f"Cost: ${message.total_cost_usd:.4f}")
            break
```

**Prompt Template**:
```python
prompt = f"""Analyze my financial data and {user_query}

I have:
- {len(bank_transactions)} bank transactions
- {len(credit_card_transactions)} credit card transactions
- {len(subscriptions)} identified subscriptions

Please:
1. Identify opportunities for savings
2. Delegate research to the research agent
3. Delegate negotiation strategies to the negotiation agent
4. Delegate tax analysis to the tax agent
5. Read their results and create a final report at data/final_report.md
"""
```

**Hints**:
- Import `AssistantMessage`, `ResultMessage`, `TextBlock` from `claude_agent_sdk`
- Use `end=''` and `flush=True` for streaming text effect
- See [architecture.md](agent/architecture.md) Section "Agent Execution" for complete example

## Testing Your Implementation

### Run the Orchestrator

```bash
cd agent
uv run python financial_orchestrator.py \
  --username john_doe \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --query "How can I save money?"
```

**Note**: Claude Agent SDK cannot run inside Claude Code sessions. Run in a regular terminal or use:
```bash
unset CLAUDECODE
uv run python financial_orchestrator.py ...
```

### Verify Outputs

Check that these files are created:
- `data/raw_data/bank_transactions.json` (from orchestrator)
- `data/raw_data/credit_card_transactions.json` (from orchestrator)
- `data/agent_outputs/research_results.md` (from research agent)
- `data/agent_outputs/negotiation_scripts.md` (from negotiation agent)
- `data/agent_outputs/tax_analysis.md` (from tax agent)
- `data/final_report.md` (from orchestrator synthesis)

### Expected Behavior

The orchestrator should:
1. Fetch data from both MCP servers
2. Detect subscriptions from transactions
3. Invoke all three sub-agents IN PARALLEL (look for simultaneous execution)
4. Read sub-agent outputs after they complete
5. Generate final comprehensive report with savings recommendations

## Verification Checklist

Before submitting, verify:
- [ ] `_detect_subscriptions()` correctly filters recurring transactions
- [ ] MCP servers are correctly configured with matching names
- [ ] Both `get_bank_transactions` and `get_credit_card_transactions` tools are called
- [ ] All three sub-agents are defined with correct prompts and models
- [ ] `ClaudeAgentOptions` includes all required parameters
- [ ] Agent execution uses `async with` and handles messages correctly
- [ ] Streaming text output works (text appears in real-time)
- [ ] All output files are created in correct locations
- [ ] Final report includes synthesis of all three sub-agent findings
- [ ] Code follows project standards (type hints, docstrings, logging)

## Resources

### Essential Reading

1. **[architecture.md](agent/architecture.md)** - Comprehensive implementation guide
   - High-level architecture
   - Low-level implementation details
   - Code walkthroughs with explanations
   - Complete working examples

2. **[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk)** - Official SDK documentation
   - Getting started guide
   - API reference
   - Multi-agent patterns
   - Tool integration

3. **[FastMCP Documentation](https://github.com/gofastmcp/fastmcp)** - MCP server framework
   - HTTP transport setup
   - Tool definitions
   - Configuration patterns

### Additional Resources

- **[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)** - Anthropic's guide
- **[Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents)** - Agent patterns

## Key Learning Points

### 1. Orchestrator-Workers Pattern
- Central orchestrator coordinates specialized workers
- Workers run in parallel for efficiency
- File-based communication for simplicity

### 2. Model Selection Strategy
- **Sonnet** for orchestrator (complex reasoning, synthesis)
- **Haiku** for all sub-agents (cost optimization, 75% reduction)
- Well-structured prompts make Haiku agents highly effective

### 3. MCP Protocol Integration
- Standardized way to connect external data sources
- Servers run independently in separate processes
- Tool discovery happens automatically

### 4. Agent vs Tool Decision
- **Use Tools** for: Data fetching, file operations, simple transformations
- **Use Sub-Agents** for: Complex reasoning, research, strategic planning

### 5. Parallel Execution
- Independent tasks should run simultaneously
- Requires explicit instruction in orchestrator prompt
- SDK handles parallel coordination automatically

## Troubleshooting

### MCP Server Not Responding
```bash
# Verify servers are running
curl http://127.0.0.1:5001/health
curl http://127.0.0.1:5002/health

# Check for port conflicts
lsof -i :5001
lsof -i :5002
```

### Agent SDK Import Errors
```bash
# Verify installation
cd agent
uv pip list | grep claude-agent-sdk

# Reinstall if needed
uv pip install -r requirements.txt
```

### Sub-Agents Not Creating Output Files
- Check that agent prompts specify exact file paths
- Verify prompts use triple enforcement (start, middle, end)
- See [architecture.md](agent/architecture.md) Section "Enforcing File Outputs with Haiku Agents"

### Running Inside Claude Code
```bash
# Claude Agent SDK detects Claude Code and blocks
# Solution: Run in regular terminal or unset environment variable
unset CLAUDECODE
uv run python financial_orchestrator.py ...
```

## Success Criteria

Your implementation is complete when:
- All TODO sections in `financial_orchestrator.py` are implemented
- Running the test command successfully fetches data from both MCP servers
- Three sub-agents execute in parallel (visible in logs/output)
- All six output files are created in correct locations
- Final report includes synthesized findings from all three agents
- Code passes syntax validation: `uv run python -m py_compile financial_orchestrator.py`
