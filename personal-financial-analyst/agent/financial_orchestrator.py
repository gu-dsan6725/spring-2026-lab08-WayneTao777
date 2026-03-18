"""Financial Optimization Orchestrator Agent.

This agent demonstrates the orchestrator-workers pattern using Claude Agent SDK.
It fetches financial data from MCP servers and coordinates specialized sub-agents
to provide comprehensive financial optimization recommendations.
"""

import argparse
import asyncio
import json
import logging
import importlib.util
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AgentDefinition,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    PermissionResultAllow,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


DATA_DIR: Path = Path(__file__).parent.parent / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw_data"
AGENT_OUTPUTS_DIR: Path = DATA_DIR / "agent_outputs"


def _load_prompt(filename: str) -> str:
    """Load prompt from prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / filename
    return prompt_path.read_text()


async def _auto_approve_all(
    tool_name: str,
    input_data: dict,
    context
):
    """Auto-approve all tools without prompting."""
    logger.debug(f"Auto-approving tool: {tool_name}")
    return PermissionResultAllow()


def _get_mcp_servers() -> dict:
    """Return MCP server configuration."""
    return {
        "Bank Account Server": {
            "type": "http",
            "url": "http://127.0.0.1:5001/mcp"
        },
        "Credit Card Server": {
            "type": "http",
            "url": "http://127.0.0.1:5002/mcp"
        }
    }


def _ensure_directories():
    """Ensure all required directories exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(
    data: dict,
    filename: str
):
    """Save data to JSON file."""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved data to {filepath}")


def _extract_tool_result_data(content) -> dict:
    """Extract structured tool result data from SDK content payloads."""
    if isinstance(content, dict):
        if "structuredContent" in content and isinstance(content["structuredContent"], dict):
            return content["structuredContent"]
        return content

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if "structuredContent" in item and isinstance(item["structuredContent"], dict):
                    return item["structuredContent"]
                if "text" in item and isinstance(item["text"], str):
                    try:
                        parsed = json.loads(item["text"])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
    return {}


def _load_local_tool(module_path: Path, func_name: str):
    """Load a local MCP tool function directly from a module path."""
    spec = importlib.util.spec_from_file_location(func_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, func_name):
        raise AttributeError(f"Function {func_name} not found in {module_path}")
    return getattr(module, func_name)


def _detect_subscriptions(
    bank_transactions: list[dict],
    credit_card_transactions: list[dict]
) -> list[dict]:
    """Detect subscription services from recurring transactions.

    TODO: Implement logic to:
    1. Filter transactions marked as recurring
    2. Identify subscription patterns (monthly charges)
    3. Categorize by service type
    4. Calculate total monthly subscription cost

    Args:
        bank_transactions: List of bank transaction dicts
        credit_card_transactions: List of credit card transaction dicts

    Returns:
        List of subscription dictionaries with service name, amount, frequency
    """
    subscriptions = []

    all_transactions = bank_transactions + credit_card_transactions

    for transaction in all_transactions:
        if not transaction.get("recurring"):
            continue

        amount = transaction.get("amount")
        if amount is None or amount >= 0:
            continue

        service_name = transaction.get("description") or transaction.get("merchant") or "Unknown Service"

        subscriptions.append({
            "service": service_name,
            "amount": round(abs(float(amount)), 2),
            "frequency": "monthly"
        })

    return subscriptions


async def _fetch_financial_data(
    username: str,
    start_date: str,
    end_date: str
) -> tuple[dict, dict]:
    """Fetch data from Bank and Credit Card MCP servers.

    TODO: Implement MCP server connections using Claude Agent SDK
    1. Configure MCP server connections (ports 5001, 5002)
    2. Call get_bank_transactions tool
    3. Call get_credit_card_transactions tool
    4. Save raw data to files

    Args:
        username: Username for the account
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Tuple of (bank_data, credit_card_data) dictionaries
    """
    logger.info(f"Fetching financial data for {username} from {start_date} to {end_date}")

    mcp_servers = _get_mcp_servers()

    options = ClaudeAgentOptions(
        model="haiku",
        system_prompt="You are a data retrieval assistant. Only call the required MCP tools and do not add commentary.",
        mcp_servers=mcp_servers,
        can_use_tool=_auto_approve_all,
        cwd=str(Path(__file__).parent.parent)
    )

    bank_data: dict = {}
    credit_card_data: dict = {}
    tool_use_map: dict[str, str] = {}

    prompt = (
        "Use the MCP tools to fetch financial data. "
        f"Call get_bank_transactions and get_credit_card_transactions for username \"{username}\" "
        f"between {start_date} and {end_date}. "
        "Do not add any commentary after tool calls."
    )

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            tool_use_map[block.id] = block.name
                        elif isinstance(block, ToolResultBlock) and not block.is_error:
                            tool_name = tool_use_map.get(block.tool_use_id, "")
                            data = _extract_tool_result_data(block.content)
                            if tool_name.endswith("get_bank_transactions"):
                                bank_data = data
                            elif tool_name.endswith("get_credit_card_transactions"):
                                credit_card_data = data
                elif isinstance(message, ResultMessage):
                    break
    except Exception as e:
        logger.error(f"Error fetching financial data: {e}", exc_info=True)
        raise

    if not bank_data or not bank_data.get("transactions"):
        logger.warning("Bank data empty from MCP tool call, falling back to local tool.")
        bank_tool_path = Path(__file__).parent.parent / "mcp_servers" / "bank_server.py"
        get_bank_transactions = _load_local_tool(bank_tool_path, "get_bank_transactions")
        bank_data = get_bank_transactions(username=username, start_date=start_date, end_date=end_date)

    if not credit_card_data or not credit_card_data.get("transactions"):
        logger.warning("Credit card data empty from MCP tool call, falling back to local tool.")
        cc_tool_path = Path(__file__).parent.parent / "mcp_servers" / "credit_card_server.py"
        get_credit_card_transactions = _load_local_tool(cc_tool_path, "get_credit_card_transactions")
        credit_card_data = get_credit_card_transactions(
            username=username,
            start_date=start_date,
            end_date=end_date
        )

    # Save raw data
    _save_json(bank_data, "bank_transactions.json")
    _save_json(credit_card_data, "credit_card_transactions.json")

    return bank_data, credit_card_data


async def _run_orchestrator(
    username: str,
    start_date: str,
    end_date: str,
    user_query: str
):
    """Main orchestrator agent logic.

    TODO: Implement the orchestrator pattern:
    1. Fetch data from MCP servers (use tools)
    2. Perform initial analysis (detect subscriptions, anomalies)
    3. Decide which sub-agents to invoke based on query
    4. Define sub-agents using AgentDefinition
    5. Invoke sub-agents (can be parallel)
    6. Read and synthesize sub-agent results
    7. Generate final report

    Args:
        username: Username for the account
        start_date: Start date for analysis
        end_date: End date for analysis
        user_query: User's financial question/request
    """
    logger.info(f"Starting financial optimization orchestrator")
    logger.info(f"User query: {user_query}")

    _ensure_directories()

    # Step 1: Fetch financial data from MCP servers
    bank_data, credit_card_data = await _fetch_financial_data(
        username,
        start_date,
        end_date
    )

    # Step 2: Initial analysis
    logger.info("Performing initial analysis...")

    bank_transactions = bank_data.get("transactions", [])
    credit_card_transactions = credit_card_data.get("transactions", [])

    subscriptions = _detect_subscriptions(
        bank_transactions,
        credit_card_transactions
    )

    logger.info(f"Detected {len(subscriptions)} subscriptions")

    # Step 3: Define sub-agents
    research_agent = AgentDefinition(
        description="Research cheaper alternatives for subscriptions and services",
        prompt=_load_prompt("research_agent_prompt.txt"),
        tools=["write"],
        model="haiku"
    )

    negotiation_agent = AgentDefinition(
        description="Create negotiation strategies and scripts for bills and services",
        prompt=_load_prompt("negotiation_agent_prompt.txt"),
        tools=["write"],
        model="haiku"
    )

    tax_agent = AgentDefinition(
        description="Identify tax-deductible expenses and optimization opportunities",
        prompt=_load_prompt("tax_agent_prompt.txt"),
        tools=["write"],
        model="haiku"
    )

    agents = {
        "research_agent": research_agent,
        "negotiation_agent": negotiation_agent,
        "tax_agent": tax_agent,
    }

    # Step 4: Configure orchestrator agent with sub-agents
    working_dir = Path(__file__).parent.parent
    mcp_servers = _get_mcp_servers()

    options = ClaudeAgentOptions(
        model="sonnet",
        system_prompt=_load_prompt("orchestrator_system_prompt.txt"),
        mcp_servers=mcp_servers,
        agents=agents,
        can_use_tool=_auto_approve_all,
        cwd=str(working_dir)
    )

    # Step 5: Run orchestrator with Claude Agent SDK
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

    try:
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
    except Exception as e:
        logger.error(f"Error during orchestration: {e}", exc_info=True)
        raise

    # Step 6: Generate final report
    logger.info("Orchestration complete. Check data/final_report.md for results.")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Financial Optimization Orchestrator Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Basic analysis
    uv run python financial_orchestrator.py \\
        --username john_doe \\
        --start-date 2026-01-01 \\
        --end-date 2026-01-31 \\
        --query "How can I save $500 per month?"

    # Subscription analysis
    uv run python financial_orchestrator.py \\
        --username jane_smith \\
        --start-date 2026-01-01 \\
        --end-date 2026-01-31 \\
        --query "Analyze my subscriptions and find better deals"
"""
    )

    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Username for account (john_doe or jane_smith)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User's financial question or request"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = _parse_args()

    await _run_orchestrator(
        username=args.username,
        start_date=args.start_date,
        end_date=args.end_date,
        user_query=args.query
    )


if __name__ == "__main__":
    asyncio.run(main())
