"""Test script to verify MCP servers using the MCP protocol."""

import asyncio
import json
import logging
from pathlib import Path

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


async def test_mcp_servers():
    """Test MCP servers by connecting and listing tools."""

    logger.info("=" * 70)
    logger.info("MCP Protocol Test")
    logger.info("=" * 70)

    # Configure MCP servers
    mcp_servers = {
        "Bank Account Server": {
            "type": "http",
            "url": "http://127.0.0.1:5001/mcp"
        },
        "Credit Card Server": {
            "type": "http",
            "url": "http://127.0.0.1:5002/mcp"
        }
    }

    # Create minimal agent options
    options = ClaudeAgentOptions(
        model="haiku",  # Use cheaper model for testing
        system_prompt="You are a test agent.",
        mcp_servers=mcp_servers,
        allowed_tools=["mcp__Bank Account Server__*", "mcp__Credit Card Server__*"],  # Allow MCP tools
        cwd=str(Path.cwd())
    )

    try:
        logger.info("Connecting to Claude Agent SDK...")
        async with ClaudeSDKClient(options=options) as client:
            logger.info("Connected successfully!")

            # Get MCP status
            logger.info("\nChecking MCP server status...")
            mcp_status = await client.get_mcp_status()
            logger.info(f"MCP Status: {json.dumps(mcp_status, indent=2, default=str)}")

            # Check if servers are connected
            servers = mcp_status.get("mcpServers", [])
            connected_servers = [s for s in servers if s.get("status") == "connected"]
            failed_servers = [s for s in servers if s.get("status") == "failed"]

            logger.info("\n" + "=" * 70)
            logger.info("Connection Results")
            logger.info("=" * 70)

            if connected_servers:
                logger.info(f"\n✓ Successfully connected to {len(connected_servers)} server(s):")
                for server in connected_servers:
                    tools = server.get("tools", [])
                    logger.info(f"  - {server['name']}: {len(tools)} tool(s) available")
                    for tool in tools:
                        logger.info(f"    * {tool['name']}")

            if failed_servers:
                logger.info(f"\n✗ Failed to connect to {len(failed_servers)} server(s):")
                for server in failed_servers:
                    logger.info(f"  - {server['name']}: {server.get('error', 'Unknown error')}")

            logger.info("\n" + "=" * 70)
            if len(connected_servers) == len(servers) and len(servers) > 0:
                logger.info("✓ ALL TESTS PASSED!")
                logger.info("\nMCP servers are configured correctly and tools are available.")
                logger.info("You can now run the full orchestrator solution.")
            else:
                logger.info("✗ SOME TESTS FAILED")
                logger.info("\nPlease check the server configuration and try again.")
            logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        logger.error("\nTroubleshooting:")
        logger.error("1. Make sure MCP servers are running on ports 5001 and 5002")
        logger.error("2. Check that ANTHROPIC_API_KEY is set")
        logger.error("3. Verify server names match FastMCP definitions")
        raise


async def main():
    """Main entry point."""
    await test_mcp_servers()


if __name__ == "__main__":
    asyncio.run(main())
