"""Interactive approval handler for Claude Agent SDK tool calls."""

import sys
from typing import Dict, List, Set


class ApprovalHandler:
    """Handles interactive approval of tool calls with smart defaults."""

    def __init__(
        self,
        auto_approve_patterns: List[str] = None,
        auto_deny_patterns: List[str] = None
    ):
        """Initialize approval handler.

        Args:
            auto_approve_patterns: Tool patterns to auto-approve (e.g., ["mcp__*", "Read"])
            auto_deny_patterns: Tool patterns to auto-deny (e.g., ["Bash"])
        """
        self.auto_approve = set(auto_approve_patterns or [])
        self.auto_deny = set(auto_deny_patterns or [])
        self.session_approved: Set[str] = set()
        self.session_denied: Set[str] = set()

    def should_approve(
        self,
        tool_name: str,
        tool_input: Dict = None,
        remember: bool = True
    ) -> bool:
        """Determine if a tool call should be approved.

        Args:
            tool_name: Name of the tool being called
            tool_input: Input parameters for the tool
            remember: Whether to remember the decision for this session

        Returns:
            True if approved, False if denied
        """
        # Check if already approved/denied this session
        if tool_name in self.session_approved:
            print(f"[Auto-approved from session memory: {tool_name}]")
            return True
        if tool_name in self.session_denied:
            print(f"[Auto-denied from session memory: {tool_name}]")
            return False

        # Check auto-approve patterns
        for pattern in self.auto_approve:
            if self._matches_pattern(tool_name, pattern):
                print(f"[Auto-approved by pattern '{pattern}': {tool_name}]")
                if remember:
                    self.session_approved.add(tool_name)
                return True

        # Check auto-deny patterns
        for pattern in self.auto_deny:
            if self._matches_pattern(tool_name, pattern):
                print(f"[Auto-denied by pattern '{pattern}': {tool_name}]")
                if remember:
                    self.session_denied.add(tool_name)
                return False

        # Prompt user
        return self._prompt_user(tool_name, tool_input, remember)

    def _matches_pattern(self, tool_name: str, pattern: str) -> bool:
        """Check if tool name matches a glob pattern."""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return tool_name.startswith(prefix)
        return tool_name == pattern

    def _prompt_user(
        self,
        tool_name: str,
        tool_input: Dict = None,
        remember: bool = True
    ) -> bool:
        """Prompt user for approval with context."""
        print("\n" + "=" * 70)
        print(f"TOOL APPROVAL REQUEST")
        print("=" * 70)
        print(f"Tool: {tool_name}")

        if tool_input:
            print(f"Parameters:")
            for key, value in tool_input.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"  {key}: {value_str}")

        print("=" * 70)

        while True:
            response = input(
                "Approve this tool call? "
                "[y]es / [n]o / [a]lways / n[e]ver / [i]nfo: "
            ).lower().strip()

            if response in ['y', 'yes']:
                print(f"✓ Approved: {tool_name}\n")
                return True
            elif response in ['n', 'no']:
                print(f"✗ Denied: {tool_name}\n")
                return False
            elif response in ['a', 'always']:
                print(f"✓ Approved and remembered for session: {tool_name}\n")
                if remember:
                    self.session_approved.add(tool_name)
                return True
            elif response in ['e', 'never']:
                print(f"✗ Denied and remembered for session: {tool_name}\n")
                if remember:
                    self.session_denied.add(tool_name)
                return False
            elif response in ['i', 'info']:
                self._show_tool_info(tool_name)
            else:
                print("Invalid response. Please enter y/n/a/e/i")

    def _show_tool_info(self, tool_name: str):
        """Show information about the tool."""
        info = {
            "mcp__": "MCP server tools - Access external data sources",
            "write": "File system - Write files to disk",
            "Read": "File system - Read files from disk",
            "Agent": "Sub-agents - Spawn specialized agents",
            "Bash": "Shell commands - Execute system commands",
        }

        print("\nTool Information:")
        for prefix, description in info.items():
            if tool_name.startswith(prefix) or tool_name == prefix:
                print(f"  {description}")
                break
        else:
            print(f"  Unknown tool: {tool_name}")
        print()


def create_approval_handler(mode: str = "interactive") -> ApprovalHandler:
    """Create an approval handler with preset configurations.

    Args:
        mode: Approval mode
            - "interactive": Prompt for all tools
            - "safe": Auto-approve safe tools, prompt for risky ones
            - "auto": Auto-approve all tools
            - "strict": Deny all tools by default

    Returns:
        Configured ApprovalHandler instance
    """
    if mode == "auto":
        return ApprovalHandler(
            auto_approve_patterns=["*"]
        )
    elif mode == "safe":
        return ApprovalHandler(
            auto_approve_patterns=[
                "mcp__*",           # MCP tools
                "Read",             # Reading files
                "write",            # Writing files
                "Agent",            # Sub-agents
            ],
            auto_deny_patterns=[
                "Bash",             # Shell commands require approval
            ]
        )
    elif mode == "strict":
        return ApprovalHandler(
            auto_deny_patterns=["*"]
        )
    else:  # interactive
        return ApprovalHandler(
            auto_approve_patterns=[],  # Nothing auto-approved
            auto_deny_patterns=[]      # Nothing auto-denied
        )


# Example usage
if __name__ == "__main__":
    # Test the approval handler
    handler = create_approval_handler(mode="safe")

    # These would auto-approve
    print(handler.should_approve("mcp__Bank_Account_Server__get_transactions"))
    print(handler.should_approve("Read"))

    # This would prompt
    print(handler.should_approve("Bash", {"command": "rm -rf /"}))
