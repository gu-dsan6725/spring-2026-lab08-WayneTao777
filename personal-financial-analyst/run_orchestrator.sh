#!/bin/bash

# Auto-approve wrapper for financial orchestrator
# This script adds the --auto-approve flag automatically

echo "Running Financial Orchestrator with auto-approval..."
echo ""

# Run with --auto-approve flag to allow all tools without prompting
uv run python agent/financial_orchestrator_solution.py "$@" --auto-approve
