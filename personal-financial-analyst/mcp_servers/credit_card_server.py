"""Credit Card MCP Server - Provides simulated credit card transaction data."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


mcp = FastMCP("Credit Card Server")


DATA_FILE: Path = Path(__file__).parent / "mock_data" / "credit_card_transactions.csv"


def _load_transactions_from_csv(
    username: str,
    start_date: str,
    end_date: str
) -> list[dict]:
    """Load transactions from CSV file for given user and date range."""
    transactions = []

    logger.info(f"Loading credit card transactions for user: {username}")
    logger.debug(f"Date range: {start_date} to {end_date}")

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        raise ValueError("Dates must be in YYYY-MM-DD format") from e

    if not DATA_FILE.exists():
        logger.error(f"Data file not found: {DATA_FILE}")
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    logger.debug(f"Reading data from: {DATA_FILE}")

    with open(DATA_FILE, "r") as f:
        reader = csv.DictReader(f)
        row_count = 0
        matched_count = 0

        for row in reader:
            row_count += 1

            if row["username"] != username:
                continue

            trans_date = datetime.strptime(row["date"], "%Y-%m-%d")
            if start_dt <= trans_date <= end_dt:
                matched_count += 1
                transactions.append({
                    "date": row["date"],
                    "merchant": row["merchant"],
                    "category": row["category"],
                    "amount": float(row["amount"]),
                    "recurring": row["recurring"].lower() == "true"
                })

        logger.info(f"Processed {row_count} rows, found {matched_count} matching transactions")

    logger.info(f"Loaded {len(transactions)} credit card transactions for {username}")
    return transactions


def _calculate_summary(transactions: list[dict]) -> dict:
    """Calculate summary statistics for transactions."""
    logger.debug(f"Calculating summary for {len(transactions)} transactions")

    total_charges = sum(abs(t["amount"]) for t in transactions)
    recurring_charges = sum(abs(t["amount"]) for t in transactions if t["recurring"])

    summary = {
        "total_charges": round(total_charges, 2),
        "recurring_charges": round(recurring_charges, 2),
        "transaction_count": len(transactions),
        "recurring_count": sum(1 for t in transactions if t["recurring"])
    }

    logger.debug(f"Summary: {json.dumps(summary, indent=2)}")
    return summary


@mcp.tool()
def get_credit_card_transactions(
    username: str,
    start_date: str,
    end_date: str
) -> dict:
    """Get credit card transactions for a user within a date range.

    Args:
        username: Username for the account (e.g., 'john_doe', 'jane_smith')
        start_date: Start date in YYYY-MM-DD format (e.g., '2026-01-01')
        end_date: End date in YYYY-MM-DD format (e.g., '2026-01-31')

    Returns:
        Dictionary containing:
        - card_name: Credit card account name
        - date_range: Date range queried
        - transactions: List of transaction objects
        - summary: Summary statistics
    """
    logger.info(f"[TOOL CALL] get_credit_card_transactions: username={username}, start_date={start_date}, end_date={end_date}")

    try:
        transactions = _load_transactions_from_csv(username, start_date, end_date)
        summary = _calculate_summary(transactions)

        result = {
            "card_name": f"{username}_credit_card",
            "card_type": "visa",
            "date_range": f"{start_date} to {end_date}",
            "transactions": transactions,
            "summary": summary
        }

        logger.info(f"[TOOL SUCCESS] Returning {len(transactions)} transactions")
        logger.debug(f"Result summary:\n{json.dumps(summary, indent=2)}")

        return result

    except Exception as e:
        logger.error(f"[TOOL ERROR] Failed to get credit card transactions: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starting Credit Card MCP Server")
    logger.info("=" * 70)
    logger.info(f"Data file: {DATA_FILE}")
    logger.info(f"Transport: streamable-http")
    logger.info(f"Port: 5002")
    logger.info(f"Host: 127.0.0.1")
    logger.info(f"Logging level: DEBUG (verbose)")
    logger.info("=" * 70)

    # Run with streamable-http transport on port 5002
    mcp.run(transport="http", port=5002, host="127.0.0.1")
