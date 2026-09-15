"""
Thin wrapper around the existing Databricks REST API used to run SQL and
get results back as a list of row dicts. This is the only place that knows
about the HTTP details - if this project ever moves to a direct Databricks
SQL connector, only this file needs to change.
"""

import logging
import requests
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.config import DATABRICKS_API_URL, DATABRICKS_API_KEY, DATABRICKS_ENGINE
from app.config import DATABRICKS_TABLE, DATABRICKS_TABLE_RECENT

logger = logging.getLogger(__name__)

def _select_table(sql: str) -> str:
    """Swap to the recent-data table when the query is day-granularity
    AND every date literal falls within [today-2, today] (IST).
    Falls back to the default table otherwise."""

    IST = ZoneInfo("Asia/Kolkata")
    _DATE_PATTERN = re.compile(r"Date\s*'(\d{4}-\d{2}-\d{2})'", re.IGNORECASE)
    _GRANULARITY_DAY_PATTERN = re.compile(r"lower\(granularity\)\s*=\s*'day'", re.IGNORECASE)

    if not _GRANULARITY_DAY_PATTERN.search(sql):
        logger.info("databricks_api: not day granularity, using default table")
        return sql

    dates_found = _DATE_PATTERN.findall(sql)
    if not dates_found:
        logger.info("databricks_api: no date literals found, using default table")
        return sql


    today_ist = datetime.now(IST).date()
    window_start = today_ist - timedelta(days=0)

    if len(dates_found) == 1:
        parsed_date = datetime.strptime(dates_found[0], "%Y-%m-%d").date()
        is_recent = window_start <= parsed_date <= today_ist
    else:
        is_recent = False

    if is_recent:
        logger.info("databricks_api: dates %s within recent window, routing to %s", dates_found, DATABRICKS_TABLE_RECENT)
        return sql.replace(DATABRICKS_TABLE, DATABRICKS_TABLE_RECENT)

    logger.info("databricks_api: dates %s outside recent window, using default table", dates_found)
    return sql

class DatabricksQueryError(Exception):
    pass

def run_query(sql: str) -> list:
    """Execute a SQL string against the gold layer and return rows as dicts."""
    sql = _select_table(sql)
    logger.info("databricks_api: sending query| sql=%s", sql)
    payload = {"engine": DATABRICKS_ENGINE, "sql": sql}
    headers = {
        "x-api-key": DATABRICKS_API_KEY,
        "Content-Type": "application/json",
    }

    response = requests.post(DATABRICKS_API_URL, json=payload, headers=headers)

    if response.status_code != 200:
        logger.error("databricks_api: request failed | status=%s", response.status_code)
        raise DatabricksQueryError(
            f"Query failed with status {response.status_code}: {response.text}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        logger.error("databricks_api: invalid JSON response")
        raise DatabricksQueryError("Invalid JSON response from Databricks API") from exc

    rows = data.get("rows") or data.get("data") or data.get("results")
    if rows is None:
        logger.error("databricks_api: no tabular data in response")
        raise DatabricksQueryError("No tabular data found in API response")

    logger.info("databricks_api: received %d rows", len(rows))
    return rows
