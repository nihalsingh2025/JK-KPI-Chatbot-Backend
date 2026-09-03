"""
Thin wrapper around the existing Databricks REST API used to run SQL and
get results back as a list of row dicts. This is the only place that knows
about the HTTP details - if this project ever moves to a direct Databricks
SQL connector, only this file needs to change.
"""

import logging
import requests

from app.config import DATABRICKS_API_URL, DATABRICKS_API_KEY, DATABRICKS_ENGINE

logger = logging.getLogger(__name__)


class DatabricksQueryError(Exception):
    pass


def run_query(sql: str) -> list:
    """Execute a SQL string against the gold layer and return rows as dicts."""
    logger.info("databricks_api: sending query")
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
