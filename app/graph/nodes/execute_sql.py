"""
Node: execute_sql

Runs the generated SQL against the Databricks REST API and caches the
result. The graph state only ever receives row_count / columns / a preview
- never the full result set - so downstream nodes and any further LLM
calls stay cheap regardless of how many rows came back.
"""

import logging

from app.graph.state import AgentState
from app.tools.databricks_api import run_query, DatabricksQueryError
from app.tools.file_cache import cache_result

logger = logging.getLogger(__name__)


def execute_sql(state: AgentState) -> AgentState:
    sql = state["generated_sql"]
    logger.info("execute_sql: start | sql=%s", sql)

    try:
        rows = run_query(sql)
    except DatabricksQueryError as exc:
        logger.error("execute_sql: query failed | %s", exc)
        state["execution_error"] = str(exc)
        return state

    if not rows:
        logger.info("execute_sql: query returned no rows")
        state["row_count"] = 0
        state["columns"] = []
        state["preview_rows"] = []
        return state

    metadata = cache_result(rows)
    state["result_id"] = metadata["result_id"]
    state["row_count"] = metadata["row_count"]
    state["columns"] = metadata["columns"]
    state["preview_rows"] = metadata["preview_rows"]
    state["download_path"] = metadata["download_path"]

    logger.info(
        "execute_sql: done | result_id=%s | row_count=%d | columns=%s",
        metadata["result_id"], metadata["row_count"], metadata["columns"],
    )

    return state
