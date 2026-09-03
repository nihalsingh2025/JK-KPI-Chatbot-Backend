"""
Node: retrieve_kpi_context

Embeds the user's question, retrieves the top-k closest KPI registry
entries, and resolves any enum values (product_type/section/etc) mentioned
in the question. This context is what gets injected into the SQL
generation prompt in the next node.
"""

import logging

from app.graph.state import AgentState
from app.retrieval.embed_store import retriever
from app.retrieval.enum_values import relevant_enum_context

logger = logging.getLogger(__name__)


def retrieve_kpi_context(state: AgentState) -> AgentState:
    logger.info("retrieve_kpi_context: start | user_query=%r", state.get("user_query"))

    state["execution_error"] = None
    state["result_id"] = None
    state["row_count"] = None
    state["columns"] = None
    state["preview_rows"] = None
    state["download_path"] = None
    state["plot_spec"] = None

    query = state["user_query"]
    history = state.get("query_history") or []

    retrieved = retriever.retrieve(query)

    if history:
        last_family = history[-1].kpi_family
        if last_family and not any(e["kpi_family"] == last_family for e in retrieved):
            forced_entry = next((e for e in retriever.entries if e["kpi_family"] == last_family), None)
            if forced_entry:
                retrieved = [forced_entry] + retrieved
                logger.info("retrieve_kpi_context: forced carry-over of kpi_family=%s from history", last_family)

    state["kpi_context"] = retrieved
    state["current_kpi_family"] = retrieved[0]["kpi_family"] if retrieved else None
    state["enum_context"] = relevant_enum_context(query)

    logger.info(
        "retrieve_kpi_context: done | matched_families=%s | enum_context_keys=%s",
        [e.get("kpi_family") for e in retrieved],
        list(state["enum_context"].keys()),
    )

    return state
