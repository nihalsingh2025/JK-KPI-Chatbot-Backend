"""
Wires all nodes into a single LangGraph StateGraph. This is the only file
that defines control flow - individual nodes stay simple, single-purpose
functions with no knowledge of what runs before or after them.

Flow:
    entry routing by state["intent"] (defaults to "query" if missing/unknown):

    query / plot / analyze:
        retrieve_kpi_context -> generate_sql -> execute_sql
            -> [query]   format_answer                 -> END
            -> [plot]    plot                           -> END
            -> [analyze] analyze_agent                  -> END

    report (bypasses the pipeline above entirely - see report_agent.py):
        report_agent -> END

detect_plot_intent is gone: with an explicit "plot" intent from the
frontend, the old fallback classifier after format_answer is redundant -
if the user wanted a chart, they used the plot intent.

Adding a 5th intent later: write its node(s), add one line to
_POST_SQL_MAP (if it shares the pipeline) or wire it like report_agent
(if it doesn't), register the node(s) below. No existing branch changes.

NOTE: the checkpointer is passed in rather than created here, so the same
graph-building logic can be compiled with either an InMemorySaver
(local/testing) or the production PostgresSaver (see app/main.py).
"""

import logging

from langgraph.graph import StateGraph, END

from app.graph.state import AgentState
from app.graph.nodes.retrieve_kpi_context import retrieve_kpi_context
from app.graph.nodes.generate_sql import generate_sql
from app.graph.nodes.execute_sql import execute_sql
from app.graph.nodes.format_answer import format_answer
from app.graph.nodes.plot import plot
from app.graph.nodes.analyze_agent import analyze_agent
from app.graph.nodes.report_agent import report_agent

logger = logging.getLogger(__name__)

_VALID_INTENTS = ("query", "plot", "analyze", "report")

# Which node the shared pipeline hands off to after execute_sql, per intent.
_POST_SQL_MAP = {
    "query": "format_answer",
    "plot": "plot",
    "analyze": "analyze_agent",
}


def _normalize_intent(state: AgentState) -> str:
    intent = (state.get("intent") or "query").strip().lower()
    if intent not in _VALID_INTENTS:
        logger.warning("build_graph: unrecognized intent=%r, falling back to 'query'", intent)
        intent = "query"
    return intent


def _route_entry(state: AgentState) -> str:
    """query/plot/analyze share the retrieval+SQL pipeline; report skips straight to report_agent."""
    intent = _normalize_intent(state)
    state["intent"] = intent
    route = "shared_pipeline" if intent != "report" else "report_agent"
    logger.info("routing entry (intent=%s) -> %s", intent, route)
    return route


def _route_after_execute_sql(state: AgentState) -> str:
    intent = _normalize_intent(state)
    route = _POST_SQL_MAP[intent]
    logger.info("routing after execute_sql (intent=%s) -> %s", intent, route)
    return route


def build_graph(checkpointer):
    logger.info("building graph")
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_kpi_context", retrieve_kpi_context)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("format_answer", format_answer)
    graph.add_node("plot", plot)
    graph.add_node("analyze_agent", analyze_agent)
    graph.add_node("report_agent", report_agent)

    graph.set_conditional_entry_point(
        _route_entry,
        {"shared_pipeline": "retrieve_kpi_context", "report_agent": "report_agent"},
    )

    graph.add_edge("retrieve_kpi_context", "generate_sql")
    graph.add_edge("generate_sql", "execute_sql")

    graph.add_conditional_edges(
        "execute_sql",
        _route_after_execute_sql,
        {"format_answer": "format_answer", "plot": "plot", "analyze_agent": "analyze_agent"},
    )

    graph.add_edge("format_answer", END)
    graph.add_edge("plot", END)
    graph.add_edge("analyze_agent", END)
    graph.add_edge("report_agent", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("graph compiled successfully")
    return compiled
