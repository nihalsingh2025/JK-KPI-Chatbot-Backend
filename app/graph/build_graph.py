"""
Wires all nodes into a single LangGraph StateGraph. This is the only file
that defines control flow - individual nodes stay simple, single-purpose
functions with no knowledge of what runs before or after them.

Flow:
    retrieve_kpi_context -> generate_sql -> execute_sql -> format_answer
        -> [plot node, only if the question asks for a chart] -> END

NOTE: the checkpointer is now passed in rather than created here, so the
same graph-building logic can be compiled with either an InMemorySaver
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
from app.graph.nodes.detect_plot_intent import detect_plot_intent

logger = logging.getLogger(__name__)


# A full LLM call just to decide "should I plot" would be wasteful for a binary decision like this.
def _route_after_format_answer(state: AgentState) -> str:
    route = "plot" if state.get("wants_plot") else "end"
    logger.info("routing after format_answer -> %s", route)
    return route


def build_graph(checkpointer):
    logger.info("building graph")
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_kpi_context", retrieve_kpi_context)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("format_answer", format_answer)
    graph.add_node("detect_plot_intent", detect_plot_intent)
    graph.add_node("plot", plot)

    graph.set_entry_point("retrieve_kpi_context")
    graph.add_edge("retrieve_kpi_context", "generate_sql")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("execute_sql", "format_answer")

    graph.add_edge("format_answer", "detect_plot_intent")

    graph.add_conditional_edges(
        "detect_plot_intent",
        _route_after_format_answer,
        {"plot": "plot", "end": END},
    )
    graph.add_edge("plot", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("graph compiled successfully")
    return compiled
