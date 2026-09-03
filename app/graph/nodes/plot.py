"""
Node: plot

Only runs when routed to by the wants_plot conditional edge (see
graph/build_graph.py). Asks the LLM to choose a plot_spec
(chart_type/x/y/group_by) from the schema and preview only
the chart deterministically via tools/plotting.py. The LLM never writes
plotting code or sees the full dataset.
"""

import json
import logging
from langchain_openai import ChatOpenAI
from typing import Optional, Literal
from pydantic import BaseModel

from app.config import FAST_MODEL, OPENAI_API_KEY
from app.graph.state import AgentState

logger = logging.getLogger(__name__)

class PlotSpec(BaseModel):
    chart_type: Literal["bar", "line", "pie"]
    x: str
    y: str
    group_by: Optional[str] = None

_llm = ChatOpenAI(model=FAST_MODEL, api_key=OPENAI_API_KEY, temperature=0).with_structured_output(PlotSpec)

SYSTEM_PROMPT = """You choose how to visualize a KPI query result. Based on
the user's question, the available columns, and a small preview of the
data, choose the chart type and columns to use.
Pick bar for comparisons across categories, line for trends over dates, and
pie only for a small number of categories that sum to a whole.
-When comparing a small number of categories (like product types or Unit_of_measurement) across multiple dates, 
put date on the x-axis and the category being compared as group_by, so each date shows the categories side by side.Do not put the category on the x-axis with date as group_by.

For bar and line charts: x is the axis category (often date), y is the
numeric value, group_by is an optional second category to split by color.

For pie charts: x is the category column that becomes the pie slice
labels (e.g. remarks, machine, product_type), y is the numeric value
column. group_by must be left null for pie charts - do not use it.

"""


def _choose_plot_spec(state: AgentState) ->  PlotSpec:
    context = {
        "user_question": state["user_query"],
        "columns": state["columns"],
        "preview_rows": state["preview_rows"],
    }

    return _llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(context, indent=2)},
    ])


def plot(state: AgentState) -> AgentState:
    logger.info("plot: start")
    if not state.get("result_id"):
        logger.info("plot: no result_id, skipping")
        return state

    plot_spec = _choose_plot_spec(state)
    state["plot_spec"] = plot_spec.model_dump()
    logger.info("plot: chosen plot_spec=%s", state["plot_spec"])

    valid_columns = set(state.get("columns") or [])
    required_valid = (
            bool(plot_spec.x) and plot_spec.x in valid_columns
            and bool(plot_spec.y) and plot_spec.y in valid_columns
            and (plot_spec.group_by is None or plot_spec.group_by in valid_columns)
    )

    if not required_valid:
        logger.warning("plot: plot_spec failed column validation, dropping chart")
        state["figure"] = None
        state["final_answer"] = (
                (state.get("final_answer") or "")
                + "\n\n(A chart could not be generated for this result.)"
        )
        return state

    # typed AgentState contract, kept as a plain dict key for simplicity.
    logger.info("plot: done")
    return state
