"""
Node: format_answer

Turns the query result metadata (never the raw rows) into a final natural
language answer. For results above the large-result threshold, the answer
points the user at the download rather than trying to summarize every row.
"""

import json
import logging
from langchain_openai import ChatOpenAI

from app.config import FAST_MODEL, OPENAI_API_KEY, LARGE_RESULT_ROW_THRESHOLD
from app.graph.state import AgentState

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model=FAST_MODEL, api_key=OPENAI_API_KEY, temperature=0)

SYSTEM_PROMPT = """You are a KPI dashboard assistant. Answer the user's
question using only the row_count, columns, and preview_rows given to you.
Never invent numbers not present in preview_rows.
If the granularity is MONTH and date is first day of month that means kpi are calculated for month that date is just a placeholder for that month, similar logic for YEAR granularity for first day of year.
Use the "large_result" flag to decide the answer style - do not judge
this yourself from row counts:
- If large_result is false: give a clean, direct summary of the preview
  rows. Do not mention downloading.
- If large_result is true: present the preview rows in sorted order as a
  quick snapshot, and mention that the full result can be downloaded for
  the detailed view."""


def format_answer(state: AgentState) -> AgentState:
    logger.info("format_answer: start")

    if state.get("execution_error"):
        state["final_answer"] = (
            f"The query could not be completed: {state['execution_error']}"
        )
        logger.info("format_answer: done | execution_error path")
        return state

    if state.get("row_count", 0) == 0:
        state["final_answer"] = "No data was found for that question."
        logger.info("format_answer: done | no rows path")
        return state

    context = {
        "user_question": state["user_query"],
        "row_count": state["row_count"],
        "columns": state["columns"],
        "preview_rows": state["preview_rows"],
        "large_result": state["row_count"] > LARGE_RESULT_ROW_THRESHOLD,
    }

    response = _llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(context, indent=2)},
    ])

    state["final_answer"] = response.content.strip()
    logger.info("format_answer: done | answer_len=%d", len(state["final_answer"]))
    return state