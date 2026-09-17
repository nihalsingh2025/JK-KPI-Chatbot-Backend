"""
Node: analyze_agent

Runs after execute_sql when intent == "analyze". generate_sql already
fetched raw, unaggregated rows for this question (see the ANALYSIS MODE
line _build_user_prompt adds when intent == "analyze" - generate_sql's
system prompt and the kpi_registry are untouched). This node loads that
cached result once, hands the model three stat tools bound to it via
closures (see analysis_tools.py), and lets it call them - possibly more
than once, e.g. "compare this month's max to last month's" - up to a
bounded number of rounds.
"""

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from app.config import FAST_MODEL, OPENAI_API_KEY, ANALYZE_MAX_TOOL_ITERATIONS
from app.graph.state import AgentState
from app.tools.file_cache import load_result
from app.graph.tools.analysis_tools.analysis_tools import build_analysis_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer analytical questions about KPI data using
only the tools provided (compute_max, compute_min, compute_std_dev). You
never see the raw rows - call a tool to get the number you need. Call
tools as many times as needed (e.g. to compare two groups), then give a
short, direct final answer in plain language once you have what you need.
Available columns in this result: {columns}
The data has already been filtered by SQL to match exactly what the
question asked (date range, product, etc.) - never ask the user to
clarify scope or dates, just call the right tool on what's given."""


def analyze_agent(state: AgentState) -> AgentState:
    logger.info("analyze_agent: start | result_id=%s", state.get("result_id"))

    if not state.get("result_id"):
        state["analysis_answer"] = "No data was found for that question, so no analysis could be run."
        state["final_answer"] = state["analysis_answer"]
        return state

    df = load_result(state["result_id"])
    tools = build_analysis_tools(df)
    tool_map = {t.name: t for t in tools}
    llm = ChatOpenAI(model=FAST_MODEL, api_key=OPENAI_API_KEY, temperature=0).bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(columns=state.get("columns"))),
        HumanMessage(content=state["user_query"]),
    ]

    final_text = None
    for i in range(ANALYZE_MAX_TOOL_ITERATIONS):
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            final_text = ai_msg.content
            logger.info("analyze_agent: done after %d tool round(s)", i)
            break

        for call in ai_msg.tool_calls:
            tool_fn = tool_map.get(call["name"])
            if tool_fn is None:
                tool_result = {"error": f"Unknown tool {call['name']}"}
            else:
                try:
                    tool_result = tool_fn.invoke(call["args"])
                except Exception as exc:
                    logger.warning("analyze_agent: tool %s failed | %s", call["name"], exc)
                    tool_result = {"error": str(exc)}
            messages.append(ToolMessage(content=json.dumps(tool_result, default=str), tool_call_id=call["id"]))

    if final_text is None:
        logger.warning("analyze_agent: hit ANALYZE_MAX_TOOL_ITERATIONS without a final answer")
        final_text = "I gathered the numbers but couldn't finish summarizing them - could you narrow the question?"

    state["analysis_answer"] = final_text
    state["final_answer"] = final_text
    return state
