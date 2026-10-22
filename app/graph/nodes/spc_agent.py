"""
Node: spc_agent

Runs after execute_sql when intent == "spc". Gets the row count, columns and a 50-row preview
(like format_answer) plus the SPC tools bound to the cached result.

- Plain questions (lookup, status list, lowest/highest, comparing machines or recipes): the model
  answers straight from the preview, no tool call.
- Change questions: it calls compare_over_time. Capability questions that need several metrics
  of the same stream: it calls streams_to_rows.

Bounded tool loop, same pattern as analyze_agent. To add a tool: add it in spc_tools.py.
"""

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from app.config import FAST_MODEL, OPENAI_API_KEY, ANALYZE_MAX_TOOL_ITERATIONS, LARGE_RESULT_ROW_THRESHOLD
from app.graph.state import AgentState
from app.tools.file_cache import load_result
from app.graph.tools.spc_tools.spc_tools import build_spc_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer SPC (process capability) questions for a tyre plant.
You are given row_count, columns and preview_rows (the first rows of the SQL result), and tools.

- Lookups, status lists, lowest/highest, comparisons of machines or recipes: answer directly from
  preview_rows. Never invent numbers.
- What got worse / better / changed / improved / deteriorated: call compare_over_time once
  (metric from the question, default Cpk), then answer from its result.
- Capability questions that need Cp, Cpk, Pp and Ppk of the same machine/recipe/parameter together:
  call streams_to_rows (one row per stream), then answer from its result.
- Reading capability: Cp (or Pp) below 1.33 = spread too wide. Cp at least 1.33 but Cpk clearly lower =
  process is off-centre. Pp (or Ppk) clearly below Cp (or Cpk) = process drifts over time.
- large_result is a flag, do not judge from row counts. If false: give a clean direct summary and do
  not mention downloading. If true: show the preview rows as a snapshot and say the full result can
  be downloaded.
- Always state the date of the values. Cp/Cpk cover the 30 days ending that date, Pp/Ppk the 90 days
  ending that date.
- status 'Insufficient Data' with an empty actual_value means not enough data - say so, never fill a number.
- Cp/Cpk bands: >=1.67 Healthy, >=1.33 Warning, >=1.00 Major, below 1.00 Critical.
  Pp/Ppk bands: >=1.50 Healthy, >=1.20 Warning, >=1.00 Major, below 1.00 Critical.
- UCL/LCL are control limits, not spec limits. Mean, sigma, spec limits and sample sizes are not available.
- Keep the answer short and simple."""


def spc_agent(state: AgentState) -> AgentState:
    logger.info("spc_agent: start | result_id=%s", state.get("result_id"))

    if state.get("execution_error"):
        state["final_answer"] = f"The query could not be completed: {state['execution_error']}"
        return state
    if not state.get("result_id") or not state.get("row_count"):
        state["final_answer"] = "No data was found for that question."
        return state

    df = load_result(state["result_id"])
    tools = build_spc_tools(df)
    tool_map = {t.name: t for t in tools}
    llm = ChatOpenAI(model=FAST_MODEL, api_key=OPENAI_API_KEY, temperature=0).bind_tools(tools)

    context = {
        "user_question": state["user_query"],
        "row_count": state["row_count"],
        "columns": state["columns"],
        "preview_rows": state["preview_rows"],
        "large_result": state["row_count"] > LARGE_RESULT_ROW_THRESHOLD,
    }
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(context, indent=2, default=str)),
    ]

    final_text = None
    for i in range(ANALYZE_MAX_TOOL_ITERATIONS):
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            final_text = ai_msg.content
            logger.info("spc_agent: done after %d tool round(s)", i)
            break

        for call in ai_msg.tool_calls:
            tool_fn = tool_map.get(call["name"])
            if tool_fn is None:
                tool_result = {"error": f"Unknown tool {call['name']}"}
            else:
                try:
                    tool_result = tool_fn.invoke(call["args"])
                except Exception as exc:
                    logger.warning("spc_agent: tool %s failed | %s", call["name"], exc)
                    tool_result = {"error": str(exc)}
            messages.append(ToolMessage(content=json.dumps(tool_result, default=str), tool_call_id=call["id"]))

    if final_text is None:
        logger.warning("spc_agent: hit max tool iterations without a final answer")
        final_text = "I gathered the numbers but couldn't finish summarizing them - could you narrow the question?"

    state["final_answer"] = final_text.strip()
    return state
