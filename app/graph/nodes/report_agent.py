"""
Node: report_agent

Entry node for intent == "report". Bypasses retrieve_kpi_context /
generate_sql / execute_sql entirely - reports are a small fixed catalog,
each tool in report_tools.py already knows its own SQL template. The LLM's
only job here is picking the right tool and filling its (few) tweakable
args from the user's wording, e.g. a date range or which downtime reasons
to include/exclude.

Single dispatch, not a ReAct loop: each report tool does its own fetch +
compute + PDF render internally and returns a result in one call, so there
is nothing to chain.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from zoneinfo import ZoneInfo
from app.config import FAST_MODEL, OPENAI_API_KEY
from app.graph.state import AgentState
from app.graph.tools.report_tools.report_tools import REPORT_TOOLS

logger = logging.getLogger(__name__)

_tool_map = {t.name: t for t in REPORT_TOOLS}
_llm = ChatOpenAI(model=FAST_MODEL, api_key=OPENAI_API_KEY, temperature=0).bind_tools(REPORT_TOOLS)

today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()

SYSTEM_PROMPT = f"""You generate KPI reports. Pick exactly one of the
available tools that matches what the user is asking for, and fill its
arguments from their wording (date range, which items to include or
exclude).

Today's date is {today} in IST(Indian Standard Time) format.

Use today's date as the reference when interpreting relative dates such
as "today", "yesterday", "this month", "last month", etc.

If they gave no date range, leave those arguments null - the tool applies
its own default. Always call exactly one tool; never answer in plain text.
"""

def report_agent(state: AgentState) -> AgentState:
    logger.info("report_agent: start | user_query=%r", state.get("user_query"))

    state["execution_error"] = None
    state["result_id"] = None
    state["row_count"] = None
    state["columns"] = None
    state["preview_rows"] = None
    state["download_path"] = None
    state["plot_spec"] = None
    state["current_kpi_family"] = None
    state["generated_sql"] = None
    state["wants_plot"] = False
    state["analysis_answer"] = None
    state["report_id"] = None
    state["report_path"] = None
    state["report_type"] = None
    state["report_error"] = None
    state["final_answer"] = None

    response = _llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["user_query"]),
    ])

    if not response.tool_calls:
        logger.info("report_agent: model returned no tool call")
        state["report_error"] = "Could not determine which report to generate."
        state["final_answer"] = (
            "I'm not sure which report you want - could you name the report "
            "(e.g. downtime report) and, if needed, a date range?"
        )
        return state

    call = response.tool_calls[0]
    if len(response.tool_calls) > 1:
        logger.warning("report_agent: model requested %d tools, only running the first", len(response.tool_calls))

    tool_fn = _tool_map.get(call["name"])
    if tool_fn is None:
        logger.error("report_agent: model picked unknown tool %s", call["name"])
        state["report_error"] = f"Unknown report tool: {call['name']}"
        state["final_answer"] = "Something went wrong picking the report type - please try rephrasing."
        return state

    logger.info("report_agent: calling tool=%s | args=%s", call["name"], call["args"])
    result = tool_fn.invoke(call["args"])

    if "error" in result:
        logger.warning("report_agent: tool returned error | %s", result["error"])
        state["report_error"] = result["error"]
        state["final_answer"] = result["error"]
        return state

    state["report_id"] = result["report_id"]
    state["report_path"] = result["report_path"]
    state["report_type"] = result["report_type"]
    state["final_answer"] = (
        f"Your {result['report_type']} report for {result.get('period_label', 'the requested period')} "
        f"is ready ({result['row_count']} rows used)."
    )

    logger.info("report_agent: done | report_id=%s", result["report_id"])
    return state
