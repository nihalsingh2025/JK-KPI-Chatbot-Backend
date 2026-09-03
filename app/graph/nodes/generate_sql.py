"""
Node: generate_sql

Builds a prompt from the retrieved KPI context plus enum context and asks
ChatOpenAI to produce a single Databricks SQL SELECT statement. Only the
retrieved fragments (not the full KPI registry) go into the prompt, keeping
token usage proportional to what the question actually needs.
"""

import json
import logging
from pydantic import  BaseModel
from langchain_openai import ChatOpenAI
from app.config import SQL_MODEL, OPENAI_API_KEY, DATABRICKS_TABLE
from app.graph.state import AgentState, QueryHistoryEntry
from datetime import date

logger = logging.getLogger(__name__)

class SQLQuery(BaseModel):
    sql: str

_llm = ChatOpenAI(model=SQL_MODEL, api_key=OPENAI_API_KEY, temperature=0).with_structured_output(SQLQuery)

SYSTEM_PROMPT = f"""You are a SQL generation assistant for a tyre
manufacturing KPI dashboard. You write a single Databricks SQL SELECT
statement against the table {DATABRICKS_TABLE}.

General rules:
- Always use lower() when comparing string columns to static values.
- Only select the columns listed under default_select_columns for the matched KPI family, plus any columns needed to answer the question.
- Follow the variant_selection_rules and gotchas given in the KPI context exactly.
- Default to the overall (non machine/sku-specific) KPI variant if the question does not specify a breakdown.
- Always sort results with ORDER BY date ASC by default, unless the user explicitly asks for a different sort order.
- Always write date literals as Date 'YYYY-MM-DD'.
- There is no separate "shift" column. Shift-level data is selected via granularity = 'SHIFT A' / 'SHIFT B' / 'SHIFT C'.
- The "Matched KPI context" below may contain more than one KPI family's information. Only one is relevant to the current question - use context and the question itself to determine which one applies, and ignore the other(s) when writing SQL.
- Judge relevance by content, not by position - do not assume earlier or later context (KPI context, history, or otherwise) is more important just because of where it appears in this prompt. Use whichever information actually answers the current question correctly.

Granularity & aggregation rules:
- If the question implies a date range without asking for a total, return per-day rows (granularity='day'). If it asks for a total/overall figure, use SUM(actual_value), drop 'date' from SELECT, and GROUP BY the remaining relevant dimensions.
- If the question asks for a kpi for a month/year, use month/year granularity with the first day of that month/year in date - do not return daily rows unless explicitly asked.
- Always GROUP BY unit_of_measurement when using SUM, to avoid summing across different units.

Date & follow-up resolution priority:
1. If the question states its own date/month/year, use that.
2. If it has none and history shows a resolved date, reuse that exact date (and granularity - if the previous question was month-level, stay month-level).
3. "Today" or "current" genuinely means today's date. But "that day", "this day", "the same day", "that date" are backward references to a date already mentioned in the recent question history - resolve these to the historical date, never to TODAY_DATE.
4. Only use today's date for relative phrases ("today", "this month") with no history to inherit from.
For all other missing filters (product, section, machine, etc.), apply the same priority - reuse the most recent relevant past question's values unless the current question fully specifies its own. Only carry over filters relevant to the current KPI family (e.g. don't carry Production's product_type into an OEE question).

OEE drill-down rules:
- "Breakdown" or "drill down" after an OEE-family question (Availability/Performance/Quality/OEE) means decomposing into Availability, Performance, and Quality at the same level and date as the previous question - not the Breakdown (downtime-reason) KPI. Only use Breakdown if the user is clearly asking about machine stoppage reasons.
"""

def _build_user_prompt(state: AgentState) -> str:
    parts = [f"USER QUESTION (most important - this defines what the user actually wants): {state['user_query']}", "", "Matched KPI context (use this to pick the correct kpi_name, columns, and rules):"]
    for entry in state.get("kpi_context", []):
        parts.append(json.dumps(entry, indent=2))

    enum_context = state.get("enum_context") or {}
    if enum_context:
        parts.append("\n\nKNOWN EXACT COLUMN VALUES (use these exact strings, not the user's own wording, for filters):")
        parts.append(json.dumps(enum_context, indent=2))

    history = state.get("query_history") or []
    if history:
        parts.append("\nRECENT QUESTION HISTORY (check if the current question is a follow-up to one of these - reuse their filters only if relevant):")
        for entry in history[-5:]:
            parts.append(f"Q: {entry.user_query}\nSQL: {entry.generated_sql}")
        parts.append("")

    parts.append(f"TODAY_DATE: {date.today().isoformat()} (** ONLY use for literal 'today'/'current' references, never for 'that day'/'this day' - see date resolution rules), Use this at last priority when you don't have any information from history as well as user's query regarding date. **" )

    return "\n".join(parts)


def generate_sql(state: AgentState) -> AgentState:
    logger.info("generate_sql: start | kpi_family=%s", state.get("current_kpi_family"))
    user_prompt = _build_user_prompt(state)

    result = _llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    state["generated_sql"] = result.sql.strip()
    logger.info("generate_sql: generated SQL | %s", state["generated_sql"])

    history = state.get("query_history") or []
    history.append(QueryHistoryEntry(
        user_query=state["user_query"],
        generated_sql=state["generated_sql"],
        kpi_family=state.get("current_kpi_family") or "",
    ))
    state["query_history"] = history[-10:]

    logger.info("generate_sql: done | history_len=%d", len(state["query_history"]))
    return state
