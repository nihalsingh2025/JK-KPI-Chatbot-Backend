"""
AgentState is the single object passed between every LangGraph node.
Keep it flat and serializable - nothing here should hold a full dataframe,
only pointers (result_id) to cached data. This keeps token usage low and
makes the graph easy to trace and debug.
"""

from typing import TypedDict, Optional, List, Dict, Any, Literal
from pydantic import BaseModel

class QueryHistoryEntry(BaseModel):
    user_query: str
    generated_sql: str
    kpi_family: str

# Intents are looked up in a dict by build_graph's router - adding a 5th
# intent later means adding a value here, one entry node, and one dict
# entry, nothing else changes.
Intent = Literal["query", "plot", "analyze", "report"]

class AgentState(TypedDict, total=False):
    # input
    user_query: str
    intent: Intent   # defaults to "query" if missing/unrecognized - see build_graph._route_by_intent

    # conversation memory
    query_history: List[QueryHistoryEntry]
    current_kpi_family: Optional[str]

    # retrieval
    kpi_context: List[Dict[str, Any]]      # top-k matched KPI registry fragments
    enum_context: Dict[str, List[str]]     # resolved enum values relevant to the query

    # sql generation / execution
    generated_sql: Optional[str]
    execution_error: Optional[str]

    # results (never the raw dataframe - see tools/file_cache.py)
    result_id: Optional[str]
    row_count: Optional[int]
    columns: Optional[List[str]]
    preview_rows: Optional[List[Dict[str, Any]]]
    download_path: Optional[str]

    # plotting
    wants_plot: bool
    plot_spec: Optional[Dict[str, Any]]    # {chart_type, x, y, group_by}
    plot_path: Optional[str]

    # analyze (intent == "analyze") - tool results stay out of state, only
    # the final narrative lands here, same "never the raw data" pattern
    # used everywhere else in this graph.
    analysis_answer: Optional[str]

    # report (intent == "report") - report tools render straight to a
    # cached PDF, state only ever holds the pointer + metadata.
    report_id: Optional[str]
    report_path: Optional[str]
    report_type: Optional[str]
    report_error: Optional[str]

    # final output
    final_answer: Optional[str]
