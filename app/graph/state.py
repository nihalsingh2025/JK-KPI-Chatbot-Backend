"""
AgentState is the single object passed between every LangGraph node.
Keep it flat and serializable - nothing here should hold a full dataframe,
only pointers (result_id) to cached data. This keeps token usage low and
makes the graph easy to trace and debug.
"""

from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel

class QueryHistoryEntry(BaseModel):
    user_query: str
    generated_sql: str
    kpi_family: str

class AgentState(TypedDict, total=False):
    # input
    user_query: str

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

    # final output
    final_answer: Optional[str]
