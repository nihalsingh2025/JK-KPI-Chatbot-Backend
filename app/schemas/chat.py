"""
Request/response contracts for the API layer. These mirror AgentState but
are explicit and validated, and deliberately leave out server-internal
fields (e.g. the on-disk download_path) that the frontend has no use for.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's natural language question")
    session_id: str = Field(..., description="Conversation/thread id, used as the LangGraph checkpointer thread_id")


class ChatResponse(BaseModel):
    session_id: str
    final_answer: Optional[str] = None
    execution_error: Optional[str] = None

    current_kpi_family: Optional[str] = None
    generated_sql: Optional[str] = None

    result_id: Optional[str] = None
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    preview_rows: Optional[List[Dict[str, Any]]] = None
    is_large_result: bool = False
    download_available: bool = False

    wants_plot: bool = False
    plot_spec: Optional[Dict[str, Any]] = None


class DownloadResponse(BaseModel):
    result_id: str
    row_count: int
    columns: List[str]
    rows: List[Dict[str, Any]]
