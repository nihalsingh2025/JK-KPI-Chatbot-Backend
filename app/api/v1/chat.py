"""
Chat endpoint. Runs the LangGraph agent once per call, using session_id as
the checkpointer thread_id so follow-up questions in the same session keep
conversational memory (query_history, current_kpi_family, etc. are all
persisted by the Postgres checkpointer between calls).
"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from app.config import LARGE_RESULT_ROW_THRESHOLD
from app.dependencies import get_graph
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def send_message(payload: ChatRequest, graph=Depends(get_graph)):
    logger.info(
        "chat: received | session_id=%s | intent=%s | query=%r",
        payload.session_id, payload.intent, payload.query,
    )

    config = {"configurable": {"thread_id": payload.session_id},
              "run_name": payload.query[:60],}

    try:
        result = graph.invoke(
            {"user_query": payload.query, "intent": payload.intent or "query"},
            config=config,
        )
    except Exception:
        logger.exception("chat: graph invocation failed | session_id=%s", payload.session_id)
        raise HTTPException(status_code=500, detail="Failed to process the query")

    row_count = result.get("row_count") or 0
    response = ChatResponse(
        session_id=payload.session_id,
        intent=result.get("intent"),
        final_answer=result.get("final_answer"),
        execution_error=result.get("execution_error"),
        current_kpi_family=result.get("current_kpi_family"),
        generated_sql=result.get("generated_sql"),
        result_id=result.get("result_id"),
        row_count=result.get("row_count"),
        is_large_result=row_count > LARGE_RESULT_ROW_THRESHOLD,
        download_available=bool(result.get("result_id")),
        wants_plot=bool(result.get("wants_plot")),
        plot_spec=result.get("plot_spec"),
        report_id=result.get("report_id"),
        report_type=result.get("report_type"),
        report_available=bool(result.get("report_id")),
    )

    logger.info(
        "chat: done | session_id=%s | intent=%s | result_id=%s | report_id=%s",
        payload.session_id, response.intent, response.result_id, response.report_id,
    )
    return response