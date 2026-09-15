"""
Download endpoints. The frontend gets a result_id back from /chat and
passes it here to fetch the full cached result - either as JSON (to
render a table) or as a CSV file (to actually download), reusing the
existing file_cache tool as-is.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas.chat import DownloadResponse
from app.tools.file_cache import load_result
from app.tools.report_cache import load_report_path
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/download", tags=["download"])


@router.get("/{result_id}", response_model=DownloadResponse)
def download_result_json(result_id: str):
    """Return the full cached result as JSON, e.g. to render a table client-side."""
    logger.info("download: JSON requested | result_id=%s", result_id)
    try:
        df = load_result(result_id)
    except FileNotFoundError:
        logger.warning("download: result not found | result_id=%s", result_id)
        raise HTTPException(status_code=404, detail="Result not found or has expired")

    logger.info("download: returning %d rows | result_id=%s", len(df), result_id)
    return DownloadResponse(
        result_id=result_id,
        row_count=len(df),
        columns=list(df.columns),
        rows=df.to_dict(orient="records"),
    )

@router.get("/{report_id}/pdf")
def download_report_pdf(report_id: str):
    """
    Serve a generated report PDF. The chat response only ever carries a
    report_id (see report_agent.py) - the frontend fetches the actual file
    here, same pattern as /download/{result_id}/csv for query results.
    """
    logger.info("download: PDF requested | report_id=%s", report_id)
    try:
        path = load_report_path(report_id)
    except FileNotFoundError:
        logger.warning("download: report not found | report_id=%s", report_id)
        raise HTTPException(status_code=404, detail="Report not found or has expired")

    return FileResponse(path, media_type="application/pdf", filename=f"{report_id}.pdf")
