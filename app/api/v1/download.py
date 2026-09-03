"""
Download endpoints. The frontend gets a result_id back from /chat and
passes it here to fetch the full cached result - either as JSON (to
render a table) or as a CSV file (to actually download), reusing the
existing file_cache tool as-is.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.schemas.chat import DownloadResponse
from app.tools.file_cache import load_result
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