"""
Caches generated report PDFs server-side, the same way file_cache.py caches
query results. Report tools render a PDF to a temp path and call
save_report() to get back a report_id; the frontend later fetches the file
via GET /api/v1/download/{report_id}/pdf.
"""

import os
import glob
import uuid
import shutil
import logging

from app.config import REPORT_CACHE_DIR, MAX_CACHED_REPORTS

logger = logging.getLogger(__name__)


def save_report(tmp_pdf_path: str, report_type: str) -> dict:
    """
    Move a rendered PDF into the report cache under a fresh report_id.

    Returns a dict with: report_id, report_path, report_type.
    """
    report_id = str(uuid.uuid4())
    dest_path = os.path.join(REPORT_CACHE_DIR, f"{report_id}.pdf")
    shutil.move(tmp_pdf_path, dest_path)

    _enforce_cache_limit()

    logger.info("report_cache: saved report_id=%s | type=%s", report_id, report_type)

    return {
        "report_id": report_id,
        "report_path": dest_path,
        "report_type": report_type,
    }


def _enforce_cache_limit():
    """Keep only the MAX_CACHED_REPORTS most recently created reports on disk."""
    pdf_files = sorted(
        glob.glob(os.path.join(REPORT_CACHE_DIR, "*.pdf")),
        key=os.path.getmtime,
        reverse=True,
    )
    for old_file in pdf_files[MAX_CACHED_REPORTS:]:
        os.remove(old_file)
        logger.info("report_cache: evicted old report | file=%s", old_file)


def load_report_path(report_id: str) -> str:
    """Return the on-disk path for a cached report, or raise FileNotFoundError."""
    path = os.path.join(REPORT_CACHE_DIR, f"{report_id}.pdf")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Report {report_id} not found or has expired")
    return path
