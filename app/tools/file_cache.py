"""
Caches query results server-side so the LLM never has to see, or pay
tokens for, raw rows. Every executed query gets a result_id; the node
layer only passes row_count / columns / a small preview back into the
graph state. The full dataframe is retrieved later by result_id when the
user wants to download it or plot it.
"""

import os
import uuid
import logging
import pandas as pd
import glob

from app.config import RESULT_CACHE_DIR, PREVIEW_ROW_COUNT

logger = logging.getLogger(__name__)

MAX_CACHED_RESULTS = 5

def cache_result(rows: list) -> dict:
    """
    Store rows as a dataframe on disk and return metadata describing it.

    Returns a dict with: result_id, row_count, columns, preview_rows,
    download_path.
    """
    df = pd.DataFrame(rows)
    result_id = str(uuid.uuid4())
    path = os.path.join(RESULT_CACHE_DIR, f"{result_id}.parquet")
    df.to_parquet(path, index=False)

    _enforce_cache_limit()

    logger.info("file_cache: cached result_id=%s | rows=%d", result_id, len(df))

    return {
        "result_id": result_id,
        "row_count": len(df),
        "columns": list(df.columns),
        "preview_rows": df.head(PREVIEW_ROW_COUNT).to_dict(orient="records"),
        "download_path": path,
    }

def _enforce_cache_limit():
    """Keep only the MAX_CACHED_RESULTS most recently created results on disk."""
    parquet_files = sorted(
        glob.glob(os.path.join(RESULT_CACHE_DIR, "*.parquet")),
        key=os.path.getmtime,
        reverse=True,
    )
    for old_file in parquet_files[MAX_CACHED_RESULTS:]:
        result_id = os.path.splitext(os.path.basename(old_file))[0]
        os.remove(old_file)
        csv_file = os.path.join(RESULT_CACHE_DIR, f"{result_id}.csv")
        if os.path.exists(csv_file):
            os.remove(csv_file)
        logger.info("file_cache: evicted old cached result_id=%s", result_id)

def load_result(result_id: str) -> pd.DataFrame:
    """Load a previously cached result back into a dataframe."""
    path = os.path.join(RESULT_CACHE_DIR, f"{result_id}.parquet")
    logger.info("file_cache: loading result_id=%s", result_id)
    return pd.read_parquet(path)
