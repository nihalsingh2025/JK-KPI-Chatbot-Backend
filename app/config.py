"""
Central configuration for the KPI chatbot.
All tunable constants live here so nothing is hardcoded inside node logic.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# --- Databricks REST API (wraps the existing script) ---
DATABRICKS_API_URL = "https://c7hzxcvdxh.execute-api.ap-south-1.amazonaws.com/query"
DATABRICKS_API_KEY = os.getenv("DATABRICKS_API_KEY", "")
DATABRICKS_ENGINE = "athena"
DATABRICKS_TABLE = "gold.gold"
DATABRICKS_TABLE_RECENT = "gold.gold_daily"

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SQL_MODEL = "gpt-5.6-luna"
FAST_MODEL = "gpt-5-nano"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Result size routing ---
# Above this row count, results are cached to a file and only a preview
# plus a download link is shown to the user / passed to the LLM.
LARGE_RESULT_ROW_THRESHOLD = 50
PREVIEW_ROW_COUNT = 50

# --- KPI retrieval ---
KPI_REGISTRY_DIR = os.path.join(os.path.dirname(__file__), "kpi_registry")
TOP_K_KPI_MATCHES = 2

# --- File cache ---
RESULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".result_cache")
os.makedirs(RESULT_CACHE_DIR, exist_ok=True)

# --- Report cache (generated PDFs) ---
REPORT_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".report_cache")
os.makedirs(REPORT_CACHE_DIR, exist_ok=True)
MAX_CACHED_REPORTS = 20

# --- Analyze node ---
# Bounds the tool-calling loop in analyze_agent so a confused model can't
# loop indefinitely - after this many rounds we return whatever the model
# has said so far instead of continuing.
ANALYZE_MAX_TOOL_ITERATIONS = 4

# --- Postgres checkpointer (production) ---
# Local default matches the host port mapped in docker-compose.yml (5442).
# Inside docker-compose, the backend service overrides this via env var to
# point at the "postgres" service on its internal port 5432.
DB_URI = os.getenv(
    "DB_URI",
    "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable",
)

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
