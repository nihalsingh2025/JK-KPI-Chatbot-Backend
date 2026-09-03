# JK KPI Assistant - Backend

FastAPI wrapper around the existing LangGraph KPI chatbot. No node logic,
prompts, or config values were changed - only the checkpointer (now
Postgres-backed) and the delivery layer (FastAPI instead of Streamlit).

## Run with Docker (recommended)

```bash
cp .env .env   # fill in OPENAI_API_KEY and DATABRICKS_API_KEY
docker compose up --build
```

API docs: http://localhost:3020/docs

## Run locally without Docker

```bash
pip install -r requirements.txt
docker run -d -p 5442:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=postgres postgres:16
cp .env .env   # fill in keys, DB_URI default already matches the command above
uvicorn app.main:app --reload
```

## Endpoints

- `POST /api/v1/chat` - body `{ "query": str, "session_id": str }`. Runs the
  graph (using `session_id` as the LangGraph thread_id for memory) and
  returns the final state: answer, SQL, result_id, preview rows, plot spec, etc.
- `GET /api/v1/download/{result_id}` - full cached result as JSON.
- `GET /api/v1/download/{result_id}/csv` - full cached result as a CSV file.
- `GET /api/v1/health` - liveness check.

## Structure

```
app/
  main.py            FastAPI app + startup/shutdown (opens/closes the Postgres checkpointer once)
  dependencies.py     checkpointer + graph lifecycle helpers
  config.py            all tunables (unchanged from original + DB_URI/LOG_LEVEL)
  logging_config.py    logging setup
  api/v1/               chat.py, download.py, health.py route modules
  schemas/              pydantic request/response models
  graph/                state.py, build_graph.py, nodes/  (unchanged logic, logging added)
  tools/                 databricks_api.py, file_cache.py, plotting.py (unchanged logic)
  retrieval/             embed_store.py, enum_values.py (unchanged logic)
  kpi_registry/           unchanged YAML files
```
