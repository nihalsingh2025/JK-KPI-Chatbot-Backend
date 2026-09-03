# Architecture — KPI Chatbot (LangGraph)

## Flow

```
POST /api/v1/chat {query, session_id}
        |
        v
retrieve_kpi_context   -- embed question, fetch top-k KPI registry fragments + enum values
        |
        v
generate_sql           -- LLM (gpt-4o) writes SQL using kpi_context + enum_context + query_history
        |
        v
execute_sql             -- runs SQL via Databricks REST API, caches result (parquet) -> result_id
        |
        v
format_answer            -- LLM (mini) turns row_count/columns/preview into a short NL answer
        |
        v
detect_plot_intent        -- LLM (mini) yes/no: does this question want a chart?
        |
        +--[wants_plot=false]--> END
        |
        +--[wants_plot=true]--> plot -- LLM (mini) picks chart_type/x/y/group_by -> plot_spec
                                    |
                                    v
                                   END
```

Each node reads/writes a single shared `AgentState` dict (see `graph/state.py`).
No node knows what runs before/after it — control flow lives only in `build_graph.py`.

## Memory / sessions

- `session_id` (from the API request) = LangGraph `thread_id`.
- State (`query_history`, `current_kpi_family`, etc.) is persisted per `thread_id` by a
  **Postgres-backed checkpointer** (`PostgresSaver`), opened once at app startup and reused
  across all requests — not per-request.
- Follow-up questions in the same session re-enter the graph with prior state auto-loaded.

## Result handling

- Raw query rows are **never** put back into `AgentState` or sent to the LLM.
- `execute_sql` caches the full result to disk (`tools/file_cache.py`, parquet) under a
  `result_id`, and only `row_count` / `columns` / a small `preview_rows` sample go into state.
- Frontend fetches the full result later via `GET /api/v1/download/{result_id}`.

## Models

- `generate_sql` — needs real reasoning (SQL correctness, filter/date resolution) → heavier model.
- `format_answer`, `detect_plot_intent`, `plot` — simple classification/summarization → cheap/fast model.

## Layers

| Layer | Files |
|---|---|
| API | `app/api/v1/{chat,download,health}.py` — thin, no business logic |
| Graph | `app/graph/state.py`, `build_graph.py`, `nodes/*.py` — all the actual logic |
| Retrieval | `app/retrieval/embed_store.py` (KPI registry embeddings), `enum_values.py` |
| Tools | `app/tools/databricks_api.py` (SQL execution), `file_cache.py` (result caching) |
| Config | `app/config.py` — models, thresholds, DB_URI, all tunables in one place |
| Lifecycle | `app/dependencies.py` + `app/main.py` — opens Postgres checkpointer + compiles graph once at startup, closes on shutdown |
