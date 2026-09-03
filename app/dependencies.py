"""
Holds the process-lifetime objects (Postgres checkpointer + compiled graph)
and exposes small helpers so routes don't need to know how they were built.

The checkpointer connection is opened once at app startup and closed once
at shutdown (see app/main.py's lifespan) - not per-request - so requests
just reuse the pool.
"""

import logging
from fastapi import Request

from langgraph.checkpoint.postgres import PostgresSaver

from app.config import DB_URI
from app.graph.build_graph import build_graph

logger = logging.getLogger(__name__)

# module-level handles for the context manager so it can be closed cleanly
_checkpointer_cm = None


def init_checkpointer_and_graph():
    """Open the Postgres checkpointer connection, run setup(), and compile the graph."""
    global _checkpointer_cm

    logger.info("dependencies: connecting to Postgres checkpointer at %s", DB_URI)
    _checkpointer_cm = PostgresSaver.from_conn_string(DB_URI)
    checkpointer = _checkpointer_cm.__enter__()

    logger.info("dependencies: running checkpointer.setup() (idempotent)")
    checkpointer.setup()

    graph = build_graph(checkpointer)
    logger.info("dependencies: checkpointer + graph ready")
    return graph


def close_checkpointer():
    """Close the Postgres checkpointer connection cleanly on shutdown."""
    global _checkpointer_cm
    if _checkpointer_cm is not None:
        logger.info("dependencies: closing Postgres checkpointer")
        _checkpointer_cm.__exit__(None, None, None)
        _checkpointer_cm = None


def get_graph(request: Request):
    """FastAPI dependency - returns the single compiled graph for this process."""
    return request.app.state.graph
