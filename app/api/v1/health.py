"""
Basic liveness endpoint. Useful for docker-compose healthchecks and for
manually confirming the backend + graph came up correctly.
"""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    graph_ready = getattr(request.app.state, "graph", None) is not None
    return {"status": "ok" if graph_ready else "starting", "graph_ready": graph_ready}
