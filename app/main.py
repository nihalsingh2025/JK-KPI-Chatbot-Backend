"""
FastAPI entrypoint.

Startup:
    - configure logging
    - open one Postgres checkpointer connection for the whole process
    - compile the graph once with that checkpointer
    - stash the compiled graph on app.state so routes can reuse it

Shutdown:
    - close the Postgres checkpointer connection cleanly
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import LOG_LEVEL
from app.logging_config import setup_logging
from app.dependencies import init_checkpointer_and_graph, close_checkpointer
from app.api.v1 import chat, download, health
from fastapi.middleware.cors import CORSMiddleware

setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup: initializing checkpointer and graph")
    app.state.graph = init_checkpointer_and_graph()
    logger.info("startup: ready")
    yield
    logger.info("shutdown: closing checkpointer")
    close_checkpointer()
    logger.info("shutdown: done")


app = FastAPI(title="JK KPI Assistant API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hey, I'm your KPI assistant. I'm up and running."}

app.include_router(health.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(download.router, prefix="/api/v1")
