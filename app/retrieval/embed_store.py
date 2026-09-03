"""
Builds and queries a small embedding index over the KPI registry YAML files.
Each KPI family (one YAML file) becomes one embedded document made from its
description plus example questions. At query time we embed the user's
question and return the top-k closest KPI family entries, which are then
injected into the SQL generation prompt.

This intentionally avoids a separate LLM call for routing - a single
embedding call is cheap and fast compared to an extra chat completion, and
is expected to be sufficient at the current registry size. A keyword/regex
pre-filter can be added later as a hybrid-retrieval boost if testing shows
embedding-only retrieval missing short or oddly phrased questions - not
needed for v1.
"""

import os
import glob
import logging
import yaml
import numpy as np
from openai import OpenAI

from app.config import KPI_REGISTRY_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, TOP_K_KPI_MATCHES

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)


def _load_registry() -> list:
    """Load every YAML file in kpi_registry/ into a list of dicts."""
    entries = []
    for path in sorted(glob.glob(os.path.join(KPI_REGISTRY_DIR, "*.yaml"))):
        with open(path, "r") as f:
            entries.append(yaml.safe_load(f))
    return entries


def _embed_text(text: str) -> np.ndarray:
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding)


def _doc_text(entry: dict) -> str:
    """Flatten one registry entry into a single string for embedding."""
    questions = "\n".join(q["question"] for q in entry.get("example_questions", []))
    return f"{entry['description']}\n{questions}"


class KpiRetriever:
    """
    Holds the registry entries and their embeddings in memory.
    Call .build() again whenever a YAML file is added or changed.
    """

    def __init__(self):
        self.entries = []
        self.embeddings = None

    def build(self):
        logger.info("embed_store: building KPI registry index")
        self.entries = _load_registry()
        vectors = [_embed_text(_doc_text(entry)) for entry in self.entries]
        self.embeddings = np.vstack(vectors)
        logger.info("embed_store: index built | families=%d", len(self.entries))

    def retrieve(self, query: str, top_k: int = TOP_K_KPI_MATCHES) -> list:
        if self.embeddings is None:
            self.build()

        query_vec = _embed_text(query)
        sims = self.embeddings @ query_vec / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        )
        top_idx = np.argsort(sims)[::-1][:top_k]
        result = [self.entries[i] for i in top_idx]
        logger.info("embed_store: retrieved families=%s", [e.get("kpi_family") for e in result])
        return result


# module-level singleton so the index is only built once per process
retriever = KpiRetriever()
retriever.build()
