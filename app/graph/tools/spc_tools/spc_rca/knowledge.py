"""Loads spc_chatbot_knowledge.json once and gives case-insensitive access to it."""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KNOWLEDGE_PATH = Path(__file__).resolve().parent / "knowledge" / "spc_chatbot_knowledge.json"


@lru_cache(maxsize=1)
def load_knowledge() -> dict:
    with open(KNOWLEDGE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("spc_rca: knowledge loaded | version=%s path=%s", data.get("version"), KNOWLEDGE_PATH)
    return data


def rules() -> dict:
    """crossing_rules constants (widen factor, spikes thresholds, one-side share)."""
    return load_knowledge()["crossing_rules"]


def find_page(group: str, parameter: str) -> Optional[dict]:
    """machines -> group -> parameter, matched case-insensitively."""
    machines = load_knowledge()["machines"]
    grp = next((v for k, v in machines.items() if k.lower() == group.lower()), None)
    if grp is None:
        return None
    return next((v for k, v in grp.items() if k.lower() == parameter.lower()), None)
