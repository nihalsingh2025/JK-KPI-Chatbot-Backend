"""
SPC root-cause analysis (RCA).

run_rca() runs one of two flows (see steps.FLOWS) and returns {"final_answer": text}.
build_rca_tool() wraps it as a tool that spc_agent can call.

Flows (same steps, different settings):
  cpk_low         why is Cp/Cpk low         30-day window, spec limits, worst 3 days
  crossing_limit  crossing the limit on day  1-day window, control limits (or spec), worst 3 hours
"""

import logging
import time
from typing import Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools.databricks_api import DatabricksQueryError
from app.graph.tools.spc_tools.spc_rca.steps import FLOWS
from app.graph.tools.spc_tools.spc_rca.writer import write_answer

logger = logging.getLogger(__name__)


def run_rca(flow: str, parameter: str, machine: str, itemcode: str,
            metric: str = "cpk", date: Optional[str] = None, limit_type: Optional[str] = None) -> dict:
    t0 = time.time()
    logger.info("spc_rca: START | flow=%s parameter=%s machine=%s itemcode=%s metric=%s date=%s limit_type=%s",
                flow, parameter, machine, itemcode, metric, date, limit_type)

    spec = FLOWS.get(flow)
    if spec is None:
        return {"final_answer": f"Unknown analysis type '{flow}'."}

    ctx = dict(flow=flow, parameter=parameter, machine=machine, itemcode=itemcode,
               metric=(metric or "cpk").lower(), date=date,
               limit_type=(limit_type or spec["limit_type"]).lower() if flow == "crossing_limit" else "spec",
               notes=[], stop=None)

    try:
        for step in spec["steps"]:
            t1 = time.time()
            step(ctx)
            logger.info("spc_rca: step=%s done | %.0f ms", step.__name__, (time.time() - t1) * 1000)
            if ctx["stop"]:
                logger.info("spc_rca: END (stopped) | %.1f s", time.time() - t0)
                return {"final_answer": ctx["stop"]}
        answer = write_answer(ctx)
    except ValueError as exc:
        logger.warning("spc_rca: bad input | %s", exc)
        return {"final_answer": f"I couldn't use that input: {exc}"}
    except DatabricksQueryError as exc:
        logger.error("spc_rca: query failed | %s", exc)
        return {"final_answer": "I couldn't fetch the data needed for this analysis. Please try again."}
    except Exception:
        logger.exception("spc_rca: unexpected error")
        return {"final_answer": "Something went wrong while analysing this. Please try again."}

    logger.info("spc_rca: END | pattern=%s | %.1f s", ctx.get("pattern"), time.time() - t0)
    return {"final_answer": answer}
