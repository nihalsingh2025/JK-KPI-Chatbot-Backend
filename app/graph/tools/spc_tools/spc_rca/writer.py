"""
Turns the RCA facts into the final answer.

The LLM only (a) writes the short "what has happened" paragraph from the facts and (b) picks and
orders causes/actions BY NUMBER from the JSON page. Code then inserts the JSON text verbatim, so
no cause or action can be invented.
"""

import json
import logging
from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import FAST_MODEL, OPENAI_API_KEY
from app.graph.tools.spc_tools.spc_rca import config as C

logger = logging.getLogger(__name__)


class RcaWriteUp(BaseModel):
    what_happened: str = Field(description="2-4 plain sentences built only from the facts. No causes here.")
    cause_indexes: List[int] = Field(description="Indexes of up to 3 candidate causes, best match to the data first.")
    action_indexes: List[int] = Field(description="Indexes of up to 4 candidate actions, best match to the data first.")


SYSTEM_PROMPT = """You write the 'what has happened' part of a root-cause answer for a tyre plant, and choose
which candidate causes and actions to show.

Rules:
- Use only the numbers and text in the facts. Never invent or round numbers differently.
- what_happened: 2-4 plain sentences. Start with the capability note if there is one (Cp/Cpk flow), then say
  how many pieces were outside the limits (above / below), the limits used with their unit, the worst
  times, and a setpoint change if there was one. Mention blank readings first if no_reading is above 0.
  If the limits are provisional, say so. Do not give causes here.
- cause_indexes / action_indexes: choose from the candidates by index only. Put first the ones that match the
  data hints (for example setpoint_changed -> setpoint causes, no_reading -> blank-reading causes).
- Write only the answer content in business language. Never mention facts, candidates, indexes, flags, JSON,
  the knowledge file, or how the answer was produced."""


def _facts_for_llm(ctx: dict, causes: list, actions: list) -> dict:
    c, lim, g = ctx["counts"], ctx["limits"], ctx.get("gold") or {}
    hints = []
    if c["setpoint_changed"]:
        hints.append("setpoint_changed")
    if c["no_reading"] > 0:
        hints.append("no_reading")
    return {
        "question_type": "Cp/Cpk low" if ctx["flow"] == "cpk_low" else "crossing the limit",
        "parameter": ctx["parameter"], "unit": ctx["unit"], "machine": ctx["machine_label"], "itemcode": ctx["item_label"],
        "capability": {k: g.get(k) for k in ("date", "cp", "cp_status", "cpk", "cpk_status", "capability_note") if g.get(k) is not None},
        "period": ctx["window"]["label"],
        "limits": {"source": lim["source"], "low": lim["low"], "high": lim["high"], "provisional": lim["provisional"]},
        "counts": c, "pattern": ctx["pattern"], "worst_times": ctx["when"],
        "pattern_meaning": [b[1].get("has_happened") for b in ctx["blocks"]],
        "notes": ctx["notes"], "page_note": ctx["page"].get("note"), "data_hints": hints,
        "candidate_causes": [{"index": i, "text": t} for i, (t, _) in enumerate(causes)],
        "candidate_actions": [{"index": i, "text": a["check"], "who": a["who"]} for i, a in enumerate(actions)],
    }


def _pick(indexes, size, limit):
    seen, out = set(), []
    for i in indexes or []:
        if isinstance(i, int) and 0 <= i < size and i not in seen:
            seen.add(i)
            out.append(i)
    return out[:limit]


def write_answer(ctx: dict) -> str:
    c, lim, unit = ctx["counts"], ctx["limits"], ctx["unit"]
    low = "n/a" if lim["low"] is None else f"{round(lim['low'], 2):g}"
    high = f"{round(lim['high'], 2):g}"
    span = f"{low} to {high}" if lim["low"] is not None else f"up to {high}"

    # nothing crossed and nothing blank: no causes
    if ctx["pattern"] == "normal" and not ctx["blocks"]:
        text = (f"No reading for {ctx['parameter']} of {ctx['item_label']} on {ctx['machine_label']} was outside "
                f"{span} {unit} in {ctx['window']['label']} ({c['pieces']} pieces). The data looks normal.")
        return text + "".join(f"\n\n{n}" for n in ctx["notes"])

    causes = [(t, name) for name, blk in ctx["blocks"] for t in blk.get("can_happen_because", [])]
    actions = [a for _, blk in ctx["blocks"] for a in blk.get("what_to_do", [])]
    facts = _facts_for_llm(ctx, causes, actions)

    what, cause_idx, action_idx = None, [], []
    try:
        llm = ChatOpenAI(model=FAST_MODEL, api_key=OPENAI_API_KEY, temperature=0).with_structured_output(RcaWriteUp)
        out = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(facts, indent=2, default=str)},
        ])
        what, cause_idx, action_idx = out.what_happened.strip(), out.cause_indexes, out.action_indexes
        logger.info("spc_rca: writer | causes=%s actions=%s", cause_idx, action_idx)
    except Exception:
        logger.exception("spc_rca: writer failed, using fallback text")

    cause_idx = _pick(cause_idx, len(causes), C.MAX_CAUSES) or list(range(min(C.MAX_CAUSES, len(causes))))
    action_idx = _pick(action_idx, len(actions), C.MAX_ACTIONS) or list(range(min(C.MAX_ACTIONS, len(actions))))
    if not what:
        first = ctx["blocks"][-1][1].get("has_happened", "")
        what = (f"{c['outside']} of {c['pieces']} pieces were outside {span} {unit} in {ctx['window']['label']}, "
                f"{c['above']} above and {c['below']} below. {first}")

    lines = ["**What has happened**", what, "", "**This can happen because** (possible, please verify)"]
    lines += [f"- {causes[i][0]}" for i in cause_idx]
    lines += ["", "**What to do**"]
    lines += [f"{n}. {actions[i]['check']} ({actions[i]['who']})" for n, i in enumerate(action_idx, 1)]
    lines += [f"\n_{n}_" for n in ctx["notes"]]
    return "\n".join(lines)
