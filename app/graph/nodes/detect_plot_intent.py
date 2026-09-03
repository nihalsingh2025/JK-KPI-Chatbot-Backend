"""
Node: detect_plot_intent

Asks the LLM a single yes/no question - does this question want a chart -
and stores the result in state["wants_plot"]. Replaces the old keyword
list, which missed phrasings like "trend" or "breakdown by".
"""

import logging
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import FAST_MODEL, OPENAI_API_KEY
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


class PlotIntent(BaseModel):
    wants_plot: bool


_llm = ChatOpenAI(model=FAST_MODEL, api_key=OPENAI_API_KEY, temperature=0).with_structured_output(PlotIntent)

SYSTEM_PROMPT = """
Decide whether the user's question is asking for a
visual chart/graph (e.g. "show me a graph", "plot this", "visualize",
"compare X and Y", "trend over time") as opposed to a plain text/data
answer. Respond with wants_plot: true only if a chart would genuinely
help answer this specific question. 
A simple "show me" doesn't means user wants to see a plot. Decide the current intent if user wants to plot something. see the trend then only answer. 
"""

def detect_plot_intent(state: AgentState) -> AgentState:
    logger.info("detect_plot_intent: start")
    result = _llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["user_query"]},
    ])
    state["wants_plot"] = result.wants_plot
    logger.info("detect_plot_intent: done | wants_plot=%s", result.wants_plot)
    return state
