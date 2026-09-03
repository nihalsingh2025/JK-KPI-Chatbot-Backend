"""
Deterministic chart-rendering functions. The LLM only ever chooses a
plot_spec dict {chart_type, x, y, group_by} from the schema and preview -
it never writes plotting code or sees raw values beyond the preview. This
keeps chart generation fast, cheap, and free of hallucinated data.

Uses Plotly as a default choice (interactive, embeds cleanly in Streamlit).
Swap this file if a different charting library is chosen later - nothing
else in the graph depends on which library is used here.
"""

import plotly.express as px
import pandas as pd


def bar_chart(df: pd.DataFrame, x: str, y: str, group_by: str = None):
    """Single bar chart, or grouped ('double') bar chart when group_by is set."""
    sort_cols = [group_by, x] if group_by else [x]
    df = df.sort_values(by=sort_cols)
    return px.bar(df, x=x, y=y, color=group_by, barmode="group")


def line_chart(df: pd.DataFrame, x: str, y: str, group_by: str = None):
    """Single line chart, or multi-series line chart when group_by is set."""
    sort_cols = [group_by, x] if group_by else [x]
    df = df.sort_values(by=sort_cols)
    return px.line(df, x=x, y=y, color=group_by, markers=True)

def pie_chart(df: pd.DataFrame, x: str, y: str,group_by: str = None):
    return px.pie(df, names=x, values=y)


CHART_FUNCTIONS = {
    "bar": bar_chart,
    "line": line_chart,
    "pie": pie_chart,
}


def render(chart_type: str, df: pd.DataFrame, **kwargs):
    """Dispatch to the correct chart function based on plot_spec['chart_type']."""
    if chart_type not in CHART_FUNCTIONS:
        raise ValueError(f"Unsupported chart_type: {chart_type}")
    return CHART_FUNCTIONS[chart_type](df, **kwargs)
