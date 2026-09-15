"""
Analysis tools for the "analyze" intent.

All three tools operate on the dataframe already cached by execute_sql
(loaded once per analyze_agent call via result_id) - never on rows sent to
the LLM. Every tool always groups by unit_of_measurement first, since a
result set can mix KPIs with different units, before applying any extra
group_by columns the model asks for. Returned results are small aggregated
tables, not the underlying rows.

build_analysis_tools() is called fresh inside analyze_agent for each
request, closing over that request's dataframe so the model never has to
name (or risk mistyping) a result_id.
"""

import logging
from typing import List, Optional

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GroupByArgs(BaseModel):
    group_by: Optional[List[str]] = Field(
        None,
        description="Extra columns to group by beyond unit_of_measurement, e.g. ['section'] or ['machine', 'kpi_name']. Leave null for a single overall figure per unit.",
    )


def _group_cols(df: pd.DataFrame, group_by: Optional[List[str]]) -> List[str]:
    cols = ["unit_of_measurement"]
    for col in group_by or []:
        if col in df.columns and col not in cols:
            cols.append(col)
    return cols


def build_analysis_tools(df: pd.DataFrame) -> List[StructuredTool]:
    """Returns [compute_max, compute_min, compute_std_dev] bound to this dataframe."""

    df["actual_value"] = pd.to_numeric(df["actual_value"], errors="coerce")

    def compute_max(group_by: Optional[List[str]] = None) -> list:
        cols = _group_cols(df, group_by)
        idx = df.groupby(cols)["actual_value"].idxmax()
        return df.loc[idx].to_dict(orient="records")

    def compute_min(group_by: Optional[List[str]] = None) -> list:
        cols = _group_cols(df, group_by)
        idx = df.groupby(cols)["actual_value"].idxmin()
        return df.loc[idx].to_dict(orient="records")

    def compute_std_dev(group_by: Optional[List[str]] = None) -> list:
        cols = _group_cols(df, group_by)
        result = df.groupby(cols)["actual_value"].std().reset_index()
        result = result.rename(columns={"actual_value": "std_dev"})
        return result.to_dict(orient="records")

    return [
        StructuredTool.from_function(
            func=compute_max,
            name="compute_max",
            description=(
                "Find the maximum actual_value, always grouped by unit_of_measurement "
                "plus any extra group_by columns. Returns the max value together with "
                "the full row it came from (date, section, machine, etc.) for each "
                "group - use this for 'which day/machine/section had the highest...' "
                "questions."
            ),
            args_schema=GroupByArgs,
        ),
        StructuredTool.from_function(
            func=compute_min,
            name="compute_min",
            description=(
                "Find the minimum actual_value, always grouped by unit_of_measurement "
                "plus any extra group_by columns. Returns the min value together with "
                "the full row it came from, for each group."
            ),
            args_schema=GroupByArgs,
        ),
        StructuredTool.from_function(
            func=compute_std_dev,
            name="compute_std_dev",
            description=(
                "Compute the standard deviation of actual_value, always grouped by "
                "unit_of_measurement plus any extra group_by columns. Use this for "
                "'how consistent/variable was X' questions. Returns std_dev per group, "
                "not a specific row."
            ),
            args_schema=GroupByArgs,
        ),
    ]
