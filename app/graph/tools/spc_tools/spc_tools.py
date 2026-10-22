"""
SPC tools for the "spc" intent.

Both tools work on the dataframe already cached by execute_sql (the full result, not the 50-row
preview) and return small tables. build_spc_tools() is called inside spc_agent for each request,
closing over that request's dataframe (same pattern as analysis_tools.py).

  compare_over_time - one index (Cp/Cpk/Pp/Ppk) between the first and last date in the result,
                      per machine x recipe x parameter ("what got worse this week").
  streams_to_rows   - the table stores one row per metric; this turns it into one row per
                      machine x recipe x parameter with Cp, Cpk, Pp, Ppk, UCL, LCL side by side
                      ("is this recipe capable").

Adding a tool later: write the function inside build_spc_tools and append it to the returned list.
"""

import logging
from typing import List, Literal, Optional

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_KEY = ["machine", "itemcode", "parameter"]
_METRICS = ["Cp", "Cpk", "Pp", "Ppk", "UCL", "LCL"]
_BAND_RANK = {"healthy": 3, "warning": 2, "major": 1, "critical": 0}


class CompareArgs(BaseModel):
    metric: Literal["Cp", "Cpk", "Pp", "Ppk"] = Field(
        "Cpk", description="Which index to compare. Default Cpk unless the user names another."
    )
    direction: Literal["worse", "better"] = Field(
        "worse", description="'worse' = values that dropped, 'better' = values that rose."
    )
    top_n: int = Field(10, description="How many biggest movers to list.")


class StreamsArgs(BaseModel):
    metrics: Optional[List[Literal["Cp", "Cpk", "Pp", "Ppk", "UCL", "LCL"]]] = Field(
        None, description="Metrics to include. Leave null for every metric present in the result."
    )
    date: Optional[str] = Field(
        None, description="YYYY-MM-DD. Leave null to use the latest date in the result."
    )
    max_rows: int = Field(50, description="Maximum rows to return (worst Cpk first).")


def _r(x):
    return None if pd.isna(x) else round(float(x), 3)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds parameter + metric columns (kpi_name = '<Parameter> <Metric>') and clean dates/values."""
    d = df.copy()
    d["actual_value"] = pd.to_numeric(d["actual_value"], errors="coerce")
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    split = d["kpi_name"].str.rsplit(" ", n=1)
    d["parameter"] = split.str[0]
    metric_lower = {m.lower(): m for m in _METRICS}
    d["metric"] = split.str[1].str.lower().map(metric_lower)
    return d[d["metric"].notna()]


def build_spc_tools(df: pd.DataFrame) -> List[StructuredTool]:
    """Returns [compare_over_time, streams_to_rows] bound to this dataframe."""

    base = _prepare(df)

    def compare_over_time(metric: str = "Cpk", direction: str = "worse", top_n: int = 10) -> dict:
        d = base[base["metric"] == metric].drop_duplicates(_KEY + ["date"])
        if d.empty:
            return {"error": f"No {metric} rows in this result."}

        dates = sorted(d["date"].unique())
        if len(dates) < 2:
            return {"error": "Only one date in the data, nothing to compare.", "dates_in_data": dates}
        start, end = dates[0], dates[-1]

        cols = ["actual_value", "status"]
        a = d[d["date"] == start].set_index(_KEY)[cols]
        b = d[d["date"] == end].set_index(_KEY)[cols]
        j = a.join(b, how="outer", lsuffix="_from", rsuffix="_to")
        total = len(j)
        j = j.dropna(subset=["actual_value_from", "actual_value_to"]).reset_index()

        j["change"] = j["actual_value_to"] - j["actual_value_from"]
        rank_from = j["status_from"].str.lower().map(_BAND_RANK)
        rank_to = j["status_to"].str.lower().map(_BAND_RANK)
        band_drop = rank_to < rank_from
        band_up = rank_to > rank_from

        movers = j[j["change"] < 0] if direction == "worse" else j[j["change"] > 0]
        movers = movers.sort_values("change", ascending=(direction == "worse")).head(top_n)

        by_param = (
            j.groupby("parameter")
            .agg(compared=("change", "size"), worse=("change", lambda s: int((s < 0).sum())), avg_change=("change", "mean"))
            .reset_index()
            .sort_values("worse", ascending=False)
        )

        def rows(frame):
            return [
                {
                    "machine": r.machine, "itemcode": r.itemcode, "parameter": r.parameter,
                    "from_value": _r(r.actual_value_from), "to_value": _r(r.actual_value_to),
                    "change": _r(r.change), "from_status": r.status_from, "to_status": r.status_to,
                }
                for r in frame.itertuples()
            ]

        return {
            "metric": metric, "from_date": start, "to_date": end,
            "streams_compared": int(len(j)),
            "streams_missing_a_value": int(total - len(j)),
            "worse": int((j["change"] < 0).sum()),
            "better": int((j["change"] > 0).sum()),
            "unchanged": int((j["change"] == 0).sum()),
            "dropped_a_status_band": int(band_drop.sum()),
            "rose_a_status_band": int(band_up.sum()),
            f"top_{direction}": rows(movers),
            "band_drop_examples": rows(j[band_drop].sort_values("change").head(top_n)),
            "by_parameter": [
                {"parameter": r.parameter, "compared": int(r.compared), "worse": int(r.worse), "avg_change": _r(r.avg_change)}
                for r in by_param.itertuples()
            ],
        }

    def streams_to_rows(metrics: Optional[List[str]] = None, date: Optional[str] = None, max_rows: int = 50) -> dict:
        dates = sorted(base["date"].unique())
        if not dates:
            return {"error": "No SPC rows in this result."}
        target = date or dates[-1]
        if target not in dates:
            return {"error": f"No rows for {target}.", "dates_in_data": sorted(set(dates[:5] + dates[-5:]))}

        d = base[base["date"] == target]
        if metrics:
            d = d[d["metric"].isin(metrics)]
        d = d.drop_duplicates(_KEY + ["metric"])
        if d.empty:
            return {"error": "No rows for those metrics on that date."}

        wide = d.pivot(index=_KEY, columns="metric", values="actual_value")
        status = d.pivot(index=_KEY, columns="metric", values="status")
        for m in ("Cpk", "Ppk"):
            if m in status.columns:
                wide[f"{m}_status"] = status[m]

        wide = wide.reset_index()
        sort_col = "Cpk" if "Cpk" in wide.columns else next((m for m in _METRICS if m in wide.columns), None)
        if sort_col:
            wide = wide.sort_values(sort_col, ascending=True, na_position="last")

        records = []
        for row in wide.head(max_rows).to_dict(orient="records"):
            records.append({k: (_r(v) if isinstance(v, float) else v) for k, v in row.items()})
        return {"date": target, "total_streams": int(len(wide)), "returned": len(records), "rows": records}

    return [
        StructuredTool.from_function(
            func=compare_over_time,
            name="compare_over_time",
            description=(
                "Compare one SPC index (Cp, Cpk, Pp or Ppk) between the first and the last date in the "
                "result, for every machine x recipe x parameter. Returns how many got worse/better, how many "
                "dropped a status band, the biggest movers and a per-parameter summary. Use for 'what got "
                "worse / better / changed / improved / deteriorated' questions."
            ),
            args_schema=CompareArgs,
        ),
        StructuredTool.from_function(
            func=streams_to_rows,
            name="streams_to_rows",
            description=(
                "The data has one row per metric. This turns it into one row per machine x recipe x parameter "
                "with Cp, Cpk, Pp, Ppk (and UCL, LCL if present) side by side, plus Cpk/Ppk status, worst Cpk first. "
                "Use for capability / 'is it capable' / 'why is Cpk low' questions that need several metrics of the "
                "same stream together."
            ),
            args_schema=StreamsArgs,
        ),
    ]
