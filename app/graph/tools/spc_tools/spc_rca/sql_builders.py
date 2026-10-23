"""
SQL templates for the RCA pipeline (the guide's queries). Every string comparison is on lower().
Values that come from the user/LLM are checked by lit() before they reach SQL.
"""

import re
from datetime import date as _date

from app.config import DATABRICKS_TABLE
from app.graph.tools.spc_tools.spc_rca import config as C

_SAFE = re.compile(r"^[A-Za-z0-9_\-\. /]+$")


def lit(value: str) -> str:
    """Lower-cased, validated text for use inside a SQL string literal."""
    v = str(value).strip().lower()
    if not v or not _SAFE.match(v):
        raise ValueError(f"Unsupported value: {value!r}")
    return v


def check_date(value: str) -> str:
    return _date.fromisoformat(str(value)[:10]).isoformat()


def ts(stamp: str) -> str:
    return f"TIMESTAMP '{stamp} UTC'"


def num(x) -> str:
    return repr(float(x))


# ---- Gold ---------------------------------------------------------------------------
def gold_latest_date_sql(machine, item, parameter) -> str:
    return (
        f"SELECT MAX(date) AS date FROM {DATABRICKS_TABLE} "
        f"WHERE lower(plant) = '{C.PLANT}' AND lower(machine) = '{lit(machine)}' "
        f"AND lower(itemcode) = '{lit(item)}' AND lower(kpi_name) = '{lit(parameter)} cpk'"
    )


def gold_metrics_sql(machine, item, parameter, suffixes, up_to_date=None) -> str:
    """suffixes e.g. ['cp','cpk'] or ['ucl','lcl']; picks the newest date on or before up_to_date."""
    m, i, p = lit(machine), lit(item), lit(parameter)
    names = ", ".join(f"'{p} {s}'" for s in suffixes)
    anchor = f"'{p} {'cpk' if 'cpk' in suffixes else suffixes[0]}'"
    upto = f" AND date <= Date '{check_date(up_to_date)}'" if up_to_date else ""
    return (
        f"SELECT date, kpi_name, actual_value, status FROM {DATABRICKS_TABLE} "
        f"WHERE lower(plant) = '{C.PLANT}' AND lower(machine) = '{m}' AND lower(itemcode) = '{i}' "
        f"AND lower(kpi_name) IN ({names}) "
        f"AND date = (SELECT MAX(date) FROM {DATABRICKS_TABLE} "
        f"WHERE lower(machine) = '{m}' AND lower(itemcode) = '{i}' AND lower(kpi_name) = {anchor}{upto})"
    )


# ---- Silver -------------------------------------------------------------------------
def readings_cte(ctx: dict) -> str:
    """WITH readings AS (...) giving event_time, setpoint, v for the machine/item/window."""
    p, m = ctx["param_cfg"], ctx["machine_cfg"]
    col = p["column"]
    v = col if p["zero_is_real"] else f"CASE WHEN {col} > 0 THEN {col} END"
    sp = f"{col}_setpoint" if p["setpoint"] else "NULL"
    start, end = ts(ctx["window"]["start"]), ts(ctx["window"]["end"])
    machine, item = lit(ctx["machine"]), lit(ctx["itemcode"])

    if m["group"] == "QuintuPlex":
        return (
            f"WITH readings AS (\n"
            f"  SELECT {C.TIME_COL}, {sp} AS setpoint, {v} AS v\n"
            f"  FROM (\n"
            f"    SELECT *, LAG(skiver_total_cuts) OVER (ORDER BY {C.TIME_COL}, skiver_total_cuts) AS prev_cut\n"
            f"    FROM {m['table']}\n"
            f"    WHERE lower({m['machine_col']}) = '{machine}' AND skiver_total_cuts IS NOT NULL\n"
            f"      AND {C.TIME_COL} >= {start} AND {C.TIME_COL} < {end}\n"
            f"  ) q\n"
            f"  WHERE (prev_cut IS NULL OR skiver_total_cuts <> prev_cut)\n"
            f"    AND lower({C.ITEM_COL}) = '{item}'\n"
            f")"
        )
    return (
        f"WITH readings AS (\n"
        f"  SELECT {C.TIME_COL}, {sp} AS setpoint, {v} AS v\n"
        f"  FROM {m['table']}\n"
        f"  WHERE lower({m['machine_col']}) = '{machine}' AND lower({C.ITEM_COL}) = '{item}'\n"
        f"    AND {C.TIME_COL} >= {start} AND {C.TIME_COL} < {end}\n"
        f")"
    )


def setpoint_sql(ctx) -> str:
    return (
        readings_cte(ctx) + "\n"
        "SELECT setpoint, COUNT(*) AS pieces FROM readings WHERE setpoint > 0 "
        "GROUP BY setpoint ORDER BY pieces DESC, setpoint LIMIT 1"
    )


def stats_sql(ctx) -> str:
    return (
        readings_cte(ctx) + "\n"
        "SELECT AVG(v) AS average, STDDEV(v) AS sd, PERCENTILE_APPROX(v, 0.5) AS median FROM readings"
    )


def counts_sql(ctx) -> str:
    lim = ctx["limits"]
    below = f"COUNT_IF(v < {num(lim['low'])})" if lim["low"] is not None else "0"
    return (
        readings_cte(ctx) + ",\n"
        "counts AS (\n"
        "  SELECT COUNT(*) AS pieces,\n"
        "         COUNT_IF(v IS NULL) AS no_reading,\n"
        f"         COUNT_IF(v > {num(lim['high'])}) AS above,\n"
        f"         {below} AS below,\n"
        "         MIN_BY(setpoint, event_time) AS setpoint_start,\n"
        "         MAX_BY(setpoint, event_time) AS setpoint_end\n"
        "  FROM readings\n"
        ")\n"
        "SELECT * FROM counts"
    )


def when_sql(ctx) -> str:
    lim = ctx["limits"]
    bucket = (
        "date_trunc('HOUR', event_time)" if ctx["window"]["kind"] == "day"
        else "to_date(event_time - INTERVAL 7 HOURS)"
    )
    cond = f"v > {num(lim['high'])}"
    if lim["low"] is not None:
        cond += f" OR v < {num(lim['low'])}"
    return (
        readings_cte(ctx) + "\n"
        f"SELECT {bucket} AS bucket, COUNT(*) AS readings_outside FROM readings "
        f"WHERE {cond} GROUP BY 1 ORDER BY readings_outside DESC LIMIT {C.TOP_WHEN}"
    )
