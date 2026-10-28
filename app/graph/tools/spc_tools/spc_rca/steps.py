"""
RCA steps. Each step takes the shared `ctx` dict, adds what it finds, and may set ctx["stop"] to a
message to end the flow early. No step passes raw readings on - only counts, limits and text.
"""

import logging
import re
from datetime import date, datetime, timedelta

from app.graph.tools.spc_tools.spc_rca import config as C
from app.graph.tools.spc_tools.spc_rca import sql_builders as S
from app.graph.tools.spc_tools.spc_rca.knowledge import find_page, rules
from app.tools.databricks_api import run_query, DatabricksQueryError

logger = logging.getLogger(__name__)


# ---- small helpers ---------------------------------------------------------------------
def _norm(name: str) -> str:
    n = re.sub(r"\s+", " ", str(name).replace("_", " ")).strip().lower()
    return re.sub(r"\s+(cp|cpk|pp|ppk|ucl|lcl)$", "", n)   # drop a trailing metric word


def _f(x):
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


def _fmt(x) -> str:
    return "n/a" if x is None else f"{round(float(x), 2):g}"


def _d(value) -> str:
    return str(value)[:10]


def _pretty(day: str) -> str:
    return datetime.strptime(day, "%Y-%m-%d").strftime("%d %b %Y")


def _query(ctx, label: str, sql: str) -> list:
    try:
        rows = run_query(sql)
    except DatabricksQueryError as exc:
        if "No tabular data" not in str(exc):      # real API failures still raise
            raise
        rows = []                                   # the API returns no table for zero rows
    logger.info("spc_rca: query=%s | rows=%d", label, len(rows))
    return rows

def _stop(ctx, message: str, why: str):
    ctx["stop"] = message
    logger.info("spc_rca: STOP | reason=%s", why)


# ---- steps -------------------------------------------------------------------------------
def resolve(ctx):
    """Validate parameter and machine, find the Silver table, the JSON page and the tolerance."""
    p = C.PARAMETERS.get(_norm(ctx["parameter"]))
    if p is None:
        return _stop(ctx, f"I don't have a root-cause guide for the parameter '{ctx['parameter']}'.", "unknown parameter")
    machine = S.lit(ctx["machine"])

    rule = next((r for r in C.MACHINE_RULES if machine.startswith(r[0])), None)
    if rule is None and machine in C.UNIFORMITY_MACHINES:
        rule = C.UNIFORMITY_RULE
    if rule is None:
        return _stop(ctx, f"I don't have a root-cause guide for the machine '{ctx['machine_label']}'.", "unknown machine")
    _, group, table, machine_col = rule

    page = find_page(group, p["name"])
    if page is None:
        return _stop(ctx, f"{p['name']} is not covered for {ctx['machine_label']}.", "no knowledge page")

    ctx.update(
        machine_label=str(ctx["machine"]).strip(), item_label=str(ctx["itemcode"]).strip(),   # for messages
        param_cfg=p, parameter=p["name"], machine=machine, itemcode=S.lit(ctx["itemcode"]),   # lower-case, for SQL
        machine_cfg=dict(group=group, table=table, machine_col=machine_col),
        page=page, tolerance=page.get("tolerance") or {}, unit=page.get("unit", ""),
    )
    if ctx.get("date"):
        ctx["date"] = S.check_date(ctx["date"])

    # RFV / LFV have an upper limit only, so there is no Cp
    if ctx["flow"] == "cpk_low" and ctx["metric"] == "cp" and ctx["tolerance"].get("type") == "upper_limit":
        ctx["metric"] = "cpk"
        ctx["notes"].append(f"{p['name']} has only an upper limit, so there is no Cp. Cpk is explained instead.")
    logger.info("spc_rca: resolved | group=%s table=%s tolerance=%s", group, table, ctx["tolerance"].get("type"))


def gold_capability(ctx):
    """Query 1: Cp and Cpk (latest date on or before ctx['date'] if given)."""
    rows = _query(ctx, "gold_capability",
                  S.gold_metrics_sql(ctx["machine"], ctx["itemcode"], ctx["parameter"], ["cp", "cpk"], ctx.get("date")))
    if not rows:
        return _stop(ctx, f"There are no Cp/Cpk values for {ctx['parameter']} of {ctx['item_label']} on {ctx['machine_label']}.",
                     "no gold rows")
    got = {str(r.get("kpi_name")).strip().lower(): r for r in rows}
    cp = got.get(f"{ctx['parameter'].lower()} cp", {})
    cpk = got.get(f"{ctx['parameter'].lower()} cpk", {})
    ctx["gold"] = dict(
        date=_d(rows[0].get("date")),
        cp=_f(cp.get("actual_value")), cp_status=str(cp.get("status") or "").lower(),
        cpk=_f(cpk.get("actual_value")), cpk_status=str(cpk.get("status") or "").lower(),
    )
    g = ctx["gold"]
    if g["cp"] is not None and g["cpk"] is not None and g["cp"] != 0:
        gap = (g["cp"] - g["cpk"]) / abs(g["cp"])
        g["capability_note"] = (
            "Cp and Cpk are close, so the spread is the main problem." if gap <= C.CENTRING_GAP_SHARE
            else "Cpk is much lower than Cp, so the process is off-centre."
        )
    logger.info("spc_rca: gold | %s", {k: v for k, v in g.items() if k != "capability_note"})


def check_not_healthy(ctx):
    g, metric = ctx["gold"], ctx["metric"]
    value, status = g[metric], g[f"{metric}_status"]
    label = f"{ctx['parameter']} {metric.capitalize()}"
    if status == "insufficient data" or value is None:
        return _stop(ctx, f"There is not enough data to calculate {label} for {ctx['item_label']} on {ctx['machine_label']} "
                          f"as of {_pretty(g['date'])}.", "insufficient data")
    if status in C.SKIP_STATUSES:
        return _stop(ctx, f"{label} for {ctx['item_label']} on {ctx['machine_label']} is {_fmt(value)} ({status.title()}) "
                          f"as of {_pretty(g['date'])}, so it isn't low.", f"status={status}")
    logger.info("spc_rca: check passed | %s=%s status=%s", metric, value, status)


def ensure_date(ctx):
    """Day flow needs a date; if the user gave none use the latest date of this stream in Gold."""
    if ctx.get("date"):
        return
    rows = _query(ctx, "gold_latest_date", S.gold_latest_date_sql(ctx["machine"], ctx["itemcode"], ctx["parameter"]))
    latest = _d(rows[0].get("date")) if rows and rows[0].get("date") else None
    if not latest:
        return _stop(ctx, f"There is no SPC data for {ctx['parameter']} of {ctx['item_label']} on {ctx['machine_label']}.",
                     "no latest date")
    ctx["date"] = latest


def window_30d(ctx):
    end_day = date.fromisoformat(ctx["gold"]["date"])
    start_day = end_day - timedelta(days=C.CP_WINDOW_DAYS - 1)
    ctx["window"] = dict(
        kind="30d", days=C.CP_WINDOW_DAYS,
        start=f"{start_day.isoformat()} {C.DAY_START}", end=f"{(end_day + timedelta(days=1)).isoformat()} {C.DAY_START}",
        label=f"the {C.CP_WINDOW_DAYS} days to {_pretty(end_day.isoformat())}",
    )
    logger.info("spc_rca: window | %s -> %s", ctx["window"]["start"], ctx["window"]["end"])


def window_day(ctx):
    day = date.fromisoformat(ctx["date"])
    ctx["window"] = dict(
        kind="day", days=1,
        start=f"{day.isoformat()} {C.DAY_START}", end=f"{(day + timedelta(days=1)).isoformat()} {C.DAY_START}",
        label=_pretty(day.isoformat()),
    )
    logger.info("spc_rca: window | %s -> %s", ctx["window"]["start"], ctx["window"]["end"])


# ---- limits -------------------------------------------------------------------------------
def _limits_control(ctx):
    """Query 2: UCL/LCL, then widen (single pieces vary more than the average of 5)."""
    rows = _query(ctx, "gold_control_limits",
                  S.gold_metrics_sql(ctx["machine"], ctx["itemcode"], ctx["parameter"], ["ucl", "lcl"], ctx["date"]))
    got = {str(r.get("kpi_name")).strip().lower(): _f(r.get("actual_value")) for r in rows}
    ucl, lcl = got.get(f"{ctx['parameter'].lower()} ucl"), got.get(f"{ctx['parameter'].lower()} lcl")
    if ucl is None or lcl is None:
        return _stop(ctx, f"I couldn't find control limits for {ctx['parameter']} of {ctx['item_label']} on "
                          f"{ctx['machine_label']} on or before {_pretty(ctx['date'])}.", "no control limits")
    centre, width = (ucl + lcl) / 2, ucl - lcl
    factor = rules()["control_widen_factor"]
    ctx["limits"] = dict(source="control", low=centre - factor * width, high=centre + factor * width,
                         ucl=ucl, lcl=lcl, provisional=False)


def _limits_spec(ctx):
    """Spec limits from the JSON tolerance, same as the KPI job."""
    tol, kind = ctx["tolerance"], ctx["tolerance"].get("type")
    lim = dict(source="spec", low=None, high=None, provisional=False)

    if kind in ("absolute", "relative"):
        rows = _query(ctx, "silver_setpoint", S.setpoint_sql(ctx))
        sp = _f(rows[0].get("setpoint")) if rows else None
        if sp is None:
            return _stop(ctx, "No setpoint was recorded in this period, so I can't check this parameter against its "
                              "spec. You could ask about the control limits instead.", "no setpoint")
        v = float(tol["value"])
        lim.update(low=sp - v, high=sp + v) if kind == "absolute" else lim.update(low=sp * (1 - v), high=sp * (1 + v))
        lim["setpoint"] = sp
    elif kind == "range":
        lim.update(low=float(tol["low"]), high=float(tol["high"]))
    elif kind == "upper_limit":
        lim.update(high=float(tol["value"]))
    elif kind in ("from_data", "median_plus_minus"):
        rows = _query(ctx, "silver_stats", S.stats_sql(ctx))
        r = rows[0] if rows else {}
        avg, sd, med = _f(r.get("average")), _f(r.get("sd")), _f(r.get("median"))
        if kind == "from_data":
            if avg is None or sd is None:
                return _stop(ctx, "There are no readings in this period to work out the limits.", "no readings for stats")
            lim.update(low=avg - 3 * sd, high=avg + 3 * sd)
        else:
            if med is None:
                return _stop(ctx, "There are no readings in this period to work out the limits.", "no readings for stats")
            lim.update(low=med - float(tol["value"]), high=med + float(tol["value"]))
        lim["provisional"] = True
        ctx["notes"].append("Cp for this parameter is provisional: its limits come from the data itself, "
                            "so it shows consistency, not a customer spec.")
    else:
        return _stop(ctx, f"I don't know how to build the limits for {ctx['parameter']}.", f"unknown tolerance {kind}")
    ctx["limits"] = lim


def limits(ctx):
    (_limits_control if ctx["limit_type"] == "control" else _limits_spec)(ctx)
    if not ctx.get("stop"):
        lim = ctx["limits"]
        logger.info("spc_rca: limits | source=%s low=%s high=%s provisional=%s",
                    lim["source"], lim["low"], lim["high"], lim["provisional"])


# ---- counting, when, pattern, page ------------------------------------------------------------
def counts(ctx):
    rows = _query(ctx, "silver_counts", S.counts_sql(ctx))
    r = rows[0] if rows else {}
    pieces = int(_f(r.get("pieces")) or 0)
    if pieces == 0:
        return _stop(ctx, f"There is no data for {ctx['machine_label']}, {ctx['item_label']} in {ctx['window']['label']}.",
                     "pieces=0")
    sp_start, sp_end = _f(r.get("setpoint_start")), _f(r.get("setpoint_end"))
    ctx["counts"] = dict(
        pieces=pieces, no_reading=int(_f(r.get("no_reading")) or 0),
        above=int(_f(r.get("above")) or 0), below=int(_f(r.get("below")) or 0),
        setpoint_start=sp_start, setpoint_end=sp_end,
        setpoint_changed=sp_start is not None and sp_end is not None and sp_start != sp_end,
    )
    logger.info("spc_rca: counts | %s", ctx["counts"])


def when(ctx):
    c = ctx["counts"]
    ctx["when"] = []
    if c["above"] + c["below"] == 0:
        return
    rows = _query(ctx, "silver_when", S.when_sql(ctx))
    fmt = "%d %b %H:00" if ctx["window"]["kind"] == "day" else "%d %b"
    for r in rows:
        b = str(r.get("bucket"))
        try:
            label = datetime.fromisoformat(b.replace("Z", "")[:19]).strftime(fmt)
        except ValueError:
            label = b
        ctx["when"].append(dict(label=label, outside=int(_f(r.get("readings_outside")) or 0)))
    logger.info("spc_rca: when | %s", ctx["when"])


def pattern(ctx):
    """normal / spikes / off_target / jumping, from the counts and the JSON constants."""
    c, rl = ctx["counts"], rules()
    outside = c["above"] + c["below"]
    if outside == 0:
        pat = "normal"
    elif outside <= max(rl["spikes_max_count"], rl["spikes_max_share"] * c["pieces"]):
        pat = "spikes"
    elif max(c["above"], c["below"]) >= rl["off_target_one_side_share"] * outside:
        pat = "off_target"
    else:
        pat = "jumping"
    ctx["pattern"] = pat
    c["outside"] = outside
    logger.info("spc_rca: pattern=%s | pieces=%d outside=%d above=%d below=%d no_reading=%d",
                pat, c["pieces"], outside, c["above"], c["below"], c["no_reading"])


def page(ctx):
    """Pick the JSON blocks to use: the pattern block, plus the no_reading block if any reading was blank."""
    pg, c = ctx["page"], ctx["counts"]
    blocks = []
    if c["no_reading"] > 0 and pg.get("no_reading"):
        blocks.append(("no_reading", pg["no_reading"]))
    if ctx["pattern"] != "normal" and pg.get(ctx["pattern"]):
        blocks.append((ctx["pattern"], pg[ctx["pattern"]]))
    ctx["blocks"] = blocks
    logger.info("spc_rca: page | blocks=%s", [b[0] for b in blocks])


# ---- flows -------------------------------------------------------------------------------------
FLOWS = {
    "cpk_low": dict(
        limit_type="spec",
        steps=[resolve, gold_capability, check_not_healthy, window_30d, limits, counts, when, pattern, page],
    ),
    "crossing_limit": dict(
        limit_type="control",
        steps=[resolve, ensure_date, window_day, limits, counts, when, pattern, page],
    ),
}
