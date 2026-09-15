"""
Report-generation tools for the "report" intent.

Each tool here is a complete, deterministic pipeline: fixed SQL template
(with a small number of user-tweakable args) -> run_query -> pandas
aggregation -> matplotlib chart -> reportlab PDF -> report_cache. The LLM
in report_agent only ever picks which tool to call and fills its args -
it never sees a data row and never writes SQL. This intentionally does
NOT go through retrieve_kpi_context / generate_sql / the kpi_registry:
reports are a small fixed catalog, so a hand-written template is more
reliable here than the general-purpose NL->SQL path.

Adding report #2 later means adding one more @tool function to this file
(or a new file under app/tools/) and listing it in report_agent's tool
list - nothing here needs to change.
"""

import os
import logging
import tempfile
from datetime import date, timedelta, datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import DATABRICKS_TABLE
from app.tools.databricks_api import run_query, DatabricksQueryError
from app.tools.report_cache import save_report

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Canonical spelling/casing, matches app/kpi_registry/downtime.yaml exactly -
# duplicated here on purpose (report tools intentionally don't read the
# registry, see module docstring).
DEFAULT_DOWNTIME_KPIS = [
    "No Schedule",
    "Operations/Material/Manpower",
    "Power Failure",
    "Preventive Maintenance",
    "Setup & Change",
    "Breakdown",
]


def _default_month_range() -> tuple[date, date]:
    """1st of the current month through today, used when the user gives no date range."""
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    start = today.replace(day=1)
    return start, today


def _parse_date(value: Optional[str], fallback: date) -> date:
    if not value:
        return fallback
    return date.fromisoformat(value)


def _sql_in_list(values: List[str]) -> str:
    return ", ".join(f"'{v.lower()}'" for v in values)


class DowntimeReportArgs(BaseModel):
    start_date: Optional[str] = Field(
        None, description="Report period start, inclusive, YYYY-MM-DD. Defaults to the 1st of the current month if omitted."
    )
    end_date: Optional[str] = Field(
        None, description="Report period end, inclusive, YYYY-MM-DD. Defaults to today if omitted."
    )
    include_kpis: Optional[List[str]] = Field(
        None, description="Downtime reasons to include, if the user wants fewer than all six default reasons. Leave null for the default full list."
    )
    exclude_kpis: Optional[List[str]] = Field(
        None, description="Downtime reasons to drop from the default list, e.g. ['Setup & Change']. Ignored if include_kpis is set."
    )


@tool(args_schema=DowntimeReportArgs)
def generate_downtime_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_kpis: Optional[List[str]] = None,
    exclude_kpis: Optional[List[str]] = None,
) -> dict:
    """
    Generate the downtime report PDF: a day-wise trend line chart of each
    downtime reason (No Schedule, Operations/Material/Manpower, Power
    Failure, Preventive Maintenance, Setup & Change, Breakdown), followed
    by a table of the average delay (minutes) per reason for the period,
    sorted highest first.

    Call this whenever the user asks for a downtime report, delay report,
    or breakdown report over some period. If they don't mention a date
    range, this defaults to the current month to date. If they ask to
    leave out or focus on specific downtime reasons, pass include_kpis or
    exclude_kpis - otherwise leave both null and all six default reasons
    are used.

    Returns a dict with report_id, report_path, row_count, period_label.
    On failure (no data, query error) returns a dict with an "error" key
    instead - never raises, so report_agent can relay the message directly.
    """
    logger.info(
        "generate_downtime_report: start | start_date=%s end_date=%s include=%s exclude=%s",
        start_date, end_date, include_kpis, exclude_kpis,
    )

    default_start, default_end = _default_month_range()
    period_start = _parse_date(start_date, default_start)
    period_end = _parse_date(end_date, default_end)
    if period_end < period_start:
        return {"error": f"end_date ({period_end}) is before start_date ({period_start})."}

    if include_kpis:
        kpi_list = include_kpis
    elif exclude_kpis:
        excluded_lower = {k.lower() for k in exclude_kpis}
        kpi_list = [k for k in DEFAULT_DOWNTIME_KPIS if k.lower() not in excluded_lower]
    else:
        kpi_list = DEFAULT_DOWNTIME_KPIS

    if not kpi_list:
        return {"error": "No downtime reasons left to report on after applying exclude_kpis."}

    exclusive_end = period_end + timedelta(days=1)
    sql = f"""SELECT kpi_name, date, machine, section, granularity, remarks, actual_value, unit_of_measurement
FROM {DATABRICKS_TABLE}
WHERE lower(kpi_name) IN ({_sql_in_list(kpi_list)})
  AND date >= DATE '{period_start.isoformat()}'
  AND date < DATE '{exclusive_end.isoformat()}'
  AND lower(granularity) = 'day'
ORDER BY date ASC"""

    try:
        rows = run_query(sql)
    except DatabricksQueryError as exc:
        logger.error("generate_downtime_report: query failed | %s", exc)
        return {"error": f"Could not fetch downtime data: {exc}"}

    if not rows:
        return {"error": f"No downtime data found between {period_start} and {period_end}."}

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["actual_value"] = pd.to_numeric(df["actual_value"], errors="coerce")

    period_label = f"{period_start.strftime('%d %B %Y')} - {period_end.strftime('%d %B %Y')}"

    chart_path = _render_trend_chart(df, period_label)
    table_df = (
        df.groupby("kpi_name", as_index=False)["actual_value"]
        .mean()
        .rename(columns={"kpi_name": "Delay reason", "actual_value": "Avg delay (min)"})
        .sort_values("Avg delay (min)", ascending=False)
    )
    table_df["Avg delay (min)"] = table_df["Avg delay (min)"].round(1)

    pdf_path = _render_pdf(period_label, chart_path, table_df)
    os.remove(chart_path)

    metadata = save_report(pdf_path, report_type="downtime")
    metadata["row_count"] = len(df)
    metadata["period_label"] = period_label

    logger.info("generate_downtime_report: done | report_id=%s | rows=%d", metadata["report_id"], len(df))
    return metadata


def _render_trend_chart(df: pd.DataFrame, period_label: str) -> str:
    """Day-wise sum of actual_value per downtime reason, one line per reason. Returns a PNG path."""
    pivot = df.groupby(["date", "kpi_name"])["actual_value"].sum().unstack(fill_value=0).sort_index()

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    for kpi_name in pivot.columns:
        ax.plot(pivot.index, pivot[kpi_name], marker="o", markersize=3, linewidth=1.4, label=kpi_name)

    ax.set_title(f"Downtime trend by reason - {period_label}", fontsize=10)
    ax.set_ylabel("Total delay (min)", fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis="both", labelsize=7)
    fig.autofmt_xdate(rotation=45)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.tight_layout()

    tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp_png.name, bbox_inches="tight")
    plt.close(fig)
    return tmp_png.name


def _render_pdf(period_label: str, chart_path: str, table_df: pd.DataFrame) -> str:
    """Builds the report PDF (title, chart, bordered table) and returns a temp file path."""
    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc = SimpleDocTemplate(tmp_pdf.name, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()

    story = [
        Paragraph(f"Downtime report for the period {period_label}", styles["Title"]),
        Spacer(1, 0.5 * cm),
        Image(chart_path, width=17 * cm, height=6.8 * cm),
        Spacer(1, 0.8 * cm),
    ]

    table_data = [list(table_df.columns)] + table_df.values.tolist()
    table = Table(table_data, colWidths=[10 * cm, 5 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1efe8")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)

    doc.build(story)
    return tmp_pdf.name
