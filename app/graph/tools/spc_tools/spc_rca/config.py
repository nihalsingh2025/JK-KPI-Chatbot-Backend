"""
Plumbing config for the SPC RCA pipeline (what the knowledge JSON does not hold):
which Silver table / columns each machine and parameter uses, plus a few constants.

Knowledge text (causes, actions, tolerances, pattern rules) lives in
app/knowledge/spc_chatbot_knowledge.json - not here.
"""

# ---- tables -------------------------------------------------------------------------
SILVER = "silver"
PLANT = "btp"
TIME_COL = "event_time"
ITEM_COL = "item_code"

# (machine name prefix, JSON group, Silver table, machine column) - matched on lower(machine)
MACHINE_RULES = [
    ("vmi_maxx",   "TBM",        f"{SILVER}.fct_production_data_vmi_maxx", "machine_no"),
    ("vmiexxium",  "TBM",        f"{SILVER}.fct_production_data_vmiexxium", "machine_no"),
    ("quintuplex", "QuintuPlex", f"{SILVER}.fct_dataparam",                 "machine_no"),
]
UNIFORMITY_MACHINES = {"30-0224", "30-0297", "micropoise"}          # lower-case
UNIFORMITY_RULE = ("", "Uniformity", f"{SILVER}.fct_productiondatatuo", "machine_name")

# ---- parameters: Silver column, setpoint column present, zero is a real reading ------------
def _p(name, column, setpoint=False, zero_is_real=False):
    return dict(name=name, column=column, setpoint=setpoint, zero_is_real=zero_is_real)

PARAMETERS = {p["name"].lower(): p for p in [
    _p("Tread Length", "tread_length", setpoint=True),
    _p("PLY1 Length", "ply1_length", setpoint=True),
    _p("PLY2 Length", "ply2_length", setpoint=True),
    _p("Belt 1 Length", "belt1_length", setpoint=True),
    _p("Belt 2 Length", "belt2_length", setpoint=True),
    _p("PA Length", "pa_length", setpoint=True),
    _p("Tread Width", "tread_width", setpoint=True),
    _p("Tread Weight", "tread_weight", setpoint=True),
    _p("Tread Splice", "tread_splice", zero_is_real=True),
    _p("PLY1 Splice", "ply1_splice", zero_is_real=True),
    _p("PLY2 Splice", "ply2_splice", zero_is_real=True),
    _p("PA Splice", "pa_splice", zero_is_real=True),
    _p("Belt 1 Splice", "belt1_splice", zero_is_real=True),
    _p("Belt 2 Splice", "belt2_splice", zero_is_real=True),
    _p("Conicity", "conicity", zero_is_real=True),
    _p("RFV", "rfv"),
    _p("LFV", "lfv"),
    _p("Inflation Pressure", "uniformity_inflation_press"),
]}

# ---- windows and rules -----------------------------------------------------------------
DAY_START = "07:00:00"          # production day starts 07:00
CP_WINDOW_DAYS = 30             # Cp/Cpk window ending on the Cp/Cpk date
TOP_WHEN = 3                    # worst hours / days to report
SKIP_STATUSES = {"healthy"}     # "why is it low?" on these statuses -> "it isn't low", no analysis
CENTRING_GAP_SHARE = 0.25       # (Cp - Cpk) / Cp above this = "off-centre", else "spread". The guide
                                # gives no number for "close" / "much lower"; adjust if needed.
MAX_CAUSES = 3
MAX_ACTIONS = 4
