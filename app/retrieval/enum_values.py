"""
Static index of known enum-like column values (product_type, section,
sub_section, category). These are injected alongside the KPI context so
SQL generation uses exact strings instead of guessing spellings.

Fuzzy matching (rapidfuzz) is used instead of exact substring matching, so
misspellings and spacing differences (e.g. "sidewal", "side wall",
"inerliner") still resolve to the correct canonical value.

Update this file whenever new values appear in the gold layer.
"""

from rapidfuzz import process, fuzz, utils

PRODUCT_TYPES = [
    "Bead Apex", "Bead Bundle", "Belt", "Belt Mother Roll", "Cap Strip",
    "Cap Strip Mother Roll", "Carcass", "Chaffer", "Chaffer Mother Roll",
    "Cured Tyre", "Final Compound", "Green Tyre", "Inner Liner",
    "Master Compound", "Ply", "Ply Mother Roll", "Sidewall", "Tread",
]

SECTIONS = ["Mixing", "Stock", "Curing", "TBM"]

# sub_section currently only applies within the Stock section
SUB_SECTIONS = {
    "Stock": ["Extruder", "4 Roll", "Bead", "3 Roll"],
    "Curing": ["Trench1","Trench2","Trench3","Trench4","Trench5","Trench6","Trench7","Trench8"]
}

CATEGORIES = [
    "SPC", "Productivity", "Breakdown Maintenance", "Quality",
    "Process Audit", "Inventory", "Uniformity Testing",
]

SCRAP_REMARKS = [
    "Bare cord", "Split Cord", "VCL Not Out", "BFC/ RIM Line crack",
    "Cony off", "IL Lumpy", "Blow Point", "Thermocouple", "L/Heal",
    "PCI Damage", "No Internal", "Conv Damage", "IL Blister",
    "Press Not Close", "Reverse Hump", "Late Open", "Carcass Buckle",
    "Bent Bead", "SW Splice open", "Shaping Cut/Fluctuation", "FM Oil",
    "Arm Damage", "SW Blow", "Tread Edge blow", "NITROGEN DROP",
    "FM White Powder", "Tread Under Cure", "Belt off Centre",
    "O Ring Leak", "Spread Cord", "A.Bridging / BNT", "FM Metal",
    "Heavy Splice", "Flash cure", "No Paint / Missing Paint", "Bead Blow",
    "Flow Crack", "Carcass Under Cure", "FM Other", "WEAK SW",
    "Thick Toe Cure", "TUO Damage", "Bead Blister", "Bladder Mark",
    "Bladder Leak", "Open Splice", "BOB", "Bead Buckle", "Assembly Off",
    "Operational Failure", "SW lamination", "Light SW", "Cap Strip Off",
    "Shoulder Below", "Bead Under cure", "TUO Con. Damage", "FM Plastic",
    "FM Dust", "TREAD EDGE OPEN", "Platen Temp. Drop", "GT scrap",
    "Thick toe", "Separation", "Paint FM", "T/up blow", "FM Curing",
    "Tread Blow", "Poor Buff", "Pulled bead/Narrow bead",
    "FM Bleeder Yarn", "Vent Cure", "Undulation", "OCL", "Development",
    "Wild Wire", "Bladder Fold", "IL Wrinkle", "IL Circ Crack",
    "Tread Cushion Gauge Heavy", "Reloading", "FM Poly", "Project",
    "LR Damage", "Tread Chipping", "HP Drop", "LR Not Raise",
    "Tread Lamination", "Light Tread", "Power Fail", "Parallel Belt",
    "Knife Cut", "Press trial", "Int. Temp. Drop", "SW Under Cure",
    "DBM Off", "MP Drop", "Light S/W flow crack", "Extra Cure",
    "Dirty Mould", "N2 drop", "IL Lamination", "Platen Temp. High",
    "Late Internal", "Open Mold",
]

GRANULARITIES = ["DAY", "MONTH", "YEAR", "SHIFT A", "SHIFT B", "SHIFT C"]

# tuned against real typo/spacing test cases - see testing/test_fuzzy_match.py
FUZZY_THRESHOLD = 70

# longest known enum value, in words - used to size the n-grams built from
# the user query (e.g. "Cap Strip Mother Roll" = 4 words)
_MAX_PHRASE_WORDS = max(
    len(value.split())
    for group in (PRODUCT_TYPES, SECTIONS,SUB_SECTIONS,CATEGORIES,SCRAP_REMARKS,GRANULARITIES)
    for value in group
)


def _candidate_phrases(user_query: str) -> list:
    """
    Build every contiguous word n-gram (1 up to _MAX_PHRASE_WORDS words)
    from the query, so multi-word enum values (e.g. "Inner Liner") can
    still be fuzzy-matched even though the match target is a phrase, not
    a single word.
    """
    words = user_query.split()
    phrases = []
    for size in range(1, _MAX_PHRASE_WORDS + 1):
        for i in range(len(words) - size + 1):
            phrases.append(" ".join(words[i:i + size]))
    return phrases


def _fuzzy_match_all(user_query: str, choices: list, threshold: int = FUZZY_THRESHOLD) -> list:
    """
    Return every choice from `choices` that fuzzy-matches some phrase in
    the user query at or above the threshold. Order-preserving, deduped.
    """
    matched = []
    for phrase in _candidate_phrases(user_query):
        result = process.extractOne(
            phrase, choices, scorer=fuzz.WRatio, processor=utils.default_process
        )
        if result is None:
            continue
        match, score, _ = result
        if score >= threshold and match not in matched:
            matched.append(match)
    return matched


def relevant_enum_context(user_query: str) -> dict:
    """
    Fuzzy-match the user's question against known enum values, tolerating
    typos and spacing differences, and return only the enum groups that
    actually matched so the prompt is not bloated with irrelevant lists.
    """
    context = {}

    matched_products = _fuzzy_match_all(user_query, PRODUCT_TYPES)
    if matched_products:
        context["product_type"] = matched_products

    matched_sections = _fuzzy_match_all(user_query, SECTIONS)
    if matched_sections:
        context["section"] = matched_sections

    all_sub_sections = [v for values in SUB_SECTIONS.values() for v in values]
    matched_sub_sections = _fuzzy_match_all(user_query, all_sub_sections)
    if matched_sub_sections:
        context["sub_section"] = matched_sub_sections

    matched_granularities = _fuzzy_match_all(user_query, GRANULARITIES)
    if matched_granularities:
        context["granularity"] = matched_granularities

    matched_remarks = _fuzzy_match_all(user_query,SCRAP_REMARKS)
    if matched_remarks:
        context["scrap_remarks"] = matched_remarks

    return context