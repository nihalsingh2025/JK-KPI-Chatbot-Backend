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
}

CATEGORIES = [
    "SPC", "Productivity", "Breakdown Maintenance", "Quality",
    "Process Audit", "Inventory", "Uniformity Testing",
]

GRANULARITIES = ["DAY", "MONTH", "YEAR", "SHIFT A", "SHIFT B", "SHIFT C"]

# tuned against real typo/spacing test cases - see testing/test_fuzzy_match.py
FUZZY_THRESHOLD = 70

# longest known enum value, in words - used to size the n-grams built from
# the user query (e.g. "Cap Strip Mother Roll" = 4 words)
_MAX_PHRASE_WORDS = max(
    len(value.split())
    for group in (PRODUCT_TYPES, SECTIONS)
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
        for section in matched_sections:
            if section in SUB_SECTIONS:
                context["sub_section"] = SUB_SECTIONS[section]

    matched_granularities = _fuzzy_match_all(user_query, GRANULARITIES)
    if matched_granularities:
        context["granularity"] = matched_granularities

    return context