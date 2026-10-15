"""Guardrails for LLM-written incident narratives.

The LLM never computes anything: every number it writes must already be a fact
in the incident context (the deterministic JSON it was shown). Each check
returns a list of short, human-readable violations; an empty list means the
narrative may be used. The violations are also fed back to the LLM when it
retries.

  (a) schema + length limits   -> check_narrative
  (b) numeric grounding        -> check_numeric_grounding
  (c) segment consistency      -> check_segment_consistency
  (d) quote grounding          -> check_quote_grounding

Customer ticket text is untrusted input (it may even carry a prompt
injection). It can be quoted verbatim as evidence, but numbers, dates and
versions that appear only in ticket text are never facts.
"""

from __future__ import annotations

import math
import numbers
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterator

from pydantic import ValidationError

from kpi_rca.config import ReportingConfig
from kpi_rca.reporting.models import (
    ACTION_MAX_WORDS,
    HEADLINE_MAX_CHARS,
    MAX_ACTIONS,
    MAX_EVIDENCE,
    MemoNarrative,
)

TRIVIAL_INT_MAX = 10  # bare integers up to this are always allowed ("3 actions")
UNTRUSTED_KEYS = frozenset({"excerpt"})  # context keys that hold customer-written text

# Country codes used by the data generator, with the names a writer would use.
COUNTRY_NAMES: dict[str, list[str]] = {
    "US": ["United States", "USA", "U.S."],
    "DE": ["Germany", "German"],
    "GB": ["United Kingdom", "UK", "Britain", "British"],
    "FR": ["France", "French"],
    "IN": ["India", "Indian"],
    "BR": ["Brazil", "Brazilian"],
    "CA": ["Canada", "Canadian"],
    "JP": ["Japan", "Japanese"],
}

# --- number extraction -------------------------------------------------------

_NUMBER = re.compile(
    r"""
    (?<![\w.])(?<![A-Za-z]-)                    # skip identifiers (P0, SEV1, INC-001) and decimal tails
    (?P<money>US\$|\$|USD\s?)?
    (?P<int>\d{1,3}(?:,\d{3})+|\d+)
    (?:\.(?P<dec>\d+))?
    (?:\s?(?P<suffix>k|K|M|B|bn|mn|m|b)(?![A-Za-z])
       |\s(?P<word>thousand|million|billion)\b)?
    (?P<unit>\s?%
       |\s?(?:percent|pct|pp|pts|points|percentage\ points?)\b
       |\s?[x×](?![A-Za-z])
       |-fold\b
       |\stimes\b)?
    """,
    re.VERBOSE,
)
_SCALE = {"k": 3, "K": 3, "thousand": 3, "M": 6, "mn": 6, "million": 6, "B": 9, "bn": 9, "billion": 9}
_MONEY_ONLY_SCALE = {"m": 6, "b": 9}  # "30m" is minutes, "$1.2m" is a million


@dataclass(frozen=True)
class ExtractedNumber:
    raw: str
    value: float
    kind: str  # "money" | "percent" | "multiplier" | "plain"
    half_unit: float  # rounding slack implied by the digits shown: "12.3k" -> 50
    is_int: bool  # a bare integer with no scale suffix


def extract_numbers(text: str) -> list[ExtractedNumber]:
    out = []
    for m in _NUMBER.finditer(text):
        decimals = m.group("dec") or ""
        digits = Decimal(m.group("int").replace(",", "") + ("." + decimals if decimals else ""))
        scale_word = m.group("suffix") or m.group("word")
        exponent = _SCALE.get(scale_word, 0)
        if m.group("money") and scale_word in _MONEY_ONLY_SCALE:
            exponent = _MONEY_ONLY_SCALE[scale_word]
        unit = (m.group("unit") or "").strip()
        if m.group("money"):
            kind = "money"
        elif unit and unit[0] in "x×-" or unit == "times":
            kind = "multiplier"
        elif unit:
            kind = "percent"
        else:
            kind = "plain"
        out.append(
            ExtractedNumber(
                raw=m.group(0).strip(),
                value=float(digits.scaleb(exponent)),
                kind=kind,
                half_unit=float(Decimal("0.5").scaleb(exponent - len(decimals))),
                is_int=not decimals and exponent == 0,
            )
        )
    return out


# --- facts the context makes available ---------------------------------------

_ISO_DATETIME = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?)?(?!\d)")
_CLOCK = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::\d{2})?(?![\d:])")
_AM_PM = re.compile(r"(?<![\w.:])(\d{1,2})(?::(\d{2}))?\s?([ap])\.?m\b\.?", re.IGNORECASE)
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_MONTH_DAY = re.compile(_MONTH + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b")
_DAY_MONTH = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH)
_VERSION = re.compile(r"(?<![\w.])v?(\d+(?:\.\d+){2,})(?!\w)(?!\.\d)")


@dataclass
class _ContextFacts:
    numbers: list[float] = field(default_factory=list)
    dates: set[date] = field(default_factory=set)
    times: set[tuple[int, int]] = field(default_factory=set)
    versions: set[str] = field(default_factory=set)

    def add_datetime(self, dt: datetime | date) -> None:
        self.dates.add(dt.date() if isinstance(dt, datetime) else dt)
        self.numbers.append(float(dt.year))
        if isinstance(dt, datetime):
            self.times.add((dt.hour, dt.minute))

    @classmethod
    def from_context(cls, context: Any) -> _ContextFacts:
        facts = cls()
        for leaf in _leaves(context):
            if isinstance(leaf, (datetime, date)):
                facts.add_datetime(leaf)
            elif isinstance(leaf, numbers.Real) and not isinstance(leaf, bool):
                if math.isfinite(float(leaf)):
                    facts.numbers.append(abs(float(leaf)))
            elif isinstance(leaf, str):
                facts._add_string(leaf)
        return facts

    def _add_string(self, text: str) -> None:
        for m in _ISO_DATETIME.finditer(text):
            parsed = _parse_iso(m)
            if parsed is not None:
                self.add_datetime(parsed)
            text = _blank(text, m)
        for m in _VERSION.finditer(text):
            self.versions.add(m.group(1))
            text = _blank(text, m)
        for m in _CLOCK.finditer(text):
            self.times.add((int(m.group(1)), int(m.group(2))))
            text = _blank(text, m)
        self.numbers.extend(abs(n.value) for n in extract_numbers(text))


def _leaves(obj: Any) -> Iterator[Any]:
    """Every scalar in a nested dict/list, except customer text under UNTRUSTED_KEYS."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in UNTRUSTED_KEYS:
                yield from _leaves(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _leaves(value)
    else:
        yield obj


def _blank(text: str, m: re.Match) -> str:
    """Replace a matched span with spaces so later patterns skip it (offsets stay valid)."""
    return text[: m.start()] + " " * (m.end() - m.start()) + text[m.end():]


def _parse_iso(m: re.Match) -> datetime | date | None:
    year, month, day, hour, minute = m.groups()
    try:
        if hour is None:
            return date(int(year), int(month), int(day))
        return datetime(int(year), int(month), int(day), int(hour), int(minute))
    except ValueError:
        return None


# --- (b) numeric grounding ----------------------------------------------------


def check_numeric_grounding(text: str, context: Any, rel_tol: float) -> list[str]:
    """Every date, time, version and number in `text` must be a context fact.

    A number matches a fact if it is the fact rounded to the digits shown
    ("$12.3k" for 12345.67, "64.8%" for 0.648) or lies within `rel_tol` of it.
    Percentages get only the rounding slack: a relative slack on a percentage
    is a slack in points, and 2% of 71.2% is already 1.4 points.
    """
    facts = context if isinstance(context, _ContextFacts) else _ContextFacts.from_context(context)
    violations = []

    for m in _ISO_DATETIME.finditer(text):
        parsed = _parse_iso(m)
        ok = parsed is not None and (
            parsed.date() in facts.dates and (parsed.hour, parsed.minute) in facts.times
            if isinstance(parsed, datetime)
            else parsed in facts.dates
        )
        if not ok:
            violations.append(f"date '{m.group(0)}' is not a context fact")
        text = _blank(text, m)

    month_days = {(d.month, d.day) for d in facts.dates}
    for pattern, month_group, day_group in ((_MONTH_DAY, 1, 2), (_DAY_MONTH, 2, 1)):
        for m in pattern.finditer(text):
            month = _MONTHS.index(m.group(month_group)) + 1
            if (month, int(m.group(day_group))) not in month_days:
                violations.append(f"date '{m.group(0)}' is not a context fact")
            text = _blank(text, m)

    for m in _CLOCK.finditer(text):
        if (int(m.group(1)), int(m.group(2))) not in facts.times:
            violations.append(f"time '{m.group(0)}' is not a context fact")
        text = _blank(text, m)

    for m in _AM_PM.finditer(text):
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
        if (hour, int(m.group(2) or 0)) not in facts.times:
            violations.append(f"time '{m.group(0).strip()}' is not a context fact")
        text = _blank(text, m)

    for m in _VERSION.finditer(text):
        if m.group(1) not in facts.versions:
            violations.append(f"version '{m.group(0)}' is not a context fact")
        text = _blank(text, m)

    for n in extract_numbers(text):
        if not _is_grounded(n, facts.numbers, rel_tol):
            violations.append(f"number '{n.raw}' is not a context fact")
    return violations


def _is_grounded(n: ExtractedNumber, facts: list[float], rel_tol: float) -> bool:
    if n.kind == "plain" and n.is_int and n.value <= TRIVIAL_INT_MAX:
        return True
    factors = (1.0, 100.0) if n.kind == "percent" else (1.0,)  # 0.648 in context == "64.8%" in text
    slack = 0.0 if n.kind == "percent" else rel_tol
    for fact in facts:
        for f in factors:
            target = fact * f
            tolerance = max(n.half_unit, slack * target) + 1e-9 * max(1.0, target)  # + float noise
            if abs(n.value - target) <= tolerance:
                return True
    return False


# --- (c) segment consistency ----------------------------------------------------


def _aliases(attribute: str, value: str) -> list[str]:
    if attribute == "country":
        return [value, *COUNTRY_NAMES.get(value, [])]
    if attribute == "app_version":
        return [value, "v" + value]
    return [value]


def _mentions(text: str, alias: str) -> bool:
    flags = 0 if alias.isupper() and len(alias) <= 3 else re.IGNORECASE  # "DE" must not match "de-risk"
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", text, flags) is not None


def check_segment_consistency(text: str, context: dict) -> list[str]:
    """Every attribute value of the top finding must be named in the narrative."""
    findings = context.get("findings") or []
    if not findings:
        return []
    violations = []
    for attr in findings[0].get("attributes", []):
        attribute, value = attr["attribute"], str(attr["value"])
        if not any(_mentions(text, alias) for alias in _aliases(attribute, value)):
            violations.append(f"segment value '{value}' ({attribute}) of the top finding is not mentioned")
    return violations


# --- (d) quote grounding --------------------------------------------------------

_QUOTE_CHARS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "«": '"', "»": '"'})


def _normalise(text: str) -> str:
    return " ".join(text.translate(_QUOTE_CHARS).split()).casefold()


def _strip_quote(quote: str) -> str:
    previous = None
    while previous != quote:
        previous = quote
        quote = quote.strip().strip("\"'").strip()
        for ellipsis in ("...", "…"):
            quote = quote.removeprefix(ellipsis).removesuffix(ellipsis)
    return quote


def check_quote_grounding(quotes: list[str], context: dict) -> list[str]:
    """Customer evidence must be verbatim (case/whitespace-insensitive) substrings of the shown tickets."""
    sources = [_normalise(t.get("excerpt") or "") for t in context.get("tickets") or []]
    violations = []
    for i, quote in enumerate(quotes):
        q = _strip_quote(_normalise(quote))
        if not q:
            violations.append(f"customer_evidence[{i}] is empty")
        elif not any(q in source for source in sources):
            violations.append(f"customer_evidence[{i}] is not a verbatim ticket quote: '{quote[:60]}'")
    return violations


# --- (a) schema + everything ------------------------------------------------------


def _written_fields(narrative: MemoNarrative) -> list[tuple[str, str]]:
    """(name, text) of every free-text field the narrator writes itself; quotes are checked separately."""
    fields = [
        ("headline", narrative.headline),
        ("executive_summary", narrative.executive_summary),
        ("root_cause_explanation", narrative.root_cause_explanation),
    ]
    for i, action in enumerate(narrative.recommended_actions):
        fields += [
            (f"recommended_actions[{i}].action", action.action),
            (f"recommended_actions[{i}].owner_team", action.owner_team),
            (f"recommended_actions[{i}].rationale", action.rationale),
        ]
    return fields


def check_narrative(narrative: MemoNarrative | dict, context: dict, cfg: ReportingConfig) -> list[str]:
    """Run all guardrails; returns [] when the narrative is safe to publish."""
    if not isinstance(narrative, MemoNarrative):
        try:
            narrative = MemoNarrative.model_validate(narrative)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"]) or "narrative"
            return [f"schema: {where}: {first['msg']} ({exc.error_count()} error(s))"]

    fields = _written_fields(narrative)
    violations = [f"{name} is empty" for name, text in fields if not text.strip()]

    if len(narrative.headline) > HEADLINE_MAX_CHARS:
        violations.append(f"headline is {len(narrative.headline)} chars (max {HEADLINE_MAX_CHARS})")
    if len(narrative.headline.strip().splitlines()) > 1:
        violations.append("headline must be a single line")
    body_words = len(narrative.executive_summary.split()) + len(narrative.root_cause_explanation.split())
    if body_words > cfg.memo_max_words:
        violations.append(
            f"executive_summary + root_cause_explanation have {body_words} words (max {cfg.memo_max_words})"
        )
    for i, action in enumerate(narrative.recommended_actions):
        words = len(action.action.split())
        if words > ACTION_MAX_WORDS:
            violations.append(f"recommended_actions[{i}].action has {words} words (max {ACTION_MAX_WORDS})")
    if len(narrative.customer_evidence) > MAX_EVIDENCE:
        violations.append(f"customer_evidence has {len(narrative.customer_evidence)} quotes (max {MAX_EVIDENCE})")
    if not 1 <= len(narrative.recommended_actions) <= MAX_ACTIONS:
        violations.append(
            f"recommended_actions has {len(narrative.recommended_actions)} items (allowed 1-{MAX_ACTIONS})"
        )

    facts = _ContextFacts.from_context(context)
    for name, text in fields:
        violations += [f"{name}: {v}" for v in check_numeric_grounding(text, facts, cfg.numeric_rel_tol)]

    story = " ".join([narrative.headline, narrative.executive_summary, narrative.root_cause_explanation])
    violations += check_segment_consistency(story, context)
    violations += check_quote_grounding(narrative.customer_evidence, context)
    return violations
