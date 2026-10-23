"""Turn a segment finding into a search query for the ticket index.

The query says WHAT failed, in the words customers use in tickets: the payment
method, the device and app, a generic "payment failed at checkout", and the
error codes split into words ("PAYPAL_SDK_TIMEOUT" -> "paypal sdk timeout").

The country is left out on purpose. Customers rarely name their country, so as
text it would only add noise; the index uses it as a score boost on the
ticket's customer_country instead.

Which attributes go in, and the customer wording for each value, come from
RetrievalConfig (query_attributes, customer_words), so another dataset can
bring its own.
"""

from __future__ import annotations

from kpi_rca.config import WEB_APP_VERSION, RetrievalConfig
from kpi_rca.schemas import SegmentFinding

GENERIC_PHRASE = "payment failed at checkout"
_DEFAULT = RetrievalConfig()


def _customer_words(attribute: str, value: str, words: dict[str, dict[str, str]]) -> str:
    if value in words.get(attribute, {}):
        return words[attribute][value]
    if attribute == "app_version":
        return "website browser checkout" if value == WEB_APP_VERSION else f"app version {value} update"
    return value


def _error_code_words(code: str) -> str:
    return " ".join(code.replace("_", " ").lower().split())


def build_query(
    finding: SegmentFinding, error_codes: list[str] | None = None, cfg: RetrievalConfig | None = None
) -> str:
    """Search text for the tickets about `finding`. Error codes default to its top_error_codes."""
    cfg = cfg or _DEFAULT
    values = {a.attribute: a.value for a in finding.attributes}
    words = [_customer_words(a, values[a], cfg.customer_words) for a in cfg.query_attributes if a in values]
    codes = list(finding.top_error_codes) if error_codes is None else error_codes
    words += [GENERIC_PHRASE, *(_error_code_words(code) for code in codes)]
    return " ".join(words)
