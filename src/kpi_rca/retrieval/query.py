"""Turn a segment finding into a search query for the ticket index.

The query says WHAT failed, in the words customers use in tickets: the payment
method, the device and app, a generic "payment failed at checkout", and the
error codes split into words ("PAYPAL_SDK_TIMEOUT" -> "paypal sdk timeout").

The country is left out on purpose. Customers rarely name their country, so as
text it would only add noise; the index uses it as a score boost on the
ticket's customer_country instead.
"""

from __future__ import annotations

from kpi_rca.config import WEB_APP_VERSION
from kpi_rca.schemas import SegmentFinding

GENERIC_PHRASE = "payment failed at checkout"
QUERY_ATTRIBUTES = ("payment_gateway", "os", "device_type", "app_version")  # fixed order; no country

# Segment value -> the words customers write. A value missing here is used as it is.
_CUSTOMER_WORDS: dict[str, dict[str, str]] = {
    "payment_gateway": {
        "PayPal": "PayPal",
        "Klarna": "Klarna",
        "Stripe": "Stripe card",  # card processors: customers say "card", rarely the processor's name
        "Adyen": "Adyen card",
        "Braintree": "Braintree card",
    },
    "os": {"iOS": "iPhone iOS app", "Android": "Android phone app", "Windows": "Windows PC", "macOS": "Mac"},
    "device_type": {"mobile": "phone", "tablet": "tablet", "desktop": "computer laptop"},
}


def _customer_words(attribute: str, value: str) -> str:
    if attribute == "app_version":
        return "website browser checkout" if value == WEB_APP_VERSION else f"app version {value} update"
    return _CUSTOMER_WORDS[attribute].get(value, value)


def _error_code_words(code: str) -> str:
    return " ".join(code.replace("_", " ").lower().split())


def build_query(finding: SegmentFinding, error_codes: list[str] | None = None) -> str:
    """Search text for the tickets about `finding`. Error codes default to its top_error_codes."""
    values = {a.attribute: a.value for a in finding.attributes}
    words = [_customer_words(attr, values[attr]) for attr in QUERY_ATTRIBUTES if attr in values]
    codes = list(finding.top_error_codes) if error_codes is None else error_codes
    words += [GENERIC_PHRASE, *(_error_code_words(code) for code in codes)]
    return " ".join(words)
