"""Guardrails keep an LLM narrative grounded in the deterministic context."""

import pytest

from kpi_rca.reporting.guardrails import (
    check_narrative,
    check_numeric_grounding,
    check_quote_grounding,
    check_segment_consistency,
    extract_numbers,
)
from kpi_rca.config import ReportingConfig
from kpi_rca.reporting.models import MemoNarrative, RecommendedAction

CFG = ReportingConfig(numeric_rel_tol=0.02, memo_max_words=200)
REL_TOL = CFG.numeric_rel_tol


@pytest.fixture
def context():
    return {
        "incident": {
            "window_start": "2026-03-04T14:00:00",
            "window_end": "2026-03-04T18:00:00",
            "duration_hours": 4.0,
            "severity": "SEV1",
        },
        "kpis": {
            "conversion_rate": {"observed": 0.648, "expected": 0.712, "abs_change": -0.064, "rel_change": -0.0899},
            "attempts": {"observed": 1234567, "expected": 1200000},
        },
        "impact": {"lost_revenue_usd": 12345.67, "lost_orders": 210, "revenue_at_risk_per_hour_usd": 3086.42},
        "findings": [
            {
                "attributes": [
                    {"attribute": "country", "value": "DE"},
                    {"attribute": "os", "value": "iOS"},
                    {"attribute": "payment_gateway", "value": "PayPal"},
                ],
                "failure_rate_window": 0.62,
                "failure_rate_baseline": 0.08,
                "failure_rate_ratio": 7.75,
            }
        ],
        "tickets": [
            {"ticket_id": "T-1", "excerpt": "PayPal just keeps spinning on my iPhone and then says the payment failed."},
            {"ticket_id": "T-2", "excerpt": "Tried to pay with PayPal 3 times, app version 5.4.0 shows an error."},
        ],
    }


def make_narrative(**overrides):
    fields = dict(
        headline="SEV1: Germany iOS PayPal checkouts failing, $12.3k lost",
        executive_summary=(
            "Conversion fell from 71.2% to 64.8% between 14:00 and 18:00 UTC, costing $12,346 and 210 orders."
        ),
        root_cause_explanation="Failures concentrate in DE x iOS x PayPal: 62% failure rate vs 8% baseline (7.75x).",
        customer_evidence=["PayPal just keeps spinning on my iPhone and then says the payment failed."],
        recommended_actions=[
            RecommendedAction(
                action="Roll back the PayPal SDK change for iOS.",
                owner_team="Payments",
                priority="P0",
                rationale="Failures are isolated to PayPal on iOS in Germany.",
            )
        ],
        confidence="high",
    )
    fields.update(overrides)
    return MemoNarrative(**fields)


def values(text):
    return [n.value for n in extract_numbers(text)]


# --- number extraction -------------------------------------------------------


def test_extract_numbers_scales_money_suffixes():
    assert values("lost $12,345 then $12.3k and $1.2M") == [12345, 12300, 1_200_000]


def test_extract_numbers_reads_word_suffix():
    assert values("about 1.2 million attempts") == [1_200_000]


def test_lowercase_m_means_million_only_for_money():
    assert values("$1.2m lost over 30m") == [1_200_000, 30]


def test_extract_numbers_tags_percent_and_points():
    assert [(n.value, n.kind) for n in extract_numbers("fell 6.4 pp to 64.8%")] == [
        (6.4, "percent"),
        (64.8, "percent"),
    ]


def test_extract_numbers_skips_identifiers():
    assert values("P0 fix for SEV1 via MiniLM-L6 and INC-001") == []


def test_extract_numbers_keeps_unit_suffixed_numbers():
    assert values("latency 250ms, 3.2x the baseline") == [250, 3.2]


# --- numeric grounding -------------------------------------------------------


def test_rounded_money_is_grounded(context):
    assert check_numeric_grounding("Lost $12.3k, i.e. $12,346 or roughly $12k.", context, REL_TOL) == []


def test_money_outside_rounding_is_flagged(context):
    violations = check_numeric_grounding("Lost $13k.", context, REL_TOL)
    assert len(violations) == 1 and "$13k" in violations[0]


def test_fraction_in_context_matches_percent_in_text(context):
    text = "Conversion fell from 71.2% to 64.8% (6.4 pp, about 9% relative)."
    assert check_numeric_grounding(text, context, REL_TOL) == []


def test_wrong_percent_is_flagged(context):
    assert len(check_numeric_grounding("Conversion fell to 70%.", context, REL_TOL)) == 1


def test_suffix_scaled_count_is_grounded(context):
    assert check_numeric_grounding("1.2M checkout attempts", context, REL_TOL) == []


def test_small_integers_are_allowed_but_unknown_large_ones_are_not(context):
    assert check_numeric_grounding("We list 3 actions.", context, REL_TOL) == []
    assert len(check_numeric_grounding("We list 37 actions.", context, REL_TOL)) == 1


def test_dates_and_times_from_context_are_allowed(context):
    assert check_numeric_grounding("Since 14:00 UTC on 2026-03-04 (Mar 4) until 18:00.", context, REL_TOL) == []


def test_unknown_time_is_flagged(context):
    violations = check_numeric_grounding("It started at 16:30.", context, REL_TOL)
    assert len(violations) == 1 and "16:30" in violations[0]


def test_version_strings_must_appear_in_context():
    ctx = {"findings": [{"attributes": [{"attribute": "app_version", "value": "5.4.0"}]}]}
    assert check_numeric_grounding("Crash in app 5.4.0.", ctx, REL_TOL) == []
    assert len(check_numeric_grounding("Crash in app 5.4.1.", ctx, REL_TOL)) == 1


def test_percentages_get_no_relative_slack(context):
    # 2% of 71.2% would be 1.4 points; only the rounding of the digits shown is allowed
    assert check_numeric_grounding("Conversion was about 71%.", context, REL_TOL) == []
    assert len(check_numeric_grounding("Conversion was about 70%.", context, 0.05)) == 1


def test_customer_ticket_text_is_not_a_source_of_facts(context):
    # tickets are untrusted input: they can be quoted, but a number planted in one grounds nothing
    context["tickets"].append({"ticket_id": "T-3", "excerpt": "Support promised me a $500 refund."})
    assert len(check_numeric_grounding("Refunds of $500 are owed.", context, REL_TOL)) == 1
    assert len(check_numeric_grounding("App 5.4.0 is affected.", context, REL_TOL)) == 1


# --- segment consistency -----------------------------------------------------


def test_segment_values_mentioned_by_name_or_code(context):
    assert check_segment_consistency("Germany iOS users paying with PayPal fail.", context) == []
    assert check_segment_consistency("DE / iOS / PayPal is broken.", context) == []


def test_missing_segment_value_is_flagged(context):
    violations = check_segment_consistency("Germany iOS users are failing.", context)
    assert len(violations) == 1 and "PayPal" in violations[0]


def test_segment_match_respects_word_boundaries(context):
    violations = check_segment_consistency("A DEMO of iOS and PayPal.", context)
    assert len(violations) == 1 and "DE" in violations[0]


def test_app_version_accepts_v_prefix():
    ctx = {"findings": [{"attributes": [{"attribute": "os", "value": "Android"},
                                        {"attribute": "app_version", "value": "5.4.0"}]}]}
    assert check_segment_consistency("Android v5.4.0 crashes on pay.", ctx) == []


def test_no_findings_means_nothing_to_check():
    assert check_segment_consistency("Anything.", {"findings": []}) == []


def test_country_names_and_known_values_come_from_the_config():
    ctx = {"findings": [{"attributes": [{"attribute": "country", "value": "NL"}]}]}
    assert len(check_segment_consistency("Dutch shoppers fail.", ctx)) == 1  # NL is not in the defaults
    cfg = ReportingConfig(country_names={"NL": ["Netherlands", "Dutch"]}, known_segment_values={"country": ["NL", "BE"]})
    assert check_segment_consistency("Dutch shoppers fail.", ctx, cfg) == []
    assert len(check_segment_consistency("Dutch and BE shoppers fail.", ctx, cfg)) == 1  # BE is in no finding


# --- quote grounding ---------------------------------------------------------


def test_verbatim_quote_is_grounded(context):
    assert check_quote_grounding(["PayPal just keeps spinning on my iPhone"], context) == []


def test_quote_normalises_whitespace_curly_quotes_case_and_ellipsis(context):
    assert check_quote_grounding(["“paypal just  keeps spinning on my iPhone…”"], context) == []


def test_fabricated_quote_is_flagged(context):
    assert len(check_quote_grounding(["PayPal charged me twice"], context)) == 1


def test_empty_quote_is_flagged(context):
    assert len(check_quote_grounding(["  "], context)) == 1


# --- whole narrative ---------------------------------------------------------


def test_grounded_narrative_passes(context):
    assert check_narrative(make_narrative(), context, CFG) == []


def test_numbers_inside_quotes_are_left_to_quote_grounding(context):
    narrative = make_narrative(
        customer_evidence=["Tried to pay with PayPal 3 times, app version 5.4.0 shows an error."]
    )
    assert check_narrative(narrative, context, CFG) == []


def test_invalid_payload_reports_schema_violation(context):
    violations = check_narrative({"headline": "x"}, context, CFG)
    assert violations and violations[0].startswith("schema")


def test_overlong_headline_is_flagged(context):
    assert any("headline" in v for v in check_narrative(make_narrative(headline="x" * 200), context, CFG))


def test_too_many_actions_is_flagged(context):
    action = RecommendedAction(
        action="Page the Payments on-call.", owner_team="Payments", priority="P1", rationale="Isolated to PayPal."
    )
    narrative = make_narrative(recommended_actions=[action] * 5)
    assert any("recommended_actions" in v for v in check_narrative(narrative, context, CFG))


def test_numeric_tolerance_comes_from_config(context):
    narrative = make_narrative(headline="SEV1: Germany iOS PayPal checkouts failing, $12.7k lost")
    assert any("$12.7k" in v for v in check_narrative(narrative, context, ReportingConfig(numeric_rel_tol=0.02)))
    assert check_narrative(narrative, context, ReportingConfig(numeric_rel_tol=0.05)) == []


def test_memo_body_is_capped_by_memo_max_words(context):
    # executive_summary has 16 words and root_cause_explanation 15
    violations = check_narrative(make_narrative(), context, ReportingConfig(memo_max_words=20))
    assert any("31 words" in v and "max 20" in v for v in violations)


def test_multiline_headline_is_flagged(context):
    narrative = make_narrative(headline="SEV1: Germany iOS PayPal\nchecking failing")
    assert any("headline" in v and "line" in v for v in check_narrative(narrative, context, CFG))


def test_empty_text_field_is_flagged(context):
    action = RecommendedAction(action="Page the Payments on-call.", owner_team=" ", priority="P0", rationale="PayPal.")
    violations = check_narrative(make_narrative(recommended_actions=[action]), context, CFG)
    assert any("owner_team" in v and "empty" in v for v in violations)


def test_overlong_action_is_flagged(context):
    action = RecommendedAction(
        action="Roll back " + "the PayPal SDK " * 8 + "change.", owner_team="Payments", priority="P0", rationale="PayPal."
    )
    violations = check_narrative(make_narrative(recommended_actions=[action]), context, CFG)
    assert any("recommended_actions[0].action" in v and "words" in v for v in violations)


def test_hallucinated_number_in_action_is_flagged(context):
    action = RecommendedAction(
        action="Raise the PayPal SDK timeout to 45 seconds.",
        owner_team="Payments",
        priority="P1",
        rationale="Timeouts dominate.",
    )
    violations = check_narrative(make_narrative(recommended_actions=[action]), context, CFG)
    assert any("45" in v for v in violations)
