"""The narrator writes the memo text; the guardrails decide whether it may be used.

No test here touches the network: ClaudeNarrator gets a fake client that
replays canned responses (or raises the SDK's own exceptions) and records
every request it receives.
"""

import email.message
import json
import urllib.error
from datetime import datetime
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import ValidationError

from kpi_rca.config import LLMConfig, ReportingConfig
from kpi_rca.reporting.context import build_incident_context
from kpi_rca.reporting.guardrails import check_narrative
from kpi_rca.reporting.llm import (
    RATE_LIMIT_RETRIES,
    RATE_LIMIT_WAIT_S,
    ClaudeNarrator,
    NarratorError,
    OpenAICompatibleNarrator,
    TemplateNarrator,
    get_narrator,
    narrate_with_guardrails,
)
from kpi_rca.reporting.models import FinancialImpact, MemoNarrative, RecommendedAction
from kpi_rca.schemas import AnomalyWindow, RCAResult, RetrievalResult, SegmentAttribute, SegmentFinding, TicketHit

CFG = ReportingConfig()
LLM = LLMConfig()  # claude-opus-5-5, 2 retries, server-side fallback on
QUOTE = "PayPal keeps timing out on my iPhone and the order never goes through."
SECOND = "Checkout with PayPal fails every time."

# --- incident context --------------------------------------------------------------

EXPECTED = {"conversion_rate": 0.8, "revenue": 128000.0, "attempts": 3200.0}
DROP = {"conversion_rate": 0.6, "revenue": 96000.0, "attempts": 3200.0}
SURGE = {"conversion_rate": 0.79, "revenue": 182000.0, "attempts": 5760.0}

FINDING = SegmentFinding(
    rank=1,
    attributes=[
        SegmentAttribute(attribute="country", value="DE"),
        SegmentAttribute(attribute="os", value="iOS"),
        SegmentAttribute(attribute="payment_gateway", value="PayPal"),
    ],
    ep=0.91,
    purity=0.7,
    score=0.79,
    failure_rate_window=0.62,
    failure_rate_baseline=0.08,
    n_window=850,
    n_baseline=12000,
    excess_failures=460.0,
    p_value=1e-12,
    shap_contribution={"payment_gateway=PayPal": 0.4},
    top_error_codes={"PAYPAL_SDK_TIMEOUT": 400, "CARD_DECLINED": 12},
)


def make_impact(lost_revenue, severity):
    return FinancialImpact(
        currency="USD",
        lost_revenue=lost_revenue,
        lost_orders=lost_revenue / 50,
        revenue_at_risk_per_hour=lost_revenue / 3,
        duration_hours=3.0,
        conversion_observed=0.6,
        conversion_expected=0.8,
        affected_attempts=850,
        segment_failure_rate_window=0.62,
        segment_failure_rate_baseline=0.08,
        severity=severity,
    )


def make_context(findings=True, tickets=True, surge=False, texts=(QUOTE, SECOND)):
    window = AnomalyWindow(
        window_id="W-001",
        start=datetime(2026, 9, 10, 14),
        end=datetime(2026, 9, 10, 17),
        kpi_type="volume_surge" if surge else "kpi_drop",
        n_buckets=3,
        if_score_min=-0.2,
        observed=SURGE if surge else DROP,
        expected=EXPECTED,
    )
    rca = RCAResult(
        window_id="W-001",
        findings=[FINDING] if findings else [],
        feature_importance={},
        top_interactions=[],
        slices_evaluated=42,
        runtime_s=1.5,
    )
    hits = [
        TicketHit(
            rank=i,
            ticket_id=f"TK{i:06d}",
            created_at=datetime(2026, 9, 10, 14, 5 * i),
            channel="chat",
            customer_country="DE",
            device_os="iOS",
            text=text,
            similarity=0.8,
            score=0.9,
        )
        for i, text in enumerate(texts, start=1)
    ]
    retrieval = RetrievalResult(
        query="paypal sdk timeout",
        backend="lsa",
        search_start=datetime(2026, 9, 10, 12),
        search_end=datetime(2026, 9, 10, 23),
        hits=hits,
        n_candidates=40,
        similar_in_window=12,
        window_rate_per_hour=4.0,
        baseline_rate_per_hour=0.5,
        spike_ratio=8.0,
    )
    impact = make_impact(0.0, "SEV3") if surge else make_impact(33000.12, "SEV1")
    return build_incident_context(window, rca, retrieval if tickets else None, impact, CFG)


# --- what Claude might write ------------------------------------------------------------


def claude_memo(summary):
    return MemoNarrative(
        headline="SEV1: checkout conversion down 25% for DE iOS PayPal users",
        executive_summary=summary,
        root_cause_explanation="Country DE, os iOS and payment_gateway PayPal fail at 62% against 8% at baseline.",
        customer_evidence=["PayPal keeps timing out on my iPhone"],
        recommended_actions=[
            RecommendedAction(
                action="Route iOS PayPal payments in Germany to the web checkout.",
                owner_team="Payments",
                priority="P0",
                rationale="PAYPAL_SDK_TIMEOUT causes most failures.",
            )
        ],
        confidence="high",
    )


GOOD = claude_memo("Conversion fell from 80% to 60% between 2026-09-10 14:00 and 17:00 UTC, losing USD 33,000.")
INVENTED = claude_memo("Conversion fell from 80% to 60% between 2026-09-10 14:00 and 17:00 UTC, losing USD 99,999.")

# --- a fake Anthropic client ----------------------------------------------------------------


class FakeEndpoint:
    """Stands in for client.messages / client.beta.messages: replays outcomes, records requests."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def parse(self, **request):
        self.calls.append(request)
        assert self.outcomes, "unexpected API call"
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    """Stands in for anthropic.Anthropic(); the outcomes go to the endpoint the narrator should call."""

    def __init__(self, *outcomes, beta=True):
        self.messages = FakeEndpoint([] if beta else outcomes)
        self.beta = SimpleNamespace(messages=FakeEndpoint(outcomes if beta else []))

    @property
    def calls(self):
        return self.beta.messages.calls + self.messages.calls


def reply(memo, stop_reason="end_turn"):
    """What parse() returns: the SDK has already validated the JSON into a MemoNarrative."""
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=memo)


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls, code):
    return cls(f"HTTP {code}", response=httpx2.Response(code, request=REQUEST), body=None)


def schema_error():
    """parse() validates the JSON itself, so a malformed memo surfaces there as a ValidationError."""
    try:
        MemoNarrative.model_validate_json('{"headline": "cut off')
    except ValidationError as exc:
        return exc


def narrate(client, llm=LLM):
    return narrate_with_guardrails(ClaudeNarrator(llm, client=client), make_context(), llm, CFG)


# --- narrate_with_guardrails with Claude ------------------------------------------------------


def test_a_draft_that_passes_the_guardrails_is_used_as_is():
    assert check_narrative(GOOD, make_context(), CFG) == []
    narrative, report = narrate(FakeClient(reply(GOOD)))
    assert narrative == GOOD
    assert report.passed and report.violations == []
    assert report.attempts == 1 and report.attempt_violations == [[]]
    assert report.narrator_used == "claude" and not report.fallback_used and report.fallback_reason is None


def test_failed_checks_are_sent_back_and_the_corrected_draft_is_used():
    client = FakeClient(reply(INVENTED), reply(GOOD))
    narrative, report = narrate(client)
    assert narrative == GOOD
    assert report.passed and report.attempts == 2 and report.narrator_used == "claude"
    first = report.attempt_violations[0]
    assert any("USD 99,999" in v for v in first) and report.attempt_violations[1] == []
    first_prompt, retry_prompt = (call["messages"][0]["content"] for call in client.calls)
    assert all(v in retry_prompt for v in first)
    assert "USD 99,999" not in first_prompt


def test_falls_back_to_the_template_when_every_draft_fails():
    client = FakeClient(reply(INVENTED), reply(INVENTED))
    narrative, report = narrate(client, llm=LLMConfig(max_retries=1))
    assert narrative == TemplateNarrator().generate(make_context())
    assert report.narrator_used == "template_fallback" and report.fallback_used
    assert report.attempts == 2 and len(report.attempt_violations) == 2
    assert report.passed and report.violations == []  # the template itself passes
    assert "guardrails" in report.fallback_reason


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens", "model_context_window_exceeded"])
def test_a_refusal_or_cut_off_falls_back_without_retrying(stop_reason):
    client = FakeClient(reply(None, stop_reason=stop_reason))
    narrative, report = narrate(client)
    assert report.narrator_used == "template_fallback" and stop_reason in report.fallback_reason
    assert len(client.calls) == 1 and report.attempts == 1 and report.attempt_violations == []


def test_a_response_without_a_parsed_memo_falls_back():
    narrative, report = narrate(FakeClient(reply(None)))
    assert report.fallback_used and "no structured memo" in report.fallback_reason


@pytest.mark.parametrize(
    "error, reason",
    [
        (status_error(anthropic.AuthenticationError, 401), "authentication"),
        (status_error(anthropic.PermissionDeniedError, 403), "permission"),
        (status_error(anthropic.NotFoundError, 404), "not found"),
        (status_error(anthropic.RateLimitError, 429), "rate limit"),
        (status_error(anthropic.InternalServerError, 500), "500"),
        (anthropic.APIConnectionError(request=REQUEST), "could not reach"),
        (anthropic.APITimeoutError(request=REQUEST), "could not reach"),
        (schema_error(), "schema"),
    ],
    ids=["401", "403", "404", "429", "500", "connection", "timeout", "schema"],
)
def test_api_errors_fall_back_to_the_template_with_the_reason(error, reason):
    client = FakeClient(error)
    narrative, report = narrate(client)
    assert narrative == TemplateNarrator().generate(make_context())
    assert report.narrator_used == "template_fallback" and report.fallback_used
    assert reason in report.fallback_reason.lower()
    assert len(client.calls) == 1  # the SDK already retried transient errors; we do not retry them again


def test_generate_raises_narrator_error_on_a_refusal():
    narrator = ClaudeNarrator(LLM, client=FakeClient(reply(None, stop_reason="refusal")))
    with pytest.raises(NarratorError, match="refusal"):
        narrator.generate(make_context())


# --- the request sent to Claude ----------------------------------------------------------------


def test_request_uses_structured_outputs_and_no_sampling_overrides():
    client = FakeClient(reply(GOOD))
    ClaudeNarrator(LLM, client=client).generate(make_context())
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["max_tokens"] == 16000 and call["timeout"] == 120.0
    assert call["output_format"] is MemoNarrative
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert not {"temperature", "top_p", "top_k", "thinking", "tool_choice", "tools"} & set(call)
    assert [m["role"] for m in call["messages"]] == ["user"]  # no assistant prefill


def test_the_context_is_sent_as_json():
    client = FakeClient(reply(GOOD))
    context = make_context()
    ClaudeNarrator(LLM, client=client).generate(context)
    content = client.calls[0]["messages"][0]["content"]
    assert json.JSONDecoder().raw_decode(content, content.index("{"))[0] == context


def test_without_server_side_fallback_the_standard_endpoint_is_called():
    client = FakeClient(reply(GOOD), beta=False)
    narrator = ClaudeNarrator(LLMConfig(server_side_fallback=False), client=client)
    assert narrator.generate(make_context()) == GOOD
    call = client.messages.calls[0]
    assert "betas" not in call and "fallbacks" not in call


def test_system_prompt_states_the_limits_and_distrusts_ticket_text():
    client = FakeClient(reply(GOOD))
    get_narrator(LLMConfig(provider="claude"), client=client, memo_max_words=150).generate(make_context())
    system = client.calls[0]["system"]
    assert "at most 150 words" in system and "at most 120 characters" in system
    assert "untrusted" in system


# --- choosing a narrator ----------------------------------------------------------------------


def test_template_provider_ignores_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert isinstance(get_narrator(LLMConfig(provider="template")), TemplateNarrator)


def test_claude_provider_uses_the_configured_model():
    narrator = get_narrator(LLMConfig(provider="claude", model="claude-sonnet-5-5"), client=FakeClient())
    assert isinstance(narrator, ClaudeNarrator) and narrator.model == "claude-sonnet-5-5"


def test_claude_provider_without_credentials_fails_at_construction(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(TypeError, match="ANTHROPIC_API_KEY"):
        get_narrator(LLMConfig(provider="claude"))
    with pytest.raises(TypeError, match="ANTHROPIC_API_KEY"):
        ClaudeNarrator(LLM)


def test_claude_provider_with_an_injected_client_needs_no_credentials(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    assert isinstance(get_narrator(LLMConfig(provider="claude"), client=FakeClient()), ClaudeNarrator)


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"ANTHROPIC_API_KEY": "test-key"}, ClaudeNarrator),
        ({"ANTHROPIC_AUTH_TOKEN": "test-token"}, ClaudeNarrator),
        ({}, TemplateNarrator),
        ({"ANTHROPIC_API_KEY": ""}, TemplateNarrator),
    ],
    ids=["api-key", "auth-token", "none", "empty-key"],
)
def test_auto_uses_claude_only_when_credentials_are_set(monkeypatch, env, expected):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert isinstance(get_narrator(LLMConfig(provider="auto"), client=FakeClient()), expected)


# --- the template ------------------------------------------------------------------------------

CONTEXTS = {
    "full": {},
    "no-findings": {"findings": False},
    "no-tickets": {"tickets": False},
    "nothing": {"findings": False, "tickets": False},
    "surge": {"surge": True},
    "surge-no-findings": {"surge": True, "findings": False},
}


@pytest.mark.parametrize("kind", CONTEXTS)
def test_template_memo_passes_every_guardrail(kind):
    context = make_context(**CONTEXTS[kind])
    assert check_narrative(TemplateNarrator().generate(context), context, CFG) == []


def test_template_memo_is_deterministic_ascii_and_ignores_feedback():
    context = make_context()
    memo = TemplateNarrator().generate(context)
    assert TemplateNarrator().generate(context, feedback=["anything"]) == memo
    assert memo.model_dump_json().isascii()


def test_template_memo_names_the_segment_the_cost_and_the_error():
    memo = TemplateNarrator().generate(make_context())
    assert memo.headline == "SEV1: checkout conversion down 25.0% for DE / iOS / PayPal"
    assert "USD 33,000" in memo.executive_summary
    assert "PAYPAL_SDK_TIMEOUT" in memo.root_cause_explanation
    assert memo.customer_evidence == [QUOTE, SECOND]
    assert (memo.recommended_actions[0].priority, memo.recommended_actions[0].owner_team) == ("P0", "Payments")
    assert memo.confidence == "high"


def test_template_quotes_only_excerpts_it_can_print_as_ascii():
    memo = TemplateNarrator().generate(make_context(texts=("Zahlung über PayPal schlägt fehl.", QUOTE)))
    assert memo.customer_evidence == [QUOTE]


def test_template_without_findings_says_the_cause_is_not_isolated():
    memo = TemplateNarrator().generate(make_context(findings=False))
    assert "not isolated" in memo.root_cause_explanation and memo.confidence == "low"


@pytest.mark.parametrize(
    "attributes, owner",
    [([("os", "Android"), ("app_version", "5.4.0")], "Mobile"), ([("country", "US"), ("device_type", "desktop")], "SRE")],
    ids=["mobile", "sre"],
)
def test_template_gives_the_p0_action_to_the_team_that_owns_the_segment(attributes, owner):
    context = make_context()
    context["findings"][0]["attributes"] = [{"attribute": a, "value": v} for a, v in attributes]
    memo = TemplateNarrator().generate(context)
    assert memo.recommended_actions[0].owner_team == owner
    assert check_narrative(memo, context, CFG) == []


def test_template_for_a_volume_surge_reports_no_loss_and_asks_to_confirm_demand():
    memo = TemplateNarrator().generate(make_context(surge=True))
    assert memo.headline.startswith("SEV3: checkout attempts up 80.0%")
    assert "No revenue loss" in memo.executive_summary
    last = memo.recommended_actions[-1]
    assert last.priority == "P2" and "surge" in last.action


def test_template_survives_a_kpi_without_a_relative_change():
    context = make_context()
    context["kpis"]["conversion_rate"]["rel_change"] = None  # nothing was expected
    memo = TemplateNarrator().generate(context)
    assert "changed" in memo.headline
    assert check_narrative(memo, context, CFG) == []


def test_template_narrator_runs_once_and_is_not_a_fallback():
    narrative, report = narrate_with_guardrails(TemplateNarrator(), make_context(), LLM, CFG)
    assert report.narrator_used == "template" and not report.fallback_used and report.fallback_reason is None
    assert report.passed and report.attempts == 1 and report.attempt_violations == [[]]


# --- the OpenAI-compatible narrator (fake transport, no network) ------------------------------

OAI = LLMConfig(provider="openai_compatible", max_retries=1)


class FakePost:
    """Stands in for the HTTP call: replays replies (dicts) or raises (exceptions), records requests."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, url, headers, body, timeout_s):
        self.calls.append({"url": url, "headers": headers, "body": body, "timeout_s": timeout_s})
        assert self.outcomes, "unexpected HTTP call"
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def chat(content, finish_reason="stop", **message):
    return {"choices": [{"finish_reason": finish_reason, "message": {"role": "assistant", "content": content, **message}}]}


def as_json(memo):
    return memo.model_dump_json()


@pytest.fixture
def clean_llm_env(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def oai_narrate(post, llm=OAI):
    return narrate_with_guardrails(OpenAICompatibleNarrator(llm, post), make_context(), llm, CFG)


def test_openai_compatible_good_json_is_used_as_is(clean_llm_env):
    post = FakePost(chat(as_json(GOOD)))
    narrative, report = oai_narrate(post)
    assert narrative == GOOD and report.passed and report.attempts == 1
    assert report.narrator_used == "openai_compatible" and not report.fallback_used


def test_openai_compatible_request_shape(clean_llm_env):
    post = FakePost(chat(as_json(GOOD)))
    OpenAICompatibleNarrator(OAI, post).generate(make_context())
    call = post.calls[0]
    assert call["url"] == "http://localhost:11434/v1/chat/completions"
    body = call["body"]
    assert body["model"] == "gemma3:1b" and body["temperature"] == 0.0
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == MemoNarrative.model_json_schema()
    assert "reasoning_effort" not in body  # only sent when llm.effort is set
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "untrusted" in body["messages"][0]["content"] and "JSON schema" in body["messages"][0]["content"]
    assert "Authorization" not in call["headers"]  # a local server needs no key


def test_openai_compatible_env_overrides_and_key_header(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example-free-llm.test/v1/")
    monkeypatch.setenv("LLM_MODEL", "some-free-model")
    monkeypatch.setenv("LLM_API_KEY", "secret-test-key")
    post = FakePost(chat(as_json(GOOD)))
    narrator = OpenAICompatibleNarrator(OAI.model_copy(update={"effort": "low", "response_format": "json_object"}), post)
    narrator.generate(make_context())
    call = post.calls[0]
    assert call["url"] == "https://example-free-llm.test/v1/chat/completions"
    assert call["body"]["model"] == "some-free-model" and narrator.model == "some-free-model"
    assert call["body"]["reasoning_effort"] == "low"
    assert call["body"]["response_format"] == {"type": "json_object"}
    assert call["headers"]["Authorization"] == "Bearer secret-test-key"
    assert "secret-test-key" not in json.dumps(call["body"])


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "https://api.anthropic.com/v1", "not a url"])
def test_openai_compatible_refuses_paid_hosts_and_bad_urls(clean_llm_env, url):
    with pytest.raises(ValueError):
        OpenAICompatibleNarrator(OAI.model_copy(update={"base_url": url}), FakePost())


def test_openai_compatible_strips_markdown_fences(clean_llm_env):
    post = FakePost(chat("```json\n" + as_json(GOOD) + "\n```"))
    assert OpenAICompatibleNarrator(OAI, post).generate(make_context()) == GOOD


def test_openai_compatible_invalid_json_is_retried_with_the_schema_problems(clean_llm_env):
    post = FakePost(chat('{"headline": "SEV1: cut off'), chat(as_json(GOOD)))
    narrative, report = oai_narrate(post)
    assert narrative == GOOD and report.passed and report.attempts == 2
    assert report.attempt_violations[0] and all(v.startswith("schema:") for v in report.attempt_violations[0])
    retry_prompt = post.calls[1]["body"]["messages"][1]["content"]
    assert all(v in retry_prompt for v in report.attempt_violations[0])


def test_my_reporting_test_an_invented_number_ends_in_the_template_memo(clean_llm_env):
    """Owner A's reporting test: an LLM that keeps inventing a loss never reaches the memo."""
    post = FakePost(chat(as_json(INVENTED)), chat(as_json(INVENTED)))
    narrative, report = oai_narrate(post)
    assert narrative == TemplateNarrator().generate(make_context())
    assert "99,999" not in narrative.executive_summary
    assert report.fallback_used and report.narrator_used == "template_fallback" and report.passed
    assert all(any("USD 99,999" in v for v in attempt) for attempt in report.attempt_violations)
    assert "USD 99,999" in post.calls[1]["body"]["messages"][1]["content"]  # the retry was told why


@pytest.mark.parametrize(
    "outcome, reason",
    [
        (urllib.error.HTTPError("u", 401, "unauthorized", None, None), "authentication failed (401): check LLM_API_KEY"),
        (urllib.error.HTTPError("u", 404, "not found", None, None), "not found (404)"),
        (urllib.error.HTTPError("u", 500, "boom", None, None), "server error (500)"),
        (urllib.error.URLError("connection refused"), "could not reach"),
        (TimeoutError(), "could not reach"),
        (chat("{}", finish_reason="length"), "finish_reason=length"),
        (chat(None, refusal="I can't help with that."), "refused"),
        ({"error": "no choices"}, "no choices[0].message"),
    ],
    ids=["401", "404", "500", "connection", "timeout", "cut-off", "refusal", "malformed"],
)
def test_openai_compatible_failures_fall_back_without_retrying(clean_llm_env, outcome, reason):
    post = FakePost(outcome)
    narrative, report = oai_narrate(post)
    assert len(post.calls) == 1
    assert narrative == TemplateNarrator().generate(make_context())
    assert report.fallback_used and reason in report.fallback_reason


def rate_limited(retry_after=None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("u", 429, "slow down", headers, None)


def test_openai_compatible_waits_out_a_429_and_resends(clean_llm_env):
    post = FakePost(rate_limited("7"), rate_limited("not a number"), chat(as_json(GOOD)))
    waits = []
    narrator = OpenAICompatibleNarrator(OAI, post, sleep=waits.append)
    narrative, report = narrate_with_guardrails(narrator, make_context(), OAI, CFG)
    assert narrative == GOOD and report.passed and report.attempts == 1  # a 429 is not a failed draft
    assert waits == [7.0, RATE_LIMIT_WAIT_S] and len(post.calls) == 3


def test_openai_compatible_falls_back_when_the_429s_do_not_stop(clean_llm_env):
    post = FakePost(*[rate_limited("600")] * (RATE_LIMIT_RETRIES + 1))
    waits = []
    narrator = OpenAICompatibleNarrator(OAI, post, sleep=waits.append)
    narrative, report = narrate_with_guardrails(narrator, make_context(), OAI, CFG)
    assert narrative == TemplateNarrator().generate(make_context())
    assert report.fallback_used and "rate limit (429)" in report.fallback_reason
    assert waits == [RATE_LIMIT_WAIT_S] * RATE_LIMIT_RETRIES  # Retry-After is capped
    assert len(post.calls) == RATE_LIMIT_RETRIES + 1


def test_openai_compatible_provider_is_opt_in(clean_llm_env):
    assert isinstance(get_narrator(LLMConfig()), TemplateNarrator)  # the default stays the template
    assert isinstance(get_narrator(LLMConfig(provider="openai_compatible")), OpenAICompatibleNarrator)
