"""Typed configuration (Pydantic v2) and YAML loading.

``load_config(path)`` merges a YAML file over the model defaults and validates the
result: config sections merge key by key, while any other value (lists and mapping
settings such as ``data.countries``) replaces its default as a whole. Every config
model forbids unknown keys, so a typo in the YAML fails loudly instead of being
silently ignored. RetrievalConfig, LLMConfig and
ReportingConfig follow INTERFACE SPEC v1 section 2 exactly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SEGMENT_ATTRIBUTES = ("country", "os", "device_type", "payment_gateway", "app_version")
DEVICE_TYPES = ("mobile", "tablet", "desktop")
WEB_APP_VERSION = "web"  # app_version recorded for desktop (browser) checkouts

BACKGROUND_TICKET_THEMES = (
    "shipping_delay",
    "refund_request",
    "login_issue",
    "promo_code",
    "generic_decline",
    "app_performance",
    "paypal_refund",
    "account_update",
    "delivery_address",
)
INCIDENT_TICKET_THEMES = (
    "paypal_timeout",
    "gateway_timeout",
    "app_crash_pay",
    "card_tokenization",
    "three_ds_failure",
    "generic_payment_failure",
)
TICKET_CHANNELS = ("email", "chat", "app_review", "social")

IncidentTheme = Literal[
    "paypal_timeout",
    "gateway_timeout",
    "app_crash_pay",
    "card_tokenization",
    "three_ds_failure",
    "generic_payment_failure",
]


class ConfigError(ValueError):
    """Raised when a configuration file cannot be read or does not validate."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _check_weights(name: str, weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError(f"{name} must not be empty")
    bad = {k: v for k, v in weights.items() if not (v > 0)}
    if bad:
        raise ValueError(f"{name} weights must be > 0, got {bad}")
    return weights


def _check_segment(name: str, segment: dict[str, str]) -> dict[str, str]:
    if not segment:
        raise ValueError(f"{name} must name at least one attribute")
    unknown = sorted(set(segment) - set(SEGMENT_ATTRIBUTES))
    if unknown:
        raise ValueError(f"{name} has unknown attributes {unknown}; allowed: {list(SEGMENT_ATTRIBUTES)}")
    empty = [k for k, v in segment.items() if not str(v).strip()]
    if empty:
        raise ValueError(f"{name} has empty values for {empty}")
    return {k: str(v) for k, v in segment.items()}


# --------------------------------------------------------------------------- data


class IncidentSpec(_Model):
    """A payment failure injected into one segment for a limited time."""

    incident_id: str
    segment: dict[str, str]  # attribute -> value, e.g. {"country": "DE", "os": "iOS"}
    start: datetime
    duration_hours: float = Field(gt=0, le=72)
    failure_rate: float = Field(gt=0, le=1)  # P(an attempt in the segment fails because of the incident)
    error_code: str
    latency_multiplier: float = Field(default=1.0, gt=0)
    ticket_rate: float = Field(default=0.03, ge=0, le=1)  # P(an incident failure produces a support ticket)
    ticket_themes: list[IncidentTheme] = Field(default_factory=lambda: ["generic_payment_failure"])

    @field_validator("start")
    @classmethod
    def _start_naive_utc(cls, v: datetime) -> datetime:
        return _naive_utc(v)

    @field_validator("segment")
    @classmethod
    def _segment_ok(cls, v: dict[str, str]) -> dict[str, str]:
        return _check_segment("incident segment", v)

    @field_validator("ticket_themes")
    @classmethod
    def _themes_ok(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("ticket_themes must not be empty")
        return v

    @field_validator("error_code")
    @classmethod
    def _error_code_ok(cls, v: str) -> str:
        v = v.strip()
        if not v or v.upper() == "NONE":
            raise ValueError("error_code must be a real failure code ('NONE' marks successful attempts)")
        return v

    @property
    def end(self) -> datetime:
        return self.start + timedelta(hours=self.duration_hours)


class BenignEventSpec(_Model):
    """A traffic change that is NOT an incident (e.g. a marketing campaign)."""

    event_id: str
    kind: Literal["volume_surge"] = "volume_surge"
    segment: dict[str, str]
    start: datetime
    duration_hours: float = Field(gt=0, le=72)
    volume_multiplier: float = Field(default=1.8, gt=0)
    description: str = ""

    @field_validator("start")
    @classmethod
    def _start_naive_utc(cls, v: datetime) -> datetime:
        return _naive_utc(v)

    @field_validator("segment")
    @classmethod
    def _segment_ok(cls, v: dict[str, str]) -> dict[str, str]:
        return _check_segment("benign event segment", v)

    @property
    def end(self) -> datetime:
        return self.start + timedelta(hours=self.duration_hours)


class TicketConfig(_Model):
    background_per_hour: float = Field(default=1.5, ge=0)  # at average traffic; follows the hourly profile
    theme_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "shipping_delay": 0.20,
            "refund_request": 0.15,
            "login_issue": 0.12,
            "promo_code": 0.10,
            "generic_decline": 0.15,
            "app_performance": 0.08,
            "paypal_refund": 0.07,
            "account_update": 0.07,
            "delivery_address": 0.06,
        }
    )
    channel_weights: dict[str, float] = Field(
        default_factory=lambda: {"email": 0.40, "chat": 0.35, "app_review": 0.15, "social": 0.10}
    )
    null_country_prob: float = Field(default=0.15, ge=0, le=1)
    null_os_prob: float = Field(default=0.25, ge=0, le=1)
    delay_mean_minutes: float = Field(default=25.0, gt=0)  # incident failure -> ticket delay (exponential)
    delay_max_hours: float = Field(default=6.0, gt=0)

    @field_validator("theme_weights")
    @classmethod
    def _themes_ok(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(v) - set(BACKGROUND_TICKET_THEMES))
        if unknown:
            raise ValueError(f"unknown background ticket themes {unknown}; allowed: {list(BACKGROUND_TICKET_THEMES)}")
        return _check_weights("tickets.theme_weights", v)

    @field_validator("channel_weights")
    @classmethod
    def _channels_ok(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(v) - set(TICKET_CHANNELS))
        if unknown:
            raise ValueError(f"unknown ticket channels {unknown}; allowed: {list(TICKET_CHANNELS)}")
        return _check_weights("tickets.channel_weights", v)


def _default_hourly_profile() -> list[float]:
    # Relative checkout traffic by UTC hour (normalised to mean 1 by the generator).
    return [
        0.55, 0.42, 0.33, 0.28, 0.27, 0.32, 0.45, 0.65, 0.85, 1.00, 1.12, 1.22,
        1.30, 1.30, 1.25, 1.22, 1.25, 1.32, 1.42, 1.50, 1.48, 1.35, 1.08, 0.78,
    ]  # fmt: skip


class DataConfig(_Model):
    """Synthetic checkout stream: dimension mixes, conversion model, incidents."""

    start: datetime = datetime(2026, 9, 1)
    days: int = Field(default=21, ge=2, le=120)
    attempts_per_hour: float = Field(default=3500.0, gt=0)  # mean over the day
    hourly_profile: list[float] = Field(default_factory=_default_hourly_profile)  # 24 values
    weekly_profile: list[float] = Field(default_factory=lambda: [0.96, 0.95, 0.97, 0.99, 1.03, 1.08, 1.02])  # Mon..Sun
    daily_traffic_noise: float = Field(default=0.03, ge=0, le=0.5)  # lognormal sigma of a per-day traffic factor

    countries: dict[str, float] = Field(
        default_factory=lambda: {
            "US": 0.28, "DE": 0.18, "GB": 0.12, "FR": 0.09, "IN": 0.11, "BR": 0.09, "CA": 0.07, "JP": 0.06,
        }  # fmt: skip
    )
    os: dict[str, float] = Field(default_factory=lambda: {"iOS": 0.36, "Android": 0.32, "Windows": 0.20, "macOS": 0.12})
    country_os: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "US": {"iOS": 0.45, "Android": 0.25, "Windows": 0.18, "macOS": 0.12},
            "DE": {"iOS": 0.42, "Android": 0.31, "Windows": 0.17, "macOS": 0.10},
            "IN": {"iOS": 0.10, "Android": 0.70, "Windows": 0.17, "macOS": 0.03},
            "BR": {"iOS": 0.18, "Android": 0.62, "Windows": 0.16, "macOS": 0.04},
            "JP": {"iOS": 0.55, "Android": 0.20, "Windows": 0.15, "macOS": 0.10},
        }
    )
    device_given_os: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "iOS": {"mobile": 0.85, "tablet": 0.15},
            "Android": {"mobile": 0.90, "tablet": 0.10},
            "Windows": {"desktop": 1.0},
            "macOS": {"desktop": 1.0},
        }
    )
    app_versions: dict[str, float] = Field(
        default_factory=lambda: {"5.2.1": 0.15, "5.3.0": 0.35, "5.4.0": 0.50}
    )  # mobile/tablet only; desktop is "web"
    payment_gateways: dict[str, float] = Field(
        default_factory=lambda: {"Stripe": 0.35, "PayPal": 0.22, "Adyen": 0.18, "Klarna": 0.10, "Braintree": 0.15}
    )
    country_gateways: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "DE": {"Stripe": 0.15, "PayPal": 0.55, "Adyen": 0.10, "Klarna": 0.15, "Braintree": 0.05},
            "BR": {"Stripe": 0.30, "PayPal": 0.15, "Adyen": 0.40, "Klarna": 0.02, "Braintree": 0.13},
            "IN": {"Stripe": 0.40, "PayPal": 0.10, "Adyen": 0.35, "Klarna": 0.02, "Braintree": 0.13},
        }
    )

    base_conversion: float = Field(default=0.86, gt=0, lt=1)  # success probability before dimension effects
    conversion_effects: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "country": {"US": 0.15, "DE": 0.10, "GB": 0.10, "FR": 0.0, "IN": -0.35, "BR": -0.75, "CA": 0.05, "JP": 0.20},
            "os": {"iOS": 0.10, "Android": -0.05, "Windows": 0.0, "macOS": 0.10},
            "device_type": {"mobile": 0.0, "tablet": 0.05, "desktop": 0.10},
            "payment_gateway": {"Stripe": 0.05, "PayPal": 0.10, "Adyen": 0.0, "Klarna": -0.10, "Braintree": 0.0},
            "app_version": {"5.2.1": -0.10, "5.3.0": 0.0, "5.4.0": 0.05, "web": 0.0},
        }
    )  # logit offsets per attribute value; unlisted values get 0
    hourly_conversion_amplitude: float = Field(default=0.05, ge=0, le=1)  # logit amplitude of a mild daily cycle

    amount_log_mean: float = 4.0  # lognormal order value in USD (mean ~ exp(mu + sigma^2/2))
    amount_log_sigma: float = Field(default=0.6, gt=0)
    country_amount_scale: dict[str, float] = Field(
        default_factory=lambda: {
            "US": 1.0, "DE": 0.95, "GB": 0.9, "FR": 0.9, "IN": 0.35, "BR": 0.5, "CA": 0.95, "JP": 1.05,
        }  # fmt: skip
    )
    latency_median_ms: float = Field(default=420.0, gt=0)
    latency_log_sigma: float = Field(default=0.35, gt=0)
    gateway_latency_scale: dict[str, float] = Field(
        default_factory=lambda: {"Stripe": 0.9, "PayPal": 1.25, "Adyen": 0.95, "Klarna": 1.1, "Braintree": 1.0}
    )
    error_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "CARD_DECLINED": 0.45, "INSUFFICIENT_FUNDS": 0.20, "USER_ABANDONED": 0.27, "FRAUD_BLOCK": 0.08,
        }  # fmt: skip
    )

    incidents: list[IncidentSpec] = Field(
        default_factory=lambda: [
            IncidentSpec(
                incident_id="INC-001",
                segment={"country": "DE", "os": "iOS", "payment_gateway": "PayPal"},
                start=datetime(2026, 9, 10, 13, 0),
                duration_hours=4.0,
                failure_rate=0.88,
                error_code="PAYPAL_SDK_TIMEOUT",
                latency_multiplier=6.0,
                ticket_rate=0.04,
                ticket_themes=["paypal_timeout"],
            ),
            IncidentSpec(
                incident_id="INC-002",
                segment={"os": "Android", "app_version": "5.4.0"},
                start=datetime(2026, 9, 16, 17, 0),
                duration_hours=3.0,
                failure_rate=0.32,
                error_code="APP_CRASH_ON_PAY",
                latency_multiplier=1.0,
                ticket_rate=0.03,
                ticket_themes=["app_crash_pay"],
            ),
        ]
    )
    benign_events: list[BenignEventSpec] = Field(
        default_factory=lambda: [
            BenignEventSpec(
                event_id="SURGE-US",
                segment={"country": "US"},
                start=datetime(2026, 9, 12, 18, 0),
                duration_hours=6.0,
                volume_multiplier=1.8,
                description="US marketing email campaign (traffic up, conversion unchanged)",
            )
        ]
    )
    tickets: TicketConfig = Field(default_factory=TicketConfig)

    @field_validator("start")
    @classmethod
    def _start_naive_utc(cls, v: datetime) -> datetime:
        return _naive_utc(v)

    @field_validator("hourly_profile")
    @classmethod
    def _hourly_ok(cls, v: list[float]) -> list[float]:
        if len(v) != 24 or any(not (x > 0) for x in v):
            raise ValueError("hourly_profile needs 24 positive values (UTC hours 0..23)")
        return v

    @field_validator("weekly_profile")
    @classmethod
    def _weekly_ok(cls, v: list[float]) -> list[float]:
        if len(v) != 7 or any(not (x > 0) for x in v):
            raise ValueError("weekly_profile needs 7 positive values (Mon..Sun)")
        return v

    @field_validator("countries", "os", "app_versions", "payment_gateways", "error_mix")
    @classmethod
    def _weights_ok(cls, v: dict[str, float], info) -> dict[str, float]:
        return _check_weights(f"data.{info.field_name}", v)

    @model_validator(mode="after")
    def _cross_checks(self) -> "DataConfig":
        for country, weights in self.country_os.items():
            if country not in self.countries:
                raise ValueError(f"country_os mentions unknown country {country!r}")
            _check_weights(f"country_os[{country}]", weights)
            unknown = sorted(set(weights) - set(self.os))
            if unknown:
                raise ValueError(f"country_os[{country}] has unknown os {unknown}")
        for os_name in self.os:
            if os_name not in self.device_given_os:
                raise ValueError(f"device_given_os is missing os {os_name!r}")
        for os_name, weights in self.device_given_os.items():
            _check_weights(f"device_given_os[{os_name}]", weights)
            unknown = sorted(set(weights) - set(DEVICE_TYPES))
            if unknown:
                raise ValueError(f"device_given_os[{os_name}] has unknown device types {unknown}")
        for country, weights in self.country_gateways.items():
            if country not in self.countries:
                raise ValueError(f"country_gateways mentions unknown country {country!r}")
            _check_weights(f"country_gateways[{country}]", weights)
            unknown = sorted(set(weights) - set(self.payment_gateways))
            if unknown:
                raise ValueError(f"country_gateways[{country}] has unknown gateways {unknown}")
        unknown_attrs = sorted(set(self.conversion_effects) - set(SEGMENT_ATTRIBUTES))
        if unknown_attrs:
            raise ValueError(f"conversion_effects has unknown attributes {unknown_attrs}")
        if WEB_APP_VERSION in self.app_versions:
            raise ValueError(f"app_versions lists mobile versions only; {WEB_APP_VERSION!r} is implied for desktop")

        data_end = self.start + timedelta(days=self.days)
        ids: set[str] = set()
        for spec in [*self.incidents, *self.benign_events]:
            spec_id = spec.incident_id if isinstance(spec, IncidentSpec) else spec.event_id
            if spec_id in ids:
                raise ValueError(f"duplicate incident/event id {spec_id!r}")
            ids.add(spec_id)
            if spec.start < self.start or spec.end > data_end:
                raise ValueError(
                    f"{spec_id} runs {spec.start:%Y-%m-%d %H:%M}..{spec.end:%Y-%m-%d %H:%M}, "
                    f"outside the data range {self.start:%Y-%m-%d %H:%M}..{data_end:%Y-%m-%d %H:%M}"
                )
            self._check_segment_values(spec_id, spec.segment)
        return self

    def _check_segment_values(self, spec_id: str, segment: dict[str, str]) -> None:
        known = {
            "country": set(self.countries),
            "os": set(self.os),
            "device_type": set(DEVICE_TYPES),
            "payment_gateway": set(self.payment_gateways),
            "app_version": set(self.app_versions) | {WEB_APP_VERSION},
        }
        for attr, value in segment.items():
            if value not in known[attr]:
                raise ValueError(f"{spec_id}: {attr}={value!r} is not one of {sorted(known[attr])}")

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=self.days)


# ---------------------------------------------------------------------- detection


class DetectionConfig(_Model):
    freq: str = "1h"  # KPI bucket size (pandas offset alias)
    period: int = Field(default=24, ge=2)  # daily seasonality in buckets
    weekly_period: int = Field(default=168, ge=2)  # weekly seasonality in buckets
    # STL seasonal smoother length in cycles (odd). Wider than statsmodels' default 7 so the
    # daily shape is estimated from ~3 weeks of the same hour instead of overfitting noise.
    stl_seasonal: int = Field(default=21, ge=7)
    # STL trend smoother length in buckets (odd, > period). None = the smallest odd length
    # above one period, so the trend follows day-to-day traffic levels (weekday effect,
    # day noise); statsmodels' default (39 for period 24) lags them and leaves day-long
    # residual drifts that the forest flags.
    stl_trend: int | None = None
    # Use MSTL(daily, weekly) once this many weeks of history exist. Each hour-of-week phase
    # then has at least this many observations; with 3 the weekly fit follows the noise
    # (measurements in kpi_rca.detection.detector, next to uses_mstl).
    mstl_min_weeks: float = Field(default=6.0, ge=2)
    rolling_window: int = Field(default=3, ge=1)  # trailing rolling mean of residual z-scores (IF features)
    n_estimators: int = Field(default=200, ge=10)
    random_state: int = 42
    min_window_buckets: int = Field(default=2, ge=1)  # debounce: a window needs this many flagged buckets
    gap_tolerance: int = Field(default=1, ge=0)  # unflagged buckets allowed inside one window
    # streaming replay
    stream_history_days: float = Field(default=14.0, gt=1)
    stream_warmup_days: float = Field(default=7.0, gt=1)
    stream_refit_every: int = Field(default=24, ge=1)  # refit the Isolation Forest every N buckets
    # live level = median of the last N deseasonalised buckets (see detection.streaming)
    stream_level_buckets: int = Field(default=25, ge=1)

    @field_validator("freq")
    @classmethod
    def _freq_ok(cls, v: str) -> str:
        from pandas.tseries.frequencies import to_offset

        try:
            to_offset(v)
        except ValueError as exc:
            raise ValueError(f"detection.freq {v!r} is not a pandas offset alias") from exc
        return v

    @field_validator("stl_seasonal")
    @classmethod
    def _odd_seasonal(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("detection.stl_seasonal must be odd")
        return v

    @model_validator(mode="after")
    def _trend_ok(self) -> "DetectionConfig":
        if self.stl_trend is not None and (self.stl_trend % 2 == 0 or self.stl_trend <= self.period):
            raise ValueError(f"detection.stl_trend must be odd and > period ({self.period})")
        return self

    @model_validator(mode="after")
    def _stream_ok(self) -> "DetectionConfig":
        if self.stream_warmup_days > self.stream_history_days:
            raise ValueError("stream_warmup_days must be <= stream_history_days")
        return self


# ---------------------------------------------------------------------------- rca


class RCAConfig(_Model):
    attributes: list[str] = Field(default_factory=lambda: list(SEGMENT_ATTRIBUTES))
    baseline_days: float = Field(default=7.0, gt=0)  # baseline = this many days before the window
    max_baseline_rows: int = Field(default=300_000, ge=1000)
    p0_clip: float = Field(default=1e-4, gt=0, lt=0.5)
    # LightGBM (M0 = baseline failure model, M1 = window change model on top of logit(p0))
    m0_rounds: int = Field(default=150, ge=1)
    m1_rounds: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    num_leaves: int = Field(default=15, ge=2)
    min_child_samples: int = Field(default=40, ge=1)
    lambda_l2: float = Field(default=1.0, ge=0)
    num_threads: int = Field(default=0, ge=0)  # 0 = LightGBM default
    # SHAP
    shap_max_rows: int = Field(default=30_000, ge=100)
    interaction_rows: int = Field(default=2000, ge=50)
    # SHAP-guided search
    shap_positive_rel: float = Field(default=0.1, ge=0, le=1)  # "clearly positive" = >= rel * row's max SHAP
    shap_positive_abs: float = Field(default=0.05, ge=0)  # ... and >= this many log-odds
    max_order: int = Field(default=4, ge=1, le=5)
    top_k_patterns: int = Field(default=25, ge=1)
    prune_tolerance: float = Field(default=0.05, ge=0, lt=1)  # drop an attribute if score falls < this fraction
    min_ep: float = Field(default=0.10, gt=0, le=1)  # stop when the next cause explains less than this
    alpha: float = Field(default=0.01, gt=0, lt=1)  # significance level of the two-proportion z-test
    max_findings: int = Field(default=3, ge=1)
    min_segment_rows: int = Field(default=30, ge=1)
    # brute-force comparator
    brute_force_max_order: int = Field(default=3, ge=1, le=5)

    @field_validator("attributes")
    @classmethod
    def _attrs_ok(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(SEGMENT_ATTRIBUTES))
        if unknown or not v:
            raise ValueError(f"rca.attributes must be a non-empty subset of {list(SEGMENT_ATTRIBUTES)}")
        return v


# ------------------------------------------------- retrieval / llm / reporting (spec)


class RetrievalConfig(_Model):
    backend: Literal["auto", "sentence_transformers", "lsa"] = "auto"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    lsa_dim: int = 128
    top_k: int = 8
    overfetch: int = 200
    pad_before_h: float = 2.0
    pad_after_h: float = 6.0
    country_boost: float = 0.05
    os_boost: float = 0.05
    similarity_threshold: float = 0.35
    random_state: int = 42


class LLMConfig(_Model):
    provider: Literal["auto", "claude", "template"] = "auto"
    model: str = "claude-opus-5-5"
    max_tokens: int = 16000
    max_retries: int = 2
    server_side_fallback: bool = True
    timeout_s: float = 120.0


class ReportingConfig(_Model):
    currency: str = "USD"
    sev1_lost_revenue: float = 20000.0  # >= this -> SEV1
    sev2_lost_revenue: float = 5000.0  # >= this -> SEV2, else SEV3
    max_ticket_excerpts: int = 8
    excerpt_chars: int = 300
    memo_max_words: int = 200
    numeric_rel_tol: float = 0.02
    make_plots: bool = True


# --------------------------------------------------------------------- evaluation


class EvaluationConfig(_Model):
    """Randomised benchmark: smaller scenarios with random incidents and a benign surge."""

    n_scenarios: int = Field(default=20, ge=1)
    seed: int = 7
    days: int = Field(default=14, ge=9)
    attempts_per_hour: float = Field(default=3000.0, gt=0)
    min_incidents: int = Field(default=1, ge=1)
    max_incidents: int = Field(default=2, ge=1)
    max_order: int = Field(default=3, ge=1, le=4)
    min_segment_share: float = Field(default=0.03, gt=0, lt=1)
    max_segment_share: float = Field(default=0.25, gt=0, le=1)
    duration_hours: tuple[int, int] = (2, 6)
    visibility_sigma: tuple[float, float] = (4.0, 9.0)  # target drop in global conversion, in hourly binomial sigmas
    failure_rate_bounds: tuple[float, float] = (0.2, 0.95)
    surge_multiplier: tuple[float, float] = (1.5, 2.2)
    surge_duration_hours: tuple[int, int] = (3, 8)
    ticket_rate: float = Field(default=0.04, ge=0, le=1)
    backends: list[Literal["lsa", "sentence_transformers"]] = Field(
        default_factory=lambda: ["lsa", "sentence_transformers"]
    )
    # streaming replay per scenario: "incidents" replays a few hours around each incident
    # (detection delay only, fast); "full" replays everything after the warm-up (also
    # streaming precision/recall, about 30 s per 14-day scenario); "off" skips it.
    stream_mode: Literal["full", "incidents", "off"] = "incidents"
    incident_free_runs: int = Field(default=1, ge=0)  # extra scenarios without incidents, for false alarms/day
    quick_scenarios: int = Field(default=3, ge=1)  # n_scenarios used by `evaluate --quick`
    bootstrap_resamples: int = Field(default=1000, ge=100)  # for the 95% confidence intervals

    @model_validator(mode="after")
    def _ranges_ok(self) -> "EvaluationConfig":
        for name in ("duration_hours", "visibility_sigma", "failure_rate_bounds", "surge_multiplier", "surge_duration_hours"):
            lo, hi = getattr(self, name)
            if not (0 < lo <= hi):
                raise ValueError(f"evaluation.{name} must satisfy 0 < low <= high, got {(lo, hi)}")
        if self.min_incidents > self.max_incidents:
            raise ValueError("evaluation.min_incidents must be <= max_incidents")
        if self.min_segment_share >= self.max_segment_share:
            raise ValueError("evaluation.min_segment_share must be < max_segment_share")
        return self


# ------------------------------------------------------------------------ app


class AppConfig(_Model):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    rca: RCAConfig = Field(default_factory=RCAConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def _section_type(model_cls: type[BaseModel], key: str) -> type[BaseModel] | None:
    field = model_cls.model_fields.get(key)
    annotation = field.annotation if field is not None else None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _deep_merge(model_cls: type[BaseModel], base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base`` following the model structure.

    Config sections (nested models such as ``detection`` or ``data.tickets``) merge key
    by key; every other value replaces the default as a whole - including mapping-valued
    settings like ``data.countries``, so an override can also remove entries, and lists.
    """
    merged = dict(base)
    for key, value in override.items():
        section = _section_type(model_cls, key)
        if section is not None and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(section, merged[key], value)
        else:
            merged[key] = value
    return merged


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        lines.append(f"  - {loc}: {err.get('msg')}")
    return "\n".join(lines)


def config_from_dict(overrides: dict[str, Any] | None = None, *, source: str = "<dict>") -> AppConfig:
    """Validate ``overrides`` merged over the defaults; raise ConfigError with readable field errors."""
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ConfigError(f"{source}: top level must be a mapping, got {type(overrides).__name__}")
    defaults = AppConfig().model_dump(mode="python")
    try:
        return AppConfig.model_validate(_deep_merge(AppConfig, defaults, overrides))
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {source}:\n{_format_errors(exc)}") from None


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load a YAML config merged over the defaults. ``None`` returns the defaults."""
    if path is None:
        return AppConfig()
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {path}: {exc}") from None
    return config_from_dict(raw or {}, source=str(path))
