"""Tests for ticket retrieval: embedders, query building and the time-windowed FAISS index."""

from __future__ import annotations

import logging
import os
import sys
import types
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from kpi_rca.config import RetrievalConfig
from kpi_rca.retrieval import embeddings
from kpi_rca.retrieval.embeddings import LSAEmbedder, SentenceTransformerEmbedder, make_embedder
from kpi_rca.retrieval.index import TicketIndex
from kpi_rca.retrieval.query import build_query
from kpi_rca.schemas import SegmentAttribute, SegmentFinding

CFG = RetrievalConfig(backend="lsa")

CORPUS = [
    "PayPal payment timed out on my iPhone.",
    "PayPal checkout keeps timing out in the iPhone app.",
    "My parcel is late and tracking has not updated in five days.",
    "The courier says my parcel is still at the depot.",
    "The password reset email never arrives, so I cannot log in.",
    "The promo code is not applying at checkout.",
]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No test may load a real model: importing sentence_transformers fails unless a test fakes it."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS_WARNING", raising=False)  # restored after the test


def fake_sentence_transformers(monkeypatch, fail: bool = False) -> list[dict]:
    """Install a stand-in sentence_transformers module; returns the kwargs of every encode() call.

    The kwargs of every model construction are kept in the module's ``loads`` list.
    """
    calls: list[dict] = []
    loads: list[dict] = []

    class FakeModel:
        def __init__(self, model_name: str, **kwargs):
            loads.append(kwargs)
            if fail:
                raise OSError(f"cannot download {model_name}")

        def encode(self, texts, **kwargs):
            calls.append(kwargs)
            return np.asfortranarray(np.full((len(texts), 4), 0.5))  # float64, unit rows, not C-ordered

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    module.loads = loads
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


def make_finding(segment: dict[str, str], top_error_codes: dict[str, int] | None = None) -> SegmentFinding:
    """A finding with dummy statistics; retrieval only reads its attributes and error codes."""
    return SegmentFinding(
        rank=1,
        attributes=[SegmentAttribute(attribute=a, value=v) for a, v in segment.items()],
        ep=0.8,
        purity=0.7,
        score=0.75,
        failure_rate_window=0.6,
        failure_rate_baseline=0.05,
        n_window=1000,
        n_baseline=20000,
        excess_failures=550.0,
        p_value=1e-12,
        shap_contribution={},
        top_error_codes=top_error_codes or {},
    )


PAYPAL_IOS_DE = make_finding({"country": "DE", "os": "iOS", "payment_gateway": "PayPal"}, {"PAYPAL_SDK_TIMEOUT": 412})


# --- LSA embedder ---------------------------------------------------------------


def test_lsa_encode_gives_float32_c_contiguous_unit_rows():
    vectors = LSAEmbedder(dim=4).fit(CORPUS).encode(CORPUS)
    assert vectors.shape == (len(CORPUS), 4)
    assert vectors.dtype == np.float32 and vectors.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


def test_lsa_unknown_words_give_a_zero_row_without_warnings():
    embedder = LSAEmbedder(dim=4).fit(CORPUS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a 0/0 normalisation would warn
        vectors = embedder.encode(["zebra quantum xylophone"])
    assert vectors.shape == (1, 4) and not np.isnan(vectors).any()
    assert np.all(vectors == 0)


# sklearn divides by the corpus variance when it reports explained variance; one document has none.
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("n_docs", [1, 2, 3])
def test_lsa_tiny_corpora_do_not_crash(n_docs):
    texts = CORPUS[:n_docs]
    vectors = LSAEmbedder(dim=128).fit(texts).encode(texts)
    assert vectors.shape == (n_docs, max(1, n_docs - 1))  # fewer components than documents, at least 1
    assert vectors.dtype == np.float32 and not np.isnan(vectors).any()


def test_lsa_encode_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        LSAEmbedder().encode(["anything"])


# --- sentence-transformers embedder (simulated) -------------------------------------


def test_sentence_transformer_embedder_returns_float32_c_contiguous(monkeypatch):
    calls = fake_sentence_transformers(monkeypatch)
    embedder = SentenceTransformerEmbedder("org/some-model")
    assert embedder.fit(["nothing is learned"]) is embedder
    vectors = embedder.encode(["a", "b"])
    assert embedder.name == "sentence_transformers"
    assert vectors.shape == (2, 4) and vectors.dtype == np.float32 and vectors.flags["C_CONTIGUOUS"]
    assert calls == [{"normalize_embeddings": True, "show_progress_bar": False}]
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"


# --- make_embedder ------------------------------------------------------------------


def test_make_embedder_lsa_uses_the_config():
    embedder = make_embedder(RetrievalConfig(backend="lsa", lsa_dim=16, random_state=7))
    assert isinstance(embedder, LSAEmbedder) and embedder.name == "lsa"
    assert (embedder.dim, embedder.random_state) == (16, 7)


def test_backend_argument_overrides_the_config():
    # The config asks for sentence-transformers, which cannot load here; the argument wins.
    embedder = make_embedder(RetrievalConfig(backend="sentence_transformers"), backend="lsa")
    assert isinstance(embedder, LSAEmbedder)


def test_auto_prefers_sentence_transformers_when_it_loads(monkeypatch):
    fake_sentence_transformers(monkeypatch)
    assert isinstance(make_embedder(RetrievalConfig(backend="auto")), SentenceTransformerEmbedder)


def test_auto_reads_only_the_local_cache_and_explicit_may_download(monkeypatch):
    fake_sentence_transformers(monkeypatch)
    make_embedder(RetrievalConfig(backend="auto"))
    make_embedder(RetrievalConfig(backend="sentence_transformers"))
    assert sys.modules["sentence_transformers"].loads == [{"local_files_only": True}, {"local_files_only": False}]


def test_auto_falls_back_to_lsa_when_the_model_is_not_cached(monkeypatch):
    fake_sentence_transformers(monkeypatch, fail=True)
    assert isinstance(make_embedder(RetrievalConfig(backend="auto")), LSAEmbedder)
    assert sys.modules["sentence_transformers"].loads == [{"local_files_only": True}]


def test_auto_falls_back_to_lsa_with_a_warning(monkeypatch, caplog):
    class Unloadable:
        def __init__(self, model_name: str, local_files_only: bool = False):
            raise OSError(f"no network to download {model_name}")

    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", Unloadable)
    with caplog.at_level(logging.WARNING, logger="kpi_rca.retrieval.embeddings"):
        embedder = make_embedder(RetrievalConfig(backend="auto"))
    assert isinstance(embedder, LSAEmbedder)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and "no network" in warned[0] and "LSA" in warned[0]


@pytest.mark.parametrize("download_fails", [False, True], ids=["package-missing", "download-fails"])
def test_explicit_sentence_transformers_failure_raises(monkeypatch, download_fails):
    if download_fails:
        fake_sentence_transformers(monkeypatch, fail=True)
    cfg = RetrievalConfig(backend="sentence_transformers", model_name="org/tiny-model")
    with pytest.raises(RuntimeError, match=r"'org/tiny-model'.*(ModuleNotFoundError|OSError)"):
        make_embedder(cfg)


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="unknown retrieval backend 'faiss'"):
        make_embedder(RetrievalConfig(), backend="faiss")


# --- build_query ----------------------------------------------------------------------


def test_query_for_the_paypal_ios_incident():
    assert build_query(PAYPAL_IOS_DE) == "PayPal iPhone iOS app payment failed at checkout paypal sdk timeout"


def test_query_contains_gateway_os_and_error_code_words():
    query = build_query(PAYPAL_IOS_DE, ["PAYPAL_SDK_TIMEOUT"])
    for words in ("PayPal", "iPhone iOS app", "paypal sdk timeout", "payment failed"):
        assert words in query


def test_query_leaves_out_the_country():
    for finding in (PAYPAL_IOS_DE, make_finding({"country": "DE"})):
        query = build_query(finding).lower()
        assert "de" not in query.split() and "germany" not in query


@pytest.mark.parametrize(
    ("attribute", "value", "words"),
    [
        ("payment_gateway", "PayPal", "PayPal"),
        ("payment_gateway", "Klarna", "Klarna"),
        ("payment_gateway", "Stripe", "Stripe card"),
        ("payment_gateway", "Adyen", "Adyen card"),
        ("payment_gateway", "Braintree", "Braintree card"),
        ("os", "iOS", "iPhone iOS app"),
        ("os", "Android", "Android phone app"),
        ("os", "Windows", "Windows PC"),
        ("os", "macOS", "Mac"),
        ("device_type", "desktop", "computer laptop"),
        ("app_version", "5.4.0", "app version 5.4.0 update"),
        ("app_version", "web", "website browser checkout"),
    ],
)
def test_segment_values_become_customer_words(attribute, value, words):
    assert words in build_query(make_finding({attribute: value}))


def test_web_app_version_is_not_called_an_app_version():
    assert "app version" not in build_query(make_finding({"app_version": "web"}))


def test_error_codes_default_to_the_findings_top_error_codes():
    finding = make_finding({"os": "Android"}, {"APP_CRASH_ON_PAY": 120, "GATEWAY_TIMEOUT": 4})
    assert "app crash on pay" in build_query(finding) and "gateway timeout" in build_query(finding)
    assert "gateway timeout" not in build_query(finding, ["APP_CRASH_ON_PAY"])


def test_query_does_not_depend_on_attribute_order():
    reordered = make_finding({"payment_gateway": "PayPal", "os": "iOS", "country": "DE"}, {"PAYPAL_SDK_TIMEOUT": 1})
    assert build_query(reordered) == build_query(PAYPAL_IOS_DE)


# --- TicketIndex: retrieval -------------------------------------------------------------

TICKET_COLUMNS = ["ticket_id", "created_at", "channel", "customer_country", "device_os", "text"]


def make_tickets(rows: list[tuple]) -> pd.DataFrame:
    """A ticket frame shaped like the loader's: created_at is datetime64, a missing country/OS is NaN."""
    tickets = pd.DataFrame(rows, columns=TICKET_COLUMNS)
    tickets["created_at"] = pd.to_datetime(tickets["created_at"])
    return tickets


# A PayPal/iOS incident window. With the default padding (2h before, 6h after) the
# search range is [2026-09-10 11:00, 2026-09-10 23:00).
WINDOW_START, WINDOW_END = datetime(2026, 9, 10, 13), datetime(2026, 9, 10, 17)
QUERY = build_query(PAYPAL_IOS_DE)

# Grouped by role, so the rows are not in time order: build() has to sort them.
ROWS = [
    # PayPal timeouts inside the search range
    ("TK000101", "2026-09-10 11:00", "email", "DE", "iOS", "Paid with PayPal on my iPhone, the payment timed out."),
    ("TK000102", "2026-09-10 13:20", "chat", "DE", "iOS", "PayPal checkout keeps timing out in the iPhone app."),
    ("TK000103", "2026-09-10 13:45", "email", "DE", "iOS", "PayPal is broken on my iPhone, failed with a timeout."),
    ("TK000104", "2026-09-10 14:30", "app", None, "iOS", "PayPal payment stuck on my iPhone, then it timed out."),
    ("TK000105", "2026-09-10 15:10", "chat", "DE", None, "Third PayPal attempt in the app, each one times out."),
    ("TK000106", "2026-09-10 16:40", "email", "DE", "iOS", "Checkout with PayPal hangs after the PayPal login."),
    ("TK000107", "2026-09-10 18:30", "chat", "DE", "iOS", "PayPal timeout again on my iPhone, payment failed."),
    # other topics inside the search range
    ("TK000108", "2026-09-10 12:00", "email", "US", "Android", "My order hasn't arrived and tracking hasn't updated."),
    ("TK000109", "2026-09-10 14:00", "chat", "GB", None, "I can't log in, the password reset email never arrives."),
    ("TK000110", "2026-09-10 15:30", "email", None, None, "The promo code isn't applying at checkout."),
    ("TK000111", "2026-09-10 17:30", "chat", "FR", "Windows", "PayPal refund for my cancelled order still not showing."),
    ("TK000112", "2026-09-10 20:00", "email", "DE", "macOS", "Delivery is late again, the parcel is still at the depot."),
    # outside the search range, including PayPal timeouts right at both edges
    ("TK000113", "2026-09-08 10:00", "email", "DE", "iOS", "PayPal payment timed out on my iPhone at checkout."),
    ("TK000114", "2026-09-10 10:59", "chat", "DE", "iOS", "PayPal timeout on my iPhone, the payment failed."),
    ("TK000115", "2026-09-10 23:00", "email", "DE", "iOS", "PayPal checkout on my iPhone failed with a timeout."),
    ("TK000116", "2026-09-11 01:00", "chat", "DE", "iOS", "PayPal timeout in the iPhone app, payment failed."),
    ("TK000117", "2026-09-12 09:00", "email", "US", None, "Paid with PayPal and returned it, where is my PayPal refund?"),
    ("TK000118", "2026-09-09 12:00", "chat", "US", "Android", "Where is my package? It should have arrived last week."),
    ("TK000119", "2026-09-11 15:00", "email", "GB", "iOS", "Sign in keeps failing with 'session expired'."),
    ("TK000120", "2026-09-12 18:00", "chat", "IN", "Android", "When will I get my money back for the cancelled order?"),
]
PAYPAL_TIMEOUTS_IN_RANGE = {f"TK000{i}" for i in range(101, 108)}
IN_RANGE = PAYPAL_TIMEOUTS_IN_RANGE | {f"TK000{i}" for i in range(108, 113)}


@pytest.fixture
def index() -> TicketIndex:
    return TicketIndex(LSAEmbedder(), CFG).build(make_tickets(ROWS))


@pytest.mark.parametrize("finding", [None, PAYPAL_IOS_DE], ids=["text-only", "with-boosts"])
def test_top_hits_are_the_in_range_paypal_timeouts(index, finding):
    result = index.search(QUERY, WINDOW_START, WINDOW_END, finding, k=len(PAYPAL_TIMEOUTS_IN_RANGE))
    assert {h.ticket_id for h in result.hits} == PAYPAL_TIMEOUTS_IN_RANGE
    assert result.query == QUERY and result.backend == "lsa"


def test_tickets_outside_the_padded_window_are_never_returned(index):
    result = index.search(QUERY, WINDOW_START, WINDOW_END, PAYPAL_IOS_DE, k=50)
    assert (result.search_start, result.search_end) == (datetime(2026, 9, 10, 11), datetime(2026, 9, 10, 23))
    assert {h.ticket_id for h in result.hits} == IN_RANGE  # not TK000114 (10:59) or TK000115 (23:00)
    assert result.n_candidates == len(IN_RANGE)


def test_ranks_run_from_one_with_non_increasing_scores(index):
    hits = index.search(QUERY, WINDOW_START, WINDOW_END, PAYPAL_IOS_DE).hits
    assert [h.rank for h in hits] == list(range(1, CFG.top_k + 1))
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_k_overrides_the_configured_top_k(index):
    assert len(index.search(QUERY, WINDOW_START, WINDOW_END, k=3).hits) == 3
    assert len(index.search(QUERY, WINDOW_START, WINDOW_END).hits) == CFG.top_k


def test_hits_carry_the_ticket_fields(index):
    hit = next(h for h in index.search(QUERY, WINDOW_START, WINDOW_END, k=50).hits if h.ticket_id == "TK000102")
    assert type(hit.created_at) is datetime and hit.created_at == datetime(2026, 9, 10, 13, 20)
    assert (hit.channel, hit.customer_country, hit.device_os) == ("chat", "DE", "iOS")
    assert hit.text == "PayPal checkout keeps timing out in the iPhone app."


def test_nanosecond_ticket_times_convert_without_a_warning():
    # The generator draws ticket times in nanoseconds; datetime keeps microseconds.
    tickets = make_tickets(ROWS)
    tickets["created_at"] += pd.Timedelta(nanoseconds=1_234_567)
    built = TicketIndex(LSAEmbedder(), CFG).build(tickets)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # "Discarding nonzero nanoseconds" would be a UserWarning
        hits = built.search(QUERY, WINDOW_START, WINDOW_END, k=50).hits
    hit = next(h for h in hits if h.ticket_id == "TK000102")
    assert hit.created_at == datetime(2026, 9, 10, 13, 20, 0, 1234)


def test_missing_country_or_os_comes_back_as_none(index):
    hits = {h.ticket_id: h for h in index.search(QUERY, WINDOW_START, WINDOW_END, k=50).hits}
    assert (hits["TK000104"].customer_country, hits["TK000104"].device_os) == (None, "iOS")
    assert (hits["TK000105"].customer_country, hits["TK000105"].device_os) == ("DE", None)
    assert (hits["TK000110"].customer_country, hits["TK000110"].device_os) == (None, None)


def test_gt_incident_id_is_optional_and_ignored(index):
    # The fixture's frame has no gt_incident_id; this one has it, and the results are the same.
    with_truth = TicketIndex(LSAEmbedder(), CFG).build(make_tickets(ROWS).assign(gt_incident_id="INC-001"))
    assert with_truth.search(QUERY, WINDOW_START, WINDOW_END) == index.search(QUERY, WINDOW_START, WINDOW_END)


def test_boosts_reorder_equally_similar_tickets():
    text = "PayPal payment timed out on my iPhone."  # same text, so the same similarity
    rows = [
        ("TK000201", "2026-09-10 14:00", "email", "US", "Android", text),  # matches neither
        ("TK000202", "2026-09-10 14:10", "email", "US", "iOS", text),  # OS only
        ("TK000203", "2026-09-10 14:20", "email", "DE", "Android", text),  # country only
        ("TK000204", "2026-09-10 14:30", "email", "DE", "iOS", text),  # both
        ("TK000205", "2026-09-10 14:40", "chat", "GB", "Windows", "The promo code isn't applying at checkout."),
    ]
    cfg = RetrievalConfig(backend="lsa", country_boost=0.05, os_boost=0.03)
    index = TicketIndex(LSAEmbedder(), cfg).build(make_tickets(rows))

    plain = index.search(QUERY, WINDOW_START, WINDOW_END, k=4).hits
    assert [h.ticket_id for h in plain] == ["TK000201", "TK000202", "TK000203", "TK000204"]  # a tie: earliest first
    assert all(h.score == h.similarity for h in plain)

    boosted = index.search(QUERY, WINDOW_START, WINDOW_END, PAYPAL_IOS_DE, k=4).hits
    assert [h.ticket_id for h in boosted] == ["TK000204", "TK000203", "TK000202", "TK000201"]
    assert len({h.similarity for h in boosted}) == 1
    assert [h.score - h.similarity for h in boosted] == pytest.approx([0.08, 0.05, 0.03, 0.0])


# --- TicketIndex: spike statistics --------------------------------------------------------


def test_spike_statistics_match_a_hand_computed_case():
    rows = [
        ("TK000301", "2026-09-10 00:00", "email", "US", "iOS", "My parcel is late."),
        ("TK000302", "2026-09-10 02:00", "email", "DE", "iOS", "PayPal timeout again."),
        ("TK000303", "2026-09-10 10:00", "chat", "DE", "iOS", "PayPal timeout at checkout."),
        ("TK000304", "2026-09-10 11:00", "chat", "DE", "iOS", "Another PayPal timeout today."),
        ("TK000305", "2026-09-10 12:00", "email", "GB", "Android", "The password reset email never arrives."),
        ("TK000306", "2026-09-10 20:00", "email", "FR", "Windows", "The courier lost my parcel."),
    ]
    cfg = RetrievalConfig(backend="lsa", pad_before_h=1, pad_after_h=1)
    index = TicketIndex(LSAEmbedder(), cfg).build(make_tickets(rows))
    result = index.search("paypal timeout", datetime(2026, 9, 10, 10), datetime(2026, 9, 10, 12))

    # The search range [09:00, 13:00) holds the 10:00 and 11:00 PayPal tickets and the 12:00 one.
    similarity = {h.ticket_id: h.similarity for h in result.hits}
    assert min(similarity["TK000303"], similarity["TK000304"]) >= cfg.similarity_threshold > similarity["TK000305"]
    assert (result.n_candidates, result.similar_in_window) == (3, 2)
    # The tickets span 00:00-20:00 = 20h: 4h inside the range, 16h outside with one similar ticket (02:00).
    assert result.window_rate_per_hour == pytest.approx(2 / 4)
    assert result.baseline_rate_per_hour == pytest.approx(1 / 16)
    assert result.spike_ratio == pytest.approx(8.0)


def test_an_empty_window_gives_no_hits_and_zero_counts(index):
    result = index.search(QUERY, datetime(2026, 10, 1, 13), datetime(2026, 10, 1, 17), PAYPAL_IOS_DE)
    assert result.hits == []
    assert (result.n_candidates, result.similar_in_window) == (0, 0)
    assert (result.window_rate_per_hour, result.spike_ratio) == (0.0, 0.0)
    assert result.baseline_rate_per_hour > 0  # the PayPal tickets are all outside the range now


# --- TicketIndex: input validation ----------------------------------------------------------

EXTRA_ROW = ("TK000999", "2026-09-10 14:00", "chat", "DE", "iOS", "PayPal failed again.")


def test_build_accepts_created_at_as_text(index):
    as_text = TicketIndex(LSAEmbedder(), CFG).build(pd.DataFrame(ROWS, columns=TICKET_COLUMNS))
    assert as_text.search(QUERY, WINDOW_START, WINDOW_END) == index.search(QUERY, WINDOW_START, WINDOW_END)


@pytest.mark.parametrize(
    ("tickets", "message"),
    [
        (make_tickets(ROWS).drop(columns="text"), r"missing columns \['text'\]"),
        (make_tickets([]), "tickets is empty"),
        (pd.DataFrame([*ROWS, (*EXTRA_ROW[:1], "yesterday", *EXTRA_ROW[2:])], columns=TICKET_COLUMNS), "convertible"),
        (make_tickets([*ROWS, (*EXTRA_ROW[:1], None, *EXTRA_ROW[2:])]), "1 missing"),
    ],
    ids=["no-text-column", "empty", "unparseable-created-at", "missing-created-at"],
)
def test_build_rejects_bad_frames(tickets, message):
    with pytest.raises(ValueError, match=message):
        TicketIndex(LSAEmbedder(), CFG).build(tickets)


def test_search_before_build_raises():
    with pytest.raises(ValueError, match="before build"):
        TicketIndex(LSAEmbedder(), CFG).search(QUERY, WINDOW_START, WINDOW_END)


@pytest.mark.parametrize(
    ("start", "end", "k", "message"),
    [
        (WINDOW_END, WINDOW_START, None, "start must be before end"),
        (WINDOW_START, WINDOW_START, None, "start must be before end"),
        (WINDOW_START, WINDOW_END, 0, "k must be >= 1"),
    ],
    ids=["reversed", "zero-length", "k-zero"],
)
def test_search_rejects_bad_arguments(index, start, end, k, message):
    with pytest.raises(ValueError, match=message):
        index.search(QUERY, start, end, k=k)
