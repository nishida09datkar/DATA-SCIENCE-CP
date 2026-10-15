"""Time-windowed nearest-neighbour search over support tickets.

build() sorts the tickets by created_at, embeds their texts and adds the
vectors to a FAISS IndexFlatIP (exact inner product, which is the cosine
similarity because the rows have unit length). Row i of the sorted frame is
FAISS id i, so the tickets of any time range are one contiguous id range
[lo, hi), found with np.searchsorted on the sorted times.

search() then:
  1. pads the anomaly window (customers write in after a failure starts and
     keep writing after it ends),
  2. asks FAISS for the most similar tickets with ids in [lo, hi) only,
  3. adds a small boost when a ticket's country / OS equals the finding's
     and re-ranks by that score,
  4. counts the tickets that look like the query (similarity >= threshold)
     per hour inside vs outside the range. A real incident shows up as a
     spike in that rate (spike_ratio well above 1).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import faiss
import numpy as np
import pandas as pd

from kpi_rca.config import RetrievalConfig
from kpi_rca.retrieval.embeddings import Embedder
from kpi_rca.schemas import RetrievalResult, SegmentFinding, TicketHit

REQUIRED_COLUMNS = ("ticket_id", "created_at", "channel", "customer_country", "device_os", "text")
_NS_PER_HOUR = 3_600 * 10**9


def _to_ns(when: datetime) -> int:
    return pd.Timestamp(when).as_unit("ns").value


def _none_if_missing(value: object) -> str | None:
    return None if pd.isna(value) else str(value)


def _matches(ticket_value: object, wanted: str | None) -> bool:
    """The finding has this attribute and the ticket has the same value (a missing value never matches)."""
    return wanted is not None and not pd.isna(ticket_value) and ticket_value == wanted


class TicketIndex:
    """Build once over all tickets, then search() once per anomaly window."""

    def __init__(self, embedder: Embedder, cfg: RetrievalConfig) -> None:
        self.embedder = embedder
        self.cfg = cfg
        self._tickets: pd.DataFrame | None = None  # sorted by created_at; row i is FAISS id i
        self._times_ns: np.ndarray | None = None  # created_at as int64 nanoseconds, for searchsorted
        self._vectors: np.ndarray | None = None
        self._index: faiss.IndexFlatIP | None = None

    def build(self, tickets: pd.DataFrame) -> TicketIndex:
        """Index the tickets. Only REQUIRED_COLUMNS are read; gt_incident_id and other columns are ignored."""
        missing = [c for c in REQUIRED_COLUMNS if c not in tickets.columns]
        if missing:
            raise ValueError(f"tickets is missing columns {missing}")
        if tickets.empty:
            raise ValueError("tickets is empty; there is nothing to index")
        try:
            created = pd.to_datetime(tickets["created_at"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tickets.created_at is not convertible to datetime: {exc}") from exc
        if created.isna().any():
            # A missing time would break the sorted order that searchsorted relies on.
            raise ValueError(f"tickets.created_at has {int(created.isna().sum())} missing value(s)")
        # TicketHit.created_at is a Python datetime, which stops at microseconds. Truncate here so
        # the index searches exactly the times it reports (and to_pydatetime() loses nothing).
        created = created.dt.floor("us")

        frame = (
            tickets[list(REQUIRED_COLUMNS)]
            .assign(created_at=created)
            .sort_values("created_at", kind="stable")
            .reset_index(drop=True)
        )
        texts = frame["text"].tolist()
        self.embedder.fit(texts)
        vectors = self.embedder.encode(texts)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._tickets = frame
        self._times_ns = frame["created_at"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self._vectors = vectors
        self._index = index
        return self

    def search(
        self,
        query: str,
        start: datetime,
        end: datetime,
        finding: SegmentFinding | None = None,
        k: int | None = None,
    ) -> RetrievalResult:
        """The k best tickets created in [start - pad_before_h, end + pad_after_h), plus the spike statistics.

        `finding` only supplies the country / OS boosts; k defaults to cfg.top_k.
        """
        if self._index is None:
            raise ValueError("TicketIndex.search() called before build()")
        if not start < end:
            raise ValueError(f"start must be before end, got start={start} end={end}")
        k = self.cfg.top_k if k is None else k
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        search_start = start - timedelta(hours=self.cfg.pad_before_h)
        search_end = end + timedelta(hours=self.cfg.pad_after_h)
        bounds = np.searchsorted(self._times_ns, [_to_ns(search_start), _to_ns(search_end)])
        lo, hi = int(bounds[0]), int(bounds[1])  # python ints: faiss.IDSelectorRange rejects np.int64
        q = self.embedder.encode([query])
        return RetrievalResult(
            query=query,
            backend=self.embedder.name,
            search_start=search_start,
            search_end=search_end,
            hits=self._top_hits(q, lo, hi, finding, k),
            **self._spike_stats(q[0], lo, hi, search_start, search_end),
        )

    def _top_hits(
        self, q: np.ndarray, lo: int, hi: int, finding: SegmentFinding | None, k: int
    ) -> list[TicketHit]:
        # Fetch more than k: a boost can lift a ticket from just below the top k by similarity.
        pool = min(max(k, self.cfg.overfetch), hi - lo)
        if pool == 0:
            return []  # nothing in the range (and FAISS refuses to search for 0 neighbours)
        params = faiss.SearchParameters(sel=faiss.IDSelectorRange(lo, hi))
        similarities, ids = self._index.search(q, pool, params=params)

        wanted = {a.attribute: a.value for a in finding.attributes} if finding is not None else {}
        scored = []
        for sim, row in zip(similarities[0].tolist(), ids[0].tolist()):
            if row >= 0:  # FAISS reports "no result" as id -1
                scored.append((sim + self._boost(row, wanted), sim, row))
        # Highest score first; ties go to the higher similarity, then the earlier ticket (rows are in time order).
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        return [self._hit(rank, row, sim, score) for rank, (score, sim, row) in enumerate(scored[:k], start=1)]

    def _boost(self, row: int, wanted: dict[str, str]) -> float:
        ticket = self._tickets.iloc[row]
        boost = 0.0
        if _matches(ticket["customer_country"], wanted.get("country")):
            boost += self.cfg.country_boost
        if _matches(ticket["device_os"], wanted.get("os")):
            boost += self.cfg.os_boost
        return boost

    def _hit(self, rank: int, row: int, similarity: float, score: float) -> TicketHit:
        ticket = self._tickets.iloc[row]
        return TicketHit(
            rank=rank,
            ticket_id=str(ticket["ticket_id"]),
            created_at=ticket["created_at"].to_pydatetime(),
            channel=str(ticket["channel"]),
            customer_country=_none_if_missing(ticket["customer_country"]),
            device_os=_none_if_missing(ticket["device_os"]),
            text=str(ticket["text"]),
            similarity=similarity,
            score=score,
        )

    def _spike_stats(
        self, q: np.ndarray, lo: int, hi: int, search_start: datetime, search_end: datetime
    ) -> dict[str, int | float]:
        # FAISS does the top-k retrieval above. Counting similar tickets needs every similarity,
        # so this is one matrix-vector product over all tickets: O(n·d), cheap at ticket volumes.
        similar = (self._vectors @ q) >= self.cfg.similarity_threshold
        similar_in_window = int(similar[lo:hi].sum())
        similar_outside = int(similar.sum()) - similar_in_window

        # Hours of the observed span (first to last ticket) inside and outside the search range.
        first, last = int(self._times_ns[0]), int(self._times_ns[-1])
        overlap_ns = max(0, min(_to_ns(search_end), last) - max(_to_ns(search_start), first))
        in_hours = overlap_ns / _NS_PER_HOUR
        out_hours = (last - first) / _NS_PER_HOUR - in_hours

        window_rate = similar_in_window / in_hours if in_hours > 0 else 0.0
        baseline_rate = similar_outside / out_hours if out_hours > 0 else 0.0
        return {
            "n_candidates": hi - lo,
            "similar_in_window": similar_in_window,
            "window_rate_per_hour": window_rate,
            "baseline_rate_per_hour": baseline_rate,
            "spike_ratio": window_rate / max(baseline_rate, 1e-9),
        }
