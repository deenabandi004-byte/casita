"""Credential-free tests for rank.stable_order.

Covers the three cases that matter to the display: no snapshot → no-op,
score gap wide enough to reorder, score gap narrow enough to suppress.

Listings are constructed with dog_policy + laundry combinations whose
score() outputs are easy to reason about (see rank.score for the bonus
table): large_ok=+12, dogs_ok=+6, laundry in-unit=+3, none=-2, shared=+1.
"""
from __future__ import annotations

from casita.models import Listing
from casita.rank import STICKINESS_THRESHOLD, score, stable_order


def _mk(key_id: str, **overrides) -> Listing:
    return Listing(
        source="zillow", source_id=key_id, url=f"https://example.test/{key_id}",
        **overrides,
    )


def test_stable_order_no_snapshot_returns_input_unchanged():
    """No prior snapshot → fresh order passes through untouched. This is the
    first-run case, before storage.save_rank_snapshot has ever written a row.
    """
    a = _mk("a", dog_policy="large_ok")
    b = _mk("b", dog_policy="dogs_ok")
    c = _mk("c", dog_policy="small_only")
    fresh = [a, b, c]

    assert [x.key for x in stable_order(fresh, None, threshold=STICKINESS_THRESHOLD)] == [a.key, b.key, c.key]
    assert [x.key for x in stable_order(fresh, {}, threshold=STICKINESS_THRESHOLD)] == [a.key, b.key, c.key]


def test_stable_order_swap_allowed_when_score_delta_exceeds_threshold():
    """a is now ahead of b with a wide score gap (large_ok+in-unit=15 vs
    dogs_ok+none=4, delta=11). The previous snapshot had b ahead of a. Because
    the current gap clears the threshold, the fresh order wins.
    """
    a = _mk("a", dog_policy="large_ok", laundry="in-unit")
    b = _mk("b", dog_policy="dogs_ok", laundry="none")
    assert score(a) - score(b) > STICKINESS_THRESHOLD  # guard the premise

    prev_snapshot = {
        a.key: {"position": 1, "score": 0.0},
        b.key: {"position": 0, "score": 0.0},
    }
    out = stable_order([a, b], prev_snapshot, threshold=STICKINESS_THRESHOLD)
    assert [x.key for x in out] == [a.key, b.key]


def test_stable_order_swap_suppressed_when_score_delta_below_threshold():
    """a is now barely ahead of b (dogs_ok+in-unit=9 vs dogs_ok+shared=7,
    delta=2). Because the previous snapshot had b ahead, and the current gap
    doesn't clear the threshold, the previous relative order wins — b stays
    ahead of a even though the fresh sort put a first.
    """
    a = _mk("a", dog_policy="dogs_ok", laundry="in-unit")
    b = _mk("b", dog_policy="dogs_ok", laundry="shared (in building)")
    assert 0 < score(a) - score(b) <= STICKINESS_THRESHOLD  # guard the premise

    prev_snapshot = {
        a.key: {"position": 1, "score": 0.0},
        b.key: {"position": 0, "score": 0.0},
    }
    out = stable_order([a, b], prev_snapshot, threshold=STICKINESS_THRESHOLD)
    assert [x.key for x in out] == [b.key, a.key]
