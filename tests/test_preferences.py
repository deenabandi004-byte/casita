"""Credential-free tests for the deterministic preference profile.

These build an in-memory SQLite matching the real schema, then assert on
behavior — decay, MIN_SUPPORT gating, no-vote no-op, walk-bucket alignment
with rank._walk_bonus, and leave-one-out exclusion.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import importlib

from casita import preferences
from casita.rank import rank
from casita.models import Listing
from casita.storage import SCHEMA, _migrate

# casita/__init__.py does `from .rank import rank`, which shadows the
# `casita.rank` submodule attribute with the function. Reach the submodule
# via importlib so we can call the private _walk_bonus helper in the
# bucket-alignment test.
rank_module = importlib.import_module("casita.rank")


UTC = timezone.utc


def _fresh_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)  # adds photo-review cols (light_quality etc.) to the base table
    return conn


def _insert_listing(conn: sqlite3.Connection, L: Listing) -> None:
    """Minimal INSERT that avoids the storage.upsert_run machinery so tests
    stay independent of scrape-run bookkeeping."""
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO listings (key, source, source_id, url, dog_policy, laundry, "
        "parking, neighborhood, light_quality, view_quality, condition_quality, "
        "first_seen, last_seen, active) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
        (L.key, L.source, L.source_id, L.url, L.dog_policy, L.laundry,
         L.parking, L.neighborhood, L.light_quality, L.view_quality,
         L.condition_quality, now, now),
    )


def _mk_listing(key_id: str, **overrides) -> Listing:
    return Listing(
        source="zillow", source_id=key_id, url=f"https://example.test/{key_id}",
        **overrides,
    )


def _insert_vote(conn, listing_key: str, direction: str, voter: str, ts: datetime,
                 reason: str = "") -> None:
    conn.execute(
        "INSERT INTO votes (listing_key, voter, direction, reason, ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (listing_key, voter, direction, reason, ts.isoformat()),
    )


# --- Walk bucket alignment ------------------------------------------------


@pytest.mark.parametrize("minutes,expected", [
    (0, "<=10"), (10, "<=10"),
    (11, "<=15"), (15, "<=15"),
    (16, "<=20"), (20, "<=20"),
    (21, "<=30"), (30, "<=30"),
    (31, ">30"), (100, ">30"),
])
def test_walk_bucket_boundaries_match_rank_walk_bonus(minutes, expected):
    """The profile's walk buckets must be the same thresholds rank._walk_bonus
    uses (sweet_spot=10). Whichever bucket a listing lands in in the
    heuristic scorer is the bucket the profile learns about."""
    assert preferences.walk_bucket(minutes) == expected


def test_walk_bucket_none_when_minutes_none():
    assert preferences.walk_bucket(None) is None


def test_walk_bucket_labels_partition_rank_walk_bonus_ranges():
    """Sanity check: each bucket label corresponds to a distinct rank._walk_bonus
    value at sweet_spot=10 (15 / 10 / 5 / 1 / -3)."""
    bonuses = {rank_module._walk_bonus(m, sweet_spot=10) for m in [5, 12, 18, 25, 40]}
    assert bonuses == {15, 10, 5, 1, -3}


# --- Decay ----------------------------------------------------------------


def test_build_profile_decay_reduces_older_vote_weight():
    """An older up-vote should carry strictly less weight than an identical
    fresh one — the whole point of recency decay."""
    conn = _fresh_conn()
    L = _mk_listing("old", dog_policy="large_ok")
    _insert_listing(conn, L)

    now = datetime(2026, 7, 1, tzinfo=UTC)
    old_ts = now - timedelta(days=90)  # 2 half-lives at default 45d
    fresh_ts = now

    _insert_vote(conn, L.key, "up", "reviewer_a", old_ts)
    old_profile = preferences.build_profile(conn, now=now)

    # Reset and re-do with a fresh vote instead.
    conn.execute("DELETE FROM votes")
    _insert_vote(conn, L.key, "up", "reviewer_a", fresh_ts)
    fresh_profile = preferences.build_profile(conn, now=now)

    old_w = old_profile.weights["dog_policy"]["large_ok"]
    fresh_w = fresh_profile.weights["dog_policy"]["large_ok"]
    assert 0 < old_w < fresh_w
    # 90 days at 45d half-life ≈ 0.25 of fresh; give a small tolerance.
    assert old_w == pytest.approx(fresh_w * 0.25, rel=0.05)


# --- MIN_SUPPORT gate -----------------------------------------------------


def test_preference_adjustment_single_event_dimension_gated_out():
    """A dimension with only one supporting event must not move the score,
    even if the raw weight is positive — mirrors the ≥2-votes bar in
    llm._ANALYZE_PREFS_SYSTEM."""
    conn = _fresh_conn()
    only = _mk_listing("only", dog_policy="large_ok")
    _insert_listing(conn, only)
    _insert_vote(conn, only.key, "up", "reviewer_a",
                 datetime(2026, 7, 1, tzinfo=UTC))

    profile = preferences.build_profile(conn, now=datetime(2026, 7, 1, tzinfo=UTC))
    other = _mk_listing("other", dog_policy="large_ok")
    assert preferences.preference_adjustment(other, profile) == 0


def test_preference_adjustment_two_events_dimension_counted():
    """Two events on the same (dim, value) clear MIN_SUPPORT and produce a
    non-zero adjustment for a matching listing."""
    conn = _fresh_conn()
    a = _mk_listing("a", dog_policy="large_ok")
    b = _mk_listing("b", dog_policy="large_ok")
    _insert_listing(conn, a); _insert_listing(conn, b)
    ts = datetime(2026, 7, 1, tzinfo=UTC)
    _insert_vote(conn, a.key, "up", "reviewer_a", ts)
    _insert_vote(conn, b.key, "up", "reviewer_a", ts)

    profile = preferences.build_profile(conn, now=ts)
    other = _mk_listing("c", dog_policy="large_ok")
    assert preferences.preference_adjustment(other, profile) > 0


# --- Cold start / no votes ------------------------------------------------


def test_preference_adjustment_no_votes_yields_zero():
    conn = _fresh_conn()
    L = _mk_listing("cold", dog_policy="large_ok", laundry="in-unit")
    _insert_listing(conn, L)
    profile = preferences.build_profile(conn, now=datetime(2026, 7, 1, tzinfo=UTC))
    assert preferences.preference_adjustment(L, profile) == 0
    assert profile.n_events == 0


def test_explain_no_matches_returns_empty_string():
    """Never fabricate a reason on cold start."""
    conn = _fresh_conn()
    L = _mk_listing("cold")
    _insert_listing(conn, L)
    profile = preferences.build_profile(conn, now=datetime(2026, 7, 1, tzinfo=UTC))
    assert preferences.explain(L, profile) == ""


def test_build_profile_hood_variants_collapse_to_same_value():
    """Zillow's title-cased "Outer Richmond" and the slugified "outer-richmond"
    are the same neighborhood. Before the normalization fix, they hashed to
    different profile keys and split credit for the same hood across two
    rival rows. Both must produce the same dim value and pool their weight."""
    conn = _fresh_conn()
    a = _mk_listing("a", dog_policy="large_ok", neighborhood="Outer Richmond")
    b = _mk_listing("b", dog_policy="large_ok", neighborhood="outer-richmond")
    _insert_listing(conn, a); _insert_listing(conn, b)
    ts = datetime(2026, 7, 1, tzinfo=UTC)
    _insert_vote(conn, a.key, "up", "reviewer_a", ts)
    _insert_vote(conn, b.key, "up", "reviewer_a", ts)

    profile = preferences.build_profile(conn, now=ts)
    # Should be one hood entry, not two.
    hood_keys = list(profile.support.get("hood", {}).keys())
    assert hood_keys == ["outer richmond"], hood_keys
    assert profile.support["hood"]["outer richmond"] == 2


def test_explain_breakdown_empty_profile_returns_empty_list():
    """The detail-page "Why this ranked here" panel omits when the profile
    is empty. Tested directly rather than inferred from explain()'s
    behavior — the two functions have separate code paths and a caller
    (listing_page.render_detail) branches on this exact return value."""
    conn = _fresh_conn()
    L = _mk_listing("cold", dog_policy="large_ok", laundry="in-unit",
                     neighborhood="Inner Richmond")
    _insert_listing(conn, L)
    profile = preferences.build_profile(conn, now=datetime(2026, 7, 1, tzinfo=UTC))
    assert profile.n_events == 0
    assert preferences.explain_breakdown(L, profile) == []


# --- passed_on as down signal --------------------------------------------


def test_collect_events_passed_on_note_becomes_down_signal():
    """A listing_status row with status='passed_on' and a non-empty note
    should register as a -1 event, matching llm._preference_examples."""
    conn = _fresh_conn()
    L = _mk_listing("passed", dog_policy="no_dogs")
    _insert_listing(conn, L)
    conn.execute(
        "INSERT INTO listing_status (listing_key, status, status_note, updated_at) "
        "VALUES (?, 'passed_on', 'no dogs', ?)",
        (L.key, datetime(2026, 7, 1, tzinfo=UTC).isoformat()),
    )
    events = preferences.collect_events(conn)
    assert len(events) == 1
    assert events[0].direction == -1
    assert events[0].listing_key == L.key


def test_collect_events_passed_on_without_note_ignored():
    """Empty-note passed_on rows are noise — the LLM prompt filters them out,
    so we do too."""
    conn = _fresh_conn()
    L = _mk_listing("empty")
    _insert_listing(conn, L)
    conn.execute(
        "INSERT INTO listing_status (listing_key, status, status_note, updated_at) "
        "VALUES (?, 'passed_on', '', ?)",
        (L.key, datetime(2026, 7, 1, tzinfo=UTC).isoformat()),
    )
    assert preferences.collect_events(conn) == []


# --- Leave-one-out exclusion ---------------------------------------------


def test_build_profile_exclude_listing_keys_drops_targeted_events():
    """Leave-one-out relies on being able to build a profile that pretends
    a specific listing was never voted on. Otherwise the eval leaks the
    label into the profile."""
    conn = _fresh_conn()
    a = _mk_listing("a", dog_policy="large_ok")
    b = _mk_listing("b", dog_policy="large_ok")
    _insert_listing(conn, a); _insert_listing(conn, b)
    ts = datetime(2026, 7, 1, tzinfo=UTC)
    _insert_vote(conn, a.key, "up", "reviewer_a", ts)
    _insert_vote(conn, b.key, "up", "reviewer_a", ts)

    full = preferences.build_profile(conn, now=ts)
    minus_a = preferences.build_profile(conn, now=ts, exclude_listing_keys={a.key})

    assert full.n_events == 2
    assert minus_a.n_events == 1
    # With only b's vote, "large_ok" has a single event → MIN_SUPPORT (2) gates it.
    assert preferences.preference_adjustment(
        _mk_listing("probe", dog_policy="large_ok"), minus_a
    ) == 0


# --- Determinism ---------------------------------------------------------


def test_build_profile_deterministic_across_repeated_calls():
    """Same conn + same anchor time → same profile. Required for the eval to
    give the same numbers on re-run, and for tests to not flake."""
    conn = _fresh_conn()
    a = _mk_listing("a", dog_policy="large_ok"); _insert_listing(conn, a)
    b = _mk_listing("b", dog_policy="large_ok"); _insert_listing(conn, b)
    ts = datetime(2026, 7, 1, tzinfo=UTC)
    _insert_vote(conn, a.key, "up", "reviewer_a", ts)
    _insert_vote(conn, b.key, "up", "reviewer_a", ts)

    p1 = preferences.build_profile(conn, now=ts)
    p2 = preferences.build_profile(conn, now=ts)
    assert p1.weights == p2.weights
    assert p1.support == p2.support


# --- rank.rank(profile=None) is a byte-identical no-op --------------------


def test_rank_with_no_profile_produces_same_order_as_before():
    """Sanity that adding the optional profile parameter didn't change the
    default behavior. Without a profile, order must be the same as before."""
    L1 = _mk_listing("l1", dog_policy="large_ok")
    L2 = _mk_listing("l2", dog_policy="small_only")
    L3 = _mk_listing("l3", dog_policy="dogs_ok")
    listings = [L1, L2, L3]
    ranked_a = rank(list(listings))
    ranked_b = rank(list(listings), profile=None)
    assert [x.key for x in ranked_a] == [x.key for x in ranked_b]
