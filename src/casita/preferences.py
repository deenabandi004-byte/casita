"""Deterministic, credential-free preference learning from vote history.

Does offline what `llm.analyze_preferences()` does with a live Gemini call:
turns vote and passed_on history into a structured, per-dimension signal that
`rank.rank()` can use to break ties. Unlike `analyze_preferences`, this never
calls an LLM, so it runs in the credential-free demo path and can be unit
tested.

Query shape overlaps with `llm._preference_examples` but this module does not
share code with it — the duplication is intentional and small. Sharing would
mean editing `llm.py`, and the point of this feature is to add a testable path
alongside the existing LLM-dependent one, not on top of it. A follow-up could
factor out a shared `iter_vote_events()` once both paths are stable.

The scoring is deliberately boring: sum of decayed weights over dimensions
that clear a small evidence bar. Every knob (half-life, MIN_SUPPORT, scaling
constant) is documented in-file so a reviewer can tell what's a modeling
choice and what's arithmetic.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from .models import Listing


# --- Tunables --------------------------------------------------------------

# Half-life for recency decay, in days. A vote this many days old counts half
# as much as a fresh one. 45 days ≈ six weekly search cycles: recent enough
# that shifting taste dominates, old enough that a single week of noisy
# feedback doesn't flip the profile. Not tuned against held-out data — this
# is a starting point, and the eval in scripts/eval_preferences.py measures
# whether the resulting profile actually helps at this setting.
HALF_LIFE_DAYS = 45.0

# Minimum number of raw events on a (dimension, value) before we let it move
# the ranking. Mirrors the ≥2-votes bar baked into `llm._ANALYZE_PREFS_SYSTEM`
# ("Only flag a pattern with real support (≥2 votes), not a one-off"). Held
# the same here on purpose: whichever audit surface a reviewer looks at, the
# evidence bar is the same.
MIN_SUPPORT = 2

# Rough magnitude ceiling for the preference adjustment. `rank.score()` runs
# in the -30..+40 range (excluding the -1000 dog-policy gate); scaling the raw
# decayed sum by ~10 keeps the adjustment comparable in magnitude to a
# meaningful heuristic term (e.g. laundry: +3, walk sweet-spot: +30) without
# ever swamping it. Documented so a reviewer can tell this is a scaling
# choice, not a magic number.
ADJUSTMENT_SCALE = 10.0

# Same buckets as rank._walk_bonus(sweet_spot=10). Kept as a tuple of
# thresholds + labels so the profile's "short trail walk" means the same
# thing as the heuristic scorer's "short trail walk". Change one and this
# has to change too — grep-friendly by design.
WALK_BUCKETS: tuple[tuple[int, str], ...] = (
    (10, "<=10"),   # sweet spot: score += 15
    (15, "<=15"),   # +10
    (20, "<=20"),   # +5
    (30, "<=30"),   # +1
)
WALK_BUCKET_FAR = ">30"  # -3 in the heuristic

DIMENSIONS: tuple[str, ...] = (
    "dog_policy",
    "laundry",
    "parking",
    "hood",
    "light_quality",
    "view_quality",
    "condition_quality",
    "trail_walk_bucket",
    "beach_walk_bucket",
)


# --- Data classes ---------------------------------------------------------


@dataclass
class VoteEvent:
    listing_key: str
    direction: int          # +1 for up, -1 for down / passed_on
    ts: datetime
    voter: str | None = None


@dataclass
class PreferenceProfile:
    """Per-dimension weight and support counts, plus a timestamp anchor.

    weights[dim][value] is the decayed sum of directional signal.
    support[dim][value] is the raw count of events touching that value —
    the ≥MIN_SUPPORT gate reads support, not weights, so a single strong
    recent up-vote doesn't get to dominate a dimension on its own.
    """
    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    support: dict[str, dict[str, int]] = field(default_factory=dict)
    n_events: int = 0
    as_of: datetime | None = None
    half_life_days: float = HALF_LIFE_DAYS


# --- Helpers --------------------------------------------------------------


def _parse_ts(raw: str | datetime | None) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    # SQLite CURRENT_TIMESTAMP writes ISO strings without a TZ. Treat as UTC.
    try:
        dt = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def walk_bucket(minutes: int | None) -> str | None:
    """Bucket a walk time into the same bins as rank._walk_bonus(sweet_spot=10)."""
    if minutes is None:
        return None
    for threshold, label in WALK_BUCKETS:
        if minutes <= threshold:
            return label
    return WALK_BUCKET_FAR


def _nearest_minutes(walk_map: dict | None, listing_key: str, anchor_names: Iterable[str]) -> int | None:
    """Minimum walk time to any of the named anchors, or None if unknown.

    We accept anchor names as an iterable rather than importing walk.TRAILS /
    walk.BEACHES so this module stays importable in tests that don't build a
    walk map.
    """
    if not walk_map:
        return None
    best: int | None = None
    for name in anchor_names:
        m = walk_map.get((listing_key, name))
        if m is None:
            continue
        if best is None or m < best:
            best = m
    return best


def _listing_dim_values(
    listing: Listing, walk_map: dict | None,
    trail_names: tuple[str, ...] | None = None,
    beach_names: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Extract the {dimension: value} snapshot for a listing.

    Values are stringified — the profile is categorical everywhere, including
    the walk-time dimensions (they bucket first, then become strings).

    Anchors are injected rather than hard-imported so tests can call this
    without touching walk.py's module-level state.
    """
    if trail_names is None or beach_names is None:
        from .walk import BEACHES, PRESIDIO_GATES  # local import: keeps preferences importable in bare-bones tests
        if trail_names is None:
            trail_names = tuple(a.name for a in PRESIDIO_GATES)
        if beach_names is None:
            beach_names = tuple(a.name for a in BEACHES)

    out: dict[str, str] = {}
    if listing.dog_policy:
        out["dog_policy"] = listing.dog_policy
    if listing.laundry:
        out["laundry"] = listing.laundry
    if listing.parking:
        # Coarsen the free-text parking field: rank.py already treats it as
        # (any / garage / none), so the profile does the same.
        p = listing.parking.lower()
        if "no parking" in p or p == "none":
            out["parking"] = "none"
        elif "garage" in p:
            out["parking"] = "garage"
        else:
            out["parking"] = "any"
    # Hood strings come in mixed forms across sources — "Outer Richmond" from
    # Zillow's title-cased label, "outer-richmond" from the slugified variant,
    # sometimes with extra whitespace. Normalize separators so they collapse
    # to the same bucket, otherwise the profile splits credit for the same
    # hood across two rival keys and under-scores both.
    hood = (listing.hood or "").strip().lower().replace("-", " ")
    hood = " ".join(hood.split())
    if hood:
        out["hood"] = hood
    if listing.light_quality:
        out["light_quality"] = listing.light_quality
    if listing.view_quality:
        out["view_quality"] = listing.view_quality
    if listing.condition_quality:
        out["condition_quality"] = listing.condition_quality

    trail = _nearest_minutes(walk_map, listing.key, trail_names)
    beach = _nearest_minutes(walk_map, listing.key, beach_names)
    tb = walk_bucket(trail)
    bb = walk_bucket(beach)
    if tb is not None:
        out["trail_walk_bucket"] = tb
    if bb is not None:
        out["beach_walk_bucket"] = bb
    return out


# --- Event collection -----------------------------------------------------


def collect_events(
    conn: sqlite3.Connection, *, before: datetime | None = None,
) -> list[VoteEvent]:
    """Up votes + down votes + passed_on notes.

    Matches `llm._preference_examples`'s UP/DOWN sourcing: `votes` for both
    directions, plus `listing_status.status='passed_on'` with a non-empty note
    as an additional DOWN signal (passed_on wins on overlap by dedup order —
    same behavior as the LLM path). `before` is an optional strict cutoff for
    time-based splits; leave-one-out uses the exclusion path in
    build_profile() instead, since it needs to remove a specific listing not
    just older events.
    """
    events: list[VoteEvent] = []

    for row in conn.execute(
        "SELECT listing_key, voter, direction, ts FROM votes ORDER BY id"
    ):
        ts = _parse_ts(row["ts"])
        if ts is None:
            continue
        if before is not None and ts >= before:
            continue
        d = 1 if row["direction"] == "up" else -1 if row["direction"] == "down" else 0
        if d == 0:
            continue
        events.append(VoteEvent(row["listing_key"], d, ts, row["voter"]))

    # passed_on with a non-empty note = down signal. Voter looked up from the
    # matching `actions` row (same lookup shape as _preference_examples).
    for row in conn.execute(
        "SELECT s.listing_key AS key, s.updated_at AS ts, "
        "(SELECT a.voter FROM actions a WHERE a.listing_key = s.listing_key "
        " AND a.kind='eliminate' AND json_extract(a.payload_json,'$.new')='passed_on' "
        " ORDER BY a.id DESC LIMIT 1) AS voter "
        "FROM listing_status s WHERE s.status='passed_on' "
        "AND s.status_note IS NOT NULL AND s.status_note != ''"
    ):
        ts = _parse_ts(row["ts"])
        if ts is None:
            continue
        if before is not None and ts >= before:
            continue
        events.append(VoteEvent(row["key"], -1, ts, row["voter"]))

    return events


# --- Profile construction --------------------------------------------------


def build_profile(
    conn: sqlite3.Connection,
    walk_map: dict | None = None,
    *,
    before: datetime | None = None,
    exclude_listing_keys: Iterable[str] | None = None,
    half_life_days: float = HALF_LIFE_DAYS,
    now: datetime | None = None,
) -> PreferenceProfile:
    """Walk events, join each event's listing, accumulate decayed weight
    per (dimension, value).

    Determinism: same conn + same `before` + same `exclude_listing_keys`
    always produces the same profile. Required for the eval to make sense
    and for tests to be reproducible.
    """
    events = collect_events(conn, before=before)
    excluded = set(exclude_listing_keys or ())
    if excluded:
        events = [e for e in events if e.listing_key not in excluded]

    if not events:
        return PreferenceProfile(as_of=now or datetime.now(timezone.utc),
                                 half_life_days=half_life_days)

    keys = [e.listing_key for e in events]
    placeholders = ",".join("?" * len(keys))
    rows_by_key: dict[str, sqlite3.Row] = {}
    for r in conn.execute(
        f"SELECT * FROM listings WHERE key IN ({placeholders})", keys
    ):
        rows_by_key[r["key"]] = r

    # Local import to avoid a hard walk.py dependency at import time.
    from .walk import BEACHES, PRESIDIO_GATES
    from . import storage
    trail_names = tuple(a.name for a in PRESIDIO_GATES)
    beach_names = tuple(a.name for a in BEACHES)

    as_of = now or max(e.ts for e in events)
    weights: dict[str, dict[str, float]] = {}
    support: dict[str, dict[str, int]] = {}

    for event in events:
        row = rows_by_key.get(event.listing_key)
        if row is None:
            continue  # listing dropped from DB — reason is durable, but we
                     # can't join to dimension values without it
        listing = storage._row_to_listing(row)
        dims = _listing_dim_values(listing, walk_map, trail_names, beach_names)
        if not dims:
            continue
        age_days = max(0.0, (as_of - event.ts).total_seconds() / 86400.0)
        decay = 0.5 ** (age_days / half_life_days)
        for dim, value in dims.items():
            weights.setdefault(dim, {})
            support.setdefault(dim, {})
            weights[dim][value] = weights[dim].get(value, 0.0) + event.direction * decay
            support[dim][value] = support[dim].get(value, 0) + 1

    return PreferenceProfile(
        weights=weights, support=support, n_events=len(events),
        as_of=as_of, half_life_days=half_life_days,
    )


# --- Scoring + explanation ------------------------------------------------


def preference_adjustment(
    listing: Listing, profile: PreferenceProfile, walk_map: dict | None = None,
) -> int:
    """Sum profile weights for this listing's values, only counting
    dimensions/values that clear MIN_SUPPORT. Scales to roughly match
    rank.score()'s natural magnitude so it can shift ties without ever
    overriding a strong heuristic signal.

    Returns 0 (not None) for cold-start / no-support cases — the caller can
    always add it in unconditionally.
    """
    if not profile.weights:
        return 0
    dims = _listing_dim_values(listing, walk_map)
    total = 0.0
    for dim, value in dims.items():
        support = profile.support.get(dim, {}).get(value, 0)
        if support < MIN_SUPPORT:
            continue
        total += profile.weights.get(dim, {}).get(value, 0.0)
    return int(round(total * ADJUSTMENT_SCALE))


def explain(
    listing: Listing, profile: PreferenceProfile, walk_map: dict | None = None,
    top_n: int = 3,
) -> str:
    """One short sentence naming up to `top_n` positive dimensions this
    listing matches. Empty string when nothing clears MIN_SUPPORT — never
    fabricate a reason.
    """
    hits = _positive_hits(listing, profile, walk_map)
    if not hits:
        return ""
    hits = hits[:top_n]
    phrases = [_phrase_for(dim, value) for dim, value, _ in hits]
    phrases = [p for p in phrases if p]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return f"Matches a learned preference: {phrases[0]}."
    return "Matches learned preferences: " + ", ".join(phrases[:-1]) + f", and {phrases[-1]}."


def explain_breakdown(
    listing: Listing, profile: PreferenceProfile, walk_map: dict | None = None,
) -> list[dict]:
    """Full per-dimension breakdown for the detail-page panel.

    One row per dimension that (a) has ≥MIN_SUPPORT for some value in the
    profile AND (b) is present on this listing. Marks whether the listing's
    value matches a value the profile learned about, and returns the
    contributed weight so the panel can order by impact.
    """
    if not profile.weights:
        return []
    dims = _listing_dim_values(listing, walk_map)
    out: list[dict] = []
    for dim in DIMENSIONS:
        dim_support = profile.support.get(dim, {})
        gated = {v: c for v, c in dim_support.items() if c >= MIN_SUPPORT}
        if not gated:
            continue
        value = dims.get(dim)
        if value is None:
            continue
        matches = value in gated
        weight = profile.weights.get(dim, {}).get(value, 0.0) if matches else 0.0
        out.append({
            "dimension": dim,
            "value": value,
            "matches": matches,
            "weight": weight,
            "support": gated.get(value, 0),
        })
    out.sort(key=lambda r: -abs(r["weight"]))
    return out


def _positive_hits(
    listing: Listing, profile: PreferenceProfile, walk_map: dict | None,
) -> list[tuple[str, str, float]]:
    """(dim, value, weight) for dimensions where this listing lands on a
    positively-weighted, adequately-supported value, ordered strongest first.
    """
    if not profile.weights:
        return []
    dims = _listing_dim_values(listing, walk_map)
    hits: list[tuple[str, str, float]] = []
    for dim, value in dims.items():
        support = profile.support.get(dim, {}).get(value, 0)
        if support < MIN_SUPPORT:
            continue
        w = profile.weights.get(dim, {}).get(value, 0.0)
        if w > 0:
            hits.append((dim, value, w))
    hits.sort(key=lambda t: -t[2])
    return hits


def _phrase_for(dim: str, value: str) -> str:
    """Human-readable phrase for a (dim, value) pair — kept small on purpose;
    unfamiliar dimensions fall through to a generic form rather than crash.
    """
    if dim == "dog_policy":
        return {"large_ok": "large-dog friendly", "dogs_ok": "dogs allowed"}.get(value, f"dog policy {value}")
    if dim == "laundry":
        return f"{value} laundry"
    if dim == "parking":
        return {"garage": "garage parking", "any": "on-site parking"}.get(value, "")
    if dim == "hood":
        return f"neighborhood: {value}"
    if dim == "light_quality":
        return f"{value} light"
    if dim == "view_quality":
        return f"{value} view"
    if dim == "condition_quality":
        return f"{value} condition"
    if dim == "trail_walk_bucket":
        return f"trail walk {value} min"
    if dim == "beach_walk_bucket":
        return f"beach walk {value} min"
    return f"{dim}={value}"
