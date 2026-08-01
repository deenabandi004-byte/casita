"""Leave-one-out eval: does the preference profile move liked listings up?

Uses fixtures/demo.sqlite only. Runs credential-free (no Vertex, no Maps API,
no network). One number to trust:

    Avg percentile-rank of held-out liked listings, baseline vs adjusted.

Compares rank.score() DIRECTLY vs rank.score() + preference_adjustment().
Deliberately does NOT go through rank.rank(), which has a bucket system
that already puts net-upvoted listings above everything else regardless of
score — sending held-out likes through rank() would show a falsely perfect
result driven by the bucket, not by the preference adjustment.

For each liked listing L:
  1. Build a profile from every event EXCEPT L's own (leave-one-out).
  2. Score all active listings with baseline = rank.score(walk_map).
  3. Score all active listings with adjusted = baseline + adjustment(profile).
  4. Record L's percentile position (lower = closer to the top) under both.

Symmetric check on passed_on listings: if the profile is working, disliked
listings should get a slightly WORSE (higher) percentile under the adjusted
score, not better. That's the "we didn't accidentally build a random
positive constant" check.

Small-n caveat is printed by the script itself, not just this docstring —
16 up-votes and ~18 non-empty passed_on notes in the fixture, so this is a
directional signal, not a validated model. That distinction matters here.
"""
from __future__ import annotations

import copy
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Make src/ importable when the script is run directly (uv run python scripts/...).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import os

# Force offline route mode BEFORE importing walk — it reads the env at
# call-time but we want to be defensive; the fixture bundles the cached
# route matrix that walk.populate_for() will read instead of calling the
# Google Routes API.
os.environ.setdefault("CASITA_ROUTES_OFFLINE", "1")

from casita import preferences, storage, walk  # noqa: E402
from casita.rank import score  # noqa: E402


FIXTURE = ROOT / "fixtures" / "demo.sqlite"


def _percentile(target_key: str, ordered_keys: list[str]) -> float:
    """Where does target_key land in ordered_keys? 0.0 = first, 1.0 = last.

    Uses (index / max(1, n-1)) rather than (index / n) so the very top slot
    is exactly 0.0 for readability. n=1 would return 0.0.
    """
    n = len(ordered_keys)
    if n <= 1:
        return 0.0
    return ordered_keys.index(target_key) / (n - 1)


def _copy_fixture_to_tmp() -> Path:
    """The demo command uses this same pattern: never touch the checked-in
    fixture. Keeps the script safe to re-run without corrupting the golden."""
    tmp = Path(tempfile.mkdtemp(prefix="casita-eval-")) / "demo.sqlite"
    shutil.copy2(FIXTURE, tmp)
    return tmp


def _liked_keys(conn: sqlite3.Connection) -> list[str]:
    """Listings with at least one up-vote AND still present in the listings
    table. A vote against a since-purged listing can't be ranked."""
    rows = conn.execute(
        "SELECT DISTINCT v.listing_key FROM votes v "
        "JOIN listings l ON l.key = v.listing_key "
        "WHERE v.direction = 'up' "
        "ORDER BY v.listing_key"
    ).fetchall()
    return [r[0] for r in rows]


def _passed_keys(conn: sqlite3.Connection) -> list[str]:
    """Listings marked passed_on with a non-empty note AND still present."""
    rows = conn.execute(
        "SELECT s.listing_key FROM listing_status s "
        "JOIN listings l ON l.key = s.listing_key "
        "WHERE s.status = 'passed_on' "
        "  AND s.status_note IS NOT NULL AND s.status_note != '' "
        "ORDER BY s.listing_key"
    ).fetchall()
    return [r[0] for r in rows]


def _rank_by_score(listings, walk_map, *, profile=None) -> list[str]:
    """Return listing keys sorted best (highest score) first.

    Ties broken by key so ordering is deterministic — needed for stable
    percentile numbers across runs."""
    def s(L):
        base = score(L, walk_map)
        if profile is not None:
            base += preferences.preference_adjustment(L, profile, walk_map)
        return base
    return [L.key for L in sorted(listings, key=lambda L: (-s(L), L.key))]


def _loo_percentiles(
    conn, target_keys: list[str], all_listings, walk_map,
) -> tuple[float, float]:
    """For each target key: baseline percentile, adjusted percentile.

    Leave-one-out exclusion: profile is built from every event EXCEPT any
    event whose listing_key is the current target. Baseline profile is None
    so the baseline is the pure heuristic — same score() the codebase
    already ships."""
    baseline_pcts: list[float] = []
    adjusted_pcts: list[float] = []

    # Baseline order doesn't depend on the fold, so compute once.
    baseline_order = _rank_by_score(all_listings, walk_map, profile=None)

    for target in target_keys:
        profile = preferences.build_profile(
            conn, walk_map, exclude_listing_keys={target},
            now=datetime.now(timezone.utc),
        )
        adjusted_order = _rank_by_score(all_listings, walk_map, profile=profile)
        baseline_pcts.append(_percentile(target, baseline_order))
        adjusted_pcts.append(_percentile(target, adjusted_order))

    return (
        sum(baseline_pcts) / len(baseline_pcts),
        sum(adjusted_pcts) / len(adjusted_pcts),
    )


def _fmt_row(name: str, n: int, baseline: float, adjusted: float) -> str:
    delta = adjusted - baseline
    return (f"{name:<20s} n={n:<3d}  baseline={baseline:.3f}  "
            f"adjusted={adjusted:.3f}  delta={delta:+.3f}")


def main() -> int:
    demo_db = _copy_fixture_to_tmp()
    os.environ["CASITA_DB_PATH"] = str(demo_db)
    os.environ["CASITA_ROUTE_CACHE_DB"] = str(demo_db)
    os.environ["CASITA_ROUTES_OFFLINE"] = "1"

    with storage.connect() as conn:
        listings = storage.active_listings(conn)
        # populate_for will use the cached route matrix in the fixture — no
        # network required. Same code path the demo render uses.
        walk_map = walk.populate_for(listings)

        liked = _liked_keys(conn)
        passed = _passed_keys(conn)

        # Filter out targets that don't appear in the active_listings set —
        # LOO on a non-active listing can't be scored the same way as the
        # rest of the set.
        active_keys = {L.key for L in listings}
        liked = [k for k in liked if k in active_keys]
        passed = [k for k in passed if k in active_keys]

        if not liked:
            print("No liked listings in fixture — nothing to evaluate.")
            return 1

        base_liked, adj_liked = _loo_percentiles(conn, liked, listings, walk_map)
        base_passed, adj_passed = (
            _loo_percentiles(conn, passed, listings, walk_map)
            if passed else (float("nan"), float("nan"))
        )

    print("Preference-adjustment eval (leave-one-out)")
    print(f"  fixture: {FIXTURE.name} ({len(listings)} active listings)")
    print("  lower percentile = closer to the top of the ranked list")
    print()
    print(_fmt_row("liked listings", len(liked), base_liked, adj_liked))
    if passed:
        print(_fmt_row("passed listings", len(passed), base_passed, adj_passed))
    print()
    print("Expected direction: liked delta < 0 (moves up), "
          "passed delta > 0 (moves down).")
    print("Small-n caveat: this fixture holds ~16 up-votes and ~18 non-empty "
          "passed_on notes.")
    print("Treat this as a directional signal, not a validated model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
