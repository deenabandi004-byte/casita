"""Stickiness eval: how much does rank.stable_order actually damp reshuffling?

Uses fixtures/demo.sqlite only. Runs credential-free (no Vertex, no Maps API,
no network). One number to trust:

    Number of adjacent swaps between run 1 and run 2, with stickiness OFF vs ON.

An "adjacent swap" is a pair (a, b) that was adjacent in run 1's order and
appears with b before a in run 2's order. It's a direct, easy-to-explain
measure of "how much did the displayed list scramble."

Both runs share the same listings; the second run applies a small
deterministic perturbation that mirrors the two realistic churn sources
between renders:
  - ±2 on llm_rank (the LLM re-ran and produced a slightly different rank)
  - ±3 minutes on every walk-map entry (Google Routes returned a slightly
    different walk time, or a cached anchor set drifted)
Most listings' heuristic scores stay unchanged; only walk-time perturbations
that cross a rank._walk_bonus bucket boundary actually move the score. So
under stickiness, the llm_rank flips that don't also come with a real
heuristic-score improvement should get suppressed.
"""
from __future__ import annotations

import os
import random
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

# Make src/ importable when the script is run directly (uv run python scripts/...).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Force offline route mode before importing walk — walk.populate_for will
# read the cached route matrix bundled with the fixture instead of calling
# the Google Routes API.
os.environ.setdefault("CASITA_ROUTES_OFFLINE", "1")

from casita import storage, walk  # noqa: E402
from casita.models import Listing  # noqa: E402
from casita.rank import STICKINESS_THRESHOLD, rank  # noqa: E402


FIXTURE = ROOT / "fixtures" / "demo.sqlite"

# Perturbation magnitudes. ±2 on llm_rank flips adjacent pairs whose ranks
# are close, without wholesale re-shuffling. ±3 on walk minutes is small
# enough that most walk times stay in their existing rank._walk_bonus bucket
# (buckets are 5 minutes wide), but a handful cross boundaries — that's the
# interesting regime where the heuristic actually moves.
LLM_RANK_NOISE = 2
WALK_NOISE_MINUTES = 3
PERTURB_SEED = 4242


def _copy_fixture_to_tmp() -> Path:
    """Same pattern as eval_preferences and the demo command: never touch the
    checked-in fixture."""
    tmp = Path(tempfile.mkdtemp(prefix="casita-stickiness-")) / "demo.sqlite"
    shutil.copy2(FIXTURE, tmp)
    return tmp


def _perturb_walk_map(walk_map: dict, *, seed: int) -> dict:
    """Add small deterministic noise to every walk-time entry.

    Uses one seeded RNG so the same fixture always produces the same
    perturbation — the eval numbers reproduce across runs, and any change
    in the perturbation logic is a visible diff instead of drift.
    """
    r = random.Random(seed)
    return {
        key: max(0, minutes + r.randint(-WALK_NOISE_MINUTES, WALK_NOISE_MINUTES))
        for key, minutes in walk_map.items()
    }


def _perturb_listings(listings: list[Listing], *, seed: int) -> list[Listing]:
    """Copy each listing and jitter its llm_rank by a small integer amount.

    Mirrors what an LLM re-run actually looks like — most items stay put or
    move by 1-2 ranks. model_copy keeps the input list unmodified so the
    baseline order can be re-derived from the originals if we ever need to.
    """
    r = random.Random(seed)
    out: list[Listing] = []
    for L in listings:
        if L.llm_rank is None:
            out.append(L.model_copy())
            continue
        delta = r.randint(-LLM_RANK_NOISE, LLM_RANK_NOISE)
        out.append(L.model_copy(update={"llm_rank": max(1, L.llm_rank + delta)}))
    return out


def _snapshot_from_order(ordered_keys: list[str]) -> dict[str, dict]:
    """In-memory snapshot in the format storage.load_rank_snapshot returns.

    Scores are placeholder 0.0s because stable_order only reads `position`
    for the ordering decision — score in the snapshot is stored for callers
    that want to inspect it, not consumed by stable_order itself.
    """
    return {k: {"position": i, "score": 0.0} for i, k in enumerate(ordered_keys)}


def _adjacent_swap_count(baseline: list[str], candidate: list[str]) -> int:
    """For each adjacent pair in `baseline`, count 1 if they appear inverted
    in `candidate`. Bounded by len(baseline) - 1."""
    pos = {k: i for i, k in enumerate(candidate)}
    count = 0
    for i in range(len(baseline) - 1):
        a, b = baseline[i], baseline[i + 1]
        if a in pos and b in pos and pos[a] > pos[b]:
            count += 1
    return count


def _moved_listings_count(baseline: list[str], candidate: list[str]) -> int:
    """How many listings ended up at a different index?"""
    pos = {k: i for i, k in enumerate(candidate)}
    return sum(1 for i, k in enumerate(baseline) if pos.get(k, i) != i)


def main() -> int:
    demo_db = _copy_fixture_to_tmp()
    os.environ["CASITA_DB_PATH"] = str(demo_db)
    os.environ["CASITA_ROUTE_CACHE_DB"] = str(demo_db)
    os.environ["CASITA_ROUTES_OFFLINE"] = "1"

    with storage.connect() as conn:
        listings = storage.active_listings(conn)
        walk_map = walk.populate_for(listings)

    # Run 1 — baseline ordering on the unmodified inputs.
    baseline = [L.key for L in rank(listings, walk_map)]
    snapshot = _snapshot_from_order(baseline)

    # Run 2 — perturbed listings + perturbed walk_map. First without
    # stickiness, then with. The delta is the value stable_order is buying.
    perturbed_listings = _perturb_listings(listings, seed=PERTURB_SEED)
    perturbed_walk = _perturb_walk_map(walk_map, seed=PERTURB_SEED)
    run2_off = [L.key for L in rank(perturbed_listings, perturbed_walk)]
    run2_on = [L.key for L in rank(
        perturbed_listings, perturbed_walk, prev_snapshot=snapshot,
        stickiness_threshold=STICKINESS_THRESHOLD,
    )]

    swaps_off = _adjacent_swap_count(baseline, run2_off)
    swaps_on = _adjacent_swap_count(baseline, run2_on)
    moved_off = _moved_listings_count(baseline, run2_off)
    moved_on = _moved_listings_count(baseline, run2_on)

    n = len(baseline)
    print("Stickiness eval (run twice, count reshuffling)")
    print(f"  fixture: {FIXTURE.name} ({n} active listings)")
    print(f"  perturbation: llm_rank ±{LLM_RANK_NOISE}, walk-map ±{WALK_NOISE_MINUTES} min "
          f"(seed={PERTURB_SEED})")
    print(f"  threshold: {STICKINESS_THRESHOLD}")
    print()
    print(f"  adjacent swaps   OFF={swaps_off:<4d}  ON={swaps_on:<4d}"
          f"  delta={swaps_on - swaps_off:+d}")
    print(f"  listings moved   OFF={moved_off:<4d}  ON={moved_on:<4d}"
          f"  delta={moved_on - moved_off:+d}")
    print()
    print("Expected direction: ON should have fewer swaps than OFF — small,")
    print("meaningless perturbations are suppressed, so the displayed list")
    print("stays closer to the run-1 baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
