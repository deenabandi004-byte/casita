# Why this change

`analyze_preferences()` in `llm.py` already tries to learn from votes. It sends every
vote and reason to Gemini and asks it to compare them against the static ranking
policy. Problem: it's one live LLM call, needs Vertex credentials, and this repo is
built so the demo and tests run without any. So `analyze_preferences` has never
been tested. Nobody actually knows if it works.

I rebuilt the same idea without the LLM call. A preference profile gets computed
straight from vote history: recency-decayed weights per dimension (laundry, parking,
dog policy, light/view/condition quality, walk-time buckets), gated so a dimension
only counts once it has real support (2+ votes, the same bar `_ANALYZE_PREFS_SYSTEM`
already uses). It plugs into the existing ranker through one optional parameter.
`profile=None` reproduces the old ranking exactly, byte for byte, so nothing about
the current app changes unless this feature is actually used.

## Why this gap

Casita's docs leave "Ways This Could Go Further" notes on most pages, and I
looked at routing anchor sets, photo-eval fixture replay, and splitting the CLI
module before picking this one.

Two things pointed here specifically.

`docs/architecture.md` names "LLM calls are Vertex-only" as a known rough edge, and
`docs/how-it-works/learning.md` says the vote loop "could gain better fixtures,
better diff output, or clearer aging of old examples." That's a real, named gap,
not something invented to have a task.

The job posting also names evals directly: "the reasoning, the wearable data behind
it, the evals that catch it being wrong." A feature that ships with a real
leave-one-out eval on real vote data is evidence of that, not a claim about it.

## Why this is close to what I already build

At Offerloop I own the ML and data infrastructure: a LightGBM lambdarank model
over a `recommendation_events` log that ranks contact and job matches, and a
warmth-scoring layer that prioritizes leads by type (alumni, dream_company,
recent_transition) based on what actually converts, not what a static rule
guesses will convert.

The part I actually like about that work is the auditing. A ranker that produces
a number nobody can interrogate is a ranker nobody should trust, including me. So
the choices I made here are the ones I make there: deterministic over LLM where
the deterministic version is good enough, weights you can read off a table
instead of a score with no explanation, a support gate so one fluke vote can't
move anything, and an eval before believing any of it. When `dated` showed up
with positive weight, the reflex to go query the fixture rather than explain it
away is the same reflex that catches the lambdarank model overfitting to a noisy
feature.

Casita's votes table is basically `recommendation_events` with a smaller schema.
Same problem, less data.

## What I built

- `src/casita/preferences.py`: `collect_events`, `build_profile`,
  `preference_adjustment`, `explain`, `explain_breakdown`. 45-day half-life on vote
  weight, `MIN_SUPPORT=2` matching `_ANALYZE_PREFS_SYSTEM`'s own ≥2-votes rule,
  walk buckets aligned to `rank._walk_bonus(sweet_spot=10)`.
- `rank.py`: one optional `profile=` parameter, used only in the tie-break term
  inside the existing buckets. `profile=None` is a no-op, verified by test.
- `tests/test_preferences.py`: 24 tests covering decay, the support gate, cold
  start, `passed_on` as a down-signal, leave-one-out exclusion, walk-bucket
  alignment, hood-variant normalization, and the no-profile no-op.
- `scripts/eval_preferences.py`: leave-one-out cross-validation. Compares
  `rank.score()` against `rank.score() + preference_adjustment()` directly, not
  through `rank.rank()`, whose favorites bucket would otherwise put held-out likes
  at the top for the wrong reason.
- `docs/how-it-works/preferences.md`: written in the repo's existing doc voice,
  including a Ways This Could Go Further section.
- Card fallback text, plus a "Why this ranked here" panel on the detail page. The
  panel only shows rows where this specific listing's value clears `MIN_SUPPORT`,
  so it isn't padded with dimensions the profile hasn't actually formed an opinion
  on for that listing.

## The number

Leave-one-out eval against `fixtures/demo.sqlite`. Lower percentile is closer to
the top of the ranked list.

|                  |  n | baseline | adjusted |  delta |
| ---------------- | -: | -------: | -------: | -----: |
| liked listings   |  9 |    0.409 |    0.311 | -0.098 |
| passed listings  | 15 |    0.343 |    0.323 | -0.020 |

Liked listings move up about 10 percentile points on average. Real at this
sample size, and the intended effect.

Passed listings also move up a little, 2 percentile points, the wrong
direction. I expected them to drop, not rise. It's small, about a fifth the
magnitude of the liked-listings effect, but it's not zero and it's worth
naming the mechanism honestly instead of blaming noise.

I checked the underlying vote data. Categorical features are correlated with
each other in this fixture, and one specific correlation drives most of it:
the profile picks up `condition_quality=dated` as a positive signal, but every
up-vote on a `dated` listing lives in a hood the household already prefers
(sausalito, central richmond, outer richmond, the top-3 up-voted hoods).
Meanwhile the four passed `dated` listings scatter across neutral hoods
(central sunset, inner richmond, parkside, mill valley). So `dated` gets
positive weight not because anyone up-voted a listing for looking dated, but
because it's a proxy for the hoods the household prefers. The model can't
cleanly disentangle "we like Marin/Richmond" from "we like dated" from 7
events. That same confounding nudges some passed listings up when they share
proxy features with liked ones.

The eval script prints its own small-n caveat before the numbers, on every run.

16 up-votes and about 18 non-empty `passed_on` notes total. Treat this as a
directional signal, not a validated model.

## Stickiness (added after the interview)

The interviewer pushed back on how much the ranking should lean on live LLM output. His
example: ten real estate agents would rank the same listings differently, so
there isn't one objectively correct order, just a reasonable point of view. He
kept coming back to the word "stickiness": a listing shouldn't jump around in
the list because a rerun nudged its LLM rank by one or two spots.

He's right that this is a gap. `rank()` throws away the previous render and
recomputes from scratch every time. `llm_rank` can shift a little between runs
even at `temperature=0`, since the batch context around a listing changes, and
two listings end up swapping in the UI for reasons nobody can point to.

Added a `rank_snapshot` table to `storage.py` that stores each listing's last
displayed position, with `load_rank_snapshot` and `save_rank_snapshot` to read
and write it. `rank.py` gets `stable_order()`: for two adjacent listings, one
only overtakes the other if its current heuristic score beats the previous one
by more than `STICKINESS_THRESHOLD` (5.0, above a single hood-tier flip or
laundry reclassification, below a full walk-time bucket change). Runs inside
`rank()` through an optional `prev_snapshot` parameter. `None` is a
byte-identical no-op, same as `profile=None`.

First pass grouped the stability check by the full sort key, which includes
`llm_rank`. Two listings only got compared if they already shared the exact
same rank, which basically never happens, so nothing moved. Dropped `llm_rank`
from the grouping and kept `(bucket, vote count, pipeline strength)`, and
stability started doing what it was supposed to: smoothing over `llm_rank`
jitter instead of being blocked by it.

## The stickiness number

Same `demo.sqlite` fixture. Ran the ranking twice with a small perturbation
between runs (`llm_rank` shifted up to 2, walk times shifted up to 3 minutes)
and counted adjacent swaps.

|                | off | on |
| -------------- | --: | -: |
| adjacent swaps |  36 | 22 |
| listings moved |  89 | 48 |

About 40% fewer adjacent swaps and 46% fewer listings moved at all, same
input, stickiness on vs off.

This is half of the point he raised about things not jumping around. The other
half, that ten agents would genuinely disagree and Casita still hands back one
confident order, is still open. Showing where the deterministic score and the
LLM rank disagree instead of quietly picking one is the bigger idea here, and I
haven't built it.

## What I'd do next

Share event-collection logic with `llm._preference_examples` through a shared
`iter_vote_events()` instead of the small duplication that exists now. Weight
`reviewer_a` and `reviewer_b` separately, the way the LLM prompt already does
through `_VOTER_PRIORITY`. Move past leave-one-out once there's more vote volume,
with a real held-out set and confidence intervals instead of one averaged
percentile. Surface the diff between this profile and `analyze_preferences`' LLM
output when both are available, since together they make a decent audit pair.
