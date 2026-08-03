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

Two things pointed here specifically.

`docs/architecture.md` names "LLM calls are Vertex-only" as a known rough edge, and
`docs/how-it-works/learning.md` says the vote loop "could gain better fixtures,
better diff output, or clearer aging of old examples." That's a real, named gap,
not something invented to have a task.

The job posting also names evals directly: "the reasoning, the wearable data behind
it, the evals that catch it being wrong." A feature that ships with a real
leave-one-out eval on real vote data is evidence of that, not a claim about it.

## Why this is close to what I already build

At Offerloop I run a LightGBM lambdarank model over a `recommendation_events` log
to rank contact/job matches, and a warmth-scoring layer that prioritizes leads by
type (alumni, dream_company, recent_transition) based on what actually converts,
not what a static rule guesses will convert. Both do the same thing this feature
does: take a stream of behavioral events, turn it into structured, gated,
per-dimension weights, feed those into a ranker, and check the result against
held-out data instead of trusting the model's judgment on faith. Casita's votes
table is basically Offerloop's `recommendation_events` table with a smaller schema.
Writing the leave-one-out eval here was the same instinct as checking whether the
lambdarank model actually beats the baseline before it ships.

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

## What I'd do next

Share event-collection logic with `llm._preference_examples` through a shared
`iter_vote_events()` instead of the small duplication that exists now. Weight
`reviewer_a` and `reviewer_b` separately, the way the LLM prompt already does
through `_VOTER_PRIORITY`. Move past leave-one-out once there's more vote volume,
with a real held-out set and confidence intervals instead of one averaged
percentile. Surface the diff between this profile and `analyze_preferences`' LLM
output when both are available, since together they make a decent audit pair.
