# Why this change

`analyze_preferences()` in `src/casita/llm.py` already tries to learn from
votes, but it's a single ungrounded LLM call — Vertex-only, so it can't run
in the credential-free demo path this repo is built around, and it has zero
test coverage as a result. I made the same underlying idea deterministic and
measurable: a preference profile built from actual vote history with recency
decay, wired into the existing heuristic scorer via one optional parameter,
with an offline eval proving it moves held-out liked listings up in rank.

## Why this gap, not a different one

`docs/architecture.md` names "LLM calls are Vertex-only" as a rough edge, and
`docs/how-it-works/learning.md` says the vote loop "could gain better
fixtures, better diff output, or clearer aging of old examples." I picked
this over routing-anchor sets, photo-eval fixture replay, or CLI module
splitting for two reasons:

**It's the same mechanism as your product.** The job posting describes
Imperfect as a coach that "adapts your training, recovery, and nutrition to
how your body is actually responding" and "iterate on the plan as life
happens." Casita's vote-learning loop is a small version of the identical
problem — take what a household actually chose, turn it into a ranking
adjustment. Building this specifically demonstrates understanding of what
the product has to do well, not general coding.

**It's checkable.** The posting also says: "the reasoning, the wearable data
behind it, the evals that catch it being wrong, and the latency that makes
it feel like a conversation." A feature that ships with a real before/after
eval is a direct demonstration of that — not a claim about it.

## What actually shipped

- `src/casita/preferences.py` — new module. `collect_events`, `build_profile`,
  `preference_adjustment`, `explain`, `explain_breakdown`. Categorical
  weights per dimension with a 45-day half-life; `MIN_SUPPORT=2` gate
  mirrors the ≥2-votes rule already baked into `_ANALYZE_PREFS_SYSTEM`.
  Walk-time buckets align with `rank._walk_bonus(sweet_spot=10)` so "short
  trail walk" means the same thing in both places.
- `src/casita/rank.py` — one optional `profile=` parameter on `rank()`. Used
  only in the tie-break term inside existing buckets, so an up-voted listing
  stays a favorite whether or not the profile agrees. `profile=None` is a
  byte-identical no-op.
- `tests/test_preferences.py` — 23 tests covering decay, MIN_SUPPORT
  gating, cold start, passed_on-as-down-signal, leave-one-out exclusion,
  walk-bucket alignment with `rank._walk_bonus`, and byte-identical
  no-profile behavior.
- `scripts/eval_preferences.py` — leave-one-out cross-validation. Compares
  `rank.score()` vs `rank.score() + preference_adjustment()` **directly**,
  not through `rank.rank()`, whose favorites bucket would otherwise put
  held-out likes at the top for the wrong reason and produce a falsely
  perfect result.
- `docs/how-it-works/preferences.md` — full doc following the existing
  page voice, including a "Ways This Could Go Further" section.
- Card fallback + a "Why this ranked here" detail-page panel. The panel
  only shows rows whose value for this specific listing clears MIN_SUPPORT
  in the profile — dimensions the profile knows about in general but where
  this listing's value has no supporting evidence contribute nothing to
  the tie-break, so they're not padding the table. Both surfaces omit at
  cold start rather than render empty.

## The number

Preference-adjustment eval (leave-one-out, run against `fixtures/demo.sqlite`).
Lower percentile = closer to the top of the ranked list.

|                    | n  | baseline | adjusted | delta   |
| ------------------ | -: | -------: | -------: | ------: |
| liked listings     |  9 |    0.409 |    0.322 | −0.087  |
| passed listings    | 15 |    0.343 |    0.333 | −0.010  |

Interpretation, honestly:

- Liked listings move up ~9 percentile points on average. Directionally
  correct and outside noise for this n.
- Passed listings barely move (−0.010 is essentially inside noise — I
  expected a small positive, got a small negative). At n=15 with a lot of
  shared structure between liked and passed listings in this fixture (both
  are SF rentals with similar layouts), this is what the data says. It's
  not proof the profile hurts negatives; it's honest reporting that the
  signal isn't strong enough for the passed set at this scale.
- Small-n caveat: 16 up-votes and ~18 non-empty passed_on notes in the
  fixture. Treat as a directional signal, not a validated model. The eval
  script prints this caveat in its own output so a reviewer sees it before
  the numbers.

## The Offerloop connection

At Offerloop we take real user behavior (which contacts a student actually
messages, which templates get replies, which company research they save)
and use it to personalize outreach targeting and content — turning implicit
signal into targeted, explainable output rather than an opaque prompt.
Preference-from-votes is the same pattern applied to housing signal: a
structured profile you can inspect, gate on evidence, and evaluate against
held-out data.

## What I'd do next

- Share event-collection logic with `llm._preference_examples` via an
  extracted `iter_vote_events()` — the two paths query the same tables and
  the small duplication is intentional-for-now, not permanent.
- Weight `reviewer_a` and `reviewer_b` separately, the way the LLM prompt
  already does through `_VOTER_PRIORITY`.
- Grow past leave-one-out once there's more vote volume: proper held-out
  set with confidence intervals rather than a single averaged percentile.
- Surface the diff between the deterministic profile and
  `analyze_preferences()`'s LLM output when both are available — the pair
  makes a nice audit signal.
