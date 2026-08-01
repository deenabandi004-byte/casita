---
icon: lucide/target
---

# Preference Profile

`src/casita/preferences.py` turns vote and pass history into a structured,
per-dimension signal that `rank.rank()` uses to break ties.

It exists to fill the same gap `analyze_preferences()` in `llm.py` tries to
fill — take what the household actually chose and let it shape ranking — but
without a live Gemini call. Two consequences follow: it runs in the
credential-free demo path, and it can be unit tested.

## What it learns

Nine dimensions, all categorical:

- `dog_policy`, `laundry`, `parking`, `hood`
- `light_quality`, `view_quality`, `condition_quality` (from photo review)
- `trail_walk_bucket`, `beach_walk_bucket` (bucketed the same way
  `rank._walk_bonus` buckets them, so a "short trail walk" means the same
  thing in the profile as it does in the heuristic scorer)

For each event — an up-vote, a down-vote, or a `passed_on` note — we accumulate
a decayed weight per `(dimension, value)`:

```
weight += direction * (0.5 ** (age_days / half_life_days))
support += 1
```

The default half-life is 45 days: recent enough that shifting taste dominates,
old enough that a single noisy week doesn't flip the profile.

## The evidence bar

A dimension/value only counts toward `preference_adjustment()` when its
`support` clears `MIN_SUPPORT = 2`. This mirrors the ≥2-votes rule already
in `_ANALYZE_PREFS_SYSTEM` ("Only flag a pattern with real support, not a
one-off"). Held on purpose: whichever audit surface a reviewer looks at, the
bar is the same.

## Where it plugs in

`rank.rank(profile=…)` adds the profile's adjustment to the tie-break term
only. Bucket assignment (pipeline / favorites / ranked / new / filtered /
eliminated) is untouched, so an up-voted listing stays a favorite whether or
not the profile agrees. `profile=None` is a byte-identical no-op — pass no
profile, get the old order.

The card falls back to `preferences.explain(...)` when neither `share_blurb`
nor `llm_reason` nor `visual_summary` is present. The detail page adds a
"Why this ranked here" panel showing every gated dimension, this listing's
value, and the contributed weight.

## Measuring it

`scripts/eval_preferences.py` runs leave-one-out cross-validation on the
fixture. It compares `rank.score()` versus `rank.score() +
preference_adjustment()` directly — not through `rank.rank()`, whose favorites
bucket would put held-out liked listings at the top for the wrong reason. The
script prints an honest small-n caveat in its own output; the fixture holds
16 up-votes and roughly 18 non-empty passed_on notes, so this is a
directional signal, not a validated model.

## Ways This Could Go Further

- Share event-collection logic with `llm._preference_examples` (they query
  the same tables with slightly different shapes; a shared
  `iter_vote_events()` would remove the duplication).
- Weight `reviewer_a` and `reviewer_b` differently, the way the LLM prompt
  already does through `_VOTER_PRIORITY`.
- Grow past leave-one-out once there's more vote volume: a proper held-out
  set with confidence intervals rather than a single averaged percentile.
- Surface the diff between the deterministic profile and
  `analyze_preferences()`'s LLM output when both are available — the pair
  makes a nice audit signal.
