# Model Release Dates — Design

**Date:** 2026-06-10
**Motivation:** Year-old models (qwen3, nemotron…) rank too high in `fit`. The catalog has no notion of model age: a 14-month-old model scores identically to last month's release of the same size. Users also can't see age at a glance.

## Goals

1. Show each model's release date as an `AGE` column in the TUI table and `paddock fit` CLI output.
2. Apply a progressive age malus to the quality sub-score so stale models sink in the ranking without being buried.

## Non-goals

- No new network requests at sync time.
- No per-variant dates (model-level only).
- No "last updated" tracking — release date only.

## Data model

- `CatalogModel` gains `released_at: Option<i64>` (epoch seconds) and `released_approx: bool` (serde default false).
- DB `models` table gains `released_at INTEGER` (nullable) and `released_approx INTEGER NOT NULL DEFAULT 0`, via the same `ALTER TABLE … ignore duplicate-column error` migration pattern as `variants.source_tag`.
- `ModelVariant` (the estimator bridge type) gains the same two fields so scoring and display read one flat struct.

## Date sources (all free — zero extra requests)

| Source | Date | Exactness |
|---|---|---|
| Curated Ollama (~78 entries) | new `"released": "YYYY-MM"` field in `curated_ollama.json`, hand-filled | exact (month precision) |
| HF GGUF + mlx-community | `createdAt` already present in the fetched `/api/models` JSON | exact |
| Discovered Ollama | oldest relative date (`N years/months/weeks/days ago`) parsed from the already-fetched tags page HTML | approximate (`released_approx = true`) |

Notes:
- `"YYYY-MM"` parses to the first day of that month, UTC.
- The Ollama tags-page date is a **lower bound on age**: a full re-push refreshes every tag date (verified live on nemotron-cascade-2: all 10 tags say "2 months ago" for a 2025 model). Taking the *oldest* tag date is the best available signal; it is displayed with a `~` prefix and still gets the malus — a wrongly-young model simply escapes it, which is the failure mode we accept.
- Curated models enriched with registry tags keep their curated (exact) date.
- Unknown date → `None`, no malus, displayed `?`.

## Scoring

Age malus subtracted from the quality sub-score (no fifth geometric-mean component; existing per-use-case weights unchanged):

```text
age_years = (now - released_at) / 365.25 days
malus     = min(20, 10 × max(0, age_years - 0.5))
```

- 6-month grace period, then 10 pts/year, capped at 20.
- Examples (now = 2026-06): qwen3 (2025-04, 14 mo) → −5.9; a 2024-07 model → −14.2; anything ≥ 2.5 y → −20.
- Applied identically for exact and approximate dates.
- `quality_subscore` takes the precomputed `age_days: Option<f64>`; callers compute it from `released_at` and an injected `now` (testability — no `SystemTime::now()` inside score.rs).
- Quality already has a quant malus; both subtract from the same base, floored at 0.

## Display

- New right-aligned `AGE` column in the TUI fit table and `paddock fit` text output, immediately after `MODEL` (age is a model attribute; QUANT onward are variant attributes).
- Format: `3w`, `8mo`, `1.2y`; `~` prefix when approximate (`~1y`); `?` when unknown.
- `fit --json` rows gain `released_at` (epoch or null) and `released_approx`.

## Error handling

- Unparseable curated `released` string: fail the curated-JSON validity test (compile-time-adjacent guard), not runtime.
- Unparseable HF `createdAt` or tags-page dates: field stays `None`, no sync error — dates are enrichment, never fatal.

## Testing

- Curated validity test: every entry's `released` parses and is within 2020-01 .. next month.
- gguf/registry tests: tags-page fixtures with per-tag dates → oldest extracted; full re-push case; missing dates → None.
- HF test: `createdAt` parsed from fixture JSON.
- DB: migration on a pre-`released_at` database file; roundtrip.
- Score: grace period boundary, 1 y / 2 y / cap values, None → no malus; ranking test: same-size model 2 y older must score strictly lower.
- CLI/TUI: AGE formatting unit tests (`3w`, `8mo`, `~1.2y`, `?`).
