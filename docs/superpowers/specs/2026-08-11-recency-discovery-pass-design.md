# Recency discovery pass design

Date: 2026-08-11
Status: approved (design)

## Problem

paddock's model discovery is popularity-capped on both sources, so a freshly
released model never surfaces until it climbs the ranks:

- **Hugging Face** (`hf.rs::fetch_hf_gguf`): one API call
  `?filter=gguf&sort=downloads&limit=N`. A brand-new repo has ~0 downloads and
  falls outside the top-N window.
- **Ollama** (`ollama_registry.rs::fetch_library_index` + `mod.rs::discover_library_models`):
  fetches `ollama.com/library` in page order (= popularity), then `take(limit)`.
  New models rank low and get dropped.

Concrete failure: Meta's `Muse Glimmer` is on both ollama.com/library and
Hugging Face, yet does not appear in paddock's catalog after a sync, because it
is outside both popularity windows.

## Goal

Add a recency-oriented discovery pass alongside the existing popularity pass on
both sources, so new-but-notable models surface at sync time. Keep the
popularity pass unchanged (do not lose established models).

## Decisions (locked)

- **Additive recency pass**, not a replacement. Union with the existing
  popularity pass, deduplicated. Both sources.
- **HF recency uses `sort=trendingScore`, NOT `sort=createdAt`.** Verified live
  (2026-08-11): `sort=createdAt` is dominated by low-quality community merges
  (`shaffhausen/...-Heretic-NM-DAU-NEO-MAX-...`); `sort=trendingScore` surfaces
  real releases - `unsloth/Muse-Glimmer-30B-GGUF` at #2, plus DeepSeek-V4,
  LiquidAI, MiniMax. Trending balances recency and notability.
- **Ollama recency uses `ollama.com/library?sort=newest`.** Verified live:
  `muse-glimmer` is #1 under that sort. Same HTML href parsing as the default
  page.
- **No org/quality filtering.** trending/newest still admit some community
  merges; they pass the existing quality gates (valid params/layers/GGUF) and
  discovery is opt-in. Org allowlists are fragile and out of scope.
- **Dedup by identity** (HF repo id, Ollama base name). `upsert_model` is
  already idempotent; dedup mainly avoids redundant network work.

## Components

### 1. HF trending pass (`crates/paddock-core/src/catalog/hf.rs`)

`fetch_hf_gguf(http, limit)` currently issues one `sort=downloads&limit={limit}`
call and processes each repo id. Change it to:

1. Fetch the popularity list (`sort=downloads&limit={limit}`) as today.
2. Fetch a trending list (`sort=trendingScore&limit={trending_limit}`).
3. Union the repo ids from both, preserving order (popularity first, then
   trending ids not already seen), deduplicated by id.
4. Run the existing `fetch_hf_repo` per unique id (unchanged: probe header,
   quants, mmproj filter, MoE active-params).

Signature becomes `fetch_hf_gguf(http, limit, trending_limit)`. A
`trending_limit` of 0 skips the trending call (keeps the old behavior available
for tests / opt-out).

Extract the "list ids from a `?...` query" step into a small helper so the two
calls share it and stay testable.

### 2. Ollama newest pass (`crates/paddock-core/src/catalog/ollama_registry.rs`)

`fetch_library_index(http)` fetches `ollama.com/library` and parses names from
the `href="/library/{name}"` pattern. Refactor the HTML->names extraction into a
pure helper (e.g. `parse_library_names(html) -> Vec<String>`), then:

1. Fetch the default page (popularity order) as today.
2. Fetch `ollama.com/library?sort=newest`.
3. Union names, popularity order first, then newest names not already seen,
   deduplicated.

`fetch_library_index` returns the unioned list; `discover_library_models` is
unchanged (it already dedups against curated bases and applies `take(limit)`).

NOTE: `take(limit)` in `discover_library_models` still caps total attempts. With
the union ordered popularity-first, a new model that only appears in the newest
list could be pushed past `limit`. To ensure recency actually surfaces, the
newest names must be given headroom: prepend a bounded slice of newest-only
names before applying the popularity list, OR raise the effective cap for the
newest slice. Concretely: interleave so the first `newest_reserve` (default ~20)
newest-only names are attempted regardless of the popularity `take(limit)`. The
plan will specify the exact ordering; the requirement is: a model present only
in the newest list, within the top `newest_reserve`, MUST be discovered.

### 3. Config + CLI (`crates/paddock-core/src/catalog/mod.rs`, `crates/paddock/src/cli.rs`)

- `SyncOptions` gains `hf_trending_limit: usize` (default 40) and
  `ollama_newest_reserve: usize` (default 20). Existing fields unchanged.
- `mod.rs::sync` passes `hf_trending_limit` to `fetch_hf_gguf` and
  `ollama_newest_reserve` into the discovery path.
- CLI `Sync` gains `--hf-trending-limit` and `--ollama-newest-reserve` flags
  (mirroring the existing `--hf-limit` / `--discover-limit` style), plus the
  defaults above when omitted.
- `SyncReport` may gain a counter for recency-only discoveries if cheap, but
  this is optional polish, not required.

## Error handling

- Either recency call failing is non-fatal: log to `report.errors` and fall
  back to the popularity results (same resilience the current sync has for a
  failed source).
- A repo appearing in both passes is fetched/upserted once (dedup); even without
  dedup, `upsert_model` idempotency makes a double-attempt safe.

## Testing

Pure/unit-testable:
- **HF id union+dedup:** given a popularity list and a trending list sharing one
  id, the merged id list contains each id once, popularity order first. A
  trending-only id is present.
- **`parse_library_names`:** extracts names from the href pattern (moved out of
  `fetch_library_index`, tested directly).
- **Ollama union+dedup:** popularity ∪ newest, dedup, newest-only name present
  within the reserve.
- **Recency surfaces a new model:** a mock where a "new" id/name appears ONLY in
  the trending/newest list ends up discovered (mock HTTP, assert the model is
  upserted / returned).
- **trending_limit = 0 / newest fetch failing:** falls back to popularity-only,
  no panic.

Live smoke: `paddock sync`, then confirm `Muse Glimmer` (or `muse-glimmer`)
appears in `paddock fit --all` / the TUI list.

## Out of scope (YAGNI)

- Org allowlists / download-count quality filtering.
- HF `sort=createdAt` pure-recency (too noisy).
- MLX recency (the mlx-community author feed is already recency-ish and not the
  reported gap).
- Scoring / ranking changes.
