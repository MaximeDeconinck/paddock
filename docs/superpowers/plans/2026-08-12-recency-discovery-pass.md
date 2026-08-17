# Recency discovery pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a recency-oriented discovery pass (HF `sort=trendingScore`, Ollama `?sort=newest`) alongside the existing popularity pass on both sources, so freshly released models surface at sync time.

**Architecture:** Each source gains a second fetch and a pure merge/dedup step: HF unions the trending id list into the popularity id list; Ollama unions the newest names, reserving headroom so newest-only models are not pushed past the discovery cap. New `SyncOptions` knobs + CLI flags thread the limits through. Per-repo processing and `upsert_model` are unchanged.

**Tech Stack:** Rust 2024, HF API, ollama.com HTML, sync pipeline.

**Spec:** `docs/superpowers/specs/2026-08-11-recency-discovery-pass-design.md`

**Locked:** additive recency pass (union, not replace) · HF uses `trendingScore` (not `createdAt`) · Ollama uses `?sort=newest` · dedup by identity · no org/quality filtering.

---

## File Structure

- `crates/paddock-core/src/catalog/hf.rs` - `fetch_hf_gguf` gains a trending pass + a pure `merge_repo_ids` helper; signature `fetch_hf_gguf(http, limit, trending_limit)`.
- `crates/paddock-core/src/catalog/ollama_registry.rs` - extract `parse_library_names(html)`; add a pure `merge_library_names(popularity, newest, newest_reserve)`; `fetch_library_index(http, newest_reserve)` fetches both pages and merges.
- `crates/paddock-core/src/catalog/mod.rs` - `SyncOptions` gains `hf_trending_limit` + `ollama_newest_reserve`; `sync` threads them.
- `crates/paddock/src/cli.rs` + `crates/paddock/src/main.rs` - `--hf-trending-limit` / `--ollama-newest-reserve` flags wired into `SyncOptions`.

---

## Task 1: HF trending pass with id merge

**Files:**
- Modify: `crates/paddock-core/src/catalog/hf.rs` (`fetch_hf_gguf` ~line 112-133; tests module; all in-crate test callers of `fetch_hf_gguf`)

- [ ] **Step 1: Write the failing test for the pure merge helper**

Add to the `#[cfg(test)] mod tests` in `hf.rs`:

```rust
#[test]
fn merge_repo_ids_dedups_popularity_first() {
    let popularity = vec!["org/a".to_string(), "org/b".to_string()];
    let trending = vec!["org/b".to_string(), "org/new".to_string()];
    let merged = merge_repo_ids(popularity, trending);
    // popularity order preserved, shared id once, trending-only appended
    assert_eq!(merged, vec!["org/a", "org/b", "org/new"]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p paddock-core merge_repo_ids_dedups_popularity_first`
Expected: compile error - `cannot find function merge_repo_ids`.

- [ ] **Step 3: Implement `merge_repo_ids`**

Add near `fetch_hf_gguf` in `hf.rs`:

```rust
/// Union two repo-id lists, popularity order first, then trending ids not
/// already seen, deduplicated by id (preserving first occurrence).
fn merge_repo_ids(popularity: Vec<String>, trending: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for id in popularity.into_iter().chain(trending.into_iter()) {
        if seen.insert(id.clone()) {
            out.push(id);
        }
    }
    out
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p paddock-core merge_repo_ids_dedups_popularity_first`
Expected: PASS.

- [ ] **Step 5: Extract a list-fetch helper + rewrite `fetch_hf_gguf` with the trending pass**

Add a helper that fetches one query's ids, then rewrite `fetch_hf_gguf` to a new signature. Replace the existing function body (lines ~112-133):

```rust
/// Fetch the repo ids for one HF models query (`?...` after `{HF_API}/models`).
/// A failed/absent list yields an empty vec (a bad query must not kill sync).
async fn fetch_repo_ids(http: &dyn HttpClient, query: &str) -> Vec<String> {
    let Ok(list) = http.get_json(&format!("{HF_API}/models?{query}")).await else {
        return Vec::new();
    };
    list.as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|item| item["id"].as_str().map(String::from))
        .collect()
}

pub async fn fetch_hf_gguf(
    http: &dyn HttpClient,
    limit: usize,
    trending_limit: usize,
) -> Result<Vec<CatalogModel>, PaddockError> {
    let popularity = fetch_repo_ids(http, &format!("filter=gguf&sort=downloads&limit={limit}")).await;
    let trending = if trending_limit > 0 {
        fetch_repo_ids(http, &format!("filter=gguf&sort=trendingScore&limit={trending_limit}")).await
    } else {
        Vec::new()
    };
    let ids = merge_repo_ids(popularity, trending);
    let mut out = Vec::new();
    for repo in ids {
        match fetch_hf_repo(http, &repo).await {
            Ok(Some(m)) => out.push(m),
            Ok(None) => {}
            Err(_) => {} // one bad repo must not kill the sync
        }
    }
    Ok(out)
}
```

NOTE: the previous `fetch_hf_gguf` returned an error if the outer list call failed (see the existing `fetch_hf_gguf_failing_http_returns_empty` test at ~hf.rs:959, which asserts on `fetch_hf_gguf(&http, 5).await` being an error). The new `fetch_repo_ids` swallows the failure to an empty vec, so `fetch_hf_gguf` now returns `Ok(vec![])` instead of `Err` when the popularity list fails. UPDATE that test to assert `.await.unwrap().is_empty()` instead of `.is_err()` (the sync loop in `mod.rs` already treats an empty result as "no HF models this run", so the behavior is equivalent and more resilient). If the implementer judges preserving the `Err` contract is safer, instead keep `fetch_hf_gguf` returning `Err` when the POPULARITY (not trending) fetch fails - either is acceptable, but the test and the code must agree. Pick one and make them consistent.

- [ ] **Step 6: Update all in-crate test callers to the new arity**

Every `fetch_hf_gguf(&http, N)` call in `hf.rs` tests (~lines 563, 612, 650, 690, 949, 962) must become `fetch_hf_gguf(&http, N, 0)` (trending pass off, so existing single-list fixtures keep working unchanged). Read each and add the `, 0` argument. Do NOT change their fixtures otherwise.

- [ ] **Step 7: Add a test proving a trending-only repo is discovered**

```rust
#[tokio::test]
async fn trending_only_repo_is_discovered() {
    let pop_url = format!("{HF_API}/models?filter=gguf&sort=downloads&limit=1");
    let trend_url = format!("{HF_API}/models?filter=gguf&sort=trendingScore&limit=5");
    let new_repo = "meta-models/Muse-Glimmer-30B-GGUF";
    let detail_url = format!("{HF_API}/models/{new_repo}?blobs=true");
    let range_url =
        format!("https://huggingface.co/{new_repo}/resolve/main/Muse-Glimmer-30B-Q4_K_M.gguf");

    let http = MockHttp::new()
        .add_json(&pop_url, json!([]))
        .add_json(&trend_url, json!([{"id": new_repo}]))
        .add_json(
            &detail_url,
            json!({
                "id": new_repo,
                "gguf": {"architecture": "llama", "context_length": 8192, "total": 30000000000u64},
                "siblings": [{"rfilename": "Muse-Glimmer-30B-Q4_K_M.gguf", "size": 18000000000u64}]
            }),
        )
        .add_range(&range_url, llama_header());

    let models = fetch_hf_gguf(&http, 1, 5).await.unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].repo.as_deref(), Some(new_repo));
}
```

NOTE: match the mock-helper API and `llama_header()` to the neighboring tests (e.g. `mmproj_mid_name_not_a_variant`). Confirm the probe/range file is the single Q4_K_M. Adjust `add_range` target if the probe picks differently.

- [ ] **Step 8: Update the `mod.rs` caller**

In `mod.rs::sync` (~line 252), change `hf::fetch_hf_gguf(http, opts.hf_limit)` to `hf::fetch_hf_gguf(http, opts.hf_limit, opts.hf_trending_limit)`. (`opts.hf_trending_limit` is added in Task 3; if compiling this task alone, temporarily pass a literal `0` and switch to `opts.hf_trending_limit` in Task 3. Prefer doing Task 3's `SyncOptions` field first if executing in order - see Task 3.)

- [ ] **Step 9: Run the suite + commit**

Run: `cargo test -p paddock-core` (green) and `cargo clippy -p paddock-core` (clean).

```bash
git add crates/paddock-core/src/catalog/hf.rs crates/paddock-core/src/catalog/mod.rs
git commit -m "feat(catalog): HF trending discovery pass merged with popularity"
```

---

## Task 2: Ollama newest pass with reserved headroom

**Files:**
- Modify: `crates/paddock-core/src/catalog/ollama_registry.rs` (`fetch_library_index` ~line 82-100; tests module; the two in-crate callers ~lines 726, 733)
- Modify: `crates/paddock-core/src/catalog/mod.rs` (`discover_library_models` caller ~line 359)

- [ ] **Step 1: Write failing tests for the pure helpers**

Add to the `#[cfg(test)] mod tests` in `ollama_registry.rs`:

```rust
#[test]
fn parse_library_names_extracts_from_href() {
    let html = r#"<a href="/library/llama3.1">..</a><a href="/library/qwen3:8b">..</a><a href="/library/llama3.1">dup</a>"#;
    let names = parse_library_names(html);
    // bare names, tag suffix stripped, deduped
    assert_eq!(names, vec!["llama3.1", "qwen3"]);
}

#[test]
fn merge_library_names_reserves_newest_headroom() {
    let popularity = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    // "new" is newest-only and must land within the reserved headroom (front).
    let newest = vec!["new".to_string(), "a".to_string()];
    let merged = merge_library_names(popularity, newest, 1);
    // reserved newest-only first, then popularity, then remaining newest; deduped
    assert_eq!(merged, vec!["new", "a", "b", "c"]);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p paddock-core -- parse_library_names_extracts_from_href merge_library_names_reserves_newest_headroom`
Expected: compile error - functions not found.

- [ ] **Step 3: Extract `parse_library_names` + add `merge_library_names`**

Refactor the extraction currently inline in `fetch_library_index` into a pure fn, and add the merge fn:

```rust
/// Extract every `{name}` from `href="/library/{name}"` in page order,
/// stripping any `:tag` suffix, deduplicated (first occurrence wins).
pub fn parse_library_names(html: &str) -> Vec<String> {
    let needle = "href=\"/library/";
    let mut names: Vec<String> = Vec::new();
    for (idx, _) in html.match_indices(needle) {
        let tail = &html[idx + needle.len()..];
        let name: String = tail
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
            .collect();
        if !name.is_empty() && !names.iter().any(|n| n == &name) {
            names.push(name);
        }
    }
    names
}

/// Union popularity + newest library names: the first `newest_reserve`
/// newest-only names (not in popularity) go first so they survive the
/// downstream discovery cap, then popularity order, then any remaining newest.
/// Deduplicated, first occurrence wins.
fn merge_library_names(
    popularity: Vec<String>,
    newest: Vec<String>,
    newest_reserve: usize,
) -> Vec<String> {
    let pop_set: std::collections::HashSet<&str> =
        popularity.iter().map(String::as_str).collect();
    let reserved: Vec<String> = newest
        .iter()
        .filter(|n| !pop_set.contains(n.as_str()))
        .take(newest_reserve)
        .cloned()
        .collect();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for name in reserved.into_iter().chain(popularity).chain(newest) {
        if seen.insert(name.clone()) {
            out.push(name);
        }
    }
    out
}
```

- [ ] **Step 4: Rewrite `fetch_library_index` to fetch both pages and merge**

```rust
/// Fetch the library index (popularity page) unioned with the `?sort=newest`
/// page, so freshly added models are discoverable. `newest_reserve` newest-only
/// names are placed first to survive the discovery cap. A failed newest fetch
/// degrades to popularity-only.
pub async fn fetch_library_index(
    http: &dyn HttpClient,
    newest_reserve: usize,
) -> Result<Vec<String>, PaddockError> {
    let popularity = parse_library_names(&http.get_text("https://ollama.com/library").await?);
    let newest = match http.get_text("https://ollama.com/library?sort=newest").await {
        Ok(html) => parse_library_names(&html),
        Err(_) => Vec::new(), // newest is best-effort; keep popularity
    };
    Ok(merge_library_names(popularity, newest, newest_reserve))
}
```

- [ ] **Step 5: Update the callers**

- In `mod.rs::discover_library_models` (~line 359): `ollama_registry::fetch_library_index(http)` becomes `ollama_registry::fetch_library_index(http, newest_reserve)`. Add a `newest_reserve: usize` parameter to `discover_library_models` and pass it from `sync` (Task 3). If executing this task alone, thread a literal `20` temporarily, switched to `opts.ollama_newest_reserve` in Task 3.
- In `ollama_registry.rs` tests (~lines 726, 733): `fetch_library_index(&http)` becomes `fetch_library_index(&http, 20)`. These existing tests register only the popularity URL; the newest fetch will fail and degrade to popularity-only, so their assertions still hold. Verify: the test at ~733 asserts an error when the page fetch fails - after the change, the POPULARITY fetch still errors (propagated via `?`), so it stays `Err`. Confirm by reading.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p paddock-core -- parse_library_names merge_library_names fetch_library_index`
Expected: PASS.

- [ ] **Step 7: Full suite + clippy + commit**

Run: `cargo test -p paddock-core` (green) and `cargo clippy -p paddock-core` (clean).

```bash
git add crates/paddock-core/src/catalog/ollama_registry.rs crates/paddock-core/src/catalog/mod.rs
git commit -m "feat(catalog): Ollama newest discovery pass with reserved headroom"
```

---

## Task 3: SyncOptions knobs + CLI flags

**Files:**
- Modify: `crates/paddock-core/src/catalog/mod.rs` (`SyncOptions` ~line 159-168, `Default` ~line 170-178, `sync` signature/usage ~line 194-250, `discover_library_models` call ~line 250)
- Modify: `crates/paddock/src/cli.rs` (`Sync` variant ~line 80-96)
- Modify: `crates/paddock/src/main.rs` (`SyncOptions` construction ~line 90-95)

- [ ] **Step 1: Add the `SyncOptions` fields + defaults**

In `mod.rs`, add to `SyncOptions`:

```rust
    /// Max HF repos to index from the trending list (recency pass). 0 disables.
    pub hf_trending_limit: usize,
    /// Newest-only Ollama library names reserved ahead of the discovery cap.
    pub ollama_newest_reserve: usize,
```

In `Default`:

```rust
            hf_trending_limit: 40,
            ollama_newest_reserve: 20,
```

- [ ] **Step 2: Thread into `sync`**

In `mod.rs::sync`: ensure the HF call is `hf::fetch_hf_gguf(http, opts.hf_limit, opts.hf_trending_limit)` and the discovery call passes `opts.ollama_newest_reserve` (i.e. `discover_library_models(http, db, &curated_models, limit, opts.ollama_newest_reserve, now, &mut report)`). Update `discover_library_models`'s signature to accept `newest_reserve: usize` and forward it to `fetch_library_index`.

- [ ] **Step 3: Add CLI flags**

In `cli.rs`, inside the `Sync { .. }` variant add:

```rust
        /// Max HuggingFace trending repos to index (recency pass; 0 disables)
        #[arg(long, default_value_t = 40)]
        hf_trending_limit: usize,
        /// Newest Ollama library models reserved ahead of the discovery cap
        #[arg(long, default_value_t = 20)]
        ollama_newest_reserve: usize,
```

- [ ] **Step 4: Wire in `main.rs`**

In the `Command::Sync { .. }` destructuring, add `hf_trending_limit` and `ollama_newest_reserve`, and set them in the `SyncOptions { .. }` construction (~line 90):

```rust
            let opts = paddock_core::catalog::SyncOptions {
                hf_limit,
                mlx_limit,
                ollama_registry: !no_ollama_registry,
                discover_limit: (!no_discover).then_some(discover_limit),
                hf_trending_limit,
                ollama_newest_reserve,
            };
```

- [ ] **Step 5: Build + test**

Run: `cargo build -p paddock`. The three test literals in `mod.rs` (~534, 740, 909) already use `..SyncOptions::default()`, so they absorb the new fields with no change. The only full literal is `main.rs` (updated in Step 4). Confirm with `grep -rn "SyncOptions {" crates/` that no other full literal exists. Then `cargo test` (green) and `cargo clippy --workspace` (clean).

- [ ] **Step 6: Commit**

```bash
git add crates/paddock-core/src/catalog/mod.rs crates/paddock/src/cli.rs crates/paddock/src/main.rs
git commit -m "feat(cli): --hf-trending-limit and --ollama-newest-reserve sync flags"
```

---

## Task 4: Live verification

**Files:** none (verification only)

- [ ] **Step 1: Build release + sync**

Run: `cargo build --release -p paddock && ./target/release/paddock sync`
Expected: completes; the summary line's `huggingface` / `discovered` counts are >= the pre-change run (recency adds models).

- [ ] **Step 2: Confirm Muse Glimmer surfaces**

Run: `./target/release/paddock fit --all -n 400 --json 2>/dev/null | python3 -c "import sys,json; [print(m['name']) for m in json.load(sys.stdin) if 'glimmer' in m['name'].lower() or 'muse' in m['name'].lower()]"`
Expected: at least one Muse Glimmer entry (HF `unsloth/Muse-Glimmer-30B-GGUF` and/or the Ollama `muse-glimmer`).

If nothing surfaces, STOP and report: dump the first ids from `{HF_API}/models?filter=gguf&sort=trendingScore&limit=40` and the `ollama.com/library?sort=newest` names to see whether the model is in the fetched window or filtered by a quality gate.

No commit (verification only).

---

## Self-Review notes

- **Spec coverage:** Task 1 = HF trending pass + merge (spec Component 1); Task 2 = Ollama newest pass + `parse_library_names`/`merge_library_names` + reserved headroom (spec Component 2, including the `take(limit)` headroom concern); Task 3 = SyncOptions + CLI knobs (spec Component 3); Task 4 = live smoke. Error-handling (best-effort recency, dedup) covered in Tasks 1-2. `createdAt`/org-filtering correctly excluded.
- **Signature consistency:** `fetch_hf_gguf(http, limit, trending_limit)`, `fetch_library_index(http, newest_reserve)`, `discover_library_models(.., newest_reserve, ..)`, `SyncOptions.hf_trending_limit` / `.ollama_newest_reserve` used identically across tasks.
- **Contract change flagged:** the `fetch_hf_gguf` Err->Ok(empty) shift on list-fetch failure is called out in Task 1 Step 5 with the test update, and a consistent alternative offered.
- **Constructor sweep:** Task 3 Step 5 explicitly greps for other `SyncOptions {` literals (tray, tests) so the new fields do not break the build.
