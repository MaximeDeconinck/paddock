# Model Release Dates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show model age in the fit table and apply a progressive age malus to the quality sub-score, per `docs/superpowers/specs/2026-06-10-release-dates-design.md`.

**Architecture:** `CatalogModel.released_at/released_approx` filled by three free sources (curated JSON hand-dates, HF `createdAt`, oldest relative date on the Ollama tags page), persisted via two new `models` columns, surfaced as an `AGE` column, and folded into `quality_subscore` as `min(20, 10 × max(0, age_years − 0.5))`.

**Tech Stack:** Rust, rusqlite, serde; no new dependencies — date math is a tiny hand-rolled civil-calendar module (no chrono).

**Plan deviation from spec (deliberate):** `ModelVariant` does NOT gain date fields. Scoring takes a precomputed `age_days: Option<f64>` parameter and display reads `ScoredModel.model.released_at` — same behavior, no churn in estimate.rs and its many test constructors.

---

### Task 1: `dates.rs` — epoch helpers (pure, TDD)

**Files:**
- Create: `crates/paddock-core/src/catalog/dates.rs`
- Modify: `crates/paddock-core/src/catalog/mod.rs` (add `pub mod dates;` after `pub mod db;`)

- [ ] **1.1 Write failing tests** in the new file:

```rust
//! Minimal civil-calendar date helpers (UTC, day precision). Hand-rolled to
//! avoid a chrono dependency: the catalog only needs "parse a date string to
//! epoch seconds" and "subtract relative ages".

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ymd_to_epoch_known_values() {
        assert_eq!(ymd_to_epoch(1970, 1, 1), Some(0));
        assert_eq!(ymd_to_epoch(2024, 7, 23), Some(1_721_692_800)); // llama3.1 day
        assert_eq!(ymd_to_epoch(2025, 4, 1), Some(1_743_465_600));
        assert_eq!(ymd_to_epoch(2024, 2, 29), Some(1_709_164_800)); // leap day
        assert_eq!(ymd_to_epoch(2024, 13, 1), None);
        assert_eq!(ymd_to_epoch(2024, 0, 1), None);
        assert_eq!(ymd_to_epoch(2024, 2, 30), None);
    }

    #[test]
    fn parse_year_month_cases() {
        assert_eq!(parse_year_month("2025-04"), ymd_to_epoch(2025, 4, 1));
        assert_eq!(parse_year_month("2024-12"), ymd_to_epoch(2024, 12, 1));
        assert_eq!(parse_year_month("2025"), None);
        assert_eq!(parse_year_month("2025-13"), None);
        assert_eq!(parse_year_month("avril 2025"), None);
    }

    #[test]
    fn parse_iso_date_prefix_cases() {
        // HF createdAt: "2024-03-07T15:45:34.000Z" — day prefix is enough.
        assert_eq!(
            parse_iso_date_prefix("2024-03-07T15:45:34.000Z"),
            ymd_to_epoch(2024, 3, 7)
        );
        assert_eq!(parse_iso_date_prefix("2024-03-07"), ymd_to_epoch(2024, 3, 7));
        assert_eq!(parse_iso_date_prefix("garbage"), None);
        assert_eq!(parse_iso_date_prefix("2024-3-7"), None);
    }

    #[test]
    fn parse_relative_ago_cases() {
        const NOW: i64 = 1_780_000_000;
        const DAY: i64 = 86_400;
        assert_eq!(parse_relative_ago("3 days ago", NOW), Some(NOW - 3 * DAY));
        assert_eq!(parse_relative_ago("1 day ago", NOW), Some(NOW - DAY));
        assert_eq!(parse_relative_ago("2 weeks ago", NOW), Some(NOW - 14 * DAY));
        assert_eq!(parse_relative_ago("6 months ago", NOW), Some(NOW - 6 * 30 * DAY));
        assert_eq!(parse_relative_ago("1 year ago", NOW), Some(NOW - 365 * DAY));
        assert_eq!(parse_relative_ago("yesterday", NOW), Some(NOW - DAY));
        assert_eq!(parse_relative_ago("5 hours ago", NOW), Some(NOW)); // < 1 day → now
        assert_eq!(parse_relative_ago("soon", NOW), None);
    }

    #[test]
    fn oldest_relative_date_scans_whole_page() {
        const NOW: i64 = 1_780_000_000;
        let html = r#"
            <span x-test-updated>2 months ago</span>
            <div>… 46e0c10c039e&nbsp;·&nbsp;1 year ago</div>
            <span>3 weeks ago</span>
        "#;
        // Oldest mention wins: 1 year ago.
        assert_eq!(
            oldest_relative_date(html, NOW),
            Some(NOW - 365 * 86_400)
        );
        assert_eq!(oldest_relative_date("<html>no dates</html>", NOW), None);
    }
}
```

- [ ] **1.2 Run, verify failure**: `cargo test -p paddock-core dates` → compile error (functions missing).

- [ ] **1.3 Implement** above the test module:

```rust
/// Days from civil date to 1970-01-01 (Howard Hinnant's algorithm), then to
/// epoch seconds. Returns None for out-of-range month/day.
pub fn ymd_to_epoch(y: i64, m: u32, d: u32) -> Option<i64> {
    if !(1..=12).contains(&m) {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let dim = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if d < 1 || d > dim[(m - 1) as usize] {
        return None;
    }
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some((era * 146_097 + doe - 719_468) * 86_400)
}

/// `"YYYY-MM"` → epoch of the first of that month (curated JSON format).
pub fn parse_year_month(s: &str) -> Option<i64> {
    let (y, m) = s.split_once('-')?;
    if y.len() != 4 || m.len() != 2 {
        return None;
    }
    ymd_to_epoch(y.parse().ok()?, m.parse().ok()?, 1)
}

/// ISO-8601 day prefix (`"2024-03-07T…"` or `"2024-03-07"`) → epoch.
/// Strict `YYYY-MM-DD` shape (HF always zero-pads).
pub fn parse_iso_date_prefix(s: &str) -> Option<i64> {
    let s = s.get(..10)?;
    let b = s.as_bytes();
    if b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    ymd_to_epoch(s[..4].parse().ok()?, s[5..7].parse().ok()?, s[8..10].parse().ok()?)
}

/// `"N years/months/weeks/days ago"` (and `"yesterday"`, sub-day units) →
/// epoch relative to `now`. Month = 30 d, year = 365 d — display precision.
pub fn parse_relative_ago(s: &str, now: i64) -> Option<i64> {
    const DAY: i64 = 86_400;
    let s = s.trim().to_lowercase();
    if s == "yesterday" {
        return Some(now - DAY);
    }
    let mut it = s.split_whitespace();
    let n: i64 = it.next()?.parse().ok()?;
    let unit = it.next()?;
    if it.next() != Some("ago") {
        return None;
    }
    let days = match unit.trim_end_matches('s') {
        "day" => n,
        "week" => n * 7,
        "month" => n * 30,
        "year" => n * 365,
        "hour" | "minute" | "second" => 0,
        _ => return None,
    };
    Some(now - days * DAY)
}

/// Scan a page for every `"N unit ago"` mention and return the OLDEST as an
/// epoch. Used on Ollama tags pages where the oldest tag date is the closest
/// available proxy for the release date (a full re-push refreshes all of
/// them — that failure mode is accepted; see the design spec).
pub fn oldest_relative_date(html: &str, now: i64) -> Option<i64> {
    let mut oldest: Option<i64> = None;
    for (idx, _) in html.match_indices(" ago") {
        // Walk back over "N unit " (max ~20 chars: "59 minutes").
        let start = html[..idx].char_indices().rev().take(20)
            .map(|(i, _)| i)
            .last()
            .unwrap_or(0);
        let window = &html[start..idx + 4];
        // Try every suffix of the window that starts at a digit.
        for (i, c) in window.char_indices() {
            if c.is_ascii_digit() {
                if let Some(e) = parse_relative_ago(&window[i..], now) {
                    oldest = Some(oldest.map_or(e, |o: i64| o.min(e)));
                    break;
                }
            }
        }
    }
    oldest
}
```

- [ ] **1.4 Run**: `cargo test -p paddock-core dates` → all pass.
- [ ] **1.5 Commit**: `git add crates/paddock-core/src/catalog/{dates.rs,mod.rs} && git commit -m "feat(core): civil-date helpers for release dates"`

---

### Task 2: model fields + DB migration (TDD)

**Files:**
- Modify: `crates/paddock-core/src/catalog/mod.rs` (CatalogModel)
- Modify: `crates/paddock-core/src/catalog/db.rs` (schema, migration, upsert, list)

- [ ] **2.1 Add fields to `CatalogModel`** (mod.rs):

```rust
    /// Release date (epoch seconds), when known. Display + age malus input.
    #[serde(default)]
    pub released_at: Option<i64>,
    /// True when the date is a lower-bound proxy (Ollama tags page), not exact.
    #[serde(default)]
    pub released_approx: bool,
```

Then `cargo build -p paddock-core 2>&1 | grep "missing field"` and add `released_at: None, released_approx: false,` to every constructor it lists (curated.rs, hf.rs ×2, ollama_registry.rs discover_model, db.rs list_models, plus test fixtures).

- [ ] **2.2 Write failing DB test** (db.rs tests; mirror the existing `source_tag` migration test):

```rust
    #[test]
    fn migration_adds_released_columns_and_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.db");
        {
            // Pre-release_at schema: current SCHEMA minus the new columns.
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE models (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL, family TEXT,
                    source TEXT NOT NULL, repo TEXT,
                    params_total INTEGER NOT NULL, params_active INTEGER NOT NULL,
                    architecture TEXT, context_max INTEGER NOT NULL,
                    UNIQUE(source, name));",
            )
            .unwrap();
        }
        let db = Db::open(&path).unwrap();
        let mut m = sample_model(); // reuse the existing test fixture helper
        m.released_at = Some(1_743_465_600);
        m.released_approx = true;
        db.upsert_model(&m).unwrap();
        let got = &db.list_models().unwrap()[0];
        assert_eq!(got.released_at, Some(1_743_465_600));
        assert!(got.released_approx);
        // None roundtrips too.
        m.released_at = None;
        m.released_approx = false;
        db.upsert_model(&m).unwrap();
        let got = &db.list_models().unwrap()[0];
        assert_eq!(got.released_at, None);
        assert!(!got.released_approx);
    }
```

(If db.rs has no `sample_model()` helper, build a minimal `CatalogModel` inline with one variant, as the existing tests do.)

- [ ] **2.3 Run, verify failure**: `cargo test -p paddock-core migration_adds_released` → fails.

- [ ] **2.4 Implement** in db.rs:
  - SCHEMA `models` gains `released_at INTEGER,` and `released_approx INTEGER NOT NULL DEFAULT 0,` before the UNIQUE line.
  - In `Db::open`, after the `source_tag` migration, same pattern twice:

```rust
        for ddl in [
            "ALTER TABLE models ADD COLUMN released_at INTEGER",
            "ALTER TABLE models ADD COLUMN released_approx INTEGER NOT NULL DEFAULT 0",
        ] {
            if let Err(e) = conn.execute(ddl, []) {
                if !e.to_string().contains("duplicate column name") {
                    return Err(e.into());
                }
            }
        }
```

  - `upsert_model`: INSERT columns + `?9, ?10` → add `released_at, released_approx` (params `m.released_at`, `m.released_approx as i64`), and `released_at=excluded.released_at, released_approx=excluded.released_approx` in the UPDATE SET.
  - `list_models`: SELECT the two columns; `released_at: r.get(9)?`, `released_approx: r.get::<_, i64>(10)? != 0` (adjust indices to the actual column order).

- [ ] **2.5 Run**: `cargo test -p paddock-core` → all pass (catalog tests compile with the new fields).
- [ ] **2.6 Commit**: `git commit -am "feat(core): released_at/released_approx on models with DB migration"`

---

### Task 3: curated hand-dates

**Files:**
- Modify: `crates/paddock-core/src/catalog/curated_ollama.json` (all 78 entries)
- Modify: `crates/paddock-core/src/catalog/curated.rs` (Entry + mapping + validity test)

- [ ] **3.1 Extend the validity test first** (curated.rs):

```rust
        // Every entry carries a parseable release month within sane bounds.
        for m in &models {
            let r = m.released_at.unwrap_or_else(|| panic!("{} missing released", m.name));
            assert!(
                r >= super::dates::ymd_to_epoch(2020, 1, 1).unwrap()
                    && r <= super::dates::ymd_to_epoch(2027, 1, 1).unwrap(),
                "{} release date out of range",
                m.name
            );
            assert!(!m.released_approx, "curated dates are exact");
        }
```

- [ ] **3.2 Run, verify failure**: `cargo test -p paddock-core curated` → panics on missing released.

- [ ] **3.3 Implement**: `Entry` gains `released: String`; mapping sets

```rust
            released_at: Some(
                super::dates::parse_year_month(&e.released)
                    .expect("curated released must be YYYY-MM (checked by unit test)"),
            ),
            released_approx: false,
```

  Then add `"released": "YYYY-MM"` to every JSON entry. Source of truth = the implementer's knowledge of each family's release announcement (e.g. llama3.1 → `2024-07`, llama3 → `2024-04`, llama2 → `2023-07`, qwen3 → `2025-04`, qwen2.5 → `2024-09`, gemma3 → `2025-03`, gemma2 → `2024-06`, phi4 → `2024-12`, deepseek-r1 → `2025-01`, mistral-small3.1 → `2025-03`, gpt-oss → `2025-08`, granite3.3 → `2025-04`, smollm2 → `2024-11`, …). **For any entry whose release month you cannot state with confidence, look it up (web search for the announcement, or the HF repo's `createdAt`) instead of guessing.**

- [ ] **3.4 Spot-check 8 dates against the web** (HF repo createdAt or announcement blog). Paste the 8 checked pairs in the task output.
- [ ] **3.5 Run**: `cargo test -p paddock-core curated` → pass.
- [ ] **3.6 Commit**: `git commit -am "feat(core): hand-curated release dates for the embedded catalog"`

---

### Task 4: HF/MLX `createdAt` (TDD)

**Files:**
- Modify: `crates/paddock-core/src/catalog/hf.rs`

- [ ] **4.1 Extend existing fixtures/tests**: in the hf.rs test module, add `"createdAt": "2024-03-07T15:45:34.000Z"` to the repo-detail JSON fixture used by the happy-path `fetch_hf_repo`/`fetch_hf_gguf` test and to one `fetch_mlx` fixture, then assert on the produced model:

```rust
        assert_eq!(m.released_at, super::super::dates::ymd_to_epoch(2024, 3, 7));
        assert!(!m.released_approx);
```

  Also assert a fixture *without* `createdAt` yields `released_at: None`.

- [ ] **4.2 Run, verify failure.**

- [ ] **4.3 Implement**: in `fetch_hf_repo` (detail JSON) and in the `fetch_mlx` item loop:

```rust
        released_at: item["createdAt"]
            .as_str()
            .and_then(super::dates::parse_iso_date_prefix),
        released_approx: false,
```

  (`detail["createdAt"]` in fetch_hf_repo; verify the MLX code path has a JSON value with `createdAt` — the list API provides it; if fetch_mlx only uses the list API, read it from the list item.)

- [ ] **4.4 Run**: `cargo test -p paddock-core hf` → pass.
- [ ] **4.5 Commit**: `git commit -am "feat(core): HF/MLX release dates from createdAt"`

---

### Task 5: discovery dates from the tags page (TDD)

**Files:**
- Modify: `crates/paddock-core/src/catalog/ollama_registry.rs`

- [ ] **5.1 Write failing test**: extend `LFM25_TAGS_HTML` with per-tag dates and assert in `discover_model_builds_first_seen_size_only`:

```rust
    // In LFM25_TAGS_HTML, add date spans like the real page:
    //   <span x-test-updated>2 months ago</span>
    //   ... per-tag rows: 46e0c10c039e&nbsp;·&nbsp;8 months ago
    // with the OLDEST being "8 months ago".
```

```rust
        // Oldest relative date on the page, marked approximate.
        assert_eq!(
            m.released_at,
            Some(NOW - 8 * 30 * 86_400)
        );
        assert!(m.released_approx);
```

  Also assert a tags page without any date yields `released_at: None, released_approx: false`. `discover_model` needs an injectable `now`: change the signature to `discover_model(http, name, now: i64)` and pass a `const NOW: i64 = 1_780_000_000;` in tests.

- [ ] **5.2 Run, verify failure** (signature change breaks mod.rs too — fix the call site in the same step with `now` threaded from sync; see 5.3).

- [ ] **5.3 Implement**:
  - `discover_model(http, name, now)`: currently calls `fetch_model_tags` which drops the HTML. Refactor: extract the pure parser from `fetch_model_tags` into `fn extract_tag_names(html: &str, base: &str) -> Vec<String>` (same URL-pattern logic), have `fetch_model_tags` fetch + delegate, and in `discover_model` fetch the page text ONCE:

```rust
    let html = http
        .get_text(&format!("https://ollama.com/library/{name}/tags"))
        .await?;
    let tags = extract_tag_names(&html, name);
    let released_at = super::dates::oldest_relative_date(&html, now);
```

  - The final `CatalogModel` gets `released_at, released_approx: released_at.is_some()`.
  - In mod.rs `discover_library_models`, compute once and thread through:

```rust
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
```

  (pass `now` as a parameter to `discover_library_models` from `sync`, computed next to the existing `last_sync` timestamp logic).

- [ ] **5.4 Run**: `cargo test -p paddock-core` → all pass (curated enrichment path untouched: enriched models keep their curated exact date).
- [ ] **5.5 Commit**: `git commit -am "feat(core): approximate release dates for discovered models from tags pages"`

---

### Task 6: age malus in scoring (TDD)

**Files:**
- Modify: `crates/paddock-core/src/score.rs`
- Modify: `crates/paddock/src/app.rs` (call site)

- [ ] **6.1 Write failing tests** (score.rs):

```rust
    #[test]
    fn age_malus_values() {
        assert_eq!(age_malus(None), 0.0);
        assert_eq!(age_malus(Some(0.0)), 0.0);
        assert_eq!(age_malus(Some(182.0)), 0.0); // inside 6-month grace
        let one_year = age_malus(Some(365.25));
        assert!((one_year - 5.0).abs() < 0.1, "1y → ~5, got {one_year}");
        let two_years = age_malus(Some(2.0 * 365.25));
        assert!((two_years - 15.0).abs() < 0.1, "2y → ~15, got {two_years}");
        assert_eq!(age_malus(Some(10.0 * 365.25)), 20.0); // cap
        assert_eq!(age_malus(Some(-50.0)), 0.0); // future date clamps
    }

    #[test]
    fn older_same_size_model_scores_lower() {
        let v = variant("Q4_K_M", 4.83, 8_030_000_000);
        let mem = estimate_memory(&v, DEFAULT_CONTEXT, &budget());
        let speed = estimate_speed(&v, 400.0);
        let fresh = score_variant(&v, &mem, &speed, UseCase::General, None);
        let stale = score_variant(&v, &mem, &speed, UseCase::General, Some(2.0 * 365.25));
        assert!(
            stale.total < fresh.total - 1.0,
            "2-year-old model must lose ground: fresh={} stale={}",
            fresh.total,
            stale.total
        );
        assert!(stale.quality < fresh.quality);
    }
```

- [ ] **6.2 Run, verify failure**: `cargo test -p paddock-core score` → compile error.

- [ ] **6.3 Implement**:

```rust
/// Age malus on quality: 6-month grace, then 10 pts/year, capped.
const AGE_GRACE_YEARS: f64 = 0.5;
const AGE_MALUS_PER_YEAR: f64 = 10.0;
const AGE_MALUS_CAP: f64 = 20.0;

/// Quality penalty for model age (days since release). None (unknown
/// release date) = no malus — absence of data must not punish a model.
fn age_malus(age_days: Option<f64>) -> f64 {
    let Some(days) = age_days else { return 0.0 };
    let years = (days / 365.25).max(0.0);
    (AGE_MALUS_PER_YEAR * (years - AGE_GRACE_YEARS).max(0.0)).min(AGE_MALUS_CAP)
}
```

  - `score_variant(v, mem, speed, uc, age_days: Option<f64>)`: pass `age_days` to `quality_subscore(v, age_days)`, which subtracts `age_malus(age_days)` alongside the existing quant malus (same `.max(0.0)` floor).
  - Update every existing `score_variant`/`score_of` call in score.rs tests with `None` (behavior unchanged).
  - app.rs `scored_models`: compute once before the loop —

```rust
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
```

  and per model: `let age_days = model.released_at.map(|r| (now - r) as f64 / 86_400.0);` → `score_variant(mv, &memory, &speed, use_case, age_days)`.

- [ ] **6.4 Run**: `cargo test` (workspace) → all pass.
- [ ] **6.5 Commit**: `git commit -am "feat: age malus on quality subscore"`

---

### Task 7: AGE column in CLI + TUI + JSON (TDD)

**Files:**
- Modify: `crates/paddock/src/output.rs` (age_label + table + JSON)
- Modify: `crates/paddock/src/tui/draw.rs:102` (header + row)

- [ ] **7.1 Write failing tests** (output.rs test module; create one if absent):

```rust
    #[test]
    fn age_label_cases() {
        const NOW: i64 = 1_780_000_000;
        const DAY: i64 = 86_400;
        assert_eq!(age_label(None, false, NOW), "?");
        assert_eq!(age_label(Some(NOW - 3 * DAY), false, NOW), "3d");
        assert_eq!(age_label(Some(NOW - 20 * DAY), false, NOW), "3w");
        assert_eq!(age_label(Some(NOW - 240 * DAY), false, NOW), "8mo");
        assert_eq!(age_label(Some(NOW - 440 * DAY), false, NOW), "1.2y");
        assert_eq!(age_label(Some(NOW - 440 * DAY), true, NOW), "~1.2y");
        assert_eq!(age_label(Some(NOW + DAY), false, NOW), "0d"); // future clamps
    }
```

- [ ] **7.2 Run, verify failure.**

- [ ] **7.3 Implement** in output.rs:

```rust
/// Compact age: `3d` < 14 days ≤ `3w` < 8 weeks ≤ `8mo` < 12 months ≤ `1.2y`.
/// `~` prefix = approximate source date; `?` = unknown.
pub fn age_label(released_at: Option<i64>, approx: bool, now: i64) -> String {
    let Some(r) = released_at else { return "?".into() };
    let days = ((now - r) as f64 / 86_400.0).max(0.0);
    let core = if days < 14.0 {
        format!("{}d", days as u64)
    } else if days < 56.0 {
        format!("{}w", (days / 7.0) as u64)
    } else if days < 365.25 {
        format!("{}mo", (days / 30.44) as u64)
    } else {
        format!("{:.1}y", days / 365.25)
    };
    if approx {
        format!("~{core}")
    } else {
        core
    }
}
```

  - `print_fit_table`: header gains `AGE` right after `MODEL` (width 6, right-aligned to match the numeric columns); each row prints `age_label(r.model.released_at, r.model.released_approx, now)` with `now` computed once at the top of the function.
  - `print_fit_json`: each row object gains `"released_at": r.model.released_at` and `"released_approx": r.model.released_approx`.
  - tui/draw.rs:102: header array gains `"AGE"` after `"MODEL"`; the row construction adds the same `age_label` call (import `crate::output::age_label`); add one narrow column `Constraint` matching the existing widths table.

- [ ] **7.4 Run**: `cargo test` → pass. Then visual check: `cargo run -q -- fit -n 5` shows the AGE column; `cargo run -q --` (TUI) renders it.
- [ ] **7.5 Commit**: `git commit -am "feat: AGE column in fit table, TUI and JSON output"`

---

### Task 8: real verification + README

**Files:**
- Modify: `README.md` (fit table example + one sentence on the age malus in the scoring section)

- [ ] **8.1 Real sync + checks**:
  - `cargo run -q -- sync` → completes; then `cargo run -q -- fit -n 15`:
    - curated models show exact ages (llama3.1 ≈ `1.9y`, qwen3 ≈ `1.2y`)
    - discovered models show `~` ages or `?`
    - year-old models (qwen3 family) sit visibly lower than in the pre-change screenshot; recent models (gpt-oss, glm) hold or gain.
  - `cargo run -q -- fit --json | python3 -c "import json,sys; rows=json.load(sys.stdin); print([r['released_at'] for r in rows[:3]])"` → epochs present.
  - Paste outputs.
- [ ] **8.2 README**: update the fit example block to include the AGE column and add one sentence to the scoring/catalog prose: quality takes a progressive age malus (6-month grace, −10 pts/year, capped at −20) so stale models sink without vanishing; `~` marks approximate dates, `?` unknown.
- [ ] **8.3 Gates**: `cargo fmt --all && cargo clippy --all-targets && cargo test` → clean.
- [ ] **8.4 Commit**: `git commit -am "feat: model release dates with age-aware ranking"` and push.

---

## Self-review notes

- Spec coverage: data model (T2), three sources (T3/T4/T5), malus (T6), display+JSON (T7), error handling (None-on-unparseable in T1 parsers; curated hard-fails via test in T3), testing (each task TDD).
- `score_variant` signature changes — all call sites enumerated: app.rs (one), score.rs tests (helper `score_of`). `discover_model` signature changes — call sites: mod.rs `discover_library_models`, ollama_registry.rs tests.
- Approx dates get the malus too (spec): nothing to implement — malus reads `released_at` only; `released_approx` is display-only.
- `oldest_relative_date` is deliberately page-global, not per-tag-row: simpler, and the oldest mention is what we want regardless of which row it sits in.
