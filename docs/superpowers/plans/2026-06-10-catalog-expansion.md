# Catalog Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Many more models, especially Ollama: live enrichment of the Ollama library through the standard OCI registry API (all quant tags per model), bigger curated name list, raised HF/MLX sync limits with CLI flags.

**Architecture:** The curated JSON becomes a NAME + arch-params seed; at sync time, `registry.ollama.ai` (OCI distribution spec) provides the live tag list per model → one CatalogVariant per recognized quant tag, each carrying its exact `source_tag` so run/serve commands reference the real Ollama tag. All HTTP stays behind the existing `HttpClient` trait. Offline fallback = today's behavior (default-tag variant from the embedded JSON).

**Verified-during-implementation facts (WebSearch/live curl, cite in code):**
1. `GET https://registry.ollama.ai/v2/library/{name}/tags/list` response shape + whether anonymous access needs the Docker token dance (401 + WWW-Authenticate → token endpoint). Implement minimal token flow only if required.
2. Real tag naming patterns (e.g. `8b`, `8b-instruct-q4_K_M`, `4b-it-q8_0`, `-text-` for base models) across llama3.1, qwen2.5, gemma3.
3. Decision recorded in code comment: per-tag manifests NOT fetched in v1 (would be ~600 extra requests); `file_size_bytes: None` for enriched variants, estimates derive from params×bpw anyway.

---

### Task A: tetro-core — Ollama registry enrichment + per-variant source tags

**Files:**
- Create: `crates/tetro-core/src/catalog/ollama_registry.rs`
- Modify: `crates/tetro-core/src/catalog/mod.rs` (CatalogVariant.source_tag, export)
- Modify: `crates/tetro-core/src/catalog/db.rs` (source_tag column + migration)
- Modify: `crates/tetro-core/src/runtime.rs` (Source::Ollama refs use source_tag)

- [ ] **A.1 — CatalogVariant gains `pub source_tag: Option<String>`** (serde default). Meaning: the exact Ollama tag (e.g. `8b-instruct-q4_K_M`) for `ollama run {base}:{tag}`; None for curated-offline/HF/MLX variants. All existing constructors add `source_tag: None`.

- [ ] **A.2 — db.rs migration (test-first)**: variants table gains `source_tag TEXT`. In `Db::open`, after the CREATE batch run `ALTER TABLE variants ADD COLUMN source_tag TEXT` ignoring the "duplicate column name" error (match on the message; rusqlite has no IF NOT EXISTS for columns). Upsert/list roundtrip the field; UNIQUE(model_id, quant) unchanged. Test: open an existing pre-migration DB file (create with the old schema SQL in the test), reopen via Db::open → column present, roundtrip works.

- [ ] **A.3 — ollama_registry.rs (test-first, MockHttp fixtures)**:

```rust
//! Live enrichment of curated Ollama models from registry.ollama.ai —
//! standard OCI distribution API, not HTML scraping.

pub struct OllamaTag { pub tag: String, pub quant: String /* normalized, e.g. Q4_K_M */ }

/// GET /v2/library/{base}/tags/list -> all tags; parse + filter to the ones
/// matching `size_prefix` (e.g. "8b") with a recognizable quant suffix.
pub async fn fetch_model_tags(http: &dyn HttpClient, base: &str)
    -> Result<Vec<String>, TetroError>;

/// Pure: keep tags for one size, map quant suffixes to catalog quants.
/// "8b-instruct-q4_K_M" -> ("8b-instruct-q4_K_M", "Q4_K_M");
/// plain "8b" -> default tag (quant from the curated entry);
/// skip "-text-"/"-base-" tags and unknown quants (q4_1, q5_0 stay unknown
/// unless added to quant_bpw — add q4_1 4.78 and q5_1 5.5? NO: YAGNI, skip).
pub fn select_variant_tags(tags: &[String], size_prefix: &str, /* … */)
    -> Vec<OllamaTag>;

/// Enrich one curated model in place: replace its single default variant with
/// one variant per selected tag (bpw via quant_bpw, arch params copied from
/// the curated entry, runtime_compat [Ollama], file_size None, source_tag set).
/// Keep the curated default variant when the registry yields nothing.
pub fn enrich_with_tags(model: &mut CatalogModel, tags: &[OllamaTag]);
```
Auth: if the live API turns out to require the Docker token flow, add it inside `fetch_model_tags` (GET token from the realm in WWW-Authenticate with scope `repository:library/{base}:pull`) via plain `http.get_json`. Tests: tags/list fixture for llama3.1 (mixed sizes, instruct/text, fp16, unknown quants) → exact expected OllamaTag set for "8b"; plain-tag handling; enrich replaces variants and preserves arch params; offline (Err) leaves model untouched.

- [ ] **A.4 — runtime.rs: Source::Ollama refs (test-first)**: `ollama_ref(model, variant)` = `"{base}:{source_tag}"` when variant.source_tag is Some (base = model.name up to ':'), else `model.name`. Used by BOTH plan_run and plan_serve (run argv + pull pre-step + model_ref). Tests: enriched variant "llama3.1:8b" + tag "8b-instruct-q4_K_M" → `ollama run llama3.1:8b-instruct-q4_K_M`; None → today's behavior.

- [ ] **A.5 — gates + commit** `feat(core): ollama registry tag enrichment with exact source tags`.

---

### Task B: curated expansion, sync wiring, CLI flags, README

**Files:**
- Modify: `crates/tetro-core/src/catalog/curated_ollama.json` (+~35 entries)
- Modify: `crates/tetro-core/src/catalog/mod.rs` (SyncOptions, sync(), SyncReport)
- Modify: `crates/tetro/src/cli.rs`, `crates/tetro/src/main.rs` (sync flags)
- Modify: `README.md`

- [ ] **B.1 — curated_ollama.json**: extend to ~75 entries. Add (with correct arch params from each family's config — same families reuse templates): qwen3:0.6b/1.7b, qwen3:235b-a22b, llama3:8b/70b, llama2:7b/13b, gemma3n:e2b/e4b, gemma:2b/7b, phi4-mini:3.8b, codegemma:7b, codestral:22b, devstral:24b, starcoder2:3b/7b/15b, deepseek-coder-v2:16b, command-r:35b, command-r7b, aya-expanse:8b/32b, olmo2:7b/13b, smollm2:135m/360m, nomic-embed? NO (embeddings out of scope), llava NO (vision out of scope), dolphin3:8b, openthinker:7b, tulu3:8b, falcon3:7b/10b, internlm2:7b, yi:6b/9b/34b, glm4:9b, magistral:24b. Keep text-gen only. Validity test already enforces shape + known quants; raise its floor to ≥70 entries.
- [ ] **B.2 — SyncOptions/sync (test-first)**: defaults hf_limit 30→100, mlx_limit 30→60; new `ollama_registry: bool` (default true). sync(): after upserting curated models, when ollama_registry, group curated entries by base name (before ':'), `fetch_model_tags` ONCE per base (~50 requests), `select_variant_tags` per entry size prefix, `enrich_with_tags`, re-upsert. Best-effort: registry errors → report.errors, curated data stands. SyncReport gains `ollama_tags: usize` (count of enriched variants). Stale-variant pruning in db.rs already replaces the old single-variant rows. Tests: mock registry happy path (variant count grows, source_tags set), registry down (curated intact, error reported).
- [ ] **B.3 — CLI**: `tetro sync --hf-limit N --mlx-limit N --no-ollama-registry` mapped to SyncOptions; human output prints `synced: N curated (M ollama tags), X huggingface, Y mlx`. Integration test: `sync --help` shows the flags (no network test).
- [ ] **B.4 — real sync verification**: `cargo run -q -- sync` on this machine. Expect: ~75 curated, several hundred ollama tag variants, ~90+ HF repos, MLX more than 5 (investigate briefly if still ~5 — likely params_from_name failures; report, don't scope-creep). Then `cargo run -q -- fit -n 15` sanity + `run <some enriched model> --json` shows a real tag ref. Paste outputs.
- [ ] **B.5 — README**: catalog section updated (3 sources + registry enrichment, sync flags, counts). Gates + commit `feat: expand catalog — ollama registry tags, bigger curated list, sync flags`.

---

## Self-review notes
- source_tag threading: CatalogVariant → db column (migration safe on old DBs) → runtime ollama_ref → CLI/TUI unchanged (they consume plan argv).
- Tag explosion control: only tags matching the entry's size prefix + known quant suffix; `-text-`/`-base-` skipped; unknown quants skipped (no bpw → no estimate).
- Offline behavior identical to today. Registry = OCI standard, not scraping.
