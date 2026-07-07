# MoE active-parameter detection from GGUF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute MoE `params_active` analytically from GGUF expert metadata (falling back to the repo-name heuristic, then dense), so any standard MoE is sized correctly regardless of its repo name.

**Architecture:** Add one field to the GGUF parser (`expert_feed_forward_length`), then refactor `hf.rs`'s `moe_active_params` into an analytic-first pure function that subtracts the FFN weights of non-activated experts from `params_total`. The build site captures the new GGUF fields and passes them in. `estimate_speed` is unchanged.

**Tech Stack:** Rust 2024, GGUF header parsing, Hugging Face sync path.

**Spec:** `docs/superpowers/specs/2026-07-07-moe-active-params-from-gguf-design.md`

**Formula:** `active = params_total - (expert_count - expert_used_count) * 3 * embedding_dim * expert_ffn_len * layers` (SwiGLU 3-matrix FFN; subtract inactive experts' weights only).

---

## File Structure

- `crates/paddock-core/src/catalog/gguf.rs` - add `expert_feed_forward_length` to `GgufMeta` + parse its key.
- `crates/paddock-core/src/catalog/hf.rs` - refactor `moe_active_params` to analytic-first with name-heuristic fallback; capture the new GGUF fields at the header-parse site; pass them at the build site.

---

## Task 1: Parse `expert_feed_forward_length` in the GGUF header

**Files:**
- Modify: `crates/paddock-core/src/catalog/gguf.rs` (`GgufMeta` struct ~line 10-25, key dispatch ~line 189-193, tests module)

- [ ] **Step 1: Write the failing test**

Find the existing GGUF builder test that checks expert counts (around `parameter_count_and_expert_counts_parsed`, ~gguf.rs:449). Add a new test right after it:

```rust
#[test]
fn expert_feed_forward_length_parsed() {
    let bytes = GgufBuilder::new()
        .string("general.architecture", "qwen3moe")
        .u32("qwen3moe.expert_count", 128)
        .u32("qwen3moe.expert_used_count", 8)
        .u32("qwen3moe.expert_feed_forward_length", 768)
        .build();
    let m = parse_gguf_header(&bytes).unwrap();
    assert_eq!(m.expert_feed_forward_length, Some(768));
    // The new key must not cross-populate the count fields.
    assert_eq!(m.expert_count, Some(128));
    assert_eq!(m.expert_used_count, Some(8));
}
```

NOTE: confirm the builder helper name (`GgufBuilder` / `.u32(...)` / `.string(...)` / `.build()`) against the existing tests in the file and match them exactly. If the builder lacks a `.u32` for arbitrary keys, use whatever method the neighboring expert-count test uses.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p paddock-core expert_feed_forward_length_parsed`
Expected: compile error - no field `expert_feed_forward_length` on `GgufMeta`.

- [ ] **Step 3: Add the field**

In `GgufMeta` (after `expert_used_count`, ~line 24):

```rust
    /// `{arch}.expert_used_count` - experts active per token.
    pub expert_used_count: Option<u64>,
    /// `{arch}.expert_feed_forward_length` - per-expert FFN hidden size (MoE).
    pub expert_feed_forward_length: Option<u64>,
```

- [ ] **Step 4: Parse the key**

In the key-dispatch chain (~line 189-193), add a branch. Place it next to the other expert keys; the suffix is unique so ordering is safe:

```rust
    } else if key.ends_with(".expert_used_count") {
        meta.expert_used_count = value_u64;
    } else if key.ends_with(".expert_feed_forward_length") {
        meta.expert_feed_forward_length = value_u64;
    } else if key.ends_with(".expert_count") {
        meta.expert_count = value_u64;
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p paddock-core expert_feed_forward_length_parsed`
Expected: PASS.

- [ ] **Step 6: Run the crate suite + commit**

Run: `cargo test -p paddock-core` (expect green - the new `Default`-derived field is `None` everywhere else).

```bash
git add crates/paddock-core/src/catalog/gguf.rs
git commit -m "feat(gguf): parse expert_feed_forward_length"
```

---

## Task 2: Analytic active-params with fallback (`hf.rs`)

**Files:**
- Modify: `crates/paddock-core/src/catalog/hf.rs` (`moe_active_params` ~line 257-289; header-parse block ~line 196-207; build site ~line 244; tests module ~line 628+)

- [ ] **Step 1: Write the failing tests**

The current `moe_active_params(repo, params_total)` is name-based. Add tests for the new analytic signature (see Step 3 for the exact signature). Add to the tests module (near the existing `moe_active_params_*` tests, ~hf.rs:628):

```rust
#[test]
fn moe_active_params_analytic_qwen3_30b() {
    // Qwen3-30B-A3B: 128 experts, 8 used, emb 2048, ffn 768, 48 layers.
    let active = moe_active_params(
        "any/name-without-marker",
        30_500_000_000,
        48,
        2048,
        Some(128),
        Some(8),
        Some(768),
    );
    // Published active ~3.3B; assert within tolerance.
    assert!(
        (3_200_000_000..=3_400_000_000).contains(&active),
        "got {active}"
    );
}

#[test]
fn moe_active_params_analytic_mixtral_8x7b() {
    // Mixtral-8x7B: 8 experts, 2 used, emb 4096, ffn 14336, 32 layers.
    let active = moe_active_params(
        "any/name",
        46_700_000_000,
        32,
        4096,
        Some(8),
        Some(2),
        Some(14336),
    );
    assert!(
        (12_500_000_000..=13_300_000_000).contains(&active),
        "got {active}"
    );
}

#[test]
fn moe_active_params_falls_back_to_name_when_no_ffn_len() {
    // No GGUF expert_ffn_len -> name heuristic still resolves a "-a3b" repo.
    let active = moe_active_params(
        "org/model-30b-a3b",
        30_000_000_000,
        48,
        2048,
        Some(128),
        Some(8),
        None,
    );
    assert_eq!(active, 3_000_000_000);
}

#[test]
fn moe_active_params_dense_when_no_moe_signal() {
    // No expert metadata, non-MoE name -> dense.
    let active = moe_active_params(
        "org/plain-13b",
        13_000_000_000,
        40,
        5120,
        None,
        None,
        None,
    );
    assert_eq!(active, 13_000_000_000);
}

#[test]
fn moe_active_params_degenerate_metadata_falls_back() {
    // used >= count -> analytic skipped, dense (non-MoE name).
    let active = moe_active_params(
        "org/weird",
        20_000_000_000,
        32,
        4096,
        Some(8),
        Some(8),
        Some(14336),
    );
    assert_eq!(active, 20_000_000_000);
}
```

NOTE: the existing name-heuristic tests (`moe_active_params_known_table`, `moe_active_params_generic_suffix`, `moe_active_params_org_name_containing_a_dash` ~hf.rs:628-651) call the OLD 2-arg signature. They must be updated in Step 3 to the new signature (passing `layers`/`emb` and `None` for the three expert args so they exercise the name fallback). Update them, do not delete them.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p paddock-core moe_active_params`
Expected: compile error - arity mismatch on `moe_active_params` (old 2-arg vs new 7-arg).

- [ ] **Step 3: Refactor `moe_active_params` to analytic-first**

Replace the existing `moe_active_params` (hf.rs ~257-289) with a new analytic-first function plus a private name-heuristic helper holding the OLD body:

```rust
/// Active per-token parameters for an MoE model. Analytic from GGUF expert
/// metadata when available (subtract the non-activated experts' FFN weights),
/// else the repo-name heuristic, else dense (`params_total`).
fn moe_active_params(
    repo: &str,
    params_total: u64,
    layers: u64,
    embedding_dim: u64,
    expert_count: Option<u64>,
    expert_used_count: Option<u64>,
    expert_ffn_len: Option<u64>,
) -> u64 {
    if let (Some(experts), Some(used), Some(ffn)) =
        (expert_count, expert_used_count, expert_ffn_len)
        && experts > used
        && embedding_dim > 0
        && layers > 0
    {
        // Subtract the FFN weights of experts not activated per token. 3 =
        // SwiGLU matrices (gate, up, down). Everything else (attention,
        // embeddings, shared + activated experts) stays counted.
        let inactive = experts - used;
        let per_expert_layer = 3u64 * embedding_dim * ffn;
        let inactive_params = inactive
            .saturating_mul(per_expert_layer)
            .saturating_mul(layers);
        return params_total.saturating_sub(inactive_params);
    }
    moe_active_params_by_name(repo, params_total)
}

/// Name-based MoE active-param heuristic: a `KNOWN` substring table plus a
/// generic `-aNNb` suffix scan. Fallback when GGUF expert metadata is absent.
fn moe_active_params_by_name(repo: &str, params_total: u64) -> u64 {
    let r = repo.to_lowercase();
    const KNOWN: &[(&str, u64)] = &[
        ("mixtral-8x7b", 12_900_000_000),
        ("mixtral-8x22b", 39_000_000_000),
        ("qwen3-30b-a3b", 3_300_000_000),
        ("qwen3-235b-a22b", 22_000_000_000),
        ("gpt-oss-20b", 3_600_000_000),
        ("gpt-oss-120b", 5_100_000_000),
        ("deepseek-v3", 37_000_000_000),
    ];
    for (pat, active) in KNOWN {
        if r.contains(pat) {
            return *active;
        }
    }
    for (idx, _) in r.match_indices("-a") {
        let tail = &r[idx + 2..];
        let digits: String = tail
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if !digits.is_empty()
            && tail[digits.len()..].starts_with('b')
            && let Ok(billions) = digits.parse::<f64>()
        {
            return (billions * 1e9) as u64;
        }
    }
    params_total
}
```

- [ ] **Step 4: Capture the new GGUF fields at the header-parse site**

In the `probe_file` header-parse block (hf.rs ~189-207), add outer mutable vars beside `layers`/`embedding_dim` and populate them from `meta`:

Before the `if let Ok(bytes) ...` block, add:

```rust
    let mut expert_count: Option<u64> = None;
    let mut expert_used_count: Option<u64> = None;
    let mut expert_ffn_len: Option<u64> = None;
```

Inside the block (after `embedding_dim = meta.embedding_length.unwrap_or(0) as u32;`), add:

```rust
        expert_count = meta.expert_count;
        expert_used_count = meta.expert_used_count;
        expert_ffn_len = meta.expert_feed_forward_length;
```

- [ ] **Step 5: Pass the fields at the build site**

Change the build site (hf.rs ~244) from:

```rust
        params_active: moe_active_params(repo, params_total),
```

to:

```rust
        params_active: moe_active_params(
            repo,
            params_total,
            layers as u64,
            embedding_dim as u64,
            expert_count,
            expert_used_count,
            expert_ffn_len,
        ),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p paddock-core moe_active_params`
Expected: PASS (5 new + the 3 updated name-heuristic tests).

- [ ] **Step 7: Full crate suite + clippy**

Run: `cargo test -p paddock-core` (expect green) and `cargo clippy -p paddock-core` (clean).

- [ ] **Step 8: Commit**

```bash
git add crates/paddock-core/src/catalog/hf.rs
git commit -m "feat(catalog): compute MoE active params from GGUF expert metadata"
```

---

## Task 3: Live verification + resync

**Files:** none (verification only)

- [ ] **Step 1: Build the release binary**

Run: `cargo build --release -p paddock`
Expected: builds clean.

- [ ] **Step 2: Resync the catalog**

Run: `./target/release/paddock sync`
Expected: completes without error (this rebuilds the catalog with the new active-param logic).

- [ ] **Step 3: Confirm Ornith is now sized correctly**

Query the catalog db:

Run: `sqlite3 "$HOME/Library/Application Support/paddock/catalog.db" "SELECT name, params_total, params_active FROM models WHERE name LIKE '%rnith%';"`
Expected: `Ornith-1.0-35B-GGUF` now shows `params_active` far below `params_total` (roughly 3-4B, not 34.66B).

If `params_active` is still equal to `params_total`, STOP: the HF `gguf` summary for this repo may not expose `expert_feed_forward_length` in the probed header - report the raw GGUF keys seen so we can decide whether the name heuristic needs an Ornith entry as a stopgap.

- [ ] **Step 4: Confirm the estimate in the TUI**

Run: `./target/release/paddock` then search `ornith`, open detail on the 35B, check Q4_K_M TOK/S is now in the tens (near the observed 35-40), not ~5. Quit.

No commit (verification only).

---

## Self-Review notes

- **Spec coverage:** Task 1 = GGUF field (spec Component 1); Task 2 = analytic function + capture + wire + fallback chain (spec Component 2, formula, fallback decision); Task 3 = live smoke + resync note (spec Testing + Re-sync). Efficiency untouched (spec Component 3 / out-of-scope).
- **Signature consistency:** `moe_active_params(repo, params_total, layers, embedding_dim, expert_count, expert_used_count, expert_ffn_len)` used identically in Task 2 tests, definition, and build site. `moe_active_params_by_name(repo, params_total)` is the renamed old body.
- **No placeholders:** all code shown; the only execution-time confirmations are the GGUF test-builder method name (Task 1) and the raw-key fallback check (Task 3), both tool/inspection-authoritative, not guesses.
- **Regression guard:** the three existing name-heuristic tests are updated (not deleted) to the new signature and still assert the name path.
