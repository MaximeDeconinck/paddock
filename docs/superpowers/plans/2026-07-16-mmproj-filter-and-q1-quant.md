# mmproj filter fix + Q1_0 quant support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop mmproj vision-projector files from surfacing as fake model variants (regardless of filename position), and recognize the `Q1_0` 1-bit quant so it appears with a sensible estimate.

**Architecture:** Two small, independent catalog-build fixes. (1) `hf.rs`: change the projector filename check from prefix-match to substring-match. (2) `mod.rs`: add `Q1_0` to the quant bpw table and the filename-recognition list. No estimator changes.

**Tech Stack:** Rust 2024, GGUF/HF catalog sync.

**Spec:** `docs/superpowers/specs/2026-07-16-mmproj-filter-and-q1-quant-design.md`

---

## File Structure

- `crates/paddock-core/src/catalog/hf.rs` - projector filter (`contains("mmproj")`), plus a new test for the mid-name shape.
- `crates/paddock-core/src/catalog/mod.rs` - `quant_bpw` + `KNOWN` gain `Q1_0`, plus tests.

---

## Task 1: mmproj filter matches projector name anywhere

**Files:**
- Modify: `crates/paddock-core/src/catalog/hf.rs` (projector check ~line 175; tests module ~line 583)

- [ ] **Step 1: Write the failing test**

Add a new test to the `#[cfg(test)] mod tests` in `hf.rs`, next to `mmproj_repo_is_llama_cpp_only` (~line 583). It mirrors that test's structure but names the projector with `mmproj` in the MIDDLE of the filename (the Bonsai shape):

```rust
#[tokio::test]
async fn mmproj_mid_name_not_a_variant() {
    // Projector files named "<model>-mmproj-<quant>.gguf" (mmproj in the middle,
    // not the prefix) must still be filtered out, not surfaced as variants.
    let repo = "prism-ml/Bonsai-27B-gguf";
    let detail_url = format!("{HF_API}/models/{repo}?blobs=true");
    let list_url = format!("{HF_API}/models?filter=gguf&sort=downloads&limit=1");
    let range_url =
        format!("https://huggingface.co/{repo}/resolve/main/Bonsai-27B-Q8_0.gguf");

    let http = MockHttp::new()
        .add_json(&list_url, json!([{"id": repo}]))
        .add_json(
            &detail_url,
            json!({
                "id": repo,
                "gguf": {
                    "architecture": "gemma3",
                    "context_length": 8192,
                    "total": 27000000000u64
                },
                "siblings": [
                    {"rfilename": "Bonsai-27B-mmproj-Q8_0.gguf", "size": 629000000u64},
                    {"rfilename": "Bonsai-27B-Q8_0.gguf", "size": 29000000000u64}
                ]
            }),
        )
        .add_range(&range_url, llama_header());

    let models = fetch_hf_gguf(&http, 1).await.unwrap();
    assert_eq!(models.len(), 1);
    let m = &models[0];
    // Only the real Q8_0 file is a variant; the mmproj projector is filtered.
    assert_eq!(m.variants.len(), 1);
    assert_eq!(m.variants[0].quant, "Q8_0");
    // A projector-bearing repo is llama.cpp-only (Ollama cannot import it).
    assert_eq!(m.variants[0].runtime_compat, vec![RuntimeKind::LlamaCpp]);
}
```

NOTE: verify the mock helpers (`MockHttp::new`, `.add_json`, `.add_range`, `llama_header()`, `HF_API`, `RuntimeKind`) and the probe-file selection against the neighboring `mmproj_repo_is_llama_cpp_only` test - they are in the same module. The probe file is the smallest by size; here `Bonsai-27B-Q8_0.gguf` (29 GB) is smaller than nothing else non-projector, and the projector (629 MB) is filtered BEFORE the min-by-size probe pick, so the range URL must point at `Bonsai-27B-Q8_0.gguf`. If the header-probe picks a different file, adjust `range_url` to match the smallest NON-projector file. Use `llama_header()` (dense) so the estimate path succeeds.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p paddock-core mmproj_mid_name_not_a_variant`
Expected: FAIL - `assert_eq!(m.variants.len(), 1)` sees 2 (the projector leaked in as a Q8_0 variant), or a runtime_compat mismatch (has_mmproj false -> Ollama+LlamaCpp).

- [ ] **Step 3: Fix the projector check**

Change (~hf.rs:175):

```rust
        if last_segment.starts_with("mmproj") && last_segment.ends_with(".gguf") {
```

to:

```rust
        if last_segment.contains("mmproj") && last_segment.ends_with(".gguf") {
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p paddock-core mmproj`
Expected: PASS - both `mmproj_mid_name_not_a_variant` (new) and `mmproj_repo_is_llama_cpp_only` (existing prefix form) green.

- [ ] **Step 5: Commit**

```bash
git add crates/paddock-core/src/catalog/hf.rs
git commit -m "fix(catalog): filter mmproj projectors named anywhere in the file"
```

---

## Task 2: Recognize the Q1_0 quant

**Files:**
- Modify: `crates/paddock-core/src/catalog/mod.rs` (`quant_bpw` ~line 103-116; `KNOWN` in `quant_from_filename` ~line 124-126; tests module)

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` in `mod.rs` (near `quant_bpw_ud_tags` / `quant_from_filename_*` tests):

```rust
#[test]
fn q1_0_quant_recognized() {
    assert_eq!(
        quant_from_filename("Bonsai-27B-Q1_0.gguf"),
        Some("Q1_0".to_string())
    );
    assert_eq!(quant_bpw("Q1_0"), Some(1.1));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p paddock-core q1_0_quant_recognized`
Expected: FAIL - `quant_from_filename` returns `None` and `quant_bpw("Q1_0")` returns `None`.

- [ ] **Step 3: Add `Q1_0` to `quant_bpw`**

In the match block (~mod.rs:103-116), add an arm (e.g. after `"Q2_K" => 3.35,`):

```rust
        "Q2_K" => 3.35,
        "Q1_0" => 1.1,
```

- [ ] **Step 4: Add `Q1_0` to the `KNOWN` filename list**

In `quant_from_filename` (~mod.rs:124-126), add `"Q1_0"` to the `KNOWN` array:

```rust
    const KNOWN: &[&str] = &[
        "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K", "Q1_0", "IQ4_XS", "BF16",
        "F16",
    ];
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p paddock-core q1_0_quant_recognized`
Expected: PASS.

- [ ] **Step 6: Full crate suite + clippy + commit**

Run: `cargo test -p paddock-core` (expect green) and `cargo clippy -p paddock-core` (clean).

```bash
git add crates/paddock-core/src/catalog/mod.rs
git commit -m "feat(catalog): recognize the Q1_0 1-bit quant"
```

---

## Task 3: Live verification + resync

**Files:** none (verification only)

- [ ] **Step 1: Build the release binary**

Run: `cargo build --release -p paddock`
Expected: builds clean.

- [ ] **Step 2: Resync the catalog**

Run: `./target/release/paddock sync`
Expected: completes without error.

- [ ] **Step 3: Confirm Bonsai-27B is corrected**

Run: `./target/release/paddock fit --all -n 300 --json 2>/dev/null | python3 -c "import sys,json; [print(m['name'], m['quant']) for m in json.load(sys.stdin) if 'onsai' in m['name'].lower()]"`

Expected: a `Q1_0` variant appears for `Bonsai-27B-gguf`. In the TUI (`./target/release/paddock`, search `bonsai`, open detail) confirm: a `Q1_0` row (~3-4 GiB, `fits`) is present and there is NO phantom `Q8_0` row (the mmproj projector no longer leaks). Quit.

If a phantom `Q8_0` still shows or `Q1_0` is absent, STOP and report the repo's file list vs what surfaced.

No commit (verification only).

---

## Self-Review notes

- **Spec coverage:** Task 1 = mmproj substring filter (spec Component 1) + its test; Task 2 = Q1_0 in `quant_bpw` + `KNOWN` (spec Component 2) + its test; Task 3 = live smoke + resync note (spec Testing / Re-sync). `Q4_1` and `QUANT_DESCENT` correctly untouched (out of scope).
- **No placeholders:** all code shown; the only execution-time confirmation is the mock-helper API + probe-file pick in Task 1, read from the neighboring existing test.
- **Independence:** the two fixes touch different files and are separately committed; either could land alone.
