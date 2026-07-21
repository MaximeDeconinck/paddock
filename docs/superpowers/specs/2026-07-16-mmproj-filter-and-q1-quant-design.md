# mmproj filter fix + Q1_0 quant support design

Date: 2026-07-16
Status: approved (design)

## Problem

Two catalog-quality bugs, surfaced by `prism-ml/Bonsai-27B-gguf`:

1. **mmproj vision-projector files leak in as fake model variants.** paddock
   skips separate vision projectors by checking
   `last_segment.starts_with("mmproj")` (`crates/paddock-core/src/catalog/hf.rs:175`).
   Bonsai names its projectors `Bonsai-27B-mmproj-Q8_0.gguf` /
   `Bonsai-27B-mmproj-BF16.gguf` - the segment starts with `bonsai`, not
   `mmproj`, so the filter misses them. The 629 MB `mmproj-Q8_0` projector then
   surfaces as a bogus `Q8_0` variant whose memory is estimated from the model's
   27B params (29.5 GiB), not the projector's real size. It also means
   `has_mmproj` stays false, so the repo is wrongly marked Ollama-importable.

2. **`Q1_0` (1-bit) quant is dropped.** `quant_bpw` and the `KNOWN` table in
   `crates/paddock-core/src/catalog/mod.rs` do not list `Q1_0`, so
   `quant_from_filename` returns `None` for `Bonsai-27B-Q1_0.gguf` and the real
   1-bit variant never appears.

Net effect for Bonsai-27B: the detail view shows a phantom Q8_0 plus F16/BF16,
while the genuinely useful small quant (Q1_0, 3.8 GB, the only one that fits) is
invisible.

## Goal

- Detect mmproj projector files by their name regardless of position, so they
  are never treated as model variants and correctly flag the repo as
  projector-bearing.
- Recognize the `Q1_0` quant so 1-bit variants appear with a sensible memory /
  speed estimate.

## Decisions (locked)

- **mmproj detection: substring match** `last_segment.contains("mmproj")` (keep
  `.ends_with(".gguf")`). `mmproj` is a specific term; a false positive on a
  real model filename is not a plausible risk.
- **Q1_0 bpw = 1.1**, derived from the observed 3.8 GB file over 27B params
  (1.13 bpw). Added to `quant_bpw` + the `KNOWN` filename table.
- **Q1_0 is NOT added to `QUANT_DESCENT`.** Unknown quants are already ranked
  after the known ladder by bpw, so Q1_0 sorts last (lowest quality) - correct
  for a 1-bit quant.
- **`Q4_1` is out of scope.** Bonsai's `dspark-Q4_1` file (1.79 GB for 27B,
  ~0.5 bpw) is a non-standard sparse format; the params-times-bpw memory model
  would be inaccurate for it.

## Components

### 1. mmproj filter (`crates/paddock-core/src/catalog/hf.rs`)

Change the projector check (~line 175) from:

```rust
if last_segment.starts_with("mmproj") && last_segment.ends_with(".gguf") {
```

to:

```rust
if last_segment.contains("mmproj") && last_segment.ends_with(".gguf") {
```

This is the only change needed: the existing block already sets
`has_mmproj = true` and `continue`s (skipping the file as a variant), and the
downstream `runtime_compat` logic (~line 218: `has_mmproj` -> `LlamaCpp` only)
already consumes that flag. No other edits.

### 2. Q1_0 quant (`crates/paddock-core/src/catalog/mod.rs`)

- In `quant_bpw` (~line 103-116), add an arm: `"Q1_0" => 1.1,`.
- In `quant_from_filename`'s `KNOWN` list (~line 124-126), add `"Q1_0"`.
  Placement in the list is not sensitive (no other KNOWN tag is a substring of
  `Q1_0` and `Q1_0` is not a substring of another), but keep the list readable.

No change to `QUANT_DESCENT` (`score.rs`).

## Error handling

- A repo whose ONLY GGUF files are projectors: after the fix, `files` is empty
  and `fetch_hf_repo` already returns `Ok(None)` (existing guard at ~hf.rs:183).
  No new handling needed.
- Q1_0 memory estimate uses the standard `params * bpw / 8` path; a 1.1 bpw
  yields a plausible small footprint. No special-casing.

## Testing

- **mmproj (extend `mmproj_repo_is_llama_cpp_only`, ~hf.rs:584):** add a sibling
  file named with mmproj in the middle (e.g. `Model-27B-mmproj-Q8_0.gguf`) and
  assert (a) no `Q8_0` variant surfaces from it, and (b) `has_mmproj` behavior
  holds - the repo is `LlamaCpp`-only. Keep the existing `mmproj-BF16.gguf`
  (prefix form) assertion so both naming shapes are covered.
- **Q1_0 (in `mod.rs` tests):** `quant_from_filename("foo-27B-Q1_0.gguf")` ->
  `Some("Q1_0")`; `quant_bpw("Q1_0")` -> `Some(1.1)`.
- **Regression:** existing quant / mmproj tests stay green.
- **Live smoke:** `paddock sync`, then Bonsai-27B detail shows a `Q1_0` variant
  (~3-4 GiB, fits) and no phantom `Q8_0`.

## Re-sync note

The fix corrects catalog build. Existing rows update on the next `paddock sync`
/ TUI `R`. Ships to other users only on a future tagged release.

## Out of scope (YAGNI)

- `Q4_1` / other non-standard sparse ("dspark") formats.
- Adding Q1_0 to the quality ladder (`QUANT_DESCENT`).
- Reading real file sizes for memory instead of the params-times-bpw estimate.
