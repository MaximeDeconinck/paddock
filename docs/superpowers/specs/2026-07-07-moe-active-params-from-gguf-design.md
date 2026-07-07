# MoE active-parameter detection from GGUF design

Date: 2026-07-07
Status: approved (design)

## Problem

paddock's speed estimate is bandwidth-bound over the *active* weights:

```
tps = bandwidth / (params_active * bpw/8 + kv_cache) * efficiency
```

For MoE models `params_active` should be the per-token active parameter count,
far below `params_total`. But on the Hugging Face sync path,
`moe_active_params(repo, params_total)` (in `crates/paddock-core/src/catalog/hf.rs`)
derives active params from the *repo name* only: a `KNOWN` substring table plus a
generic `-aNNb` suffix pattern. Models that do not follow that naming fall
through to `params_total` and are treated as dense.

Concrete failure: `deepreinforce-ai/Ornith-1.0-35B-GGUF` is a Qwen3.5 MoE (256
experts, 8 active per token). Its name has no `-aNNb` marker and no `KNOWN`
entry, so the catalog stored `params_active == params_total == 34.66B`. paddock
estimated ~5 tok/s for Q4_K_M; the real machine (M5, 154 GB/s) delivers 35-40
tok/s (~7x off). The GGUF header already exposes the expert counts paddock reads
but never uses for this.

## Goal

Compute `params_active` for MoE models from the GGUF header (which paddock
already downloads at sync time), analytically, so any standard MoE is sized
correctly regardless of its repo name. Keep the name heuristic as a fallback.

## Decisions (locked)

- **MoE efficiency (`MOE_EFFICIENCY = 0.3`) is out of scope.** This fix only
  corrects `params_active`. Recalibrating the efficiency constant needs real
  measurements (the `paddock bench` roadmap item).
- **Fallback chain:** GGUF-analytic -> name heuristic (`KNOWN` / `-aNNb`) ->
  dense. No regression for models the name heuristic already catches (gpt-oss,
  mixtral, etc.).
- **Scope: the Hugging Face sync path only** (`hf.rs`), which is where the bug
  lives. The curated Ollama JSON keeps its hand-maintained `params_active`; the
  ollama-registry path is untouched.

## Formula

```
active = params_total
       - (expert_count - expert_used_count) * 3 * embedding_dim * expert_ffn_len * layers
```

Rationale: subtract the FFN weights of the experts that are NOT activated per
token. Everything else stays counted - attention, embeddings, shared experts,
and the activated experts. `3` is the SwiGLU FFN matrix count (gate, up, down),
which holds for every major MoE (Qwen, Mixtral, DeepSeek).

Verified against known models:
- Qwen3-30B-A3B: 30.5B - (128-8)*3*2048*768*48 = 30.5B - 27.2B = **3.32B**
  (published active 3.3B).
- Mixtral-8x7B: 46.7B - (8-2)*3*4096*14336*32 = 46.7B - 33.8B = **12.9B**
  (published active 12.9B).

Use saturating subtraction; if the computed inactive mass exceeds total (should
not happen for valid metadata), the result clamps at 0 and the fallback is not
re-invoked (the analytic branch owns the result once its inputs are present).
Guard the analytic branch on `expert_count > expert_used_count` and
`embedding_dim > 0 && layers > 0` so degenerate metadata skips to the fallback.

## Components

### 1. GGUF parser (`crates/paddock-core/src/catalog/gguf.rs`)

Add to `GgufMeta`:

```rust
/// `{arch}.expert_feed_forward_length` - per-expert FFN hidden size (MoE).
pub expert_feed_forward_length: Option<u64>,
```

Parse the key alongside the existing `.expert_count` / `.expert_used_count`
handling (the parser matches on key suffix `.expert_feed_forward_length`).
Unit test via the existing GGUF builder: a header with
`qwen3moe.expert_feed_forward_length = 768` yields `Some(768)`, and the key must
not cross-populate `expert_count` / `expert_used_count`.

### 2. Active-params computation (`crates/paddock-core/src/catalog/hf.rs`)

Refactor the active-param derivation into an analytic-first pure function.
Signature (final names to match existing style during implementation):

```rust
/// Active per-token parameters for an MoE model. Analytic from GGUF expert
/// metadata when available (subtract non-activated experts' FFN weights),
/// else the repo-name heuristic, else dense (params_total).
fn moe_active_params(
    repo: &str,
    params_total: u64,
    layers: u64,
    embedding_dim: u64,
    expert_count: Option<u64>,
    expert_used_count: Option<u64>,
    expert_ffn_len: Option<u64>,
) -> u64
```

- Analytic branch: when `expert_count`, `expert_used_count`, `expert_ffn_len`
  are all `Some`, `expert_count > expert_used_count`, and `embedding_dim > 0 &&
  layers > 0`: apply the formula with `saturating_sub`, return it.
- Else: the current name heuristic (the existing `KNOWN` table + `-aNNb`
  scan), unchanged, kept as a private helper.
- Else: `params_total`.

The build site (currently `params_active: moe_active_params(repo, params_total)`
at ~hf.rs:244) passes the captured GGUF values. The `if let Ok(bytes) ...`
block that parses the header (~hf.rs:196-207) must also capture
`expert_count`, `expert_used_count`, and `expert_feed_forward_length` into outer
variables (like `layers` / `embedding_dim` today) so they are available at the
build site.

### 3. No estimate changes

`estimate_speed` and `MOE_EFFICIENCY` are untouched. The fix only changes what
`params_active` the catalog stores.

## Error handling

- Missing any expert field -> name heuristic -> dense. Never panics.
- Polluted `params_total` (mmproj projector case, already handled at hf.rs:155):
  the analytic subtraction operates on whatever `params_total` the existing
  logic resolved; for real MoE repos the GGUF total is trustworthy. No new
  handling needed.
- Non-SwiGLU FFN (2-matrix): would slightly overestimate the subtracted mass.
  No known major MoE uses this; accepted risk, not guarded.

## Testing

- **gguf.rs:** `expert_feed_forward_length` parses from a builder header; does
  not cross-populate other expert fields.
- **hf.rs `moe_active_params`** (pure, unit-tested):
  - analytic Qwen3-30B-A3B inputs (total 30.5B, 128 experts, 8 used, emb 2048,
    ffn 768, 48 layers) -> ~3.3B (assert within a tolerance, e.g. 3.2-3.4B);
  - analytic Mixtral-8x7B inputs (total 46.7B, 8 experts, 2 used, emb 4096,
    ffn 14336, 32 layers) -> ~12.9B;
  - `expert_ffn_len = None` -> falls back to the name heuristic (a `-aNNb`
    repo still resolves via name);
  - no expert metadata and a non-MoE name -> dense (`params_total`);
  - degenerate metadata (`expert_used_count >= expert_count`, or `layers == 0`)
    -> falls back, does not divide/underflow.
- **estimate:** existing tests still pass (unchanged).
- **Live smoke:** `paddock sync` (or TUI `R`), then confirm Ornith-1.0-35B
  Q4_K_M now reports a tok/s in the tens (near the observed 35-40), not ~5.

## Re-sync note

The fix corrects catalog *build*. Existing rows (Ornith and any other
name-missed MoE) update on the next `paddock sync` / TUI `R`. Ship a note in the
PR so the change is visible after a resync, not immediately on the old snapshot.

## Out of scope (YAGNI)

- MoE efficiency recalibration / `paddock bench`.
- The ollama-registry discovery path and curated JSON (already correct or
  hand-maintained).
- Non-SwiGLU FFN accounting.
- Shared-expert-specific corrections (shared experts are always active and
  already remain counted by the subtract-inactive-only formula).
