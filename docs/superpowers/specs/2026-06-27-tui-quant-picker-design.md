# TUI quant picker design

Date: 2026-06-27
Status: approved (design)

## Problem

paddock auto-selects one quantization per model: `best_variant` walks the
quality ladder (Q8 → Q2) and picks the highest-quality quant that fits the GPU
(then sysctl-tunable, then RAM). That is the right default, but a user often
wants a *smaller* quant on purpose: lower memory and higher tok/s for a small
quality hit. Today there is no way to choose. The other quants exist in the
catalog (`CatalogModel.variants`) but are invisible in the UI.

## Goal

Let the user pick the quantization in the TUI's detail popup: list every quant
the model has with its per-quant memory / tok-s / fit, navigate with the arrow
keys, and have `x` (run) / `s` (serve) launch the chosen quant.

## Decisions (locked)

- **Detail popup only.** The global models list stays best-per-model; `x`/`s`
  there remain the quick path (best quant). To pick a quant, open the detail
  popup (`enter`) and choose. No quant cycling in the global list.
- **Arrow keys** (`↑`/`↓`) move the quant selection within the detail popup.
- **All quants selectable**, including non-fitting ones, with their fit verdict
  shown (`fits` / `tune sysctl` / `ram only`). The user's choice, consistent
  with the CLI already serving `FitsRamOnly`.
- CLI `--quant` flag is a later phase (TUI first).

## Components

### 1. Shared quality ordering (`crates/paddock-core/src/score.rs`)

Extract the quant ordering `best_variant` already computes:

```rust
/// Variant indices ordered best-quality-first: by the quant-quality ladder
/// (QUANT_DESCENT), then higher bpw, then quant label. Same order best_variant
/// walks.
pub fn variants_by_quality(variants: &[ModelVariant]) -> Vec<usize>
```

Refactor `best_variant` to walk `variants_by_quality(variants)` over the
fit-verdict ladder, so the ordering lives in one place (its existing tests are
the safety net; behavior is unchanged).

### 2. State (`crates/paddock/src/tui/state.rs`)

- New field `detail_variant: usize` — the index into the selected row's
  `model.variants` currently chosen in the detail popup.
- On entering Detail (`enter` in the models list): set `detail_variant =
  row.variant_idx` (the scored best) and compute `detail_plan` /
  `detail_serve_plan` for it.
- In Detail mode, `↑`/`↓` move `detail_variant` within
  `variants_by_quality(&row.model.variants)` order (clamp at ends), then
  recompute `detail_plan` / `detail_serve_plan` for the new variant.
- Plan builders are parameterized by variant index:
  `plan_run_for(row, idx)` / `plan_serve_for(row, idx)`. The List-tab quick path
  (`x`/`s` without opening detail) uses `row.variant_idx` (best); the Detail-tab
  `x`/`s` use `detail_variant`.

State stays pure; `estimate_memory`/`estimate_speed` for the per-quant display
use `self.budget` (already on `TuiState`) and are computed in the render layer.

### 3. Rendering (`crates/paddock/src/tui/draw.rs`)

`draw_detail` renders, inside the popup:

```
+- detail · Qwen3.6-35B-A3B-MTP-GGUF --------------------------+
| QUANT        MEMORY     TOK/S  FIT                           |
|> UD-Q4_K_XL  23.3 GiB     24   fits                          |
|  Q8_0        37.1 GiB     14   ram only                      |
|  Q6_K        29.3 GiB     18   tune sysctl                   |
|  Q2_K        15.0 GiB     31   fits                          |
|                                                             |
|  up/down pick quant · x run · s serve · esc back            |
| [tok/s vs context chart for the selected quant]             |
+-------------------------------------------------------------+
```

- One row per `model.variants`, ordered by `variants_by_quality`. For each:
  MEMORY (`gib(estimate_memory(v, DEFAULT_CONTEXT, budget).total_bytes)`),
  TOK/S (`estimate_speed(v, bandwidth, kv_cache_bytes(v, DEFAULT_CONTEXT)).generation_tps`),
  FIT (`verdict_label` + `verdict_style`).
- The row whose index == `detail_variant` is highlighted (ACCENT_DEEP), matching
  the list-selection style.
- The speed chart and the run/serve hint use `detail_variant`'s variant (not the
  scored `variant_idx`).
- Single-variant models: the list shows one row; arrows are no-ops.

### 4. Key handling (`state.rs` Detail arm)

Detail mode currently: `Esc`/`q`/`Enter` close, `x` run, `s` serve. Add:
- `↑` / `Up`: `detail_variant` moves one step up the quality order (toward the
  best), clamped; recompute detail plans.
- `↓` / `Down`: one step down (toward smaller), clamped; recompute.
- `x` / `s`: run / serve the `detail_variant` quant (was: the row's best).

## Error handling

- A chosen quant that does not fit GPU still launches (its fit verdict was shown;
  the existing serve/run lifecycle handles `FitsWithSysctlTuning` / `FitsRamOnly`
  exactly as the CLI does).
- A quant whose `plan_serve`/`plan_run` errors (e.g. a non-GGUF MLX pseudo-quant
  on a GGUF runtime) surfaces the error in the footer via the existing
  `last_error` path; the popup stays open on the previous valid plan.

## Testing

State is pure and unit-testable:
- entering Detail sets `detail_variant = variant_idx`;
- `↑`/`↓` move `detail_variant` within the quality order and clamp at both ends;
- after moving, `x`/`s` return a plan for the chosen quant (assert the plan's
  model_ref/quant reflects `detail_variant`, not the best);
- a single-variant model: arrows are no-ops.
Core: `variants_by_quality` orders Q8→Q2 correctly (and `best_variant`'s
existing tests still pass after the refactor). The per-quant render numbers and
the chart are visual, verified by a live smoke test (open detail, arrow through
quants, watch MEMORY/TOK/S/FIT change, serve a smaller quant, confirm it runs).

## Out of scope (YAGNI)

CLI `--quant` flag (later phase), quant cycling in the global list, per-quant
columns in the global table, remembering a per-model quant preference across
sessions.
