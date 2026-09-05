# paddock bench design

Date: 2026-08-17
Status: approved (design)

## Problem

paddock's speed estimates are analytic: `tps = bandwidth / bytes_per_token *
efficiency`, with hardcoded efficiency constants in `estimate.rs`
(`SPEED_EFFICIENCY = 0.75` dense, `MOE_SPEED_EFFICIENCY = 0.3` MoE). The MoE
constant demonstrably underestimates on real hardware: Ornith-1.0-35B (Qwen3.5
MoE) estimated ~16 tok/s on an M5 while the real machine delivers 35-40. The
constants' own doc comments promise "a future `paddock bench` module will
recalibrate this per-machine". This is that module.

## Goal

`paddock bench` measures the real generation tok/s of an already-running
server, derives the machine's actual efficiency, persists it, and every future
estimate (TUI detail table, chart, fit scoring) uses the measured value. After
one bench of a MoE model, the 16-vs-38 gap closes.

## Decisions (locked)

- **Measure + recalibrate.** The bench persists a per-machine calibration that
  estimates consume. Not a report-only tool.
- **Benches already-running servers only.** No launch-bench-stop lifecycle in
  v1; the user serves, then benches. Target resolution reuses the `stop`/`logs`
  matching (substring / pid; no target = the single running server, error
  listing candidates when several).
- **Last measurement wins, per class (dense / moe).** No history, no smoothing;
  re-bench at will.
- **Unresolvable model = measure-only.** If the server's `model_ref` cannot be
  matched to a catalog variant (needed for params/bpw), print the measured
  tok/s with a warning and skip the calibration update.

## Components

### 1. Measurement (`paddock-core`, new `bench` module)

One short generation request (default 128 tokens, `--tokens N` override)
against the server's endpoint, preceded by a small warm-up request so cold
start does not pollute the timing. Per runtime:

- **Ollama**: `POST /api/generate` with `stream: false` and
  `options.num_predict = N`. The response carries `eval_count` and
  `eval_duration` (ns): exact tok/s, no wall-clock noise.
- **llama.cpp** (`llama-server`): `POST /completion` with `n_predict = N`. The
  response's `timings` object reports the server-measured generation speed
  (`predicted_per_second`; exact field name verified at implementation time
  against the running llama-server version).
- **mlx** (`mlx_lm.server`): OpenAI `POST /v1/chat/completions` with
  `stream: false` and `max_tokens = N`; no native timings, so tok/s =
  `usage.completion_tokens / wall_time`.

Transport: the existing tiny TcpStream HTTP client (`probe.rs`,
`http_post_local` / `http_body`). Response-size caveat: generation responses
are small (a few KiB), within the client's limits.

Fallback: if a runtime-specific timing field is missing from the response,
fall back to wall-clock + token count when a token count is available;
otherwise fail with a clear error.

### 2. Calibration math + storage (`paddock-core`)

At near-empty context (short bench prompt), KV traffic is negligible, so:

```
eff_measured = measured_tps * (params_active * bpw / 8) / (bandwidth_gbps * 1e9)
```

(consistent units with `estimate_speed`; the estimate formula already models KV
separately, so calibrating at KV~0 is correct - efficiency captures kernel
quality, not context depth).

- **Variant resolution**: the server's `model_ref`:
  - `hf.co/{org}/{repo}:{quant}` (spawned llama.cpp) -> catalog model by repo,
    variant by quant label (case-insensitive, like `resolve_quant`).
  - Ollama tag (`name:tag`) -> curated variant by `source_tag`, else model by
    base name + best-matching variant.
  - MLX refs -> catalog by repo.
  - No match -> measure-only path (warning, no calibration write).
- **Class**: `moe` if the resolved model has `params_active < params_total`,
  else `dense`.
- **Storage**: `calibration.json` in `paths::app_support_dir()`:

```json
{
  "dense": {"efficiency": 0.71, "model": "...", "measured_at": 1786...},
  "moe":   {"efficiency": 0.68, "model": "...", "measured_at": 1786...}
}
```

Either entry optional. Corrupt/missing file = no calibration (defaults apply).
Sanity clamp on write: efficiency outside (0.05, 1.5) is rejected with an error
(a wildly implausible value means the measurement or resolution is wrong; do
not poison future estimates).

### 3. Calibration consumption (`estimate.rs` + call sites)

- New `SpeedCalibration { dense: f64, moe: f64 }` with
  `Default = { SPEED_EFFICIENCY, MOE_SPEED_EFFICIENCY }`.
- `estimate_speed` keeps its exact signature and behavior (wrapper over the
  calibrated variant with `SpeedCalibration::default()`), so every existing
  test and caller is untouched by default.
- New `estimate_speed_calibrated(v, bandwidth_gbps, kv_cache_bytes, &SpeedCalibration)`
  picks `dense`/`moe` by the existing `params_active < params_total` test.
- `App` loads `calibration.json` once at startup into a `SpeedCalibration`
  (defaults when absent) and the production call sites use it:
  - the scoring path in `score.rs` (via however `App::scored_models` threads
    budget/bandwidth today - the calibration travels the same road),
  - the TUI detail table + speed chart in `tui/draw.rs`.
- Result: no bench run = identical behavior to today; after a bench, TUI and
  fit reflect the machine's measured efficiency.

### 4. CLI (`paddock bench`)

```
paddock bench [target] [--tokens N] [--json]
```

- Target resolution: same matcher as `stop`/`logs` over the running servers
  (paddock-spawned + Ollama-loaded, i.e. `list_all_servers`).
- Text output (one line per fact, no em-dash):

```
model      Ornith-1.0-35B-GGUF Q4_K_M (moe)
measured   38.2 tok/s
estimated  16.5 tok/s (before calibration)
efficiency 0.68
calibration updated: moe 0.30 -> 0.68
```

- Measure-only path prints measured tok/s + `calibration not updated: <reason>`.
- `--json` emits the same fields machine-readable.

## Error handling

- No running server: clear message ("nothing to bench - serve a model first").
- Several servers, no target: list them, ask for a target (non-zero exit).
- Server dies / connection refused mid-bench: clean error, no calibration write.
- Response missing expected fields: wall-clock fallback when token count is
  available, else error naming the missing field.
- Unresolvable model_ref: measured tok/s printed, calibration skipped, warning.
- Implausible efficiency (outside 0.05-1.5): error, no write.

## Testing

Pure units:
- calibration math: eff from (tps, params_active, bpw, bandwidth); MoE and
  dense examples with known numbers.
- per-runtime response parsing from JSON fixtures (ollama eval fields,
  llama.cpp timings, OpenAI usage + injected wall time).
- `calibration.json` round-trip (write, load, corrupt file -> defaults,
  clamp rejection).
- `estimate_speed_calibrated` applies dense vs moe factors;
  `estimate_speed` (default wrapper) unchanged - existing tests prove it.
- variant resolution from model_ref shapes (hf.co ref, ollama tag, no match).

Live smoke: serve a MoE (e.g. the Qwen3.6-35B-A3B already cached), `paddock
bench`, confirm measured ~35-40, calibration written, and the TUI detail then
estimates near the measured value.

## Out of scope (YAGNI)

- Bench from the TUI (v2 candidate).
- Prompt-speed (prefill) calibration; `PROMPT_SPEED_FACTOR` untouched.
- Launch-bench-stop lifecycle.
- Multi-measurement history / smoothing / per-model calibration.
- Cross-machine calibration sharing.
