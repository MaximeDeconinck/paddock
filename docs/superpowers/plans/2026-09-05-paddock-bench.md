# paddock bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `paddock bench [target]` measures the real generation tok/s of an already-running server, derives this machine's bandwidth efficiency for the model's class (dense / moe), persists it in `calibration.json`, and every speed estimate (fit scoring, TUI detail table, speed chart) consumes the measured value.

**Architecture:** Three pure core pieces plus one thin CLI. `estimate.rs` gains a `SpeedCalibration` struct and `estimate_speed_calibrated`; the old `estimate_speed` becomes a wrapper with defaults so nothing changes until a bench runs. A new `calibration.rs` owns the math (`eff = tps * bytes_per_token / bandwidth`), the sanity clamp and the JSON file. A new `bench.rs` builds the per-runtime generation request, parses the per-runtime timing fields (wall-clock fallback), and maps a server `model_ref` back to a catalog variant. `App` loads the calibration once; `main.rs` wires the `bench` subcommand over `list_all_servers`.

**Tech Stack:** Rust 2024, serde_json, the existing `SystemProbe::http_post_local` TcpStream client (no new deps).

**Spec:** `docs/superpowers/specs/2026-08-17-paddock-bench-design.md`

## Global Constraints

- Measure + recalibrate: the bench writes a per-machine calibration that estimates consume.
- Benches already-running servers only (no launch/stop lifecycle).
- Last measurement wins, per class (`dense` / `moe`). No history, no smoothing.
- Unresolvable `model_ref` = measure-only: print measured tok/s, warn, skip the calibration write.
- Efficiency outside the open interval (0.05, 1.5) is rejected with an error and never written.
- `estimate_speed(v, bandwidth_gbps, kv_cache_bytes)` keeps its exact signature and default behavior.
- Default bench length 128 tokens, `--tokens N` override.
- Storage: `calibration.json` in `paths::app_support_dir()`; `PADDOCK_CALIBRATION_PATH` env override for tests.
- No em-dash anywhere (code, docs, output). Use `-` or `->`.
- Text output is one fact per line, aligned like `paddock scan`.

---

## File Structure

- `crates/paddock-core/src/estimate.rs` - add `SpeedCalibration` (+ `Default`, `for_variant`) and `estimate_speed_calibrated`; `estimate_speed` delegates to it with defaults. Update the two efficiency-constant doc comments (the bench module now exists).
- `crates/paddock-core/src/calibration.rs` (new) - `ModelClass`, `CalibrationEntry`, `CalibrationFile`, `default_calibration_path`, `load`, `save`, `efficiency_from_measurement`, `validate_efficiency`.
- `crates/paddock-core/src/bench.rs` (new) - `TimingSource`, `BenchMeasurement`, `ParsedTiming`, `bench_request`, `parse_timing`, `finalize`, `measure`, `resolve_model_ref`.
- `crates/paddock-core/src/serving.rs` - `ServerRowMatch` + `match_server_rows` (target resolution over `ServerRow`, since `list_all_servers` returns rows, not records).
- `crates/paddock-core/src/lib.rs` - export the two new modules.
- `crates/paddock/src/app.rs` - `App.calibration: SpeedCalibration` loaded at startup; `scored_models` uses `estimate_speed_calibrated`.
- `crates/paddock/src/tui/state.rs`, `tui/mod.rs`, `tui/draw.rs` - thread the calibration into `TuiState` and the detail table / speed chart.
- `crates/paddock/src/cli.rs`, `main.rs`, `output.rs` - `Bench { target, tokens }` subcommand, `bench_server`, `BenchReport` + `print_bench`.
- `crates/paddock/tests/cli.rs` - CLI smoke for the no-match path.
- `README.md` - `paddock bench` section, subcommand count, roadmap and "How the estimates work" updates.

---

## Task 1: `SpeedCalibration` and `estimate_speed_calibrated`

**Files:**
- Modify: `crates/paddock-core/src/estimate.rs` (constants doc comments lines 11-21, `estimate_speed` lines 203-241, tests module)

**Interfaces:**
- Produces: `pub struct SpeedCalibration { pub dense: f64, pub moe: f64 }` with `Default` = `{ SPEED_EFFICIENCY, MOE_SPEED_EFFICIENCY }` and `pub fn for_variant(&self, v: &ModelVariant) -> f64`; `pub fn estimate_speed_calibrated(v: &ModelVariant, bandwidth_gbps: f64, kv_cache_bytes: u64, cal: &SpeedCalibration) -> SpeedEstimate`.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `estimate.rs`:

```rust
    #[test]
    fn default_calibration_matches_constants() {
        let cal = SpeedCalibration::default();
        assert_eq!(cal.dense, SPEED_EFFICIENCY);
        assert_eq!(cal.moe, MOE_SPEED_EFFICIENCY);
        let dense = llama31_8b_q4km();
        let mut moe = llama31_8b_q4km();
        moe.params_total = 30_000_000_000;
        moe.params_active = 3_000_000_000;
        assert_eq!(cal.for_variant(&dense), SPEED_EFFICIENCY);
        assert_eq!(cal.for_variant(&moe), MOE_SPEED_EFFICIENCY);
    }

    /// The calibrated estimator scales linearly with the class factor and the
    /// default wrapper is exactly the calibrated one at default factors.
    #[test]
    fn calibrated_speed_scales_with_class_factor() {
        let dense = llama31_8b_q4km();
        let mut moe = llama31_8b_q4km();
        moe.params_total = 30_000_000_000;
        moe.params_active = 3_000_000_000;
        let cal = SpeedCalibration { dense: 0.5, moe: 0.6 };

        let d_default = estimate_speed(&dense, 400.0, 0).generation_tps;
        let d_cal = estimate_speed_calibrated(&dense, 400.0, 0, &cal).generation_tps;
        assert!((d_cal / d_default - 0.5 / SPEED_EFFICIENCY).abs() < 1e-9, "{d_cal} vs {d_default}");

        let m_default = estimate_speed(&moe, 400.0, 0).generation_tps;
        let m_cal = estimate_speed_calibrated(&moe, 400.0, 0, &cal).generation_tps;
        assert!((m_cal / m_default - 0.6 / MOE_SPEED_EFFICIENCY).abs() < 1e-9, "{m_cal} vs {m_default}");

        let same = estimate_speed_calibrated(&dense, 400.0, 0, &SpeedCalibration::default());
        assert_eq!(same.generation_tps, d_default);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p paddock-core calibrat`
Expected: compile error, `cannot find type SpeedCalibration`.

- [ ] **Step 3: Implement**

Replace the two constant doc comments and `estimate_speed` in `estimate.rs`:

```rust
/// Fraction of theoretical bandwidth actually achieved by llama.cpp/MLX kernels
/// for dense models. Default calibrated on community benchmarks; `paddock bench`
/// overrides it per machine (see `SpeedCalibration`).
pub const SPEED_EFFICIENCY: f64 = 0.75;
/// Fraction of theoretical bandwidth achieved for MoE models: expert routing
/// scatters weight reads, so kernels reach far less of peak bandwidth than the
/// dense streaming case. Measured ≈0.29 on M5 (Qwen3.6-35B-A3B UD-Q4_K_XL,
/// 2026-06) and ≈0.3 implied by community Qwen3-30B-A3B numbers on M3 Max;
/// known to underestimate on some machines, which is what `paddock bench`
/// corrects per machine (see `SpeedCalibration`).
pub const MOE_SPEED_EFFICIENCY: f64 = 0.3;
```

```rust
/// Per-machine bandwidth-efficiency factors consumed by
/// `estimate_speed_calibrated`. Defaults are the community-calibrated
/// constants; `paddock bench` replaces them with values measured on this
/// machine (persisted by `calibration.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeedCalibration {
    pub dense: f64,
    pub moe: f64,
}

impl Default for SpeedCalibration {
    fn default() -> Self {
        Self {
            dense: SPEED_EFFICIENCY,
            moe: MOE_SPEED_EFFICIENCY,
        }
    }
}

impl SpeedCalibration {
    /// Efficiency factor for a variant: MoE when `params_active < params_total`,
    /// dense otherwise (the same class test the estimator always used).
    pub fn for_variant(&self, v: &ModelVariant) -> f64 {
        if v.params_active < v.params_total {
            self.moe
        } else {
            self.dense
        }
    }
}

/// Estimate generation speed given memory bandwidth, at the default
/// (uncalibrated) efficiency factors. Thin wrapper over
/// `estimate_speed_calibrated`; kept so existing callers and tests are untouched.
/// `bandwidth_gbps`: must be finite and ≥ 0; non-finite/negative values are treated as 0.0.
/// `kv_cache_bytes`: every decoded token re-streams the KV cache built so far
/// on top of the active weights, so speed decays with context depth. Callers
/// pass the DEFAULT_CONTEXT cache from `estimate_memory`, keeping the MEMORY
/// and TOK/S columns consistent at the same 8k depth; 0 models an empty
/// context.
pub fn estimate_speed(v: &ModelVariant, bandwidth_gbps: f64, kv_cache_bytes: u64) -> SpeedEstimate {
    estimate_speed_calibrated(v, bandwidth_gbps, kv_cache_bytes, &SpeedCalibration::default())
}

/// `estimate_speed` with explicit efficiency factors. The bench calibrates at
/// near-empty context (KV ~ 0), which is correct here: the KV term is modeled
/// separately, so the efficiency captures kernel quality, not context depth.
pub fn estimate_speed_calibrated(
    v: &ModelVariant,
    bandwidth_gbps: f64,
    kv_cache_bytes: u64,
    cal: &SpeedCalibration,
) -> SpeedEstimate {
    let bpw = if v.bpw.is_finite() && v.bpw > 0.0 {
        v.bpw
    } else {
        0.0
    };
    let safe_bw = if bandwidth_gbps.is_finite() && bandwidth_gbps >= 0.0 {
        bandwidth_gbps
    } else {
        0.0
    };
    let efficiency = cal.for_variant(v);
    let bytes_per_token = v.params_active as f64 * bpw / 8.0 + kv_cache_bytes as f64;
    let generation_tps = if bytes_per_token > 0.0 {
        safe_bw * 1e9 / bytes_per_token * efficiency
    } else {
        0.0
    };
    SpeedEstimate {
        generation_tps,
        prompt_tps_range: (
            generation_tps * PROMPT_SPEED_FACTOR.0,
            generation_tps * PROMPT_SPEED_FACTOR.1,
        ),
        tier: SpeedTier::from_tps(generation_tps),
    }
}
```

- [ ] **Step 4: Run the core tests**

Run: `cargo test -p paddock-core`
Expected: all pass, including the two new tests and every pre-existing `estimate_speed` test.

- [ ] **Step 5: Commit**

```bash
git add crates/paddock-core/src/estimate.rs
git commit -m "feat(estimate): SpeedCalibration and estimate_speed_calibrated"
```

---

## Task 2: Calibration math and `calibration.json` storage

**Files:**
- Create: `crates/paddock-core/src/calibration.rs`
- Modify: `crates/paddock-core/src/lib.rs` (add `pub mod calibration;`)

**Interfaces:**
- Consumes: `estimate::{ModelVariant, SpeedCalibration}`, `paths::app_support_dir`, `PaddockError::Other`.
- Produces:
  - `pub const EFFICIENCY_MIN: f64 = 0.05; pub const EFFICIENCY_MAX: f64 = 1.5;`
  - `pub enum ModelClass { Dense, Moe }` with `pub fn of(v: &ModelVariant) -> Self` and `pub fn label(&self) -> &'static str` ("dense" / "moe").
  - `pub struct CalibrationEntry { pub efficiency: f64, pub model: String, pub measured_at: i64 }`
  - `pub struct CalibrationFile { pub dense: Option<CalibrationEntry>, pub moe: Option<CalibrationEntry> }` with `pub fn set(&mut self, class: ModelClass, entry: CalibrationEntry)` and `pub fn to_speed_calibration(&self) -> SpeedCalibration`.
  - `pub fn default_calibration_path() -> PathBuf`
  - `pub fn load(path: &Path) -> CalibrationFile`
  - `pub fn save(path: &Path, file: &CalibrationFile) -> Result<(), PaddockError>`
  - `pub fn efficiency_from_measurement(measured_tps: f64, params_active: u64, bpw: f64, bandwidth_gbps: f64) -> f64`
  - `pub fn validate_efficiency(eff: f64) -> Result<f64, PaddockError>`

- [ ] **Step 1: Write the failing tests**

Create `crates/paddock-core/src/calibration.rs` containing only the tests module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::{MOE_SPEED_EFFICIENCY, SPEED_EFFICIENCY};

    /// Llama 3.1 8B Q4_K_M on an M2 Max (400 GB/s) measured 62.2 tok/s:
    /// bytes/token = 8.03e9 * 4.83 / 8 = 4.848e9, eff = 62.2 * 4.848e9 / 400e9.
    #[test]
    fn dense_efficiency_from_measurement() {
        let eff = efficiency_from_measurement(62.2, 8_030_000_000, 4.83, 400.0);
        assert!((eff - 0.754).abs() < 0.002, "got {eff}");
    }

    /// Qwen3.5 MoE 35B-A3B Q4_K_M on an M5 (153.6 GB/s) measured 38.2 tok/s:
    /// bytes/token = 3e9 * 4.83 / 8 = 1.811e9, eff = 38.2 * 1.811e9 / 153.6e9.
    #[test]
    fn moe_efficiency_uses_active_params() {
        let eff = efficiency_from_measurement(38.2, 3_000_000_000, 4.83, 153.6);
        assert!((eff - 0.450).abs() < 0.002, "got {eff}");
    }

    #[test]
    fn hostile_measurement_inputs_yield_zero() {
        assert_eq!(efficiency_from_measurement(f64::NAN, 1, 4.0, 400.0), 0.0);
        assert_eq!(efficiency_from_measurement(30.0, 0, 4.0, 400.0), 0.0);
        assert_eq!(efficiency_from_measurement(30.0, 1_000_000_000, 4.0, 0.0), 0.0);
        assert_eq!(efficiency_from_measurement(30.0, 1_000_000_000, f64::NAN, 400.0), 0.0);
    }

    #[test]
    fn validate_rejects_implausible_efficiency() {
        assert_eq!(validate_efficiency(0.68).unwrap(), 0.68);
        assert!(validate_efficiency(0.0).is_err());
        assert!(validate_efficiency(0.05).is_err());
        assert!(validate_efficiency(1.5).is_err());
        assert!(validate_efficiency(3.2).is_err());
        assert!(validate_efficiency(f64::NAN).is_err());
        let msg = validate_efficiency(3.2).unwrap_err().to_string();
        assert!(msg.contains("implausible efficiency"), "{msg}");
        assert!(msg.contains("3.2"), "{msg}");
    }

    #[test]
    fn model_class_from_variant() {
        let mut v = crate::estimate::ModelVariant {
            model_name: "m".into(),
            quant: "Q4_K_M".into(),
            bpw: 4.83,
            params_total: 8_000_000_000,
            params_active: 8_000_000_000,
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
            embedding_dim: 4096,
            context_max: 8192,
        };
        assert_eq!(ModelClass::of(&v), ModelClass::Dense);
        assert_eq!(ModelClass::Dense.label(), "dense");
        v.params_active = 3_000_000_000;
        assert_eq!(ModelClass::of(&v), ModelClass::Moe);
        assert_eq!(ModelClass::Moe.label(), "moe");
    }

    #[test]
    fn file_roundtrips_and_partial_entries_fall_back_to_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("calibration.json");
        let mut file = CalibrationFile::default();
        file.set(
            ModelClass::Moe,
            CalibrationEntry {
                efficiency: 0.68,
                model: "Ornith-1.0-35B-GGUF Q4_K_M".into(),
                measured_at: 1_786_000_000,
            },
        );
        save(&path, &file).unwrap();
        let back = load(&path);
        assert_eq!(back, file);
        let cal = back.to_speed_calibration();
        assert_eq!(cal.moe, 0.68);
        assert_eq!(cal.dense, SPEED_EFFICIENCY, "missing class keeps the default");
        // The JSON has no `dense` key at all when the entry is absent.
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("\"dense\""), "{raw}");
    }

    #[test]
    fn missing_or_corrupt_file_means_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope.json");
        assert_eq!(load(&missing), CalibrationFile::default());
        let corrupt = dir.path().join("bad.json");
        std::fs::write(&corrupt, b"{ not json").unwrap();
        assert_eq!(load(&corrupt), CalibrationFile::default());
        let cal = load(&corrupt).to_speed_calibration();
        assert_eq!(cal.dense, SPEED_EFFICIENCY);
        assert_eq!(cal.moe, MOE_SPEED_EFFICIENCY);
    }

    #[test]
    fn set_replaces_the_class_entry() {
        let mut file = CalibrationFile::default();
        let e = |eff: f64| CalibrationEntry {
            efficiency: eff,
            model: "x".into(),
            measured_at: 1,
        };
        file.set(ModelClass::Dense, e(0.6));
        file.set(ModelClass::Dense, e(0.7));
        assert_eq!(file.dense.as_ref().unwrap().efficiency, 0.7);
        assert!(file.moe.is_none());
    }

    #[test]
    fn env_override_wins_for_calibration_path() {
        // Set, read, unset BEFORE asserting so a failure never leaks the var.
        // SAFETY: env mutation; no other test in this crate reads this var.
        unsafe { std::env::set_var("PADDOCK_CALIBRATION_PATH", "/tmp/xyz-cal.json") };
        let got = default_calibration_path();
        unsafe { std::env::remove_var("PADDOCK_CALIBRATION_PATH") };
        assert_eq!(got, std::path::PathBuf::from("/tmp/xyz-cal.json"));
    }
}
```

Add `pub mod calibration;` to `lib.rs` (alphabetical, after `catalog`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p paddock-core calibration::`
Expected: compile errors, `cannot find function efficiency_from_measurement` etc.

- [ ] **Step 3: Implement the module**

Prepend to `calibration.rs` (above the tests module):

```rust
//! Per-machine speed calibration: written by `paddock bench`, consumed by
//! every speed estimate through `estimate::SpeedCalibration`.
//!
//! At near-empty context the KV term is negligible, so a measured generation
//! speed pins the efficiency factor directly:
//!     eff = measured_tps * (params_active * bpw / 8) / (bandwidth * 1e9)
//! One entry per model class (dense / moe); the last measurement wins.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::PaddockError;
use crate::estimate::{ModelVariant, SpeedCalibration};

/// Open interval of plausible efficiency factors. Outside it, the measurement
/// or the catalog match is wrong; never persist such a value.
pub const EFFICIENCY_MIN: f64 = 0.05;
pub const EFFICIENCY_MAX: f64 = 1.5;

/// Speed class a calibration entry applies to: the same `params_active <
/// params_total` test `SpeedCalibration::for_variant` uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelClass {
    Dense,
    Moe,
}

impl ModelClass {
    pub fn of(v: &ModelVariant) -> Self {
        if v.params_active < v.params_total {
            Self::Moe
        } else {
            Self::Dense
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Moe => "moe",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationEntry {
    pub efficiency: f64,
    /// Human label of what was benched, e.g. `Ornith-1.0-35B-GGUF Q4_K_M`.
    pub model: String,
    /// Unix seconds.
    pub measured_at: i64,
}

/// On-disk shape of `calibration.json`. Either entry optional.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense: Option<CalibrationEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<CalibrationEntry>,
}

impl CalibrationFile {
    /// Replace the entry for `class` (last measurement wins).
    pub fn set(&mut self, class: ModelClass, entry: CalibrationEntry) {
        match class {
            ModelClass::Dense => self.dense = Some(entry),
            ModelClass::Moe => self.moe = Some(entry),
        }
    }

    /// Factors for the estimator: measured where present, defaults elsewhere.
    pub fn to_speed_calibration(&self) -> SpeedCalibration {
        let d = SpeedCalibration::default();
        SpeedCalibration {
            dense: self.dense.as_ref().map_or(d.dense, |e| e.efficiency),
            moe: self.moe.as_ref().map_or(d.moe, |e| e.efficiency),
        }
    }
}

/// `~/Library/Application Support/paddock/calibration.json`;
/// `PADDOCK_CALIBRATION_PATH` overrides (tests).
pub fn default_calibration_path() -> PathBuf {
    if let Ok(p) = std::env::var("PADDOCK_CALIBRATION_PATH") {
        return PathBuf::from(p);
    }
    crate::paths::app_support_dir().join("calibration.json")
}

/// Missing or corrupt file = no calibration (defaults apply). Never errors:
/// a bad calibration file must not take the whole app down.
pub fn load(path: &Path) -> CalibrationFile {
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

/// Atomic write (tmp + rename), creating the parent directory.
pub fn save(path: &Path, file: &CalibrationFile) -> Result<(), PaddockError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| PaddockError::Other(format!("cannot create {parent:?}: {e}")))?;
    }
    let json =
        serde_json::to_vec_pretty(file).map_err(|e| PaddockError::Other(e.to_string()))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)
        .map_err(|e| PaddockError::Other(format!("cannot write calibration: {e}")))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| PaddockError::Other(format!("cannot finalize calibration: {e}")))
}

/// Efficiency implied by a measurement at near-empty context. Returns 0.0 on
/// hostile inputs (NaN speed, zero params, zero bandwidth), which
/// `validate_efficiency` then rejects.
pub fn efficiency_from_measurement(
    measured_tps: f64,
    params_active: u64,
    bpw: f64,
    bandwidth_gbps: f64,
) -> f64 {
    if !measured_tps.is_finite() || !bpw.is_finite() || !bandwidth_gbps.is_finite() {
        return 0.0;
    }
    let bytes_per_token = params_active as f64 * bpw / 8.0;
    if bytes_per_token <= 0.0 || bandwidth_gbps <= 0.0 || measured_tps <= 0.0 {
        return 0.0;
    }
    measured_tps * bytes_per_token / (bandwidth_gbps * 1e9)
}

/// Sanity clamp before persisting.
pub fn validate_efficiency(eff: f64) -> Result<f64, PaddockError> {
    if eff.is_finite() && eff > EFFICIENCY_MIN && eff < EFFICIENCY_MAX {
        Ok(eff)
    } else {
        Err(PaddockError::Other(format!(
            "implausible efficiency {eff:.3} (expected between {EFFICIENCY_MIN} and {EFFICIENCY_MAX}); \
             the served model probably does not match its catalog entry"
        )))
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p paddock-core calibration::`
Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/paddock-core/src/calibration.rs crates/paddock-core/src/lib.rs
git commit -m "feat(core): calibration math and calibration.json storage"
```

---

## Task 3: Bench measurement (request, parsing, wall-clock fallback)

**Files:**
- Create: `crates/paddock-core/src/bench.rs`
- Modify: `crates/paddock-core/src/lib.rs` (add `pub mod bench;`)

**Interfaces:**
- Consumes: `catalog::RuntimeKind`, `hardware::SystemProbe::http_post_local`, `PaddockError::Other`.
- Produces:
  - `pub const DEFAULT_BENCH_TOKENS: u32 = 128;`
  - `pub enum TimingSource { ServerTimings, WallClock }` (Serialize snake_case) with `pub fn label(&self) -> &'static str` ("server timings" / "wall clock").
  - `pub struct BenchMeasurement { pub tps: f64, pub tokens: u32, pub timing: TimingSource }`
  - `pub struct ParsedTiming { pub tokens: u32, pub server_tps: Option<f64> }`
  - `pub fn bench_request(runtime: RuntimeKind, endpoint: &str, model_ref: &str, tokens: u32) -> (String, String)` (url, json body)
  - `pub fn parse_timing(runtime: RuntimeKind, body: &str) -> Result<ParsedTiming, PaddockError>`
  - `pub fn finalize(parsed: ParsedTiming, wall: std::time::Duration) -> Result<BenchMeasurement, PaddockError>`
  - `pub fn measure(probe: &dyn SystemProbe, runtime: RuntimeKind, endpoint: &str, model_ref: &str, tokens: u32) -> Result<BenchMeasurement, PaddockError>`

Field names verified against the runtimes: Ollama `/api/generate` (non-stream) returns `eval_count` and `eval_duration` (ns); llama-server `/completion` returns `tokens_predicted` and `timings.{predicted_n, predicted_ms, predicted_per_second}`; mlx_lm.server `/v1/chat/completions` returns OpenAI `usage.completion_tokens` and no timings.

- [ ] **Step 1: Write the failing tests**

Create `crates/paddock-core/src/bench.rs` with the tests module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::MockProbe;
    use std::time::Duration;

    #[test]
    fn ollama_request_shape() {
        let (url, body) = bench_request(RuntimeKind::Ollama, "http://127.0.0.1:11434", "llama3.2:3b", 128);
        assert_eq!(url, "http://127.0.0.1:11434/api/generate");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"], "llama3.2:3b");
        assert_eq!(v["stream"], false);
        assert_eq!(v["options"]["num_predict"], 128);
        assert!(v["prompt"].as_str().unwrap().len() > 10);
    }

    #[test]
    fn llama_cpp_request_shape() {
        let (url, body) = bench_request(RuntimeKind::LlamaCpp, "http://127.0.0.1:8080", "x", 64);
        assert_eq!(url, "http://127.0.0.1:8080/completion");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["n_predict"], 64);
        assert_eq!(v["stream"], false);
    }

    #[test]
    fn mlx_request_shape() {
        let (url, body) = bench_request(RuntimeKind::MlxLm, "http://127.0.0.1:8080", "mlx-community/x-4bit", 32);
        assert_eq!(url, "http://127.0.0.1:8080/v1/chat/completions");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"], "mlx-community/x-4bit");
        assert_eq!(v["max_tokens"], 32);
        assert_eq!(v["stream"], false);
        assert_eq!(v["messages"][0]["role"], "user");
    }

    #[test]
    fn parse_ollama_eval_fields() {
        // 128 tokens in 3.2 s = 40 tok/s, exact from the server's own clock.
        let body = r#"{"model":"x","response":"...","done":true,"eval_count":128,"eval_duration":3200000000}"#;
        let p = parse_timing(RuntimeKind::Ollama, body).unwrap();
        assert_eq!(p.tokens, 128);
        assert!((p.server_tps.unwrap() - 40.0).abs() < 1e-9);
    }

    #[test]
    fn parse_ollama_without_duration_falls_back_to_wall_clock() {
        let body = r#"{"done":true,"eval_count":100}"#;
        let p = parse_timing(RuntimeKind::Ollama, body).unwrap();
        assert_eq!(p.tokens, 100);
        assert!(p.server_tps.is_none());
    }

    #[test]
    fn parse_ollama_missing_eval_count_names_the_field() {
        let err = parse_timing(RuntimeKind::Ollama, r#"{"done":true}"#).unwrap_err().to_string();
        assert!(err.contains("eval_count"), "{err}");
    }

    #[test]
    fn parse_llama_cpp_timings() {
        let body = r#"{"content":"...","tokens_predicted":128,"timings":{"prompt_n":12,"prompt_ms":80.0,"predicted_n":128,"predicted_ms":2560.0,"predicted_per_second":50.0}}"#;
        let p = parse_timing(RuntimeKind::LlamaCpp, body).unwrap();
        assert_eq!(p.tokens, 128);
        assert_eq!(p.server_tps, Some(50.0));
    }

    #[test]
    fn parse_llama_cpp_without_timings_uses_tokens_predicted() {
        let body = r#"{"content":"...","tokens_predicted":77}"#;
        let p = parse_timing(RuntimeKind::LlamaCpp, body).unwrap();
        assert_eq!(p.tokens, 77);
        assert!(p.server_tps.is_none());
    }

    #[test]
    fn parse_openai_usage() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"..."}}],"usage":{"prompt_tokens":12,"completion_tokens":128,"total_tokens":140}}"#;
        let p = parse_timing(RuntimeKind::MlxLm, body).unwrap();
        assert_eq!(p.tokens, 128);
        assert!(p.server_tps.is_none());
        let err = parse_timing(RuntimeKind::MlxLm, r#"{"choices":[]}"#).unwrap_err().to_string();
        assert!(err.contains("usage.completion_tokens"), "{err}");
    }

    #[test]
    fn parse_rejects_non_json() {
        assert!(parse_timing(RuntimeKind::Ollama, "<html>").is_err());
    }

    #[test]
    fn finalize_prefers_server_timings_else_wall_clock() {
        let s = finalize(ParsedTiming { tokens: 128, server_tps: Some(40.0) }, Duration::from_secs(9)).unwrap();
        assert_eq!(s.tps, 40.0);
        assert_eq!(s.timing, TimingSource::ServerTimings);
        let w = finalize(ParsedTiming { tokens: 128, server_tps: None }, Duration::from_secs(4)).unwrap();
        assert_eq!(w.tps, 32.0);
        assert_eq!(w.timing, TimingSource::WallClock);
        assert_eq!(w.tokens, 128);
    }

    #[test]
    fn finalize_rejects_zero_tokens_and_zero_wall() {
        assert!(finalize(ParsedTiming { tokens: 0, server_tps: None }, Duration::from_secs(1)).is_err());
        assert!(finalize(ParsedTiming { tokens: 10, server_tps: None }, Duration::ZERO).is_err());
    }

    #[test]
    fn measure_warms_up_then_times_one_request() {
        let mut probe = MockProbe::default();
        probe.posts.insert(
            "http://127.0.0.1:11434/api/generate".into(),
            r#"{"done":true,"eval_count":64,"eval_duration":1600000000}"#.into(),
        );
        let m = measure(&probe, RuntimeKind::Ollama, "http://127.0.0.1:11434", "llama3.2:3b", 64).unwrap();
        assert!((m.tps - 40.0).abs() < 1e-9);
        assert_eq!(m.timing, TimingSource::ServerTimings);
        let posts = probe.post_bodies.lock().unwrap();
        assert_eq!(posts.len(), 2, "one warm-up + one timed request");
        let warm: serde_json::Value = serde_json::from_str(&posts[0].1).unwrap();
        let timed: serde_json::Value = serde_json::from_str(&posts[1].1).unwrap();
        assert!(warm["options"]["num_predict"].as_u64().unwrap() < 64);
        assert_eq!(timed["options"]["num_predict"], 64);
    }

    #[test]
    fn measure_unreachable_server_is_a_clean_error() {
        let probe = MockProbe::default(); // no POST fixtures = connection refused
        let err = measure(&probe, RuntimeKind::LlamaCpp, "http://127.0.0.1:8080", "x", 16)
            .unwrap_err()
            .to_string();
        assert!(err.contains("127.0.0.1:8080"), "{err}");
    }
}
```

Add `pub mod bench;` to `lib.rs` (first, alphabetical).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p paddock-core bench::`
Expected: compile errors (`bench_request` not found).

- [ ] **Step 3: Implement the module**

Prepend to `bench.rs`:

```rust
//! `paddock bench`: measure the real generation tok/s of a running server.
//!
//! One warm-up request (so cold start does not pollute the timing), then one
//! timed generation of `tokens` tokens. Each runtime reports timings its own
//! way; when the server gives none, tok/s = tokens / wall time.

use std::time::{Duration, Instant};

use serde::Serialize;

use crate::PaddockError;
use crate::catalog::RuntimeKind;
use crate::hardware::SystemProbe;

/// Default `--tokens`: long enough to amortize per-request overhead, short
/// enough that the KV cache stays negligible (the calibration assumes KV ~ 0).
pub const DEFAULT_BENCH_TOKENS: u32 = 128;
const WARM_UP_TOKENS: u32 = 8;
/// Open-ended so the model does not stop early; ~20 prompt tokens.
const BENCH_PROMPT: &str =
    "Write a detailed, multi-paragraph history of the horse, from domestication to the present day.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingSource {
    /// The server measured its own generation speed (Ollama, llama.cpp).
    ServerTimings,
    /// tokens / wall-clock of the whole request (mlx-lm, or missing fields).
    WallClock,
}

impl TimingSource {
    pub fn label(&self) -> &'static str {
        match self {
            Self::ServerTimings => "server timings",
            Self::WallClock => "wall clock",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchMeasurement {
    pub tps: f64,
    pub tokens: u32,
    pub timing: TimingSource,
}

/// What a runtime's response told us: generated token count and, when the
/// server clocked it, the generation speed.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedTiming {
    pub tokens: u32,
    pub server_tps: Option<f64>,
}

/// (url, JSON body) of a non-streaming generation of `tokens` tokens.
pub fn bench_request(
    runtime: RuntimeKind,
    endpoint: &str,
    model_ref: &str,
    tokens: u32,
) -> (String, String) {
    let endpoint = endpoint.trim_end_matches('/');
    match runtime {
        RuntimeKind::Ollama => (
            format!("{endpoint}/api/generate"),
            serde_json::json!({
                "model": model_ref,
                "prompt": BENCH_PROMPT,
                "stream": false,
                "options": { "num_predict": tokens },
            })
            .to_string(),
        ),
        RuntimeKind::LlamaCpp => (
            format!("{endpoint}/completion"),
            serde_json::json!({
                "prompt": BENCH_PROMPT,
                "n_predict": tokens,
                "stream": false,
            })
            .to_string(),
        ),
        RuntimeKind::MlxLm => (
            format!("{endpoint}/v1/chat/completions"),
            serde_json::json!({
                "model": model_ref,
                "messages": [{ "role": "user", "content": BENCH_PROMPT }],
                "max_tokens": tokens,
                "stream": false,
            })
            .to_string(),
        ),
    }
}

fn missing(field: &str) -> PaddockError {
    PaddockError::Other(format!(
        "bench: server response has no `{field}` field; cannot count generated tokens"
    ))
}

/// Extract token count and server-side speed from a generation response.
pub fn parse_timing(runtime: RuntimeKind, body: &str) -> Result<ParsedTiming, PaddockError> {
    let v: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| PaddockError::Other(format!("bench: unparseable server response: {e}")))?;
    match runtime {
        RuntimeKind::Ollama => {
            let tokens = v["eval_count"].as_u64().ok_or_else(|| missing("eval_count"))? as u32;
            let server_tps = v["eval_duration"]
                .as_u64()
                .filter(|&ns| ns > 0)
                .map(|ns| tokens as f64 / (ns as f64 / 1e9));
            Ok(ParsedTiming { tokens, server_tps })
        }
        RuntimeKind::LlamaCpp => {
            let timings = &v["timings"];
            let tokens = timings["predicted_n"]
                .as_u64()
                .or_else(|| v["tokens_predicted"].as_u64())
                .ok_or_else(|| missing("tokens_predicted"))? as u32;
            let server_tps = timings["predicted_per_second"]
                .as_f64()
                .filter(|t| t.is_finite() && *t > 0.0);
            Ok(ParsedTiming { tokens, server_tps })
        }
        RuntimeKind::MlxLm => {
            let tokens = v["usage"]["completion_tokens"]
                .as_u64()
                .ok_or_else(|| missing("usage.completion_tokens"))? as u32;
            Ok(ParsedTiming { tokens, server_tps: None })
        }
    }
}

/// Server timings when present, else tokens / wall time.
pub fn finalize(parsed: ParsedTiming, wall: Duration) -> Result<BenchMeasurement, PaddockError> {
    if parsed.tokens == 0 {
        return Err(PaddockError::Other(
            "bench: server generated 0 tokens; is the model loaded?".into(),
        ));
    }
    if let Some(tps) = parsed.server_tps {
        return Ok(BenchMeasurement {
            tps,
            tokens: parsed.tokens,
            timing: TimingSource::ServerTimings,
        });
    }
    let secs = wall.as_secs_f64();
    if secs <= 0.0 {
        return Err(PaddockError::Other(
            "bench: zero wall time for the generation request".into(),
        ));
    }
    Ok(BenchMeasurement {
        tps: parsed.tokens as f64 / secs,
        tokens: parsed.tokens,
        timing: TimingSource::WallClock,
    })
}

fn unreachable(endpoint: &str) -> PaddockError {
    PaddockError::Other(format!(
        "bench: no answer from {endpoint}; the server died or refused the request (check `paddock ps` / `paddock logs`)"
    ))
}

/// Warm up, then time one generation of `tokens` tokens against a running server.
pub fn measure(
    probe: &dyn SystemProbe,
    runtime: RuntimeKind,
    endpoint: &str,
    model_ref: &str,
    tokens: u32,
) -> Result<BenchMeasurement, PaddockError> {
    let (url, warm_body) = bench_request(runtime, endpoint, model_ref, WARM_UP_TOKENS);
    probe
        .http_post_local(&url, &warm_body)
        .ok_or_else(|| unreachable(endpoint))?;

    let (url, body) = bench_request(runtime, endpoint, model_ref, tokens);
    let start = Instant::now();
    let response = probe
        .http_post_local(&url, &body)
        .ok_or_else(|| unreachable(endpoint))?;
    let wall = start.elapsed();
    finalize(parse_timing(runtime, &response)?, wall)
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p paddock-core bench::`
Expected: 14 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/paddock-core/src/bench.rs crates/paddock-core/src/lib.rs
git commit -m "feat(bench): measure generation tok/s of a running server"
```

---

## Task 4: Resolve a server `model_ref` to a catalog variant

**Files:**
- Modify: `crates/paddock-core/src/bench.rs` (add `resolve_model_ref` + tests)

**Interfaces:**
- Consumes: `catalog::CatalogModel` (`name`, `repo`, `variants[].quant`, `variants[].source_tag`, `to_model_variant`), `score::variants_by_quality`.
- Produces: `pub fn resolve_model_ref(models: &[CatalogModel], model_ref: &str) -> Option<(usize, usize)>` returning `(model index, variant index)`; `None` = measure-only.

Ref shapes, from `runtime.rs`: llama-server / Ollama-HF `hf.co/{org}/{repo}:{quant}` or `{org}/{repo}:{quant}`; mlx `{org}/{repo}` (no quant); Ollama `{base}:{source_tag}` or the curated full name (`llama3.2:3b`, `qwen3-coder:30b`).

- [ ] **Step 1: Write the failing tests**

Add a second tests module to `bench.rs`:

```rust
#[cfg(test)]
mod resolve_tests {
    use super::*;
    use crate::catalog::{CatalogModel, CatalogVariant, RuntimeKind, Source};

    fn variant(quant: &str, tag: Option<&str>) -> CatalogVariant {
        CatalogVariant {
            quant: quant.into(),
            bpw: crate::catalog::quant_bpw(quant).unwrap_or(4.5),
            file_size_bytes: None,
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
            embedding_dim: 4096,
            runtime_compat: vec![RuntimeKind::Ollama, RuntimeKind::LlamaCpp],
            source_tag: tag.map(str::to_string),
        }
    }

    fn model(name: &str, source: Source, repo: Option<&str>, variants: Vec<CatalogVariant>) -> CatalogModel {
        CatalogModel {
            id: 0,
            name: name.into(),
            family: None,
            source,
            repo: repo.map(str::to_string),
            params_total: 8_000_000_000,
            params_active: 8_000_000_000,
            architecture: None,
            context_max: 32_768,
            released_at: None,
            released_approx: false,
            variants,
        }
    }

    fn catalog() -> Vec<CatalogModel> {
        vec![
            model(
                "llama3.1:8b",
                Source::Ollama,
                None,
                vec![
                    variant("Q4_K_M", Some("8b-instruct-q4_K_M")),
                    variant("Q8_0", Some("8b-instruct-q8_0")),
                ],
            ),
            model("qwen3-coder:30b", Source::Ollama, None, vec![variant("Q4_K_M", None)]),
            model(
                "Qwen3.6-35B-A3B-GGUF",
                Source::HuggingFace,
                Some("unsloth/Qwen3.6-35B-A3B-GGUF"),
                vec![variant("Q4_K_M", None), variant("UD-Q4_K_XL", None), variant("Q8_0", None)],
            ),
            model(
                "Llama-3.1-8B-Instruct-4bit",
                Source::Mlx,
                Some("mlx-community/Llama-3.1-8B-Instruct-4bit"),
                vec![variant("MLX_4BIT", None)],
            ),
        ]
    }

    #[test]
    fn hf_ref_with_quant_matches_repo_and_quant_case_insensitively() {
        let c = catalog();
        assert_eq!(resolve_model_ref(&c, "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:ud-q4_k_xl"), Some((2, 1)));
        assert_eq!(resolve_model_ref(&c, "unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0"), Some((2, 2)));
    }

    #[test]
    fn hf_ref_with_unknown_quant_is_unresolved() {
        assert_eq!(resolve_model_ref(&catalog(), "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:IQ2_XXS"), None);
    }

    #[test]
    fn mlx_repo_ref_picks_its_single_variant() {
        assert_eq!(resolve_model_ref(&catalog(), "mlx-community/Llama-3.1-8B-Instruct-4bit"), Some((3, 0)));
    }

    #[test]
    fn ollama_source_tag_matches_exact_variant() {
        assert_eq!(resolve_model_ref(&catalog(), "llama3.1:8b-instruct-q8_0"), Some((0, 1)));
    }

    #[test]
    fn ollama_full_curated_name_picks_best_quality_variant() {
        assert_eq!(resolve_model_ref(&catalog(), "qwen3-coder:30b"), Some((1, 0)));
        // Curated name with two variants and no tag hit: Q8_0 is the higher quality.
        assert_eq!(resolve_model_ref(&catalog(), "llama3.1:8b"), Some((0, 1)));
    }

    #[test]
    fn ollama_base_name_with_unknown_tag_matches_quant_substring_else_best() {
        let c = catalog();
        assert_eq!(resolve_model_ref(&c, "llama3.1:latest-q4_K_M"), Some((0, 0)));
        assert_eq!(resolve_model_ref(&c, "llama3.1:latest"), Some((0, 1)));
    }

    #[test]
    fn unknown_ref_is_unresolved() {
        assert_eq!(resolve_model_ref(&catalog(), "definitely-not-a-model:1b"), None);
        assert_eq!(resolve_model_ref(&catalog(), "hf.co/nobody/nothing:Q4_K_M"), None);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p paddock-core resolve_tests`
Expected: compile error, `cannot find function resolve_model_ref`.

- [ ] **Step 3: Implement**

Add to `bench.rs` (above the tests modules), with `use crate::catalog::CatalogModel;` added to the imports:

```rust
/// Map a running server's `model_ref` back to a catalog (model, variant) pair,
/// so the bench can read `params_active` / `bpw`. Indexes are into `models`
/// and `models[i].variants`. None = measure-only (no calibration update).
///
/// Shapes (see `runtime.rs`): `hf.co/{org}/{repo}:{quant}` and
/// `{org}/{repo}[:{quant}]` for HF / MLX repos; `{base}:{tag}` for Ollama,
/// where `tag` is an exact `source_tag`, part of a curated full name, or a
/// library tag that merely contains a quant label.
pub fn resolve_model_ref(models: &[CatalogModel], model_ref: &str) -> Option<(usize, usize)> {
    let stripped = model_ref.strip_prefix("hf.co/").unwrap_or(model_ref);

    if stripped.contains('/') {
        let (repo, quant) = match stripped.rsplit_once(':') {
            Some((r, q)) => (r, Some(q)),
            None => (stripped, None),
        };
        let mi = models.iter().position(|m| {
            m.repo.as_deref().is_some_and(|r| r.eq_ignore_ascii_case(repo))
        })?;
        let vi = match quant {
            Some(q) => models[mi].variants.iter().position(|v| v.quant.eq_ignore_ascii_case(q))?,
            None => best_quality_idx(&models[mi])?,
        };
        return Some((mi, vi));
    }

    let (base, tag) = match stripped.split_once(':') {
        Some((b, t)) => (b, Some(t)),
        None => (stripped, None),
    };
    let base_of = |m: &CatalogModel| m.name.split(':').next().unwrap_or(&m.name).to_string();

    // 1. Exact `source_tag` on a model with the same base name.
    if let Some(tag) = tag {
        for (mi, m) in models.iter().enumerate() {
            if !base_of(m).eq_ignore_ascii_case(base) {
                continue;
            }
            if let Some(vi) = m.variants.iter().position(|v| {
                v.source_tag.as_deref().is_some_and(|t| t.eq_ignore_ascii_case(tag))
            }) {
                return Some((mi, vi));
            }
        }
    }
    // 2. Curated full name (`llama3.2:3b` carries its tag in the name).
    if let Some(mi) = models.iter().position(|m| m.name.eq_ignore_ascii_case(stripped)) {
        return Some((mi, best_quality_idx(&models[mi])?));
    }
    // 3. Base name; variant whose quant label appears in the tag, else best quality.
    let mi = models.iter().position(|m| base_of(m).eq_ignore_ascii_case(base))?;
    let m = &models[mi];
    let vi = tag
        .and_then(|t| {
            let t = t.to_lowercase();
            m.variants.iter().position(|v| t.contains(&v.quant.to_lowercase()))
        })
        .or_else(|| best_quality_idx(m))?;
    Some((mi, vi))
}

/// Index of the highest-quality variant (first in `variants_by_quality`).
fn best_quality_idx(m: &CatalogModel) -> Option<usize> {
    let mvs: Vec<_> = m.variants.iter().map(|v| m.to_model_variant(v)).collect();
    crate::score::variants_by_quality(&mvs).first().copied()
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p paddock-core bench`
Expected: all bench and resolve tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/paddock-core/src/bench.rs
git commit -m "feat(bench): resolve a server model_ref to a catalog variant"
```

---

## Task 5: Estimates consume the calibration (App, scoring, TUI)

**Files:**
- Modify: `crates/paddock/src/app.rs` (`App` struct + `load` + `scored_models` line 93)
- Modify: `crates/paddock/src/tui/state.rs` (`TuiState` fields + `new`, and every `TuiState::new` call in its tests)
- Modify: `crates/paddock/src/tui/mod.rs` (`TuiState::new` call ~line 30)
- Modify: `crates/paddock/src/tui/draw.rs` (`draw_detail`, `draw_speed_chart`, `detail_lines`)

**Interfaces:**
- Consumes: `paddock_core::calibration::{default_calibration_path, load}`, `paddock_core::estimate::{SpeedCalibration, estimate_speed_calibrated}`.
- Produces: `App.calibration: SpeedCalibration`; `TuiState.calibration: SpeedCalibration`; `TuiState::new(rows, use_case, runtimes, budget, calibration)`.

- [ ] **Step 1: Write the failing test**

Add to the tests module of `crates/paddock/src/tui/state.rs` (find the existing helper that builds a `TuiState` in that module; the new field is the fifth `new` argument):

```rust
    #[test]
    fn tui_state_carries_calibration() {
        use paddock_core::estimate::SpeedCalibration;
        let cal = SpeedCalibration { dense: 0.55, moe: 0.45 };
        let s = TuiState::new(
            Vec::new(),
            UseCase::General,
            RuntimesStatus::default(),
            MemoryBudget { gpu_effective_bytes: 1, ram_total_bytes: 2 },
            cal,
        );
        assert_eq!(s.calibration, cal);
    }
```

If `RuntimesStatus` does not implement `Default`, build it the way the neighbouring state tests do (copy their helper).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p paddock tui_state_carries_calibration`
Expected: compile error, `new` takes 4 arguments / no field `calibration`.

- [ ] **Step 3: App loads the calibration and scores with it**

In `app.rs`:

```rust
use paddock_core::estimate::{
    DEFAULT_CONTEXT, FitVerdict, MemoryBudget, MemoryEstimate, SpeedCalibration, SpeedEstimate,
    estimate_memory, estimate_speed_calibrated,
};
```

```rust
pub struct App {
    pub profile: HardwareProfile,
    pub budget: MemoryBudget,
    /// Per-machine efficiency factors from `paddock bench` (defaults when
    /// no bench has run). Loaded once; every speed estimate uses it.
    pub calibration: SpeedCalibration,
}

impl App {
    pub fn load() -> Self {
        let profile = scan(&RealSystemProbe);
        let budget = MemoryBudget {
            gpu_effective_bytes: profile.gpu.effective_limit_bytes,
            ram_total_bytes: profile.ram_total_bytes,
        };
        let calibration =
            paddock_core::calibration::load(&paddock_core::calibration::default_calibration_path())
                .to_speed_calibration();
        Self {
            profile,
            budget,
            calibration,
        }
    }
```

and in `scored_models`:

```rust
            let speed = estimate_speed_calibrated(
                mv,
                self.profile.bandwidth_gbps,
                memory.kv_cache_bytes,
                &self.calibration,
            );
```

- [ ] **Step 4: Thread it through the TUI**

`state.rs`: add the field and the `new` parameter.

```rust
    /// Per-machine speed calibration used by the detail table and speed chart.
    pub calibration: SpeedCalibration,
```

```rust
    pub fn new(
        rows: Vec<ScoredModel>,
        use_case: UseCase,
        runtimes: RuntimesStatus,
        budget: MemoryBudget,
        calibration: SpeedCalibration,
    ) -> Self {
        Self {
            // ... existing fields unchanged ...
            calibration,
        }
    }
```

Import `SpeedCalibration` from `paddock_core::estimate` in `state.rs`. Update every existing `TuiState::new(` call in the `state.rs` tests to pass `SpeedCalibration::default()` as the fifth argument (grep: `rg "TuiState::new\(" crates/paddock/src`).

`tui/mod.rs` ~line 30: pass `app.calibration` as the fifth argument.

`draw.rs`: replace the `estimate_speed` import with `estimate_speed_calibrated`:

```rust
use paddock_core::estimate::{FitVerdict, SpeedCalibration, estimate_speed_calibrated, kv_cache_bytes};
```

In `draw_detail`, pass `&state.calibration` to both helpers:

```rust
    let lines = detail_lines(
        r,
        sel,
        &state.budget,
        profile.bandwidth_gbps,
        &state.calibration,
        state.detail_plan.as_ref(),
        state.detail_serve_plan.as_ref(),
    );
    // ...
    draw_speed_chart(
        frame,
        chart_area,
        &r.model.to_model_variant(&r.model.variants[sel]),
        r.model.context_max,
        profile.bandwidth_gbps,
        &state.calibration,
    );
```

`draw_speed_chart` gains `cal: &SpeedCalibration` after `bandwidth_gbps` and calls:

```rust
            let tps = estimate_speed_calibrated(v, bandwidth_gbps, kv_cache_bytes(v, ctx as u32), cal)
                .generation_tps;
```

`detail_lines` gains `cal: &SpeedCalibration` after `bandwidth_gbps` and calls:

```rust
        let tps = estimate_speed_calibrated(&v, bandwidth_gbps, kv_cache_bytes(&v, DEFAULT_CONTEXT), cal)
            .generation_tps;
```

- [ ] **Step 5: Build and run the whole workspace tests**

Run: `cargo test --workspace`
Expected: everything compiles; all tests pass (the state test included). `cargo clippy --workspace --all-targets` clean.

- [ ] **Step 6: Commit**

```bash
git add crates/paddock/src/app.rs crates/paddock/src/tui
git commit -m "feat(app): load calibration.json and estimate with it (scoring, TUI)"
```

---

## Task 6: `paddock bench` CLI

**Files:**
- Modify: `crates/paddock-core/src/serving.rs` (add `ServerRowMatch`, `match_server_rows` + tests, after `list_all_servers`)
- Modify: `crates/paddock/src/cli.rs` (new `Bench` variant)
- Modify: `crates/paddock/src/output.rs` (`BenchReport`, `print_bench`)
- Modify: `crates/paddock/src/main.rs` (dispatch + `bench_server`)
- Modify: `crates/paddock/tests/cli.rs` (no-match smoke test)

**Interfaces:**
- Consumes: `bench::{measure, resolve_model_ref, DEFAULT_BENCH_TOKENS, TimingSource}`, `calibration::{default_calibration_path, load, save, efficiency_from_measurement, validate_efficiency, CalibrationEntry, ModelClass}`, `estimate::estimate_speed_calibrated`, `serving::list_all_servers`.
- Produces: `pub enum ServerRowMatch<'a> { Matched(&'a ServerRow), Ambiguous(Vec<&'a ServerRow>), NotFound }`; `pub fn match_server_rows<'a>(rows: &'a [ServerRow], target: Option<&str>) -> ServerRowMatch<'a>`; `output::BenchReport` (Serialize) and `output::print_bench(&BenchReport)`.

- [ ] **Step 1: Write the failing test for `match_server_rows`**

Add to `serving.rs` a new tests module:

```rust
#[cfg(test)]
mod server_row_match_tests {
    use super::*;

    fn row(model: &str, stop: StopHandle) -> ServerRow {
        ServerRow {
            model: model.into(),
            runtime: RuntimeKind::LlamaCpp,
            endpoint: "http://127.0.0.1:8080".into(),
            openai_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
            ctx: None,
            started_at: None,
            stop,
        }
    }

    #[test]
    fn no_target_means_the_single_server() {
        let rows = vec![row("qwen3-8b", StopHandle::Pid(1))];
        assert!(matches!(match_server_rows(&rows, None), ServerRowMatch::Matched(r) if r.model == "qwen3-8b"));
        assert!(matches!(match_server_rows(&[], None), ServerRowMatch::NotFound));
        let two = vec![row("a", StopHandle::Pid(1)), row("b", StopHandle::Pid(2))];
        assert!(matches!(match_server_rows(&two, None), ServerRowMatch::Ambiguous(v) if v.len() == 2));
    }

    #[test]
    fn pid_and_substring_targets() {
        let rows = vec![
            row("hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M", StopHandle::Pid(51234)),
            row("llama3.2:3b", StopHandle::OllamaModel("llama3.2:3b".into())),
            row("llama3.1:8b", StopHandle::OllamaModel("llama3.1:8b".into())),
        ];
        assert!(matches!(match_server_rows(&rows, Some("51234")), ServerRowMatch::Matched(r) if r.model.starts_with("hf.co")));
        assert!(matches!(match_server_rows(&rows, Some("qwen3.6")), ServerRowMatch::Matched(r) if r.model.contains("Qwen3.6")));
        assert!(matches!(match_server_rows(&rows, Some("llama3")), ServerRowMatch::Ambiguous(v) if v.len() == 2));
        assert!(matches!(match_server_rows(&rows, Some("nothing")), ServerRowMatch::NotFound));
        assert!(matches!(match_server_rows(&rows, Some("99")), ServerRowMatch::NotFound));
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p paddock-core server_row_match`
Expected: compile error, `match_server_rows` not found.

- [ ] **Step 3: Implement `match_server_rows`**

Add to `serving.rs` after `list_all_servers`:

```rust
/// Result of resolving a `bench` target against unified server rows.
pub enum ServerRowMatch<'a> {
    Matched(&'a ServerRow),
    /// Several rows hit (or several servers run and no target was given).
    Ambiguous(Vec<&'a ServerRow>),
    NotFound,
}

/// Resolve a `bench` target over `list_all_servers` rows: no target = the
/// single running server; all-digits = pid of a paddock-spawned server;
/// otherwise a case-insensitive substring of the model name.
pub fn match_server_rows<'a>(rows: &'a [ServerRow], target: Option<&str>) -> ServerRowMatch<'a> {
    let Some(target) = target else {
        return match rows {
            [] => ServerRowMatch::NotFound,
            [one] => ServerRowMatch::Matched(one),
            many => ServerRowMatch::Ambiguous(many.iter().collect()),
        };
    };
    if let Ok(pid) = target.parse::<u32>() {
        return match rows.iter().find(|r| r.stop == StopHandle::Pid(pid)) {
            Some(r) => ServerRowMatch::Matched(r),
            None => ServerRowMatch::NotFound,
        };
    }
    let needle = target.to_lowercase();
    let hits: Vec<&ServerRow> = rows
        .iter()
        .filter(|r| r.model.to_lowercase().contains(&needle))
        .collect();
    match hits.as_slice() {
        [] => ServerRowMatch::NotFound,
        [one] => ServerRowMatch::Matched(one),
        _ => ServerRowMatch::Ambiguous(hits),
    }
}
```

Run: `cargo test -p paddock-core server_row_match` - expected PASS.

- [ ] **Step 4: Add the subcommand**

`cli.rs`, after `Logs { .. }`:

```rust
    /// Measure a running server's real tok/s and calibrate speed estimates
    Bench {
        /// Target: model name substring or a pid; omit when one server runs
        target: Option<String>,
        /// Tokens to generate for the timed run
        #[arg(long, default_value_t = paddock_core::bench::DEFAULT_BENCH_TOKENS)]
        tokens: u32,
    },
```

`output.rs`, at the end:

```rust
/// Everything `paddock bench` learned, for the text and `--json` printers.
#[derive(Debug, serde::Serialize)]
pub struct BenchReport {
    pub model_ref: String,
    pub runtime: paddock_core::catalog::RuntimeKind,
    pub measured_tps: f64,
    pub tokens: u32,
    pub timing: paddock_core::bench::TimingSource,
    /// Catalog model name + quant + class when the ref resolved.
    pub model: Option<String>,
    pub quant: Option<String>,
    pub class: Option<paddock_core::calibration::ModelClass>,
    /// Estimate at KV ~ 0 with the calibration in force before this run.
    pub estimated_tps: Option<f64>,
    pub efficiency: Option<f64>,
    pub previous_efficiency: Option<f64>,
    pub calibration_updated: bool,
    /// Why the calibration was not updated (unresolved ref, implausible value).
    pub reason: Option<String>,
}

pub fn print_bench(r: &BenchReport) {
    match (&r.model, &r.quant, r.class) {
        (Some(m), Some(q), Some(c)) => println!("model      {m} {q} ({})", c.label()),
        _ => println!("model      {} (not in catalog)", r.model_ref),
    }
    println!(
        "measured   {:.1} tok/s ({} tokens, {})",
        r.measured_tps,
        r.tokens,
        r.timing.label()
    );
    if let Some(est) = r.estimated_tps {
        println!("estimated  {est:.1} tok/s (before calibration)");
    }
    if let Some(eff) = r.efficiency {
        println!("efficiency {eff:.2}");
    }
    match (&r.reason, r.class, r.previous_efficiency, r.efficiency) {
        (Some(reason), ..) => println!("calibration not updated: {reason}"),
        (None, Some(c), Some(prev), Some(eff)) if r.calibration_updated => {
            println!("calibration updated: {} {prev:.2} -> {eff:.2}", c.label())
        }
        _ => {}
    }
}
```

`main.rs`: dispatch after `Logs`:

```rust
        Some(Command::Bench { target, tokens }) => bench_server(&app, target.as_deref(), tokens, cli.json)?,
```

and the handler, placed after `show_logs`:

```rust
/// `paddock bench`: time one generation against a running server, derive this
/// machine's efficiency for the model's class, persist it.
fn bench_server(app: &App, target: Option<&str>, tokens: u32, json: bool) -> Result<()> {
    use paddock_core::bench::{measure, resolve_model_ref};
    use paddock_core::calibration::{
        self, CalibrationEntry, ModelClass, default_calibration_path, efficiency_from_measurement,
        validate_efficiency,
    };
    use paddock_core::estimate::estimate_speed_calibrated;
    use paddock_core::serving::{ServerRowMatch, list_all_servers, match_server_rows};

    if tokens == 0 {
        bail!("--tokens must be at least 1");
    }
    let probe = RealSystemProbe;
    let rows = list_all_servers(&Registry::open_default(), &probe);
    let row = match match_server_rows(&rows, target) {
        ServerRowMatch::Matched(r) => r,
        ServerRowMatch::Ambiguous(cands) => {
            match target {
                None => eprintln!("several servers are running - pick one with `paddock bench <target>`:"),
                Some(t) => eprintln!("`{t}` matches several servers - be specific:"),
            }
            for r in cands {
                eprintln!("  {} ({})", r.model, r.endpoint);
            }
            std::process::exit(1);
        }
        ServerRowMatch::NotFound => {
            match target {
                None => eprintln!("nothing to bench - serve a model first"),
                Some(t) => {
                    eprintln!("no running server matches `{t}`");
                    if !rows.is_empty() {
                        eprintln!(
                            "running: {}",
                            rows.iter().map(|r| r.model.as_str()).collect::<Vec<_>>().join(", ")
                        );
                    }
                }
            }
            std::process::exit(1);
        }
    };

    if !json {
        eprintln!("benching {} on {} ({tokens} tokens)...", row.model, row.endpoint);
    }
    let measured = measure(&probe, row.runtime, &row.endpoint, &row.model, tokens)?;

    let db = app.open_db()?;
    let models = db.list_models().context("reading catalog")?;
    let resolved = resolve_model_ref(&models, &row.model)
        .map(|(mi, vi)| (&models[mi], models[mi].to_model_variant(&models[mi].variants[vi])));

    let mut report = output::BenchReport {
        model_ref: row.model.clone(),
        runtime: row.runtime,
        measured_tps: measured.tps,
        tokens: measured.tokens,
        timing: measured.timing,
        model: None,
        quant: None,
        class: None,
        estimated_tps: None,
        efficiency: None,
        previous_efficiency: None,
        calibration_updated: false,
        reason: None,
    };
    let mut rejected = false;

    match resolved {
        None => {
            report.reason = Some(format!(
                "`{}` not found in catalog (run `paddock sync`); measured only",
                row.model
            ));
        }
        Some((model, mv)) => {
            let class = ModelClass::of(&mv);
            let previous = match class {
                ModelClass::Dense => app.calibration.dense,
                ModelClass::Moe => app.calibration.moe,
            };
            report.model = Some(model.name.clone());
            report.quant = Some(mv.quant.clone());
            report.class = Some(class);
            report.previous_efficiency = Some(previous);
            // KV ~ 0: the same near-empty-context condition the bench runs under.
            report.estimated_tps = Some(
                estimate_speed_calibrated(&mv, app.profile.bandwidth_gbps, 0, &app.calibration)
                    .generation_tps,
            );
            let eff = efficiency_from_measurement(
                measured.tps,
                mv.params_active,
                mv.bpw,
                app.profile.bandwidth_gbps,
            );
            report.efficiency = Some(eff);
            match validate_efficiency(eff) {
                Err(e) => {
                    rejected = true;
                    report.reason = Some(e.to_string());
                }
                Ok(eff) => {
                    let path = default_calibration_path();
                    let mut file = calibration::load(&path);
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);
                    file.set(
                        class,
                        CalibrationEntry {
                            efficiency: eff,
                            model: format!("{} {}", model.name, mv.quant),
                            measured_at: now,
                        },
                    );
                    calibration::save(&path, &file).context("writing calibration.json")?;
                    report.calibration_updated = true;
                }
            }
        }
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        output::print_bench(&report);
    }
    if rejected {
        std::process::exit(1);
    }
    Ok(())
}
```

- [ ] **Step 5: CLI smoke test**

In `crates/paddock/tests/cli.rs`, extend `paddock()` so a test never touches the real calibration file:

```rust
    c.env("PADDOCK_CALIBRATION_PATH", dir.path().join("calibration.json"));
```

and add:

```rust
#[test]
fn bench_unknown_target_fails_cleanly() {
    let (mut cmd, _dir) = paddock();
    cmd.args(["bench", "definitely-not-running-anywhere"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("no running server matches"));
}

#[test]
fn bench_rejects_zero_tokens() {
    let (mut cmd, _dir) = paddock();
    cmd.args(["bench", "--tokens", "0", "x"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--tokens must be at least 1"));
}
```

- [ ] **Step 6: Build, test, lint**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all -- --check`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add crates/paddock-core/src/serving.rs crates/paddock/src/cli.rs crates/paddock/src/output.rs crates/paddock/src/main.rs crates/paddock/tests/cli.rs
git commit -m "feat(cli): paddock bench measures tok/s and calibrates estimates"
```

---

## Task 7: Docs and live smoke

**Files:**
- Modify: `README.md` (subcommand count line "Nine subcommands", new `### paddock bench` section after `paddock ps/stop/logs`, the "Speed" paragraph in "How the estimates work", the Roadmap bullet)

- [ ] **Step 1: README**

Change `Nine subcommands cover everything scriptable:` to `Ten subcommands cover everything scriptable:`.

Insert after the `paddock logs` paragraph (before the `mmproj` paragraph):

````markdown
### `paddock bench`: measure, then trust the numbers

The speed column is an estimate. `paddock bench` replaces the guesswork with a measurement of *your* machine: it times one short generation against a server that is already running (llama.cpp, mlx-lm, or a model loaded in Ollama), derives the bandwidth efficiency the kernels actually reach on this Mac, and stores it so every future estimate uses it.

```text
$ paddock serve Qwen3.6-35B-A3B-GGUF
...
$ paddock bench
model      Qwen3.6-35B-A3B-GGUF Q4_K_M (moe)
measured   38.2 tok/s (128 tokens, server timings)
estimated  16.5 tok/s (before calibration)
efficiency 0.45
calibration updated: moe 0.30 -> 0.45
```

With one server running no target is needed; otherwise pass a model-name substring or a pid, like `stop` and `logs`. `--tokens N` changes the length of the timed run (default 128). Ollama and llama.cpp report their own generation timings, so the number is exact; mlx-lm has none, so paddock falls back to tokens over wall time and says so.

Calibration is per machine and per model class: one factor for dense models, one for MoE, last measurement wins, stored in `~/Library/Application Support/paddock/calibration.json`. Delete the file to go back to the defaults. If the served model cannot be matched to a catalog entry, paddock prints the measured speed and leaves the calibration alone; if the implied efficiency is implausible (outside 0.05 to 1.5, which means the model on the wire is not the one in the catalog), it refuses to write it.
````

In "How the estimates work" > "Speed", replace the closing sentence `A real `paddock bench` module that measures *your* machine and recalibrates the factors per-device is on the roadmap.` with:

```markdown
`paddock bench` measures your machine and replaces the efficiency factors with the values it observes, per model class; after one bench the TOK/S column reflects this Mac rather than the community average.
```

In the Roadmap, replace the `paddock bench` bullet with:

```markdown
- **`paddock bench` v2**: bench from the TUI, prefill (prompt-speed) calibration, offload-penalty modeling
```

- [ ] **Step 2: Grep for em-dashes in everything touched**

Run: `rg -nP '\x{2014}' README.md crates/paddock-core/src/bench.rs crates/paddock-core/src/calibration.rs crates/paddock/src/main.rs crates/paddock/src/output.rs docs/superpowers/plans/2026-09-05-paddock-bench.md`
Expected: no output.

- [ ] **Step 3: Live smoke (needs a running server on this Mac)**

```bash
cargo run -p paddock -- ps
cargo run -p paddock -- bench --json
cat ~/Library/Application\ Support/paddock/calibration.json
cargo run -p paddock -- fit -n 5
```

Expected: `bench` prints a measurement, `calibration.json` holds the class entry, and the `fit` TOK/S of same-class models moves toward the measured value. If no server is running, the smoke reports `nothing to bench - serve a model first` and the step is recorded as skipped in the commit message.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: paddock bench usage and calibrated speed estimates"
```

---

## Self-review

- **Spec coverage.** Measurement per runtime + warm-up + wall-clock fallback: Task 3. Calibration math, storage, clamp, corrupt-file defaults: Task 2. `SpeedCalibration`, wrapper, calibrated variant, App/score/TUI consumption: Tasks 1 and 5. Variant resolution for the three ref shapes and the measure-only path: Tasks 4 and 6. CLI with target matching over `list_all_servers`, text and `--json` output, all error paths: Task 6. Tests listed in the spec map to Tasks 1-6; live smoke to Task 7.
- **Placeholders.** None; every step carries its code.
- **Type consistency.** `SpeedCalibration { dense, moe }` and `for_variant` (Task 1) are what Task 2's `to_speed_calibration`, Task 5's `App`/`TuiState` and Task 6's `bench_server` use. `ModelClass::{of, label}` (Task 2) is used by Task 6 and `output.rs`. `BenchMeasurement { tps, tokens, timing }` and `TimingSource::label` (Task 3) feed `BenchReport` (Task 6). `resolve_model_ref -> Option<(usize, usize)>` (Task 4) is consumed as `(mi, vi)` in Task 6. `match_server_rows(rows, Option<&str>) -> ServerRowMatch` (Task 6) matches its own call site. `TuiState::new` gains exactly one trailing `SpeedCalibration` argument everywhere (Task 5).
