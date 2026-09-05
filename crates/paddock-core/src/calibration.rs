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
        assert_eq!(
            efficiency_from_measurement(30.0, 1_000_000_000, f64::NAN, 400.0),
            0.0
        );
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
