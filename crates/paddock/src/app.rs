//! Shared loading path: hardware profile + catalog + scored rows.
//! Used by both the CLI subcommands and (Task 7) the TUI.

use anyhow::{Context, Result};
use paddock_core::catalog::CatalogModel;
use paddock_core::catalog::db::{Db, default_db_path};
use paddock_core::estimate::{
    DEFAULT_CONTEXT, FitVerdict, MemoryBudget, MemoryEstimate, SpeedCalibration, SpeedEstimate,
    estimate_memory, estimate_speed_calibrated,
};
use paddock_core::hardware::{HardwareProfile, RealSystemProbe, scan};
use paddock_core::score::{Score, UseCase, best_variant, score_variant};

#[derive(Clone)]
pub struct ScoredModel {
    pub model: CatalogModel,
    /// Index into model.variants of the picked "best quant that fits".
    pub variant_idx: usize,
    pub memory: MemoryEstimate,
    pub speed: SpeedEstimate,
    pub score: Score,
}

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
        let calibration = paddock_core::calibration::load(
            &paddock_core::calibration::default_calibration_path(),
        )
        .to_speed_calibration();
        Self {
            profile,
            budget,
            calibration,
        }
    }

    pub fn open_db(&self) -> Result<Db> {
        Db::open(default_db_path()).context("opening catalog database")
    }

    /// All models scored for a use case, sorted by descending score.
    /// `include_unfit` keeps DoesNotFit rows (verdict still shown).
    pub fn scored_models(
        &self,
        db: &Db,
        use_case: UseCase,
        include_unfit: bool,
    ) -> Result<Vec<ScoredModel>> {
        let models = db.list_models().context("reading catalog")?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let mut rows = Vec::new();
        for model in models {
            let mvs: Vec<_> = model
                .variants
                .iter()
                .map(|v| model.to_model_variant(v))
                .collect();
            // Pick the best fitting variant; when nothing fits at all, fall
            // back to the smallest variant so --all can still show the model.
            // Pointer identity, not quant-label equality: two variants can
            // share the same quant string, and `bv` borrows from `mvs`.
            let picked = best_variant(&mvs, &self.budget)
                .map(|bv| {
                    mvs.iter()
                        .position(|v| std::ptr::eq(v, bv))
                        .expect("best_variant borrows from mvs")
                })
                .or_else(|| {
                    if include_unfit {
                        mvs.iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| {
                                a.bpw
                                    .partial_cmp(&b.bpw)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                    } else {
                        None
                    }
                });
            let Some(variant_idx) = picked else { continue };
            let mv = &mvs[variant_idx];
            let memory = estimate_memory(mv, DEFAULT_CONTEXT, &self.budget);
            if !include_unfit && memory.verdict == FitVerdict::DoesNotFit {
                continue;
            }
            let speed = estimate_speed_calibrated(
                mv,
                self.profile.bandwidth_gbps,
                memory.kv_cache_bytes,
                &self.calibration,
            );
            let age_days = model.released_at.map(|r| (now - r) as f64 / 86_400.0);
            let score = score_variant(mv, &memory, &speed, use_case, age_days);
            rows.push(ScoredModel {
                model,
                variant_idx,
                memory,
                speed,
                score,
            });
        }
        // total_cmp: deterministic descending order even if a score is NaN.
        rows.sort_by(|a, b| b.score.total.total_cmp(&a.score.total));
        Ok(rows)
    }
}
