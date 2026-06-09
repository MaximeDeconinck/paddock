//! Shared loading path: hardware profile + catalog + scored rows.
//! Used by both the CLI subcommands and (Task 7) the TUI.

use anyhow::{Context, Result};
use tetro_core::catalog::db::{default_db_path, Db};
use tetro_core::catalog::CatalogModel;
use tetro_core::estimate::{
    estimate_memory, estimate_speed, FitVerdict, MemoryBudget, MemoryEstimate, SpeedEstimate,
    DEFAULT_CONTEXT,
};
use tetro_core::hardware::{scan, HardwareProfile, RealSystemProbe};
use tetro_core::score::{best_variant, score_variant, Score, UseCase};

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
}

impl App {
    pub fn load() -> Self {
        let profile = scan(&RealSystemProbe);
        let budget = MemoryBudget {
            gpu_effective_bytes: profile.gpu.effective_limit_bytes,
            ram_total_bytes: profile.ram_total_bytes,
        };
        Self { profile, budget }
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
        let mut rows = Vec::new();
        for model in models {
            let mvs: Vec<_> = model
                .variants
                .iter()
                .map(|v| model.to_model_variant(v))
                .collect();
            // Pick the best fitting variant; when nothing fits at all, fall
            // back to the smallest variant so --all can still show the model.
            let picked = best_variant(&mvs, &self.budget)
                .map(|bv| mvs.iter().position(|v| v.quant == bv.quant).unwrap())
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
            let speed = estimate_speed(mv, self.profile.bandwidth_gbps);
            let score = score_variant(mv, &memory, &speed, use_case);
            rows.push(ScoredModel {
                model,
                variant_idx,
                memory,
                speed,
                score,
            });
        }
        rows.sort_by(|a, b| {
            b.score
                .total
                .partial_cmp(&a.score.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(rows)
    }
}
