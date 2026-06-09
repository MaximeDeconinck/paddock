//! Composite 0-100 scoring of model variants per use case.

use serde::{Deserialize, Serialize};

use crate::estimate::{
    estimate_memory, FitVerdict, MemoryBudget, MemoryEstimate, ModelVariant, SpeedEstimate,
    DEFAULT_CONTEXT,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum UseCase {
    #[default]
    General,
    Coding,
    Chat,
    Reasoning,
}

#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub fit: f64,
    pub speed: f64,
    pub quality: f64,
    pub context: f64,
}

pub fn weights(uc: UseCase) -> Weights {
    match uc {
        UseCase::Coding => Weights {
            fit: 15.0,
            speed: 35.0,
            quality: 35.0,
            context: 15.0,
        },
        UseCase::Chat => Weights {
            fit: 20.0,
            speed: 40.0,
            quality: 30.0,
            context: 10.0,
        },
        UseCase::Reasoning => Weights {
            fit: 15.0,
            speed: 20.0,
            quality: 50.0,
            context: 15.0,
        },
        UseCase::General => Weights {
            fit: 25.0,
            speed: 25.0,
            quality: 25.0,
            context: 25.0,
        },
    }
}

/// Quantization quality malus thresholds (effective bpw).
const QUALITY_MALUS_SUB_Q4_BPW: f64 = 4.25;
const QUALITY_MALUS_SUB_Q4: f64 = 12.0;
const QUALITY_MALUS_SUB_Q3_BPW: f64 = 3.6;
const QUALITY_MALUS_SUB_Q3: f64 = 25.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub total: f64,
    pub fit: f64,
    pub speed: f64,
    pub quality: f64,
    pub context: f64,
}

pub fn score_variant(
    v: &ModelVariant,
    mem: &MemoryEstimate,
    speed: &SpeedEstimate,
    uc: UseCase,
) -> Score {
    let fit = fit_subscore(mem);
    let speed_s = speed_subscore(speed.generation_tps);
    let quality = quality_subscore(v);
    let context = context_subscore(v.context_max);
    let w = weights(uc);
    let total =
        (w.fit * fit + w.speed * speed_s + w.quality * quality + w.context * context) / 100.0;
    Score {
        total,
        fit,
        speed: speed_s,
        quality,
        context,
    }
}

/// FitsGpu rewards memory headroom: 60 base + up to 40 for margin.
fn fit_subscore(mem: &MemoryEstimate) -> f64 {
    match mem.verdict {
        FitVerdict::FitsGpu => {
            let margin = if mem.gpu_limit_bytes > 0 {
                1.0 - mem.total_bytes as f64 / mem.gpu_limit_bytes as f64
            } else {
                0.0
            };
            60.0 + 40.0 * margin.clamp(0.0, 1.0)
        }
        FitVerdict::FitsWithSysctlTuning => 45.0,
        FitVerdict::FitsRamOnly => 25.0,
        FitVerdict::DoesNotFit => 0.0,
    }
}

/// Piecewise-linear over experience tiers: (0,0) (5,25) (15,55) (30,85) (60,100).
fn speed_subscore(tps: f64) -> f64 {
    let points = [
        (0.0f64, 0.0f64),
        (5.0, 25.0),
        (15.0, 55.0),
        (30.0, 85.0),
        (60.0, 100.0),
    ];
    if tps >= 60.0 {
        return 100.0;
    }
    for w in points.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if tps <= x1 {
            return y0 + (y1 - y0) * (tps - x0) / (x1 - x0);
        }
    }
    100.0
}

/// v0.1 proxy: log10(params) normalized 1B->0, 1T->100, minus low-quant malus.
fn quality_subscore(v: &ModelVariant) -> f64 {
    let p = (v.params_total.max(1)) as f64;
    let base = ((p.log10() - 9.0) / 3.0 * 100.0).clamp(0.0, 100.0);
    let malus = if v.bpw < QUALITY_MALUS_SUB_Q3_BPW {
        QUALITY_MALUS_SUB_Q3
    } else if v.bpw < QUALITY_MALUS_SUB_Q4_BPW {
        QUALITY_MALUS_SUB_Q4
    } else {
        0.0
    };
    (base - malus).max(0.0)
}

/// log2 normalized: 4k -> 0, 128k -> 100.
fn context_subscore(context_max: u32) -> f64 {
    let c = f64::from(context_max.max(1));
    ((c.log2() - 12.0) / 5.0 * 100.0).clamp(0.0, 100.0)
}

/// Quant descent order for "best quant that fits".
pub const QUANT_DESCENT: &[&str] = &["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"];

/// Pick the best variant of one model: walk the descent, first FitsGpu wins;
/// if none fits the GPU, retry accepting FitsWithSysctlTuning, then FitsRamOnly.
/// Unknown quants (IQ4_XS, F16, MLX bits) are appended after the descent list,
/// sorted by descending bpw.
pub fn best_variant<'a>(
    variants: &'a [ModelVariant],
    budget: &MemoryBudget,
) -> Option<&'a ModelVariant> {
    let rank = |v: &ModelVariant| {
        QUANT_DESCENT
            .iter()
            .position(|q| *q == v.quant)
            .unwrap_or(QUANT_DESCENT.len())
    };
    let mut ordered: Vec<&ModelVariant> = variants.iter().collect();
    ordered.sort_by(|a, b| {
        rank(a).cmp(&rank(b)).then(
            b.bpw
                .partial_cmp(&a.bpw)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    for accept in [
        FitVerdict::FitsGpu,
        FitVerdict::FitsWithSysctlTuning,
        FitVerdict::FitsRamOnly,
    ] {
        if let Some(v) = ordered
            .iter()
            .find(|v| estimate_memory(v, DEFAULT_CONTEXT, budget).verdict == accept)
        {
            return Some(v);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::{
        estimate_memory, estimate_speed, MemoryBudget, ModelVariant, DEFAULT_CONTEXT,
    };

    fn variant(quant: &str, bpw: f64, params: u64) -> ModelVariant {
        ModelVariant {
            model_name: "test".into(),
            quant: quant.into(),
            bpw,
            params_total: params,
            params_active: params,
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
            embedding_dim: 4096,
            context_max: 131_072,
        }
    }

    fn budget() -> MemoryBudget {
        MemoryBudget {
            gpu_effective_bytes: 27 * (1u64 << 30),
            ram_total_bytes: 36 * (1u64 << 30),
        }
    }

    fn score_of(v: &ModelVariant, uc: UseCase) -> Score {
        let mem = estimate_memory(v, DEFAULT_CONTEXT, &budget());
        let speed = estimate_speed(v, 400.0);
        score_variant(v, &mem, &speed, uc)
    }

    #[test]
    fn score_in_0_100() {
        let s = score_of(&variant("Q4_K_M", 4.83, 8_030_000_000), UseCase::General);
        assert!(s.total > 0.0 && s.total <= 100.0);
    }

    #[test]
    fn reasoning_prefers_bigger_model_than_chat() {
        let small = variant("Q4_K_M", 4.83, 8_030_000_000);
        let big = variant("Q4_K_M", 4.83, 32_000_000_000);
        let chat_gap = score_of(&small, UseCase::Chat).total - score_of(&big, UseCase::Chat).total;
        let reasoning_gap =
            score_of(&small, UseCase::Reasoning).total - score_of(&big, UseCase::Reasoning).total;
        // Reasoning weighs quality 50: the big model must close the gap (or win).
        assert!(reasoning_gap < chat_gap);
    }

    #[test]
    fn low_quant_gets_quality_malus() {
        let q4 = score_of(&variant("Q4_K_M", 4.83, 8_030_000_000), UseCase::General);
        let q2 = score_of(&variant("Q2_K", 3.35, 8_030_000_000), UseCase::General);
        assert!(q2.quality < q4.quality);
    }

    #[test]
    fn doesnotfit_zeroes_fit_subscore() {
        let huge = variant("Q4_K_M", 4.83, 70_550_000_000);
        let s = score_of(&huge, UseCase::General);
        assert_eq!(s.fit, 0.0);
    }

    #[test]
    fn best_variant_picks_highest_quant_that_fits_gpu() {
        // 27 GiB GPU limit: Q8_0 of a 32B (~37.3 GB) misses GPU (FitsRamOnly),
        // Q6_K (~29.3 GB) misses GPU (FitsWithSysctlTuning), Q4_K_M (~21.9 GB) fits.
        let variants = vec![
            variant("Q8_0", 8.5, 32_000_000_000),
            variant("Q6_K", 6.59, 32_000_000_000),
            variant("Q4_K_M", 4.83, 32_000_000_000),
            variant("Q2_K", 3.35, 32_000_000_000),
        ];
        let best = best_variant(&variants, &budget()).unwrap();
        assert_eq!(best.quant, "Q4_K_M");
    }

    #[test]
    fn best_variant_falls_back_when_nothing_fits_gpu() {
        let variants = vec![variant("Q4_K_M", 4.83, 70_550_000_000)];
        // 70B Q4_K_M (~46.3GB total) doesn't fit 36GB RAM at all => None
        assert!(best_variant(&variants, &budget()).is_none());
        // But a sysctl-tunable one is returned:
        let tunable = vec![variant("Q4_K_M", 4.83, 45_000_000_000)]; // ~30.1GB total < 34.4GB tunable
        let b2 = best_variant(&tunable, &budget());
        assert!(b2.is_some());
    }

    #[test]
    fn weights_sum_to_100() {
        for uc in [
            UseCase::General,
            UseCase::Coding,
            UseCase::Chat,
            UseCase::Reasoning,
        ] {
            let w = weights(uc);
            assert!((w.fit + w.speed + w.quality + w.context - 100.0).abs() < 1e-9);
        }
    }
}
