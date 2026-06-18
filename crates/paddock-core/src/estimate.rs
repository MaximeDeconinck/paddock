//! Pure estimation formulas — the scientific core of paddock.
//!
//! Generation speed model: token generation on Apple Silicon is memory-bandwidth
//! bound. Each generated token streams all active weights once:
//!     tps = bandwidth / (params_active * bpw / 8) * EFFICIENCY
//! KV cache is GQA-aware (kv_heads, not head_count):
//!     kv = 2 (K+V) * layers * kv_heads * head_dim * context * 2 bytes (fp16)

use serde::{Deserialize, Serialize};

/// Fraction of theoretical bandwidth actually achieved by llama.cpp/MLX kernels
/// for dense models. Default calibrated on community benchmarks; a future
/// `paddock bench` module will recalibrate this per-machine.
pub const SPEED_EFFICIENCY: f64 = 0.75;
/// Fraction of theoretical bandwidth achieved for MoE models: expert routing
/// scatters weight reads, so kernels reach far less of peak bandwidth than the
/// dense streaming case. Measured ≈0.29 on M5 (Qwen3.6-35B-A3B UD-Q4_K_XL,
/// 2026-06) and ≈0.3 implied by community Qwen3-30B-A3B numbers on M3 Max;
/// to be refined by the future bench module.
pub const MOE_SPEED_EFFICIENCY: f64 = 0.3;
/// Flat runtime overhead (buffers, tokenizer, Metal heaps).
pub const OVERHEAD_BASE_BYTES: u64 = 500 * 1024 * 1024;
/// Activation overhead proportional to weights.
pub const OVERHEAD_WEIGHTS_FRACTION: f64 = 0.05;
/// Memory kept back from the GPU even with aggressive `iogpu.wired_limit_mb` tuning.
pub const SYSCTL_TUNING_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Prompt processing is compute-bound; rough multiplier range over generation speed.
pub const PROMPT_SPEED_FACTOR: (f64, f64) = (5.0, 10.0);
/// Context length used for default fit computations across the app.
pub const DEFAULT_CONTEXT: u32 = 8192;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVariant {
    pub model_name: String,
    pub quant: String,
    /// Effective bits per weight (K-quants carry metadata overhead).
    /// Must be finite and > 0; out-of-domain values degrade to zero-weight/zero-speed estimates rather than panicking.
    pub bpw: f64,
    pub params_total: u64,
    /// == params_total for dense models; active params for MoE.
    pub params_active: u64,
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub embedding_dim: u32,
    pub context_max: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitVerdict {
    /// Fits within the effective GPU (Metal) limit.
    FitsGpu,
    /// Would fit if `iogpu.wired_limit_mb` were raised (still leaving a system reserve).
    FitsWithSysctlTuning,
    /// Fits in total RAM only — partial CPU offload, degraded speed.
    FitsRamOnly,
    DoesNotFit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBudget {
    pub gpu_effective_bytes: u64,
    pub ram_total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub overhead_bytes: u64,
    pub total_bytes: u64,
    /// Mirrors `budget.gpu_effective_bytes` for forward-compat with later tasks.
    pub gpu_limit_bytes: u64,
    pub verdict: FitVerdict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SpeedTier {
    Slow,
    Usable,
    Smooth,
    Instant,
}

impl SpeedTier {
    /// Classify tps into a tier.
    /// Boundary semantics: exactly 30.0 → Smooth, exactly 15.0 → Smooth, exactly 5.0 → Usable.
    /// Matches spec: >30 instant, 15–30 smooth, 5–15 usable, <5 slow.
    pub fn from_tps(tps: f64) -> Self {
        if tps > 30.0 {
            Self::Instant
        } else if tps >= 15.0 {
            Self::Smooth
        } else if tps >= 5.0 {
            Self::Usable
        } else {
            Self::Slow
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Instant => "instant",
            Self::Smooth => "smooth",
            Self::Usable => "usable",
            Self::Slow => "slow",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedEstimate {
    pub generation_tps: f64,
    /// Rough compute-bound prompt-processing range (low, high) in tok/s.
    pub prompt_tps_range: (f64, f64),
    pub tier: SpeedTier,
}

/// KV cache size at a given context depth: K + V (×2), per layer, per KV
/// head, head_dim wide, fp16 (×2 bytes). Linear in `context_len` — this is
/// also the slope of the speed decay, since every decoded token re-streams
/// the whole cache.
pub fn kv_cache_bytes(v: &ModelVariant, context_len: u32) -> u64 {
    2u64.saturating_mul(v.layers as u64)
        .saturating_mul(v.kv_heads as u64)
        .saturating_mul(v.head_dim as u64)
        .saturating_mul(context_len as u64)
        .saturating_mul(2)
}

pub fn estimate_memory(
    v: &ModelVariant,
    context_len: u32,
    budget: &MemoryBudget,
) -> MemoryEstimate {
    let bpw = if v.bpw.is_finite() && v.bpw > 0.0 {
        v.bpw
    } else {
        0.0
    };
    let weights_bytes = (v.params_total as f64 * bpw / 8.0) as u64;
    let kv_cache_bytes = kv_cache_bytes(v, context_len);
    let overhead_bytes =
        OVERHEAD_BASE_BYTES + (weights_bytes as f64 * OVERHEAD_WEIGHTS_FRACTION) as u64;
    let total_bytes = weights_bytes
        .saturating_add(kv_cache_bytes)
        .saturating_add(overhead_bytes);

    let tunable_max = budget
        .ram_total_bytes
        .saturating_sub(SYSCTL_TUNING_RESERVE_BYTES);
    let verdict = if total_bytes <= budget.gpu_effective_bytes {
        FitVerdict::FitsGpu
    } else if total_bytes <= tunable_max {
        FitVerdict::FitsWithSysctlTuning
    } else if total_bytes <= budget.ram_total_bytes {
        FitVerdict::FitsRamOnly
    } else {
        FitVerdict::DoesNotFit
    };

    MemoryEstimate {
        weights_bytes,
        kv_cache_bytes,
        overhead_bytes,
        total_bytes,
        gpu_limit_bytes: budget.gpu_effective_bytes,
        verdict,
    }
}

/// Estimate generation speed given memory bandwidth.
/// `bandwidth_gbps`: must be finite and ≥ 0; non-finite/negative values are treated as 0.0.
/// `kv_cache_bytes`: every decoded token re-streams the KV cache built so far
/// on top of the active weights, so speed decays with context depth. Callers
/// pass the DEFAULT_CONTEXT cache from `estimate_memory`, keeping the MEMORY
/// and TOK/S columns consistent at the same 8k depth; 0 models an empty
/// context. NOTE: SPEED_EFFICIENCY/MOE_SPEED_EFFICIENCY were calibrated on
/// shallow-context benchmarks — recalibrate both when `paddock bench` lands,
/// not before, to avoid double-counting the decay.
pub fn estimate_speed(v: &ModelVariant, bandwidth_gbps: f64, kv_cache_bytes: u64) -> SpeedEstimate {
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
    let efficiency = if v.params_active < v.params_total {
        MOE_SPEED_EFFICIENCY
    } else {
        SPEED_EFFICIENCY
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn llama31_8b_q4km() -> ModelVariant {
        ModelVariant {
            model_name: "llama-3.1-8b".into(),
            quant: "Q4_K_M".into(),
            bpw: 4.83,
            params_total: 8_030_000_000,
            params_active: 8_030_000_000,
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
            embedding_dim: 4096,
            context_max: 131_072,
        }
    }

    fn llama31_70b_q4km() -> ModelVariant {
        ModelVariant {
            model_name: "llama-3.1-70b".into(),
            quant: "Q4_K_M".into(),
            bpw: 4.83,
            params_total: 70_550_000_000,
            params_active: 70_550_000_000,
            layers: 80,
            kv_heads: 8,
            head_dim: 128,
            embedding_dim: 8192,
            context_max: 131_072,
        }
    }

    /// M2 Max 36 GB: 400 GB/s, Metal limit ~27 GB.
    fn m2_max_36gb() -> MemoryBudget {
        MemoryBudget {
            gpu_effective_bytes: 27 * (1u64 << 30),
            ram_total_bytes: 36 * (1u64 << 30),
        }
    }

    #[test]
    fn speed_decays_with_kv_cache_depth() {
        let v = llama31_8b_q4km();
        let empty = estimate_speed(&v, 400.0, 0).generation_tps;
        let deep = estimate_speed(&v, 400.0, 1_000_000_000).generation_tps; // ~8k ctx
        assert!(deep < empty);
        // 8B Q4_K_M: ~4.85 GB of weights + ~1 GB of cache per token.
        let weights = 8_030_000_000.0 * 4.83 / 8.0;
        let expected_ratio = weights / (weights + 1_000_000_000.0);
        assert!(
            (deep / empty - expected_ratio).abs() < 0.02,
            "deep/empty = {}, expected ~{expected_ratio}",
            deep / empty
        );
    }

    #[test]
    fn speed_8b_q4km_on_m2_max_within_field_range() {
        let s = estimate_speed(&llama31_8b_q4km(), 400.0, 0);
        assert!(
            s.generation_tps > 55.0 && s.generation_tps < 70.0,
            "got {}",
            s.generation_tps
        );
        assert_eq!(s.tier, SpeedTier::Instant);
        assert!(s.prompt_tps_range.0 >= 5.0 * s.generation_tps - 1e-6);
        assert!(s.prompt_tps_range.1 <= 10.0 * s.generation_tps + 1e-6);
    }

    #[test]
    fn memory_70b_q4km_does_not_fit_36gb() {
        let m = estimate_memory(&llama31_70b_q4km(), 8192, &m2_max_36gb());
        // ~42.6 GB of weights alone > 36 GB RAM
        assert_eq!(m.verdict, FitVerdict::DoesNotFit);
        assert!(m.weights_bytes > 40_000_000_000);
    }

    #[test]
    fn kv_cache_8b_gqa_at_8k_under_1_1_gb() {
        let m = estimate_memory(&llama31_8b_q4km(), 8192, &m2_max_36gb());
        // 2 * 32 layers * 8 kv_heads * 128 head_dim * 8192 ctx * 2 bytes = 1.0 GiB
        assert!(m.kv_cache_bytes < 1_100_000_000, "got {}", m.kv_cache_bytes);
        assert!(m.kv_cache_bytes > 900_000_000);
    }

    #[test]
    fn kv_cache_is_gqa_aware() {
        let mut mha = llama31_8b_q4km();
        mha.kv_heads = 32; // pretend MHA instead of GQA
        let gqa = estimate_memory(&llama31_8b_q4km(), 8192, &m2_max_36gb());
        let full = estimate_memory(&mha, 8192, &m2_max_36gb());
        assert_eq!(full.kv_cache_bytes, gqa.kv_cache_bytes * 4);
    }

    #[test]
    fn verdict_ladder() {
        let v = llama31_8b_q4km(); // total ~5.6 GB
        let tight = MemoryBudget {
            gpu_effective_bytes: 5 * (1u64 << 30),
            ram_total_bytes: 16 * (1u64 << 30),
        };
        // Does not fit 5 GiB GPU limit but fits RAM-4GiB => sysctl tuning
        assert_eq!(
            estimate_memory(&v, 8192, &tight).verdict,
            FitVerdict::FitsWithSysctlTuning
        );
        let tiny = MemoryBudget {
            gpu_effective_bytes: 4 * (1u64 << 30),
            ram_total_bytes: 7 * (1u64 << 30),
        };
        // Fits in 7 GiB RAM total, but not in ram-4GiB tunable window => ram only (partial offload)
        assert_eq!(
            estimate_memory(&v, 8192, &tiny).verdict,
            FitVerdict::FitsRamOnly
        );
        let nope = MemoryBudget {
            gpu_effective_bytes: 2 * (1u64 << 30),
            ram_total_bytes: 4 * (1u64 << 30),
        };
        assert_eq!(
            estimate_memory(&v, 8192, &nope).verdict,
            FitVerdict::DoesNotFit
        );
        let roomy = m2_max_36gb();
        assert_eq!(
            estimate_memory(&v, 8192, &roomy).verdict,
            FitVerdict::FitsGpu
        );
    }

    /// MoE speed = bandwidth over *active* params at the MoE efficiency (0.3),
    /// and remains far faster than a dense model of the same *total* size.
    #[test]
    fn moe_uses_active_params_for_speed() {
        let mut moe = llama31_8b_q4km();
        moe.params_total = 30_000_000_000;
        moe.params_active = 3_000_000_000;
        let moe_speed = estimate_speed(&moe, 400.0, 0).generation_tps;
        // (a) 400e9 / (3e9 × 4.83 / 8) × 0.3 ≈ 66 tok/s
        assert!(moe_speed > 60.0 && moe_speed < 72.0, "got {moe_speed}");
        // (b) a dense model of the same TOTAL size would crawl (~16.5 tok/s);
        // the MoE must beat it by a wide margin.
        let mut dense_30b = llama31_8b_q4km();
        dense_30b.params_total = 30_000_000_000;
        dense_30b.params_active = 30_000_000_000;
        let dense_speed = estimate_speed(&dense_30b, 400.0, 0).generation_tps;
        assert!(
            moe_speed > dense_speed * 3.0,
            "moe={moe_speed} dense={dense_speed}"
        );
    }

    /// Field truth (2026-06): Qwen3.6-35B-A3B UD-Q4_K_XL on M5 (153.6 GB/s)
    /// measured 22.6 tok/s generation via llama-cli Metal.
    #[test]
    fn moe_qwen36_35b_on_m5_matches_measurement() {
        let moe = ModelVariant {
            model_name: "qwen3.6-35b-a3b".into(),
            quant: "UD-Q4_K_XL".into(),
            bpw: 5.18,
            params_total: 35_505_251_456,
            params_active: 3_000_000_000,
            layers: 48,
            kv_heads: 4,
            head_dim: 128,
            embedding_dim: 2048,
            context_max: 262_144,
        };
        let s = estimate_speed(&moe, 153.6, 0);
        assert!(
            s.generation_tps > 18.0 && s.generation_tps < 28.0,
            "measured 22.6 tok/s, estimated {}",
            s.generation_tps
        );
    }

    #[test]
    fn tiers() {
        assert_eq!(SpeedTier::from_tps(45.0), SpeedTier::Instant);
        assert_eq!(SpeedTier::from_tps(20.0), SpeedTier::Smooth);
        assert_eq!(SpeedTier::from_tps(8.0), SpeedTier::Usable);
        assert_eq!(SpeedTier::from_tps(2.0), SpeedTier::Slow);
    }

    #[test]
    fn hostile_inputs_degrade_gracefully() {
        let mut bad = llama31_8b_q4km();
        bad.bpw = f64::NAN;
        let m = estimate_memory(&bad, 8192, &m2_max_36gb());
        assert_eq!(m.weights_bytes, 0);
        let s = estimate_speed(&bad, 400.0, 0);
        assert_eq!(s.generation_tps, 0.0);
        assert_eq!(s.tier, SpeedTier::Slow);
        let s2 = estimate_speed(&llama31_8b_q4km(), f64::NAN, 0);
        assert!(s2.generation_tps == 0.0);
    }

    #[test]
    fn kv_cache_saturates_instead_of_wrapping() {
        let mut huge = llama31_8b_q4km();
        huge.layers = u32::MAX;
        huge.kv_heads = u32::MAX;
        huge.head_dim = u32::MAX;
        let m = estimate_memory(&huge, u32::MAX, &m2_max_36gb());
        assert_eq!(m.kv_cache_bytes, u64::MAX);
        assert_eq!(m.verdict, FitVerdict::DoesNotFit);
    }
}
