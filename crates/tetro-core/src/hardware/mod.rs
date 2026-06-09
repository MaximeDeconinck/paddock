pub mod chip;
pub mod probe;
pub mod runtimes;

use serde::{Deserialize, Serialize};

pub use chip::{ChipFamily, ChipVariant};
pub use probe::{MockProbe, RealSystemProbe, SystemProbe};
pub use runtimes::{RuntimeStatus, RuntimesStatus};

/// GPU-addressable memory: all three values exposed so the UI can suggest
/// `sysctl iogpu.wired_limit_mb` tuning when a model barely misses the fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemory {
    pub metal_limit_bytes: Option<u64>,
    pub sysctl_wired_limit_bytes: Option<u64>,
    pub effective_limit_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub chip_name: String,
    pub family: ChipFamily,
    pub variant: ChipVariant,
    pub ram_total_bytes: u64,
    pub perf_cores: u32,
    pub efficiency_cores: u32,
    pub bandwidth_gbps: f64,
    pub bandwidth_estimated: bool,
    pub gpu: GpuMemory,
    pub runtimes: RuntimesStatus,
}

pub fn scan(probe: &dyn SystemProbe) -> HardwareProfile {
    let chip_name = probe
        .sysctl_string("machdep.cpu.brand_string")
        .unwrap_or_else(|| "unknown".to_string());
    let (family, variant) = chip::parse_brand_string(&chip_name);
    let ram_total_bytes = probe.sysctl_u64("hw.memsize").unwrap_or(0);
    let perf_cores = probe
        .sysctl_u64("hw.perflevel0.physicalcpucount")
        .unwrap_or(0) as u32;
    let efficiency_cores = probe
        .sysctl_u64("hw.perflevel1.physicalcpucount")
        .unwrap_or(0) as u32;
    let (bandwidth_gbps, bandwidth_estimated) = chip::bandwidth_gbps(family, variant, perf_cores);

    let metal_limit_bytes = probe.gpu_recommended_working_set();
    let sysctl_wired_limit_bytes = probe
        .sysctl_u64("iogpu.wired_limit_mb")
        .filter(|&v| v > 0)
        .map(|mb| mb.saturating_mul(1024 * 1024));
    let effective_limit_bytes = metal_limit_bytes
        .or(sysctl_wired_limit_bytes)
        .unwrap_or(ram_total_bytes * 3 / 4);

    HardwareProfile {
        chip_name,
        family,
        variant,
        ram_total_bytes,
        perf_cores,
        efficiency_cores,
        bandwidth_gbps,
        bandwidth_estimated,
        gpu: GpuMemory {
            metal_limit_bytes,
            sysctl_wired_limit_bytes,
            effective_limit_bytes,
        },
        runtimes: runtimes::detect_runtimes(probe),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::probe::MockProbe;

    fn m2_max_probe() -> MockProbe {
        let mut p = MockProbe::default();
        p.strings
            .insert("machdep.cpu.brand_string".into(), "Apple M2 Max".into());
        p.u64s.insert("hw.memsize".into(), 36u64 * (1 << 30));
        p.u64s.insert("hw.perflevel0.physicalcpucount".into(), 8);
        p.u64s.insert("hw.perflevel1.physicalcpucount".into(), 4);
        p.gpu_working_set = Some(27u64 * (1 << 30));
        p
    }

    #[test]
    fn scan_m2_max_profile() {
        let hw = scan(&m2_max_probe());
        assert_eq!(hw.family, ChipFamily::M2);
        assert_eq!(hw.variant, ChipVariant::Max);
        assert_eq!(hw.ram_total_bytes, 36u64 * (1 << 30));
        assert_eq!(hw.bandwidth_gbps, 400.0);
        assert_eq!(hw.gpu.effective_limit_bytes, 27u64 * (1 << 30));
        assert_eq!(hw.gpu.metal_limit_bytes, Some(27u64 * (1 << 30)));
    }

    #[test]
    fn gpu_fallback_sysctl_then_75_percent() {
        let mut p = m2_max_probe();
        p.gpu_working_set = None;
        p.u64s.insert("iogpu.wired_limit_mb".into(), 30000);
        let hw = scan(&p);
        assert_eq!(hw.gpu.effective_limit_bytes, 30000u64 * 1024 * 1024);

        let mut p2 = m2_max_probe();
        p2.gpu_working_set = None;
        let hw2 = scan(&p2);
        assert_eq!(hw2.gpu.effective_limit_bytes, 36u64 * (1 << 30) * 3 / 4);
    }

    #[test]
    fn scan_survives_missing_sysctls() {
        let hw = scan(&MockProbe::default());
        assert_eq!(hw.family, ChipFamily::Unknown);
        assert!(hw.bandwidth_estimated);
        assert_eq!(hw.ram_total_bytes, 0);
    }

    #[test]
    #[ignore]
    fn real_probe_smoke() {
        let hw = scan(&RealSystemProbe);
        // On any Mac (including Apple Silicon CI), RAM must be detectable.
        assert!(
            hw.ram_total_bytes > 0,
            "expected non-zero RAM, got {}",
            hw.ram_total_bytes
        );
        // On this Apple Silicon Mac, Metal must return a working-set limit.
        assert!(
            hw.gpu.metal_limit_bytes.is_some(),
            "expected Metal GPU limit to be present on Apple Silicon"
        );
        // Bandwidth should be non-zero and not flagged as estimated on a known chip.
        assert!(hw.bandwidth_gbps > 0.0);
        assert!(
            !hw.bandwidth_estimated,
            "expected exact bandwidth on a known chip, got estimate (family={:?})",
            hw.family
        );
        println!(
            "chip={} family={:?} variant={:?} ram={}GiB bw={}GB/s bw_estimated={} metal_limit={:?}GiB runtimes={:?}",
            hw.chip_name,
            hw.family,
            hw.variant,
            hw.ram_total_bytes / (1 << 30),
            hw.bandwidth_gbps,
            hw.bandwidth_estimated,
            hw.gpu.metal_limit_bytes.map(|b| b / (1 << 30)),
            hw.runtimes,
        );
    }
}
