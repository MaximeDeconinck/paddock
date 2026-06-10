use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChipFamily {
    M1,
    M2,
    M3,
    M4,
    M5,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChipVariant {
    Base,
    Pro,
    Max,
    Ultra,
}

/// Parse `machdep.cpu.brand_string`, e.g. "Apple M2 Max".
pub fn parse_brand_string(s: &str) -> (ChipFamily, ChipVariant) {
    let family = if s.contains("M5") {
        ChipFamily::M5
    } else if s.contains("M4") {
        ChipFamily::M4
    } else if s.contains("M3") {
        ChipFamily::M3
    } else if s.contains("M2") {
        ChipFamily::M2
    } else if s.contains("M1") {
        ChipFamily::M1
    } else {
        ChipFamily::Unknown
    };
    let variant = if s.contains("Ultra") {
        ChipVariant::Ultra
    } else if s.contains("Max") {
        ChipVariant::Max
    } else if s.contains("Pro") {
        ChipVariant::Pro
    } else {
        ChipVariant::Base
    };
    (family, variant)
}

/// Unified memory bandwidth in GB/s, plus `true` when the value is an estimate.
///
/// Sources:
///   - M1/M2 series: apple.com/newsroom "Apple unveils M1 Pro and M1 Max" (Oct 2021),
///     "Apple unveils M2 Pro and M2 Max" (Jan 2023).
///   - M3 series: apple.com/newsroom "Apple unveils M3, M3 Pro, and M3 Max" (Oct 2023).
///     M3 Pro: 150 GB/s (192-bit bus, down from 200 GB/s on M2 Pro).
///     M3 Max: 300 GB/s (14-core, 10P+4E) or 400 GB/s (16-core, 12P+4E).
///     M3 Ultra: 819.2 GB/s (newsroom Mar 2025).
///   - M4 series: apple.com/newsroom "Apple introduces M4 Pro and M4 Max" (Oct 2024).
///     M4 Pro: 273 GB/s. M4 Max: 410 GB/s (14-core, 10P) or 546 GB/s (16-core, 12P).
///     Sources confirmed via support.apple.com/en-us/121553 and /en-us/121554.
///   - M5 series: apple.com/newsroom "Apple unleashes M5" (Oct 2025, LPDDR5X-9600).
///     M5 Base: 153.6 GB/s.
///     apple.com/newsroom "Apple debuts M5 Pro and M5 Max" (Mar 2026).
///     M5 Pro: 307.0 GB/s. M5 Max: 460.0 GB/s (10P) or 614.0 GB/s (12P).
///     M5 Ultra: not yet shipped at time of writing.
///
/// Max-tier chips ship in two memory configs distinguished by CPU perf-core count:
///   M3 Max: 10P=300 GB/s / 12P=400 GB/s
///   M4 Max: 10P=410 GB/s / 12P=546 GB/s
///   M5 Max: 10P=460 GB/s / 12P=614 GB/s
pub fn bandwidth_gbps(family: ChipFamily, variant: ChipVariant, perf_cores: u32) -> (f64, bool) {
    use ChipFamily as F;
    use ChipVariant as V;
    let known = match (family, variant) {
        (F::M1, V::Base) => Some(68.25),
        (F::M1, V::Pro) => Some(200.0),
        (F::M1, V::Max) => Some(400.0),
        (F::M1, V::Ultra) => Some(800.0),
        (F::M2, V::Base) => Some(100.0),
        (F::M2, V::Pro) => Some(200.0),
        (F::M2, V::Max) => Some(400.0),
        (F::M2, V::Ultra) => Some(800.0),
        (F::M3, V::Base) => Some(102.4),
        (F::M3, V::Pro) => Some(150.0),
        (F::M3, V::Max) => Some(if perf_cores >= 12 { 400.0 } else { 300.0 }),
        (F::M3, V::Ultra) => Some(819.2),
        (F::M4, V::Base) => Some(120.0),
        (F::M4, V::Pro) => Some(273.0),
        (F::M4, V::Max) => Some(if perf_cores >= 12 { 546.0 } else { 410.0 }),
        // No M4 Ultra shipped at time of writing; conservative guess from M2/M3 Ultra pattern.
        (F::M4, V::Ultra) => None,
        (F::M5, V::Base) => Some(153.6),
        (F::M5, V::Pro) => Some(307.0),
        (F::M5, V::Max) => Some(if perf_cores >= 12 { 614.0 } else { 460.0 }),
        // No M5 Ultra shipped at time of writing; conservative guess from prior Ultra pattern.
        (F::M5, V::Ultra) => None,
        (F::Unknown, _) => None,
    };
    match known {
        Some(bw) => (bw, false),
        None => {
            // Conservative estimate by tier for unknown/future chips.
            let est = match variant {
                V::Base => 100.0,
                V::Pro => 200.0,
                V::Max => 400.0,
                V::Ultra => 800.0,
            };
            // Truly unknown family + base tier: assume the slowest Apple Silicon ever shipped.
            let est = if family == F::Unknown && variant == V::Base {
                68.25
            } else {
                est
            };
            (est, true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_brand_strings() {
        assert_eq!(
            parse_brand_string("Apple M1"),
            (ChipFamily::M1, ChipVariant::Base)
        );
        assert_eq!(
            parse_brand_string("Apple M2 Max"),
            (ChipFamily::M2, ChipVariant::Max)
        );
        assert_eq!(
            parse_brand_string("Apple M3 Pro"),
            (ChipFamily::M3, ChipVariant::Pro)
        );
        assert_eq!(
            parse_brand_string("Apple M4 Ultra"),
            (ChipFamily::M4, ChipVariant::Ultra)
        );
        assert_eq!(
            parse_brand_string("Apple M5"),
            (ChipFamily::M5, ChipVariant::Base)
        );
        assert_eq!(
            parse_brand_string("Intel(R) Core(TM) i7"),
            (ChipFamily::Unknown, ChipVariant::Base)
        );
    }

    #[test]
    fn bandwidth_known_chips_exact() {
        assert_eq!(
            bandwidth_gbps(ChipFamily::M1, ChipVariant::Base, 4),
            (68.25, false)
        );
        assert_eq!(
            bandwidth_gbps(ChipFamily::M2, ChipVariant::Max, 8),
            (400.0, false)
        );
        assert_eq!(
            bandwidth_gbps(ChipFamily::M4, ChipVariant::Pro, 10),
            (273.0, false)
        );
        assert_eq!(
            bandwidth_gbps(ChipFamily::M5, ChipVariant::Base, 4),
            (153.6, false)
        );
        assert_eq!(
            bandwidth_gbps(ChipFamily::M5, ChipVariant::Pro, 12),
            (307.0, false)
        );
    }

    #[test]
    fn bandwidth_max_depends_on_perf_cores() {
        // M3 Max 14-core CPU (10P+4E) = 300 GB/s, 16-core (12P+4E) = 400 GB/s
        assert_eq!(
            bandwidth_gbps(ChipFamily::M3, ChipVariant::Max, 10),
            (300.0, false)
        );
        assert_eq!(
            bandwidth_gbps(ChipFamily::M3, ChipVariant::Max, 12),
            (400.0, false)
        );
        // M4 Max 14-core = 410, 16-core = 546
        assert_eq!(
            bandwidth_gbps(ChipFamily::M4, ChipVariant::Max, 10),
            (410.0, false)
        );
        assert_eq!(
            bandwidth_gbps(ChipFamily::M4, ChipVariant::Max, 12),
            (546.0, false)
        );
        // M5 Max 14-core (10P) = 460, 16-core (12P) = 614
        assert_eq!(
            bandwidth_gbps(ChipFamily::M5, ChipVariant::Max, 10),
            (460.0, false)
        );
        assert_eq!(
            bandwidth_gbps(ChipFamily::M5, ChipVariant::Max, 12),
            (614.0, false)
        );
    }

    #[test]
    fn bandwidth_unknown_chip_is_conservative_estimate() {
        let (bw, est) = bandwidth_gbps(ChipFamily::Unknown, ChipVariant::Base, 4);
        assert!(est);
        assert!(bw <= 100.0);
    }
}
