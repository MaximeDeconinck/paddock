//! Human tables and JSON serialization for the CLI subcommands.

use tetro_core::estimate::FitVerdict;
use tetro_core::hardware::HardwareProfile;

use crate::app::ScoredModel;

/// Binary gibibytes, matching what macOS reports for RAM sizes.
pub fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

pub fn verdict_label(v: FitVerdict) -> &'static str {
    match v {
        FitVerdict::FitsGpu => "fits",
        FitVerdict::FitsWithSysctlTuning => "tune sysctl",
        FitVerdict::FitsRamOnly => "ram only",
        FitVerdict::DoesNotFit => "no fit",
    }
}

pub fn print_profile(p: &HardwareProfile) {
    println!("chip        {}", p.chip_name);
    println!("ram         {}", gib(p.ram_total_bytes));
    println!("cores       {}P + {}E", p.perf_cores, p.efficiency_cores);
    println!(
        "bandwidth   {:.0} GB/s{}",
        p.bandwidth_gbps,
        if p.bandwidth_estimated {
            " (estimated)"
        } else {
            ""
        }
    );
    match p.gpu.metal_limit_bytes {
        Some(b) => println!(
            "gpu limit   {} (Metal recommendedMaxWorkingSetSize)",
            gib(b)
        ),
        None => println!(
            "gpu limit   {} (fallback: 75% of RAM)",
            gib(p.gpu.effective_limit_bytes)
        ),
    }
    let rt = |s: &tetro_core::hardware::RuntimeStatus, name: &str| {
        if s.installed {
            format!(
                "{name} {}{}",
                s.version.as_deref().unwrap_or("?"),
                if s.running { " (running)" } else { "" }
            )
        } else {
            format!("{name} not installed")
        }
    };
    println!("runtimes    {}", rt(&p.runtimes.ollama, "ollama"));
    println!("            {}", rt(&p.runtimes.llama_cpp, "llama.cpp"));
    println!("            {}", rt(&p.runtimes.mlx, "mlx-lm"));
}

/// The product output of `tetro serve`: where to point an OpenAI client.
/// The curl line is display-only (single quotes are literal, no shell here).
pub fn print_endpoint(plan: &tetro_core::runtime::ServePlan) {
    println!("endpoint    {}", plan.endpoint);
    println!("openai      {}", plan.openai_url);
    println!("model       {}", plan.model_ref);
    println!("try it      curl -s {} \\", plan.openai_url);
    println!(
        "              -d '{{\"model\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'",
        plan.model_ref
    );
}

pub fn print_fit_table(rows: &[ScoredModel]) {
    println!(
        "{:<32} {:<9} {:>9} {:>9}  {:<12} {:>5}",
        "MODEL", "QUANT", "MEMORY", "TOK/S", "FIT", "SCORE"
    );
    for r in rows {
        let v = &r.model.variants[r.variant_idx];
        println!(
            "{:<32} {:<9} {:>9} {:>9.0}  {:<12} {:>5.0}",
            truncate(&r.model.name, 32),
            v.quant,
            gib(r.memory.total_bytes),
            r.speed.generation_tps,
            verdict_label(r.memory.verdict),
            r.score.total
        );
    }
}

#[derive(serde::Serialize)]
struct FitRow<'a> {
    name: &'a str,
    quant: &'a str,
    memory: &'a tetro_core::estimate::MemoryEstimate,
    speed: &'a tetro_core::estimate::SpeedEstimate,
    score: &'a tetro_core::score::Score,
}

pub fn print_fit_json(rows: &[ScoredModel]) -> anyhow::Result<()> {
    let out: Vec<FitRow> = rows
        .iter()
        .map(|r| FitRow {
            name: &r.model.name,
            quant: &r.model.variants[r.variant_idx].quant,
            memory: &r.memory,
            speed: &r.speed,
            score: &r.score,
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

/// The part after the colon in a recommendation line, e.g.
/// "fits GPU with 14.2 GiB to spare, ~34 tok/s (instant), 128k context".
pub fn justification(r: &ScoredModel) -> String {
    let fit = match r.memory.verdict {
        FitVerdict::FitsGpu => format!(
            "fits GPU with {} to spare",
            gib(r
                .memory
                .gpu_limit_bytes
                .saturating_sub(r.memory.total_bytes))
        ),
        FitVerdict::FitsWithSysctlTuning => "fits after sysctl tuning".to_string(),
        FitVerdict::FitsRamOnly => "fits in RAM only (CPU offload)".to_string(),
        FitVerdict::DoesNotFit => "does not fit this machine".to_string(),
    };
    format!(
        "{fit}, ~{:.0} tok/s ({}), {} context",
        r.speed.generation_tps,
        r.speed.tier.label(),
        context_label(r.model.context_max)
    )
}

fn context_label(c: u32) -> String {
    if c >= 1024 {
        format!("{}k", c / 1024)
    } else {
        c.to_string()
    }
}

pub fn print_recommendations(rows: &[ScoredModel]) {
    for r in rows {
        let v = &r.model.variants[r.variant_idx];
        println!(
            "{} {} — {:.0}/100: {}",
            r.model.name,
            v.quant,
            r.score.total,
            justification(r)
        );
    }
}

#[derive(serde::Serialize)]
struct RecommendRow<'a> {
    model: &'a str,
    quant: &'a str,
    score: f64,
    justification: String,
}

pub fn print_recommendations_json(rows: &[ScoredModel]) -> anyhow::Result<()> {
    let out: Vec<RecommendRow> = rows
        .iter()
        .map(|r| RecommendRow {
            model: &r.model.name,
            quant: &r.model.variants[r.variant_idx].quant,
            score: r.score.total,
            justification: justification(r),
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let cut: String = s.chars().take(n.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}
