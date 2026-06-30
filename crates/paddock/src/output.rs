//! Human tables and JSON serialization for the CLI subcommands.

use paddock_core::estimate::FitVerdict;
use paddock_core::hardware::HardwareProfile;
use paddock_core::serving::{AvailableRow, ServerRow, StopHandle};

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
    let rt = |s: &paddock_core::hardware::RuntimeStatus, name: &str| {
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

/// The product output of `paddock serve`: where to point an OpenAI client.
/// The curl line is display-only (single quotes are literal, no shell here).
pub fn print_endpoint(plan: &paddock_core::runtime::ServePlan) {
    println!("endpoint    {}", plan.endpoint);
    println!("openai      {}", plan.openai_url);
    println!("model       {}", plan.model_ref);
    println!("try it      curl -s {} \\", plan.openai_url);
    println!(
        "              -d '{{\"model\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'",
        plan.model_ref
    );
}

/// Compact age: `3d` < 14 days ≤ `2w` < 56 days ≤ `7mo` < 1 year ≤ `1.2y`.
/// `~` prefix = approximate source date; `?` = unknown release date.
pub fn age_label(released_at: Option<i64>, approx: bool, now: i64) -> String {
    let Some(r) = released_at else {
        return "?".into();
    };
    let days = ((now - r) as f64 / 86_400.0).max(0.0);
    let core = if days < 14.0 {
        format!("{}d", days as u64)
    } else if days < 56.0 {
        format!("{}w", (days / 7.0) as u64)
    } else if days < 365.25 {
        format!("{}mo", (days / 30.44) as u64)
    } else {
        format!("{:.1}y", days / 365.25)
    };
    if approx { format!("~{core}") } else { core }
}

/// Compact "time since" for the sync indicator: `5s` / `1m` / `3h` / `2d`.
pub fn humanize_since(secs: i64) -> String {
    let s = secs.max(0);
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m", s / 60)
    } else if s < 86400 {
        format!("{}h", s / 3600)
    } else {
        format!("{}d", s / 86400)
    }
}

pub fn print_fit_table(rows: &[ScoredModel]) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    println!(
        "{:<32} {:>6} {:<9} {:>9} {:>9}  {:<12} {:>5}",
        "MODEL", "AGE", "QUANT", "MEMORY", "TOK/S", "FIT", "SCORE"
    );
    for r in rows {
        let v = &r.model.variants[r.variant_idx];
        println!(
            "{:<32} {:>6} {:<9} {:>9} {:>9.0}  {:<12} {:>5.0}",
            truncate(&r.model.name, 32),
            age_label(r.model.released_at, r.model.released_approx, now),
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
    released_at: Option<i64>,
    released_approx: bool,
    quant: &'a str,
    memory: &'a paddock_core::estimate::MemoryEstimate,
    speed: &'a paddock_core::estimate::SpeedEstimate,
    score: &'a paddock_core::score::Score,
}

pub fn print_fit_json(rows: &[ScoredModel]) -> anyhow::Result<()> {
    let out: Vec<FitRow> = rows
        .iter()
        .map(|r| FitRow {
            name: &r.model.name,
            released_at: r.model.released_at,
            released_approx: r.model.released_approx,
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
            "{} {} - {:.0}/100: {}",
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

pub fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let cut: String = s.chars().take(n.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

/// `paddock ps` text view: a RUNNING section then an AVAILABLE section.
/// Empty sections are skipped; both empty prints "no servers running".
/// `now` is unix seconds, injected for deterministic tests.
pub fn servers_view(running: &[ServerRow], available: &[AvailableRow], now: i64) -> String {
    if running.is_empty() && available.is_empty() {
        return "no servers running".to_string();
    }
    let mut s = String::new();
    if !running.is_empty() {
        s.push_str("RUNNING\n");
        s.push_str(&format!(
            "{:<28} {:<9} {:<26} {:>7} {:>8} {:>7}\n",
            "MODEL", "RUNTIME", "ENDPOINT", "CTX", "UPTIME", "PID"
        ));
        for r in running {
            let ctx = r.ctx.map(|c| c.to_string()).unwrap_or_else(|| "-".into());
            let uptime = r
                .started_at
                .map(|t| humanize_since(now - t))
                .unwrap_or_else(|| "-".into());
            let pid = match &r.stop {
                StopHandle::Pid(p) => p.to_string(),
                _ => "-".into(),
            };
            s.push_str(&format!(
                "{:<28} {:<9} {:<26} {:>7} {:>8} {:>7}\n",
                truncate(&r.model, 28),
                runtime_label(r.runtime),
                truncate(&r.endpoint, 26),
                ctx,
                uptime,
                pid,
            ));
        }
    }
    if !available.is_empty() {
        if !running.is_empty() {
            s.push('\n');
        }
        s.push_str("AVAILABLE\n");
        s.push_str(&format!(
            "{:<28} {:<9} {:>10} {:>12}\n",
            "MODEL", "RUNTIME", "SIZE", "LAST-SERVED"
        ));
        for r in available {
            let size = r.size_bytes.map(gib).unwrap_or_else(|| "-".into());
            s.push_str(&format!(
                "{:<28} {:<9} {:>10} {:>12}\n",
                truncate(&r.model, 28),
                runtime_label(r.runtime),
                size,
                age_label(r.last_served_at, false, now),
            ));
        }
    }
    s.trim_end().to_string()
}

/// Print the `paddock ps` view, computing `now` from the system clock.
pub fn print_servers(running: &[ServerRow], available: &[AvailableRow]) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    println!("{}", servers_view(running, available, now));
}

pub fn runtime_label(rt: paddock_core::catalog::RuntimeKind) -> &'static str {
    use paddock_core::catalog::RuntimeKind::*;
    match rt {
        Ollama => "ollama",
        LlamaCpp => "llama.cpp",
        MlxLm => "mlx-lm",
    }
}

/// Humanized uptime (`30s`, `12m`, `3h`, `2d`) from a unix `started_at`.
pub fn uptime_label(started_at: i64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    humanize_since(now - started_at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_core::catalog::RuntimeKind;
    use paddock_core::runtime::ServePlan;

    fn ollama_plan(name: &str) -> ServePlan {
        ServePlan {
            server_argv: None,
            pre_steps: vec![],
            endpoint: "http://127.0.0.1:11434".into(),
            openai_url: "http://127.0.0.1:11434/v1/chat/completions".into(),
            model_ref: name.into(),
            ready_path: "/api/version".into(),
            install: None,
            port_ignored: false,
            runtime: RuntimeKind::Ollama,
            ctx: 0,
            port: None,
        }
    }

    #[test]
    fn servers_view_empty_says_none() {
        assert_eq!(servers_view(&[], &[], 1_780_000_000), "no servers running");
    }

    #[test]
    fn servers_view_has_sections_and_dashes() {
        let running = vec![ServerRow {
            model: "qwen3:8b".into(),
            runtime: RuntimeKind::Ollama,
            endpoint: "http://127.0.0.1:11434".into(),
            openai_url: "http://127.0.0.1:11434/v1/chat/completions".into(),
            ctx: None,
            started_at: None,
            stop: StopHandle::OllamaModel("qwen3:8b".into()),
        }];
        let available = vec![AvailableRow {
            model: "llama3".into(),
            runtime: RuntimeKind::Ollama,
            size_bytes: Some(4 * 1024 * 1024 * 1024),
            last_served_at: None,
            plan: ollama_plan("llama3"),
        }];
        let out = servers_view(&running, &available, 1_780_000_000);
        assert!(out.contains("RUNNING"));
        assert!(out.contains("AVAILABLE"));
        assert!(out.contains("qwen3:8b"));
        assert!(out.contains("llama3"));
        assert!(out.contains('-'));
        assert!(out.contains('?'));
    }

    #[test]
    fn humanize_since_buckets() {
        assert_eq!(humanize_since(5), "5s");
        assert_eq!(humanize_since(90), "1m");
        assert_eq!(humanize_since(3 * 3600 + 5), "3h");
        assert_eq!(humanize_since(2 * 86400), "2d");
        assert_eq!(humanize_since(-10), "0s"); // clock skew clamps
    }

    #[test]
    fn age_label_cases() {
        const NOW: i64 = 1_780_000_000;
        const DAY: i64 = 86_400;
        assert_eq!(age_label(None, false, NOW), "?");
        assert_eq!(age_label(Some(NOW - 3 * DAY), false, NOW), "3d");
        assert_eq!(age_label(Some(NOW - 20 * DAY), false, NOW), "2w");
        assert_eq!(age_label(Some(NOW - 240 * DAY), false, NOW), "7mo");
        assert_eq!(age_label(Some(NOW - 440 * DAY), false, NOW), "1.2y");
        assert_eq!(age_label(Some(NOW - 440 * DAY), true, NOW), "~1.2y");
        assert_eq!(age_label(Some(NOW + DAY), false, NOW), "0d"); // future clamps
    }
}
