mod app;
mod cli;
mod clipboard;
mod lifecycle;
mod output;
mod tray;
mod tui;

use std::io::Write;

use anyhow::{Context, Result, bail};
use clap::Parser;
use paddock_core::hardware::RealSystemProbe;
use paddock_core::runtime::{InstallPlan, RunPlan, ServePlan, plan_run, plan_serve};
use paddock_core::score::UseCase;
use paddock_core::serving::Registry;

use crate::app::App;
use crate::cli::{Cli, Command};
use crate::lifecycle::{
    LifecycleError, RegistryGuard, StderrProgress, resolve_model, resolved_ctx,
};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let app = App::load();

    match cli.command {
        Some(Command::Scan) => {
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&app.profile)?);
            } else {
                output::print_profile(&app.profile);
            }
        }
        Some(Command::Fit {
            all,
            use_case,
            limit,
        }) => fit(&app, all, use_case.into(), limit, cli.json)?,
        Some(Command::Recommend { use_case }) => {
            let db = app.open_db()?;
            let mut rows = app.scored_models(&db, use_case.into(), false)?;
            if rows.is_empty() {
                eprintln!("catalog is empty - run `paddock sync` first");
            }
            rows.truncate(5);
            if cli.json {
                output::print_recommendations_json(&rows)?;
            } else {
                output::print_recommendations(&rows);
            }
        }
        Some(Command::Run { model, ctx, quant }) => run_model(&app, &model, ctx, quant, cli.json)?,
        Some(Command::Serve {
            model,
            port,
            ctx,
            foreground,
            quant,
        }) => serve_model(&app, &model, port, ctx, foreground, quant, cli.json)?,
        Some(Command::Ps) => {
            let registry = Registry::open_default();
            let probe = RealSystemProbe;
            let running = paddock_core::serving::list_all_servers(&registry, &probe);
            let history = paddock_core::serving::History::open_default();
            let available = paddock_core::serving::list_available(&history, &probe, &running);
            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "running": &running,
                        "available": &available,
                    }))?
                );
            } else {
                output::print_servers(&running, &available);
            }
        }
        Some(Command::Stop { target, yes }) => stop_servers(&target, yes)?,
        Some(Command::Logs { target, follow }) => show_logs(&target, follow)?,
        Some(Command::Bench { target, tokens }) => {
            bench_server(&app, target.as_deref(), tokens, cli.json)?
        }
        Some(Command::Sync {
            hf_limit,
            mlx_limit,
            no_ollama_registry,
            discover_limit,
            no_discover,
            hf_trending_limit,
            ollama_newest_reserve,
        }) => {
            let db = app.open_db()?;
            let http = paddock_core::catalog::hf::ReqwestClient::new()?;
            let opts = paddock_core::catalog::SyncOptions {
                hf_limit,
                mlx_limit,
                ollama_registry: !no_ollama_registry,
                discover_limit: (!no_discover).then_some(discover_limit),
                hf_trending_limit,
                ollama_newest_reserve,
            };
            let report = tokio::runtime::Runtime::new()?
                .block_on(paddock_core::catalog::sync(&http, &db, &opts))?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "synced: {} curated ({} ollama tags), {} discovered, {} huggingface, {} mlx",
                    report.curated,
                    report.ollama_tags,
                    report.discovered,
                    report.huggingface,
                    report.mlx
                );
                for e in &report.errors {
                    eprintln!("warning: {e}");
                }
            }
        }
        Some(Command::Tray) => tray::run()?,
        None => {
            if cli.cli || cli.json {
                fit(&app, false, UseCase::General, 20, cli.json)?;
            } else {
                tui::run(app)?;
            }
        }
    }
    Ok(())
}

/// CLI rendering of lifecycle errors. The interactive-disambiguation cases
/// print exactly today's multi-line message and exit 1; everything else
/// propagates through anyhow (Rust prints `Error: <Display>`, as `bail!` did).
fn cli_fail(e: LifecycleError) -> anyhow::Error {
    match e {
        LifecycleError::Ambiguous { .. }
        | LifecycleError::AmbiguousServer { .. }
        | LifecycleError::NoServerMatch { .. }
        | LifecycleError::InstallDeclined { .. } => {
            eprintln!("{e}");
            std::process::exit(1);
        }
        other => other.into(),
    }
}

/// Default listing shared by `paddock fit` and bare `paddock --cli/--json`.
fn fit(app: &App, all: bool, use_case: UseCase, limit: usize, json: bool) -> Result<()> {
    let db = app.open_db()?;
    let mut rows = app.scored_models(&db, use_case, all)?;
    if rows.is_empty() {
        eprintln!("catalog is empty - run `paddock sync` first");
    }
    rows.truncate(limit);
    if json {
        output::print_fit_json(&rows)?;
    } else {
        output::print_fit_table(&rows);
    }
    Ok(())
}

fn run_model(
    app: &App,
    query: &str,
    ctx: Option<u32>,
    quant: Option<String>,
    json: bool,
) -> Result<()> {
    let (model, idx) = resolve_model(app, query, quant.as_deref()).map_err(cli_fail)?;

    // API delta vs the original plan: plan_run is fallible (repo-less HF/MLX
    // models, non-GGUF quants). Surface the actionable error and exit non-zero.
    let ctx = Some(resolved_ctx(app, &model, idx, ctx));
    let plan: RunPlan = plan_run(&model, &model.variants[idx], &app.profile.runtimes, ctx)?;

    if json {
        // Machine mode never launches interactive processes.
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }

    println!("$ {}", plan.display());
    launch(plan)
}

fn serve_model(
    app: &App,
    query: &str,
    port: Option<u16>,
    ctx: Option<u32>,
    foreground: bool,
    quant: Option<String>,
    json: bool,
) -> Result<()> {
    let (model, idx) = resolve_model(app, query, quant.as_deref()).map_err(cli_fail)?;
    let ctx = Some(resolved_ctx(app, &model, idx, ctx));
    let plan = plan_serve(
        &model,
        &model.variants[idx],
        &app.profile.runtimes,
        port,
        ctx,
    )?;

    if json {
        // Machine mode: print the plan, zero side effects (no spawn, no pull).
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }

    serve_with_plan(plan, foreground)
}

/// CLI/TUI adapter over `lifecycle::serve`: same terminal output as before
/// (endpoint block on stdout, progress on stderr), install confirmed on the
/// tty, no readiness timeout. Shared with the TUI `s` key.
pub(crate) fn serve_with_plan(plan: ServePlan, foreground: bool) -> Result<()> {
    use crate::lifecycle::{InstallPolicy, ServeMode, serve};
    use paddock_core::catalog::RuntimeKind;

    let mode = if foreground {
        ServeMode::Foreground
    } else {
        ServeMode::Detached
    };
    let outcome = serve(
        plan,
        mode,
        InstallPolicy::Ask(&confirm_and_install),
        None,
        &StderrProgress,
    )
    .map_err(cli_fail)?;
    let plan = outcome.plan;
    output::print_endpoint(&plan);

    match outcome.child {
        Some(mut c) => {
            // Best-effort registry entry for tray/UIs; the guard unregisters
            // on every exit path including `?`. SIGINT kills paddock and the
            // child together (default tty behavior) without running Drop -
            // the stale file is reaped by the next `list_live`.
            let _guard = (plan.runtime != RuntimeKind::Ollama)
                .then(|| RegistryGuard::register(&plan, c.id(), None));
            eprintln!("serving - press Ctrl-C to stop");
            let status = c.wait()?;
            if !status.success() {
                bail!("server exited with {status}");
            }
            Ok(())
        }
        None => {
            match outcome.pid {
                // Cold-started the Ollama daemon; it serves in the background on
                // its fixed port. ollama ps / ollama stop manage it, not paddock.
                Some(_) if plan.runtime == RuntimeKind::Ollama => {
                    eprintln!("ollama daemon started in the background");
                }
                Some(pid) => {
                    eprintln!(
                        "serving in background · pid {pid} · paddock logs {}",
                        plan.model_ref
                    );
                }
                // Already-running Ollama daemon: nothing was spawned, so nothing
                // to detach or track - the daemon owns the model and `ollama ps`
                // lists it. paddock's ps/stop/logs cover llama.cpp/mlx only.
                None => {}
            }
            Ok(())
        }
    }
}

fn stop_servers(target: &str, yes: bool) -> Result<()> {
    use crate::lifecycle::{resolve_servers, stop_records};

    let chosen = resolve_servers(target).map_err(cli_fail)?;

    if target == "all" && !yes {
        eprintln!("about to stop {} server(s):", chosen.len());
        for r in &chosen {
            eprintln!("  {} (pid {})", r.model_ref, r.pid);
        }
        eprint!("proceed? [y/N] ");
        std::io::stderr().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        if !matches!(line.trim(), "y" | "Y") {
            eprintln!("aborted");
            return Ok(());
        }
    }

    for s in stop_records(chosen, &StderrProgress) {
        println!("stopped {} (pid {})", s.model_ref, s.pid);
    }
    Ok(())
}

fn show_logs(target: &str, follow: bool) -> Result<()> {
    use paddock_core::serving::{RecordMatch, match_records};

    let records = Registry::open_default().list_live(&RealSystemProbe);
    let chosen = match match_records(&records, target) {
        RecordMatch::Matched(v) if v.len() == 1 => v[0].clone(),
        RecordMatch::Matched(_) | RecordMatch::Ambiguous(_) => {
            eprintln!("`{target}` matches several servers - use a pid");
            std::process::exit(1);
        }
        RecordMatch::NotFound => {
            eprintln!("no running server matches `{target}`");
            std::process::exit(1);
        }
    };

    let Some(path) = chosen.log_path.clone() else {
        eprintln!(
            "{} runs under {:?} which keeps its own logs (no paddock log file)",
            chosen.model_ref, chosen.runtime
        );
        return Ok(());
    };

    if follow {
        // Delegate to `tail -f` for follow semantics. Unlike `run_checked`, a
        // non-zero exit is NOT an error here: the user ends `tail -f` with
        // Ctrl-C (exit 130), which is the normal way to stop following.
        std::process::Command::new("tail")
            .args(["-f".as_ref(), path.as_os_str()])
            .status()
            .map_err(|e| anyhow::anyhow!("failed to run tail: {e}. Is it in PATH?"))?;
        Ok(())
    } else {
        let body = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("cannot read log {path:?}: {e}"))?;
        print!("{body}");
        Ok(())
    }
}

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
                None => eprintln!(
                    "several servers are running - pick one with `paddock bench <target>`:"
                ),
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
                            rows.iter()
                                .map(|r| r.model.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                }
            }
            std::process::exit(1);
        }
    };

    if !json {
        eprintln!(
            "benching {} on {} ({tokens} tokens)...",
            row.model, row.endpoint
        );
    }
    let measured = measure(&probe, row.runtime, &row.endpoint, &row.model, tokens)?;

    let db = app.open_db()?;
    let models = db.list_models().context("reading catalog")?;
    let resolved = resolve_model_ref(&models, &row.model).map(|(mi, vi)| {
        (
            &models[mi],
            models[mi].to_model_variant(&models[mi].variants[vi]),
        )
    });

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

/// Shared launch path for `paddock run` and the TUI: confirm any required
/// runtime install (never auto-install), then replace this process with the
/// run command. Keeping confirmation here keeps the guarantee in one place.
pub(crate) fn launch(plan: RunPlan) -> Result<()> {
    if let Some(install) = &plan.install {
        confirm_and_install(install).map_err(cli_fail)?;
    }
    exec(&plan.argv)
}

pub(crate) fn confirm_and_install(install: &InstallPlan) -> Result<(), LifecycleError> {
    use std::io::IsTerminal;

    let cmd = install
        .argv
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ");
    if !std::io::stdin().is_terminal() {
        eprintln!(
            "required runtime is not installed and stdin is not a terminal - \
             re-run interactively to confirm install (`{cmd}`)."
        );
        std::process::exit(1);
    }
    eprint!("required runtime is not installed. install with `{cmd}`? [y/N] ");
    std::io::stderr().flush().map_err(anyhow::Error::from)?;
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(anyhow::Error::from)?;
    let answer = answer.trim().to_ascii_lowercase();
    if answer != "y" && answer != "yes" {
        return Err(LifecycleError::InstallDeclined { command: cmd });
    }
    // Check the installer binary exists before running it (avoid exec-ENOENT).
    let installer = &install.argv[0];
    if !find_in_path(installer) {
        eprintln!("{}", installer_missing_hint(installer));
        std::process::exit(1);
    }
    let status = std::process::Command::new(installer)
        .args(&install.argv[1..])
        .status()
        .with_context(|| format!("running `{cmd}`"))?;
    if !status.success() {
        return Err(anyhow::anyhow!("`{cmd}` failed ({status}); fix the install and retry").into());
    }
    Ok(())
}

fn installer_missing_hint(bin: &str) -> String {
    match bin {
        "brew" => "brew not found - install Homebrew from https://brew.sh first".to_string(),
        "uv" => "uv not found - install uv from https://docs.astral.sh/uv first".to_string(),
        other => format!("{other} not found - install it and make sure it is in PATH first"),
    }
}

/// `which`-style PATH scan; absolute/relative paths are checked directly.
fn find_in_path(bin: &str) -> bool {
    if bin.contains('/') {
        return std::path::Path::new(bin).is_file();
    }
    std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).any(|dir| dir.join(bin).is_file()))
        .unwrap_or(false)
}

/// Replace this process with the run command. Shared with the TUI (Task 7).
pub(crate) fn exec(argv: &[String]) -> Result<()> {
    use std::os::unix::process::CommandExt;
    let err = std::process::Command::new(&argv[0]).args(&argv[1..]).exec();
    Err(anyhow::anyhow!(
        "failed to launch {}: {err}. Is it in PATH?",
        argv[0]
    ))
}
