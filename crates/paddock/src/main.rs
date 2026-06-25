mod app;
mod cli;
mod output;
mod tray;
mod tui;

use std::io::Write;

use anyhow::{Context, Result, bail};
use clap::Parser;
use paddock_core::PaddockError;
use paddock_core::catalog::CatalogModel;
use paddock_core::hardware::{RealSystemProbe, SystemProbe};
use paddock_core::runtime::{InstallPlan, RunPlan, ServePlan, plan_run, plan_serve};
use paddock_core::score::{UseCase, best_variant};
use paddock_core::serving::{Registry, ServingRecord};

use crate::app::App;
use crate::cli::{Cli, Command};

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
                eprintln!("catalog is empty — run `paddock sync` first");
            }
            rows.truncate(5);
            if cli.json {
                output::print_recommendations_json(&rows)?;
            } else {
                output::print_recommendations(&rows);
            }
        }
        Some(Command::Run { model, ctx }) => run_model(&app, &model, ctx, cli.json)?,
        Some(Command::Serve {
            model,
            port,
            ctx,
            foreground,
        }) => serve_model(&app, &model, port, ctx, foreground, cli.json)?,
        Some(Command::Ps) => {
            let records = Registry::open_default().list_live(&RealSystemProbe);
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&records)?);
            } else {
                output::print_ps_table(&records);
            }
        }
        Some(Command::Stop { target, yes }) => stop_servers(&target, yes)?,
        Some(Command::Sync {
            hf_limit,
            mlx_limit,
            no_ollama_registry,
            discover_limit,
            no_discover,
        }) => {
            let db = app.open_db()?;
            let http = paddock_core::catalog::hf::ReqwestClient::new()?;
            let opts = paddock_core::catalog::SyncOptions {
                hf_limit,
                mlx_limit,
                ollama_registry: !no_ollama_registry,
                discover_limit: (!no_discover).then_some(discover_limit),
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

/// Default listing shared by `paddock fit` and bare `paddock --cli/--json`.
fn fit(app: &App, all: bool, use_case: UseCase, limit: usize, json: bool) -> Result<()> {
    let db = app.open_db()?;
    let mut rows = app.scored_models(&db, use_case, all)?;
    if rows.is_empty() {
        eprintln!("catalog is empty — run `paddock sync` first");
    }
    rows.truncate(limit);
    if json {
        output::print_fit_json(&rows)?;
    } else {
        output::print_fit_table(&rows);
    }
    Ok(())
}

/// Catalog lookup + best-fitting variant pick, shared by `run` and `serve`.
/// Returns the model and the index into `model.variants` of the chosen quant.
/// Exits the process on an ambiguous name (interactive disambiguation UX).
fn resolve_model(app: &App, query: &str) -> Result<(CatalogModel, usize)> {
    let db = app.open_db()?;
    let models = db.list_models().context("reading catalog")?;
    let model = match find_model(&models, query) {
        Lookup::Found(m) => m.clone(),
        Lookup::Ambiguous(names) => {
            eprintln!("model name `{query}` is ambiguous — candidates:");
            for n in names {
                eprintln!("  {n}");
            }
            std::process::exit(1);
        }
        Lookup::NotFound => return Err(PaddockError::ModelNotFound(query.to_string()).into()),
    };

    let mvs: Vec<_> = model
        .variants
        .iter()
        .map(|v| model.to_model_variant(v))
        .collect();
    let Some(best) = best_variant(&mvs, &app.budget) else {
        bail!(
            "no quantization of `{}` fits this machine ({} RAM); try a smaller model from `paddock fit`",
            model.name,
            output::gib(app.budget.ram_total_bytes)
        );
    };
    // Pointer identity, not quant-label equality: two variants can share the
    // same quant string, and `best` borrows from `mvs` (same order as
    // `model.variants`).
    let best_idx = mvs
        .iter()
        .position(|v| std::ptr::eq(v, best))
        .expect("best_variant borrows from mvs");
    Ok((model, best_idx))
}

/// Resolve the launch context for a chosen model variant: explicit `--ctx`
/// wins, otherwise auto-size against this machine's memory budget.
fn resolved_ctx(app: &App, model: &CatalogModel, idx: usize, ctx: Option<u32>) -> u32 {
    let mv = model.to_model_variant(&model.variants[idx]);
    paddock_core::estimate::resolve_ctx(ctx, &mv, &app.budget, model.context_max)
}

fn run_model(app: &App, query: &str, ctx: Option<u32>, json: bool) -> Result<()> {
    let (model, idx) = resolve_model(app, query)?;

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
    json: bool,
) -> Result<()> {
    let (model, idx) = resolve_model(app, query)?;
    let ctx = Some(resolved_ctx(app, &model, idx, ctx));
    let plan = plan_serve(&model, &model.variants[idx], &app.profile.runtimes, port, ctx)?;

    if json {
        // Machine mode: print the plan, zero side effects (no spawn, no pull).
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }

    serve_with_plan(plan, foreground)
}

/// Full serve lifecycle: confirm install, spawn the server child when needed,
/// wait for readiness, run pre-steps (e.g. `ollama pull`), print the endpoint
/// block, then wait on the child. Shared with the TUI (Task 3).
pub(crate) fn serve_with_plan(plan: ServePlan, foreground: bool) -> Result<()> {
    if plan.port_ignored {
        eprintln!("warning: --port is ignored for the Ollama daemon (fixed 11434)");
    }
    if let Some(install) = &plan.install {
        confirm_and_install(install)?;
    }

    let log_dir = paddock_core::serving::default_serving_dir().join("logs");

    let mut detached_log: Option<std::path::PathBuf> = None;
    let mut child = match &plan.server_argv {
        Some(argv) => {
            eprintln!("$ {}", argv.join(" "));
            if foreground {
                Some(spawn_checked(argv)?)
            } else {
                // Detached: spawn to a per-invocation placeholder (our own pid
                // makes it unique across concurrent `serve`s), then rename to
                // <child-pid>.log once the child pid is known.
                let tmp_log = log_dir.join(format!("pending-{}.log", std::process::id()));
                let c = spawn_detached(argv, &tmp_log)?;
                let final_log = log_dir.join(format!("{}.log", c.id()));
                let actual = match std::fs::rename(&tmp_log, &final_log) {
                    Ok(()) => final_log,
                    Err(_) => tmp_log, // rename failed → the data is still at the pending path
                };
                detached_log = Some(actual);
                Some(c)
            }
        }
        None => None,
    };

    // Readiness + pre-steps; never leave an orphaned server behind on failure.
    let prepared = wait_ready(&plan, child.as_mut()).and_then(|()| {
        for step in &plan.pre_steps {
            eprintln!("$ {}", step.join(" "));
            run_checked(step)?;
        }
        Ok(())
    });
    if let Err(e) = prepared {
        if let Some(c) = child.as_mut() {
            let _ = c.kill();
            let _ = c.wait();
        }
        return Err(e);
    }

    // Ollama loads a model only on its first request and the daemon outlives
    // us — warm it up now so "ready" means ready (and the model shows in
    // /api/ps + the tray). Best-effort: a failure leaves a working endpoint
    // that simply cold-starts on first use.
    if plan.runtime == paddock_core::catalog::RuntimeKind::Ollama {
        eprintln!("loading {} into memory…", plan.model_ref);
        if !paddock_core::serving::warm_up_ollama(&RealSystemProbe, &plan.model_ref) {
            eprintln!("warning: warm-up failed — the model will load on the first request");
        }
    }

    output::print_endpoint(&plan);

    match child {
        Some(mut c) if foreground => {
            // Best-effort registry entry for tray/UIs; the guard unregisters
            // on every exit path including `?`. SIGINT kills paddock and the
            // child together (default tty behavior) without running Drop —
            // the stale file is reaped by the next `list_live`.
            let _guard = RegistryGuard::register(&plan, c.id(), None);
            eprintln!("serving — press Ctrl-C to stop");
            let status = c.wait()?;
            if !status.success() {
                bail!("server exited with {status}");
            }
            Ok(())
        }
        // Detached child: register WITHOUT the drop-guard so it outlives us.
        Some(c) => {
            register_detached(&plan, c.id(), detached_log.take());
            eprintln!(
                "serving in background · pid {} · paddock logs {}",
                c.id(),
                plan.model_ref
            );
            Ok(())
        }
        // Ollama daemon path: nothing to detach or track — the daemon owns the
        // model and `ollama ps` lists it. Matches today's behavior. paddock's
        // ps/stop/logs cover the spawned (llama.cpp/mlx) servers only.
        None => Ok(()),
    }
}

/// Build a serving registry record for a running server.
fn build_record(plan: &ServePlan, pid: u32, log_path: Option<std::path::PathBuf>) -> ServingRecord {
    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    ServingRecord {
        pid,
        runtime: plan.runtime,
        endpoint: plan.endpoint.clone(),
        openai_url: plan.openai_url.clone(),
        model_ref: plan.model_ref.clone(),
        ready_path: plan.ready_path.clone(),
        started_at,
        ctx: plan.ctx,
        log_path,
        port: plan.port,
    }
}

/// Register a detached child server. Unlike `RegistryGuard`, this does NOT
/// unregister on drop — the server must survive this process exiting.
fn register_detached(plan: &ServePlan, pid: u32, log_path: Option<std::path::PathBuf>) {
    let record = build_record(plan, pid, log_path);
    if let Err(e) = Registry::open_default().register(&record) {
        eprintln!("warning: could not record serving state: {e}");
    }
}

/// RAII wrapper around the serving registry: best-effort register on
/// creation, unregister on drop (normal return and `?` early-returns alike).
struct RegistryGuard {
    registry: Registry,
    pid: u32,
}

impl RegistryGuard {
    fn register(plan: &ServePlan, pid: u32, log_path: Option<std::path::PathBuf>) -> Self {
        let registry = Registry::open_default();
        let record = build_record(plan, pid, log_path);
        if let Err(e) = registry.register(&record) {
            eprintln!("warning: could not record serving state: {e}");
        }
        Self { registry, pid }
    }
}

impl Drop for RegistryGuard {
    fn drop(&mut self) {
        let _ = self.registry.unregister(self.pid);
    }
}

/// How long to wait for readiness. A spawned child gets no deadline at all:
/// runtimes like `llama-server -hf` and mlx_lm.server may be DOWNLOADING a
/// multi-GB model on first run (tens of minutes on slow links), and any fixed
/// cap conflates "still downloading" with "hung". Liveness is covered by
/// `try_wait` instead. Without a child the Ollama daemon is expected up
/// already, so refusal should be near-instant.
fn readiness_deadline(child_spawned: bool) -> Option<std::time::Duration> {
    if child_spawned {
        None
    } else {
        Some(std::time::Duration::from_secs(3))
    }
}

/// Poll `{endpoint}{ready_path}` until it answers 2xx. With a spawned child
/// this loops indefinitely — the child exiting is the only failure mode; a
/// notice after 5 s and a heartbeat every 60 s keep the user informed. Each
/// iteration blocks at most ~800 ms (300 ms connect + 500 ms read in
/// `http_get_local`, plus a 250 ms sleep), so Ctrl-C — which kills paddock and
/// the child together via default tty behavior — feels instant.
fn wait_ready(plan: &ServePlan, mut child: Option<&mut std::process::Child>) -> Result<()> {
    use std::time::{Duration, Instant};

    let url = format!("{}{}", plan.endpoint, plan.ready_path);
    let deadline = readiness_deadline(child.is_some());
    let start = Instant::now();
    let mut notified = false;
    let mut next_heartbeat = Duration::from_secs(60);
    loop {
        if RealSystemProbe.http_get_local(&url).is_some() {
            return Ok(());
        }
        if let Some(c) = child.as_deref_mut()
            && let Some(status) = c.try_wait()?
        {
            let argv = plan.server_argv.as_deref().unwrap_or_default().join(" ");
            bail!(
                "server exited with {status} before becoming ready — \
                     run `{argv}` manually to see the error"
            );
        }
        if let Some(deadline) = deadline
            && start.elapsed() >= deadline
        {
            bail!("ollama daemon not reachable on 11434 — is it running?");
        }
        if !notified && start.elapsed() >= Duration::from_secs(5) {
            eprintln!("downloading/loading model — this can take a while");
            notified = true;
        }
        if start.elapsed() >= next_heartbeat {
            eprintln!("still waiting for {} — Ctrl-C to stop", plan.endpoint);
            next_heartbeat += Duration::from_secs(60);
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Spawn a server child, with an actionable error when the binary is missing.
fn spawn_checked(argv: &[String]) -> Result<std::process::Child> {
    std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to start {}: {e}. Is it in PATH?", argv[0]))
}

// libc-free, matching serving.rs style: detach into a new session.
unsafe extern "C" {
    #[link_name = "setsid"]
    fn libc_setsid() -> i32;
}

/// Spawn a server child detached from the controlling terminal, with stdout +
/// stderr captured to `log_path`. Returns the child handle (its PID is the
/// session leader). Dropping the handle does NOT kill the process.
fn spawn_detached(argv: &[String], log_path: &std::path::Path) -> Result<std::process::Child> {
    use std::os::unix::process::CommandExt;

    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("cannot create log dir {parent:?}: {e}"))?;
    }
    let log = std::fs::File::create(log_path)
        .map_err(|e| anyhow::anyhow!("cannot create log file {log_path:?}: {e}"))?;
    let log_err = log.try_clone().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut cmd = std::process::Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .stdin(std::process::Stdio::null())
        .stdout(log)
        .stderr(log_err);
    // SAFETY: setsid only creates a new session; async-signal-safe, no allocation.
    unsafe {
        cmd.pre_exec(|| {
            libc_setsid();
            Ok(())
        });
    }
    cmd.spawn()
        .map_err(|e| anyhow::anyhow!("failed to start {}: {e}. Is it in PATH?", argv[0]))
}

fn stop_servers(target: &str, yes: bool) -> Result<()> {
    use paddock_core::catalog::RuntimeKind;
    use paddock_core::serving::{RecordMatch, match_records, terminate};

    let registry = Registry::open_default();
    let records = registry.list_live(&RealSystemProbe);
    let chosen = match match_records(&records, target) {
        RecordMatch::Matched(v) => v,
        RecordMatch::Ambiguous(cands) => {
            eprintln!("`{target}` matches several servers — be specific:");
            for r in cands {
                eprintln!("  {} (pid {})", r.model_ref, r.pid);
            }
            std::process::exit(1);
        }
        RecordMatch::NotFound => {
            eprintln!("no running server matches `{target}`");
            if !records.is_empty() {
                eprintln!(
                    "running: {}",
                    records
                        .iter()
                        .map(|r| r.model_ref.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            std::process::exit(1);
        }
    };

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

    for r in chosen {
        if r.runtime == RuntimeKind::Ollama {
            let _ = run_checked(&["ollama".into(), "stop".into(), r.model_ref.clone()]);
        } else {
            terminate(r.pid);
        }
        let _ = registry.unregister(r.pid);
        println!("stopped {} (pid {})", r.model_ref, r.pid);
    }
    Ok(())
}

/// Run a pre-step to completion (stdout/stderr inherited — progress streams
/// to the tty) and fail on non-zero exit.
fn run_checked(argv: &[String]) -> Result<()> {
    let cmd = argv.join(" ");
    let status = std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .status()
        .with_context(|| format!("running `{cmd}`"))?;
    if !status.success() {
        bail!("`{cmd}` failed ({status}); fix it and retry");
    }
    Ok(())
}

/// Shared launch path for `paddock run` and the TUI: confirm any required
/// runtime install (never auto-install), then replace this process with the
/// run command. Keeping confirmation here keeps the guarantee in one place.
pub(crate) fn launch(plan: RunPlan) -> Result<()> {
    if let Some(install) = &plan.install {
        confirm_and_install(install)?;
    }
    exec(&plan.argv)
}

fn confirm_and_install(install: &InstallPlan) -> Result<()> {
    use std::io::IsTerminal;

    let cmd = install
        .argv
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ");
    if !std::io::stdin().is_terminal() {
        eprintln!(
            "required runtime is not installed and stdin is not a terminal — \
             re-run interactively to confirm install (`{cmd}`)."
        );
        std::process::exit(1);
    }
    eprint!("required runtime is not installed. install with `{cmd}`? [y/N] ");
    std::io::stderr().flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    let answer = answer.trim().to_ascii_lowercase();
    if answer != "y" && answer != "yes" {
        eprintln!("install declined — nothing launched. Run `{cmd}` yourself, then retry.");
        std::process::exit(1);
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
        bail!("`{cmd}` failed ({status}); fix the install and retry");
    }
    Ok(())
}

fn installer_missing_hint(bin: &str) -> String {
    match bin {
        "brew" => "brew not found — install Homebrew from https://brew.sh first".to_string(),
        "uv" => "uv not found — install uv from https://docs.astral.sh/uv first".to_string(),
        other => format!("{other} not found — install it and make sure it is in PATH first"),
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

enum Lookup<'a> {
    Found(&'a CatalogModel),
    Ambiguous(Vec<&'a str>),
    NotFound,
}

/// Exact name match first, then case-insensitive exact, then
/// case-insensitive contains.
fn find_model<'a>(models: &'a [CatalogModel], query: &str) -> Lookup<'a> {
    if let Some(m) = models.iter().find(|m| m.name == query) {
        return Lookup::Found(m);
    }
    let q = query.to_lowercase();
    if let Some(m) = models.iter().find(|m| m.name.to_lowercase() == q) {
        return Lookup::Found(m);
    }
    let matches: Vec<&CatalogModel> = models
        .iter()
        .filter(|m| m.name.to_lowercase().contains(&q))
        .collect();
    match matches.as_slice() {
        [] => Lookup::NotFound,
        [one] => Lookup::Found(one),
        many => Lookup::Ambiguous(many.iter().map(|m| m.name.as_str()).collect()),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str) -> CatalogModel {
        CatalogModel {
            id: 0,
            name: name.to_string(),
            family: None,
            source: paddock_core::catalog::Source::HuggingFace,
            repo: None,
            params_total: 8_000_000_000,
            params_active: 8_000_000_000,
            architecture: None,
            context_max: 8192,
            released_at: None,
            released_approx: false,
            variants: vec![],
        }
    }

    #[test]
    fn exact_match_wins_over_contains() {
        let models = vec![model("Llama3"), model("Llama3-70B")];
        match find_model(&models, "Llama3") {
            Lookup::Found(m) => assert_eq!(m.name, "Llama3"),
            _ => panic!("expected exact match"),
        }
    }

    #[test]
    fn case_insensitive_exact_match_beats_ambiguous_contains() {
        let models = vec![model("Llama3"), model("Llama3-70B")];
        match find_model(&models, "llama3") {
            Lookup::Found(m) => assert_eq!(m.name, "Llama3"),
            other => panic!(
                "expected case-insensitive exact match, got {}",
                match other {
                    Lookup::Ambiguous(_) => "Ambiguous",
                    Lookup::NotFound => "NotFound",
                    Lookup::Found(_) => unreachable!(),
                }
            ),
        }
    }

    #[test]
    fn contains_still_resolves_unique_substring() {
        let models = vec![model("Llama3-70B"), model("Qwen2.5-Coder")];
        match find_model(&models, "qwen") {
            Lookup::Found(m) => assert_eq!(m.name, "Qwen2.5-Coder"),
            _ => panic!("expected contains match"),
        }
    }

    #[test]
    fn ambiguous_when_no_exact_and_multiple_contains() {
        let models = vec![model("Llama3-8B"), model("Llama3-70B")];
        match find_model(&models, "llama3") {
            Lookup::Ambiguous(names) => assert_eq!(names.len(), 2),
            _ => panic!("expected ambiguous"),
        }
    }

    #[test]
    fn not_found_when_nothing_matches() {
        let models = vec![model("Llama3")];
        assert!(matches!(find_model(&models, "mistral"), Lookup::NotFound));
    }

    #[test]
    fn spawned_child_waits_without_deadline() {
        // First-run model downloads can take tens of minutes; any fixed cap
        // would kill a healthy child mid-download.
        assert_eq!(readiness_deadline(true), None);
    }

    #[test]
    fn daemon_probe_keeps_short_deadline() {
        assert_eq!(
            readiness_deadline(false),
            Some(std::time::Duration::from_secs(3))
        );
    }
}
