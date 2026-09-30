//! Serve / stop lifecycle shared by the CLI, the TUI and the MCP server.
//! Everything here returns values: no stdout, no stdin, no `process::exit`.
//! Progress goes through `Progress`; the only direct output is best-effort
//! registry warnings on stderr. Adapters render `LifecycleError` and report
//! `Progress` their own way.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use paddock_core::catalog::{CatalogModel, RuntimeKind};
use paddock_core::estimate::{MemoryBudget, ModelVariant};
use paddock_core::hardware::{RealSystemProbe, SystemProbe};
use paddock_core::runtime::{InstallPlan, ServePlan};
use paddock_core::score::best_variant;
use paddock_core::serving::{History, Registry, ServingRecord};

use crate::app::App;

/// How the lifecycle reports what it is doing. The CLI/TUI print to stderr
/// (byte-for-byte today's messages); the MCP server discards everything
/// because stdout is the protocol channel and stderr is for internal errors.
pub trait Progress {
    /// Informational line: "downloading/loading model - this can take a while",
    /// "port 8080 is busy - serving on 8081 instead", heartbeats.
    fn note(&self, msg: &str);
    /// A command about to run ("$ ollama pull ...").
    fn command(&self, argv: &[String]);
    /// True when child commands (pre-steps, `ollama stop`) must not inherit
    /// this process's stdout/stderr.
    fn quiet(&self) -> bool {
        false
    }
}

pub struct StderrProgress;

impl Progress for StderrProgress {
    fn note(&self, msg: &str) {
        eprintln!("{msg}");
    }
    fn command(&self, argv: &[String]) {
        eprintln!("$ {}", argv.join(" "));
    }
}

pub struct SilentProgress;

impl Progress for SilentProgress {
    fn note(&self, _msg: &str) {}
    fn command(&self, _argv: &[String]) {}
    fn quiet(&self) -> bool {
        true
    }
}

/// Every way serve/stop can fail. `Display` reproduces the CLI wording so the
/// CLI adapter can print it as-is; the MCP adapter maps variants to codes.
#[derive(Debug, thiserror::Error)]
pub enum LifecycleError {
    #[error(
        "model `{0}` not found in catalog. Run `paddock sync` or check `paddock fit` for names."
    )]
    ModelNotFound(String),
    #[error("model name `{query}` is ambiguous - candidates:{}", candidates.iter().map(|c| format!("\n  {c}")).collect::<String>())]
    Ambiguous {
        query: String,
        candidates: Vec<String>,
    },
    #[error(
        "no quantization of `{model}` fits this machine ({} RAM); try a smaller model from `paddock fit`",
        crate::output::gib(*ram_bytes)
    )]
    NoFit { model: String, ram_bytes: u64 },
    #[error("no quant `{label}` for this model; available: {}", available.join(", "))]
    UnknownQuant {
        label: String,
        available: Vec<String>,
    },
    #[error("required runtime is not installed - run `{}` then retry", install.argv.join(" "))]
    NoRuntime { install: InstallPlan },
    #[error("install declined - nothing launched. Run `{command}` yourself, then retry.")]
    InstallDeclined { command: String },
    #[error(
        "server exited with {status} before becoming ready - run `{}` manually to see the error",
        argv.join(" ")
    )]
    ServerExited {
        status: String,
        argv: Vec<String>,
        log_path: Option<PathBuf>,
    },
    /// Readiness timeout on a detached child. The server is still running and
    /// (for llama.cpp/mlx) already registered, so `paddock ps` can see it.
    #[error(
        "server not ready yet (pid {}) - it keeps running; check `paddock ps`",
        pid.map(|p| p.to_string()).unwrap_or_else(|| "?".into())
    )]
    NotReady {
        pid: Option<u32>,
        log_path: Option<PathBuf>,
        plan: Box<ServePlan>,
    },
    #[error("ollama daemon not reachable on 11434 - is it running?")]
    OllamaUnreachable,
    #[error("no running server matches `{target}`{}", if running.is_empty() { String::new() } else { format!("\nrunning: {}", running.join(", ")) })]
    NoServerMatch {
        target: String,
        running: Vec<String>,
    },
    #[error("`{target}` matches several servers - be specific:{}", candidates.iter().map(|(m, p)| format!("\n  {m} (pid {p})")).collect::<String>())]
    AmbiguousServer {
        target: String,
        candidates: Vec<(String, u32)>,
    },
    #[error("catalog is empty - run `paddock sync` first")]
    CatalogEmpty,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<paddock_core::PaddockError> for LifecycleError {
    fn from(e: paddock_core::PaddockError) -> Self {
        LifecycleError::Other(e.into())
    }
}

/// What to do when the plan needs a runtime that is not installed.
pub enum InstallPolicy<'a> {
    /// Ask the human (CLI/TUI): the callback prompts and installs, or returns
    /// `InstallDeclined`.
    Ask(&'a dyn Fn(&InstallPlan) -> Result<(), LifecycleError>),
    /// Never install (MCP): return `NoRuntime` before touching anything.
    Refuse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeMode {
    /// Child inherits the tty; the caller waits on `ServeOutcome.child`.
    Foreground,
    /// Child is detached (own session, log file) and registered; it outlives
    /// this process.
    Detached,
}

#[derive(Debug)]
pub struct ServeOutcome {
    /// Final plan (the port may have moved).
    pub plan: ServePlan,
    /// Spawned child pid; None when the already-running Ollama daemon serves.
    pub pid: Option<u32>,
    /// Detached log file, when a child was spawned detached.
    pub log_path: Option<PathBuf>,
    /// Foreground only: the child handle for the caller to wait on.
    pub child: Option<std::process::Child>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stopped {
    pub model_ref: String,
    pub pid: u32,
    pub runtime: RuntimeKind,
}

pub(crate) enum Lookup<'a> {
    Found(&'a CatalogModel),
    Ambiguous(Vec<&'a str>),
    NotFound,
}

/// Exact name match first, then case-insensitive exact, then
/// case-insensitive contains.
pub(crate) fn find_model<'a>(models: &'a [CatalogModel], query: &str) -> Lookup<'a> {
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

/// Index into `variants` of the variant whose quant label equals `label`
/// (case-insensitive). On a label shared by several variants, returns the
/// best-quality one (first in `variants_by_quality` order).
pub(crate) fn resolve_quant(
    variants: &[ModelVariant],
    label: &str,
) -> Result<usize, LifecycleError> {
    let order = paddock_core::score::variants_by_quality(variants);
    if let Some(&idx) = order
        .iter()
        .find(|&&i| variants[i].quant.eq_ignore_ascii_case(label))
    {
        return Ok(idx);
    }
    Err(LifecycleError::UnknownQuant {
        label: label.to_string(),
        available: order.iter().map(|&i| variants[i].quant.clone()).collect(),
    })
}

/// Catalog lookup + best-fitting variant pick over an in-memory model list.
/// Returns the model and the index into `model.variants` of the chosen quant.
pub fn resolve_in(
    models: &[CatalogModel],
    budget: &MemoryBudget,
    query: &str,
    quant: Option<&str>,
) -> Result<(CatalogModel, usize), LifecycleError> {
    if models.is_empty() {
        return Err(LifecycleError::CatalogEmpty);
    }
    let model = match find_model(models, query) {
        Lookup::Found(m) => m.clone(),
        Lookup::Ambiguous(names) => {
            return Err(LifecycleError::Ambiguous {
                query: query.to_string(),
                candidates: names.into_iter().map(str::to_string).collect(),
            });
        }
        Lookup::NotFound => return Err(LifecycleError::ModelNotFound(query.to_string())),
    };

    let mvs: Vec<_> = model
        .variants
        .iter()
        .map(|v| model.to_model_variant(v))
        .collect();

    // Explicit quant launches that variant even if it does not fit; the
    // verdict is informational (consistent with the TUI quant picker).
    if let Some(label) = quant {
        let idx = resolve_quant(&mvs, label)?;
        return Ok((model, idx));
    }

    let Some(best) = best_variant(&mvs, budget) else {
        return Err(LifecycleError::NoFit {
            model: model.name.clone(),
            ram_bytes: budget.ram_total_bytes,
        });
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

/// `resolve_in` over the on-disk catalog. Shared by `run`, `serve` and MCP.
pub fn resolve_model(
    app: &App,
    query: &str,
    quant: Option<&str>,
) -> Result<(CatalogModel, usize), LifecycleError> {
    let db = app.open_db()?;
    let models = db
        .list_models()
        .map_err(|e| LifecycleError::Other(anyhow::Error::from(e).context("reading catalog")))?;
    resolve_in(&models, &app.budget, query, quant)
}

/// Resolve the launch context for a chosen model variant: explicit `--ctx`
/// wins, otherwise auto-size against this machine's memory budget.
pub fn resolved_ctx(app: &App, model: &CatalogModel, idx: usize, ctx: Option<u32>) -> u32 {
    let mv = model.to_model_variant(&model.variants[idx]);
    paddock_core::estimate::resolve_ctx(ctx, &mv, &app.budget, model.context_max)
}

/// How long to wait for readiness. A spawned child gets whatever the caller
/// asked for (the CLI asks for None: runtimes like `llama-server -hf` may be
/// downloading a multi-GB model on first run, and any fixed cap conflates
/// "still downloading" with "hung"). Without a child the Ollama daemon is
/// expected up already, so refusal should be near-instant.
pub(crate) fn readiness_deadline(
    child_spawned: bool,
    requested: Option<Duration>,
) -> Option<Duration> {
    if child_spawned {
        requested
    } else {
        Some(Duration::from_secs(3))
    }
}

/// Poll `{endpoint}{ready_path}` until it answers 2xx. Each iteration blocks
/// at most ~800 ms (300 ms connect + 500 ms read in `http_get_local`, plus a
/// 250 ms sleep), so Ctrl-C feels instant. A notice after 5 s and a heartbeat
/// every 60 s go through `progress`. On `timeout`: with a child, `NotReady`
/// (the child is NOT killed here, the caller decides); without, the daemon
/// case, `OllamaUnreachable`.
pub(crate) fn wait_ready(
    probe: &dyn SystemProbe,
    plan: &ServePlan,
    mut child: Option<&mut std::process::Child>,
    timeout: Option<Duration>,
    progress: &dyn Progress,
) -> Result<(), LifecycleError> {
    let url = format!("{}{}", plan.endpoint, plan.ready_path);
    let start = Instant::now();
    let mut notified = false;
    let mut next_heartbeat = Duration::from_secs(60);
    loop {
        if probe.http_get_local(&url).is_some() {
            return Ok(());
        }
        if let Some(c) = child.as_deref_mut()
            && let Some(status) = c.try_wait().map_err(anyhow::Error::from)?
        {
            return Err(LifecycleError::ServerExited {
                status: status.to_string(),
                argv: plan.server_argv.clone().unwrap_or_default(),
                log_path: None,
            });
        }
        if let Some(deadline) = timeout
            && start.elapsed() >= deadline
        {
            return Err(match child.as_deref() {
                Some(c) => LifecycleError::NotReady {
                    pid: Some(c.id()),
                    log_path: None,
                    plan: Box::new(plan.clone()),
                },
                None => LifecycleError::OllamaUnreachable,
            });
        }
        if !notified && start.elapsed() >= Duration::from_secs(5) {
            progress.note("downloading/loading model - this can take a while");
            notified = true;
        }
        if start.elapsed() >= next_heartbeat {
            progress.note(&format!(
                "still waiting for {} - Ctrl-C to stop",
                plan.endpoint
            ));
            next_heartbeat += Duration::from_secs(60);
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Spawn a server child on the tty, with an actionable error when the binary
/// is missing.
pub(crate) fn spawn_checked(argv: &[String]) -> Result<std::process::Child, LifecycleError> {
    std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to start {}: {e}. Is it in PATH?", argv[0]).into())
}

// libc-free, matching serving.rs style: detach into a new session.
unsafe extern "C" {
    #[link_name = "setsid"]
    fn libc_setsid() -> i32;
}

/// Spawn a server child detached from the controlling terminal, with stdout +
/// stderr captured to `log_path`. Dropping the handle does NOT kill it.
pub(crate) fn spawn_detached(
    argv: &[String],
    log_path: &Path,
) -> Result<std::process::Child, LifecycleError> {
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
        .map_err(|e| anyhow::anyhow!("failed to start {}: {e}. Is it in PATH?", argv[0]).into())
}

/// `started + budget`, or `None` (unbounded) when the sum overflows `Instant`
/// (e.g. an MCP `timeout_secs` of u64::MAX). Never panics.
pub(crate) fn deadline_after(started: Instant, budget: Duration) -> Option<Instant> {
    started.checked_add(budget)
}

/// Process-wide sequence for pending log names: MCP dispatches tool calls
/// concurrently in one process, so the pid alone is not unique.
static PENDING_LOG_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A fresh `pending-<pid>-<n>.log` name, unique across concurrent serves in
/// this process and across processes.
pub(crate) fn pending_log_name() -> String {
    let n = PENDING_LOG_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("pending-{}-{n}.log", std::process::id())
}

/// Run a pre-step to completion and fail on non-zero exit. Output streams to
/// the tty unless `progress.quiet()` (MCP: stdout is the protocol channel).
pub(crate) fn run_checked(argv: &[String], progress: &dyn Progress) -> Result<(), LifecycleError> {
    run_checked_until(argv, progress, None).map(|_| ())
}

/// `run_checked` with an optional deadline. `None` waits for the command to
/// finish (the CLI path). `Some(deadline)` polls it every 250 ms and returns
/// `Ok(Some(pid))` if it is still running at the deadline; the command is
/// left running (e.g. an `ollama pull` that keeps downloading).
fn run_checked_until(
    argv: &[String],
    progress: &dyn Progress,
    deadline: Option<Instant>,
) -> Result<Option<u32>, LifecycleError> {
    use std::process::Stdio;
    let cmd = argv.join(" ");
    let stdio = || {
        if progress.quiet() {
            Stdio::null()
        } else {
            Stdio::inherit()
        }
    };
    let mut command = std::process::Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(stdio())
        .stderr(stdio());
    let spawn_err = |e: std::io::Error| anyhow::anyhow!("running `{cmd}`: {e}");
    let status = match deadline {
        None => command.status().map_err(spawn_err)?,
        Some(deadline) => {
            let mut child = command.spawn().map_err(spawn_err)?;
            loop {
                if let Some(status) = child.try_wait().map_err(spawn_err)? {
                    break status;
                }
                if Instant::now() >= deadline {
                    let pid = child.id();
                    // Reap it when it finishes so a long MCP session does
                    // not accumulate zombies.
                    std::thread::spawn(move || {
                        let _ = child.wait();
                    });
                    return Ok(Some(pid));
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        }
    };
    if !status.success() {
        return Err(anyhow::anyhow!("`{cmd}` failed ({status}); fix it and retry").into());
    }
    Ok(None)
}

/// Run `f` and return its result, or `None` if `deadline` passes first.
/// `None` deadline: `f` runs inline on this thread (the CLI path). With a
/// deadline, `f` runs on a spawned thread that is NOT aborted on expiry: it
/// finishes in the background (e.g. a warm-up request that keeps the model
/// loading after the caller moved on).
pub(crate) fn run_with_deadline<T: Send + 'static>(
    deadline: Option<Instant>,
    f: impl FnOnce() -> T + Send + 'static,
) -> Option<T> {
    let Some(deadline) = deadline else {
        return Some(f());
    };
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        // The receiver is gone on expiry; the result is simply dropped.
        let _ = tx.send(f());
    });
    rx.recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .ok()
}

/// True for runtimes that load the model lazily on the first request, so
/// readiness alone does not mean "loaded".
pub(crate) fn needs_warm_up(runtime: RuntimeKind) -> bool {
    matches!(runtime, RuntimeKind::Ollama | RuntimeKind::MlxLm)
}

/// Send the request that makes a lazily-loading runtime load the model, and
/// wait for it. `None` when the runtime needs no warm-up (llama.cpp: its
/// `/health` is 200 only once loaded); otherwise whether the request
/// succeeded. Ollama: `warm_up_ollama`. mlx: a one-token chat completion
/// (`mlx_lm.server` answers `/v1/models` before loading the model).
pub(crate) fn warm_up(probe: &dyn SystemProbe, plan: &ServePlan) -> Option<bool> {
    match plan.runtime {
        RuntimeKind::Ollama => Some(paddock_core::serving::warm_up_ollama(
            probe,
            &plan.model_ref,
        )),
        RuntimeKind::MlxLm => {
            let body = serde_json::json!({
                "model": plan.model_ref,
                "messages": [{ "role": "user", "content": "hi" }],
                "max_tokens": 1,
            })
            .to_string();
            Some(probe.http_post_local(&plan.openai_url, &body).is_some())
        }
        RuntimeKind::LlamaCpp => None,
    }
}

/// Build a serving registry record for a running server.
pub(crate) fn build_record(plan: &ServePlan, pid: u32, log_path: Option<PathBuf>) -> ServingRecord {
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
/// unregister on drop - the server must survive this process exiting.
pub(crate) fn register_detached(plan: &ServePlan, pid: u32, log_path: Option<PathBuf>) {
    let record = build_record(plan, pid, log_path);
    if let Err(e) = Registry::open_default().register(&record) {
        eprintln!("warning: could not record serving state: {e}");
    }
}

/// RAII wrapper around the serving registry for foreground children:
/// best-effort register on creation, unregister on drop.
pub struct RegistryGuard {
    registry: Registry,
    pid: u32,
}

impl RegistryGuard {
    pub fn register(plan: &ServePlan, pid: u32, log_path: Option<PathBuf>) -> Self {
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

/// Full serve lifecycle: install policy, port fallback, history record, spawn
/// (foreground or detached), readiness wait, pre-steps, warm-up (Ollama and
/// mlx load lazily), registry entry. Returns once the model is loaded. In
/// `Foreground` mode the child handle is returned for the caller to wait on;
/// in `Detached` mode the registry entry has already been written.
///
/// On a readiness, pre-step or warm-up timeout with a detached serve, the
/// child (if any) is left running and registered, and
/// `NotReady { pid, log_path, plan }` is returned so the caller can report
/// "starting". Every other failure kills the child.
pub fn serve(
    mut plan: ServePlan,
    mode: ServeMode,
    policy: InstallPolicy<'_>,
    ready_timeout: Option<Duration>,
    progress: &dyn Progress,
) -> Result<ServeOutcome, LifecycleError> {
    if plan.port_ignored {
        progress.note("warning: --port is ignored for the Ollama daemon (fixed 11434)");
    }
    if let Some(install) = &plan.install {
        match policy {
            InstallPolicy::Ask(confirm) => confirm(install)?,
            InstallPolicy::Refuse => {
                return Err(LifecycleError::NoRuntime {
                    install: install.clone(),
                });
            }
        }
    }

    // Spawned servers (llama.cpp/mlx) default to 8080; pick a free port so
    // concurrent servers don't collide (and so the readiness probe can't be
    // answered by a different process already on the port). The Ollama daemon
    // has a fixed port and no server_argv, so it's untouched.
    if plan.server_argv.is_some()
        && let Some(requested) = plan.port
        && let Some(free) = paddock_core::serving::free_port(requested)
        && free != requested
    {
        progress.note(&format!(
            "port {requested} is busy - serving on {free} instead"
        ));
        plan = plan.with_port(free);
    }

    // Remember spawned (llama.cpp/mlx) serves so the TUI can offer one-key
    // relaunch. Best-effort; Ollama is covered by /api/tags, not recorded.
    if plan.server_argv.is_some() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        History::open_default().record(&plan, now);
    }

    let log_dir = paddock_core::serving::default_serving_dir().join("logs");
    let mut log_path: Option<PathBuf> = None;
    let mut child = match &plan.server_argv {
        Some(argv) => {
            progress.command(argv);
            match mode {
                ServeMode::Foreground => Some(spawn_checked(argv)?),
                ServeMode::Detached => {
                    // Spawn to a per-invocation placeholder (our pid plus a
                    // process-wide counter makes it unique across concurrent
                    // serves, including concurrent MCP tool calls), then
                    // rename to <child-pid>.log once the child pid is known.
                    let tmp_log = log_dir.join(pending_log_name());
                    let c = spawn_detached(argv, &tmp_log)?;
                    let final_log = log_dir.join(format!("{}.log", c.id()));
                    let actual = match std::fs::rename(&tmp_log, &final_log) {
                        Ok(()) => final_log,
                        Err(_) => tmp_log, // rename failed: the data is still at the pending path
                    };
                    log_path = Some(actual);
                    Some(c)
                }
            }
        }
        None => None,
    };

    // Readiness + pre-steps; never leave an orphaned server behind on failure,
    // except the detached-timeout case, which is the whole point of `NotReady`.
    let timeout = readiness_deadline(child.is_some(), ready_timeout);
    // Detached serves with a caller budget (MCP) bound the pre-steps and the
    // warm-up too, so a multi-GB `ollama pull` or model load cannot outlast
    // `ready_timeout`. The CLI (None) and foreground serves wait for both to
    // finish.
    let started = Instant::now();
    let prestep_deadline = match (mode, ready_timeout) {
        (ServeMode::Detached, Some(total)) => deadline_after(started, total),
        _ => None,
    };
    let server_pid = child.as_ref().map(|c| c.id());
    let prepared =
        wait_ready(&RealSystemProbe, &plan, child.as_mut(), timeout, progress).and_then(|()| {
            for step in &plan.pre_steps {
                progress.command(step);
                if let Some(pid) = run_checked_until(step, progress, prestep_deadline)? {
                    return Err(LifecycleError::NotReady {
                        pid: Some(pid),
                        log_path: None,
                        plan: Box::new(plan.clone()),
                    });
                }
            }
            warm_up_until(&plan, server_pid, prestep_deadline, progress)
        });
    match prepared {
        Ok(()) => {}
        Err(LifecycleError::NotReady { pid, .. }) if mode == ServeMode::Detached => {
            // Register the spawned server (if any), left running. `pid` is
            // that server (readiness or warm-up timeout; None for the Ollama
            // daemon) or a still-running pre-step.
            if plan.runtime != RuntimeKind::Ollama
                && let Some(server_pid) = server_pid
            {
                register_detached(&plan, server_pid, log_path.clone());
            }
            let log_path = if pid == server_pid { log_path } else { None };
            return Err(LifecycleError::NotReady {
                pid,
                log_path,
                plan: Box::new(plan),
            });
        }
        Err(e) => {
            if let Some(c) = child.as_mut() {
                let _ = c.kill();
                let _ = c.wait();
            }
            // `wait_ready` has no log path; point the caller at the detached log.
            return Err(match e {
                LifecycleError::ServerExited {
                    status,
                    argv,
                    log_path: None,
                } if log_path.is_some() => LifecycleError::ServerExited {
                    status,
                    argv,
                    log_path,
                },
                e => e,
            });
        }
    }

    let pid = child.as_ref().map(|c| c.id());
    match mode {
        ServeMode::Foreground => Ok(ServeOutcome {
            plan,
            pid,
            log_path: None,
            child,
        }),
        ServeMode::Detached => {
            // Register WITHOUT the drop-guard so the child outlives us. A
            // cold-started Ollama daemon is managed by ollama itself.
            if let Some(pid) = pid
                && plan.runtime != RuntimeKind::Ollama
            {
                register_detached(&plan, pid, log_path.clone());
            }
            Ok(ServeOutcome {
                plan,
                pid,
                log_path,
                child: None,
            })
        }
    }
}

/// Warm-up step of `serve`, after readiness and pre-steps. Ollama and mlx
/// load a model only on its first request (and the Ollama daemon outlives
/// us), so warm them up now: "ready" means loaded (and an Ollama model shows
/// in /api/ps + the tray). Best-effort: a failed request leaves a working
/// endpoint that simply cold-starts on first use. With a `deadline`
/// (detached MCP serves), an unfinished warm-up is `NotReady` for
/// `server_pid`; its request stays open in the background so the load
/// completes.
fn warm_up_until(
    plan: &ServePlan,
    server_pid: Option<u32>,
    deadline: Option<Instant>,
    progress: &dyn Progress,
) -> Result<(), LifecycleError> {
    if !needs_warm_up(plan.runtime) {
        return Ok(());
    }
    progress.note(&format!("loading {} into memory…", plan.model_ref));
    let request = plan.clone();
    match run_with_deadline(deadline, move || warm_up(&RealSystemProbe, &request)) {
        None => Err(LifecycleError::NotReady {
            pid: server_pid,
            log_path: None,
            plan: Box::new(plan.clone()),
        }),
        Some(Some(false)) => {
            progress.note("warning: warm-up failed - the model will load on the first request");
            Ok(())
        }
        Some(_) => Ok(()),
    }
}

/// Resolve a stop target (model name substring, pid, or `all`) against the
/// live registry. `all` on an empty registry is `NoServerMatch`.
pub fn resolve_servers(target: &str) -> Result<Vec<ServingRecord>, LifecycleError> {
    use paddock_core::serving::{RecordMatch, match_records};

    let records = Registry::open_default().list_live(&RealSystemProbe);
    match match_records(&records, target) {
        RecordMatch::Matched(v) => Ok(v.into_iter().cloned().collect()),
        RecordMatch::Ambiguous(cands) => Err(LifecycleError::AmbiguousServer {
            target: target.to_string(),
            candidates: cands.iter().map(|r| (r.model_ref.clone(), r.pid)).collect(),
        }),
        RecordMatch::NotFound => Err(LifecycleError::NoServerMatch {
            target: target.to_string(),
            running: records.iter().map(|r| r.model_ref.clone()).collect(),
        }),
    }
}

/// Stop every record: `ollama stop` for Ollama-served models, SIGTERM for
/// paddock-spawned servers; unregister each. Best-effort per record.
pub fn stop_records(records: Vec<ServingRecord>, progress: &dyn Progress) -> Vec<Stopped> {
    use paddock_core::serving::terminate;

    let registry = Registry::open_default();
    let mut stopped = Vec::with_capacity(records.len());
    for r in records {
        if r.runtime == RuntimeKind::Ollama {
            let _ = run_checked(
                &["ollama".into(), "stop".into(), r.model_ref.clone()],
                progress,
            );
        } else {
            terminate(r.pid);
        }
        let _ = registry.unregister(r.pid);
        stopped.push(Stopped {
            model_ref: r.model_ref,
            pid: r.pid,
            runtime: r.runtime,
        });
    }
    stopped
}

/// `resolve_servers` + `stop_records`, for callers that need no confirmation
/// step in between (MCP rejects `all` before calling this).
pub fn stop(target: &str, progress: &dyn Progress) -> Result<Vec<Stopped>, LifecycleError> {
    let records = resolve_servers(target)?;
    Ok(stop_records(records, progress))
}

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_core::catalog::{CatalogVariant, RuntimeKind, Source};
    use paddock_core::estimate::ModelVariant;

    fn model(name: &str) -> CatalogModel {
        CatalogModel {
            id: 0,
            name: name.to_string(),
            family: None,
            source: Source::HuggingFace,
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

    /// One-variant Ollama model small enough to fit any budget used below.
    fn small_ollama_model(name: &str) -> CatalogModel {
        let mut m = model(name);
        m.source = Source::Ollama;
        m.params_total = 1_000_000_000;
        m.params_active = 1_000_000_000;
        m.variants = vec![CatalogVariant {
            quant: "Q4_K_M".into(),
            bpw: 4.83,
            file_size_bytes: None,
            layers: 16,
            kv_heads: 8,
            head_dim: 64,
            embedding_dim: 2048,
            runtime_compat: vec![RuntimeKind::Ollama],
            source_tag: None,
        }];
        m
    }

    fn budget_gib(gib: u64) -> MemoryBudget {
        let bytes = gib * 1024 * 1024 * 1024;
        MemoryBudget {
            gpu_effective_bytes: bytes,
            ram_total_bytes: bytes,
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
            _ => panic!("expected case-insensitive exact match"),
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

    fn mv(quant: &str, bpw: f64) -> ModelVariant {
        ModelVariant {
            model_name: "test".into(),
            quant: quant.into(),
            bpw,
            params_total: 8_000_000_000,
            params_active: 8_000_000_000,
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
            embedding_dim: 4096,
            context_max: 8192,
        }
    }

    #[test]
    fn resolve_quant_exact_and_case_insensitive() {
        let vs = vec![mv("Q8_0", 8.5), mv("Q4_K_M", 4.83), mv("Q2_K", 3.35)];
        assert_eq!(resolve_quant(&vs, "Q4_K_M").unwrap(), 1);
        assert_eq!(resolve_quant(&vs, "q4_k_m").unwrap(), 1);
    }

    #[test]
    fn resolve_quant_collision_picks_best_quality() {
        let vs = vec![mv("Q4_K_M", 4.5), mv("Q4_K_M", 5.0)];
        assert_eq!(resolve_quant(&vs, "Q4_K_M").unwrap(), 1);
    }

    #[test]
    fn resolve_quant_no_match_lists_available() {
        let vs = vec![mv("Q8_0", 8.5), mv("Q4_K_M", 4.83)];
        let err = resolve_quant(&vs, "Q3_K_M").unwrap_err();
        match &err {
            LifecycleError::UnknownQuant { label, available } => {
                assert_eq!(label, "Q3_K_M");
                assert_eq!(available, &["Q8_0", "Q4_K_M"]);
            }
            other => panic!("expected UnknownQuant, got {other}"),
        }
        assert_eq!(
            err.to_string(),
            "no quant `Q3_K_M` for this model; available: Q8_0, Q4_K_M"
        );
    }

    #[test]
    fn resolve_in_empty_catalog_is_catalog_empty() {
        let err = resolve_in(&[], &budget_gib(16), "anything", None).unwrap_err();
        assert!(matches!(err, LifecycleError::CatalogEmpty));
        assert_eq!(
            err.to_string(),
            "catalog is empty - run `paddock sync` first"
        );
    }

    #[test]
    fn resolve_in_unknown_model_is_model_not_found() {
        let models = vec![small_ollama_model("fake-model")];
        let err = resolve_in(&models, &budget_gib(16), "nope", None).unwrap_err();
        assert!(matches!(err, LifecycleError::ModelNotFound(ref q) if q == "nope"));
        assert!(err.to_string().contains("paddock sync"));
    }

    #[test]
    fn resolve_in_ambiguous_lists_candidates() {
        let models = vec![
            small_ollama_model("llama3-8b"),
            small_ollama_model("llama3-70b"),
        ];
        let err = resolve_in(&models, &budget_gib(16), "llama3", None).unwrap_err();
        match &err {
            LifecycleError::Ambiguous { query, candidates } => {
                assert_eq!(query, "llama3");
                assert_eq!(candidates, &["llama3-8b", "llama3-70b"]);
            }
            other => panic!("expected Ambiguous, got {other}"),
        }
        assert_eq!(
            err.to_string(),
            "model name `llama3` is ambiguous - candidates:\n  llama3-8b\n  llama3-70b"
        );
    }

    #[test]
    fn resolve_in_picks_best_fitting_variant() {
        let models = vec![small_ollama_model("fake-model")];
        let (m, idx) = resolve_in(&models, &budget_gib(16), "fake-model", None).unwrap();
        assert_eq!(m.name, "fake-model");
        assert_eq!(idx, 0);
    }

    #[test]
    fn resolve_in_no_fit_reports_ram() {
        // 1 GiB budget cannot hold a 1B-parameter Q4 model plus its KV cache.
        let models = vec![small_ollama_model("fake-model")];
        let err = resolve_in(&models, &budget_gib(1), "fake-model", None).unwrap_err();
        match &err {
            LifecycleError::NoFit { model, ram_bytes } => {
                assert_eq!(model, "fake-model");
                assert_eq!(*ram_bytes, 1024 * 1024 * 1024);
            }
            other => panic!("expected NoFit, got {other}"),
        }
        assert_eq!(
            err.to_string(),
            "no quantization of `fake-model` fits this machine (1.0 GiB RAM); try a smaller model from `paddock fit`"
        );
    }

    #[test]
    fn explicit_quant_bypasses_fit_check() {
        let models = vec![small_ollama_model("fake-model")];
        let (_, idx) = resolve_in(&models, &budget_gib(1), "fake-model", Some("q4_k_m")).unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn error_display_matches_cli_wording() {
        let e = LifecycleError::NoServerMatch {
            target: "nope".into(),
            running: vec![],
        };
        assert_eq!(e.to_string(), "no running server matches `nope`");

        let e = LifecycleError::NoServerMatch {
            target: "nope".into(),
            running: vec!["a".into(), "b".into()],
        };
        assert_eq!(
            e.to_string(),
            "no running server matches `nope`\nrunning: a, b"
        );

        let e = LifecycleError::AmbiguousServer {
            target: "qwen".into(),
            candidates: vec![("qwen3-8b".into(), 1), ("qwen3-4b".into(), 2)],
        };
        assert_eq!(
            e.to_string(),
            "`qwen` matches several servers - be specific:\n  qwen3-8b (pid 1)\n  qwen3-4b (pid 2)"
        );

        let e = LifecycleError::OllamaUnreachable;
        assert_eq!(
            e.to_string(),
            "ollama daemon not reachable on 11434 - is it running?"
        );

        let e = LifecycleError::ServerExited {
            status: "exit status: 1".into(),
            argv: vec!["llama-server".into(), "-hf".into(), "x".into()],
            log_path: None,
        };
        assert_eq!(
            e.to_string(),
            "server exited with exit status: 1 before becoming ready - run `llama-server -hf x` manually to see the error"
        );

        let e = LifecycleError::NoRuntime {
            install: InstallPlan {
                kind: RuntimeKind::LlamaCpp,
                argv: vec!["brew".into(), "install".into(), "llama.cpp".into()],
            },
        };
        assert_eq!(
            e.to_string(),
            "required runtime is not installed - run `brew install llama.cpp` then retry"
        );

        let e = LifecycleError::InstallDeclined {
            command: "brew install llama.cpp".into(),
        };
        assert_eq!(
            e.to_string(),
            "install declined - nothing launched. Run `brew install llama.cpp` yourself, then retry."
        );
    }

    use paddock_core::hardware::{MockProbe, SystemProbe};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Probe that answers the readiness GET only from the Nth call on.
    struct CountingProbe {
        calls: AtomicUsize,
        answer_from: usize,
    }

    impl SystemProbe for CountingProbe {
        fn sysctl_string(&self, _: &str) -> Option<String> {
            None
        }
        fn sysctl_u64(&self, _: &str) -> Option<u64> {
            None
        }
        fn gpu_recommended_working_set(&self) -> Option<u64> {
            None
        }
        fn which(&self, _: &str) -> Option<String> {
            None
        }
        fn run_command(&self, _: &str, _: &[&str]) -> Option<String> {
            None
        }
        fn http_get_local(&self, _: &str) -> Option<String> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            (n >= self.answer_from).then(|| "ok".to_string())
        }
        fn http_post_local(&self, _: &str, _: &str) -> Option<String> {
            None
        }
    }

    fn sleeping_child() -> std::process::Child {
        std::process::Command::new("sleep")
            .arg("30")
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("sleep is available")
    }

    fn spawned_plan() -> ServePlan {
        ServePlan {
            server_argv: Some(vec!["llama-server".into(), "--port".into(), "8080".into()]),
            pre_steps: vec![],
            endpoint: "http://127.0.0.1:8080".into(),
            openai_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
            model_ref: "x".into(),
            ready_path: "/health".into(),
            install: None,
            port_ignored: false,
            runtime: RuntimeKind::LlamaCpp,
            ctx: 4096,
            port: Some(8080),
        }
    }

    #[test]
    fn spawned_child_waits_without_deadline_by_default() {
        // First-run model downloads can take tens of minutes; the CLI passes
        // None and any fixed cap would kill a healthy child mid-download.
        assert_eq!(readiness_deadline(true, None), None);
        assert_eq!(
            readiness_deadline(true, Some(Duration::from_secs(600))),
            Some(Duration::from_secs(600))
        );
    }

    #[test]
    fn daemon_probe_keeps_short_deadline() {
        assert_eq!(
            readiness_deadline(false, None),
            Some(Duration::from_secs(3))
        );
        assert_eq!(
            readiness_deadline(false, Some(Duration::from_secs(600))),
            Some(Duration::from_secs(3))
        );
    }

    #[test]
    fn wait_ready_timeout_leaves_child_alive() {
        let probe = MockProbe::default(); // never answers
        let mut child = sleeping_child();
        let pid = child.id();
        let err = wait_ready(
            &probe,
            &spawned_plan(),
            Some(&mut child),
            Some(Duration::from_millis(300)),
            &SilentProgress,
        )
        .unwrap_err();
        match err {
            LifecycleError::NotReady {
                pid: Some(p),
                log_path: None,
                ..
            } => assert_eq!(p, pid),
            other => panic!("expected NotReady, got {other}"),
        }
        assert!(
            child.try_wait().unwrap().is_none(),
            "child must still be running"
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn wait_ready_returns_once_probe_answers() {
        let probe = CountingProbe {
            calls: AtomicUsize::new(0),
            answer_from: 3,
        };
        let mut child = sleeping_child();
        wait_ready(
            &probe,
            &spawned_plan(),
            Some(&mut child),
            Some(Duration::from_secs(10)),
            &SilentProgress,
        )
        .unwrap();
        assert_eq!(probe.calls.load(Ordering::SeqCst), 3);
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn wait_ready_daemon_timeout_is_ollama_unreachable() {
        let probe = MockProbe::default();
        let err = wait_ready(
            &probe,
            &spawned_plan(),
            None,
            Some(Duration::ZERO),
            &SilentProgress,
        )
        .unwrap_err();
        assert!(matches!(err, LifecycleError::OllamaUnreachable));
    }

    #[test]
    fn wait_ready_child_exit_is_server_exited() {
        let probe = MockProbe::default();
        let mut child = std::process::Command::new("false").spawn().unwrap();
        let err = wait_ready(
            &probe,
            &spawned_plan(),
            Some(&mut child),
            Some(Duration::from_secs(5)),
            &SilentProgress,
        )
        .unwrap_err();
        match err {
            LifecycleError::ServerExited { argv, .. } => {
                assert_eq!(argv[0], "llama-server");
            }
            other => panic!("expected ServerExited, got {other}"),
        }
    }

    #[test]
    fn run_checked_quiet_does_not_fail_on_success() {
        run_checked(&["true".into()], &SilentProgress).unwrap();
        let err = run_checked(&["false".into()], &SilentProgress).unwrap_err();
        assert!(err.to_string().contains("`false` failed"));
    }

    #[test]
    fn run_checked_until_returns_pid_of_step_still_running_at_deadline() {
        let deadline = Instant::now() + Duration::from_millis(300);
        let pid = run_checked_until(
            &["sleep".into(), "30".into()],
            &SilentProgress,
            Some(deadline),
        )
        .unwrap()
        .expect("sleep 30 is still running at the deadline");
        // Left running, not killed.
        let alive = std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .unwrap();
        assert!(alive.success(), "pre-step {pid} should still be alive");
        let _ = std::process::Command::new("kill")
            .arg(pid.to_string())
            .status();
    }

    #[test]
    fn run_checked_until_with_deadline_reports_success_and_failure() {
        let deadline = || Some(Instant::now() + Duration::from_secs(5));
        assert_eq!(
            run_checked_until(&["true".into()], &SilentProgress, deadline()).unwrap(),
            None
        );
        let err = run_checked_until(&["false".into()], &SilentProgress, deadline()).unwrap_err();
        assert!(err.to_string().contains("`false` failed"), "{err}");
    }

    #[test]
    fn other_error_source_does_not_duplicate_top_line() {
        // `#[error(transparent)]`: Display and source() both delegate to the
        // inner anyhow error, so the top line is not printed twice in a
        // `Caused by:` chain.
        let e = LifecycleError::Other(anyhow::anyhow!("x").context("top"));
        assert_eq!(e.to_string(), "top");
        let chain: Vec<String> = anyhow::Error::from(e)
            .chain()
            .map(|c| c.to_string())
            .collect();
        assert_eq!(chain, vec!["top".to_string(), "x".to_string()]);

        let e = LifecycleError::Other(anyhow::anyhow!("x").context("top"));
        let src = std::error::Error::source(&e).map(|s| s.to_string());
        assert_ne!(src.as_deref(), Some("top"));
    }

    #[test]
    fn serve_refuse_policy_returns_no_runtime_before_spawning() {
        let dir = tempfile::tempdir().unwrap();
        let _env = ServingDirGuard::isolate(dir.path());
        let mut plan = spawned_plan();
        plan.server_argv = Some(vec!["definitely-not-a-binary-xyz".into()]);
        plan.install = Some(InstallPlan {
            kind: RuntimeKind::LlamaCpp,
            argv: vec!["brew".into(), "install".into(), "llama.cpp".into()],
        });
        let err = serve(
            plan,
            ServeMode::Detached,
            InstallPolicy::Refuse,
            Some(Duration::from_secs(1)),
            &SilentProgress,
        )
        .unwrap_err();
        match err {
            LifecycleError::NoRuntime { install } => assert_eq!(install.argv[0], "brew"),
            other => panic!("expected NoRuntime, got {other}"),
        }
    }

    #[test]
    fn serve_ask_policy_propagates_install_declined() {
        let dir = tempfile::tempdir().unwrap();
        let _env = ServingDirGuard::isolate(dir.path());
        let mut plan = spawned_plan();
        plan.server_argv = Some(vec!["definitely-not-a-binary-xyz".into()]);
        plan.install = Some(InstallPlan {
            kind: RuntimeKind::LlamaCpp,
            argv: vec!["brew".into(), "install".into(), "llama.cpp".into()],
        });
        let decline = |i: &InstallPlan| {
            Err(LifecycleError::InstallDeclined {
                command: i.argv.join(" "),
            })
        };
        let err = serve(
            plan,
            ServeMode::Detached,
            InstallPolicy::Ask(&decline),
            None,
            &SilentProgress,
        )
        .unwrap_err();
        assert!(matches!(err, LifecycleError::InstallDeclined { .. }));
    }

    /// Serializes every test that touches `PADDOCK_SERVING_DIR`: cargo runs
    /// tests on parallel threads and the env is process-wide.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Points `PADDOCK_SERVING_DIR` at an isolated dir while holding
    /// `ENV_LOCK`; restores the previous value on drop (even on panic), then
    /// releases the lock.
    struct ServingDirGuard {
        prev: Option<std::ffi::OsString>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl ServingDirGuard {
        fn isolate(dir: &Path) -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let prev = std::env::var_os("PADDOCK_SERVING_DIR");
            // SAFETY: every writer of PADDOCK_SERVING_DIR in this process goes
            // through this guard and holds ENV_LOCK.
            unsafe { std::env::set_var("PADDOCK_SERVING_DIR", dir) };
            Self { prev, _lock: lock }
        }
    }

    impl Drop for ServingDirGuard {
        fn drop(&mut self) {
            // SAFETY: ENV_LOCK is still held (fields drop after this body).
            unsafe {
                match self.prev.take() {
                    Some(v) => std::env::set_var("PADDOCK_SERVING_DIR", v),
                    None => std::env::remove_var("PADDOCK_SERVING_DIR"),
                }
            }
        }
    }

    /// True while `pid` exists (signal 0 probe).
    fn pid_alive(pid: u32) -> bool {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[test]
    fn serve_missing_binary_is_actionable() {
        // Isolated serving dir so the history/registry writes never touch the
        // real one.
        let dir = tempfile::tempdir().unwrap();
        let _env = ServingDirGuard::isolate(dir.path());
        let mut plan = spawned_plan();
        plan.server_argv = Some(vec![
            "definitely-not-a-binary-xyz".into(),
            "--port".into(),
            "8080".into(),
        ]);
        let err = serve(
            plan,
            ServeMode::Detached,
            InstallPolicy::Refuse,
            Some(Duration::from_secs(1)),
            &SilentProgress,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("failed to start definitely-not-a-binary-xyz"),
            "got: {err}"
        );
    }

    #[test]
    fn serve_detached_timeout_leaves_child_running_and_registered() {
        let dir = tempfile::tempdir().unwrap();
        let _env = ServingDirGuard::isolate(dir.path());
        let mut plan = spawned_plan();
        plan.server_argv = Some(vec!["sleep".into(), "30".into()]);
        let err = serve(
            plan,
            ServeMode::Detached,
            InstallPolicy::Refuse,
            Some(Duration::from_secs(1)),
            &SilentProgress,
        )
        .unwrap_err();
        let pid = match err {
            LifecycleError::NotReady { pid: Some(pid), .. } => pid,
            other => panic!("expected NotReady with a pid, got {other}"),
        };
        let alive = pid_alive(pid);
        let registered = dir.path().join(format!("{pid}.json")).exists();
        paddock_core::serving::terminate(pid);
        assert!(alive, "detached child must keep running after NotReady");
        assert!(registered, "detached child must be registered on NotReady");
    }

    #[test]
    fn serve_detached_early_exit_reports_log_path() {
        let dir = tempfile::tempdir().unwrap();
        let _env = ServingDirGuard::isolate(dir.path());
        let mut plan = spawned_plan();
        plan.server_argv = Some(vec!["sh".into(), "-c".into(), "exit 1".into()]);
        let err = serve(
            plan,
            ServeMode::Detached,
            InstallPolicy::Refuse,
            Some(Duration::from_secs(5)),
            &SilentProgress,
        )
        .unwrap_err();
        match err {
            LifecycleError::ServerExited {
                log_path: Some(p), ..
            } => assert!(p.exists(), "log {p:?} must exist"),
            other => panic!("expected ServerExited with a log_path, got {other}"),
        }
    }

    #[test]
    fn run_with_deadline_returns_result_when_it_completes() {
        let deadline = Instant::now() + Duration::from_secs(5);
        assert_eq!(run_with_deadline(Some(deadline), || 7), Some(7));
    }

    #[test]
    fn run_with_deadline_returns_none_quickly_on_expiry() {
        let deadline = Instant::now() + Duration::from_millis(200);
        let t0 = Instant::now();
        let got = run_with_deadline(Some(deadline), || {
            std::thread::sleep(Duration::from_secs(5));
            1
        });
        assert_eq!(got, None);
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "must return at the deadline, took {:?}",
            t0.elapsed()
        );
    }

    #[test]
    fn run_with_deadline_past_deadline_is_none() {
        let past = Instant::now();
        std::thread::sleep(Duration::from_millis(10));
        let got = run_with_deadline(Some(past), || {
            std::thread::sleep(Duration::from_millis(500));
            1
        });
        assert_eq!(got, None);
    }

    #[test]
    fn run_with_deadline_without_deadline_runs_inline() {
        let caller = std::thread::current().id();
        let got = run_with_deadline(None, move || std::thread::current().id() == caller);
        assert_eq!(got, Some(true));
    }

    fn mlx_plan() -> ServePlan {
        ServePlan {
            server_argv: Some(vec![
                "mlx_lm.server".into(),
                "--model".into(),
                "mlx-community/Qwen2.5-0.5B-Instruct-4bit".into(),
                "--port".into(),
                "8080".into(),
            ]),
            pre_steps: vec![],
            endpoint: "http://127.0.0.1:8080".into(),
            openai_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
            model_ref: "mlx-community/Qwen2.5-0.5B-Instruct-4bit".into(),
            ready_path: "/v1/models".into(),
            install: None,
            port_ignored: false,
            runtime: RuntimeKind::MlxLm,
            ctx: 0,
            port: Some(8080),
        }
    }

    #[test]
    fn warm_up_mlx_posts_one_token_completion_to_openai_url() {
        let plan = mlx_plan();
        let mut probe = MockProbe::default();
        probe
            .posts
            .insert(plan.openai_url.clone(), "{\"choices\":[]}".into());
        assert_eq!(warm_up(&probe, &plan), Some(true));
        let sent = probe.post_bodies.lock().unwrap().clone();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, "http://127.0.0.1:8080/v1/chat/completions");
        let body: serde_json::Value = serde_json::from_str(&sent[0].1).unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "messages": [{ "role": "user", "content": "hi" }],
                "max_tokens": 1,
            })
        );
        // No response (connection refused, non-2xx, read timeout) = failed.
        assert_eq!(warm_up(&MockProbe::default(), &plan), Some(false));
        assert!(needs_warm_up(RuntimeKind::MlxLm));
    }

    #[test]
    fn warm_up_ollama_goes_to_generate_endpoint() {
        let mut plan = mlx_plan();
        plan.runtime = RuntimeKind::Ollama;
        plan.server_argv = None;
        plan.model_ref = "qwen3:8b".into();
        let mut probe = MockProbe::default();
        probe
            .posts
            .insert("http://127.0.0.1:11434/api/generate".into(), "{}".into());
        assert_eq!(warm_up(&probe, &plan), Some(true));
        let sent = probe.post_bodies.lock().unwrap().clone();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, "http://127.0.0.1:11434/api/generate");
        let body: serde_json::Value = serde_json::from_str(&sent[0].1).unwrap();
        assert_eq!(body["model"], "qwen3:8b");
        assert!(needs_warm_up(RuntimeKind::Ollama));
    }

    #[test]
    fn warm_up_llama_cpp_does_nothing() {
        let probe = MockProbe::default();
        assert_eq!(warm_up(&probe, &spawned_plan()), None);
        assert!(probe.post_bodies.lock().unwrap().is_empty());
        assert!(!needs_warm_up(RuntimeKind::LlamaCpp));
    }

    /// A stand-in for `mlx_lm.server` (python3 stdlib, no model): answers
    /// every GET 200 at once (readiness), and the POST warm-up either at once
    /// (`fast`) or after 60 s (`hang`, a model still loading).
    fn fake_mlx_argv(post: &str) -> Vec<String> {
        const SCRIPT: &str = r#"
import sys, time, http.server
mode = sys.argv[3]
class H(http.server.BaseHTTPRequestHandler):
    def reply(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")
    def do_GET(self):
        self.reply()
    def do_POST(self):
        if mode == "hang":
            time.sleep(60)
        self.reply()
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[2])), H).serve_forever()
"#;
        vec![
            "python3".into(),
            "-c".into(),
            SCRIPT.into(),
            "--port".into(),
            "8080".into(),
            post.into(),
        ]
    }

    fn fake_mlx_plan(post: &str) -> ServePlan {
        // Start the free-port search on an ephemeral port so parallel test
        // runs do not race for 8080.
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut plan = mlx_plan();
        plan.server_argv = Some(fake_mlx_argv(post));
        plan.with_port(port)
    }

    #[test]
    fn serve_detached_mlx_waits_for_warm_up_then_is_ready() {
        let dir = tempfile::tempdir().unwrap();
        let _env = ServingDirGuard::isolate(dir.path());
        let outcome = serve(
            fake_mlx_plan("fast"),
            ServeMode::Detached,
            InstallPolicy::Refuse,
            Some(Duration::from_secs(20)),
            &SilentProgress,
        )
        .expect("fake mlx server answers readiness and warm-up");
        let pid = outcome.pid.expect("spawned child");
        let registered = dir.path().join(format!("{pid}.json")).exists();
        paddock_core::serving::terminate(pid);
        assert!(registered, "ready mlx server must be registered");
    }

    #[test]
    fn serve_detached_mlx_warm_up_past_timeout_is_not_ready_and_registered() {
        let dir = tempfile::tempdir().unwrap();
        let _env = ServingDirGuard::isolate(dir.path());
        let t0 = Instant::now();
        let result = serve(
            fake_mlx_plan("hang"),
            ServeMode::Detached,
            InstallPolicy::Refuse,
            Some(Duration::from_secs(4)),
            &SilentProgress,
        );
        let err = match result {
            Err(e) => e,
            Ok(outcome) => {
                // Never leak the fake server when the assertion fails.
                if let Some(pid) = outcome.pid {
                    paddock_core::serving::terminate(pid);
                }
                panic!("expected NotReady, got ready (warm-up not awaited)");
            }
        };
        let elapsed = t0.elapsed();
        let (pid, log_path) = match err {
            LifecycleError::NotReady {
                pid: Some(pid),
                log_path,
                ..
            } => (pid, log_path),
            other => panic!("expected NotReady with the server pid, got {other}"),
        };
        let alive = pid_alive(pid);
        let registered = dir.path().join(format!("{pid}.json")).exists();
        paddock_core::serving::terminate(pid);
        assert!(
            elapsed < Duration::from_secs(8),
            "warm-up must be bounded by timeout, took {elapsed:?}"
        );
        assert!(alive, "server must keep running after a warm-up timeout");
        assert!(
            registered,
            "server must be registered after a warm-up timeout"
        );
        let log_path = log_path.expect("server log path");
        assert!(log_path.ends_with(format!("{pid}.log")), "{log_path:?}");
    }

    #[test]
    fn pending_log_names_are_unique_per_call() {
        let a = pending_log_name();
        let b = pending_log_name();
        assert_ne!(a, b);
        assert!(a.starts_with(&format!("pending-{}-", std::process::id())));
    }

    #[test]
    fn overflowing_deadline_is_unbounded_and_does_not_panic() {
        let now = Instant::now();
        assert_eq!(deadline_after(now, Duration::from_secs(u64::MAX)), None);
        assert!(deadline_after(now, Duration::from_secs(1)).is_some());
        let argv = vec!["true".to_string()];
        let deadline = deadline_after(now, Duration::from_secs(u64::MAX));
        assert_eq!(
            run_checked_until(&argv, &SilentProgress, deadline).unwrap(),
            None
        );
    }

    #[test]
    fn serve_foreground_timeout_kills_child() {
        let dir = tempfile::tempdir().unwrap();
        let _env = ServingDirGuard::isolate(dir.path());
        let mut plan = spawned_plan();
        plan.server_argv = Some(vec!["sleep".into(), "30".into()]);
        let err = serve(
            plan,
            ServeMode::Foreground,
            InstallPolicy::Refuse,
            Some(Duration::from_secs(1)),
            &SilentProgress,
        )
        .unwrap_err();
        let pid = match err {
            LifecycleError::NotReady { pid: Some(pid), .. } => pid,
            other => panic!("expected NotReady with a pid, got {other}"),
        };
        let deadline = Instant::now() + Duration::from_secs(2);
        while pid_alive(pid) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        let alive = pid_alive(pid);
        if alive {
            paddock_core::serving::terminate(pid);
        }
        assert!(!alive, "foreground child must be killed on NotReady");
    }

    #[test]
    fn resolve_servers_unknown_target_lists_running() {
        let dir = tempfile::tempdir().unwrap();
        let _env = ServingDirGuard::isolate(dir.path());
        match resolve_servers("nope").unwrap_err() {
            LifecycleError::NoServerMatch { target, running } => {
                assert_eq!(target, "nope");
                assert!(running.is_empty());
            }
            other => panic!("expected NoServerMatch, got {other}"),
        }
    }

    #[test]
    fn stop_records_terminates_and_unregisters() {
        let dir = tempfile::tempdir().unwrap();
        let _env = ServingDirGuard::isolate(dir.path());
        let mut child = sleeping_child();
        let pid = child.id();
        let record = build_record(&spawned_plan(), pid, None);
        Registry::open_default().register(&record).unwrap();
        let json = dir.path().join(format!("{pid}.json"));
        assert!(json.exists());

        let stopped = stop_records(vec![record], &SilentProgress);
        assert_eq!(
            stopped,
            vec![Stopped {
                model_ref: "x".into(),
                pid,
                runtime: spawned_plan().runtime,
            }]
        );
        assert!(!json.exists(), "record must be unregistered");
        // SIGTERM delivered: the child exits (reaped so the pid is not reused).
        let status = child.wait().unwrap();
        assert!(!status.success(), "child must die from SIGTERM");
    }
}
