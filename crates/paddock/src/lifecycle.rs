//! Serve / stop lifecycle shared by the CLI, the TUI and the MCP server.
//! Everything here returns values: no printing, no stdin, no process::exit.
//! Adapters render `LifecycleError` and report `Progress` their own way.

use std::path::PathBuf;

use paddock_core::catalog::CatalogModel;
use paddock_core::estimate::{MemoryBudget, ModelVariant};
use paddock_core::runtime::{InstallPlan, ServePlan};
use paddock_core::score::best_variant;

use crate::app::App;

/// How the lifecycle reports what it is doing. The CLI/TUI print to stderr
/// (byte-for-byte today's messages); the MCP server discards everything
/// because stdout is the protocol channel and stderr is for internal errors.
#[allow(dead_code)] // wired up when serve/stop move here (later tasks)
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

#[allow(dead_code)] // wired up when serve/stop move here (later tasks)
pub struct StderrProgress;

impl Progress for StderrProgress {
    fn note(&self, msg: &str) {
        eprintln!("{msg}");
    }
    fn command(&self, argv: &[String]) {
        eprintln!("$ {}", argv.join(" "));
    }
}

#[allow(dead_code)] // wired up when serve/stop move here (later tasks)
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
#[allow(dead_code)] // wired up when serve/stop move here (later tasks)
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
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl From<paddock_core::PaddockError> for LifecycleError {
    fn from(e: paddock_core::PaddockError) -> Self {
        LifecycleError::Other(e.into())
    }
}

/// What to do when the plan needs a runtime that is not installed.
#[allow(dead_code)] // wired up when serve/stop move here (later tasks)
pub enum InstallPolicy<'a> {
    /// Ask the human (CLI/TUI): the callback prompts and installs, or returns
    /// `InstallDeclined`.
    Ask(&'a dyn Fn(&InstallPlan) -> Result<(), LifecycleError>),
    /// Never install (MCP): return `NoRuntime` before touching anything.
    Refuse,
}

#[allow(dead_code)] // wired up when serve/stop move here (later tasks)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeMode {
    /// Child inherits the tty; the caller waits on `ServeOutcome.child`.
    Foreground,
    /// Child is detached (own session, log file) and registered; it outlives
    /// this process.
    Detached,
}

#[allow(dead_code)] // wired up when serve/stop move here (later tasks)
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

#[allow(dead_code)] // wired up when serve/stop move here (later tasks)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stopped {
    pub model_ref: String,
    pub pid: u32,
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
}
