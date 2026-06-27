//! Builds the exact command to run a model - pure data, no exec, no prompt.
//! The binary owns display, confirmation and process replacement.

use serde::{Deserialize, Serialize};

use crate::catalog::{self, CatalogModel, CatalogVariant, RuntimeKind, Source};
use crate::error::PaddockError;
use crate::estimate::DEFAULT_CONTEXT;
use crate::hardware::RuntimesStatus;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallPlan {
    pub kind: RuntimeKind,
    pub argv: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunPlan {
    pub argv: Vec<String>,
    /// Set when the required runtime is missing; the binary MUST ask for
    /// explicit confirmation before executing it.
    pub install: Option<InstallPlan>,
}

impl RunPlan {
    pub fn display(&self) -> String {
        self.argv
            .iter()
            .map(|a| shell_quote(a))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Single-quote an argv element unless every char is shell-safe (POSIX style).
fn shell_quote(arg: &str) -> String {
    let safe = !arg.is_empty()
        && arg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._:/=@,+-".contains(c));
    if safe {
        arg.to_string()
    } else {
        format!("'{}'", arg.replace('\'', "'\\''"))
    }
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// Ollama library reference for a variant: `{base}:{source_tag}` when the
/// catalog carries the exact library tag (registry-enriched variants), else
/// the curated model name as-is (e.g. `llama3.1:8b`).
fn ollama_ref(model: &CatalogModel, variant: &CatalogVariant) -> String {
    match &variant.source_tag {
        Some(tag) => {
            let base = model.name.split(':').next().unwrap_or(&model.name);
            format!("{base}:{tag}")
        }
        None => model.name.clone(),
    }
}

/// Hybrid run strategy (see README): prefer Ollama for GGUF, mlx_lm for MLX.
/// mlx-lm chat command verified against ml-explore/mlx-lm README (2026-06).
pub fn plan_run(
    model: &CatalogModel,
    variant: &CatalogVariant,
    rt: &RuntimesStatus,
    ctx: Option<u32>,
) -> Result<RunPlan, PaddockError> {
    // `-c` size for llama.cpp paths; None falls back to the fit-verdict default.
    // Ollama/MLX are unaffected - they manage their own context window.
    let ctx = ctx.unwrap_or(DEFAULT_CONTEXT);
    match model.source {
        Source::Mlx => {
            let repo = model.repo.as_deref().ok_or_else(|| no_repo(&model.name))?;
            Ok(RunPlan {
                argv: s(&["mlx_lm.chat", "--model", repo]),
                install: (!rt.mlx.installed).then(|| InstallPlan {
                    kind: RuntimeKind::MlxLm,
                    argv: s(&["uv", "tool", "install", "mlx-lm"]),
                }),
            })
        }
        Source::Ollama => Ok(RunPlan {
            argv: s(&["ollama", "run", &ollama_ref(model, variant)]),
            install: (!rt.ollama.installed).then(ollama_install),
        }),
        Source::HuggingFace => {
            let repo = model.repo.as_deref().ok_or_else(|| no_repo(&model.name))?;
            validate_gguf_quant(model, variant)?;
            // Some repos cannot go through Ollama at all (e.g. separate
            // mmproj vision files, ollama/ollama#15447): llama.cpp only.
            if !variant.runtime_compat.contains(&RuntimeKind::Ollama) {
                let model_ref = format!("{repo}:{}", variant.quant);
                // `-c` aligns runtime memory with the fit verdict's context
                // assumption: llama-cli defaults --ctx-size to 0 = the model's
                // full context, which can be 262k → tens of GB of KV cache.
                // `--no-mmproj` skips the vision tower - text-only in v0.1,
                // and the estimator doesn't count vision weights (flag
                // verified present in llama.cpp b9580 for both binaries).
                return Ok(RunPlan {
                    argv: s(&[
                        "llama-cli",
                        "-hf",
                        &model_ref,
                        "-c",
                        &ctx.to_string(),
                        "--no-mmproj",
                    ]),
                    install: (!rt.llama_cpp.installed).then(llama_cpp_install),
                });
            }
            let model_ref = format!("hf.co/{repo}:{}", variant.quant);
            Ok(RunPlan {
                argv: s(&["ollama", "run", &model_ref]),
                install: (!rt.ollama.installed).then(ollama_install),
            })
        }
    }
}

fn no_repo(name: &str) -> PaddockError {
    PaddockError::Other(format!(
        "model `{name}` has no HuggingFace repo reference; re-run `paddock sync`"
    ))
}

/// A quant tag usable as `hf.co/{repo}:{quant}` / `-hf {repo}:{quant}` must be
/// a known GGUF tag (not an MLX pseudo-quant, not DB garbage).
fn validate_gguf_quant(model: &CatalogModel, variant: &CatalogVariant) -> Result<(), PaddockError> {
    if catalog::quant_bpw(&variant.quant).is_none() || variant.quant.starts_with("MLX_") {
        return Err(PaddockError::Other(format!(
            "variant `{}` of `{}` is not a GGUF quant tag usable by GGUF runtimes",
            variant.quant, model.name
        )));
    }
    Ok(())
}

/// Default port for llama-server / mlx_lm.server (Ollama's daemon port is fixed).
pub const DEFAULT_SERVE_PORT: u16 = 8080;
const OLLAMA_ENDPOINT: &str = "http://127.0.0.1:11434";

/// Everything the binary needs to start serving and print a usable endpoint.
/// Pure data: no spawn, no IO here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServePlan {
    /// Server process to spawn as a foreground child; None when an already
    /// running daemon (Ollama) will do the serving.
    pub server_argv: Option<Vec<String>>,
    /// Commands run to completion after readiness, before printing the
    /// endpoint (e.g. `ollama pull`). Each is a full argv.
    pub pre_steps: Vec<Vec<String>>,
    /// Base URL, e.g. http://127.0.0.1:11434
    pub endpoint: String,
    /// OpenAI-compatible chat completions URL.
    pub openai_url: String,
    /// Value to put in the `model` field of API payloads.
    pub model_ref: String,
    /// Path polled (GET, 2xx) to detect server readiness, e.g. "/api/version".
    pub ready_path: String,
    /// Same contract as RunPlan: the binary MUST confirm before installing.
    pub install: Option<InstallPlan>,
    /// True when caller requested a port but the plan uses Ollama's fixed
    /// daemon port (11434) instead.
    pub port_ignored: bool,
    /// Runtime this plan serves with (registry label, readiness policy).
    pub runtime: RuntimeKind,
    /// Resolved context window (llama.cpp); 0 when not applicable.
    pub ctx: u32,
    /// Bound port for spawned servers; None for the Ollama daemon.
    pub port: Option<u16>,
}

impl ServePlan {
    /// Rebind a spawned-server plan to a different local port: rewrites the
    /// `--port` argument, `endpoint`, `openai_url`, and `port`. Intended for
    /// llama.cpp/mlx plans (the Ollama daemon has a fixed port and no --port).
    pub fn with_port(mut self, port: u16) -> Self {
        if let Some(argv) = self.server_argv.as_mut()
            && let Some(i) = argv.iter().position(|a| a == "--port")
            && let Some(slot) = argv.get_mut(i + 1)
        {
            *slot = port.to_string();
        }
        self.endpoint = format!("http://127.0.0.1:{port}");
        self.openai_url = format!("http://127.0.0.1:{port}/v1/chat/completions");
        self.port = Some(port);
        self
    }
}

/// Serve strategy mirrors `plan_run`, plus a llama.cpp fallback for HF GGUF
/// (llama-server can download from HF directly via `-hf repo:quant`).
///
/// Facts verified 2026-06-10:
/// - `mlx_lm.server --model {repo} --port N`, default port 8080, serves
///   OpenAI-compatible `/v1/chat/completions` and `/v1/models`
///   (ml-explore/mlx-lm `mlx_lm/SERVER.md` and `server.py` argparse).
/// - `llama-server -hf <user>/<model>[:quant]` (alias of `--hf-repo`),
///   default port 8080, `GET /health` → 200 when ready / 503 while loading
///   (ggml-org/llama.cpp `tools/server/README.md`).
/// - Ollama daemon on 11434 exposes OpenAI-compat `POST /v1/chat/completions`
///   (docs.ollama.com/api/openai-compatibility) and `GET /api/version` as a
///   cheap readiness check (ollama/ollama `docs/api.md`).
pub fn plan_serve(
    model: &CatalogModel,
    variant: &CatalogVariant,
    rt: &RuntimesStatus,
    port: Option<u16>,
    ctx: Option<u32>,
) -> Result<ServePlan, PaddockError> {
    // llama-server `-c` size; None → fit-verdict default. Ollama/MLX ignore it.
    let ctx = ctx.unwrap_or(DEFAULT_CONTEXT);
    if port == Some(0) {
        return Err(PaddockError::Other(
            "port 0 is not supported; pick a fixed port so the endpoint is known upfront"
                .to_string(),
        ));
    }
    let port_requested = port.is_some();
    let port = port.unwrap_or(DEFAULT_SERVE_PORT);
    let local = format!("http://127.0.0.1:{port}");
    match model.source {
        Source::Mlx => {
            let repo = model.repo.clone().ok_or_else(|| no_repo(&model.name))?;
            Ok(ServePlan {
                server_argv: Some(s(&[
                    "mlx_lm.server",
                    "--model",
                    &repo,
                    "--port",
                    &port.to_string(),
                ])),
                pre_steps: vec![],
                endpoint: local.clone(),
                openai_url: format!("{local}/v1/chat/completions"),
                model_ref: repo,
                ready_path: "/v1/models".to_string(),
                install: (!rt.mlx.installed).then(|| InstallPlan {
                    kind: RuntimeKind::MlxLm,
                    argv: s(&["uv", "tool", "install", "mlx-lm"]),
                }),
                port_ignored: false,
                runtime: RuntimeKind::MlxLm,
                // mlx_lm.server is launched without a context flag, so the
                // resolved ctx is not applicable here (ServePlan.ctx doc: 0).
                ctx: 0,
                port: Some(port),
            })
        }
        Source::Ollama | Source::HuggingFace => {
            // repo:quant ref, validated once for the HF case.
            let mut hf_ref = if model.source == Source::HuggingFace {
                let repo = model.repo.as_deref().ok_or_else(|| no_repo(&model.name))?;
                validate_gguf_quant(model, variant)?;
                Some(format!("{repo}:{}", variant.quant))
            } else {
                None
            };
            // Variants Ollama cannot import (e.g. separate mmproj vision
            // files, ollama/ollama#15447) always go through llama-server,
            // proposing its install when missing.
            if model.source == Source::HuggingFace
                && !variant.runtime_compat.contains(&RuntimeKind::Ollama)
            {
                // Infallible: the HF arm above always sets `hf_ref`.
                let hf_ref = hf_ref.take().unwrap();
                let install = (!rt.llama_cpp.installed).then(llama_cpp_install);
                return Ok(llama_server_plan(hf_ref, port, ctx, &local, install));
            }
            // Prefer Ollama; fall back to llama-server for HF GGUF when only
            // llama.cpp is installed.
            if !rt.ollama.installed
                && rt.llama_cpp.installed
                && let Some(hf_ref) = hf_ref.take()
            {
                return Ok(llama_server_plan(hf_ref, port, ctx, &local, None));
            }
            let model_ref = match hf_ref {
                Some(hf_ref) => format!("hf.co/{hf_ref}"),
                None => ollama_ref(model, variant),
            };
            Ok(ServePlan {
                server_argv: (!rt.ollama.running).then(|| s(&["ollama", "serve"])),
                pre_steps: vec![vec![
                    "ollama".to_string(),
                    "pull".to_string(),
                    model_ref.clone(),
                ]],
                endpoint: OLLAMA_ENDPOINT.to_string(),
                openai_url: format!("{OLLAMA_ENDPOINT}/v1/chat/completions"),
                model_ref,
                ready_path: "/api/version".to_string(),
                install: (!rt.ollama.installed).then(ollama_install),
                port_ignored: port_requested,
                runtime: RuntimeKind::Ollama,
                ctx,
                port: None,
            })
        }
    }
}

fn ollama_install() -> InstallPlan {
    InstallPlan {
        kind: RuntimeKind::Ollama,
        argv: s(&["brew", "install", "ollama"]),
    }
}

fn llama_cpp_install() -> InstallPlan {
    InstallPlan {
        kind: RuntimeKind::LlamaCpp,
        argv: s(&["brew", "install", "llama.cpp"]),
    }
}

/// Foreground `llama-server -hf {repo}:{quant}` plan (HF GGUF).
fn llama_server_plan(
    hf_ref: String,
    port: u16,
    ctx: u32,
    local: &str,
    install: Option<InstallPlan>,
) -> ServePlan {
    ServePlan {
        // `-c` aligns runtime memory with the fit verdict's context
        // assumption: llama-server defaults --ctx-size to 0 = the model's
        // full context, which can be 262k → tens of GB of KV cache.
        // `--no-mmproj` skips the vision tower - text-only in v0.1, and the
        // estimator doesn't count vision weights (flag verified present in
        // llama.cpp b9580 for both binaries).
        server_argv: Some(s(&[
            "llama-server",
            "-hf",
            &hf_ref,
            "--port",
            &port.to_string(),
            "-c",
            &ctx.to_string(),
            "--no-mmproj",
        ])),
        pre_steps: vec![],
        endpoint: local.to_string(),
        openai_url: format!("{local}/v1/chat/completions"),
        model_ref: hf_ref,
        ready_path: "/health".to_string(),
        install,
        port_ignored: false,
        runtime: RuntimeKind::LlamaCpp,
        ctx,
        port: Some(port),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{CatalogModel, CatalogVariant, RuntimeKind, Source};
    use crate::hardware::{RuntimeStatus, RuntimesStatus};

    fn hf_model() -> CatalogModel {
        CatalogModel {
            id: 0,
            name: "Meta-Llama-3.1-8B-Instruct-GGUF".into(),
            family: Some("llama".into()),
            source: Source::HuggingFace,
            repo: Some("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF".into()),
            params_total: 8_030_000_000,
            params_active: 8_030_000_000,
            architecture: Some("llama".into()),
            context_max: 131_072,
            released_at: None,
            released_approx: false,
            variants: vec![variant(
                "Q4_K_M",
                vec![RuntimeKind::Ollama, RuntimeKind::LlamaCpp],
            )],
        }
    }

    fn variant(quant: &str, compat: Vec<RuntimeKind>) -> CatalogVariant {
        CatalogVariant {
            quant: quant.into(),
            bpw: 4.83,
            file_size_bytes: None,
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
            embedding_dim: 4096,
            runtime_compat: compat,
            source_tag: None,
        }
    }

    fn with_ollama() -> RuntimesStatus {
        RuntimesStatus {
            ollama: RuntimeStatus {
                installed: true,
                version: None,
                running: true,
            },
            ..Default::default()
        }
    }

    #[test]
    fn hf_gguf_with_ollama_uses_hf_co_ref() {
        let m = hf_model();
        let plan = plan_run(&m, &m.variants[0], &with_ollama(), None).unwrap();
        assert_eq!(
            plan.argv,
            vec![
                "ollama",
                "run",
                "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
            ]
        );
        assert!(plan.install.is_none());
    }

    #[test]
    fn ollama_library_model_uses_plain_name() {
        let mut m = hf_model();
        m.source = Source::Ollama;
        m.repo = None;
        m.name = "llama3.1:8b".into();
        let plan = plan_run(&m, &m.variants[0], &with_ollama(), None).unwrap();
        assert_eq!(plan.argv, vec!["ollama", "run", "llama3.1:8b"]);
    }

    /// Curated Ollama model enriched with an exact library tag.
    fn enriched_ollama_model() -> CatalogModel {
        let mut m = hf_model();
        m.source = Source::Ollama;
        m.repo = None;
        m.name = "llama3.1:8b".into();
        m.variants[0].source_tag = Some("8b-instruct-q4_K_M".into());
        m
    }

    #[test]
    fn run_enriched_ollama_variant_uses_exact_source_tag() {
        let m = enriched_ollama_model();
        let plan = plan_run(&m, &m.variants[0], &with_ollama(), None).unwrap();
        assert_eq!(
            plan.argv,
            vec!["ollama", "run", "llama3.1:8b-instruct-q4_K_M"]
        );
    }

    #[test]
    fn serve_enriched_ollama_variant_pulls_exact_source_tag() {
        let m = enriched_ollama_model();
        let p = plan_serve(&m, &m.variants[0], &with_ollama(), None, None).unwrap();
        assert_eq!(
            p.pre_steps,
            vec![vec![
                "ollama".to_string(),
                "pull".to_string(),
                "llama3.1:8b-instruct-q4_K_M".to_string()
            ]]
        );
        assert_eq!(p.model_ref, "llama3.1:8b-instruct-q4_K_M");
    }

    #[test]
    fn mlx_model_with_mlx_installed() {
        let mut m = hf_model();
        m.source = Source::Mlx;
        m.repo = Some("mlx-community/Llama-3.1-8B-Instruct-4bit".into());
        m.variants = vec![variant("MLX_4BIT", vec![RuntimeKind::MlxLm])];
        let rt = RuntimesStatus {
            mlx: RuntimeStatus {
                installed: true,
                version: Some("0.24.0".into()),
                running: false,
            },
            ..Default::default()
        };
        let plan = plan_run(&m, &m.variants[0], &rt, None).unwrap();
        assert_eq!(
            plan.argv,
            vec![
                "mlx_lm.chat",
                "--model",
                "mlx-community/Llama-3.1-8B-Instruct-4bit"
            ]
        );
    }

    #[test]
    fn no_runtime_yields_install_plan_never_auto_runs() {
        let m = hf_model();
        let plan = plan_run(&m, &m.variants[0], &RuntimesStatus::default(), None).unwrap();
        let install = plan.install.expect("must propose an install");
        assert_eq!(install.argv, vec!["brew", "install", "ollama"]);
        // run command still present so the UI can show what WILL run after install
        assert_eq!(plan.argv[0], "ollama");
    }

    #[test]
    fn mlx_without_repo_errors() {
        let mut m = hf_model();
        m.source = Source::Mlx;
        m.repo = None;
        m.variants = vec![variant("MLX_4BIT", vec![RuntimeKind::MlxLm])];
        let err = plan_run(&m, &m.variants[0], &RuntimesStatus::default(), None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no HuggingFace repo reference"), "{msg}");
        assert!(msg.contains("paddock sync"), "{msg}");
    }

    #[test]
    fn hf_without_repo_errors() {
        let mut m = hf_model();
        m.repo = None;
        let err = plan_run(&m, &m.variants[0], &with_ollama(), None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no HuggingFace repo reference"), "{msg}");
        assert!(msg.contains("paddock sync"), "{msg}");
    }

    #[test]
    fn hf_with_mlx_quant_errors() {
        let mut m = hf_model();
        m.variants = vec![variant("MLX_4BIT", vec![RuntimeKind::MlxLm])];
        let err = plan_run(&m, &m.variants[0], &with_ollama(), None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not a GGUF quant tag"), "{msg}");
        assert!(msg.contains("MLX_4BIT"), "{msg}");

        // DB-roundtripped garbage (unknown quant) must also be refused
        m.variants = vec![variant("Q4_BOGUS", vec![RuntimeKind::Ollama])];
        let err = plan_run(&m, &m.variants[0], &with_ollama(), None).unwrap_err();
        assert!(err.to_string().contains("not a GGUF quant tag"));
    }

    fn ollama_running() -> RuntimesStatus {
        RuntimesStatus {
            ollama: RuntimeStatus {
                installed: true,
                version: Some("0.30.6".into()),
                running: true,
            },
            ..Default::default()
        }
    }

    fn ollama_stopped() -> RuntimesStatus {
        RuntimesStatus {
            ollama: RuntimeStatus {
                installed: true,
                version: None,
                running: false,
            },
            ..Default::default()
        }
    }

    fn llama_cpp_only() -> RuntimesStatus {
        RuntimesStatus {
            llama_cpp: RuntimeStatus {
                installed: true,
                version: None,
                running: false,
            },
            ..Default::default()
        }
    }

    #[test]
    fn serve_hf_with_running_ollama_pulls_and_reuses_daemon() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &ollama_running(), None, None).unwrap();
        assert!(p.server_argv.is_none());
        assert_eq!(
            p.pre_steps,
            vec![vec![
                "ollama".to_string(),
                "pull".to_string(),
                "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M".to_string()
            ]]
        );
        assert_eq!(p.endpoint, "http://127.0.0.1:11434");
        assert_eq!(p.openai_url, "http://127.0.0.1:11434/v1/chat/completions");
        assert_eq!(
            p.model_ref,
            "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
        );
        assert_eq!(p.ready_path, "/api/version");
        assert!(p.install.is_none());
        assert_eq!(p.runtime, RuntimeKind::Ollama);
    }

    #[test]
    fn serve_hf_with_stopped_ollama_boots_the_daemon() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &ollama_stopped(), None, None).unwrap();
        assert_eq!(
            p.server_argv,
            Some(vec!["ollama".to_string(), "serve".to_string()])
        );
        // pull still listed: the binary runs pre_steps AFTER readiness
        assert_eq!(p.pre_steps.len(), 1);
        assert_eq!(p.endpoint, "http://127.0.0.1:11434");
        assert_eq!(p.runtime, RuntimeKind::Ollama);
    }

    #[test]
    fn serve_hf_without_ollama_falls_back_to_llama_server() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &llama_cpp_only(), None, None).unwrap();
        assert_eq!(
            p.server_argv,
            Some(vec![
                "llama-server".to_string(),
                "-hf".to_string(),
                "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M".to_string(),
                "--port".to_string(),
                "8080".to_string(),
                "-c".to_string(),
                "8192".to_string(),
                "--no-mmproj".to_string(),
            ])
        );
        assert!(p.pre_steps.is_empty());
        assert_eq!(p.endpoint, "http://127.0.0.1:8080");
        assert_eq!(p.ready_path, "/health");
        assert!(p.install.is_none());
        assert_eq!(p.runtime, RuntimeKind::LlamaCpp);
    }

    #[test]
    fn serve_ctx_override_applies_to_llama_server() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &llama_cpp_only(), None, Some(32768)).unwrap();
        let argv = p.server_argv.expect("llama-server argv");
        // `-c` immediately followed by the requested ctx, default replaced.
        let c = argv.iter().position(|a| a == "-c").expect("-c flag");
        assert_eq!(argv[c + 1], "32768");
    }

    #[test]
    fn run_ctx_override_applies_to_llama_cli() {
        let m = mmproj_model();
        let plan = plan_run(&m, &m.variants[0], &ollama_and_llama_cpp(), Some(16384)).unwrap();
        let c = plan.argv.iter().position(|a| a == "-c").expect("-c flag");
        assert_eq!(plan.argv[c + 1], "16384");
    }

    #[test]
    fn serve_port_override_applies_to_non_ollama_servers() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &llama_cpp_only(), Some(9999), None).unwrap();
        assert_eq!(p.endpoint, "http://127.0.0.1:9999");
        assert!(p.server_argv.unwrap().contains(&"9999".to_string()));
        assert!(!p.port_ignored);
        // ollama ignores --port (fixed daemon port)
        let p2 = plan_serve(&m, &m.variants[0], &ollama_running(), Some(9999), None).unwrap();
        assert_eq!(p2.endpoint, "http://127.0.0.1:11434");
        assert!(p2.port_ignored);
    }

    #[test]
    fn with_port_rewrites_argv_endpoint_and_url() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &llama_cpp_only(), None, None)
            .unwrap()
            .with_port(8090);
        assert_eq!(p.port, Some(8090));
        assert_eq!(p.endpoint, "http://127.0.0.1:8090");
        assert_eq!(p.openai_url, "http://127.0.0.1:8090/v1/chat/completions");
        let argv = p.server_argv.unwrap();
        let i = argv.iter().position(|a| a == "--port").unwrap();
        assert_eq!(argv[i + 1], "8090");
    }

    #[test]
    fn serve_rejects_port_zero() {
        let m = hf_model();
        assert!(plan_serve(&m, &m.variants[0], &llama_cpp_only(), Some(0), None).is_err());
    }

    #[test]
    fn serve_mlx_model() {
        let mut m = hf_model();
        m.source = Source::Mlx;
        m.repo = Some("mlx-community/Llama-3.1-8B-Instruct-4bit".into());
        m.variants = vec![variant("MLX_4BIT", vec![RuntimeKind::MlxLm])];
        let rt = RuntimesStatus {
            mlx: RuntimeStatus {
                installed: true,
                version: Some("0.24.0".into()),
                running: false,
            },
            ..Default::default()
        };
        let p = plan_serve(&m, &m.variants[0], &rt, None, None).unwrap();
        assert_eq!(
            p.server_argv,
            Some(vec![
                "mlx_lm.server".to_string(),
                "--model".to_string(),
                "mlx-community/Llama-3.1-8B-Instruct-4bit".to_string(),
                "--port".to_string(),
                "8080".to_string(),
            ])
        );
        assert_eq!(p.openai_url, "http://127.0.0.1:8080/v1/chat/completions");
        assert_eq!(p.model_ref, "mlx-community/Llama-3.1-8B-Instruct-4bit");
        assert_eq!(p.ready_path, "/v1/models");
        assert_eq!(p.runtime, RuntimeKind::MlxLm);
    }

    #[test]
    fn serve_without_any_runtime_proposes_install() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &RuntimesStatus::default(), None, None).unwrap();
        let install = p.install.expect("must propose install");
        assert_eq!(install.argv, vec!["brew", "install", "ollama"]);
        // plan still describes what will happen post-install (ollama path)
        assert_eq!(p.endpoint, "http://127.0.0.1:11434");
    }

    #[test]
    fn serve_rejects_repo_less_and_bad_quants_like_run() {
        let mut m = hf_model();
        m.repo = None;
        assert!(plan_serve(&m, &m.variants[0], &ollama_running(), None, None).is_err());
        let m2 = hf_model();
        let bad = variant("MLX_4BIT", vec![RuntimeKind::MlxLm]);
        assert!(plan_serve(&m2, &bad, &ollama_running(), None, None).is_err());
    }

    /// HF model whose repo ships a separate mmproj file: llama.cpp-only.
    fn mmproj_model() -> CatalogModel {
        let mut m = hf_model();
        m.name = "Qwen3.6-35B-A3B-MTP-GGUF".into();
        m.repo = Some("unsloth/Qwen3.6-35B-A3B-MTP-GGUF".into());
        m.variants = vec![variant("UD-Q4_K_M", vec![RuntimeKind::LlamaCpp])];
        m
    }

    fn ollama_and_llama_cpp() -> RuntimesStatus {
        let mut rt = ollama_running();
        rt.llama_cpp = RuntimeStatus {
            installed: true,
            version: None,
            running: false,
        };
        rt
    }

    #[test]
    fn serve_llama_cpp_only_variant_never_uses_ollama_even_if_running() {
        let m = mmproj_model();
        let p = plan_serve(&m, &m.variants[0], &ollama_and_llama_cpp(), None, None).unwrap();
        assert_eq!(
            p.server_argv,
            Some(vec![
                "llama-server".to_string(),
                "-hf".to_string(),
                "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M".to_string(),
                "--port".to_string(),
                "8080".to_string(),
                "-c".to_string(),
                "8192".to_string(),
                "--no-mmproj".to_string(),
            ])
        );
        assert!(
            p.pre_steps.is_empty(),
            "must not ollama pull: {:?}",
            p.pre_steps
        );
        assert_eq!(p.runtime, RuntimeKind::LlamaCpp);
        assert!(p.install.is_none());
    }

    #[test]
    fn serve_llama_cpp_only_variant_without_runtimes_proposes_llama_cpp_install() {
        let m = mmproj_model();
        let p = plan_serve(&m, &m.variants[0], &RuntimesStatus::default(), None, None).unwrap();
        let install = p.install.expect("must propose install");
        assert_eq!(install.kind, RuntimeKind::LlamaCpp);
        assert_eq!(install.argv, vec!["brew", "install", "llama.cpp"]);
        // plan still shows what will run post-install: llama-server shape
        let argv = p.server_argv.expect("llama-server argv");
        assert_eq!(argv[0], "llama-server");
        assert_eq!(p.endpoint, "http://127.0.0.1:8080");
        assert_eq!(p.ready_path, "/health");
        assert_eq!(p.runtime, RuntimeKind::LlamaCpp);
    }

    #[test]
    fn serve_llama_cpp_only_variant_with_only_ollama_running_still_llama_server() {
        let m = mmproj_model();
        let p = plan_serve(&m, &m.variants[0], &ollama_running(), None, None).unwrap();
        assert_eq!(p.runtime, RuntimeKind::LlamaCpp);
        assert!(p.pre_steps.is_empty());
        let install = p.install.expect("llama.cpp not installed → propose it");
        assert_eq!(install.argv, vec!["brew", "install", "llama.cpp"]);
    }

    #[test]
    fn run_llama_cpp_only_variant_uses_llama_cli() {
        let m = mmproj_model();
        let plan = plan_run(&m, &m.variants[0], &ollama_and_llama_cpp(), None).unwrap();
        assert_eq!(
            plan.argv,
            vec![
                "llama-cli",
                "-hf",
                "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M",
                "-c",
                "8192",
                "--no-mmproj"
            ]
        );
        assert!(plan.install.is_none());
    }

    #[test]
    fn run_llama_cpp_only_variant_without_runtimes_proposes_llama_cpp_install() {
        let m = mmproj_model();
        let plan = plan_run(&m, &m.variants[0], &RuntimesStatus::default(), None).unwrap();
        assert_eq!(plan.argv[0], "llama-cli");
        let install = plan.install.expect("must propose install");
        assert_eq!(install.kind, RuntimeKind::LlamaCpp);
        assert_eq!(install.argv, vec!["brew", "install", "llama.cpp"]);
    }

    #[test]
    fn ud_quants_accepted_on_ollama_path_too() {
        // UD- tags now have a known bpw and name a real file in the repo, so
        // the hf.co ref is valid for ollama-compatible variants.
        let mut m = hf_model();
        m.variants = vec![variant(
            "UD-Q4_K_M",
            vec![RuntimeKind::Ollama, RuntimeKind::LlamaCpp],
        )];
        let plan = plan_run(&m, &m.variants[0], &with_ollama(), None).unwrap();
        assert_eq!(
            plan.argv,
            vec![
                "ollama",
                "run",
                "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:UD-Q4_K_M"
            ]
        );
    }

    #[test]
    fn display_shell_quotes_unsafe_elements() {
        let plan = RunPlan {
            argv: vec!["ollama".into(), "run".into(), "weird name".into()],
            install: None,
        };
        assert_eq!(plan.display(), "ollama run 'weird name'");
    }
}
