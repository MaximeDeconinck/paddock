//! Builds the exact command to run a model — pure data, no exec, no prompt.
//! The binary owns display, confirmation and process replacement.

use serde::Serialize;

use crate::catalog::{self, CatalogModel, CatalogVariant, RuntimeKind, Source};
use crate::error::TetroError;
use crate::hardware::RuntimesStatus;

#[derive(Debug, Clone, Serialize)]
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

/// Hybrid run strategy (see README): prefer Ollama for GGUF, mlx_lm for MLX.
/// mlx-lm chat command verified against ml-explore/mlx-lm README (2026-06).
pub fn plan_run(
    model: &CatalogModel,
    variant: &CatalogVariant,
    rt: &RuntimesStatus,
) -> Result<RunPlan, TetroError> {
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
            argv: s(&["ollama", "run", &model.name]),
            install: (!rt.ollama.installed).then(ollama_install),
        }),
        Source::HuggingFace => {
            let repo = model.repo.as_deref().ok_or_else(|| no_repo(&model.name))?;
            validate_gguf_quant(model, variant)?;
            let model_ref = format!("hf.co/{repo}:{}", variant.quant);
            Ok(RunPlan {
                argv: s(&["ollama", "run", &model_ref]),
                install: (!rt.ollama.installed).then(ollama_install),
            })
        }
    }
}

fn no_repo(name: &str) -> TetroError {
    TetroError::Other(format!(
        "model `{name}` has no HuggingFace repo reference; re-run `tetro sync`"
    ))
}

/// A quant tag usable as `hf.co/{repo}:{quant}` / `-hf {repo}:{quant}` must be
/// a known GGUF tag (not an MLX pseudo-quant, not DB garbage).
fn validate_gguf_quant(model: &CatalogModel, variant: &CatalogVariant) -> Result<(), TetroError> {
    if catalog::quant_bpw(&variant.quant).is_none() || variant.quant.starts_with("MLX_") {
        return Err(TetroError::Other(format!(
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
#[derive(Debug, Clone, Serialize)]
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
) -> Result<ServePlan, TetroError> {
    if port == Some(0) {
        return Err(TetroError::Other(
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
            // Prefer Ollama; fall back to llama-server for HF GGUF when only
            // llama.cpp is installed.
            if !rt.ollama.installed && rt.llama_cpp.installed {
                if let Some(hf_ref) = hf_ref.take() {
                    return Ok(ServePlan {
                        server_argv: Some(s(&[
                            "llama-server",
                            "-hf",
                            &hf_ref,
                            "--port",
                            &port.to_string(),
                        ])),
                        pre_steps: vec![],
                        endpoint: local.clone(),
                        openai_url: format!("{local}/v1/chat/completions"),
                        model_ref: hf_ref,
                        ready_path: "/health".to_string(),
                        install: None,
                        port_ignored: false,
                    });
                }
            }
            let model_ref = match hf_ref {
                Some(hf_ref) => format!("hf.co/{hf_ref}"),
                None => model.name.clone(),
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
        let plan = plan_run(&m, &m.variants[0], &with_ollama()).unwrap();
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
        let plan = plan_run(&m, &m.variants[0], &with_ollama()).unwrap();
        assert_eq!(plan.argv, vec!["ollama", "run", "llama3.1:8b"]);
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
        let plan = plan_run(&m, &m.variants[0], &rt).unwrap();
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
        let plan = plan_run(&m, &m.variants[0], &RuntimesStatus::default()).unwrap();
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
        let err = plan_run(&m, &m.variants[0], &RuntimesStatus::default()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no HuggingFace repo reference"), "{msg}");
        assert!(msg.contains("tetro sync"), "{msg}");
    }

    #[test]
    fn hf_without_repo_errors() {
        let mut m = hf_model();
        m.repo = None;
        let err = plan_run(&m, &m.variants[0], &with_ollama()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no HuggingFace repo reference"), "{msg}");
        assert!(msg.contains("tetro sync"), "{msg}");
    }

    #[test]
    fn hf_with_mlx_quant_errors() {
        let mut m = hf_model();
        m.variants = vec![variant("MLX_4BIT", vec![RuntimeKind::MlxLm])];
        let err = plan_run(&m, &m.variants[0], &with_ollama()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not a GGUF quant tag"), "{msg}");
        assert!(msg.contains("MLX_4BIT"), "{msg}");

        // DB-roundtripped garbage (unknown quant) must also be refused
        m.variants = vec![variant("Q4_BOGUS", vec![RuntimeKind::Ollama])];
        let err = plan_run(&m, &m.variants[0], &with_ollama()).unwrap_err();
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
        let p = plan_serve(&m, &m.variants[0], &ollama_running(), None).unwrap();
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
    }

    #[test]
    fn serve_hf_with_stopped_ollama_boots_the_daemon() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &ollama_stopped(), None).unwrap();
        assert_eq!(
            p.server_argv,
            Some(vec!["ollama".to_string(), "serve".to_string()])
        );
        // pull still listed: the binary runs pre_steps AFTER readiness
        assert_eq!(p.pre_steps.len(), 1);
        assert_eq!(p.endpoint, "http://127.0.0.1:11434");
    }

    #[test]
    fn serve_hf_without_ollama_falls_back_to_llama_server() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &llama_cpp_only(), None).unwrap();
        assert_eq!(
            p.server_argv,
            Some(vec![
                "llama-server".to_string(),
                "-hf".to_string(),
                "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M".to_string(),
                "--port".to_string(),
                "8080".to_string(),
            ])
        );
        assert!(p.pre_steps.is_empty());
        assert_eq!(p.endpoint, "http://127.0.0.1:8080");
        assert_eq!(p.ready_path, "/health");
        assert!(p.install.is_none());
    }

    #[test]
    fn serve_port_override_applies_to_non_ollama_servers() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &llama_cpp_only(), Some(9999)).unwrap();
        assert_eq!(p.endpoint, "http://127.0.0.1:9999");
        assert!(p.server_argv.unwrap().contains(&"9999".to_string()));
        assert!(!p.port_ignored);
        // ollama ignores --port (fixed daemon port)
        let p2 = plan_serve(&m, &m.variants[0], &ollama_running(), Some(9999)).unwrap();
        assert_eq!(p2.endpoint, "http://127.0.0.1:11434");
        assert!(p2.port_ignored);
    }

    #[test]
    fn serve_rejects_port_zero() {
        let m = hf_model();
        assert!(plan_serve(&m, &m.variants[0], &llama_cpp_only(), Some(0)).is_err());
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
        let p = plan_serve(&m, &m.variants[0], &rt, None).unwrap();
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
    }

    #[test]
    fn serve_without_any_runtime_proposes_install() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &RuntimesStatus::default(), None).unwrap();
        let install = p.install.expect("must propose install");
        assert_eq!(install.argv, vec!["brew", "install", "ollama"]);
        // plan still describes what will happen post-install (ollama path)
        assert_eq!(p.endpoint, "http://127.0.0.1:11434");
    }

    #[test]
    fn serve_rejects_repo_less_and_bad_quants_like_run() {
        let mut m = hf_model();
        m.repo = None;
        assert!(plan_serve(&m, &m.variants[0], &ollama_running(), None).is_err());
        let m2 = hf_model();
        let bad = variant("MLX_4BIT", vec![RuntimeKind::MlxLm]);
        assert!(plan_serve(&m2, &bad, &ollama_running(), None).is_err());
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
