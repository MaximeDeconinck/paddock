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
            if catalog::quant_bpw(&variant.quant).is_none() || variant.quant.starts_with("MLX_") {
                return Err(TetroError::Other(format!(
                    "variant `{}` of `{}` is not a valid GGUF tag for `ollama run hf.co/...`",
                    variant.quant, model.name
                )));
            }
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
        assert!(msg.contains("not a valid GGUF tag"), "{msg}");
        assert!(msg.contains("MLX_4BIT"), "{msg}");

        // DB-roundtripped garbage (unknown quant) must also be refused
        m.variants = vec![variant("Q4_BOGUS", vec![RuntimeKind::Ollama])];
        let err = plan_run(&m, &m.variants[0], &with_ollama()).unwrap_err();
        assert!(err.to_string().contains("not a valid GGUF tag"));
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
