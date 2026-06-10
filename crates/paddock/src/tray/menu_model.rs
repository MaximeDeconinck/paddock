//! Pure construction of the tray menu content. No tray-icon types here so it
//! is unit-testable and portable to the future Tauri app.

use paddock_core::catalog::RuntimeKind;
use paddock_core::hardware::RuntimesStatus;
use paddock_core::serving::{LoadedModel, ServingRecord};

use crate::output::gib;

const OLLAMA_HOST: &str = "127.0.0.1:11434";
const OLLAMA_OPENAI_URL: &str = "http://127.0.0.1:11434/v1/chat/completions";

#[derive(Debug, PartialEq, Eq)]
pub enum MenuEntry {
    /// Disabled section header, e.g. "Ollama — 127.0.0.1:11434".
    Header(String),
    /// Clickable model row; activating copies `copy_url`.
    Model {
        label: String,
        copy_url: String,
    },
    /// Disabled informational line.
    Info(String),
    Separator,
}

#[derive(Debug, PartialEq, Eq)]
pub struct MenuModel {
    pub entries: Vec<MenuEntry>,
}

/// Merge paddock-launched servers (registry) with live Ollama state (`/api/ps`)
/// into a flat list of menu entries. `ollama_ps: None` means the daemon is
/// unreachable; the section only exists at all when Ollama is installed.
/// The footer (Rafraîchir/Quitter) belongs to the rendering layer, not here.
pub fn build_menu_model(
    records: &[ServingRecord],
    ollama_ps: Option<&[LoadedModel]>,
    runtimes: &RuntimesStatus,
) -> MenuModel {
    let mut entries = Vec::new();

    // Ollama daemon section first, gated on the runtime being installed.
    // Registry records for an ollama paddock booted are skipped below: the
    // daemon section already covers them (dedup).
    if runtimes.ollama.installed {
        match ollama_ps {
            Some(models) if !models.is_empty() => {
                entries.push(MenuEntry::Header(format!("Ollama — {OLLAMA_HOST}")));
                for m in models {
                    entries.push(MenuEntry::Model {
                        label: format!("{} — {}", m.name, gib(m.size_bytes)),
                        copy_url: OLLAMA_OPENAI_URL.to_string(),
                    });
                }
            }
            Some(_) => {
                entries.push(MenuEntry::Header(format!("Ollama — {OLLAMA_HOST}")));
                entries.push(MenuEntry::Info("aucun modèle chargé".into()));
            }
            None => entries.push(MenuEntry::Info("Ollama — injoignable".into())),
        }
    }

    // One section per paddock-launched server (llama-server / mlx_lm.server).
    for r in records {
        let label = match r.runtime {
            RuntimeKind::Ollama => continue, // covered by the daemon section
            RuntimeKind::LlamaCpp => "llama-server",
            RuntimeKind::MlxLm => "mlx-lm",
        };
        if !entries.is_empty() {
            entries.push(MenuEntry::Separator);
        }
        entries.push(MenuEntry::Header(format!(
            "{label} — {}",
            host_port(&r.endpoint)
        )));
        entries.push(MenuEntry::Model {
            label: r.model_ref.clone(),
            copy_url: r.openai_url.clone(),
        });
    }

    if entries.is_empty() {
        entries.push(MenuEntry::Info("Aucun serveur actif".into()));
    }
    if !runtimes.ollama.installed && !runtimes.llama_cpp.installed && !runtimes.mlx.installed {
        entries.push(MenuEntry::Info(
            "Aucun runtime installé — lance `paddock run` pour installer".into(),
        ));
    }

    MenuModel { entries }
}

/// "http://127.0.0.1:8080" → "127.0.0.1:8080" for header lines.
fn host_port(endpoint: &str) -> &str {
    endpoint
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_end_matches('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(runtime: RuntimeKind, port: u16, model_ref: &str) -> ServingRecord {
        ServingRecord {
            pid: 1234,
            runtime,
            endpoint: format!("http://127.0.0.1:{port}"),
            openai_url: format!("http://127.0.0.1:{port}/v1/chat/completions"),
            model_ref: model_ref.to_string(),
            ready_path: "/health".into(),
            started_at: 1_770_000_000,
        }
    }

    fn runtimes(ollama: bool, llama: bool, mlx: bool) -> RuntimesStatus {
        let mut s = RuntimesStatus::default();
        s.ollama.installed = ollama;
        s.llama_cpp.installed = llama;
        s.mlx.installed = mlx;
        s
    }

    const GIB18: u64 = 18 * 1024 * 1024 * 1024;

    #[test]
    fn ollama_with_loaded_models() {
        let ps = vec![LoadedModel {
            name: "qwen3:30b-a3b".into(),
            size_bytes: GIB18,
        }];
        let m = build_menu_model(&[], Some(&ps), &runtimes(true, false, false));
        assert_eq!(
            m.entries,
            vec![
                MenuEntry::Header("Ollama — 127.0.0.1:11434".into()),
                MenuEntry::Model {
                    label: "qwen3:30b-a3b — 18.0 GiB".into(),
                    copy_url: "http://127.0.0.1:11434/v1/chat/completions".into(),
                },
            ]
        );
    }

    #[test]
    fn ollama_reachable_but_empty_shows_no_model_loaded() {
        let m = build_menu_model(&[], Some(&[]), &runtimes(true, false, false));
        assert_eq!(
            m.entries,
            vec![
                MenuEntry::Header("Ollama — 127.0.0.1:11434".into()),
                MenuEntry::Info("aucun modèle chargé".into()),
            ]
        );
    }

    #[test]
    fn ollama_unreachable_shows_injoignable() {
        let m = build_menu_model(&[], None, &runtimes(true, false, false));
        assert_eq!(
            m.entries,
            vec![MenuEntry::Info("Ollama — injoignable".into())]
        );
    }

    #[test]
    fn ollama_not_installed_no_section() {
        let ps = vec![LoadedModel {
            name: "qwen3:30b-a3b".into(),
            size_bytes: GIB18,
        }];
        // Even with /api/ps data, no Ollama section when it is not installed.
        let m = build_menu_model(&[], Some(&ps), &runtimes(false, true, false));
        assert_eq!(
            m.entries,
            vec![MenuEntry::Info("Aucun serveur actif".into())]
        );
    }

    #[test]
    fn registry_records_get_sections() {
        let records = vec![
            record(RuntimeKind::LlamaCpp, 8080, "unsloth/Qwen3-8B-GGUF:Q4_K_M"),
            record(RuntimeKind::MlxLm, 8081, "mlx-community/Qwen3-8B-4bit"),
        ];
        let m = build_menu_model(&records, None, &runtimes(false, true, true));
        assert_eq!(
            m.entries,
            vec![
                MenuEntry::Header("llama-server — 127.0.0.1:8080".into()),
                MenuEntry::Model {
                    label: "unsloth/Qwen3-8B-GGUF:Q4_K_M".into(),
                    copy_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
                },
                MenuEntry::Separator,
                MenuEntry::Header("mlx-lm — 127.0.0.1:8081".into()),
                MenuEntry::Model {
                    label: "mlx-community/Qwen3-8B-4bit".into(),
                    copy_url: "http://127.0.0.1:8081/v1/chat/completions".into(),
                },
            ]
        );
    }

    #[test]
    fn ollama_registry_record_deduped() {
        // paddock booted `ollama serve` → registry record; the daemon section
        // (from /api/ps) already covers it, so the record adds nothing.
        let records = vec![record(RuntimeKind::Ollama, 11434, "qwen3:30b-a3b")];
        let ps = vec![LoadedModel {
            name: "qwen3:30b-a3b".into(),
            size_bytes: GIB18,
        }];
        let m = build_menu_model(&records, Some(&ps), &runtimes(true, false, false));
        assert_eq!(
            m.entries,
            vec![
                MenuEntry::Header("Ollama — 127.0.0.1:11434".into()),
                MenuEntry::Model {
                    label: "qwen3:30b-a3b — 18.0 GiB".into(),
                    copy_url: "http://127.0.0.1:11434/v1/chat/completions".into(),
                },
            ]
        );
    }

    #[test]
    fn empty_all() {
        // Nothing running, nothing installed → both hint lines.
        let m = build_menu_model(&[], None, &runtimes(false, false, false));
        assert_eq!(
            m.entries,
            vec![
                MenuEntry::Info("Aucun serveur actif".into()),
                MenuEntry::Info(
                    "Aucun runtime installé — lance `paddock run` pour installer".into()
                ),
            ]
        );
    }

    #[test]
    fn model_label_uses_gib_formatting() {
        let ps = vec![LoadedModel {
            name: "llama3.2:1b".into(),
            size_bytes: 1_400_000_000,
        }];
        let m = build_menu_model(&[], Some(&ps), &runtimes(true, false, false));
        let MenuEntry::Model { label, .. } = &m.entries[1] else {
            panic!("expected a model row, got {:?}", m.entries[1]);
        };
        // Same binary-GiB formatting as the rest of the CLI (output::gib).
        assert_eq!(label, &format!("llama3.2:1b — {}", gib(1_400_000_000)));
        assert_eq!(label, "llama3.2:1b — 1.3 GiB");
    }
}
