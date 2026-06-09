mod app;
mod cli;
mod output;

// Temporary stub — the real TUI lands in Task 7.
mod tui {
    pub fn run(_app: crate::app::App) -> anyhow::Result<()> {
        anyhow::bail!("TUI lands in Task 7; use --cli")
    }
}

use std::io::Write;

use anyhow::{bail, Context, Result};
use clap::Parser;
use tetro_core::catalog::CatalogModel;
use tetro_core::runtime::{plan_run, InstallPlan, RunPlan};
use tetro_core::score::{best_variant, UseCase};
use tetro_core::TetroError;

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
                eprintln!("catalog is empty — run `tetro sync` first");
            }
            rows.truncate(5);
            if cli.json {
                output::print_recommendations_json(&rows)?;
            } else {
                output::print_recommendations(&rows);
            }
        }
        Some(Command::Run { model }) => run_model(&app, &model, cli.json)?,
        Some(Command::Sync) => {
            let db = app.open_db()?;
            let http = tetro_core::catalog::hf::ReqwestClient::new()?;
            let report = tokio::runtime::Runtime::new()?.block_on(tetro_core::catalog::sync(
                &http,
                &db,
                &Default::default(),
            ))?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "synced: {} curated, {} huggingface, {} mlx",
                    report.curated, report.huggingface, report.mlx
                );
                for e in &report.errors {
                    eprintln!("warning: {e}");
                }
            }
        }
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

/// Default listing shared by `tetro fit` and bare `tetro --cli/--json`.
fn fit(app: &App, all: bool, use_case: UseCase, limit: usize, json: bool) -> Result<()> {
    let db = app.open_db()?;
    let mut rows = app.scored_models(&db, use_case, all)?;
    rows.truncate(limit);
    if rows.is_empty() {
        eprintln!("catalog is empty — run `tetro sync` first");
    }
    if json {
        output::print_fit_json(&rows)?;
    } else {
        output::print_fit_table(&rows);
    }
    Ok(())
}

fn run_model(app: &App, query: &str, json: bool) -> Result<()> {
    let db = app.open_db()?;
    let models = db.list_models().context("reading catalog")?;
    let model = match find_model(&models, query) {
        Lookup::Found(m) => m,
        Lookup::Ambiguous(names) => {
            eprintln!("model name `{query}` is ambiguous — candidates:");
            for n in names {
                eprintln!("  {n}");
            }
            std::process::exit(1);
        }
        Lookup::NotFound => return Err(TetroError::ModelNotFound(query.to_string()).into()),
    };

    let mvs: Vec<_> = model
        .variants
        .iter()
        .map(|v| model.to_model_variant(v))
        .collect();
    let Some(best) = best_variant(&mvs, &app.budget) else {
        bail!(
            "no quantization of `{}` fits this machine ({} RAM); try a smaller model from `tetro fit`",
            model.name,
            output::gb(app.budget.ram_total_bytes)
        );
    };
    let variant = model
        .variants
        .iter()
        .find(|v| v.quant == best.quant)
        .expect("best_variant returns a quant present in the input");

    // API delta vs the original plan: plan_run is fallible (repo-less HF/MLX
    // models, non-GGUF quants). Surface the actionable error and exit non-zero.
    let plan: RunPlan = plan_run(model, variant, &app.profile.runtimes)?;

    if json {
        // Machine mode never launches interactive processes.
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }

    println!("$ {}", plan.display());
    if let Some(install) = &plan.install {
        confirm_and_install(install)?;
    }
    exec(&plan.argv)
}

fn confirm_and_install(install: &InstallPlan) -> Result<()> {
    let cmd = install
        .argv
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ");
    print!("required runtime is not installed. install with `{cmd}`? [y/N] ");
    std::io::stdout().flush()?;
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

/// Exact name match first, then case-insensitive contains.
fn find_model<'a>(models: &'a [CatalogModel], query: &str) -> Lookup<'a> {
    if let Some(m) = models.iter().find(|m| m.name == query) {
        return Lookup::Found(m);
    }
    let q = query.to_lowercase();
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
