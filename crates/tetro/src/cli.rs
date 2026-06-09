use clap::{Parser, Subcommand};
use tetro_core::score::UseCase;

#[derive(Parser)]
#[command(
    name = "tetro",
    version,
    about = "Which LLMs fit your Mac, how fast, and how to run them"
)]
pub struct Cli {
    /// Machine-readable JSON output
    #[arg(long, global = true)]
    pub json: bool,
    /// Force plain table output (skip the TUI)
    #[arg(long, global = true)]
    pub cli: bool,
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Show the hardware profile
    Scan,
    /// List models that fit this machine
    Fit {
        /// Include models that do not fit
        #[arg(long)]
        all: bool,
        #[arg(long, value_enum, default_value = "general")]
        use_case: UseCaseArg,
        /// Limit the number of rows
        #[arg(short = 'n', long, default_value_t = 20)]
        limit: usize,
    },
    /// Top 5 recommendations with one-line justifications
    Recommend {
        #[arg(long, value_enum, default_value = "general")]
        use_case: UseCaseArg,
    },
    /// Launch a model with the best available runtime
    Run { model: String },
    /// Refresh the model catalog
    Sync,
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum UseCaseArg {
    General,
    Coding,
    Chat,
    Reasoning,
}

impl From<UseCaseArg> for UseCase {
    fn from(v: UseCaseArg) -> Self {
        match v {
            UseCaseArg::General => UseCase::General,
            UseCaseArg::Coding => UseCase::Coding,
            UseCaseArg::Chat => UseCase::Chat,
            UseCaseArg::Reasoning => UseCase::Reasoning,
        }
    }
}
