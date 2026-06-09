use thiserror::Error;

/// All errors produced by tetro-core. Messages must be actionable.
#[derive(Debug, Error)]
pub enum TetroError {
    #[error("sysctl `{name}` unavailable ({reason}); is this a macOS machine?")]
    Sysctl { name: String, reason: String },
    #[error("catalog database error: {0}. Try deleting the catalog and re-running `tetro sync`.")]
    Db(#[from] rusqlite::Error),
    #[error("network error: {0}. Check your connection and retry `tetro sync`.")]
    Network(String),
    #[error("malformed GGUF header: {0}")]
    Gguf(String),
    #[error("model `{0}` not found in catalog. Run `tetro sync` or check `tetro fit` for names.")]
    ModelNotFound(String),
    #[error("{0}")]
    Other(String),
}
