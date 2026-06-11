use thiserror::Error;

/// All errors produced by paddock-core. Messages must be actionable.
#[derive(Debug, Error)]
pub enum PaddockError {
    #[error("sysctl `{name}` unavailable ({reason}); is this a macOS machine?")]
    Sysctl { name: String, reason: String },
    #[error("catalog database error: {0}. Try deleting the catalog and re-running `paddock sync`.")]
    Db(#[from] rusqlite::Error),
    #[error("network error: {0}. Check your connection and retry `paddock sync`.")]
    Network(String),
    #[error("malformed GGUF header: {0}")]
    Gguf(String),
    #[error(
        "model `{0}` not found in catalog. Run `paddock sync` or check `paddock fit` for names."
    )]
    ModelNotFound(String),
    #[error("{0}")]
    Other(String),
}
