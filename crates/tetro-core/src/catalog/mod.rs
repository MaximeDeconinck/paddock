pub mod curated;
pub mod db;
pub mod gguf;
pub mod hf;

use serde::{Deserialize, Serialize};

use crate::estimate::ModelVariant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    HuggingFace,
    Ollama,
    Mlx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeKind {
    Ollama,
    LlamaCpp,
    MlxLm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogVariant {
    pub quant: String,
    pub bpw: f64,
    pub file_size_bytes: Option<u64>,
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub embedding_dim: u32,
    pub runtime_compat: Vec<RuntimeKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: i64,
    pub name: String,
    pub family: Option<String>,
    pub source: Source,
    /// HF repo id when applicable (enables `ollama run hf.co/{repo}:{quant}`).
    pub repo: Option<String>,
    pub params_total: u64,
    pub params_active: u64,
    pub architecture: Option<String>,
    pub context_max: u32,
    pub variants: Vec<CatalogVariant>,
}

impl CatalogModel {
    /// Bridge to the estimator's flat variant type.
    pub fn to_model_variant(&self, v: &CatalogVariant) -> ModelVariant {
        ModelVariant {
            model_name: self.name.clone(),
            quant: v.quant.clone(),
            bpw: v.bpw,
            params_total: self.params_total,
            params_active: self.params_active,
            layers: v.layers,
            kv_heads: v.kv_heads,
            head_dim: v.head_dim,
            embedding_dim: v.embedding_dim,
            context_max: self.context_max,
        }
    }
}

/// Effective bits-per-weight per quant (K-quants carry metadata overhead).
pub fn quant_bpw(quant: &str) -> Option<f64> {
    Some(match quant {
        "Q8_0" => 8.5,
        "Q6_K" => 6.59,
        "Q5_K_M" => 5.69,
        "Q4_K_M" => 4.83,
        "Q4_0" => 4.55,
        "Q3_K_M" => 3.91,
        "Q2_K" => 3.35,
        "IQ4_XS" => 4.25,
        "F16" | "BF16" => 16.0,
        "MLX_4BIT" => 4.5,
        "MLX_8BIT" => 8.5,
        _ => return None,
    })
}

/// Extract a known quant tag from a GGUF filename, e.g. "llama-3.1-8b-Q4_K_M.gguf".
pub fn quant_from_filename(name: &str) -> Option<String> {
    const KNOWN: &[&str] = &[
        "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K", "IQ4_XS", "BF16", "F16",
    ];
    let upper = name.to_uppercase();
    KNOWN
        .iter()
        .find(|q| upper.contains(*q))
        .map(|q| q.to_string())
}

/// Options for `sync`. Limits keep first sync fast; raise via CLI later if wanted.
pub struct SyncOptions {
    pub hf_limit: usize,
    pub mlx_limit: usize,
}

impl Default for SyncOptions {
    fn default() -> Self {
        Self {
            hf_limit: 30,
            mlx_limit: 30,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct SyncReport {
    pub curated: usize,
    pub huggingface: usize,
    pub mlx: usize,
    pub errors: Vec<String>,
}

/// Idempotent catalog sync: curated list always, network sources best-effort.
pub async fn sync(
    http: &dyn hf::HttpClient,
    db: &db::Db,
    opts: &SyncOptions,
) -> Result<SyncReport, crate::TetroError> {
    let mut report = SyncReport::default();
    for m in curated::curated_ollama_models() {
        db.upsert_model(&m)?;
        report.curated += 1;
    }
    match hf::fetch_hf_gguf(http, opts.hf_limit).await {
        Ok(models) => {
            for m in models {
                db.upsert_model(&m)?;
                report.huggingface += 1;
            }
        }
        Err(e) => report.errors.push(format!("huggingface: {e}")),
    }
    match hf::fetch_mlx(http, opts.mlx_limit).await {
        Ok(models) => {
            for m in models {
                db.upsert_model(&m)?;
                report.mlx += 1;
            }
        }
        Err(e) => report.errors.push(format!("mlx: {e}")),
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    db.set_last_sync(now)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use serde_json::Value;

    use super::*;
    use crate::TetroError;

    /// MockHttp that always fails.
    struct FailingHttp;

    #[async_trait]
    impl hf::HttpClient for FailingHttp {
        async fn get_json(&self, url: &str) -> Result<Value, TetroError> {
            Err(TetroError::Network(format!("mock failure: {url}")))
        }

        async fn get_range(
            &self,
            url: &str,
            _start: u64,
            _end: u64,
        ) -> Result<Vec<u8>, TetroError> {
            Err(TetroError::Network(format!("mock failure: {url}")))
        }
    }

    #[tokio::test]
    async fn sync_failing_http_still_persists_curated() {
        let dir = tempfile::tempdir().unwrap();
        let db = db::Db::open(dir.path().join("catalog.db")).unwrap();
        let http = FailingHttp;

        let report = sync(&http, &db, &SyncOptions::default()).await.unwrap();

        // Curated always inserted regardless of network
        assert!(
            report.curated >= 35,
            "expected >= 35 curated, got {}",
            report.curated
        );
        // Network sources failed
        assert_eq!(
            report.errors.len(),
            2,
            "expected 2 network errors, got {:?}",
            report.errors
        );
        assert!(report.errors[0].contains("huggingface"));
        assert!(report.errors[1].contains("mlx"));
        // last_sync set even on network failure
        let ts = db.last_sync().unwrap();
        assert!(ts.is_some(), "last_sync should be set");
        assert!(ts.unwrap() > 0);

        // Curated models actually in DB
        let models = db.list_models().unwrap();
        assert!(models.len() >= 35, "expected >= 35 models in DB");
    }
}
