pub mod curated;
pub mod db;
pub mod gguf;
pub mod hf;
pub mod ollama_registry;

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
    /// Exact Ollama library tag (e.g. `8b-instruct-q4_K_M`) for
    /// `ollama run {base}:{tag}`; None for curated-offline/HF/MLX variants.
    #[serde(default)]
    pub source_tag: Option<String>,
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
    // Unsloth Dynamic (UD-*) tags: same layout as the base quant. The _XL
    // family upcasts the important layers; +0.35 bpw over the closest base
    // quant is a conservative estimate of that upcast cost.
    if let Some(base) = quant.strip_prefix("UD-") {
        let xl_base = match base {
            "Q2_K_XL" => Some("Q2_K"),
            "Q3_K_XL" => Some("Q3_K_M"),
            "Q4_K_XL" => Some("Q4_K_M"),
            "Q5_K_XL" => Some("Q5_K_M"),
            "Q6_K_XL" => Some("Q6_K"),
            "Q8_K_XL" => Some("Q8_0"),
            _ => None,
        };
        return match xl_base {
            Some(b) => quant_bpw(b).map(|v| v + 0.35),
            None => quant_bpw(base),
        };
    }
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
/// Unsloth Dynamic tags are returned EXACTLY as published (`UD-Q4_K_M`,
/// `UD-Q4_K_XL`): the tag doubles as the `hf.co/{repo}:{quant}` file selector,
/// so it must match a real filename in the repo.
pub fn quant_from_filename(name: &str) -> Option<String> {
    const KNOWN: &[&str] = &[
        "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K", "IQ4_XS", "BF16", "F16",
    ];
    // UD_XL tags checked first because several contain a KNOWN base quant as
    // a substring (UD-Q2_K_XL contains Q2_K, etc.).
    const UD_XL: &[&str] = &[
        "UD-Q2_K_XL",
        "UD-Q3_K_XL",
        "UD-Q4_K_XL",
        "UD-Q5_K_XL",
        "UD-Q6_K_XL",
        "UD-Q8_K_XL",
    ];
    let upper = name.to_uppercase();
    if let Some(q) = UD_XL.iter().find(|q| upper.contains(*q)) {
        return Some(q.to_string());
    }
    for q in KNOWN {
        if let Some(idx) = upper.find(q) {
            // `UD-` immediately before the base quant → Unsloth Dynamic tag.
            // `get` handles idx < 3 and non-char-boundary slices without panic.
            // Word boundary: "CLOUD-Q4_K_M" must not match as UD-.
            if upper.get(idx.wrapping_sub(3)..idx) == Some("UD-")
                && (idx == 3 || !upper.as_bytes()[idx - 4].is_ascii_alphanumeric())
            {
                return Some(format!("UD-{q}"));
            }
            return Some(q.to_string());
        }
    }
    None
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
                match db.upsert_model(&m) {
                    Ok(_) => report.huggingface += 1,
                    Err(e) => report
                        .errors
                        .push(format!("huggingface upsert {}: {e}", m.name)),
                }
            }
        }
        Err(e) => report.errors.push(format!("huggingface: {e}")),
    }
    match hf::fetch_mlx(http, opts.mlx_limit).await {
        Ok(models) => {
            for m in models {
                match db.upsert_model(&m) {
                    Ok(_) => report.mlx += 1,
                    Err(e) => report.errors.push(format!("mlx upsert {}: {e}", m.name)),
                }
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

        async fn get_text(&self, url: &str) -> Result<String, TetroError> {
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

    #[test]
    fn quant_from_filename_ud_prefix_kept_exactly() {
        assert_eq!(
            quant_from_filename("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
            Some("UD-Q4_K_M".to_string())
        );
        assert_eq!(
            quant_from_filename("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
            Some("UD-Q4_K_XL".to_string())
        );
        // lowercase ud- in the filename still yields the canonical tag
        assert_eq!(
            quant_from_filename("model-ud-q4_k_m.gguf"),
            Some("UD-Q4_K_M".to_string())
        );
        // plain tags unchanged
        assert_eq!(
            quant_from_filename("x-Q4_K_M.gguf"),
            Some("Q4_K_M".to_string())
        );
    }

    #[test]
    fn quant_from_filename_ud_word_boundary_and_multibyte() {
        // "UD-" preceded by an alphanumeric char is not an Unsloth tag.
        assert_eq!(
            quant_from_filename("Cloud-Q4_K_M.gguf"),
            Some("Q4_K_M".to_string())
        );
        // Multibyte char earlier in the name must not panic.
        assert_eq!(
            quant_from_filename("Éx-UD-Q4_K_M.gguf"),
            Some("UD-Q4_K_M".to_string())
        );
        // Multibyte within 3 bytes of the tag must not panic.
        assert_eq!(
            quant_from_filename("Méga-Q4_K_M.gguf"),
            Some("Q4_K_M".to_string())
        );
    }

    #[test]
    fn quant_bpw_ud_tags() {
        // plain UD- maps to the base quant
        assert_eq!(quant_bpw("UD-Q4_K_M"), Some(4.83));
        // _XL family: base bpw + 0.35
        let xl = quant_bpw("UD-Q4_K_XL").expect("UD-Q4_K_XL must have a bpw");
        assert!((xl - 5.18).abs() < 1e-9, "got {xl}");
        let q6 = quant_bpw("UD-Q6_K_XL").expect("UD-Q6_K_XL must have a bpw");
        assert!((q6 - 6.94).abs() < 1e-9, "got {q6}");
        // existing exact entries unchanged
        assert_eq!(quant_bpw("Q4_K_M"), Some(4.83));
        assert_eq!(quant_bpw("nonsense"), None);
        assert_eq!(quant_bpw("UD-nonsense"), None);
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
