pub mod db;
pub mod gguf;
// curated and hf added in later sub-tasks

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
