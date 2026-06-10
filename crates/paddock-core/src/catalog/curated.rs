//! Curated list of popular Ollama library models. The Ollama site has no
//! public API and scraping is brittle; any HF GGUF repo remains runnable via
//! `ollama run hf.co/{repo}:{quant}` regardless of this list.

use super::{quant_bpw, CatalogModel, CatalogVariant, RuntimeKind, Source};

#[derive(serde::Deserialize)]
struct Entry {
    name: String,
    family: String,
    params_total: u64,
    #[serde(default)]
    params_active: Option<u64>,
    context_max: u32,
    layers: u32,
    kv_heads: u32,
    head_dim: u32,
    embedding_dim: u32,
    quant: String,
    file_size_bytes: u64,
}

// NOTE (curated_ollama.json cannot carry comments): `deepseek-coder-v2:16b`
// uses MLA (Multi-head Latent Attention); its entry encodes the
// GQA-equivalent shape (kv_heads=16, head_dim=128), so the GQA KV-cache math
// overestimates the real MLA compressed-KV footprint by roughly 7x. The
// error is in the conservative direction (never under-provisions memory),
// so the numbers stay as published; modeling it exactly would take an
// MLA-aware KV estimator.
pub fn curated_ollama_models() -> Vec<CatalogModel> {
    let entries: Vec<Entry> = serde_json::from_str(include_str!("curated_ollama.json"))
        .expect("embedded curated_ollama.json must be valid (checked by unit test)");
    entries
        .into_iter()
        .map(|e| CatalogModel {
            id: 0,
            name: e.name,
            family: Some(e.family.clone()),
            source: Source::Ollama,
            repo: None,
            params_total: e.params_total,
            params_active: e.params_active.unwrap_or(e.params_total),
            architecture: Some(e.family),
            context_max: e.context_max,
            released_at: None,
            released_approx: false,
            variants: vec![CatalogVariant {
                bpw: quant_bpw(&e.quant).unwrap_or(4.83),
                quant: e.quant,
                file_size_bytes: Some(e.file_size_bytes),
                layers: e.layers,
                kv_heads: e.kv_heads,
                head_dim: e.head_dim,
                embedding_dim: e.embedding_dim,
                runtime_compat: vec![RuntimeKind::Ollama],
                source_tag: None,
            }],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_count_and_validity() {
        let models = curated_ollama_models();
        assert!(
            models.len() >= 70,
            "expected >= 70 entries, got {}",
            models.len()
        );
        for m in &models {
            let v = &m.variants[0];
            assert!(v.layers > 0, "model {} has layers=0", m.name);
            assert!(v.kv_heads > 0, "model {} has kv_heads=0", m.name);
            assert!(v.bpw > 0.0, "model {} has bpw=0", m.name);
        }
        // Every raw entry must use a known quant: a typo'd quant must fail
        // here instead of silently falling back to the runtime default bpw.
        let entries: Vec<Entry> =
            serde_json::from_str(include_str!("curated_ollama.json")).unwrap();
        for e in &entries {
            assert!(
                quant_bpw(&e.quant).is_some(),
                "model {} has unknown quant {}",
                e.name,
                e.quant
            );
        }
    }

    #[test]
    fn spot_check_llama31_8b() {
        let models = curated_ollama_models();
        let m = models.iter().find(|m| m.name == "llama3.1:8b").unwrap();
        // From meta-llama/Llama-3.1-8B-Instruct config.json
        let v = &m.variants[0];
        assert_eq!(v.layers, 32);
        assert_eq!(v.kv_heads, 8);
        assert_eq!(v.head_dim, 128);
        assert_eq!(v.embedding_dim, 4096);
        assert_eq!(m.context_max, 131072);
    }

    #[test]
    fn spot_check_qwen25_7b() {
        let models = curated_ollama_models();
        let m = models.iter().find(|m| m.name == "qwen2.5:7b").unwrap();
        // From Qwen/Qwen2.5-7B-Instruct config.json
        let v = &m.variants[0];
        assert_eq!(v.layers, 28);
        assert_eq!(v.kv_heads, 4);
        assert_eq!(v.embedding_dim, 3584);
        assert_eq!(m.context_max, 131072);
    }

    #[test]
    fn spot_check_starcoder2_15b() {
        let models = curated_ollama_models();
        let m = models.iter().find(|m| m.name == "starcoder2:15b").unwrap();
        // From bigcode/starcoder2-15b config.json
        let v = &m.variants[0];
        assert_eq!(v.layers, 40);
        assert_eq!(v.kv_heads, 4);
        assert_eq!(v.head_dim, 128);
        assert_eq!(v.embedding_dim, 6144);
        assert_eq!(m.context_max, 16384);
    }

    #[test]
    fn spot_check_qwen3_235b_moe_active_params() {
        let models = curated_ollama_models();
        let m = models.iter().find(|m| m.name == "qwen3:235b-a22b").unwrap();
        assert_eq!(m.params_total, 235_000_000_000);
        assert_eq!(m.params_active, 22_000_000_000);
    }

    #[test]
    fn spot_check_gemma2_9b() {
        let models = curated_ollama_models();
        let m = models.iter().find(|m| m.name == "gemma2:9b").unwrap();
        // From google/gemma-2-9b config.json
        let v = &m.variants[0];
        assert_eq!(v.layers, 42);
        assert_eq!(v.kv_heads, 8);
        assert_eq!(v.head_dim, 256);
        assert_eq!(v.embedding_dim, 3584);
        assert_eq!(m.context_max, 8192);
    }
}
