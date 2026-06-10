pub mod curated;
pub mod dates;
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
    /// Release date (epoch seconds), when known. Display + age malus input.
    #[serde(default)]
    pub released_at: Option<i64>,
    /// True when the date is a lower-bound proxy (Ollama tags page), not exact.
    #[serde(default)]
    pub released_approx: bool,
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

/// Options for `sync`. Limits keep first sync fast; raise via CLI if wanted.
pub struct SyncOptions {
    pub hf_limit: usize,
    pub mlx_limit: usize,
    /// Enrich curated Ollama models with the real library tag list
    /// (one request per model base name; best-effort).
    pub ollama_registry: bool,
    /// Auto-discover this many top library models beyond the curated set
    /// (tags page + manifest + 256 KiB GGUF probe each; best-effort).
    /// None disables discovery.
    pub discover_limit: Option<usize>,
}

impl Default for SyncOptions {
    fn default() -> Self {
        Self {
            hf_limit: 100,
            mlx_limit: 60,
            ollama_registry: true,
            discover_limit: Some(60),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct SyncReport {
    pub curated: usize,
    /// Curated variants enriched with an exact Ollama library tag.
    pub ollama_tags: usize,
    /// Uncurated Ollama library models discovered via registry manifests.
    pub discovered: usize,
    pub huggingface: usize,
    pub mlx: usize,
    pub errors: Vec<String>,
}

/// Idempotent catalog sync: curated list always, network sources best-effort.
pub async fn sync(
    http: &dyn hf::HttpClient,
    db: &db::Db,
    opts: &SyncOptions,
) -> Result<SyncReport, crate::PaddockError> {
    let mut report = SyncReport::default();
    let mut curated_models = curated::curated_ollama_models();
    // Enrich IN MEMORY first, then upsert each model exactly once. Upserting
    // the single-variant baseline before enrichment would prune previously
    // enriched variants from the DB whenever the registry is unreachable.
    let mut preserve = std::collections::HashSet::new();
    if opts.ollama_registry {
        enrich_curated_with_registry(http, db, &mut curated_models, &mut preserve, &mut report)
            .await;
    } else {
        // Registry disabled: no tag data this sync. Models already in the DB
        // may carry enriched variants from a previous sync; a baseline upsert
        // would prune them, so preserve every existing model and only insert
        // the curated baseline for absent ones.
        for m in &curated_models {
            match db.model_exists(m.source, &m.name) {
                Ok(true) => {
                    preserve.insert(m.name.clone());
                }
                Ok(false) => {}
                Err(e) => report
                    .errors
                    .push(format!("curated lookup {}: {e}", m.name)),
            }
        }
    }
    for m in &curated_models {
        if preserve.contains(&m.name) {
            // Preserved as-is (no registry data this sync): still curated.
            report.curated += 1;
            continue;
        }
        match db.upsert_model(m) {
            Ok(_) => {
                report.curated += 1;
                report.ollama_tags += m.variants.iter().filter(|v| v.source_tag.is_some()).count();
            }
            Err(e) => report
                .errors
                .push(format!("curated upsert {}: {e}", m.name)),
        }
    }
    // One clock read per sync: anchors discovered models' relative dates AND
    // the last-sync stamp below.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    if let Some(limit) = opts.discover_limit {
        discover_library_models(http, db, &curated_models, limit, now, &mut report).await;
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
    db.set_last_sync(now)?;
    Ok(report)
}

/// Best-effort live tag enrichment, IN MEMORY ONLY (the caller upserts): one
/// tags-page fetch per curated base name (`llama3.1` for `llama3.1:8b` and
/// `llama3.1:70b`), then per curated size keep one tag per known quant and
/// replace the model's variants with `source_tag` set.
///
/// On a per-base fetch failure the error lands in `report.errors` and each
/// affected model already present in the DB is added to `preserve`: its
/// existing (possibly enriched) row must not be degraded to the
/// single-variant baseline. Models absent from the DB are NOT preserved, so
/// a first sync while offline still inserts the curated baseline.
async fn enrich_curated_with_registry(
    http: &dyn hf::HttpClient,
    db: &db::Db,
    curated_models: &mut [CatalogModel],
    preserve: &mut std::collections::HashSet<String>,
    report: &mut SyncReport,
) {
    // Base names in curated order, deduplicated (entries lacking ':' have no
    // library page to consult and are skipped).
    let mut bases: Vec<String> = Vec::new();
    for m in curated_models.iter() {
        if let Some((base, _)) = m.name.split_once(':') {
            if !bases.iter().any(|b| b == base) {
                bases.push(base.to_string());
            }
        }
    }
    for base in &bases {
        let tags = match ollama_registry::fetch_model_tags(http, base).await {
            Ok(t) => t,
            Err(e) => {
                report.errors.push(format!("ollama tags {base}: {e}"));
                for m in curated_models.iter() {
                    if m.name.split_once(':').map(|(b, _)| b) != Some(base) {
                        continue;
                    }
                    match db.model_exists(m.source, &m.name) {
                        Ok(true) => {
                            preserve.insert(m.name.clone());
                        }
                        Ok(false) => {}
                        Err(e) => report
                            .errors
                            .push(format!("ollama preserve {}: {e}", m.name)),
                    }
                }
                continue;
            }
        };
        for m in curated_models.iter_mut() {
            let Some((b, size)) = m.name.split_once(':') else {
                continue;
            };
            if b != base {
                continue;
            }
            let Some(default_quant) = m.variants.first().map(|v| v.quant.clone()) else {
                continue;
            };
            let selected = ollama_registry::select_variant_tags(&tags, size, &default_quant);
            // Empty selection: the registry answered but lists no usable tag
            // for this size — the curated baseline stands (and is upserted).
            ollama_registry::enrich_with_tags(m, &selected);
        }
    }
}

/// Best-effort discovery of uncurated Ollama library models: index page in
/// popularity order, minus the curated base names, first `limit` entries.
/// Each candidate costs a tags page + one manifest + one 256 KiB header
/// probe; failures are reported per model and never fatal.
async fn discover_library_models(
    http: &dyn hf::HttpClient,
    db: &db::Db,
    curated_models: &[CatalogModel],
    limit: usize,
    now: i64,
    report: &mut SyncReport,
) {
    let index = match ollama_registry::fetch_library_index(http).await {
        Ok(names) => names,
        Err(e) => {
            report.errors.push(format!("ollama discovery index: {e}"));
            return;
        }
    };
    let curated_bases: std::collections::HashSet<&str> = curated_models
        .iter()
        .map(|m| m.name.split(':').next().unwrap_or(&m.name))
        .collect();
    for name in index
        .iter()
        .filter(|n| !curated_bases.contains(n.as_str()))
        .take(limit)
    {
        match ollama_registry::discover_model(http, name, now).await {
            Ok(Some(m)) => match db.upsert_model(&m) {
                Ok(_) => report.discovered += 1,
                Err(e) => report
                    .errors
                    .push(format!("discover upsert {}: {e}", m.name)),
            },
            Ok(None) => {} // deliberately skipped (cloud-only, embeddings, …)
            Err(e) => report.errors.push(format!("discover {name}: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use serde_json::Value;

    use super::*;
    use crate::PaddockError;

    /// MockHttp that always fails.
    struct FailingHttp;

    #[async_trait]
    impl hf::HttpClient for FailingHttp {
        async fn get_json(&self, url: &str) -> Result<Value, PaddockError> {
            Err(PaddockError::Network(format!("mock failure: {url}")))
        }

        async fn get_text(&self, url: &str) -> Result<String, PaddockError> {
            Err(PaddockError::Network(format!("mock failure: {url}")))
        }

        async fn get_range(
            &self,
            url: &str,
            _start: u64,
            _end: u64,
        ) -> Result<Vec<u8>, PaddockError> {
            Err(PaddockError::Network(format!("mock failure: {url}")))
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
            report.curated >= 70,
            "expected >= 70 curated, got {}",
            report.curated
        );
        // Registry down: zero enriched variants, per-base errors reported,
        // curated data stands untouched.
        assert_eq!(report.ollama_tags, 0);
        assert!(
            report.errors.iter().any(|e| e.contains("ollama tags")),
            "expected per-base registry errors, got {:?}",
            report.errors
        );
        // Discovery degrades to a single index error, nothing discovered.
        assert_eq!(report.discovered, 0);
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("ollama discovery index")));
        assert!(report.errors.iter().any(|e| e.contains("huggingface")));
        assert!(report.errors.iter().any(|e| e.contains("mlx")));
        // last_sync set even on network failure
        let ts = db.last_sync().unwrap();
        assert!(ts.is_some(), "last_sync should be set");
        assert!(ts.unwrap() > 0);

        // Curated models actually in DB, with the offline single variant.
        let models = db.list_models().unwrap();
        assert!(models.len() >= 70, "expected >= 70 models in DB");
        let m = models.iter().find(|m| m.name == "llama3.1:8b").unwrap();
        assert_eq!(m.variants.len(), 1);
        assert_eq!(m.variants[0].source_tag, None);
    }

    #[tokio::test]
    async fn sync_with_registry_disabled_skips_tag_fetches() {
        let dir = tempfile::tempdir().unwrap();
        let db = db::Db::open(dir.path().join("catalog.db")).unwrap();
        let http = FailingHttp;
        let opts = SyncOptions {
            ollama_registry: false,
            discover_limit: None, // isolate the registry knob
            ..SyncOptions::default()
        };

        let report = sync(&http, &db, &opts).await.unwrap();

        assert_eq!(report.ollama_tags, 0);
        // Only the two non-registry network sources may fail.
        assert_eq!(
            report.errors.len(),
            2,
            "expected exactly hf+mlx errors, got {:?}",
            report.errors
        );
    }

    /// MockHttp serving tag pages for some bases; everything else fails.
    struct PagesHttp {
        pages: std::collections::HashMap<String, String>,
        calls: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl hf::HttpClient for PagesHttp {
        async fn get_json(&self, url: &str) -> Result<Value, PaddockError> {
            Err(PaddockError::Network(format!("mock failure: {url}")))
        }

        async fn get_text(&self, url: &str) -> Result<String, PaddockError> {
            self.calls.lock().unwrap().push(url.to_string());
            self.pages
                .get(url)
                .cloned()
                .ok_or_else(|| PaddockError::Network(format!("mock failure: {url}")))
        }

        async fn get_range(
            &self,
            url: &str,
            _start: u64,
            _end: u64,
        ) -> Result<Vec<u8>, PaddockError> {
            Err(PaddockError::Network(format!("mock failure: {url}")))
        }
    }

    #[tokio::test]
    async fn sync_enriches_curated_with_registry_tags() {
        let dir = tempfile::tempdir().unwrap();
        let db = db::Db::open(dir.path().join("catalog.db")).unwrap();

        // llama3.1 covers two curated sizes (8b, 70b) sharing one base;
        // qwen3 is a distinct base where only the 8b size has variant tags.
        let llama_page = r#"
            <a href="/library/llama3.1:8b">8b</a>
            <a href="/library/llama3.1:8b-instruct-q4_K_M">x</a>
            <a href="/library/llama3.1:8b-instruct-q8_0">x</a>
            <a href="/library/llama3.1:70b-instruct-q4_K_M">x</a>
            <a href="/library/llama3.1:70b-instruct-q6_K">x</a>
        "#;
        let qwen_page = r#"
            <a href="/library/qwen3:8b-q4_K_M">x</a>
            <a href="/library/qwen3:8b-fp16">x</a>
        "#;
        let mut pages = std::collections::HashMap::new();
        pages.insert(
            "https://ollama.com/library/llama3.1/tags".to_string(),
            llama_page.to_string(),
        );
        pages.insert(
            "https://ollama.com/library/qwen3/tags".to_string(),
            qwen_page.to_string(),
        );
        let http = PagesHttp {
            pages,
            calls: std::sync::Mutex::new(Vec::new()),
        };

        let report = sync(&http, &db, &SyncOptions::default()).await.unwrap();

        // 2 variants each for llama3.1:8b, llama3.1:70b and qwen3:8b.
        assert_eq!(report.ollama_tags, 6, "errors: {:?}", report.errors);

        let models = db.list_models().unwrap();
        let m8 = models.iter().find(|m| m.name == "llama3.1:8b").unwrap();
        assert_eq!(m8.variants.len(), 2);
        // Plain `8b` tag aliases the curated default quant.
        let q4 = m8.variants.iter().find(|v| v.quant == "Q4_K_M").unwrap();
        assert_eq!(q4.source_tag.as_deref(), Some("8b"));
        let q8 = m8.variants.iter().find(|v| v.quant == "Q8_0").unwrap();
        assert_eq!(q8.source_tag.as_deref(), Some("8b-instruct-q8_0"));
        // Architecture copied from the curated entry.
        assert_eq!(q8.layers, 32);
        assert_eq!(q8.embedding_dim, 4096);

        let m70 = models.iter().find(|m| m.name == "llama3.1:70b").unwrap();
        assert_eq!(m70.variants.len(), 2);
        assert!(m70.variants.iter().all(|v| v.source_tag.is_some()));

        let q3 = models.iter().find(|m| m.name == "qwen3:8b").unwrap();
        assert_eq!(q3.variants.len(), 2);
        assert!(q3
            .variants
            .iter()
            .any(|v| v.source_tag.as_deref() == Some("8b-fp16")));

        // Other qwen3 sizes had no matching tags: curated variant stands.
        let q14 = models.iter().find(|m| m.name == "qwen3:14b").unwrap();
        assert_eq!(q14.variants.len(), 1);
        assert_eq!(q14.variants[0].source_tag, None);

        // One fetch per base, even with two curated llama3.1 sizes.
        let calls = http.calls.lock().unwrap();
        let llama_calls = calls
            .iter()
            .filter(|u| u.ends_with("/library/llama3.1/tags"))
            .count();
        assert_eq!(llama_calls, 1, "expected one fetch per base");

        // Unserved bases were reported, not fatal.
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("ollama tags gemma3")));
    }

    #[tokio::test]
    async fn resync_with_registry_down_preserves_enriched_variants() {
        let dir = tempfile::tempdir().unwrap();
        let db = db::Db::open(dir.path().join("catalog.db")).unwrap();

        // First sync: registry serves llama3.1 tags → enriched variants land
        // in the DB.
        let llama_page = r#"
            <a href="/library/llama3.1:8b-instruct-q4_K_M">x</a>
            <a href="/library/llama3.1:8b-instruct-q8_0">x</a>
        "#;
        let mut pages = std::collections::HashMap::new();
        pages.insert(
            "https://ollama.com/library/llama3.1/tags".to_string(),
            llama_page.to_string(),
        );
        let http = PagesHttp {
            pages,
            calls: std::sync::Mutex::new(Vec::new()),
        };
        sync(&http, &db, &SyncOptions::default()).await.unwrap();
        let models = db.list_models().unwrap();
        let m = models.iter().find(|m| m.name == "llama3.1:8b").unwrap();
        assert_eq!(m.variants.len(), 2, "precondition: enriched in DB");

        // Re-sync with the registry fully down: the previously enriched
        // variants must survive (no degradation to single-variant baseline).
        let report = sync(&FailingHttp, &db, &SyncOptions::default())
            .await
            .unwrap();
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("ollama tags llama3.1")),
            "expected per-base registry error, got {:?}",
            report.errors
        );

        let models = db.list_models().unwrap();
        let m = models.iter().find(|m| m.name == "llama3.1:8b").unwrap();
        assert_eq!(
            m.variants.len(),
            2,
            "enriched variants must survive a registry-down re-sync"
        );
        assert!(m
            .variants
            .iter()
            .any(|v| v.source_tag.as_deref() == Some("8b-instruct-q8_0")));
    }

    #[tokio::test]
    async fn resync_with_registry_disabled_preserves_enriched_variants() {
        let dir = tempfile::tempdir().unwrap();
        let db = db::Db::open(dir.path().join("catalog.db")).unwrap();

        // First sync enriches llama3.1:8b from the registry.
        let llama_page = r#"
            <a href="/library/llama3.1:8b-instruct-q4_K_M">x</a>
            <a href="/library/llama3.1:8b-instruct-q8_0">x</a>
        "#;
        let mut pages = std::collections::HashMap::new();
        pages.insert(
            "https://ollama.com/library/llama3.1/tags".to_string(),
            llama_page.to_string(),
        );
        let http = PagesHttp {
            pages,
            calls: std::sync::Mutex::new(Vec::new()),
        };
        sync(&http, &db, &SyncOptions::default()).await.unwrap();

        // Re-sync with `--no-ollama-registry`: existing rows preserved.
        let opts = SyncOptions {
            ollama_registry: false,
            ..SyncOptions::default()
        };
        let report = sync(&FailingHttp, &db, &opts).await.unwrap();
        assert!(report.curated >= 70, "preserved models still count");

        let models = db.list_models().unwrap();
        let m = models.iter().find(|m| m.name == "llama3.1:8b").unwrap();
        assert_eq!(m.variants.len(), 2);
        assert!(m
            .variants
            .iter()
            .any(|v| v.source_tag.as_deref() == Some("8b-instruct-q8_0")));
    }

    #[tokio::test]
    async fn registry_down_still_inserts_models_absent_from_db() {
        // First-sync-offline case: nothing in the DB yet, every registry
        // fetch fails → the curated baseline must still be inserted.
        let dir = tempfile::tempdir().unwrap();
        let db = db::Db::open(dir.path().join("catalog.db")).unwrap();

        let report = sync(&FailingHttp, &db, &SyncOptions::default())
            .await
            .unwrap();
        assert!(report.curated >= 70);

        let models = db.list_models().unwrap();
        let m = models.iter().find(|m| m.name == "llama3.1:8b").unwrap();
        assert_eq!(m.variants.len(), 1);
        assert_eq!(m.variants[0].source_tag, None);
    }

    /// MockHttp serving text pages, JSON manifests and blob ranges; anything
    /// unregistered fails like a network error.
    #[derive(Default)]
    struct DiscoveryHttp {
        text: std::collections::HashMap<String, String>,
        json: std::collections::HashMap<String, Value>,
        ranges: std::collections::HashMap<String, Vec<u8>>,
    }

    #[async_trait]
    impl hf::HttpClient for DiscoveryHttp {
        async fn get_json(&self, url: &str) -> Result<Value, PaddockError> {
            self.json
                .get(url)
                .cloned()
                .ok_or_else(|| PaddockError::Network(format!("mock failure: {url}")))
        }

        async fn get_text(&self, url: &str) -> Result<String, PaddockError> {
            self.text
                .get(url)
                .cloned()
                .ok_or_else(|| PaddockError::Network(format!("mock failure: {url}")))
        }

        async fn get_range(
            &self,
            url: &str,
            _start: u64,
            _end: u64,
        ) -> Result<Vec<u8>, PaddockError> {
            self.ranges
                .get(url)
                .cloned()
                .ok_or_else(|| PaddockError::Network(format!("mock failure: {url}")))
        }
    }

    /// Register the full discovery chain for one uncurated model.
    fn add_lfm25_fixtures(http: &mut DiscoveryHttp) {
        http.text.insert(
            "https://ollama.com/library/lfm2.5/tags".to_string(),
            r#"
                <a href="/library/lfm2.5:1.2b">1.2b</a>
                <a href="/library/lfm2.5:1.2b-q8_0">x</a>
            "#
            .to_string(),
        );
        http.json.insert(
            "https://registry.ollama.ai/v2/library/lfm2.5/manifests/1.2b".to_string(),
            serde_json::json!({
                "schemaVersion": 2,
                "layers": [{
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": "sha256:blob",
                    "size": 736_000_000u64
                }]
            }),
        );
        http.ranges.insert(
            "https://registry.ollama.ai/v2/library/lfm2.5/blobs/sha256:blob".to_string(),
            gguf::tests::GgufBuilder::new()
                .string("general.architecture", "lfm2moe")
                .u64("general.parameter_count", 1_170_000_000)
                .u32("lfm2moe.block_count", 16)
                .u32("lfm2moe.attention.head_count", 16)
                .u32("lfm2moe.attention.head_count_kv", 4)
                .u32("lfm2moe.embedding_length", 2048)
                .u32("lfm2moe.context_length", 32768)
                .build(),
        );
    }

    #[tokio::test]
    async fn sync_discovers_uncurated_library_models() {
        let dir = tempfile::tempdir().unwrap();
        let db = db::Db::open(dir.path().join("catalog.db")).unwrap();

        // Index lists a curated base (llama3.1 — must be skipped, no fixtures
        // needed) and one uncurated model with a full discovery chain.
        let mut http = DiscoveryHttp::default();
        http.text.insert(
            "https://ollama.com/library".to_string(),
            r#"
                <a href="/library/llama3.1"><h2>llama3.1</h2></a>
                <a href="/library/lfm2.5"><h2>lfm2.5</h2></a>
            "#
            .to_string(),
        );
        add_lfm25_fixtures(&mut http);

        let report = sync(&http, &db, &SyncOptions::default()).await.unwrap();

        assert_eq!(report.discovered, 1, "errors: {:?}", report.errors);
        // Curated base on the index must not be re-discovered (its manifest
        // fixture does not exist, so an attempt would surface as an error).
        assert!(
            !report
                .errors
                .iter()
                .any(|e| e.contains("discover llama3.1")),
            "curated bases must be excluded from discovery: {:?}",
            report.errors
        );

        let models = db.list_models().unwrap();
        let m = models.iter().find(|m| m.name == "lfm2.5:1.2b").unwrap();
        assert_eq!(m.source, Source::Ollama);
        assert_eq!(m.architecture.as_deref(), Some("lfm2moe"));
        assert_eq!(m.params_total, 1_170_000_000);
        assert_eq!(m.variants.len(), 2);
        assert!(m.variants.iter().all(|v| v.source_tag.is_some()));
        // Fixture tags page carries no relative dates → no release proxy.
        assert_eq!(m.released_at, None);
        assert!(!m.released_approx);
    }

    #[tokio::test]
    async fn sync_discover_limit_caps_attempts() {
        let dir = tempfile::tempdir().unwrap();
        let db = db::Db::open(dir.path().join("catalog.db")).unwrap();

        // Two uncurated names; limit 1 → the second (fixture-less, would
        // error if attempted) must never be fetched.
        let mut http = DiscoveryHttp::default();
        http.text.insert(
            "https://ollama.com/library".to_string(),
            r#"
                <a href="/library/lfm2.5"><h2>lfm2.5</h2></a>
                <a href="/library/somemodel"><h2>somemodel</h2></a>
            "#
            .to_string(),
        );
        add_lfm25_fixtures(&mut http);
        let opts = SyncOptions {
            discover_limit: Some(1),
            ..SyncOptions::default()
        };

        let report = sync(&http, &db, &opts).await.unwrap();

        assert_eq!(report.discovered, 1, "errors: {:?}", report.errors);
        assert!(
            !report
                .errors
                .iter()
                .any(|e| e.contains("discover somemodel")),
            "limit must cap discovery attempts: {:?}",
            report.errors
        );
    }
}
