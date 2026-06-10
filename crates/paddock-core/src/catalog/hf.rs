//! HuggingFace catalog sources. All HTTP behind `HttpClient` for testing.

use async_trait::async_trait;
use serde_json::Value;

use super::{quant_bpw, quant_from_filename, CatalogModel, CatalogVariant, RuntimeKind, Source};
use crate::PaddockError;

#[async_trait]
pub trait HttpClient: Send + Sync {
    async fn get_json(&self, url: &str) -> Result<Value, PaddockError>;
    /// GET with an explicit `Accept` header, JSON body expected. Needed by
    /// OCI registries (`registry.ollama.ai` manifests require
    /// `application/vnd.docker.distribution.manifest.v2+json`); plain JSON
    /// APIs keep using `get_json`. Default impl ignores the header so test
    /// mocks only have to implement `get_json`.
    async fn get_json_with_accept(&self, url: &str, _accept: &str) -> Result<Value, PaddockError> {
        self.get_json(url).await
    }
    /// Plain GET returning the body as text (e.g. an HTML page).
    async fn get_text(&self, url: &str) -> Result<String, PaddockError>;
    /// GET with `Range: bytes=start-end` (inclusive), follows redirects.
    async fn get_range(&self, url: &str, start: u64, end: u64) -> Result<Vec<u8>, PaddockError>;
}

pub struct ReqwestClient {
    client: reqwest::Client,
}

impl ReqwestClient {
    pub fn new() -> Result<Self, PaddockError> {
        let client = reqwest::Client::builder()
            .user_agent(concat!("paddock/", env!("CARGO_PKG_VERSION")))
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| PaddockError::Network(e.to_string()))?;
        Ok(Self { client })
    }
}

#[async_trait]
impl HttpClient for ReqwestClient {
    async fn get_json(&self, url: &str) -> Result<Value, PaddockError> {
        let resp = self.client.get(url).send().await.map_err(net)?;
        if !resp.status().is_success() {
            return Err(PaddockError::Network(format!(
                "{url}: HTTP {}",
                resp.status()
            )));
        }
        resp.json().await.map_err(net)
    }

    async fn get_json_with_accept(&self, url: &str, accept: &str) -> Result<Value, PaddockError> {
        let resp = self
            .client
            .get(url)
            .header("Accept", accept)
            .send()
            .await
            .map_err(net)?;
        if !resp.status().is_success() {
            return Err(PaddockError::Network(format!(
                "{url}: HTTP {}",
                resp.status()
            )));
        }
        resp.json().await.map_err(net)
    }

    async fn get_text(&self, url: &str) -> Result<String, PaddockError> {
        let resp = self.client.get(url).send().await.map_err(net)?;
        if !resp.status().is_success() {
            return Err(PaddockError::Network(format!(
                "{url}: HTTP {}",
                resp.status()
            )));
        }
        resp.text().await.map_err(net)
    }

    async fn get_range(&self, url: &str, start: u64, end: u64) -> Result<Vec<u8>, PaddockError> {
        let resp = self
            .client
            .get(url)
            .header("Range", format!("bytes={start}-{end}"))
            .send()
            .await
            .map_err(net)?;
        // Only 206 Partial Content is acceptable: a 200 means the server
        // ignored the Range header and would stream the whole multi-GB file
        // into memory via `bytes()`.
        if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            return Err(PaddockError::Network(format!(
                "{url}: server ignored Range request (status {})",
                resp.status()
            )));
        }
        Ok(resp.bytes().await.map_err(net)?.to_vec())
    }
}

fn net(e: reqwest::Error) -> PaddockError {
    PaddockError::Network(e.to_string())
}

const HF_API: &str = "https://huggingface.co/api";
/// How many bytes of a GGUF file to fetch when the API lacks metadata.
const GGUF_HEADER_PROBE_BYTES: u64 = 2 * 1024 * 1024;

/// Sync popular GGUF repos from HuggingFace. Returns models (not yet persisted).
pub async fn fetch_hf_gguf(
    http: &dyn HttpClient,
    limit: usize,
) -> Result<Vec<CatalogModel>, PaddockError> {
    let list = http
        .get_json(&format!(
            "{HF_API}/models?filter=gguf&sort=downloads&limit={limit}"
        ))
        .await?;
    let mut out = Vec::new();
    for item in list.as_array().unwrap_or(&Vec::new()) {
        let Some(repo) = item["id"].as_str() else {
            continue;
        };
        match fetch_hf_repo(http, repo).await {
            Ok(Some(m)) => out.push(m),
            Ok(None) => {}
            Err(_) => {} // one bad repo must not kill the sync
        }
    }
    Ok(out)
}

async fn fetch_hf_repo(
    http: &dyn HttpClient,
    repo: &str,
) -> Result<Option<CatalogModel>, PaddockError> {
    let detail = http
        .get_json(&format!("{HF_API}/models/{repo}?blobs=true"))
        .await?;
    let gguf = &detail["gguf"];
    let mut architecture = gguf["architecture"].as_str().map(String::from);
    let mut context_max = gguf["context_length"].as_u64().unwrap_or(0) as u32;
    let params_total = gguf["total"].as_u64().unwrap_or(0);

    // Collect GGUF files with a recognizable quant.
    let mut files: Vec<(String, String, u64)> = Vec::new(); // (filename, quant, size)
    let mut has_mmproj = false;
    for sib in detail["siblings"].as_array().unwrap_or(&Vec::new()) {
        let Some(name) = sib["rfilename"].as_str() else {
            continue;
        };
        if !name.ends_with(".gguf") || name.contains("-of-") {
            // skip split files
            continue;
        }
        // Separate vision projector file: not a model variant, and Ollama
        // cannot import such repos via hf.co (ollama/ollama#15447).
        let last_segment = name.rsplit('/').next().unwrap_or(name).to_lowercase();
        if last_segment.starts_with("mmproj") && last_segment.ends_with(".gguf") {
            has_mmproj = true;
            continue;
        }
        if let Some(q) = quant_from_filename(name) {
            files.push((name.to_string(), q, sib["size"].as_u64().unwrap_or(0)));
        }
    }
    if files.is_empty() {
        return Ok(None);
    }

    // Architecture detail (layers, kv heads) is not in the list API: parse the
    // header of the smallest file via a Range request.
    let mut layers = 0u32;
    let mut kv_heads = 0u32;
    let mut head_dim = 0u32;
    let mut embedding_dim = 0u32;
    // Infallible: `files.is_empty()` returned `Ok(None)` above.
    let probe_file = &files.iter().min_by_key(|f| f.2).unwrap().0.clone();
    let url = format!("https://huggingface.co/{repo}/resolve/main/{probe_file}");
    if let Ok(bytes) = http.get_range(&url, 0, GGUF_HEADER_PROBE_BYTES - 1).await {
        if let Ok(meta) = super::gguf::parse_gguf_header(&bytes) {
            architecture = architecture.or(meta.architecture.clone());
            layers = meta.block_count.unwrap_or(0) as u32;
            kv_heads = meta.head_count_kv.or(meta.head_count).unwrap_or(0) as u32;
            head_dim = meta.head_dim().unwrap_or(0) as u32;
            embedding_dim = meta.embedding_length.unwrap_or(0) as u32;
            if context_max == 0 {
                context_max = meta.context_length.unwrap_or(0) as u32;
            }
        }
    }
    if layers == 0 || kv_heads == 0 || head_dim == 0 || params_total == 0 {
        return Ok(None); // cannot estimate without these — skip repo
    }

    let runtime_compat = if has_mmproj {
        vec![RuntimeKind::LlamaCpp]
    } else {
        vec![RuntimeKind::Ollama, RuntimeKind::LlamaCpp]
    };
    let variants = files
        .into_iter()
        .filter_map(|(_, quant, size)| {
            quant_bpw(&quant).map(|bpw| CatalogVariant {
                quant,
                bpw,
                file_size_bytes: (size > 0).then_some(size),
                layers,
                kv_heads,
                head_dim,
                embedding_dim,
                runtime_compat: runtime_compat.clone(),
                source_tag: None,
            })
        })
        .collect::<Vec<_>>();
    if variants.is_empty() {
        return Ok(None);
    }

    Ok(Some(CatalogModel {
        id: 0,
        name: repo.split('/').next_back().unwrap_or(repo).to_string(),
        family: architecture.clone(),
        source: Source::HuggingFace,
        repo: Some(repo.to_string()),
        params_total,
        params_active: moe_active_params(repo, params_total),
        architecture,
        context_max: if context_max == 0 { 4096 } else { context_max },
        released_at: None,
        released_approx: false,
        variants,
    }))
}

/// Known MoE active-parameter counts, keyed by substring of repo name (lowercase).
/// v0.1 heuristic; HF API does not expose active params.
fn moe_active_params(repo: &str, params_total: u64) -> u64 {
    let r = repo.to_lowercase();
    const KNOWN: &[(&str, u64)] = &[
        ("mixtral-8x7b", 12_900_000_000),
        ("mixtral-8x22b", 39_000_000_000),
        ("qwen3-30b-a3b", 3_300_000_000),
        ("qwen3-235b-a22b", 22_000_000_000),
        ("gpt-oss-20b", 3_600_000_000),
        ("gpt-oss-120b", 5_100_000_000),
        ("deepseek-v3", 37_000_000_000),
    ];
    for (pat, active) in KNOWN {
        if r.contains(pat) {
            return *active;
        }
    }
    // Generic "aXb" suffix pattern, e.g. "...-30B-A3B...". Check every "-a"
    // occurrence: org names may contain "-a" before the real suffix.
    for (idx, _) in r.match_indices("-a") {
        let tail = &r[idx + 2..];
        let digits: String = tail
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if !digits.is_empty() && tail[digits.len()..].starts_with('b') {
            if let Ok(billions) = digits.parse::<f64>() {
                return (billions * 1e9) as u64;
            }
        }
    }
    params_total
}

/// Sync mlx-community models (MLX runtime).
pub async fn fetch_mlx(
    http: &dyn HttpClient,
    limit: usize,
) -> Result<Vec<CatalogModel>, PaddockError> {
    let list = http
        .get_json(&format!(
            "{HF_API}/models?author=mlx-community&sort=downloads&limit={limit}"
        ))
        .await?;
    let mut out = Vec::new();
    for item in list.as_array().unwrap_or(&Vec::new()) {
        let Some(repo) = item["id"].as_str() else {
            continue;
        };
        let Ok(config) = http
            .get_json(&format!(
                "https://huggingface.co/{repo}/resolve/main/config.json"
            ))
            .await
        else {
            continue;
        };
        let layers = config["num_hidden_layers"].as_u64().unwrap_or(0) as u32;
        let heads = config["num_attention_heads"].as_u64().unwrap_or(0) as u32;
        let kv_heads = config["num_key_value_heads"]
            .as_u64()
            .unwrap_or(heads as u64) as u32;
        let hidden = config["hidden_size"].as_u64().unwrap_or(0) as u32;
        let context = config["max_position_embeddings"].as_u64().unwrap_or(4096) as u32;
        let bits = config["quantization"]["bits"].as_u64();
        if layers == 0 || heads == 0 || hidden == 0 {
            continue;
        }
        let Some(params_total) = params_from_name(repo) else {
            continue;
        };
        let (quant, bpw) = match bits {
            Some(4) => ("MLX_4BIT".to_string(), 4.5),
            Some(8) => ("MLX_8BIT".to_string(), 8.5),
            _ => ("F16".to_string(), 16.0),
        };
        out.push(CatalogModel {
            id: 0,
            name: repo.split('/').next_back().unwrap_or(repo).to_string(),
            family: config["model_type"].as_str().map(String::from),
            source: Source::Mlx,
            repo: Some(repo.to_string()),
            params_total,
            params_active: moe_active_params(repo, params_total),
            architecture: config["model_type"].as_str().map(String::from),
            context_max: context,
            released_at: None,
            released_approx: false,
            variants: vec![CatalogVariant {
                quant,
                bpw,
                file_size_bytes: None,
                layers,
                kv_heads,
                head_dim: if heads > 0 { hidden / heads } else { 0 },
                embedding_dim: hidden,
                runtime_compat: vec![RuntimeKind::MlxLm],
                source_tag: None,
            }],
        });
    }
    Ok(out)
}

/// Parse "...-7B-..." style parameter counts from a repo name.
fn params_from_name(repo: &str) -> Option<u64> {
    let lower = repo.to_lowercase();
    let bytes = lower.as_bytes();
    let mut best: Option<f64> = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            if i < bytes.len()
                && bytes[i] == b'b'
                && (i + 1 == bytes.len() || !bytes[i + 1].is_ascii_alphanumeric())
            {
                if let Ok(v) = lower[start..i].parse::<f64>() {
                    // MoE "NxMb" pattern (e.g. "8x7b"): the token is preceded
                    // by "<digits>x", total params are N * M billions.
                    let mut val = v;
                    if start >= 2 && bytes[start - 1] == b'x' {
                        let x = start - 1;
                        let mut k = x;
                        while k > 0 && bytes[k - 1].is_ascii_digit() {
                            k -= 1;
                        }
                        if k < x {
                            if let Ok(n) = lower[k..x].parse::<f64>() {
                                val = n * v;
                            }
                        }
                    }
                    best = Some(best.map_or(val, |b: f64| b.max(val)));
                }
            }
        } else {
            i += 1;
        }
    }
    best.map(|b| (b * 1e9) as u64)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_trait::async_trait;
    use serde_json::{json, Value};

    use super::*;
    use crate::catalog::gguf::tests::{llama_header, GgufBuilder};

    /// Mock HTTP client for unit tests.
    struct MockHttp {
        json: HashMap<String, Value>,
        text: HashMap<String, String>,
        ranges: HashMap<String, Vec<u8>>,
    }

    impl MockHttp {
        fn new() -> Self {
            Self {
                json: HashMap::new(),
                text: HashMap::new(),
                ranges: HashMap::new(),
            }
        }

        fn add_json(mut self, url: &str, val: Value) -> Self {
            self.json.insert(url.to_string(), val);
            self
        }

        fn add_range(mut self, url: &str, bytes: Vec<u8>) -> Self {
            self.ranges.insert(url.to_string(), bytes);
            self
        }
    }

    #[async_trait]
    impl HttpClient for MockHttp {
        async fn get_json(&self, url: &str) -> Result<Value, PaddockError> {
            self.json
                .get(url)
                .cloned()
                .ok_or_else(|| PaddockError::Network(format!("mock: no json for {url}")))
        }

        async fn get_text(&self, url: &str) -> Result<String, PaddockError> {
            self.text
                .get(url)
                .cloned()
                .ok_or_else(|| PaddockError::Network(format!("mock: no text for {url}")))
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
                .ok_or_else(|| PaddockError::Network(format!("mock: no range for {url}")))
        }
    }

    fn failing_http() -> MockHttp {
        MockHttp::new()
    }

    #[tokio::test]
    async fn fetch_hf_repo_full_flow() {
        let repo = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF";
        let detail_url = format!("{HF_API}/models/{repo}?blobs=true");
        let list_url = format!("{HF_API}/models?filter=gguf&sort=downloads&limit=1");
        let range_url = format!(
            "https://huggingface.co/{repo}/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        );

        let gguf_bytes = llama_header();

        let http = MockHttp::new()
            .add_json(&list_url, json!([{"id": repo}]))
            .add_json(
                &detail_url,
                json!({
                    "id": repo,
                    "gguf": {
                        "architecture": "llama",
                        "context_length": 131072,
                        "total": 8030000000u64
                    },
                    "siblings": [
                        {
                            "rfilename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                            "size": 4920000000u64
                        },
                        {
                            "rfilename": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
                            "size": 8540000000u64
                        }
                    ]
                }),
            )
            .add_range(&range_url, gguf_bytes);

        let models = fetch_hf_gguf(&http, 1).await.unwrap();
        assert_eq!(models.len(), 1);
        let m = &models[0];
        assert_eq!(m.params_total, 8030000000);
        // Both Q4_K_M and Q8_0 should appear as variants
        assert_eq!(m.variants.len(), 2);
        // The header gives 32 layers, 8 kv_heads, 128 head_dim
        for v in &m.variants {
            assert_eq!(v.layers, 32);
            assert_eq!(v.kv_heads, 8);
            assert_eq!(v.head_dim, 128);
        }
    }

    #[tokio::test]
    async fn mmproj_repo_is_llama_cpp_only() {
        // Repos shipping a separate vision projector (mmproj-*.gguf) cannot be
        // imported by Ollama via hf.co (ollama/ollama#15447).
        let repo = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF";
        let detail_url = format!("{HF_API}/models/{repo}?blobs=true");
        let list_url = format!("{HF_API}/models?filter=gguf&sort=downloads&limit=1");
        let range_url =
            format!("https://huggingface.co/{repo}/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf");

        let http = MockHttp::new()
            .add_json(&list_url, json!([{"id": repo}]))
            .add_json(
                &detail_url,
                json!({
                    "id": repo,
                    "gguf": {
                        "architecture": "qwen3moe",
                        "context_length": 262144,
                        "total": 35000000000u64
                    },
                    "siblings": [
                        {"rfilename": "mmproj-BF16.gguf", "size": 1200000000u64},
                        {"rfilename": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "size": 22000000000u64}
                    ]
                }),
            )
            .add_range(&range_url, llama_header());

        let models = fetch_hf_gguf(&http, 1).await.unwrap();
        assert_eq!(models.len(), 1);
        let m = &models[0];
        // mmproj-BF16.gguf must NOT surface as a BF16 model variant
        assert_eq!(m.variants.len(), 1);
        assert_eq!(m.variants[0].quant, "UD-Q4_K_M");
        assert_eq!(m.variants[0].runtime_compat, vec![RuntimeKind::LlamaCpp]);
    }

    #[test]
    fn params_from_name_cases() {
        assert_eq!(
            params_from_name("mlx-community/Llama-3.1-8B-Instruct-4bit"),
            Some(8_000_000_000)
        );
        assert_eq!(params_from_name("Qwen2.5-0.5B"), Some(500_000_000));
    }

    #[test]
    fn params_from_name_moe_nxmb() {
        // 8x7B is 56B total, not 7B.
        assert_eq!(
            params_from_name("mlx-community/Mixtral-8x7B-Instruct-4bit"),
            Some(56_000_000_000)
        );
    }

    #[test]
    fn moe_active_params_known_table() {
        // Qwen3-30B-A3B hits the known table
        let repo = "Qwen/Qwen3-30B-A3B-GGUF";
        let total = 30_500_000_000u64;
        let active = moe_active_params(repo, total);
        assert_eq!(active, 3_300_000_000);
    }

    #[test]
    fn moe_active_params_generic_suffix() {
        // Generic "-a3b" style not in known table
        let repo = "some-org/SomeMoE-20B-A3B-GGUF";
        let total = 20_000_000_000u64;
        let active = moe_active_params(repo, total);
        assert_eq!(active, 3_000_000_000);
    }

    #[test]
    fn moe_active_params_org_name_containing_a_dash() {
        // "-a" in the org name must not shadow the real "-A3B" suffix.
        let repo = "meta-ai/SomeMoE-20B-A3B-GGUF";
        let total = 20_000_000_000u64;
        let active = moe_active_params(repo, total);
        assert_eq!(active, 3_000_000_000);
    }

    #[tokio::test]
    async fn fetch_mlx_basic() {
        let list_url = format!("{HF_API}/models?author=mlx-community&sort=downloads&limit=1");
        let repo = "mlx-community/Llama-3.1-8B-Instruct-4bit";
        let config_url = format!("https://huggingface.co/{repo}/resolve/main/config.json");

        let http = MockHttp::new()
            .add_json(&list_url, json!([{"id": repo}]))
            .add_json(
                &config_url,
                json!({
                    "model_type": "llama",
                    "num_hidden_layers": 32,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "hidden_size": 4096,
                    "max_position_embeddings": 131072,
                    "quantization": {"bits": 4}
                }),
            );

        let models = fetch_mlx(&http, 1).await.unwrap();
        assert_eq!(models.len(), 1);
        let m = &models[0];
        assert_eq!(m.params_total, 8_000_000_000);
        let v = &m.variants[0];
        assert_eq!(v.quant, "MLX_4BIT");
        assert_eq!(v.bpw, 4.5);
        assert_eq!(v.layers, 32);
        assert_eq!(v.kv_heads, 8);
        assert_eq!(v.head_dim, 128); // 4096 / 32
    }

    #[tokio::test]
    async fn bad_repo_does_not_kill_sync() {
        // One repo has bad data (missing gguf field), other is fine
        let list_url = format!("{HF_API}/models?filter=gguf&sort=downloads&limit=2");
        let good_repo = "bartowski/Qwen2.5-7B-Instruct-GGUF";
        let bad_repo = "bad/broken-GGUF";
        let good_detail_url = format!("{HF_API}/models/{good_repo}?blobs=true");
        let range_url = format!(
            "https://huggingface.co/{good_repo}/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        );

        let gguf_bytes = GgufBuilder::new()
            .string("general.architecture", "qwen2")
            .u32("qwen2.block_count", 28)
            .u32("qwen2.attention.head_count", 28)
            .u32("qwen2.attention.head_count_kv", 4)
            .u32("qwen2.embedding_length", 3584)
            .u32("qwen2.context_length", 131072)
            .build();

        let http = MockHttp::new()
            .add_json(&list_url, json!([{"id": bad_repo}, {"id": good_repo}]))
            // bad_repo has no detail url registered → get_json returns error → skipped
            .add_json(
                &good_detail_url,
                json!({
                    "id": good_repo,
                    "gguf": {
                        "architecture": "qwen2",
                        "context_length": 131072,
                        "total": 7620000000u64
                    },
                    "siblings": [
                        {"rfilename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "size": 4680000000u64}
                    ]
                }),
            )
            .add_range(&range_url, gguf_bytes);

        let models = fetch_hf_gguf(&http, 2).await.unwrap();
        assert_eq!(
            models.len(),
            1,
            "bad repo should be skipped, good repo should pass"
        );
        assert_eq!(models[0].params_total, 7620000000);
    }

    #[tokio::test]
    async fn fetch_hf_gguf_failing_http_returns_empty() {
        let http = failing_http();
        // failing_http has no list url registered; fetch_hf_gguf returns error from outer call
        let result = fetch_hf_gguf(&http, 5).await;
        assert!(result.is_err());
    }
}
