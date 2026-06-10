//! Live enrichment of curated Ollama models with the real tag list of the
//! Ollama library.
//!
//! PRE-VERIFIED (live curl, 2026-06-10): the OCI `tags/list` endpoint is NOT
//! exposed (`GET https://registry.ollama.ai/v2/library/llama3.1/tags/list`
//! → 404). Tag enumeration therefore uses the public library page
//! `GET https://ollama.com/library/{base}/tags` (HTML, 200), extracting tag
//! names from the URL scheme `href="/library/{base}:{tag}"` — URL patterns
//! are far more stable than DOM structure. Treat as semi-stable best-effort:
//! errors are reported, never fatal.
//!
//! Future size enrichment (NOT used in v1, ~1 request per kept tag):
//! `GET https://registry.ollama.ai/v2/library/{base}/manifests/{tag}` with
//! `Accept: application/vnd.docker.distribution.manifest.v2+json` works
//! anonymously and returns layer sizes.

use super::hf::HttpClient;
use super::{quant_bpw, CatalogModel, CatalogVariant, RuntimeKind};
use crate::TetroError;

/// One Ollama library tag kept for a curated model size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OllamaTag {
    /// Exact library tag, e.g. `8b-instruct-q4_K_M` or plain `8b`.
    pub tag: String,
    /// Normalized catalog quant, e.g. `Q4_K_M`.
    pub quant: String,
}

/// Fetch every tag of `{base}` (e.g. "llama3.1") from the Ollama library
/// tags page, in page order, deduplicated.
pub async fn fetch_model_tags(
    http: &dyn HttpClient,
    base: &str,
) -> Result<Vec<String>, TetroError> {
    let html = http
        .get_text(&format!("https://ollama.com/library/{base}/tags"))
        .await?;
    let needle = format!("/library/{base}:");
    let mut tags: Vec<String> = Vec::new();
    for (idx, _) in html.match_indices(&needle) {
        let tail = &html[idx + needle.len()..];
        // A tag name stops at the closing quote (or any char outside the
        // Ollama tag alphabet [A-Za-z0-9._-]).
        let tag: String = tail
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
            .collect();
        if !tag.is_empty() && !tags.iter().any(|t| t == &tag) {
            tags.push(tag);
        }
    }
    Ok(tags)
}

/// Map a lowercased tag quant suffix to a catalog quant known to `quant_bpw`.
/// `q4_K_M` → `Q4_K_M`, `q8_0` → `Q8_0`, `fp16` → `F16`; unknown → None.
fn normalize_quant(suffix: &str) -> Option<String> {
    let lower = suffix.to_ascii_lowercase();
    let candidate = match lower.as_str() {
        "fp16" => "F16".to_string(),
        _ => lower.to_ascii_uppercase(),
    };
    quant_bpw(&candidate).is_some().then_some(candidate)
}

/// Pure tag selection for one curated size (e.g. "8b"):
/// - the plain `{size_prefix}` tag maps to `default_quant` (the curated
///   entry's quant — that's what the default tag aliases on the library);
/// - `{size_prefix}-…-{quant}` tags are kept when the last `-` segment is a
///   known quant (per `quant_bpw` after normalization);
/// - `-text`/`-base` (non-chat) tags and unknown quants (q4_1, q5_0, …) are
///   skipped;
/// - one tag per normalized quant, shortest tag wins (proxy for the
///   canonical `8b-instruct-q4_K_M` over longer exotic variants).
pub fn select_variant_tags(
    tags: &[String],
    size_prefix: &str,
    default_quant: &str,
) -> Vec<OllamaTag> {
    let mut out: Vec<OllamaTag> = Vec::new();
    for tag in tags {
        let quant = if tag == size_prefix {
            default_quant.to_string()
        } else {
            let Some(rest) = tag.strip_prefix(size_prefix) else {
                continue;
            };
            if !rest.starts_with('-') {
                continue;
            }
            if tag.contains("-text") || tag.contains("-base") {
                continue;
            }
            // Infallible: `rest` starts with '-', so rsplit yields a segment.
            let Some(quant) = tag.rsplit('-').next().and_then(normalize_quant) else {
                continue;
            };
            quant
        };
        match out.iter_mut().find(|t| t.quant == quant) {
            Some(existing) => {
                if tag.len() < existing.tag.len() {
                    existing.tag = tag.clone();
                }
            }
            None => out.push(OllamaTag {
                tag: tag.clone(),
                quant,
            }),
        }
    }
    out
}

/// Replace the model's variants with one per selected tag: bpw via
/// `quant_bpw`, architecture fields copied from the existing first variant,
/// file size unknown (estimates derive from params × bpw), `source_tag` set
/// to the exact library tag. No-op when `tags` is empty (offline fallback:
/// the curated default variant stands).
pub fn enrich_with_tags(model: &mut CatalogModel, tags: &[OllamaTag]) {
    if tags.is_empty() {
        return;
    }
    let Some(arch) = model.variants.first().cloned() else {
        return;
    };
    let variants: Vec<CatalogVariant> = tags
        .iter()
        .filter_map(|t| {
            quant_bpw(&t.quant).map(|bpw| CatalogVariant {
                quant: t.quant.clone(),
                bpw,
                file_size_bytes: None,
                layers: arch.layers,
                kv_heads: arch.kv_heads,
                head_dim: arch.head_dim,
                embedding_dim: arch.embedding_dim,
                runtime_compat: vec![RuntimeKind::Ollama],
                source_tag: Some(t.tag.clone()),
            })
        })
        .collect();
    if !variants.is_empty() {
        model.variants = variants;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_trait::async_trait;
    use serde_json::Value;

    use super::*;
    use crate::catalog::Source;

    struct MockHttp {
        text: HashMap<String, String>,
    }

    #[async_trait]
    impl HttpClient for MockHttp {
        async fn get_json(&self, url: &str) -> Result<Value, TetroError> {
            Err(TetroError::Network(format!("mock: no json for {url}")))
        }

        async fn get_text(&self, url: &str) -> Result<String, TetroError> {
            self.text
                .get(url)
                .cloned()
                .ok_or_else(|| TetroError::Network(format!("mock: no text for {url}")))
        }

        async fn get_range(
            &self,
            url: &str,
            _start: u64,
            _end: u64,
        ) -> Result<Vec<u8>, TetroError> {
            Err(TetroError::Network(format!("mock: no range for {url}")))
        }
    }

    /// Realistic excerpt of https://ollama.com/library/llama3.1/tags —
    /// hrefs are the load-bearing pattern, surrounding DOM is noise.
    const LLAMA31_TAGS_HTML: &str = r#"<!DOCTYPE html>
<html><body>
<a class="group" href="/library/llama3.1:latest"><span>latest</span></a>
<a class="group" href="/library/llama3.1:8b"><span>8b</span></a>
<a class="group" href="/library/llama3.1:70b"><span>70b</span></a>
<a class="group" href="/library/llama3.1:405b"><span>405b</span></a>
<a class="group" href="/library/llama3.1:8b"><span>8b duplicate row</span></a>
<a href="/library/llama3.1:8b-instruct-q4_K_M">8b-instruct-q4_K_M</a>
<a href="/library/llama3.1:8b-instruct-q8_0">8b-instruct-q8_0</a>
<a href="/library/llama3.1:8b-instruct-fp16">8b-instruct-fp16</a>
<a href="/library/llama3.1:8b-instruct-q4_1">8b-instruct-q4_1</a>
<a href="/library/llama3.1:8b-text-q4_K_M">8b-text-q4_K_M</a>
<a href="/library/llama3.1:70b-instruct-q4_K_M">70b-instruct-q4_K_M</a>
<a href="/library/mistral:7b">unrelated model link</a>
<a href="/blog/llama3">blog link</a>
</body></html>"#;

    fn http_with_llama31() -> MockHttp {
        let mut text = HashMap::new();
        text.insert(
            "https://ollama.com/library/llama3.1/tags".to_string(),
            LLAMA31_TAGS_HTML.to_string(),
        );
        MockHttp { text }
    }

    #[tokio::test]
    async fn fetch_model_tags_extracts_hrefs_deduped_in_order() {
        let http = http_with_llama31();
        let tags = fetch_model_tags(&http, "llama3.1").await.unwrap();
        assert_eq!(
            tags,
            vec![
                "latest",
                "8b",
                "70b",
                "405b",
                "8b-instruct-q4_K_M",
                "8b-instruct-q8_0",
                "8b-instruct-fp16",
                "8b-instruct-q4_1",
                "8b-text-q4_K_M",
                "70b-instruct-q4_K_M",
            ]
        );
    }

    #[tokio::test]
    async fn fetch_model_tags_propagates_network_error() {
        let http = MockHttp {
            text: HashMap::new(),
        };
        assert!(fetch_model_tags(&http, "llama3.1").await.is_err());
    }

    #[test]
    fn select_variant_tags_llama31_8b() {
        let tags: Vec<String> = [
            "latest",
            "8b",
            "70b",
            "405b",
            "8b-instruct-q4_K_M",
            "8b-instruct-q8_0",
            "8b-instruct-fp16",
            "8b-instruct-q4_1",
            "8b-text-q4_K_M",
            "70b-instruct-q4_K_M",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let kept = select_variant_tags(&tags, "8b", "Q4_K_M");
        // - plain "8b" → curated default quant, and being shortest it wins
        //   the Q4_K_M slot over "8b-instruct-q4_K_M"
        // - "-text" skipped, unknown q4_1 skipped, 70b/405b/latest excluded
        assert_eq!(
            kept,
            vec![
                OllamaTag {
                    tag: "8b".into(),
                    quant: "Q4_K_M".into()
                },
                OllamaTag {
                    tag: "8b-instruct-q8_0".into(),
                    quant: "Q8_0".into()
                },
                OllamaTag {
                    tag: "8b-instruct-fp16".into(),
                    quant: "F16".into()
                },
            ]
        );
    }

    #[test]
    fn select_variant_tags_without_plain_tag_keeps_shortest_per_quant() {
        let tags: Vec<String> = [
            "8b-instruct-2024-q4_K_M", // longer exotic variant…
            "8b-instruct-q4_K_M",      // …loses to the canonical one
            "8b-instruct-q6_K",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let kept = select_variant_tags(&tags, "8b", "Q4_K_M");
        assert_eq!(
            kept,
            vec![
                OllamaTag {
                    tag: "8b-instruct-q4_K_M".into(),
                    quant: "Q4_K_M".into()
                },
                OllamaTag {
                    tag: "8b-instruct-q6_K".into(),
                    quant: "Q6_K".into()
                },
            ]
        );
    }

    #[test]
    fn select_variant_tags_prefix_must_be_whole_segment() {
        // "8b1-q4_K_M" must not match size prefix "8b".
        let tags = vec!["8b1-instruct-q4_K_M".to_string()];
        assert!(select_variant_tags(&tags, "8b", "Q4_K_M").is_empty());
    }

    #[test]
    fn select_variant_tags_q4_0_is_a_known_quant() {
        let tags = vec!["8b-instruct-q4_0".to_string(), "8b-text-fp16".to_string()];
        let kept = select_variant_tags(&tags, "8b", "Q4_K_M");
        assert_eq!(
            kept,
            vec![OllamaTag {
                tag: "8b-instruct-q4_0".into(),
                quant: "Q4_0".into()
            }]
        );
    }

    fn curated_llama31() -> CatalogModel {
        CatalogModel {
            id: 0,
            name: "llama3.1:8b".into(),
            family: Some("llama".into()),
            source: Source::Ollama,
            repo: None,
            params_total: 8_030_000_000,
            params_active: 8_030_000_000,
            architecture: Some("llama".into()),
            context_max: 131_072,
            variants: vec![CatalogVariant {
                quant: "Q4_K_M".into(),
                bpw: 4.83,
                file_size_bytes: Some(4_920_000_000),
                layers: 32,
                kv_heads: 8,
                head_dim: 128,
                embedding_dim: 4096,
                runtime_compat: vec![RuntimeKind::Ollama],
                source_tag: None,
            }],
        }
    }

    #[test]
    fn enrich_with_tags_replaces_variants_and_copies_arch() {
        let mut m = curated_llama31();
        let tags = vec![
            OllamaTag {
                tag: "8b".into(),
                quant: "Q4_K_M".into(),
            },
            OllamaTag {
                tag: "8b-instruct-q8_0".into(),
                quant: "Q8_0".into(),
            },
        ];
        enrich_with_tags(&mut m, &tags);

        assert_eq!(m.variants.len(), 2);
        let q8 = m.variants.iter().find(|v| v.quant == "Q8_0").unwrap();
        assert_eq!(q8.source_tag.as_deref(), Some("8b-instruct-q8_0"));
        assert_eq!(q8.bpw, 8.5);
        assert_eq!(q8.file_size_bytes, None);
        assert_eq!(q8.runtime_compat, vec![RuntimeKind::Ollama]);
        // Architecture fields copied from the curated variant.
        assert_eq!(q8.layers, 32);
        assert_eq!(q8.kv_heads, 8);
        assert_eq!(q8.head_dim, 128);
        assert_eq!(q8.embedding_dim, 4096);
        // Default tag variant carries its exact tag too.
        let q4 = m.variants.iter().find(|v| v.quant == "Q4_K_M").unwrap();
        assert_eq!(q4.source_tag.as_deref(), Some("8b"));
    }

    #[test]
    fn enrich_with_empty_tags_keeps_curated_variant() {
        let mut m = curated_llama31();
        enrich_with_tags(&mut m, &[]);
        assert_eq!(m.variants.len(), 1);
        assert_eq!(m.variants[0].quant, "Q4_K_M");
        assert_eq!(m.variants[0].source_tag, None);
        assert_eq!(m.variants[0].file_size_bytes, Some(4_920_000_000));
    }

    /// Live sanity check against the real Ollama library page. Run manually:
    /// `cargo test -p tetro-core live_llama31 -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "network: hits ollama.com"]
    async fn live_llama31_tags_page_yields_tags() {
        let http = super::super::hf::ReqwestClient::new().unwrap();
        let tags = fetch_model_tags(&http, "llama3.1").await.unwrap();
        println!("live llama3.1 tags ({}): {:?}", tags.len(), tags);
        assert!(tags.len() >= 5, "expected >= 5 tags, got {tags:?}");
    }
}
