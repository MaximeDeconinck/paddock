//! Live enrichment of curated Ollama models with the real tag list of the
//! Ollama library.
//!
//! PRE-VERIFIED (live curl, 2026-06-10): the OCI `tags/list` endpoint is NOT
//! exposed (`GET https://registry.ollama.ai/v2/library/llama3.1/tags/list`
//! → 404). Tag enumeration therefore uses the public library page
//! `GET https://ollama.com/library/{base}/tags` (HTML, 200), extracting tag
//! names from the URL scheme `href="/library/{base}:{tag}"` - URL patterns
//! are far more stable than DOM structure. Treat as semi-stable best-effort:
//! errors are reported, never fatal.
//!
//! Future size enrichment (NOT used in v1, ~1 request per kept tag):
//! `GET https://registry.ollama.ai/v2/library/{base}/manifests/{tag}` with
//! `Accept: application/vnd.docker.distribution.manifest.v2+json` works
//! anonymously and returns layer sizes.

use super::hf::HttpClient;
use super::{CatalogModel, CatalogVariant, RuntimeKind, Source, quant_bpw};
use crate::PaddockError;

const REGISTRY: &str = "https://registry.ollama.ai/v2/library";
/// PRE-VERIFIED (live curl, 2026-06-10): manifests require this Accept header
/// and answer anonymously with the layer list (digest + size per layer).
const MANIFEST_ACCEPT: &str = "application/vnd.docker.distribution.manifest.v2+json";
/// PRE-VERIFIED (live curl, 2026-06-10): a blob GET with `Range:
/// bytes=0-262143` returns HTTP 206 (after a redirect reqwest follows by
/// default) whose first bytes are a parseable GGUF v3 header.
const BLOB_PROBE_BYTES: u64 = 256 * 1024;
/// Plain library tags (`8b`) conventionally alias the q4_K_M build; the
/// manifest does not name the quant, so discovery assumes the convention.
const DEFAULT_LIBRARY_QUANT: &str = "Q4_K_M";
/// Embeddings/vision architectures are out of scope for a text-gen catalog.
const ARCH_SKIPLIST: &[&str] = &["bert", "nomic-bert", "clip"];

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
) -> Result<Vec<String>, PaddockError> {
    let html = http
        .get_text(&format!("https://ollama.com/library/{base}/tags"))
        .await?;
    Ok(extract_tag_names(&html, base))
}

/// Pure extraction of every `{base}` tag from a library tags page, in page
/// order, deduplicated.
fn extract_tag_names(html: &str, base: &str) -> Vec<String> {
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
    tags
}

/// Extract every `{name}` from `href="/library/{name}"` in page order,
/// stripping any `:tag` suffix, deduplicated (first occurrence wins).
///
/// PRE-VERIFIED (live curl, 2026-06-10): `https://ollama.com/library` exposes
/// 234 model names via the URL pattern `href="/library/{name}"` - same
/// semi-stable URL-scheme extraction as `fetch_model_tags`.
pub fn parse_library_names(html: &str) -> Vec<String> {
    let needle = "href=\"/library/";
    let mut names: Vec<String> = Vec::new();
    for (idx, _) in html.match_indices(needle) {
        let tail = &html[idx + needle.len()..];
        // A name stops at any char outside the Ollama name alphabet - note
        // ':' is excluded, so tag links (`/library/{name}:{tag}`) still yield
        // the bare name and deduplicate away.
        let name: String = tail
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
            .collect();
        if !name.is_empty() && !names.iter().any(|n| n == &name) {
            names.push(name);
        }
    }
    names
}

/// Union popularity + newest library names: the first `newest_reserve`
/// newest-only names (not in popularity) go first so they survive the
/// downstream discovery cap, then popularity order, then any remaining newest.
/// Deduplicated, first occurrence wins.
fn merge_library_names(
    popularity: Vec<String>,
    newest: Vec<String>,
    newest_reserve: usize,
) -> Vec<String> {
    let pop_set: std::collections::HashSet<&str> =
        popularity.iter().map(String::as_str).collect();
    let reserved: Vec<String> = newest
        .iter()
        .filter(|n| !pop_set.contains(n.as_str()))
        .take(newest_reserve)
        .cloned()
        .collect();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for name in reserved.into_iter().chain(popularity).chain(newest) {
        if seen.insert(name.clone()) {
            out.push(name);
        }
    }
    out
}

/// Fetch the library index: the popularity page unioned with the
/// `?sort=newest` page, so freshly added models are discoverable.
/// `newest_reserve` newest-only names are placed first to survive the
/// discovery cap. A failed newest fetch degrades to popularity-only.
pub async fn fetch_library_index(
    http: &dyn HttpClient,
    newest_reserve: usize,
) -> Result<Vec<String>, PaddockError> {
    let popularity = parse_library_names(&http.get_text("https://ollama.com/library").await?);
    let newest = match http.get_text("https://ollama.com/library?sort=newest").await {
        Ok(html) => parse_library_names(&html),
        Err(_) => Vec::new(),
    };
    Ok(merge_library_names(popularity, newest, newest_reserve))
}

/// Fetch the OCI manifest of `{base}:{tag}` and return the (digest, size) of
/// its model-weights layer (mediaType `application/vnd.ollama.image.model`).
pub async fn fetch_manifest_model_blob(
    http: &dyn HttpClient,
    base: &str,
    tag: &str,
) -> Result<(String, u64), PaddockError> {
    let manifest = http
        .get_json_with_accept(
            &format!("{REGISTRY}/{base}/manifests/{tag}"),
            MANIFEST_ACCEPT,
        )
        .await?;
    for layer in manifest["layers"].as_array().unwrap_or(&Vec::new()) {
        if !layer["mediaType"].as_str().unwrap_or("").contains("model") {
            continue;
        }
        let Some(digest) = layer["digest"].as_str() else {
            continue;
        };
        return Ok((digest.to_string(), layer["size"].as_u64().unwrap_or(0)));
    }
    Err(PaddockError::Network(format!(
        "{base}:{tag}: manifest has no model layer"
    )))
}

/// Parse a library size token (`8b`, `1.2b`, `350m`) into a parameter count.
fn params_from_size_token(size: &str) -> Option<u64> {
    let digits = size.strip_suffix('b').or_else(|| size.strip_suffix('m'))?;
    let v: f64 = digits.parse().ok()?;
    if !(v.is_finite() && v > 0.0) {
        return None;
    }
    let scale = if size.ends_with('b') { 1e9 } else { 1e6 };
    Some((v * scale) as u64)
}

/// Discover one uncurated library model from its tags page + one registry
/// manifest + one 256 KiB GGUF header probe. `Ok(None)` = deliberately
/// skipped (cloud-only, embeddings/vision architecture, unusable header);
/// `Err` = network failure worth reporting.
///
/// v1 probes ONLY the first-seen (most popular) size and creates that one
/// model; other sizes are skipped - arch params can differ per size, so each
/// size would need its own header probe. Per-size probes are a
/// request-budget tradeoff; revisit.
///
/// `now` (epoch seconds) anchors the page's relative dates ("8 months ago");
/// the OLDEST one on the tags page becomes the approximate release date.
pub async fn discover_model(
    http: &dyn HttpClient,
    name: &str,
    now: i64,
) -> Result<Option<CatalogModel>, PaddockError> {
    let html = http
        .get_text(&format!("https://ollama.com/library/{name}/tags"))
        .await?;
    let tags = extract_tag_names(&html, name);
    let released_at = super::dates::oldest_relative_date(&html, now);
    // Ollama Cloud builds are excluded by user decision: they do not run
    // locally. A model whose every quant-bearing tag is cloud yields None.
    let tags: Vec<String> = tags
        .into_iter()
        .filter(|t| !t.ends_with("-cloud"))
        .collect();
    // First-seen size prefix = first tag segment carrying a digit ("8b",
    // "1.2b", "e2b", …); "latest" and word-only tags ("instruct") skipped.
    let Some(size) = tags
        .iter()
        .filter(|t| *t != "latest")
        .map(|t| t.split('-').next().unwrap_or(t))
        .find(|s| s.chars().any(|c| c.is_ascii_digit()))
        .map(String::from)
    else {
        return Ok(None);
    };
    let selected = select_variant_tags(&tags, &size, DEFAULT_LIBRARY_QUANT);
    let Some(probe) = selected.first() else {
        return Ok(None);
    };
    let (digest, blob_size) = fetch_manifest_model_blob(http, name, &probe.tag).await?;
    let bytes = http
        .get_range(
            &format!("{REGISTRY}/{name}/blobs/{digest}"),
            0,
            BLOB_PROBE_BYTES - 1,
        )
        .await?;
    let Ok(meta) = super::gguf::parse_gguf_header(&bytes) else {
        return Ok(None);
    };
    let arch = meta.architecture.clone();
    if let Some(a) = arch.as_deref()
        && ARCH_SKIPLIST.contains(&a)
    {
        return Ok(None);
    }
    let layers = meta.block_count.unwrap_or(0) as u32;
    let kv_heads = meta.head_count_kv.or(meta.head_count).unwrap_or(0) as u32;
    let head_dim = meta.head_dim().unwrap_or(0) as u32;
    let embedding_dim = meta.embedding_length.unwrap_or(0) as u32;
    if layers == 0 || kv_heads == 0 || head_dim == 0 || embedding_dim == 0 {
        return Ok(None); // cannot estimate without the attention shape
    }
    // Param fallback chain: exact header count → weights-blob size divided by
    // the probed quant's bits-per-weight → the size token itself.
    let Some(params_total) = meta
        .parameter_count
        .filter(|p| *p > 0)
        .or_else(|| {
            quant_bpw(&probe.quant)
                .filter(|_| blob_size > 0)
                .map(|bpw| (blob_size as f64 * 8.0 / bpw) as u64)
        })
        .or_else(|| params_from_size_token(&size))
    else {
        return Ok(None);
    };
    let params_active = match (meta.expert_count, meta.expert_used_count) {
        // Rough MoE approximation: active ≈ total × used/total experts
        // (ignores the dense shared layers - good enough for fit estimates).
        (Some(experts), Some(used)) if used > 0 && used < experts => {
            (params_total as f64 * used as f64 / experts as f64) as u64
        }
        _ => params_total,
    };
    let variants: Vec<CatalogVariant> = selected
        .iter()
        .filter_map(|t| {
            quant_bpw(&t.quant).map(|bpw| CatalogVariant {
                quant: t.quant.clone(),
                bpw,
                // Only the probed tag's weights-blob size is known; other
                // tags would each cost a manifest request.
                file_size_bytes: (t.tag == probe.tag && blob_size > 0).then_some(blob_size),
                layers,
                kv_heads,
                head_dim,
                embedding_dim,
                runtime_compat: vec![RuntimeKind::Ollama],
                source_tag: Some(t.tag.clone()),
            })
        })
        .collect();
    if variants.is_empty() {
        return Ok(None);
    }
    let context_max = meta.context_length.unwrap_or(0) as u32;
    Ok(Some(CatalogModel {
        id: 0,
        name: format!("{name}:{size}"),
        family: arch.clone(),
        source: Source::Ollama,
        repo: None,
        params_total,
        params_active,
        architecture: arch,
        context_max: if context_max == 0 { 4096 } else { context_max },
        released_at,
        released_approx: released_at.is_some(),
        variants,
    }))
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
///   entry's quant - that's what the default tag aliases on the library);
/// - `{size_prefix}-…-{quant}` tags are kept when the last `-` segment is a
///   known quant (per `quant_bpw` after normalization);
/// - `-text`/`-base` (non-chat) tags and unknown quants (q4_1, q5_0, …) are
///   skipped;
/// - one tag per normalized quant, picked by canonical-form preference:
///   plain `{size}` > exact `{size}-instruct-{q}` > exact `{size}-it-{q}` >
///   exact `{size}-{q}` > shortest remaining (fallback). This avoids pinning
///   stale versioned tags (`35b-v0.1-q4_K_M`) when a canonical form exists.
pub fn select_variant_tags(
    tags: &[String],
    size_prefix: &str,
    default_quant: &str,
) -> Vec<OllamaTag> {
    struct Kept {
        tag: String,
        quant: String,
        rank: u8,
    }
    let mut out: Vec<Kept> = Vec::new();
    for tag in tags {
        let (quant, rank) = if tag == size_prefix {
            // Plain default alias: the library's own canonical pick.
            (default_quant.to_string(), 0u8)
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
            let raw = tag.rsplit('-').next().unwrap_or_default();
            let Some(quant) = normalize_quant(raw) else {
                continue;
            };
            let rank = if *tag == format!("{size_prefix}-instruct-{raw}") {
                1
            } else if *tag == format!("{size_prefix}-it-{raw}") {
                2
            } else if *tag == format!("{size_prefix}-{raw}") {
                3
            } else {
                4
            };
            (quant, rank)
        };
        match out.iter_mut().find(|t| t.quant == quant) {
            Some(existing) => {
                if rank < existing.rank || (rank == existing.rank && tag.len() < existing.tag.len())
                {
                    existing.tag = tag.clone();
                    existing.rank = rank;
                }
            }
            None => out.push(Kept {
                tag: tag.clone(),
                quant,
                rank,
            }),
        }
    }
    out.into_iter()
        .map(|k| OllamaTag {
            tag: k.tag,
            quant: k.quant,
        })
        .collect()
}

/// Replace the model's variants with one per selected tag: bpw via
/// `quant_bpw`, architecture fields copied from the existing first variant,
/// file size kept from the curated default variant when the quant matches
/// (unknown otherwise; estimates derive from params × bpw), `source_tag` set
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
                file_size_bytes: if t.quant == arch.quant {
                    arch.file_size_bytes
                } else {
                    None
                },
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
    use serde_json::{Value, json};

    use super::*;
    use crate::catalog::gguf::tests::GgufBuilder;

    #[derive(Default)]
    struct MockHttp {
        text: HashMap<String, String>,
        json: HashMap<String, Value>,
        ranges: HashMap<String, Vec<u8>>,
    }

    #[async_trait]
    impl HttpClient for MockHttp {
        // `get_json_with_accept` keeps the trait's default impl, which must
        // delegate here - manifest fixtures are registered as plain json.
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

    /// Realistic excerpt of https://ollama.com/library/llama3.1/tags -
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
        let mut http = MockHttp::default();
        http.text.insert(
            "https://ollama.com/library/llama3.1/tags".to_string(),
            LLAMA31_TAGS_HTML.to_string(),
        );
        http
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
        let http = MockHttp::default();
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
    fn select_variant_tags_prefers_canonical_instruct_over_stale_versioned() {
        // The stale `-v0.1-` tag is SHORTER, but the exact canonical
        // `{size}-instruct-{q}` form must win the quant slot.
        let tags: Vec<String> = ["8b-v0.1-q4_K_M", "8b-instruct-q4_K_M"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let kept = select_variant_tags(&tags, "8b", "Q4_K_M");
        assert_eq!(
            kept,
            vec![OllamaTag {
                tag: "8b-instruct-q4_K_M".into(),
                quant: "Q4_K_M".into()
            }]
        );
    }

    #[test]
    fn select_variant_tags_prefers_it_and_plain_size_quant_forms() {
        // `{size}-it-{q}` beats non-canonical forms - even at equal length
        // and listed second (page order / shortest must not decide).
        let tags: Vec<String> = ["9b-v1-q4_K_M", "9b-it-q4_K_M"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let kept = select_variant_tags(&tags, "9b", "Q4_K_M");
        assert_eq!(kept[0].tag, "9b-it-q4_K_M");

        // …and exact `{size}-{q}` beats non-canonical too.
        let tags: Vec<String> = ["7b-2407-q4_K_M", "7b-q4_K_M"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let kept = select_variant_tags(&tags, "7b", "Q4_K_M");
        assert_eq!(kept[0].tag, "7b-q4_K_M");
    }

    #[test]
    fn select_variant_tags_no_canonical_falls_back_to_shortest() {
        let tags: Vec<String> = ["35b-v0.1-q4_K_M", "35b-08-2024-q4_K_M"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let kept = select_variant_tags(&tags, "35b", "Q4_K_M");
        assert_eq!(
            kept,
            vec![OllamaTag {
                tag: "35b-v0.1-q4_K_M".into(),
                quant: "Q4_K_M".into()
            }]
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
            released_at: None,
            released_approx: false,
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
        // Default tag variant carries its exact tag too, and keeps the
        // curated file size (same quant as the curated default).
        let q4 = m.variants.iter().find(|v| v.quant == "Q4_K_M").unwrap();
        assert_eq!(q4.source_tag.as_deref(), Some("8b"));
        assert_eq!(q4.file_size_bytes, Some(4_920_000_000));
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

    /// Realistic excerpt of https://ollama.com/library - names appear once
    /// per card, sometimes again as tag links; non-library links are noise.
    const LIBRARY_INDEX_HTML: &str = r#"<!DOCTYPE html>
<html><body>
<a href="/library/gemma3"><h2>gemma3</h2></a>
<a href="/library/deepseek-r1"><h2>deepseek-r1</h2></a>
<a href="/library/llama3.1"><h2>llama3.1</h2></a>
<a href="/library/gemma3">duplicate card link</a>
<a href="/library/lfm2.5:1.2b">tag link must dedup to the base name</a>
<a href="/library/lfm2.5"><h2>lfm2.5</h2></a>
<a href="/blog/new-models">blog link</a>
<a href="/download">download</a>
</body></html>"#;

    #[test]
    fn parse_library_names_extracts_from_href() {
        let html = r#"<a href="/library/llama3.1">..</a><a href="/library/qwen3:8b">..</a><a href="/library/llama3.1">dup</a>"#;
        let names = parse_library_names(html);
        assert_eq!(names, vec!["llama3.1", "qwen3"]);
    }

    #[test]
    fn merge_library_names_reserves_newest_headroom() {
        let popularity = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let newest = vec!["new".to_string(), "a".to_string()];
        let merged = merge_library_names(popularity, newest, 1);
        assert_eq!(merged, vec!["new", "a", "b", "c"]);
    }

    #[tokio::test]
    async fn fetch_library_index_extracts_names_deduped_in_order() {
        let mut http = MockHttp::default();
        http.text.insert(
            "https://ollama.com/library".to_string(),
            LIBRARY_INDEX_HTML.to_string(),
        );
        let names = fetch_library_index(&http, 20).await.unwrap();
        assert_eq!(names, vec!["gemma3", "deepseek-r1", "llama3.1", "lfm2.5"]);
    }

    #[tokio::test]
    async fn fetch_library_index_propagates_network_error() {
        let http = MockHttp::default();
        assert!(fetch_library_index(&http, 20).await.is_err());
    }

    /// Manifest fixture shaped like a real registry.ollama.ai answer.
    fn manifest_json(digest: &str, size: u64) -> Value {
        json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": "sha256:config",
                "size": 489
            },
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": digest,
                    "size": size
                },
                {
                    "mediaType": "application/vnd.ollama.image.template",
                    "digest": "sha256:tmpl",
                    "size": 120
                }
            ]
        })
    }

    #[tokio::test]
    async fn fetch_manifest_model_blob_picks_model_layer() {
        let mut http = MockHttp::default();
        http.json.insert(
            "https://registry.ollama.ai/v2/library/lfm2.5/manifests/1.2b".to_string(),
            manifest_json("sha256:abc123", 736_000_000),
        );
        let (digest, size) = fetch_manifest_model_blob(&http, "lfm2.5", "1.2b")
            .await
            .unwrap();
        assert_eq!(digest, "sha256:abc123");
        assert_eq!(size, 736_000_000);
    }

    #[tokio::test]
    async fn fetch_manifest_without_model_layer_is_error() {
        let mut http = MockHttp::default();
        http.json.insert(
            "https://registry.ollama.ai/v2/library/x/manifests/1b".to_string(),
            json!({"layers": [{"mediaType": "application/vnd.ollama.image.template",
                               "digest": "sha256:t", "size": 1}]}),
        );
        assert!(fetch_manifest_model_blob(&http, "x", "1b").await.is_err());
    }

    #[test]
    fn params_from_size_token_cases() {
        assert_eq!(params_from_size_token("8b"), Some(8_000_000_000));
        assert_eq!(params_from_size_token("1.2b"), Some(1_200_000_000));
        assert_eq!(params_from_size_token("350m"), Some(350_000_000));
        assert_eq!(params_from_size_token("e2b"), None); // gemma3n effective size
        assert_eq!(params_from_size_token("latest"), None);
    }

    /// Full discovery fixture for an lfm2.5-like model: tags page + manifest
    /// + blob header. `header` lets each test vary the GGUF metadata.
    fn discovery_http(tags_html: &str, blob_size: u64, header: Vec<u8>) -> MockHttp {
        let mut http = MockHttp::default();
        http.text.insert(
            "https://ollama.com/library/lfm2.5/tags".to_string(),
            tags_html.to_string(),
        );
        http.json.insert(
            "https://registry.ollama.ai/v2/library/lfm2.5/manifests/1.2b".to_string(),
            manifest_json("sha256:blob", blob_size),
        );
        http.ranges.insert(
            "https://registry.ollama.ai/v2/library/lfm2.5/blobs/sha256:blob".to_string(),
            header,
        );
        http
    }

    /// Tags page with realistic date markup: a page-level "updated" span and
    /// per-tag rows `digest&nbsp;·&nbsp;N months ago`. Oldest = 8 months ago.
    const LFM25_TAGS_HTML: &str = r#"
        <span x-test-updated>2 months ago</span>
        <a href="/library/lfm2.5:latest">latest</a>
        <div>46e0c10c039e&nbsp;&middot;&nbsp;731MB&nbsp;&middot;&nbsp;2 months ago</div>
        <a href="/library/lfm2.5:1.2b">1.2b</a>
        <a href="/library/lfm2.5:350m">350m</a>
        <a href="/library/lfm2.5:1.2b-q4_K_M">x</a>
        <div>46e0c10c039e&nbsp;·&nbsp;8 months ago</div>
        <a href="/library/lfm2.5:1.2b-q8_0">x</a>
        <a href="/library/lfm2.5:1.2b-cloud">cloud build must be skipped</a>
        <a href="/library/lfm2.5:350m-q4_K_M">x</a>
    "#;

    /// Same tag set, no date markup anywhere (degraded/changed page layout).
    const LFM25_TAGS_HTML_NO_DATES: &str = r#"
        <a href="/library/lfm2.5:latest">latest</a>
        <a href="/library/lfm2.5:1.2b">1.2b</a>
        <a href="/library/lfm2.5:350m">350m</a>
        <a href="/library/lfm2.5:1.2b-q4_K_M">x</a>
        <a href="/library/lfm2.5:1.2b-q8_0">x</a>
        <a href="/library/lfm2.5:1.2b-cloud">cloud build must be skipped</a>
        <a href="/library/lfm2.5:350m-q4_K_M">x</a>
    "#;

    /// Fixed "now" for discovery tests so relative-date math is deterministic.
    const NOW: i64 = 1_780_000_000;

    fn lfm25_header() -> GgufBuilder {
        GgufBuilder::new()
            .string("general.architecture", "lfm2moe")
            .u64("general.parameter_count", 1_170_000_000)
            .u32("lfm2moe.block_count", 16)
            .u32("lfm2moe.attention.head_count", 16)
            .u32("lfm2moe.attention.head_count_kv", 4)
            .u32("lfm2moe.embedding_length", 2048)
            .u32("lfm2moe.context_length", 32768)
    }

    #[tokio::test]
    async fn discover_model_builds_first_seen_size_only() {
        let http = discovery_http(LFM25_TAGS_HTML, 736_000_000, lfm25_header().build());
        let m = discover_model(&http, "lfm2.5", NOW).await.unwrap().unwrap();

        // First-seen size only (1.2b); 350m skipped in v1.
        assert_eq!(m.name, "lfm2.5:1.2b");
        assert_eq!(m.source, Source::Ollama);
        assert_eq!(m.architecture.as_deref(), Some("lfm2moe"));
        assert_eq!(m.params_total, 1_170_000_000); // exact header count wins
        assert_eq!(m.params_active, 1_170_000_000); // no expert keys → dense
        assert_eq!(m.context_max, 32768);

        // Oldest relative date on the page ("8 months ago") is the release
        // proxy - always approximate.
        assert_eq!(m.released_at, Some(NOW - 8 * 30 * 86_400));
        assert!(m.released_approx);

        // Plain `1.2b` aliases Q4_K_M and wins that slot; cloud tag excluded.
        assert_eq!(m.variants.len(), 2);
        let q4 = m.variants.iter().find(|v| v.quant == "Q4_K_M").unwrap();
        assert_eq!(q4.source_tag.as_deref(), Some("1.2b"));
        assert_eq!(q4.file_size_bytes, Some(736_000_000)); // probed tag only
        assert_eq!(q4.layers, 16);
        assert_eq!(q4.kv_heads, 4);
        assert_eq!(q4.head_dim, 128); // 2048 / 16
        assert_eq!(q4.embedding_dim, 2048);
        assert_eq!(q4.runtime_compat, vec![RuntimeKind::Ollama]);
        let q8 = m.variants.iter().find(|v| v.quant == "Q8_0").unwrap();
        assert_eq!(q8.source_tag.as_deref(), Some("1.2b-q8_0"));
        assert_eq!(q8.file_size_bytes, None);
    }

    #[tokio::test]
    async fn discover_model_moe_active_params_from_expert_ratio() {
        let header = lfm25_header()
            .u32("lfm2moe.expert_count", 32)
            .u32("lfm2moe.expert_used_count", 4)
            .build();
        let http = discovery_http(LFM25_TAGS_HTML, 736_000_000, header);
        let m = discover_model(&http, "lfm2.5", NOW).await.unwrap().unwrap();
        assert_eq!(m.params_total, 1_170_000_000);
        // Rough MoE approximation: total × 4/32.
        assert_eq!(m.params_active, 146_250_000);
    }

    #[tokio::test]
    async fn discover_model_without_page_dates_has_no_release_date() {
        let http = discovery_http(
            LFM25_TAGS_HTML_NO_DATES,
            736_000_000,
            lfm25_header().build(),
        );
        let m = discover_model(&http, "lfm2.5", NOW).await.unwrap().unwrap();
        assert_eq!(m.released_at, None);
        assert!(!m.released_approx);
    }

    #[tokio::test]
    async fn discover_model_params_fallback_blob_size_then_size_token() {
        // No general.parameter_count → blob_size × 8 / bpw of the probed
        // quant (Q4_K_M, 4.83).
        let header = GgufBuilder::new()
            .string("general.architecture", "lfm2")
            .u32("lfm2.block_count", 16)
            .u32("lfm2.attention.head_count", 16)
            .u32("lfm2.attention.head_count_kv", 4)
            .u32("lfm2.embedding_length", 2048)
            .u32("lfm2.context_length", 32768)
            .build();
        let http = discovery_http(LFM25_TAGS_HTML, 736_000_000, header.clone());
        let m = discover_model(&http, "lfm2.5", NOW).await.unwrap().unwrap();
        assert_eq!(m.params_total, (736_000_000f64 * 8.0 / 4.83) as u64);

        // Manifest reports size 0 → last resort: the size token "1.2b".
        let http = discovery_http(LFM25_TAGS_HTML, 0, header);
        let m = discover_model(&http, "lfm2.5", NOW).await.unwrap().unwrap();
        assert_eq!(m.params_total, 1_200_000_000);
        // No blob size known on any variant.
        assert!(m.variants.iter().all(|v| v.file_size_bytes.is_none()));
    }

    #[tokio::test]
    async fn discover_model_cloud_only_is_none() {
        let html = r#"
            <a href="/library/lfm2.5:latest">latest</a>
            <a href="/library/lfm2.5:120b-cloud">x</a>
            <a href="/library/lfm2.5:480b-cloud">x</a>
        "#;
        let http = discovery_http(html, 0, Vec::new());
        assert!(
            discover_model(&http, "lfm2.5", NOW)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn discover_model_skiplist_architecture_is_none() {
        for arch in ["bert", "nomic-bert", "clip"] {
            let header = GgufBuilder::new()
                .string("general.architecture", arch)
                .u32("x.block_count", 12)
                .u32("x.attention.head_count", 12)
                .u32("x.attention.head_count_kv", 12)
                .u32("x.embedding_length", 768)
                .u32("x.context_length", 512)
                .build();
            let http = discovery_http(LFM25_TAGS_HTML, 736_000_000, header);
            assert!(
                discover_model(&http, "lfm2.5", NOW)
                    .await
                    .unwrap()
                    .is_none(),
                "arch {arch} must be skipped"
            );
        }
    }

    #[tokio::test]
    async fn discover_model_unparseable_header_is_none() {
        let http = discovery_http(LFM25_TAGS_HTML, 736_000_000, b"not a gguf file".to_vec());
        assert!(
            discover_model(&http, "lfm2.5", NOW)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn discover_model_network_errors_propagate() {
        // Tags page missing entirely.
        let http = MockHttp::default();
        assert!(discover_model(&http, "lfm2.5", NOW).await.is_err());
        // Tags OK but manifest fetch fails.
        let mut http = MockHttp::default();
        http.text.insert(
            "https://ollama.com/library/lfm2.5/tags".to_string(),
            LFM25_TAGS_HTML.to_string(),
        );
        assert!(discover_model(&http, "lfm2.5", NOW).await.is_err());
    }

    /// Live sanity check against the real Ollama library page. Run manually:
    /// `cargo test -p paddock-core live_llama31 -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "network: hits ollama.com"]
    async fn live_llama31_tags_page_yields_tags() {
        let http = super::super::hf::ReqwestClient::new().unwrap();
        let tags = fetch_model_tags(&http, "llama3.1").await.unwrap();
        println!("live llama3.1 tags ({}): {:?}", tags.len(), tags);
        assert!(tags.len() >= 5, "expected >= 5 tags, got {tags:?}");
    }
}
