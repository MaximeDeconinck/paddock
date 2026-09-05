//! `paddock bench`: measure the real generation tok/s of a running server.
//!
//! One warm-up request (so cold start does not pollute the timing), then one
//! timed generation of `tokens` tokens. Each runtime reports timings its own
//! way; when the server gives none, tok/s = tokens / wall time.

use std::time::{Duration, Instant};

use serde::Serialize;

use crate::PaddockError;
use crate::catalog::{CatalogModel, RuntimeKind};
use crate::hardware::SystemProbe;

/// Default `--tokens`: long enough to amortize per-request overhead, short
/// enough that the KV cache stays negligible (the calibration assumes KV ~ 0).
pub const DEFAULT_BENCH_TOKENS: u32 = 128;
const WARM_UP_TOKENS: u32 = 8;
/// Open-ended so the model does not stop early; ~20 prompt tokens.
const BENCH_PROMPT: &str =
    "Write a detailed, multi-paragraph history of the horse, from domestication to the present day.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingSource {
    /// The server measured its own generation speed (Ollama, llama.cpp).
    ServerTimings,
    /// tokens / wall-clock of the whole request (mlx-lm, or missing fields).
    WallClock,
}

impl TimingSource {
    pub fn label(&self) -> &'static str {
        match self {
            Self::ServerTimings => "server timings",
            Self::WallClock => "wall clock",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchMeasurement {
    pub tps: f64,
    pub tokens: u32,
    pub timing: TimingSource,
}

/// What a runtime's response told us: generated token count and, when the
/// server clocked it, the generation speed.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedTiming {
    pub tokens: u32,
    pub server_tps: Option<f64>,
}

/// (url, JSON body) of a non-streaming generation of `tokens` tokens.
pub fn bench_request(
    runtime: RuntimeKind,
    endpoint: &str,
    model_ref: &str,
    tokens: u32,
) -> (String, String) {
    let endpoint = endpoint.trim_end_matches('/');
    match runtime {
        RuntimeKind::Ollama => (
            format!("{endpoint}/api/generate"),
            serde_json::json!({
                "model": model_ref,
                "prompt": BENCH_PROMPT,
                "stream": false,
                "options": { "num_predict": tokens },
            })
            .to_string(),
        ),
        RuntimeKind::LlamaCpp => (
            format!("{endpoint}/completion"),
            serde_json::json!({
                "prompt": BENCH_PROMPT,
                "n_predict": tokens,
                "stream": false,
            })
            .to_string(),
        ),
        RuntimeKind::MlxLm => (
            format!("{endpoint}/v1/chat/completions"),
            serde_json::json!({
                "model": model_ref,
                "messages": [{ "role": "user", "content": BENCH_PROMPT }],
                "max_tokens": tokens,
                "stream": false,
            })
            .to_string(),
        ),
    }
}

fn missing(field: &str) -> PaddockError {
    PaddockError::Other(format!(
        "bench: server response has no `{field}` field; cannot count generated tokens"
    ))
}

/// Extract token count and server-side speed from a generation response.
pub fn parse_timing(runtime: RuntimeKind, body: &str) -> Result<ParsedTiming, PaddockError> {
    let v: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| PaddockError::Other(format!("bench: unparseable server response: {e}")))?;
    match runtime {
        RuntimeKind::Ollama => {
            let tokens = v["eval_count"]
                .as_u64()
                .ok_or_else(|| missing("eval_count"))? as u32;
            let server_tps = v["eval_duration"]
                .as_u64()
                .filter(|&ns| ns > 0)
                .map(|ns| tokens as f64 / (ns as f64 / 1e9));
            Ok(ParsedTiming { tokens, server_tps })
        }
        RuntimeKind::LlamaCpp => {
            let timings = &v["timings"];
            let tokens = timings["predicted_n"]
                .as_u64()
                .or_else(|| v["tokens_predicted"].as_u64())
                .ok_or_else(|| missing("tokens_predicted"))? as u32;
            let server_tps = timings["predicted_per_second"]
                .as_f64()
                .filter(|t| t.is_finite() && *t > 0.0);
            Ok(ParsedTiming { tokens, server_tps })
        }
        RuntimeKind::MlxLm => {
            let tokens = v["usage"]["completion_tokens"]
                .as_u64()
                .ok_or_else(|| missing("usage.completion_tokens"))? as u32;
            Ok(ParsedTiming {
                tokens,
                server_tps: None,
            })
        }
    }
}

/// Server timings when present, else tokens / wall time.
pub fn finalize(parsed: ParsedTiming, wall: Duration) -> Result<BenchMeasurement, PaddockError> {
    if parsed.tokens == 0 {
        return Err(PaddockError::Other(
            "bench: server generated 0 tokens; is the model loaded?".into(),
        ));
    }
    if let Some(tps) = parsed.server_tps {
        return Ok(BenchMeasurement {
            tps,
            tokens: parsed.tokens,
            timing: TimingSource::ServerTimings,
        });
    }
    let secs = wall.as_secs_f64();
    if secs <= 0.0 {
        return Err(PaddockError::Other(
            "bench: zero wall time for the generation request".into(),
        ));
    }
    Ok(BenchMeasurement {
        tps: parsed.tokens as f64 / secs,
        tokens: parsed.tokens,
        timing: TimingSource::WallClock,
    })
}

fn unreachable(endpoint: &str) -> PaddockError {
    PaddockError::Other(format!(
        "bench: no answer from {endpoint}; the server died or refused the request (check `paddock ps` / `paddock logs`)"
    ))
}

/// Warm up, then time one generation of `tokens` tokens against a running server.
pub fn measure(
    probe: &dyn SystemProbe,
    runtime: RuntimeKind,
    endpoint: &str,
    model_ref: &str,
    tokens: u32,
) -> Result<BenchMeasurement, PaddockError> {
    let (url, warm_body) = bench_request(runtime, endpoint, model_ref, WARM_UP_TOKENS);
    probe
        .http_post_local(&url, &warm_body)
        .ok_or_else(|| unreachable(endpoint))?;

    let (url, body) = bench_request(runtime, endpoint, model_ref, tokens);
    let start = Instant::now();
    let response = probe
        .http_post_local(&url, &body)
        .ok_or_else(|| unreachable(endpoint))?;
    let wall = start.elapsed();
    finalize(parse_timing(runtime, &response)?, wall)
}

/// Map a running server's `model_ref` back to a catalog (model, variant) pair,
/// so the bench can read `params_active` / `bpw`. Indexes are into `models`
/// and `models[i].variants`. None = measure-only (no calibration update).
///
/// Shapes (see `runtime.rs`): `hf.co/{org}/{repo}:{quant}` and
/// `{org}/{repo}[:{quant}]` for HF / MLX repos; `{base}:{tag}` for Ollama,
/// where `tag` is an exact `source_tag`, part of a curated full name, or a
/// library tag that merely contains a quant label.
pub fn resolve_model_ref(models: &[CatalogModel], model_ref: &str) -> Option<(usize, usize)> {
    let stripped = model_ref.strip_prefix("hf.co/").unwrap_or(model_ref);

    if stripped.contains('/') {
        let (repo, quant) = match stripped.rsplit_once(':') {
            Some((r, q)) => (r, Some(q)),
            None => (stripped, None),
        };
        let mi = models.iter().position(|m| {
            m.repo
                .as_deref()
                .is_some_and(|r| r.eq_ignore_ascii_case(repo))
        })?;
        let vi = match quant {
            Some(q) => models[mi]
                .variants
                .iter()
                .position(|v| v.quant.eq_ignore_ascii_case(q))?,
            None => best_quality_idx(&models[mi])?,
        };
        return Some((mi, vi));
    }

    let (base, tag) = match stripped.split_once(':') {
        Some((b, t)) => (b, Some(t)),
        None => (stripped, None),
    };
    let base_of = |m: &CatalogModel| m.name.split(':').next().unwrap_or(&m.name).to_string();

    // 1. Exact `source_tag` on a model with the same base name.
    if let Some(tag) = tag {
        for (mi, m) in models.iter().enumerate() {
            if !base_of(m).eq_ignore_ascii_case(base) {
                continue;
            }
            if let Some(vi) = m.variants.iter().position(|v| {
                v.source_tag
                    .as_deref()
                    .is_some_and(|t| t.eq_ignore_ascii_case(tag))
            }) {
                return Some((mi, vi));
            }
        }
    }
    // 2. Curated full name (`llama3.2:3b` carries its tag in the name).
    if let Some(mi) = models
        .iter()
        .position(|m| m.name.eq_ignore_ascii_case(stripped))
    {
        return Some((mi, best_quality_idx(&models[mi])?));
    }
    // 3. Base name; variant whose quant label appears in the tag, else best quality.
    let mi = models
        .iter()
        .position(|m| base_of(m).eq_ignore_ascii_case(base))?;
    let m = &models[mi];
    let vi = tag
        .and_then(|t| {
            let t = t.to_lowercase();
            m.variants
                .iter()
                .position(|v| t.contains(&v.quant.to_lowercase()))
        })
        .or_else(|| best_quality_idx(m))?;
    Some((mi, vi))
}

/// Index of the highest-quality variant (first in `variants_by_quality`).
fn best_quality_idx(m: &CatalogModel) -> Option<usize> {
    let mvs: Vec<_> = m.variants.iter().map(|v| m.to_model_variant(v)).collect();
    crate::score::variants_by_quality(&mvs).first().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::MockProbe;
    use std::time::Duration;

    #[test]
    fn ollama_request_shape() {
        let (url, body) = bench_request(
            RuntimeKind::Ollama,
            "http://127.0.0.1:11434",
            "llama3.2:3b",
            128,
        );
        assert_eq!(url, "http://127.0.0.1:11434/api/generate");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"], "llama3.2:3b");
        assert_eq!(v["stream"], false);
        assert_eq!(v["options"]["num_predict"], 128);
        assert!(v["prompt"].as_str().unwrap().len() > 10);
    }

    #[test]
    fn llama_cpp_request_shape() {
        let (url, body) = bench_request(RuntimeKind::LlamaCpp, "http://127.0.0.1:8080", "x", 64);
        assert_eq!(url, "http://127.0.0.1:8080/completion");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["n_predict"], 64);
        assert_eq!(v["stream"], false);
    }

    #[test]
    fn mlx_request_shape() {
        let (url, body) = bench_request(
            RuntimeKind::MlxLm,
            "http://127.0.0.1:8080",
            "mlx-community/x-4bit",
            32,
        );
        assert_eq!(url, "http://127.0.0.1:8080/v1/chat/completions");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"], "mlx-community/x-4bit");
        assert_eq!(v["max_tokens"], 32);
        assert_eq!(v["stream"], false);
        assert_eq!(v["messages"][0]["role"], "user");
    }

    #[test]
    fn parse_ollama_eval_fields() {
        // 128 tokens in 3.2 s = 40 tok/s, exact from the server's own clock.
        let body = r#"{"model":"x","response":"...","done":true,"eval_count":128,"eval_duration":3200000000}"#;
        let p = parse_timing(RuntimeKind::Ollama, body).unwrap();
        assert_eq!(p.tokens, 128);
        assert!((p.server_tps.unwrap() - 40.0).abs() < 1e-9);
    }

    #[test]
    fn parse_ollama_without_duration_falls_back_to_wall_clock() {
        let body = r#"{"done":true,"eval_count":100}"#;
        let p = parse_timing(RuntimeKind::Ollama, body).unwrap();
        assert_eq!(p.tokens, 100);
        assert!(p.server_tps.is_none());
    }

    #[test]
    fn parse_ollama_missing_eval_count_names_the_field() {
        let err = parse_timing(RuntimeKind::Ollama, r#"{"done":true}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("eval_count"), "{err}");
    }

    #[test]
    fn parse_llama_cpp_timings() {
        let body = r#"{"content":"...","tokens_predicted":128,"timings":{"prompt_n":12,"prompt_ms":80.0,"predicted_n":128,"predicted_ms":2560.0,"predicted_per_second":50.0}}"#;
        let p = parse_timing(RuntimeKind::LlamaCpp, body).unwrap();
        assert_eq!(p.tokens, 128);
        assert_eq!(p.server_tps, Some(50.0));
    }

    #[test]
    fn parse_llama_cpp_without_timings_uses_tokens_predicted() {
        let body = r#"{"content":"...","tokens_predicted":77}"#;
        let p = parse_timing(RuntimeKind::LlamaCpp, body).unwrap();
        assert_eq!(p.tokens, 77);
        assert!(p.server_tps.is_none());
    }

    #[test]
    fn parse_openai_usage() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"..."}}],"usage":{"prompt_tokens":12,"completion_tokens":128,"total_tokens":140}}"#;
        let p = parse_timing(RuntimeKind::MlxLm, body).unwrap();
        assert_eq!(p.tokens, 128);
        assert!(p.server_tps.is_none());
        let err = parse_timing(RuntimeKind::MlxLm, r#"{"choices":[]}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("usage.completion_tokens"), "{err}");
    }

    #[test]
    fn parse_rejects_non_json() {
        assert!(parse_timing(RuntimeKind::Ollama, "<html>").is_err());
    }

    #[test]
    fn finalize_prefers_server_timings_else_wall_clock() {
        let s = finalize(
            ParsedTiming {
                tokens: 128,
                server_tps: Some(40.0),
            },
            Duration::from_secs(9),
        )
        .unwrap();
        assert_eq!(s.tps, 40.0);
        assert_eq!(s.timing, TimingSource::ServerTimings);
        let w = finalize(
            ParsedTiming {
                tokens: 128,
                server_tps: None,
            },
            Duration::from_secs(4),
        )
        .unwrap();
        assert_eq!(w.tps, 32.0);
        assert_eq!(w.timing, TimingSource::WallClock);
        assert_eq!(w.tokens, 128);
    }

    #[test]
    fn finalize_rejects_zero_tokens_and_zero_wall() {
        assert!(
            finalize(
                ParsedTiming {
                    tokens: 0,
                    server_tps: None
                },
                Duration::from_secs(1)
            )
            .is_err()
        );
        assert!(
            finalize(
                ParsedTiming {
                    tokens: 10,
                    server_tps: None
                },
                Duration::ZERO
            )
            .is_err()
        );
    }

    #[test]
    fn measure_warms_up_then_times_one_request() {
        let mut probe = MockProbe::default();
        probe.posts.insert(
            "http://127.0.0.1:11434/api/generate".into(),
            r#"{"done":true,"eval_count":64,"eval_duration":1600000000}"#.into(),
        );
        let m = measure(
            &probe,
            RuntimeKind::Ollama,
            "http://127.0.0.1:11434",
            "llama3.2:3b",
            64,
        )
        .unwrap();
        assert!((m.tps - 40.0).abs() < 1e-9);
        assert_eq!(m.timing, TimingSource::ServerTimings);
        let posts = probe.post_bodies.lock().unwrap();
        assert_eq!(posts.len(), 2, "one warm-up + one timed request");
        let warm: serde_json::Value = serde_json::from_str(&posts[0].1).unwrap();
        let timed: serde_json::Value = serde_json::from_str(&posts[1].1).unwrap();
        assert!(warm["options"]["num_predict"].as_u64().unwrap() < 64);
        assert_eq!(timed["options"]["num_predict"], 64);
    }

    #[test]
    fn measure_unreachable_server_is_a_clean_error() {
        let probe = MockProbe::default(); // no POST fixtures = connection refused
        let err = measure(&probe, RuntimeKind::LlamaCpp, "http://127.0.0.1:8080", "x", 16)
            .unwrap_err()
            .to_string();
        assert!(err.contains("127.0.0.1:8080"), "{err}");
    }
}

#[cfg(test)]
mod resolve_tests {
    use super::*;
    use crate::catalog::{CatalogModel, CatalogVariant, RuntimeKind, Source};

    fn variant(quant: &str, tag: Option<&str>) -> CatalogVariant {
        CatalogVariant {
            quant: quant.into(),
            bpw: crate::catalog::quant_bpw(quant).unwrap_or(4.5),
            file_size_bytes: None,
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
            embedding_dim: 4096,
            runtime_compat: vec![RuntimeKind::Ollama, RuntimeKind::LlamaCpp],
            source_tag: tag.map(str::to_string),
        }
    }

    fn model(
        name: &str,
        source: Source,
        repo: Option<&str>,
        variants: Vec<CatalogVariant>,
    ) -> CatalogModel {
        CatalogModel {
            id: 0,
            name: name.into(),
            family: None,
            source,
            repo: repo.map(str::to_string),
            params_total: 8_000_000_000,
            params_active: 8_000_000_000,
            architecture: None,
            context_max: 32_768,
            released_at: None,
            released_approx: false,
            variants,
        }
    }

    fn catalog() -> Vec<CatalogModel> {
        vec![
            model(
                "llama3.1:8b",
                Source::Ollama,
                None,
                vec![
                    variant("Q4_K_M", Some("8b-instruct-q4_K_M")),
                    variant("Q8_0", Some("8b-instruct-q8_0")),
                ],
            ),
            model(
                "qwen3-coder:30b",
                Source::Ollama,
                None,
                vec![variant("Q4_K_M", None)],
            ),
            model(
                "Qwen3.6-35B-A3B-GGUF",
                Source::HuggingFace,
                Some("unsloth/Qwen3.6-35B-A3B-GGUF"),
                vec![
                    variant("Q4_K_M", None),
                    variant("UD-Q4_K_XL", None),
                    variant("Q8_0", None),
                ],
            ),
            model(
                "Llama-3.1-8B-Instruct-4bit",
                Source::Mlx,
                Some("mlx-community/Llama-3.1-8B-Instruct-4bit"),
                vec![variant("MLX_4BIT", None)],
            ),
        ]
    }

    #[test]
    fn hf_ref_with_quant_matches_repo_and_quant_case_insensitively() {
        let c = catalog();
        assert_eq!(
            resolve_model_ref(&c, "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:ud-q4_k_xl"),
            Some((2, 1))
        );
        assert_eq!(
            resolve_model_ref(&c, "unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0"),
            Some((2, 2))
        );
    }

    #[test]
    fn hf_ref_with_unknown_quant_is_unresolved() {
        assert_eq!(
            resolve_model_ref(&catalog(), "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:IQ2_XXS"),
            None
        );
    }

    #[test]
    fn mlx_repo_ref_picks_its_single_variant() {
        assert_eq!(
            resolve_model_ref(&catalog(), "mlx-community/Llama-3.1-8B-Instruct-4bit"),
            Some((3, 0))
        );
    }

    #[test]
    fn ollama_source_tag_matches_exact_variant() {
        assert_eq!(
            resolve_model_ref(&catalog(), "llama3.1:8b-instruct-q8_0"),
            Some((0, 1))
        );
    }

    #[test]
    fn ollama_full_curated_name_picks_best_quality_variant() {
        assert_eq!(resolve_model_ref(&catalog(), "qwen3-coder:30b"), Some((1, 0)));
        // Curated name with two variants and no tag hit: Q8_0 is the higher quality.
        assert_eq!(resolve_model_ref(&catalog(), "llama3.1:8b"), Some((0, 1)));
    }

    #[test]
    fn ollama_base_name_with_unknown_tag_matches_quant_substring_else_best() {
        let c = catalog();
        assert_eq!(resolve_model_ref(&c, "llama3.1:latest-q4_K_M"), Some((0, 0)));
        assert_eq!(resolve_model_ref(&c, "llama3.1:latest"), Some((0, 1)));
    }

    #[test]
    fn unknown_ref_is_unresolved() {
        assert_eq!(resolve_model_ref(&catalog(), "definitely-not-a-model:1b"), None);
        assert_eq!(resolve_model_ref(&catalog(), "hf.co/nobody/nothing:Q4_K_M"), None);
    }
}
