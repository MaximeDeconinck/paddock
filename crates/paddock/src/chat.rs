//! `paddock_chat`: send a prompt, plus files read locally, to a running local
//! model and return only its answer. File contents never leave this process
//! except through the model's own output.

use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

use paddock_core::catalog::RuntimeKind;
use paddock_core::serving::{ServerRow, ServerRowMatch, ServerStatus, match_server_rows};
use serde_json::{Value, json};

/// Total size cap for `files`.
pub(crate) const MAX_FILES_BYTES: u64 = 1024 * 1024;
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 2048;
pub(crate) const DEFAULT_TEMPERATURE: f32 = 0.2;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Debug, thiserror::Error)]
pub(crate) enum ChatError {
    #[error("no running server matches `{target}`")]
    NotServing {
        target: String,
        running: Vec<String>,
    },
    #[error("`{target}` matches several servers - be specific")]
    Ambiguous {
        target: String,
        candidates: Vec<String>,
    },
    #[error("{model} is still loading")]
    NotReady { model: String },
    #[error("cannot use {path}: {reason}")]
    File { path: String, reason: String },
    #[error("prompt needs ~{estimated} tokens, the server context is {limit}")]
    ContextOverflow { estimated: u64, limit: u32 },
    #[error("chat request failed: {message}")]
    Failed {
        status: Option<u16>,
        message: String,
    },
}

impl ChatError {
    /// MCP error body: `{ code, message, ...extra }`.
    pub(crate) fn body(&self) -> Value {
        let (code, extra) = match self {
            ChatError::NotServing { running, .. } => (
                "not_serving",
                json!({ "running": running, "hint": "start one with paddock_serve" }),
            ),
            ChatError::Ambiguous { candidates, .. } => {
                ("ambiguous_server", json!({ "candidates": candidates }))
            }
            ChatError::NotReady { .. } => (
                "not_ready",
                json!({ "hint": "call again once paddock_ps shows it ready" }),
            ),
            ChatError::File { path, .. } => ("file_error", json!({ "path": path })),
            ChatError::ContextOverflow { estimated, limit } => (
                "context_overflow",
                json!({
                    "estimated_tokens": estimated,
                    "limit": limit,
                    "hint": "split the files across several calls, or re-serve with a larger ctx",
                }),
            ),
            ChatError::Failed { status, .. } => ("chat_failed", json!({ "http_status": status })),
        };
        let mut body = json!({ "code": code, "message": self.to_string() });
        if let (Some(dst), Some(src)) = (body.as_object_mut(), extra.as_object()) {
            for (k, v) in src {
                dst.insert(k.clone(), v.clone());
            }
        }
        body
    }
}

/// Pick the ready server `target` designates among `rows`. Loading servers
/// never make a ready match ambiguous; they only explain a miss (`not_ready`).
pub(crate) fn resolve_target(rows: &[ServerRow], target: &str) -> Result<ServerRow, ChatError> {
    let (ready, starting): (Vec<ServerRow>, Vec<ServerRow>) = rows
        .iter()
        .cloned()
        .partition(|r| r.status == ServerStatus::Ready);
    match find(&ready, target) {
        ServerRowMatch::Matched(r) => Ok(r.clone()),
        ServerRowMatch::NotFound
            if !matches!(find(&starting, target), ServerRowMatch::NotFound) =>
        {
            Err(ChatError::NotReady {
                model: target.to_string(),
            })
        }
        ServerRowMatch::Ambiguous(c) => Err(ChatError::Ambiguous {
            target: target.to_string(),
            candidates: c.iter().map(|r| r.model.clone()).collect(),
        }),
        ServerRowMatch::NotFound => Err(ChatError::NotServing {
            target: target.to_string(),
            running: rows.iter().map(|r| r.model.clone()).collect(),
        }),
    }
}

/// `match_server_rows`, except that an all-digits target matching no pid is
/// retried as a name substring (model names carry numbers: "2507", "8b").
fn find<'a>(rows: &'a [ServerRow], target: &str) -> ServerRowMatch<'a> {
    match match_server_rows(rows, Some(target)) {
        ServerRowMatch::NotFound if target.parse::<u32>().is_ok() => {
            let hits: Vec<&ServerRow> = rows.iter().filter(|r| r.model.contains(target)).collect();
            match hits.as_slice() {
                [] => ServerRowMatch::NotFound,
                [one] => ServerRowMatch::Matched(one),
                _ => ServerRowMatch::Ambiguous(hits),
            }
        }
        m => m,
    }
}

/// Read `paths` as UTF-8, returning `(file name, content)` pairs. Errors name
/// the path, never the content.
pub(crate) fn read_files(paths: &[String]) -> Result<Vec<(String, String)>, ChatError> {
    let mut total = 0u64;
    let mut out = Vec::with_capacity(paths.len());
    for p in paths {
        let err = |reason: String| ChatError::File {
            path: p.clone(),
            reason,
        };
        let path = Path::new(p);
        if !path.is_absolute() {
            return Err(err("path must be absolute".into()));
        }
        let meta = std::fs::metadata(path).map_err(|e| err(e.to_string()))?;
        if !meta.is_file() {
            return Err(err("not a regular file".into()));
        }
        total += meta.len();
        if total > MAX_FILES_BYTES {
            return Err(err(format!(
                "files exceed {} KiB in total",
                MAX_FILES_BYTES / 1024
            )));
        }
        // Bounded read: the file may have grown since the size check.
        let budget = MAX_FILES_BYTES - (total - meta.len());
        let mut bytes = Vec::with_capacity(meta.len() as usize);
        std::fs::File::open(path)
            .and_then(|f| f.take(budget + 1).read_to_end(&mut bytes))
            .map_err(|e| err(e.to_string()))?;
        if bytes.len() as u64 > budget {
            return Err(err(format!(
                "files exceed {} KiB in total",
                MAX_FILES_BYTES / 1024
            )));
        }
        let text = String::from_utf8(bytes).map_err(|_| err("not UTF-8 text".into()))?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.clone());
        out.push((name, text));
    }
    Ok(out)
}

/// The user message: the prompt, then each file under a `### name` heading.
pub(crate) fn user_message(prompt: &str, files: &[(String, String)]) -> String {
    let mut msg = prompt.to_string();
    for (name, content) in files {
        msg.push_str(&format!("\n\n### {name}\n```\n{content}"));
        if !content.ends_with('\n') {
            msg.push('\n');
        }
        msg.push_str("```");
    }
    msg
}

/// Conservative token estimate (~3.5 bytes per token).
pub(crate) fn estimate_tokens(text_bytes: usize) -> u64 {
    (text_bytes as u64 * 2).div_ceil(7)
}

/// Pre-flight against the server's known context window (llama.cpp only).
pub(crate) fn check_context(
    server: &ServerRow,
    message_bytes: usize,
    max_tokens: u32,
) -> Result<(), ChatError> {
    let Some(limit) = server
        .ctx
        .filter(|_| server.runtime == RuntimeKind::LlamaCpp)
    else {
        return Ok(());
    };
    let estimated = estimate_tokens(message_bytes) + u64::from(max_tokens);
    if estimated > u64::from(limit) {
        return Err(ChatError::ContextOverflow { estimated, limit });
    }
    Ok(())
}

/// Drop a leading `<think>…</think>` block (reasoning models).
pub(crate) fn strip_think(text: &str) -> String {
    let trimmed = text.trim_start();
    if let Some(rest) = trimmed.strip_prefix("<think>")
        && let Some(end) = rest.find("</think>")
    {
        return rest[end + "</think>".len()..].trim_start().to_string();
    }
    text.to_string()
}

pub(crate) struct ChatRequest {
    pub system: Option<String>,
    pub user: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

pub(crate) fn request_body(model_ref: &str, req: &ChatRequest) -> Value {
    let mut messages = Vec::new();
    if let Some(s) = &req.system {
        messages.push(json!({ "role": "system", "content": s }));
    }
    messages.push(json!({ "role": "user", "content": req.user }));
    json!({
        "model": model_ref,
        "messages": messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": false,
    })
}

/// The tool result from a chat-completions answer.
pub(crate) fn chat_result(server: &ServerRow, answer: &Value, elapsed: Duration) -> Value {
    let message = &answer["choices"][0]["message"];
    let finish_reason = answer["choices"][0]["finish_reason"].as_str();
    let mut text = message["content"].as_str().unwrap_or_default();
    // Some GGUF chat templates make Ollama / llama.cpp file the whole answer
    // as reasoning. A finished generation with no content: that is the answer.
    if text.trim().is_empty() && finish_reason != Some("length") {
        text = message["reasoning"]
            .as_str()
            .or_else(|| message["reasoning_content"].as_str())
            .unwrap_or_default();
    }
    let prompt_tokens = answer["usage"]["prompt_tokens"].as_u64();
    let completion_tokens = answer["usage"]["completion_tokens"].as_u64();
    let secs = elapsed.as_secs_f64();
    let tok_per_s = completion_tokens
        .filter(|_| secs > 0.0)
        .map(|n| ((n as f64 / secs) * 10.0).round() / 10.0);
    json!({
        "text": strip_think(text).trim().to_string(),
        "finish_reason": finish_reason,
        "model_ref": server.model,
        "runtime": server.runtime,
        "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens },
        "elapsed_ms": elapsed.as_millis() as u64,
        "tok_per_s": tok_per_s,
    })
}

/// POST the request to the server's OpenAI endpoint.
pub(crate) async fn send(server: &ServerRow, req: &ChatRequest) -> Result<Value, ChatError> {
    let failed = |status: Option<u16>, message: String| ChatError::Failed { status, message };
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| failed(None, e.to_string()))?;
    let started = Instant::now();
    let resp = client
        .post(&server.openai_url)
        .json(&request_body(&server.model, req))
        .send()
        .await
        .map_err(|e| failed(None, e.to_string()))?;
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        let message = body["error"]["message"]
            .as_str()
            .or_else(|| body["error"].as_str())
            .unwrap_or("server returned an error")
            .to_string();
        return Err(failed(Some(status.as_u16()), message));
    }
    if body["choices"][0]["message"]["content"].as_str().is_none() {
        return Err(failed(
            Some(status.as_u16()),
            "answer has no choices[0].message.content".into(),
        ));
    }
    let result = chat_result(server, &body, started.elapsed());
    if result["text"] == "" && result["finish_reason"] == "length" {
        // Reasoning models can spend the whole budget thinking (Ollama puts
        // it in a separate `reasoning` field) and never answer.
        return Err(failed(
            Some(status.as_u16()),
            format!(
                "the model used all {} max_tokens before answering (likely reasoning): raise max_tokens or use a non-reasoning model",
                req.max_tokens
            ),
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_core::serving::StopHandle;

    fn row(model: &str, runtime: RuntimeKind, ctx: Option<u32>, status: ServerStatus) -> ServerRow {
        ServerRow {
            model: model.into(),
            runtime,
            endpoint: "http://127.0.0.1:8080".into(),
            openai_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
            ctx,
            started_at: None,
            stop: StopHandle::Pid(1),
            status,
        }
    }

    #[test]
    fn resolves_ready_server_and_rejects_loading_one() {
        let rows = vec![
            row(
                "qwen3-4b",
                RuntimeKind::LlamaCpp,
                Some(8192),
                ServerStatus::Ready,
            ),
            row(
                "gemma3:12b",
                RuntimeKind::Ollama,
                None,
                ServerStatus::Starting,
            ),
        ];
        assert_eq!(resolve_target(&rows, "QWEN").unwrap().model, "qwen3-4b");
        assert!(matches!(
            resolve_target(&rows, "gemma"),
            Err(ChatError::NotReady { .. })
        ));
        // A number that is no pid is a name substring.
        let dated = vec![row(
            "Qwen3-30B-A3B-Instruct-2507",
            RuntimeKind::Ollama,
            None,
            ServerStatus::Ready,
        )];
        assert_eq!(
            resolve_target(&dated, "2507").unwrap().model,
            "Qwen3-30B-A3B-Instruct-2507"
        );
        // A loading namesake does not make the ready server ambiguous.
        let mut both = rows.clone();
        both.push(row(
            "qwen3-4b-2",
            RuntimeKind::LlamaCpp,
            Some(8192),
            ServerStatus::Starting,
        ));
        assert_eq!(resolve_target(&both, "qwen3-4b").unwrap().model, "qwen3-4b");
        let err = resolve_target(&rows, "llama").unwrap_err();
        assert_eq!(err.body()["code"], "not_serving");
        assert_eq!(err.body()["running"][1], "gemma3:12b");
    }

    #[test]
    fn user_message_fences_each_file() {
        let msg = user_message(
            "Classe ces lignes.",
            &[
                ("a.csv".into(), "x;1\n".into()),
                ("b.csv".into(), "y;2".into()),
            ],
        );
        assert_eq!(
            msg,
            "Classe ces lignes.\n\n### a.csv\n```\nx;1\n```\n\n### b.csv\n```\ny;2\n```"
        );
    }

    #[test]
    fn read_files_errors_name_the_path_not_the_content() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("secret.bin");
        std::fs::write(&bin, [0xff, 0xfe, 0x00]).unwrap();
        let p = bin.to_string_lossy().into_owned();
        let err = read_files(std::slice::from_ref(&p)).unwrap_err();
        let body = err.body();
        assert_eq!(body["code"], "file_error");
        assert_eq!(body["path"], p.as_str());
        assert!(matches!(
            read_files(&["relative.csv".into()]),
            Err(ChatError::File { .. })
        ));
        let ok = dir.path().join("ok.csv");
        std::fs::write(&ok, "date;libelle\n").unwrap();
        let files = read_files(&[ok.to_string_lossy().into_owned()]).unwrap();
        assert_eq!(files[0], ("ok.csv".into(), "date;libelle\n".into()));
    }

    #[test]
    fn read_files_enforces_total_cap() {
        let dir = tempfile::tempdir().unwrap();
        let big = dir.path().join("big.csv");
        std::fs::write(&big, vec![b'a'; MAX_FILES_BYTES as usize + 1]).unwrap();
        let err = read_files(&[big.to_string_lossy().into_owned()]).unwrap_err();
        assert!(err.to_string().contains("1024 KiB"));
    }

    #[test]
    fn context_check_only_applies_to_llama_cpp() {
        let llama = row("m", RuntimeKind::LlamaCpp, Some(4096), ServerStatus::Ready);
        assert!(check_context(&llama, 7_000, 2048).is_ok()); // 2000 + 2048
        let err = check_context(&llama, 14_000, 2048).unwrap_err();
        assert_eq!(err.body()["code"], "context_overflow");
        assert_eq!(err.body()["limit"], 4096);
        let ollama = row("m", RuntimeKind::Ollama, None, ServerStatus::Ready);
        assert!(check_context(&ollama, 10_000_000, 2048).is_ok());
    }

    #[test]
    fn strips_leading_think_block() {
        assert_eq!(strip_think("<think>hmm\nok</think>\n\nRéponse"), "Réponse");
        assert_eq!(
            strip_think("Réponse <think>x</think>"),
            "Réponse <think>x</think>"
        );
        assert_eq!(strip_think("<think>never closed"), "<think>never closed");
    }

    #[test]
    fn request_body_puts_system_first() {
        let body = request_body(
            "qwen3",
            &ChatRequest {
                system: Some("Tu es comptable.".into()),
                user: "u".into(),
                max_tokens: 64,
                temperature: 0.2,
            },
        );
        assert_eq!(body["model"], "qwen3");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "u");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn chat_result_reports_usage_and_speed() {
        let server = row(
            "qwen3",
            RuntimeKind::LlamaCpp,
            Some(8192),
            ServerStatus::Ready,
        );
        let answer = json!({
            "choices": [{ "message": { "content": "<think>x</think>Total: 42 €" } }],
            "usage": { "prompt_tokens": 900, "completion_tokens": 50 },
        });
        let v = chat_result(&server, &answer, Duration::from_millis(2000));
        assert_eq!(v["text"], "Total: 42 €");
        assert!(v["finish_reason"].is_null());
        assert_eq!(v["usage"]["prompt_tokens"], 900);
        assert_eq!(v["tok_per_s"], 25.0);
        assert_eq!(v["elapsed_ms"], 2000);
        assert_eq!(v["runtime"], "llama_cpp");
        let misfiled = json!({
            "choices": [{ "message": { "content": "", "reasoning": "1;loisirs;-12,00" }, "finish_reason": "stop" }],
        });
        assert_eq!(
            chat_result(&server, &misfiled, Duration::from_secs(1))["text"],
            "1;loisirs;-12,00"
        );
        let bare = chat_result(
            &server,
            &json!({ "choices": [{ "message": { "content": "ok" } }] }),
            Duration::from_secs(1),
        );
        assert!(bare["tok_per_s"].is_null());
    }

    /// One-shot fake OpenAI server: answers the first request with `reply`.
    async fn fake_server(reply: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 65536];
            let _ = sock.read(&mut buf).await;
            sock.write_all(reply.as_bytes()).await.unwrap();
        });
        format!("http://{addr}/v1/chat/completions")
    }

    fn req() -> ChatRequest {
        ChatRequest {
            system: None,
            user: "hi".into(),
            max_tokens: 16,
            temperature: 0.2,
        }
    }

    #[tokio::test]
    async fn send_returns_model_text() {
        let body = r#"{"choices":[{"message":{"content":"bonjour"}}],"usage":{"prompt_tokens":3,"completion_tokens":1}}"#;
        let reply: &'static str = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
        let mut server = row("m", RuntimeKind::LlamaCpp, Some(8192), ServerStatus::Ready);
        server.openai_url = fake_server(reply).await;
        let v = send(&server, &req()).await.unwrap();
        assert_eq!(v["text"], "bonjour");
        assert_eq!(v["usage"]["completion_tokens"], 1);
    }

    #[tokio::test]
    async fn send_flags_budget_spent_before_answering() {
        let body = r#"{"choices":[{"message":{"content":"","reasoning":"..."},"finish_reason":"length"}],"usage":{"prompt_tokens":3,"completion_tokens":16}}"#;
        let reply: &'static str = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
        let mut server = row("m", RuntimeKind::Ollama, None, ServerStatus::Ready);
        server.openai_url = fake_server(reply).await;
        let err = send(&server, &req()).await.unwrap_err();
        assert_eq!(err.body()["code"], "chat_failed");
        assert!(err.to_string().contains("raise max_tokens"));
    }

    #[tokio::test]
    async fn send_surfaces_server_error_message() {
        let body = r#"{"error":{"message":"the request exceeds the available context size"}}"#;
        let reply: &'static str = Box::leak(
            format!(
                "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
        let mut server = row("m", RuntimeKind::LlamaCpp, Some(8192), ServerStatus::Ready);
        server.openai_url = fake_server(reply).await;
        let err = send(&server, &req()).await.unwrap_err();
        let body = err.body();
        assert_eq!(body["code"], "chat_failed");
        assert_eq!(body["http_status"], 400);
        assert!(body["message"].as_str().unwrap().contains("context size"));
    }
}
