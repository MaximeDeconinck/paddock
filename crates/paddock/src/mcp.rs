//! `paddock mcp`: the provisioning tools over the Model Context Protocol on
//! stdio. Third adapter over `lifecycle` (after the CLI and the TUI): no
//! prompts, no install, JSON in, JSON out. stdout is the protocol channel;
//! nothing else may write to it.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use paddock_core::hardware::RealSystemProbe;
use paddock_core::runtime::{ServePlan, plan_serve};
use paddock_core::score::UseCase;
use paddock_core::serving::Registry;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::app::{App, ScoredModel};
use crate::lifecycle::{
    InstallPolicy, LifecycleError, ServeMode, SilentProgress, resolve_model, resolved_ctx,
};
use crate::output;

const DEFAULT_SERVE_TIMEOUT_SECS: u64 = 600;

/// Shown to the agent at `initialize`. Written for the model, not the human.
pub(crate) const INSTRUCTIONS: &str = "\
paddock knows which LLMs fit this Apple Silicon Mac, how fast they run, and how to serve them.
Workflow: paddock_scan (hardware) -> paddock_fit or paddock_recommend (ranked models) -> \
paddock_serve (start one, get an endpoint) -> paddock_ps / paddock_stop (manage).
paddock_serve blocks until the server answers, up to timeout_secs (default 600); a first \
run may download a multi-GB model. status \"starting\" means the timeout elapsed but the \
server is still coming up. With runtime \"ollama\" the model is still downloading: call \
paddock_serve again later with the same arguments (it returns ready once the pull finished). \
With llama.cpp / mlx the server is loading: poll paddock_ps until it is listed, then use the \
endpoint; if it never appears the server may have exited, and log_path from the starting \
result holds its log.
openai_url speaks the OpenAI chat-completions protocol; put model_ref in the `model` field.
An error with code \"no_runtime\" means a runtime (ollama, llama.cpp, mlx-lm) is not installed: \
show install_command to the user and ask before anything is installed. paddock never installs \
on an agent's request.
Fit verdicts: \"fits\" = whole model in GPU memory; \"tune sysctl\" = fits after raising the \
Metal working-set limit; \"ram only\" = CPU offload, slow; \"no fit\" = do not try.";

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum UseCaseInput {
    General,
    Coding,
    Chat,
    Reasoning,
}

impl From<UseCaseInput> for UseCase {
    fn from(v: UseCaseInput) -> Self {
        match v {
            UseCaseInput::General => UseCase::General,
            UseCaseInput::Coding => UseCase::Coding,
            UseCaseInput::Chat => UseCase::Chat,
            UseCaseInput::Reasoning => UseCase::Reasoning,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FitInput {
    /// Scoring profile: general (default), coding, chat or reasoning.
    #[serde(default)]
    pub use_case: Option<UseCaseInput>,
    /// Max rows (default 10).
    #[serde(default)]
    pub limit: Option<usize>,
    /// Also list models that do not fit (default false).
    #[serde(default)]
    pub include_unfit: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecommendInput {
    /// Scoring profile: general (default), coding, chat or reasoning.
    #[serde(default)]
    pub use_case: Option<UseCaseInput>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ServeInput {
    /// Catalog model name, or a unique substring of one (see paddock_fit).
    pub model: String,
    /// Quantization label (e.g. Q4_K_M); default picks the best fitting one.
    #[serde(default)]
    pub quant: Option<String>,
    /// Context window in tokens (llama.cpp only); default auto-sizes to memory.
    #[serde(default)]
    pub ctx: Option<u32>,
    /// Port for llama.cpp / mlx servers (Ollama always uses 11434).
    #[serde(default)]
    pub port: Option<u16>,
    /// Seconds to wait for readiness before returning status "starting" (default 600).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StopInput {
    /// Model name substring or pid of a running server. "all" is rejected.
    pub target: String,
}

/// MCP error body: `{ code, message, ...extra }`, one code per variant.
pub(crate) fn error_body(e: &LifecycleError) -> Value {
    let (code, extra) = match e {
        LifecycleError::ModelNotFound(_) => (
            "model_not_found",
            json!({ "hint": "call paddock_fit for names" }),
        ),
        LifecycleError::Ambiguous { candidates, .. } => {
            ("ambiguous", json!({ "candidates": candidates }))
        }
        LifecycleError::NoFit { ram_bytes, .. } => (
            "no_fit",
            json!({ "ram_gib": *ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0) }),
        ),
        LifecycleError::UnknownQuant { available, .. } => {
            ("unknown_quant", json!({ "available": available }))
        }
        LifecycleError::NoRuntime { install } => (
            "no_runtime",
            json!({
                "install_command": install.argv.join(" "),
                "hint": "ask the user before installing",
            }),
        ),
        LifecycleError::InstallDeclined { .. } => ("install_declined", json!({})),
        LifecycleError::ServerExited { argv, log_path, .. } => (
            "server_exited",
            json!({ "argv": argv, "log_path": log_path }),
        ),
        // Mapped to a `starting` result by `paddock_serve`; listed for completeness.
        LifecycleError::NotReady { pid, log_path, .. } => {
            ("not_ready", json!({ "pid": pid, "log_path": log_path }))
        }
        LifecycleError::OllamaUnreachable => ("ollama_unreachable", json!({})),
        LifecycleError::NoServerMatch { running, .. } => {
            ("no_server_match", json!({ "running": running }))
        }
        LifecycleError::AmbiguousServer { candidates, .. } => (
            "ambiguous_server",
            json!({
                "candidates": candidates
                    .iter()
                    .map(|(m, p)| json!({ "model_ref": m, "pid": p }))
                    .collect::<Vec<_>>()
            }),
        ),
        LifecycleError::CatalogEmpty => ("catalog_empty", json!({ "hint": "run paddock sync" })),
        LifecycleError::Other(_) => ("internal", json!({})),
    };
    let mut body = json!({ "code": code, "message": e.to_string() });
    if let (Some(dst), Some(src)) = (body.as_object_mut(), extra.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }
    body
}

/// `paddock_fit` result: an object (MCP `structuredContent` must be one).
pub(crate) fn fit_result(rows: &[ScoredModel]) -> Result<Value, LifecycleError> {
    Ok(
        json!({ "models": serde_json::to_value(output::fit_rows(rows)).map_err(anyhow::Error::from)? }),
    )
}

/// `paddock_recommend` result: an object (MCP `structuredContent` must be one).
pub(crate) fn recommend_result(rows: &[ScoredModel]) -> Result<Value, LifecycleError> {
    Ok(json!({
        "recommendations": serde_json::to_value(output::recommend_rows(rows)).map_err(anyhow::Error::from)?
    }))
}

/// `paddock_stop` refuses `all` over MCP: stopping every server is a human
/// decision. Returns the error result when `target` is rejected.
pub(crate) fn reject_stop_target(target: &str) -> Option<CallToolResult> {
    (target == "all").then(|| {
        CallToolResult::structured_error(json!({
            "code": "invalid_target",
            "message": "`all` is not accepted over MCP",
            "hint": "stop servers one at a time",
        }))
    })
}

/// The `paddock_serve` result for both `ready` and `starting`.
pub(crate) fn serve_result(
    status: &str,
    plan: &ServePlan,
    pid: Option<u32>,
    log_path: Option<PathBuf>,
) -> Value {
    json!({
        "status": status,
        "endpoint": plan.endpoint,
        "openai_url": plan.openai_url,
        "model_ref": plan.model_ref,
        "runtime": plan.runtime,
        "ctx": plan.ctx,
        "port": plan.port,
        "pid": pid,
        "log_path": log_path,
    })
}

fn ok(value: Value) -> CallToolResult {
    CallToolResult::structured(value)
}

fn fail(e: &LifecycleError) -> CallToolResult {
    CallToolResult::structured_error(error_body(e))
}

/// Run blocking lifecycle code off the async executor and render the result.
/// A panic in the closure becomes an `internal` error instead of killing the
/// session.
async fn blocking<F>(f: F) -> CallToolResult
where
    F: FnOnce() -> Result<Value, LifecycleError> + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(v)) => ok(v),
        Ok(Err(e)) => fail(&e),
        Err(join) => fail(&LifecycleError::Other(anyhow::anyhow!(
            "tool panicked: {join}"
        ))),
    }
}

#[derive(Clone)]
pub struct PaddockMcp {
    app: Arc<App>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl PaddockMcp {
    pub fn new(app: App) -> Self {
        Self {
            app: Arc::new(app),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "paddock_scan",
        description = "Hardware profile of this Mac: chip, RAM, GPU memory limit, memory bandwidth, and which runtimes (ollama, llama.cpp, mlx-lm) are installed or running. Call first to know what you are working with.",
        annotations(read_only_hint = true, idempotent_hint = true)
    )]
    async fn scan(&self) -> CallToolResult {
        match serde_json::to_value(&self.app.profile) {
            Ok(v) => ok(v),
            Err(e) => fail(&LifecycleError::Other(e.into())),
        }
    }

    #[tool(
        name = "paddock_fit",
        description = "Catalog models ranked for this machine: quant picked, memory estimate, generation tok/s estimate, fit verdict (fits / tune sysctl / ram only / no fit) and score, as { \"models\": [...] }. An empty models array means nothing matched: limit was 0, nothing fits this machine (retry with include_unfit true to see why), or the catalog is empty (then call paddock_recommend: a catalog_empty error confirms it and the user must run `paddock sync`).",
        annotations(read_only_hint = true)
    )]
    async fn fit(&self, Parameters(input): Parameters<FitInput>) -> CallToolResult {
        let app = self.app.clone();
        blocking(move || {
            let db = app.open_db()?;
            let use_case = input
                .use_case
                .map(UseCase::from)
                .unwrap_or(UseCase::General);
            let mut rows =
                app.scored_models(&db, use_case, input.include_unfit.unwrap_or(false))?;
            rows.truncate(input.limit.unwrap_or(10));
            fit_result(&rows)
        })
        .await
    }

    #[tool(
        name = "paddock_recommend",
        description = "Top 5 models for this machine with a one-line justification each (fit headroom, speed tier, context), as { \"recommendations\": [...] }. Use when you want a short answer instead of the full ranking.",
        annotations(read_only_hint = true)
    )]
    async fn recommend(&self, Parameters(input): Parameters<RecommendInput>) -> CallToolResult {
        let app = self.app.clone();
        blocking(move || {
            let db = app.open_db()?;
            let use_case = input
                .use_case
                .map(UseCase::from)
                .unwrap_or(UseCase::General);
            let mut rows = app.scored_models(&db, use_case, false)?;
            rows.truncate(5);
            recommend_result(&rows)
        })
        .await
    }

    #[tool(
        name = "paddock_serve",
        description = "Start serving a catalog model with the best available runtime and return an OpenAI-compatible endpoint. Picks the best fitting quant unless `quant` is given. Blocks until ready (up to timeout_secs, default 600; a first run may download the model). status \"starting\" means the timeout elapsed first: with runtime \"ollama\" the model is still downloading, so call paddock_serve again later with the same arguments (it returns ready once the pull finished); with llama.cpp / mlx the server is loading, so poll paddock_ps (log_path holds its log). Never installs a runtime: a no_runtime error carries the command for the user."
    )]
    async fn serve(&self, Parameters(input): Parameters<ServeInput>) -> CallToolResult {
        let app = self.app.clone();
        blocking(move || {
            let (model, idx) = resolve_model(&app, &input.model, input.quant.as_deref())?;
            let ctx = Some(resolved_ctx(&app, &model, idx, input.ctx));
            let plan = plan_serve(
                &model,
                &model.variants[idx],
                &app.profile.runtimes,
                input.port,
                ctx,
            )?;
            let timeout =
                Duration::from_secs(input.timeout_secs.unwrap_or(DEFAULT_SERVE_TIMEOUT_SECS));
            match crate::lifecycle::serve(
                plan,
                ServeMode::Detached,
                InstallPolicy::Refuse,
                Some(timeout),
                &SilentProgress,
            ) {
                Ok(outcome) => Ok(serve_result(
                    "ready",
                    &outcome.plan,
                    outcome.pid,
                    outcome.log_path,
                )),
                Err(LifecycleError::NotReady {
                    pid,
                    log_path,
                    plan,
                }) => Ok(serve_result("starting", &plan, pid, log_path)),
                Err(e) => Err(e),
            }
        })
        .await
    }

    #[tool(
        name = "paddock_ps",
        description = "Running servers (paddock-spawned llama.cpp / mlx servers and models loaded in the Ollama daemon) with endpoint, context and pid, plus models available locally but not running.",
        annotations(read_only_hint = true)
    )]
    async fn ps(&self) -> CallToolResult {
        blocking(move || {
            let probe = RealSystemProbe;
            let registry = Registry::open_default();
            let running = paddock_core::serving::list_all_servers(&registry, &probe);
            let history = paddock_core::serving::History::open_default();
            let available = paddock_core::serving::list_available(&history, &probe, &running);
            Ok(json!({ "running": running, "available": available }))
        })
        .await
    }

    #[tool(
        name = "paddock_stop",
        description = "Stop one running server by model name substring or pid. Ollama-loaded models are unloaded from the daemon; paddock-spawned servers are terminated. \"all\" is rejected: stop servers one at a time.",
        annotations(destructive_hint = true)
    )]
    async fn stop(&self, Parameters(input): Parameters<StopInput>) -> CallToolResult {
        if let Some(rejected) = reject_stop_target(&input.target) {
            return rejected;
        }
        blocking(move || {
            let stopped = crate::lifecycle::stop(&input.target, &SilentProgress)?;
            Ok(json!({
                "stopped": stopped
                    .iter()
                    .map(|s| json!({ "model_ref": s.model_ref, "pid": s.pid }))
                    .collect::<Vec<_>>()
            }))
        })
        .await
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for PaddockMcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("paddock", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }
}

/// Block on a stdio MCP session until the client closes stdin.
pub fn run(app: App) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let service = PaddockMcp::new(app).serve(rmcp::transport::stdio()).await?;
        service.waiting().await?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_core::catalog::RuntimeKind;
    use paddock_core::runtime::InstallPlan;

    #[test]
    fn error_body_no_runtime_carries_install_command_and_hint() {
        let e = LifecycleError::NoRuntime {
            install: InstallPlan {
                kind: RuntimeKind::LlamaCpp,
                argv: vec!["brew".into(), "install".into(), "llama.cpp".into()],
            },
        };
        let v = error_body(&e);
        assert_eq!(v["code"], "no_runtime");
        assert_eq!(v["install_command"], "brew install llama.cpp");
        assert_eq!(v["hint"], "ask the user before installing");
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("brew install llama.cpp")
        );
    }

    #[test]
    fn error_body_codes_match_spec_table() {
        let cases: Vec<(LifecycleError, &str)> = vec![
            (LifecycleError::ModelNotFound("x".into()), "model_not_found"),
            (
                LifecycleError::Ambiguous {
                    query: "l".into(),
                    candidates: vec!["a".into(), "b".into()],
                },
                "ambiguous",
            ),
            (
                LifecycleError::NoFit {
                    model: "x".into(),
                    ram_bytes: 8 << 30,
                },
                "no_fit",
            ),
            (
                LifecycleError::UnknownQuant {
                    label: "Q9".into(),
                    available: vec!["Q4_K_M".into()],
                },
                "unknown_quant",
            ),
            (
                LifecycleError::ServerExited {
                    status: "exit status: 1".into(),
                    argv: vec!["llama-server".into()],
                    log_path: None,
                },
                "server_exited",
            ),
            (LifecycleError::OllamaUnreachable, "ollama_unreachable"),
            (
                LifecycleError::NoServerMatch {
                    target: "x".into(),
                    running: vec!["a".into()],
                },
                "no_server_match",
            ),
            (
                LifecycleError::AmbiguousServer {
                    target: "x".into(),
                    candidates: vec![("a".into(), 1)],
                },
                "ambiguous_server",
            ),
            (LifecycleError::CatalogEmpty, "catalog_empty"),
            (LifecycleError::Other(anyhow::anyhow!("boom")), "internal"),
        ];
        for (e, code) in cases {
            let v = error_body(&e);
            assert_eq!(v["code"], code, "{e}");
            assert!(v["message"].is_string());
        }
    }

    #[test]
    fn error_body_extra_fields() {
        let v = error_body(&LifecycleError::ModelNotFound("x".into()));
        assert_eq!(v["hint"], "call paddock_fit for names");

        let v = error_body(&LifecycleError::Ambiguous {
            query: "l".into(),
            candidates: vec!["a".into(), "b".into()],
        });
        assert_eq!(v["candidates"], serde_json::json!(["a", "b"]));

        let v = error_body(&LifecycleError::NoFit {
            model: "x".into(),
            ram_bytes: 8 << 30,
        });
        assert_eq!(v["ram_gib"], 8.0);

        let v = error_body(&LifecycleError::UnknownQuant {
            label: "Q9".into(),
            available: vec!["Q4_K_M".into()],
        });
        assert_eq!(v["available"], serde_json::json!(["Q4_K_M"]));

        let v = error_body(&LifecycleError::ServerExited {
            status: "exit status: 1".into(),
            argv: vec!["llama-server".into()],
            log_path: Some("/tmp/1.log".into()),
        });
        assert_eq!(v["argv"], serde_json::json!(["llama-server"]));
        assert_eq!(v["log_path"], "/tmp/1.log");

        let v = error_body(&LifecycleError::NoServerMatch {
            target: "x".into(),
            running: vec!["a".into()],
        });
        assert_eq!(v["running"], serde_json::json!(["a"]));

        let v = error_body(&LifecycleError::AmbiguousServer {
            target: "x".into(),
            candidates: vec![("a".into(), 1)],
        });
        assert_eq!(
            v["candidates"],
            serde_json::json!([{ "model_ref": "a", "pid": 1 }])
        );

        let v = error_body(&LifecycleError::CatalogEmpty);
        assert_eq!(v["hint"], "run paddock sync");
    }

    #[test]
    fn error_body_other_is_internal() {
        // Review Focus 1: plan_serve rejects port 0 with a PaddockError; it
        // reaches the agent as `internal` with the message, never a panic.
        let e: LifecycleError = paddock_core::PaddockError::Other(
            "port 0 is not supported; pick a fixed port so the endpoint is known upfront".into(),
        )
        .into();
        let v = error_body(&e);
        assert_eq!(v["code"], "internal");
        assert!(v["message"].as_str().unwrap().contains("port 0"));
    }

    #[test]
    fn serve_result_starting_carries_pid_and_endpoint() {
        let plan = paddock_core::runtime::ServePlan {
            server_argv: Some(vec!["llama-server".into()]),
            pre_steps: vec![],
            endpoint: "http://127.0.0.1:8081".into(),
            openai_url: "http://127.0.0.1:8081/v1/chat/completions".into(),
            model_ref: "x".into(),
            ready_path: "/health".into(),
            install: None,
            port_ignored: false,
            runtime: RuntimeKind::LlamaCpp,
            ctx: 4096,
            port: Some(8081),
        };
        let v = serve_result(
            "starting",
            &plan,
            Some(42),
            Some(std::path::PathBuf::from("/tmp/42.log")),
        );
        assert_eq!(v["status"], "starting");
        assert_eq!(v["endpoint"], "http://127.0.0.1:8081");
        assert_eq!(v["openai_url"], "http://127.0.0.1:8081/v1/chat/completions");
        assert_eq!(v["model_ref"], "x");
        assert_eq!(v["runtime"], "llama_cpp");
        assert_eq!(v["ctx"], 4096);
        assert_eq!(v["port"], 8081);
        assert_eq!(v["pid"], 42);
        assert_eq!(v["log_path"], "/tmp/42.log");
    }

    #[test]
    fn fit_and_recommend_results_are_objects_wrapping_arrays() {
        let rows = vec![crate::output::tests::scored()];
        let v = fit_result(&rows).unwrap();
        assert!(v.is_object());
        assert_eq!(v["models"].as_array().unwrap().len(), 1);
        assert_eq!(v["models"][0]["name"], "fake-model");
        let v = recommend_result(&rows).unwrap();
        assert!(v.is_object());
        assert_eq!(v["recommendations"].as_array().unwrap().len(), 1);

        assert_eq!(
            fit_result(&[]).unwrap(),
            serde_json::json!({ "models": [] })
        );
        assert_eq!(
            recommend_result(&[]).unwrap(),
            serde_json::json!({ "recommendations": [] })
        );
    }

    #[tokio::test]
    async fn stop_all_is_rejected_with_invalid_target() {
        let mcp = PaddockMcp::new(crate::app::App::load());
        let r = mcp
            .stop(Parameters(StopInput {
                target: "all".into(),
            }))
            .await;
        assert_eq!(r.is_error, Some(true));
        let body = r.structured_content.expect("structured error body");
        assert_eq!(body["code"], "invalid_target");
        assert!(reject_stop_target("llama").is_none());
    }

    #[test]
    fn tool_list_has_six_tools_with_spec_names_and_annotations() {
        let router = PaddockMcp::tool_router();
        let mut names: Vec<String> = router
            .list_all()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "paddock_fit",
                "paddock_ps",
                "paddock_recommend",
                "paddock_scan",
                "paddock_serve",
                "paddock_stop",
            ]
        );
        for t in router.list_all() {
            let ann = t.annotations.clone().unwrap_or_default();
            match t.name.as_ref() {
                "paddock_serve" => assert_ne!(ann.read_only_hint, Some(true)),
                "paddock_stop" => assert_eq!(ann.destructive_hint, Some(true)),
                _ => assert_eq!(ann.read_only_hint, Some(true), "{}", t.name),
            }
            assert!(
                t.description.as_deref().unwrap_or("").len() > 20,
                "{}",
                t.name
            );
        }
    }
}
