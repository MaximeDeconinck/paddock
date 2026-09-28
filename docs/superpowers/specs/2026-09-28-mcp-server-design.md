# paddock mcp design

Date: 2026-09-28
Status: approved (design)

## Problem

paddock knows what fits this Mac, how fast it runs, and how to launch it, but
only a human at a terminal can ask. Coding agents (Claude Code, Cursor, Claude
Desktop) cannot: they would have to shell out to `paddock fit --json`, parse
tables, and drive `paddock serve` whose lifecycle is written for a TTY
(`eprintln!` progress, `stdin` install confirmation, `process::exit(1)` on
ambiguity). The README roadmap promises an MCP server so an agent can ask "what
can this machine run?" and get an OpenAI-compatible endpoint back. This is that
server.

## Goal

`paddock mcp` exposes provisioning over the Model Context Protocol on stdio:
an agent discovers the hardware, ranks models, serves one, gets its endpoint,
lists and stops servers. Any MCP client that can spawn a process can use it
with zero configuration beyond the command line.

This is the first brick of a local delegation setup (a remote orchestrator
handing sub-tasks to local models). Relaying prompts (`chat`) and spawning
local agent runners (`run_agent`) are explicitly out of scope; they are the
v2 items listed at the end.

## Decisions (locked)

- **Provisioning only.** Six tools: `paddock_scan`, `paddock_fit`,
  `paddock_recommend`, `paddock_serve`, `paddock_ps`, `paddock_stop`. No
  `sync` (slow, networked), no `logs`, no `run` (interactive, needs a TTY).
- **stdio transport only.** No HTTP listener, no auth, no port to manage.
- **`rmcp` 3.x (official Rust SDK)**, features `server`, `transport-io`,
  `macros`, plus `schemars` for input schemas. No hand-rolled JSON-RPC.
- **Same binary.** `paddock mcp` is a subcommand. The serve lifecycle lives in
  the binary crate; a separate crate would force it into `paddock-core` (which
  spawns no processes today) or duplicate it. `paddock-core` is untouched.
- **Lifecycle extracted, not duplicated.** The TTY-coupled functions in
  `main.rs` move to a `lifecycle` module that returns values and reports
  progress through a trait. CLI, TUI and MCP are three adapters over one
  implementation.
- **`serve` blocks until ready, with a timeout.** `timeout_secs` (default 600).
  On timeout the server is left running and the tool returns
  `status: "starting"` with the pid; the agent polls `paddock_ps`.
- **No install from MCP.** A missing runtime is a structured error carrying the
  install command. The agent asks the human; paddock never installs on an
  agent's say-so.
- **No `stop all` from MCP.** `target` is a model name or pid. The agent loops
  if it wants everything down.

## Components

### 1. `lifecycle` module (`crates/paddock/src/lifecycle.rs`)

Everything in `main.rs` that prints, reads stdin, or exits becomes a function
that returns a value.

```rust
pub trait Progress {
    /// Informational line (today's `eprintln!`): "downloading/loading model…",
    /// "port 8080 is busy - serving on 8081 instead", heartbeats.
    fn note(&self, msg: &str);
    /// A command about to run (today's "$ ollama pull …").
    fn command(&self, argv: &[String]);
}
pub struct StderrProgress; // CLI + TUI: byte-for-byte today's messages
pub struct SilentProgress; // MCP: discards

pub enum InstallPolicy { Ask, Refuse }
pub enum ServeMode { Foreground, Detached }

pub enum LifecycleError {
    ModelNotFound(String),
    Ambiguous { query: String, candidates: Vec<String> },
    NoFit { model: String, ram_bytes: u64 },
    UnknownQuant { label: String, available: Vec<String> },
    NoRuntime { install: InstallPlan },
    InstallDeclined,
    ServerExited { status: String, argv: Vec<String>, log_path: Option<PathBuf> },
    /// Readiness timeout on a detached child. The server is still running.
    NotReady { pid: Option<u32>, log_path: Option<PathBuf> },
    OllamaUnreachable,
    NoServerMatch { target: String, running: Vec<String> },
    AmbiguousServer { target: String, candidates: Vec<(String, u32)> },
    CatalogEmpty,
    Other(anyhow::Error),
}

pub struct ServeOutcome {
    pub plan: ServePlan,          // final plan (port may have moved)
    pub pid: Option<u32>,         // spawned child; None for the Ollama daemon
    pub log_path: Option<PathBuf>,
}

pub struct Stopped { pub model_ref: String, pub pid: u32 }

pub fn resolve_model(app: &App, query: &str, quant: Option<&str>)
    -> Result<(CatalogModel, usize), LifecycleError>;

pub fn serve(
    plan: ServePlan,
    mode: ServeMode,
    policy: InstallPolicy,
    ready_timeout: Option<Duration>,
    progress: &dyn Progress,
) -> Result<ServeOutcome, LifecycleError>;

pub fn stop(target: &str, progress: &dyn Progress)
    -> Result<Vec<Stopped>, LifecycleError>;
```

Behavior:

- `resolve_model` is today's function minus `eprintln!`/`exit`: ambiguity is
  `Ambiguous { candidates }`, an empty catalog is `CatalogEmpty`.
- `serve` is today's `serve_with_plan`: port fallback, history record, spawn
  (foreground or detached with the pending-log rename), readiness wait,
  pre-steps, Ollama warm-up, registry entry. Differences:
  - `InstallPolicy::Ask` calls today's `confirm_and_install` (stays in
    `main.rs`, passed as a closure or reached through the policy); `Refuse`
    returns `NoRuntime { install }` before touching anything.
  - `wait_ready(plan, child, timeout)` takes an explicit `Option<Duration>`.
    The CLI passes `None` for spawned servers (today: wait forever, heartbeat
    every 60 s through `progress.note`). The Ollama-daemon case keeps its
    existing short deadline and maps to `OllamaUnreachable`. When `timeout`
    elapses on a **detached** child, the child is *not* killed and the result
    is `NotReady { pid, log_path }`; the registry entry is written first so
    `ps` can see it. A foreground child on timeout is killed as today
    (foreground is CLI-only and never passes a timeout, so this branch is
    defensive).
  - `serve` returns as soon as the endpoint is ready, in both modes. In
    `Foreground` mode `ServeOutcome.child` is `Some(std::process::Child)`;
    the CLI prints the endpoint block, registers the `RegistryGuard`, and
    waits on the child itself (today's "serving - press Ctrl-C to stop"
    branch). In `Detached` mode `child` is `None` and the registry entry has
    already been written by `serve`. The struct therefore gains a
    `pub child: Option<std::process::Child>` field.
- `stop` is today's `stop_servers` minus the confirmation prompt and the
  `exit(1)` paths. `"all"` is still understood (via `match_records`); the
  y/N prompt moves up into `main.rs`, which calls `lifecycle::stop` only after
  confirmation. The MCP adapter rejects `"all"` before calling.

Every `LifecycleError` variant has a `Display` rendering that reproduces
today's CLI wording exactly, so the 14 `assert_cmd` tests pass unchanged.
`main.rs` maps `Ambiguous`/`AmbiguousServer`/`NoServerMatch` to the same
stderr text + exit code 1 as before.

### 2. MCP server (`crates/paddock/src/mcp.rs`)

- `Command::Mcp` in `cli.rs`. `--json` and `--cli` are ignored with it.
- `struct PaddockMcp { app: Arc<App> }` with `#[tool_router]`. `App::load()`
  runs once at startup (hardware probe, calibration); the catalog DB is
  reopened per call so a `paddock sync` in another terminal is picked up.
- `ServerHandler::get_info` sets `instructions` for the agent:
  what paddock is, that `paddock_serve` may block up to `timeout_secs`, that
  `status: "starting"` means poll `paddock_ps`, that `openai_url` speaks the
  OpenAI chat-completions protocol with `model_ref` in the `model` field, and
  that a `no_runtime` error must be shown to the human, not acted on.
- Tool bodies run under `tokio::task::spawn_blocking` (lifecycle code sleeps
  and does blocking HTTP through `RealSystemProbe`).
- Results: `CallToolResult` with one text block containing the JSON and the
  same value in `structured_content`. Errors: `CallToolResult::error` with a
  JSON `{ code, message, ... }` body (table below).
- stdout is the protocol channel: nothing else may write to it. Internal
  failures go to stderr.

#### Tools

| Tool | Input | Output | Annotations |
|---|---|---|---|
| `paddock_scan` | none | `HardwareProfile` as serialized by `scan --json` | readOnly |
| `paddock_fit` | `use_case?` (`general`\|`coding`\|`chat`\|`reasoning`, default general), `limit?` (default 10), `include_unfit?` (default false) | `[FitRow]`: same shape as `fit --json` | readOnly |
| `paddock_recommend` | `use_case?` | top 5 `FitRow` each with a `justification` string | readOnly |
| `paddock_serve` | `model` (name or substring), `quant?`, `ctx?`, `port?`, `timeout_secs?` (default 600) | `{ status: "ready"\|"starting", endpoint, openai_url, model_ref, runtime, ctx, port, pid?, log_path? }` | none |
| `paddock_ps` | none | `{ running: [ServerRow], available: [AvailableRow] }`, same as `ps --json` | readOnly |
| `paddock_stop` | `target` (model name or pid) | `{ stopped: [{ model_ref, pid }] }` | destructive |

Tool descriptions are written for the agent: when to call, what `fits` /
`tune sysctl` / `ram only` mean, that `serve` picks the best fitting quant
unless `quant` is given.

#### Error codes

| `LifecycleError` | `code` | extra fields |
|---|---|---|
| `ModelNotFound` | `model_not_found` | `hint: "call paddock_fit for names"` |
| `Ambiguous` | `ambiguous` | `candidates` |
| `NoFit` | `no_fit` | `ram_gib` |
| `UnknownQuant` | `unknown_quant` | `available` |
| `NoRuntime` | `no_runtime` | `install_command`, `hint: "ask the user before installing"` |
| `ServerExited` | `server_exited` | `argv`, `log_path` |
| `OllamaUnreachable` | `ollama_unreachable` | |
| `NoServerMatch` | `no_server_match` | `running` |
| `AmbiguousServer` | `ambiguous_server` | `candidates` |
| `CatalogEmpty` | `catalog_empty` | `hint: "run paddock sync"` |
| `target == "all"` on stop | `invalid_target` | `hint: "stop servers one at a time"` |
| `Other` | `internal` | |

`NotReady` is not an error at the MCP layer: it becomes a normal result with
`status: "starting"`.

### 3. CLI and TUI adapters

- `main.rs`: `serve_model`, `run_model`'s install path, `stop_servers` become
  thin wrappers: build the plan, call `lifecycle`, render `LifecycleError`
  through its `Display`, keep the `exit(1)` sites for the ambiguity cases.
  The `stop all` y/N prompt stays here.
- `tui/mod.rs`: the `s` key path already calls `serve_with_plan(plan, false)`
  after suspending the TUI; it calls `lifecycle::serve(plan, Detached, Ask,
  None, &StderrProgress)` instead. No behavior change.

### 4. Distribution and docs

- `rmcp`, `schemars` added to `crates/paddock/Cargo.toml`. Binary size grows;
  acceptable, one binary to distribute via cargo-dist.
- README: new `paddock mcp` section under the subcommands, with
  `claude mcp add paddock -- paddock mcp` and the Claude Desktop JSON
  snippet, the tool list, and the `starting` / `no_runtime` contracts.
  Roadmap entry replaced by v2 items (below).

## Testing

- `lifecycle` unit tests: error mapping for each variant; `wait_ready` with a
  timeout against a fake `SystemProbe` that never answers (returns
  `NotReady`, child left alive), and one that answers on the third poll.
- `mcp` integration test (`assert_cmd`): spawn `paddock mcp`, write
  `initialize` + `notifications/initialized` + `tools/list` to stdin, assert
  six tools with the expected names and that each input schema parses. One
  `tools/call paddock_ps` against an empty registry (reuse the temp-dir env
  var the existing `ps` tests use) returns `{ running: [], available: [...] }`.
  One `tools/call paddock_stop` with `target: "all"` returns `invalid_target`.
- Regression: the existing 14 CLI tests and 54 TUI tests pass with no edits.
  That is the contract that the CLI wording did not change.

## Out of scope (v2 candidates)

- `paddock_chat(model, messages)`: relay a completion to a served endpoint so
  an orchestrator can delegate text sub-tasks without leaving MCP.
- `paddock_run_agent(model, task, cwd)`: spawn a local agent runner (aider,
  opencode, …) against a served endpoint and return its diff.
- HTTP transport for remote or multi-client use.
- `paddock_sync` and `paddock_logs`.
- MCP resources (`paddock://hardware`, `paddock://catalog`).
