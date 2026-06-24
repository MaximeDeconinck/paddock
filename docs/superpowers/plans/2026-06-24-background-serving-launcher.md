# Background Serving Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `paddock serve` into a detached, manageable launcher — no dedicated terminal per model — with `ps`/`stop`/`logs` lifecycle and a context window that auto-sizes to available memory.

**Architecture:** Detached spawn via `setsid` in `pre_exec`, with stdout/stderr redirected to a per-pid log file and the process recorded in the existing on-disk `Registry`. New CLI subcommands read/act on that registry. Context auto-sizing computes the largest context whose memory still fits the GPU budget, reusing the existing estimator.

**Tech Stack:** Rust (edition 2024), clap (CLI), serde/serde_json (registry), `std::os::unix::process::CommandExt` + tiny `extern "C"` syscalls (`setsid`, `kill`) following the codebase's existing libc-free style, `assert_cmd` + `tempfile` (integration tests).

---

## File structure

- `crates/paddock-core/src/serving.rs` — extend `ServingRecord`; add `match_records` target resolver. (Modify)
- `crates/paddock-core/src/estimate.rs` — add `auto_ctx` + `resolve_ctx`. (Modify)
- `crates/paddock/src/cli.rs` — `--foreground` on `Serve`, new `Ps`/`Stop`/`Logs` subcommands. (Modify)
- `crates/paddock/src/main.rs` — detached serve path, ctx resolution, command dispatch + handlers. (Modify)
- `crates/paddock/src/output.rs` — `ps` table + JSON formatting. (Modify)
- `crates/paddock/tests/cli.rs` — integration tests for new commands. (Modify)
- `README.md` — document the launcher. (Modify)

---

## Task 1: Extend `ServingRecord` with `ctx`, `log_path`, `port`

**Files:**
- Modify: `crates/paddock-core/src/serving.rs:12-21`
- Test: `crates/paddock-core/src/serving.rs` (inline `#[cfg(test)]`)

- [ ] **Step 1: Write the failing tests**

Add to the test module at the bottom of `serving.rs` (create the module if absent):

```rust
#[cfg(test)]
mod record_tests {
    use super::*;

    #[test]
    fn record_roundtrips_new_fields() {
        let r = ServingRecord {
            pid: 42,
            runtime: RuntimeKind::LlamaCpp,
            endpoint: "http://127.0.0.1:8080".into(),
            openai_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
            model_ref: "repo:Q4_K_M".into(),
            ready_path: "/health".into(),
            started_at: 1000,
            ctx: 32768,
            log_path: Some(std::path::PathBuf::from("/tmp/42.log")),
            port: Some(8080),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: ServingRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ctx, 32768);
        assert_eq!(back.log_path, Some(std::path::PathBuf::from("/tmp/42.log")));
        assert_eq!(back.port, Some(8080));
    }

    #[test]
    fn old_record_without_new_fields_still_deserializes() {
        // A record written before this change has no ctx/log_path/port.
        let old = r#"{
            "pid": 7, "runtime": "Ollama",
            "endpoint": "http://127.0.0.1:11434",
            "openai_url": "http://127.0.0.1:11434/v1/chat/completions",
            "model_ref": "llama3.2:1b", "ready_path": "/api/version",
            "started_at": 5
        }"#;
        let r: ServingRecord = serde_json::from_str(old).unwrap();
        assert_eq!(r.ctx, 0);
        assert_eq!(r.log_path, None);
        assert_eq!(r.port, None);
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test -p paddock-core record_tests`
Expected: FAIL — compile error, `ServingRecord` has no field `ctx`.

- [ ] **Step 3: Add the fields**

In `crates/paddock-core/src/serving.rs`, replace the struct (lines 12-21):

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServingRecord {
    pub pid: u32,
    pub runtime: RuntimeKind,
    pub endpoint: String,
    pub openai_url: String,
    pub model_ref: String,
    pub ready_path: String,
    pub started_at: i64,
    /// Resolved context window the server was launched with (0 for legacy records).
    #[serde(default)]
    pub ctx: u32,
    /// Log file for detached spawned children; None for foreground / Ollama.
    #[serde(default)]
    pub log_path: Option<std::path::PathBuf>,
    /// Port for spawned servers (None for the Ollama daemon).
    #[serde(default)]
    pub port: Option<u16>,
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cargo test -p paddock-core record_tests`
Expected: PASS (2 tests).

- [ ] **Step 5: Fix the existing constructor**

The only constructor is `RegistryGuard::register` in `crates/paddock/src/main.rs:273-281`. Add the three new fields so the workspace compiles:

```rust
        let record = ServingRecord {
            pid,
            runtime: plan.runtime,
            endpoint: plan.endpoint.clone(),
            openai_url: plan.openai_url.clone(),
            model_ref: plan.model_ref.clone(),
            ready_path: plan.ready_path.clone(),
            started_at,
            ctx: 0,
            log_path: None,
            port: None,
        };
```

Run: `cargo build`
Expected: builds clean (the detached path in Task 4 will set real values).

- [ ] **Step 6: Commit**

```bash
git add crates/paddock-core/src/serving.rs crates/paddock/src/main.rs
git commit -m "feat(core): ctx/log_path/port fields on ServingRecord"
```

---

## Task 2: `auto_ctx` and `resolve_ctx` in the estimator

**Files:**
- Modify: `crates/paddock-core/src/estimate.rs` (after `kv_cache_bytes`, ~line 129)
- Test: `crates/paddock-core/src/estimate.rs` (existing `#[cfg(test)]` module)

- [ ] **Step 1: Write the failing tests**

Add to the test module in `estimate.rs`. Reuse the existing test fixtures if present (`llama31_70b_q4km()`, `m2_max_36gb()` are referenced elsewhere in this file's tests); otherwise build a `ModelVariant` inline.

```rust
#[test]
fn auto_ctx_picks_largest_fitting_context() {
    // Small model on a roomy budget → should reach context_max.
    let mut v = llama31_8b_q4km();
    v.context_max = 32_768;
    let budget = MemoryBudget {
        gpu_effective_bytes: 24 * 1024 * 1024 * 1024,
        ram_total_bytes: 32 * 1024 * 1024 * 1024,
    };
    assert_eq!(auto_ctx(&v, &budget, v.context_max), 32_768);
}

#[test]
fn auto_ctx_clamps_down_when_kv_cache_is_huge() {
    // Tiny GPU budget → auto_ctx must drop below context_max but never below 4096.
    let mut v = llama31_70b_q4km();
    v.context_max = 131_072;
    let budget = MemoryBudget {
        gpu_effective_bytes: 40 * 1024 * 1024 * 1024,
        ram_total_bytes: 64 * 1024 * 1024 * 1024,
    };
    let c = auto_ctx(&v, &budget, v.context_max);
    assert!(c >= 4096, "never below the floor");
    assert!(c <= 131_072);
    assert!(c % 4096 == 0, "rounded to a 4k step");
}

#[test]
fn resolve_ctx_prefers_explicit_flag() {
    let v = llama31_8b_q4km();
    let budget = MemoryBudget {
        gpu_effective_bytes: 24 * 1024 * 1024 * 1024,
        ram_total_bytes: 32 * 1024 * 1024 * 1024,
    };
    assert_eq!(resolve_ctx(Some(16_384), &v, &budget, v.context_max), 16_384);
    // None falls through to auto_ctx (non-zero).
    assert!(resolve_ctx(None, &v, &budget, v.context_max) >= 4096);
}
```

If `llama31_8b_q4km()` does not already exist in the test module, add it next to the existing `llama31_70b_q4km()` fixture:

```rust
fn llama31_8b_q4km() -> ModelVariant {
    ModelVariant {
        model_name: "llama-3.1-8b".into(),
        quant: "Q4_K_M".into(),
        bpw: 4.83,
        params_total: 8_000_000_000,
        params_active: 8_000_000_000,
        layers: 32,
        kv_heads: 8,
        head_dim: 128,
        embedding_dim: 4096,
        context_max: 131_072,
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test -p paddock-core auto_ctx`
Expected: FAIL — `auto_ctx` / `resolve_ctx` not found.

- [ ] **Step 3: Implement `auto_ctx` and `resolve_ctx`**

Add after `kv_cache_bytes` (around line 129):

```rust
/// Step size for auto-sized context — round to a 4k boundary.
const CTX_STEP: u32 = 4096;
/// Never auto-size below this; a variant that cannot hold 4k would not have
/// been selected by `best_variant`.
const CTX_FLOOR: u32 = 4096;

/// Largest context whose total memory (weights + KV cache + overhead) still
/// fits the GPU budget, rounded down to a 4k step and clamped to
/// `[CTX_FLOOR, context_max]`. Falls back to the floor when even 4k overflows.
pub fn auto_ctx(v: &ModelVariant, budget: &MemoryBudget, context_max: u32) -> u32 {
    let ceil = context_max.max(CTX_FLOOR);
    // Walk down from the rounded ceiling in 4k steps.
    let start = (ceil / CTX_STEP) * CTX_STEP;
    let mut ctx = start.max(CTX_FLOOR);
    loop {
        if estimate_memory(v, ctx, budget).total_bytes <= budget.gpu_effective_bytes {
            return ctx.min(context_max).max(CTX_FLOOR);
        }
        if ctx <= CTX_FLOOR {
            return CTX_FLOOR;
        }
        ctx = ctx.saturating_sub(CTX_STEP).max(CTX_FLOOR);
    }
}

/// Resolve the context to launch with: explicit `--ctx` wins, else auto-size.
pub fn resolve_ctx(
    explicit: Option<u32>,
    v: &ModelVariant,
    budget: &MemoryBudget,
    context_max: u32,
) -> u32 {
    explicit.unwrap_or_else(|| auto_ctx(v, budget, context_max))
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cargo test -p paddock-core auto_ctx resolve_ctx`
Expected: PASS (4 tests). If `auto_ctx_clamps_down...` returns 4096 because the 70B weights alone exceed 40 GiB, that is correct (floor reached) — the assertions still hold.

- [ ] **Step 5: Commit**

```bash
git add crates/paddock-core/src/estimate.rs
git commit -m "feat(core): auto_ctx/resolve_ctx context auto-sizing"
```

---

## Task 3: Wire ctx resolution into the CLI serve/run paths

The `ctx: Option<u32>` parameter is already threaded into `plan_run`/`plan_serve` (they fall back to `DEFAULT_CONTEXT` on `None`). This task makes the **CLI** resolve `None` to `auto_ctx` before calling them, so detached records and the launched server use the auto-sized value. The TUI keeps passing `None` (8k default) — auto-sizing in the TUI is out of scope.

**Files:**
- Modify: `crates/paddock/src/main.rs:154-190` (`run_model`, `serve_model`)

- [ ] **Step 1: Add a ctx-resolution helper in `main.rs`**

Above `run_model` (line 154), add:

```rust
/// Resolve the launch context for a chosen model variant: explicit `--ctx`
/// wins, otherwise auto-size against this machine's GPU budget.
fn resolved_ctx(app: &App, model: &CatalogModel, idx: usize, ctx: Option<u32>) -> u32 {
    let mv = model.to_model_variant(&model.variants[idx]);
    paddock_core::estimate::resolve_ctx(ctx, &mv, &app.budget, model.context_max)
}
```

- [ ] **Step 2: Use it in `run_model`**

Replace the `plan_run` call (line 159):

```rust
    let ctx = Some(resolved_ctx(app, &model, idx, ctx));
    let plan: RunPlan = plan_run(&model, &model.variants[idx], &app.profile.runtimes, ctx)?;
```

- [ ] **Step 3: Use it in `serve_model`**

Replace the `plan_serve` call (line 181 area):

```rust
    let ctx = Some(resolved_ctx(app, &model, idx, ctx));
    let plan = plan_serve(&model, &model.variants[idx], &app.profile.runtimes, port, ctx)?;
```

- [ ] **Step 4: Verify build + existing tests**

Run: `cargo build && cargo test -p paddock-core runtime`
Expected: builds; runtime tests still PASS (they call `plan_*` directly with explicit values, unaffected).

- [ ] **Step 5: Commit**

```bash
git add crates/paddock/src/main.rs
git commit -m "feat(cli): auto-size context on serve/run when --ctx omitted"
```

---

## Task 4: Detached serve by default, `--foreground` to attach

**Files:**
- Modify: `crates/paddock/src/cli.rs:43-49` (`Serve` variant)
- Modify: `crates/paddock/src/main.rs` (`serve_model`, `serve_with_plan`, new detached path)

- [ ] **Step 1: Add the `--foreground` flag**

In `cli.rs`, the `Serve` variant becomes:

```rust
    /// Serve a model over HTTP and print the endpoint (OpenAI-compatible)
    Serve {
        model: String,
        /// Port for llama.cpp / mlx servers (Ollama always uses 11434)
        #[arg(long)]
        port: Option<u16>,
        /// Context window in tokens (llama.cpp only; default auto-sizes to memory)
        #[arg(long)]
        ctx: Option<u32>,
        /// Stay attached and stream logs (Ctrl-C stops); default runs detached
        #[arg(long, short = 'f')]
        foreground: bool,
    },
```

Update the dispatch arm in `main.rs` (line ~52):

```rust
        Some(Command::Serve { model, port, ctx, foreground }) => {
            serve_model(&app, &model, port, ctx, foreground, cli.json)?
        }
```

- [ ] **Step 2: Thread `foreground` through `serve_model`**

```rust
fn serve_model(
    app: &App,
    query: &str,
    port: Option<u16>,
    ctx: Option<u32>,
    foreground: bool,
    json: bool,
) -> Result<()> {
    let (model, idx) = resolve_model(app, query)?;
    let ctx = Some(resolved_ctx(app, &model, idx, ctx));
    let plan = plan_serve(&model, &model.variants[idx], &app.profile.runtimes, port, ctx)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }

    serve_with_plan(plan, foreground)
}
```

- [ ] **Step 3: Add the detached-spawn primitive**

Add near `spawn_checked` in `main.rs`. This redirects output to a log file and starts a new session so the child survives the terminal closing:

```rust
use std::os::unix::process::CommandExt;

// libc-free, matching serving.rs style: detach into a new session.
unsafe extern "C" {
    #[link_name = "setsid"]
    fn libc_setsid() -> i32;
}

/// Spawn a server child detached from the controlling terminal, with stdout +
/// stderr captured to `log_path`. Returns the child handle (its PID is the
/// session leader). Dropping the handle does NOT kill the process.
fn spawn_detached(argv: &[String], log_path: &std::path::Path) -> Result<std::process::Child> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("cannot create log dir {parent:?}: {e}"))?;
    }
    let log = std::fs::File::create(log_path)
        .map_err(|e| anyhow::anyhow!("cannot create log file {log_path:?}: {e}"))?;
    let log_err = log.try_clone().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut cmd = std::process::Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .stdin(std::process::Stdio::null())
        .stdout(log)
        .stderr(log_err);
    // SAFETY: setsid only creates a new session; async-signal-safe, no allocation.
    unsafe {
        cmd.pre_exec(|| {
            libc_setsid();
            Ok(())
        });
    }
    cmd.spawn()
        .map_err(|e| anyhow::anyhow!("failed to start {}: {e}. Is it in PATH?", argv[0]))
}
```

- [ ] **Step 4: Branch `serve_with_plan` on `foreground`**

Replace `serve_with_plan` (lines 195-257). The shared prefix (install confirm) stays; the spawn and tail differ:

```rust
pub(crate) fn serve_with_plan(plan: ServePlan, foreground: bool) -> Result<()> {
    if plan.port_ignored {
        eprintln!("warning: --port is ignored for the Ollama daemon (fixed 11434)");
    }
    if let Some(install) = &plan.install {
        confirm_and_install(install)?;
    }

    // Pre-compute the detached log path from the eventual PID? PID is only known
    // after spawn. So: spawn first (detached writes to a temp name, then rename),
    // or name the log by PID after spawn. We name by PID after spawn by spawning
    // to a path we know post-hoc — simplest: spawn, then we already have the PID.
    let log_dir = paddock_core::serving::default_serving_dir().join("logs");

    let mut child = match &plan.server_argv {
        Some(argv) => {
            eprintln!("$ {}", argv.join(" "));
            if foreground {
                Some(spawn_checked(argv)?)
            } else {
                // Detached: spawn to a per-invocation placeholder (our own pid
                // makes it unique across concurrent `serve`s), then rename to
                // <child-pid>.log once the child pid is known.
                let tmp_log = log_dir.join(format!("pending-{}.log", std::process::id()));
                let c = spawn_detached(argv, &tmp_log)?;
                let final_log = log_dir.join(format!("{}.log", c.id()));
                let _ = std::fs::rename(&tmp_log, &final_log);
                Some(c)
            }
        }
        None => None,
    };

    let prepared = wait_ready(&plan, child.as_mut()).and_then(|()| {
        for step in &plan.pre_steps {
            eprintln!("$ {}", step.join(" "));
            run_checked(step)?;
        }
        Ok(())
    });
    if let Err(e) = prepared {
        if let Some(c) = child.as_mut() {
            let _ = c.kill();
            let _ = c.wait();
        }
        return Err(e);
    }

    if plan.runtime == paddock_core::catalog::RuntimeKind::Ollama {
        eprintln!("loading {} into memory…", plan.model_ref);
        if !paddock_core::serving::warm_up_ollama(&RealSystemProbe, &plan.model_ref) {
            eprintln!("warning: warm-up failed — the model will load on the first request");
        }
    }

    output::print_endpoint(&plan);

    match child {
        Some(mut c) if foreground => {
            let _guard = RegistryGuard::register(&plan, c.id(), plan.ctx, None);
            eprintln!("serving — press Ctrl-C to stop");
            let status = c.wait()?;
            if !status.success() {
                bail!("server exited with {status}");
            }
            Ok(())
        }
        // Detached child: register WITHOUT the drop-guard so it outlives us.
        Some(c) => {
            let log_path = log_dir.join(format!("{}.log", c.id()));
            register_detached(&plan, c.id(), Some(log_path));
            eprintln!(
                "serving in background · pid {} · paddock logs {}",
                c.id(),
                plan.model_ref
            );
            Ok(())
        }
        // Ollama daemon path: nothing to detach or track — the daemon owns the
        // model and `ollama ps` lists it. Matches today's behavior. paddock's
        // ps/stop/logs cover the spawned (llama.cpp/mlx) servers only.
        None => Ok(()),
    }
}

/// Register a detached child server. Unlike `RegistryGuard`, this does NOT
/// unregister on drop — the server must survive this process exiting.
fn register_detached(plan: &ServePlan, pid: u32, log_path: Option<std::path::PathBuf>) {
    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let record = ServingRecord {
        pid,
        runtime: plan.runtime,
        endpoint: plan.endpoint.clone(),
        openai_url: plan.openai_url.clone(),
        model_ref: plan.model_ref.clone(),
        ready_path: plan.ready_path.clone(),
        started_at,
        ctx: plan.ctx,
        log_path,
        port: plan.port,
    };
    if let Err(e) = Registry::open_default().register(&record) {
        eprintln!("warning: could not record serving state: {e}");
    }
}
```

Note: this references `plan.ctx` and `plan.port`. Add those two fields to `ServePlan` in `crates/paddock-core/src/runtime.rs` (struct at line ~149) and populate them in all three `ServePlan` constructors (mlx, llama_server_plan, ollama arm) plus the test expectations. For ollama/mlx `port` = `Some(port)`; for the ollama daemon arm `port` = `None`. `ctx` is already a resolved local in `plan_serve` — store it on every `ServePlan` it builds and pass it into `llama_server_plan`.

- [ ] **Step 5: Add `ctx`/`port` to `ServePlan` and `RegistryGuard::register` signature**

In `runtime.rs`, add to `ServePlan`:

```rust
    /// Resolved context window (llama.cpp); 0 when not applicable.
    pub ctx: u32,
    /// Bound port for spawned servers; None for the Ollama daemon.
    pub port: Option<u16>,
```

Populate: mlx arm → `ctx, port: Some(port)`; ollama daemon arm → `ctx, port: None`; `llama_server_plan(hf_ref, port, ctx, local, install)` → set `ctx, port: Some(port)` (it already receives `ctx`). The `ctx` local already exists in `plan_serve` (`let ctx = ctx.unwrap_or(DEFAULT_CONTEXT);`).

Update `RegistryGuard::register` to accept ctx/log_path so the foreground path records them too:

```rust
    fn register(plan: &ServePlan, pid: u32, ctx: u32, log_path: Option<std::path::PathBuf>) -> Self {
        // ...existing body, but set:
        //   ctx, log_path, port: plan.port,
    }
```

(Foreground call passes `plan.ctx` instead of literal `0`: `RegistryGuard::register(&plan, c.id(), plan.ctx, None)`.)

- [ ] **Step 6: Update existing `ServePlan` tests for the new fields**

Existing `plan_serve` tests in `runtime.rs` assert on whole `ServePlan` shape only via `server_argv` (field access), so they keep compiling. Any test constructing a `ServePlan` literal must add `ctx` and `port`. Search and fix:

Run: `cargo build 2>&1 | grep -A2 "missing field"` and add `ctx: 8192, port: Some(8080)` (or `None` for daemon literals) where the compiler points.

- [ ] **Step 7: Build + tests**

Run: `cargo build && cargo test -p paddock-core && cargo test -p paddock`
Expected: PASS. Detached spawn is not unit-tested (process side effects); it is covered by manual verification in Step 8.

- [ ] **Step 8: Manual smoke test**

```bash
# requires a real GGUF + llama.cpp; otherwise skip and rely on `serve --json`
cargo run -- serve --json some-hf-gguf-model   # confirm plan carries ctx + port
```

- [ ] **Step 9: Commit**

```bash
git add crates/paddock/src/cli.rs crates/paddock/src/main.rs crates/paddock-core/src/runtime.rs
git commit -m "feat(cli): detached serve by default with --foreground opt-in"
```

---

## Task 5: `paddock ps`

**Files:**
- Modify: `crates/paddock/src/cli.rs` (add `Ps` variant)
- Modify: `crates/paddock/src/main.rs` (dispatch + handler)
- Modify: `crates/paddock/src/output.rs` (table + json)
- Test: `crates/paddock/tests/cli.rs`

- [ ] **Step 1: Write the failing integration test**

Add to `crates/paddock/tests/cli.rs`:

```rust
#[test]
fn ps_empty_registry_reports_no_servers() {
    let (mut cmd, _dir) = paddock();
    let out = cmd.arg("ps").assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout);
    assert!(stdout.contains("no servers running"), "got: {stdout}");
}

#[test]
fn ps_json_empty_is_empty_array() {
    let (mut cmd, _dir) = paddock();
    let out = cmd.args(["ps", "--json"]).assert().success();
    let v: serde_json::Value =
        serde_json::from_slice(&out.get_output().stdout).expect("valid json");
    assert_eq!(v, serde_json::json!([]));
}
```

- [ ] **Step 2: Run, verify failure**

Run: `cargo test -p paddock ps_empty_registry_reports_no_servers`
Expected: FAIL — unrecognized subcommand `ps`.

- [ ] **Step 3: Add the `Ps` command**

In `cli.rs`, after `Serve`:

```rust
    /// List running paddock servers
    Ps,
```

`--json` is the existing global flag on `Cli`, so no per-command flag is needed.

- [ ] **Step 4: Add the output formatter**

In `output.rs`:

```rust
use paddock_core::serving::ServingRecord;

/// Render `ps` as a table; empty → a short notice.
pub fn print_ps_table(records: &[ServingRecord]) {
    if records.is_empty() {
        println!("no servers running");
        return;
    }
    println!("{:<28} {:<9} {:<26} {:>7} {:>8} {:>7}", "MODEL", "RUNTIME", "ENDPOINT", "CTX", "UPTIME", "PID");
    for r in records {
        println!(
            "{:<28} {:<9} {:<26} {:>7} {:>8} {:>7}",
            truncate(&r.model_ref, 28),
            runtime_label(r.runtime),
            truncate(&r.endpoint, 26),
            r.ctx,
            uptime_label(r.started_at),
            r.pid,
        );
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { s.to_string() } else { format!("{}…", &s[..n - 1]) }
}

fn runtime_label(rt: paddock_core::catalog::RuntimeKind) -> &'static str {
    use paddock_core::catalog::RuntimeKind::*;
    match rt { Ollama => "ollama", LlamaCpp => "llama.cpp", MlxLm => "mlx-lm" }
}

/// Whole-seconds uptime as `12m`, `3h`, `2d` from a unix `started_at`.
fn uptime_label(started_at: i64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let secs = (now - started_at).max(0);
    if secs < 60 { format!("{secs}s") }
    else if secs < 3600 { format!("{}m", secs / 60) }
    else if secs < 86400 { format!("{}h", secs / 3600) }
    else { format!("{}d", secs / 86400) }
}
```

- [ ] **Step 5: Add the handler + dispatch**

In `main.rs` dispatch:

```rust
        Some(Command::Ps) => {
            let records = paddock_core::serving::Registry::open_default().list_live(&RealSystemProbe);
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&records)?);
            } else {
                output::print_ps_table(&records);
            }
        }
```

- [ ] **Step 6: Run tests, verify pass**

Run: `cargo test -p paddock ps_`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add crates/paddock/src/cli.rs crates/paddock/src/main.rs crates/paddock/src/output.rs crates/paddock/tests/cli.rs
git commit -m "feat(cli): paddock ps lists running servers"
```

---

## Task 6: `paddock stop <model|pid|all>` with confirmation

**Files:**
- Modify: `crates/paddock-core/src/serving.rs` (add `match_records`)
- Modify: `crates/paddock/src/cli.rs` (add `Stop`)
- Modify: `crates/paddock/src/main.rs` (handler)
- Test: `crates/paddock-core/src/serving.rs` (match logic), `crates/paddock/tests/cli.rs` (command)

- [ ] **Step 1: Write the failing unit test for target matching**

In `serving.rs` test module:

```rust
#[cfg(test)]
mod match_tests {
    use super::*;

    fn rec(pid: u32, model: &str) -> ServingRecord {
        ServingRecord {
            pid, runtime: RuntimeKind::LlamaCpp,
            endpoint: "e".into(), openai_url: "o".into(),
            model_ref: model.into(), ready_path: "/health".into(),
            started_at: 0, ctx: 8192, log_path: None, port: Some(8080),
        }
    }

    #[test]
    fn match_all_returns_everything() {
        let rs = vec![rec(1, "a"), rec(2, "b")];
        assert_eq!(match_records(&rs, "all").matched().len(), 2);
    }

    #[test]
    fn match_by_pid() {
        let rs = vec![rec(10, "a"), rec(20, "b")];
        let m = match_records(&rs, "20");
        assert_eq!(m.matched().len(), 1);
        assert_eq!(m.matched()[0].pid, 20);
    }

    #[test]
    fn match_by_model_substring() {
        let rs = vec![rec(1, "qwen3-35b"), rec(2, "llama3-8b")];
        let m = match_records(&rs, "qwen");
        assert_eq!(m.matched().len(), 1);
        assert_eq!(m.matched()[0].pid, 1);
    }

    #[test]
    fn ambiguous_substring_lists_candidates() {
        let rs = vec![rec(1, "qwen3-35b"), rec(2, "qwen3-8b")];
        assert!(matches!(match_records(&rs, "qwen"), RecordMatch::Ambiguous(_)));
    }

    #[test]
    fn no_match_is_not_found() {
        let rs = vec![rec(1, "a")];
        assert!(matches!(match_records(&rs, "zzz"), RecordMatch::NotFound));
    }
}
```

- [ ] **Step 2: Run, verify failure**

Run: `cargo test -p paddock-core match_tests`
Expected: FAIL — `match_records`/`RecordMatch` not found.

- [ ] **Step 3: Implement `match_records`**

In `serving.rs`:

```rust
/// Result of resolving a `stop`/`logs` target against live records.
pub enum RecordMatch<'a> {
    /// One or more records to act on (also the `all` case).
    Matched(Vec<&'a ServingRecord>),
    /// A name substring hit several models — caller lists and aborts.
    Ambiguous(Vec<&'a ServingRecord>),
    NotFound,
}

impl<'a> RecordMatch<'a> {
    pub fn matched(&self) -> &[&'a ServingRecord] {
        match self {
            RecordMatch::Matched(v) => v,
            _ => &[],
        }
    }
}

/// Resolve a target: `all` → every record; all-digits → exact PID; otherwise a
/// case-insensitive substring of `model_ref` (Ambiguous if >1 distinct model).
pub fn match_records<'a>(records: &'a [ServingRecord], target: &str) -> RecordMatch<'a> {
    if target == "all" {
        return if records.is_empty() {
            RecordMatch::NotFound
        } else {
            RecordMatch::Matched(records.iter().collect())
        };
    }
    if let Ok(pid) = target.parse::<u32>() {
        return match records.iter().find(|r| r.pid == pid) {
            Some(r) => RecordMatch::Matched(vec![r]),
            None => RecordMatch::NotFound,
        };
    }
    let needle = target.to_lowercase();
    let hits: Vec<&ServingRecord> = records
        .iter()
        .filter(|r| r.model_ref.to_lowercase().contains(&needle))
        .collect();
    match hits.len() {
        0 => RecordMatch::NotFound,
        1 => RecordMatch::Matched(hits),
        _ => RecordMatch::Ambiguous(hits),
    }
}

/// Send SIGTERM to a pid. Best-effort; a dead pid is a no-op.
pub fn terminate(pid: u32) {
    if pid == 0 || pid > i32::MAX as u32 {
        return;
    }
    // SAFETY: kill with SIGTERM (15) signals an existing process.
    unsafe { libc_kill(pid as i32, 15) };
}
```

- [ ] **Step 4: Run, verify pass**

Run: `cargo test -p paddock-core match_tests`
Expected: PASS (5 tests).

- [ ] **Step 5: Add the `Stop` command + handler**

`cli.rs`:

```rust
    /// Stop a running server (by model name, pid, or `all`)
    Stop {
        /// Target: model name substring, a pid, or `all`
        target: String,
        /// Skip the confirmation prompt for `all`
        #[arg(long, short = 'y')]
        yes: bool,
    },
```

`main.rs` dispatch:

```rust
        Some(Command::Stop { target, yes }) => stop_servers(&target, yes)?,
```

`main.rs` handler:

```rust
fn stop_servers(target: &str, yes: bool) -> Result<()> {
    use paddock_core::serving::{Registry, RecordMatch, match_records, terminate};
    use paddock_core::catalog::RuntimeKind;

    let registry = Registry::open_default();
    let records = registry.list_live(&RealSystemProbe);
    let chosen = match match_records(&records, target) {
        RecordMatch::Matched(v) => v,
        RecordMatch::Ambiguous(cands) => {
            eprintln!("`{target}` matches several servers — be specific:");
            for r in cands {
                eprintln!("  {} (pid {})", r.model_ref, r.pid);
            }
            std::process::exit(1);
        }
        RecordMatch::NotFound => {
            eprintln!("no running server matches `{target}`");
            if !records.is_empty() {
                eprintln!("running: {}", records.iter().map(|r| r.model_ref.as_str()).collect::<Vec<_>>().join(", "));
            }
            std::process::exit(1);
        }
    };

    if target == "all" && !yes {
        eprintln!("about to stop {} server(s):", chosen.len());
        for r in &chosen {
            eprintln!("  {} (pid {})", r.model_ref, r.pid);
        }
        eprint!("proceed? [y/N] ");
        use std::io::Write;
        std::io::stderr().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        if !matches!(line.trim(), "y" | "Y") {
            eprintln!("aborted");
            return Ok(());
        }
    }

    for r in chosen {
        if r.runtime == RuntimeKind::Ollama {
            let _ = run_checked(&["ollama".into(), "stop".into(), r.model_ref.clone()]);
        } else {
            terminate(r.pid);
        }
        let _ = registry.unregister(r.pid);
        println!("stopped {} (pid {})", r.model_ref, r.pid);
    }
    Ok(())
}
```

- [ ] **Step 6: Write the failing command test**

`tests/cli.rs`:

```rust
#[test]
fn stop_unknown_target_errors() {
    let (mut cmd, _dir) = paddock();
    cmd.args(["stop", "nope"]).assert().failure();
}
```

- [ ] **Step 7: Run all the Task-6 tests**

Run: `cargo test -p paddock-core match_tests && cargo test -p paddock stop_`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/paddock-core/src/serving.rs crates/paddock/src/cli.rs crates/paddock/src/main.rs crates/paddock/tests/cli.rs
git commit -m "feat(cli): paddock stop with all-confirmation and per-runtime semantics"
```

---

## Task 7: `paddock logs <model|pid> [-f]`

**Files:**
- Modify: `crates/paddock/src/cli.rs` (add `Logs`)
- Modify: `crates/paddock/src/main.rs` (handler)
- Test: `crates/paddock/tests/cli.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn logs_unknown_target_errors() {
    let (mut cmd, _dir) = paddock();
    cmd.args(["logs", "nope"]).assert().failure();
}
```

- [ ] **Step 2: Run, verify failure**

Run: `cargo test -p paddock logs_unknown_target_errors`
Expected: FAIL — unrecognized subcommand `logs`.

- [ ] **Step 3: Add the `Logs` command**

`cli.rs`:

```rust
    /// Show a detached server's log (by model name or pid)
    Logs {
        /// Target: model name substring or a pid
        target: String,
        /// Follow the log (like `tail -f`)
        #[arg(long, short = 'f')]
        follow: bool,
    },
```

`main.rs` dispatch:

```rust
        Some(Command::Logs { target, follow }) => show_logs(&target, follow)?,
```

- [ ] **Step 4: Implement the handler**

```rust
fn show_logs(target: &str, follow: bool) -> Result<()> {
    use paddock_core::serving::{Registry, RecordMatch, match_records};

    let records = Registry::open_default().list_live(&RealSystemProbe);
    let chosen = match match_records(&records, target) {
        RecordMatch::Matched(v) if v.len() == 1 => v[0].clone(),
        RecordMatch::Matched(_) | RecordMatch::Ambiguous(_) => {
            eprintln!("`{target}` matches several servers — use a pid");
            std::process::exit(1);
        }
        RecordMatch::NotFound => {
            eprintln!("no running server matches `{target}`");
            std::process::exit(1);
        }
    };

    let Some(path) = chosen.log_path.clone() else {
        eprintln!(
            "{} runs under {:?} which keeps its own logs (no paddock log file)",
            chosen.model_ref, chosen.runtime
        );
        return Ok(());
    };

    if follow {
        // Delegate to `tail -f` for follow semantics.
        run_checked(&["tail".into(), "-f".into(), path.to_string_lossy().into_owned()])
    } else {
        let body = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("cannot read log {path:?}: {e}"))?;
        print!("{body}");
        Ok(())
    }
}
```

Note: `match_records` returns borrows; clone the chosen record (`v[0].clone()`) so it outlives the `records` borrow. `ServingRecord` is `Clone`.

- [ ] **Step 5: Run, verify pass**

Run: `cargo test -p paddock logs_unknown_target_errors`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/paddock/src/cli.rs crates/paddock/src/main.rs crates/paddock/tests/cli.rs
git commit -m "feat(cli): paddock logs tails a detached server's log"
```

---

## Task 8: Documentation

**Files:**
- Modify: `README.md` (serve section, lines ~115-137)

- [ ] **Step 1: Update the serve section**

Rewrite the lifecycle paragraph and flag list to describe detached-by-default, `--foreground`, ctx auto-sizing, and the new `ps`/`stop`/`logs` commands. Add a short example block:

```text
$ paddock serve qwen3-8b
…
endpoint    http://127.0.0.1:8080
serving in background · pid 51234 · paddock logs qwen3-8b

$ paddock ps
MODEL          RUNTIME    ENDPOINT                CTX   UPTIME   PID
qwen3-8b       llama.cpp  http://127.0.0.1:8080  32768     2m    51234

$ paddock stop qwen3-8b
stopped qwen3-8b (pid 51234)
```

Document: `serve` runs detached by default (no terminal stays open); `--foreground`/`-f` attaches and streams logs with Ctrl-C to stop; `--ctx` overrides the default auto-sizing; `paddock logs <model> -f` follows a detached server's log; `paddock stop all` asks for confirmation (`-y` to skip).

- [ ] **Step 2: Verify the doc matches behavior**

Run: `cargo run -- serve --help && cargo run -- ps --help && cargo run -- stop --help && cargo run -- logs --help`
Expected: help text matches the README claims.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document detached serve and ps/stop/logs lifecycle"
```

---

## Final verification

- [ ] `cargo build` — clean
- [ ] `cargo test` — full workspace green
- [ ] `cargo clippy` — no new warnings
- [ ] Manual: `serve` (detached) → `ps` shows it → `logs` reads it → `stop` removes it (requires a real GGUF + llama.cpp).
