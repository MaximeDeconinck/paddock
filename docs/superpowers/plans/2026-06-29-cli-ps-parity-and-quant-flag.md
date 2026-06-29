# CLI `ps` parity + `--quant` flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `paddock ps` show the same running + AVAILABLE picture as the TUI servers tab, and let `paddock run`/`serve` launch a chosen quantization with `--quant`.

**Architecture:** Reuse the existing core `list_all_servers` / `list_available` (already powering the TUI) in the CLI `Ps` handler; add `Serialize` to the row types for JSON mode and a new `output::print_servers` for text mode. For `--quant`, add a pure `resolve_quant` helper that maps a label to a variant index via the existing `variants_by_quality` ordering, and thread an `Option<String>` quant override through `resolve_model`.

**Tech Stack:** Rust 2024, clap, serde/serde_json, anyhow.

**Spec:** `docs/superpowers/specs/2026-06-29-cli-ps-parity-and-quant-flag-design.md`

---

## File Structure

- `crates/paddock-core/src/serving.rs` - add `#[derive(Serialize)]` to `StopHandle`, `ServerRow`, `AvailableRow` (logic unchanged; `list_all_servers`/`list_available`/`History::open_default` already exist). `ServePlan` (runtime.rs) already derives `Serialize`.
- `crates/paddock/src/output.rs` - replace `print_ps_table` with `print_servers` + a testable `servers_view(running, available, now) -> String`.
- `crates/paddock/src/main.rs` - rewrite the `Command::Ps` arm; add `resolve_quant`; thread `quant` through `resolve_model` / `run_model` / `serve_model`.
- `crates/paddock/src/cli.rs` - add `--quant` to `Run` and `Serve`.

---

## Task 1: Serialize derives on the row types (core)

**Files:**
- Modify: `crates/paddock-core/src/serving.rs` (the `StopHandle` enum ~line 310, `ServerRow` struct ~line 322, `AvailableRow` struct ~line 388)
- Test: `crates/paddock-core/src/serving.rs` (inline `#[cfg(test)]` module)

`serde::{Serialize, Deserialize}` are already imported and used in this file (e.g. `ServingRecord`). `ServePlan` (embedded in `AvailableRow`) already derives `Serialize`.

- [ ] **Step 1: Write the failing test**

Add to the existing `#[cfg(test)] mod tests` in `serving.rs`:

```rust
#[test]
fn server_and_available_rows_serialize() {
    let row = ServerRow {
        model: "qwen3:8b".into(),
        runtime: RuntimeKind::Ollama,
        endpoint: "http://127.0.0.1:11434".into(),
        openai_url: "http://127.0.0.1:11434/v1/chat/completions".into(),
        ctx: None,
        started_at: None,
        stop: StopHandle::OllamaModel("qwen3:8b".into()),
    };
    let j = serde_json::to_string(&row).unwrap();
    assert!(j.contains("qwen3:8b"));

    let avail = AvailableRow {
        model: "llama3".into(),
        runtime: RuntimeKind::Ollama,
        size_bytes: Some(4_000_000_000),
        last_served_at: None,
        plan: ollama_serve_plan("llama3"),
    };
    let j = serde_json::to_string(&avail).unwrap();
    assert!(j.contains("llama3"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p paddock-core server_and_available_rows_serialize`
Expected: compile error - `the trait bound 'ServerRow: Serialize' is not satisfied` (and same for `AvailableRow`, `StopHandle`).

- [ ] **Step 3: Add the derives**

On `StopHandle` (currently `#[derive(Debug, Clone, PartialEq, Eq)]`):

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum StopHandle {
```

On `ServerRow` (currently `#[derive(Debug, Clone)]`):

```rust
#[derive(Debug, Clone, Serialize)]
pub struct ServerRow {
```

On `AvailableRow` (currently `#[derive(Debug, Clone)]`):

```rust
#[derive(Debug, Clone, Serialize)]
pub struct AvailableRow {
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p paddock-core server_and_available_rows_serialize`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/paddock-core/src/serving.rs
git commit -m "feat(core): derive Serialize on ServerRow/AvailableRow/StopHandle"
```

---

## Task 2: `print_servers` text view (output.rs)

**Files:**
- Modify: `crates/paddock/src/output.rs` (replace `print_ps_table` ~line 236; add imports at top)
- Test: `crates/paddock/src/output.rs` (`#[cfg(test)] mod tests`)

`gib`, `age_label(released_at: Option<i64>, approx: bool, now: i64)`, `uptime_label`, and the private `humanize_since(secs: i64) -> String` already live in this module. `runtime_label(RuntimeKind) -> &'static str` too.

- [ ] **Step 1: Add imports**

At the top of `output.rs`, extend the `serving` import:

```rust
use paddock_core::serving::{AvailableRow, ServerRow, ServingRecord, StopHandle};
```

(`ServingRecord` stays - other functions still use it.)

- [ ] **Step 2: Write the failing test**

Add to `#[cfg(test)] mod tests` in `output.rs`:

```rust
use paddock_core::catalog::RuntimeKind;
use paddock_core::serving::{AvailableRow, ServerRow, StopHandle};
use paddock_core::runtime::ServePlan;

fn ollama_plan(name: &str) -> ServePlan {
    ServePlan {
        server_argv: None,
        pre_steps: vec![],
        endpoint: "http://127.0.0.1:11434".into(),
        openai_url: "http://127.0.0.1:11434/v1/chat/completions".into(),
        model_ref: name.into(),
        ready_path: "/api/version".into(),
        install: None,
        port_ignored: false,
        runtime: RuntimeKind::Ollama,
        ctx: 0,
        port: None,
    }
}

#[test]
fn servers_view_empty_says_none() {
    assert_eq!(servers_view(&[], &[], 1_780_000_000), "no servers running");
}

#[test]
fn servers_view_has_sections_and_dashes() {
    let running = vec![ServerRow {
        model: "qwen3:8b".into(),
        runtime: RuntimeKind::Ollama,
        endpoint: "http://127.0.0.1:11434".into(),
        openai_url: "http://127.0.0.1:11434/v1/chat/completions".into(),
        ctx: None,
        started_at: None,
        stop: StopHandle::OllamaModel("qwen3:8b".into()),
    }];
    let available = vec![AvailableRow {
        model: "llama3".into(),
        runtime: RuntimeKind::Ollama,
        size_bytes: Some(4 * 1024 * 1024 * 1024),
        last_served_at: None,
        plan: ollama_plan("llama3"),
    }];
    let out = servers_view(&running, &available, 1_780_000_000);
    assert!(out.contains("RUNNING"));
    assert!(out.contains("AVAILABLE"));
    assert!(out.contains("qwen3:8b"));
    assert!(out.contains("llama3"));
    // Ollama running row: no pid, no ctx, no uptime -> dashes.
    assert!(out.contains('-'));
    // Available row with unknown last-served renders "?".
    assert!(out.contains('?'));
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p paddock servers_view`
Expected: compile error - `cannot find function 'servers_view'`.

- [ ] **Step 4: Replace `print_ps_table` with `servers_view` + `print_servers`**

Delete the existing `print_ps_table` function (lines ~236-256) and replace with:

```rust
/// `paddock ps` text view: a RUNNING section then an AVAILABLE section.
/// Empty sections are skipped; both empty prints "no servers running".
/// `now` is unix seconds, injected for deterministic tests.
pub fn servers_view(running: &[ServerRow], available: &[AvailableRow], now: i64) -> String {
    if running.is_empty() && available.is_empty() {
        return "no servers running".to_string();
    }
    let mut s = String::new();
    if !running.is_empty() {
        s.push_str("RUNNING\n");
        s.push_str(&format!(
            "{:<28} {:<9} {:<26} {:>7} {:>8} {:>7}\n",
            "MODEL", "RUNTIME", "ENDPOINT", "CTX", "UPTIME", "PID"
        ));
        for r in running {
            let ctx = r.ctx.map(|c| c.to_string()).unwrap_or_else(|| "-".into());
            let uptime = r
                .started_at
                .map(|t| humanize_since(now - t))
                .unwrap_or_else(|| "-".into());
            let pid = match &r.stop {
                StopHandle::Pid(p) => p.to_string(),
                _ => "-".into(),
            };
            s.push_str(&format!(
                "{:<28} {:<9} {:<26} {:>7} {:>8} {:>7}\n",
                truncate(&r.model, 28),
                runtime_label(r.runtime),
                truncate(&r.endpoint, 26),
                ctx,
                uptime,
                pid,
            ));
        }
    }
    if !available.is_empty() {
        if !running.is_empty() {
            s.push('\n');
        }
        s.push_str("AVAILABLE\n");
        s.push_str(&format!(
            "{:<28} {:<9} {:>10} {:>12}\n",
            "MODEL", "RUNTIME", "SIZE", "LAST-SERVED"
        ));
        for r in available {
            let size = r.size_bytes.map(gib).unwrap_or_else(|| "-".into());
            s.push_str(&format!(
                "{:<28} {:<9} {:>10} {:>12}\n",
                truncate(&r.model, 28),
                runtime_label(r.runtime),
                size,
                age_label(r.last_served_at, false, now),
            ));
        }
    }
    s.trim_end().to_string()
}

/// Print the `paddock ps` view, computing `now` from the system clock.
pub fn print_servers(running: &[ServerRow], available: &[AvailableRow]) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    println!("{}", servers_view(running, available, now));
}
```

Note: `truncate` is the existing private helper in this module (used by the old `print_ps_table`). If the test module already imports `super::*`, the new `use` lines in the test only need the types not already in scope - drop any duplicate-import warnings by keeping a single `use paddock_core::serving::{AvailableRow, ServerRow, StopHandle};` inside the test module.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p paddock servers_view`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add crates/paddock/src/output.rs
git commit -m "feat(cli): print_servers view with RUNNING + AVAILABLE sections"
```

---

## Task 3: Wire the `Ps` handler to running + available

**Files:**
- Modify: `crates/paddock/src/main.rs` (the `Command::Ps` arm ~lines 59-66; imports at top)

- [ ] **Step 1: Update imports**

The `Registry` import stays. `ServingRecord` may become unused after this task - if `cargo build` warns it's unused, drop it from the `use paddock_core::serving::{...}` line. Leave `RealSystemProbe` (already imported).

- [ ] **Step 2: Replace the `Command::Ps` arm**

Replace lines ~59-66 with:

```rust
        Some(Command::Ps) => {
            let registry = Registry::open_default();
            let probe = RealSystemProbe;
            let running = paddock_core::serving::list_all_servers(&registry, &probe);
            let history = paddock_core::serving::History::open_default();
            let available = paddock_core::serving::list_available(&history, &probe, &running);
            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "running": &running,
                        "available": &available,
                    }))?
                );
            } else {
                output::print_servers(&running, &available);
            }
        }
```

- [ ] **Step 3: Build and verify it compiles**

Run: `cargo build -p paddock`
Expected: builds. Fix any unused-import warning for `ServingRecord` by removing it from the import list.

- [ ] **Step 4: Smoke-test text + JSON**

Run:
```bash
cargo run -q -p paddock -- ps
cargo run -q -p paddock -- ps --json
```
Expected: text shows a RUNNING and/or AVAILABLE section (or `no servers running`); JSON is an object with `running` and `available` arrays. (With Ollama models installed, AVAILABLE is non-empty.)

- [ ] **Step 5: Run the full test suite**

Run: `cargo test`
Expected: PASS (no regressions; the old `print_ps_table` tests, if any, were replaced in Task 2).

- [ ] **Step 6: Commit**

```bash
git add crates/paddock/src/main.rs
git commit -m "feat(cli): paddock ps shows running + AVAILABLE like the TUI"
```

---

## Task 4: `resolve_quant` helper

**Files:**
- Modify: `crates/paddock/src/main.rs` (add the helper near `resolve_model` ~line 130; add `ModelVariant` import)
- Test: `crates/paddock/src/main.rs` (`#[cfg(test)] mod tests` - create one if absent)

`paddock_core::score::variants_by_quality(variants: &[ModelVariant]) -> Vec<usize>` exists. `ModelVariant` (with a `pub quant: String`) lives in `paddock_core::estimate`.

- [ ] **Step 1: Add the `ModelVariant` import**

Extend the estimate-related imports at the top of `main.rs` (add a line):

```rust
use paddock_core::estimate::ModelVariant;
```

- [ ] **Step 2: Write the failing test**

Add to `main.rs` a test module (append at the end of the file if none exists):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use paddock_core::catalog::RuntimeKind;

    fn mv(quant: &str, bpw: f32) -> ModelVariant {
        ModelVariant {
            quant: quant.into(),
            bytes_per_weight: bpw,
            ..ModelVariant::test_default()
        }
    }

    #[test]
    fn resolve_quant_exact_and_case_insensitive() {
        let vs = vec![mv("Q8_0", 8.5), mv("Q4_K_M", 4.8), mv("Q2_K", 3.0)];
        assert_eq!(resolve_quant(&vs, "Q4_K_M").unwrap(), 1);
        assert_eq!(resolve_quant(&vs, "q4_k_m").unwrap(), 1);
    }

    #[test]
    fn resolve_quant_collision_picks_best_quality() {
        // Two variants share the label; the higher-bpw one wins (it sorts
        // first in variants_by_quality).
        let vs = vec![mv("Q4_K_M", 4.5), mv("Q4_K_M", 4.9)];
        assert_eq!(resolve_quant(&vs, "Q4_K_M").unwrap(), 1);
    }

    #[test]
    fn resolve_quant_no_match_lists_available() {
        let vs = vec![mv("Q8_0", 8.5), mv("Q4_K_M", 4.8)];
        let err = resolve_quant(&vs, "Q3_K_M").unwrap_err().to_string();
        assert!(err.contains("Q8_0"));
        assert!(err.contains("Q4_K_M"));
    }
}
```

NOTE for the implementer: `ModelVariant` likely has no `test_default()` and several
required fields. Before writing the test, open `crates/paddock-core/src/estimate.rs`,
read the `ModelVariant` struct (~line 33), and construct it with its real fields
(set whatever `params`, `file_bytes`, etc. it requires to any valid value - the
test only exercises `quant` and the field `variants_by_quality` sorts on). If a
`#[cfg(test)]` constructor helper would keep the test readable, add one in
`estimate.rs` and use it; otherwise build the struct literally. Do not invent
fields. Confirm the exact field name `variants_by_quality` sorts on (bpw / quant
label) by reading `score.rs::variants_by_quality`, and make the collision test's
two variants differ on that field so the assertion is deterministic.

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p paddock resolve_quant`
Expected: compile error - `cannot find function 'resolve_quant'`.

- [ ] **Step 4: Implement `resolve_quant`**

Add near `resolve_model` in `main.rs`:

```rust
/// Index into `variants` of the variant whose quant label equals `label`
/// (case-insensitive). On a label shared by several variants, returns the
/// best-quality one (first in `variants_by_quality` order). Errors listing the
/// available quants when nothing matches.
fn resolve_quant(variants: &[ModelVariant], label: &str) -> Result<usize> {
    let order = paddock_core::score::variants_by_quality(variants);
    if let Some(&idx) = order
        .iter()
        .find(|&&i| variants[i].quant.eq_ignore_ascii_case(label))
    {
        return Ok(idx);
    }
    let available: Vec<&str> = order.iter().map(|&i| variants[i].quant.as_str()).collect();
    bail!(
        "no quant `{label}` for this model; available: {}",
        available.join(", ")
    );
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p paddock resolve_quant`
Expected: PASS (all three tests).

- [ ] **Step 6: Commit**

```bash
git add crates/paddock/src/main.rs crates/paddock-core/src/estimate.rs
git commit -m "feat(cli): resolve_quant maps a label to a variant index"
```

---

## Task 5: Thread `--quant` through run/serve

**Files:**
- Modify: `crates/paddock/src/cli.rs` (the `Run` and `Serve` variants ~lines 41-60)
- Modify: `crates/paddock/src/main.rs` (`resolve_model` ~line 133, its two callers ~lines 52-58, `run_model` ~line 177, `serve_model` ~line 195)

- [ ] **Step 1: Add `--quant` to the CLI variants**

In `cli.rs`, inside `Run { .. }` add:

```rust
        /// Quantization label to launch (e.g. Q4_K_M); default auto-picks the best fit
        #[arg(long)]
        quant: Option<String>,
```

Inside `Serve { .. }` add the same field.

- [ ] **Step 2: Update `resolve_model` to take the override**

Change the signature and the variant-selection logic in `main.rs`:

```rust
fn resolve_model(app: &App, query: &str, quant: Option<&str>) -> Result<(CatalogModel, usize)> {
    let db = app.open_db()?;
    let models = db.list_models().context("reading catalog")?;
    let model = match find_model(&models, query) {
        Lookup::Found(m) => m.clone(),
        Lookup::Ambiguous(names) => {
            eprintln!("model name `{query}` is ambiguous - candidates:");
            for n in names {
                eprintln!("  {n}");
            }
            std::process::exit(1);
        }
        Lookup::NotFound => return Err(PaddockError::ModelNotFound(query.to_string()).into()),
    };

    let mvs: Vec<_> = model
        .variants
        .iter()
        .map(|v| model.to_model_variant(v))
        .collect();

    // Explicit --quant launches that variant even if it does not fit; the
    // verdict is informational (consistent with the TUI quant picker).
    if let Some(label) = quant {
        let idx = resolve_quant(&mvs, label)?;
        return Ok((model, idx));
    }

    let Some(best) = best_variant(&mvs, &app.budget) else {
        bail!(
            "no quantization of `{}` fits this machine ({} RAM); try a smaller model from `paddock fit`",
            model.name,
            output::gib(app.budget.ram_total_bytes)
        );
    };
    let best_idx = mvs
        .iter()
        .position(|v| std::ptr::eq(v, best))
        .expect("best_variant borrows from mvs");
    Ok((model, best_idx))
}
```

- [ ] **Step 3: Update `run_model` and `serve_model` signatures + the call to `resolve_model`**

`run_model`:

```rust
fn run_model(app: &App, query: &str, ctx: Option<u32>, quant: Option<String>, json: bool) -> Result<()> {
    let (model, idx) = resolve_model(app, query, quant.as_deref())?;
    // ... rest unchanged ...
```

`serve_model`:

```rust
fn serve_model(
    app: &App,
    query: &str,
    port: Option<u16>,
    ctx: Option<u32>,
    foreground: bool,
    quant: Option<String>,
    json: bool,
) -> Result<()> {
    let (model, idx) = resolve_model(app, query, quant.as_deref())?;
    // ... rest unchanged ...
```

- [ ] **Step 4: Update the match arms**

In the `match cli.command` block:

```rust
        Some(Command::Run { model, ctx, quant }) => run_model(&app, &model, ctx, quant, cli.json)?,
        Some(Command::Serve {
            model,
            port,
            ctx,
            foreground,
            quant,
        }) => serve_model(&app, &model, port, ctx, foreground, quant, cli.json)?,
```

- [ ] **Step 5: Build and run the full suite**

Run: `cargo build -p paddock && cargo test`
Expected: builds, all tests pass.

- [ ] **Step 6: Smoke-test `--quant`**

Run (pick a multi-quant catalog model, e.g. one shown by `paddock fit`):
```bash
cargo run -q -p paddock -- serve <model> --quant Q2_K --json
cargo run -q -p paddock -- serve <model> --quant bogus --json
```
Expected: the first prints a plan whose `model_ref` reflects the `Q2_K` quant (or launches it in non-JSON mode); the second exits non-zero with `no quant \`bogus\` for this model; available: ...`.

- [ ] **Step 7: Commit**

```bash
git add crates/paddock/src/cli.rs crates/paddock/src/main.rs
git commit -m "feat(cli): --quant flag on run/serve to launch a chosen quantization"
```

---

## Task 6: Docs

**Files:**
- Modify: `README.md` (the `ps` and `serve`/`run` usage sections)

- [ ] **Step 1: Update the README**

Find the section documenting `paddock ps` and `paddock serve`/`run`. Update the `ps` description to mention it now lists running servers (paddock + Ollama) and a locally-available group. Add `--quant <label>` to the `run`/`serve` usage with a one-line note: "launch a specific quantization (default auto-picks the best fit); any quant is selectable, even one that does not fit." Use `-` or ` · ` as separators - the em-dash `—` is banned project-wide.

- [ ] **Step 2: Verify no em-dash slipped in**

Run: `! grep -n "—" README.md`
Expected: no output (exit 1 from grep). If any line matches, replace the `—` with `-` or ` · `.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document paddock ps parity and the --quant flag"
```

---

## Self-Review notes

- **Spec coverage:** Task 1 (Serialize derives) + Task 3 (handler) + Task 2 (text view) cover Feature 1; Task 4 (`resolve_quant`) + Task 5 (wiring) cover Feature 2; Task 6 covers the docs. All spec decisions (Running+AVAILABLE, launch-even-if-unfit, error-listing-quants, collision picks best, case-insensitive) are implemented in Tasks 2/4/5.
- **`ServePlan` Serialize:** already derived (runtime.rs:149) - the spec's note to add it is moot; only `StopHandle`/`ServerRow`/`AvailableRow` need it (Task 1).
- **Determinism:** `servers_view` takes injected `now`; `print_servers` supplies the wall clock. Tests use `now` + `started_at: None` to stay deterministic.
- **`ModelVariant` construction in tests:** flagged inline in Task 4 - the implementer must read the real struct and build valid instances rather than assume fields.
