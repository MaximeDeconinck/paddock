# TUI Servers View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an in-TUI servers tab (list running servers, stop, copy endpoint) and make serving from the TUI run detached via suspend/resume, without leaving the TUI.

**Architecture:** A new `Tab { Models, Servers }` dimension on the pure `TuiState`, a background mpsc refresh task (mirroring `sync_task`) that re-probes `Registry::list_live` off the UI thread, and event-loop handling for the new IO-bearing actions (stop, copy, inline detached serve). Rendering gains a tab indicator and a servers table.

**Tech Stack:** Rust (edition 2024), ratatui/crossterm (TUI), `std::sync::mpsc` (background task), `arboard` (clipboard, already a dependency), `assert`-style inline unit tests on the pure state machine.

---

## File structure

- `crates/paddock/src/tui/state.rs` — `Tab` enum, tab/servers fields, key handling for the servers tab + Tab toggle, `set_servers`/`selected_server` helpers, new `Action` variants. (Modify)
- `crates/paddock/src/tui/servers_task.rs` — background periodic `list_live` refresh over mpsc. (Create)
- `crates/paddock/src/clipboard.rs` — shared `copy_to_clipboard` helper. (Create)
- `crates/paddock/src/tray/mod.rs` — call the shared clipboard helper. (Modify)
- `crates/paddock/src/main.rs` — declare `mod clipboard;`. (Modify)
- `crates/paddock/src/tui/draw.rs` — tab indicator, servers table, per-tab footer, empty state. (Modify)
- `crates/paddock/src/tui/mod.rs` — spawn the task, drain its channel, handle stop/copy/inline-serve, remove `Exit::Serve`. (Modify)

---

## Task 1: State — `Tab` dimension, servers fields, navigation, Tab toggle

**Files:**
- Modify: `crates/paddock/src/tui/state.rs`
- Test: `crates/paddock/src/tui/state.rs` (existing `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing tests**

Add to the test module in `state.rs`. First add a `ServingRecord` import and a fixture helper at the top of the test module (next to `fake_row`):

```rust
    use paddock_core::serving::ServingRecord;

    fn srv(pid: u32, model: &str, port: u16) -> ServingRecord {
        ServingRecord {
            pid,
            runtime: RuntimeKind::LlamaCpp,
            endpoint: format!("http://127.0.0.1:{port}"),
            openai_url: format!("http://127.0.0.1:{port}/v1/chat/completions"),
            model_ref: model.into(),
            ready_path: "/health".into(),
            started_at: 0,
            ctx: 8192,
            log_path: None,
            port: Some(port),
        }
    }
```

Then the tests:

```rust
    #[test]
    fn tab_toggles_between_models_and_servers() {
        let mut s = state();
        assert_eq!(s.tab, Tab::Models);
        s.handle_key(key(KeyCode::Tab));
        assert_eq!(s.tab, Tab::Servers);
        s.handle_key(key(KeyCode::Tab));
        assert_eq!(s.tab, Tab::Models);
    }

    #[test]
    fn servers_tab_navigation_is_clamped() {
        let mut s = state();
        s.set_servers(vec![srv(1, "a", 8080), srv(2, "b", 8081)]);
        s.handle_key(key(KeyCode::Tab)); // -> Servers
        assert_eq!(s.server_selected, 0);
        s.handle_key(key(KeyCode::Char('k'))); // clamped at top
        assert_eq!(s.server_selected, 0);
        s.handle_key(key(KeyCode::Char('j')));
        assert_eq!(s.server_selected, 1);
        s.handle_key(key(KeyCode::Char('j'))); // clamped at bottom
        assert_eq!(s.server_selected, 1);
    }

    #[test]
    fn set_servers_preserves_selection_by_pid() {
        let mut s = state();
        s.set_servers(vec![srv(10, "a", 8080), srv(20, "b", 8081)]);
        s.tab = Tab::Servers;
        s.server_selected = 1; // pid 20
        // a refresh drops pid 10; pid 20 is now at index 0
        s.set_servers(vec![srv(20, "b", 8081)]);
        assert_eq!(s.server_selected, 0);
        assert_eq!(s.selected_server().unwrap().pid, 20);
    }
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test -p paddock tab_toggles_between`
Expected: FAIL — `Tab` not found / no field `tab`.

- [ ] **Step 3: Add the `Tab` enum and state fields**

In `state.rs`, after the `Mode` enum, add:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Models,
    Servers,
}
```

Add an import near the top of `state.rs` (with the other `paddock_core` uses):

```rust
use paddock_core::serving::ServingRecord;
```

Add fields to `TuiState` (after `last_sync`):

```rust
    /// Active top-level tab. Search/Detail overlays only apply on `Models`.
    pub tab: Tab,
    /// Live servers shown on the Servers tab; refreshed by the background task.
    pub servers: Vec<ServingRecord>,
    /// Cursor within `servers`.
    pub server_selected: usize,
```

Initialize them in `TuiState::new` (after `last_sync: None,`):

```rust
            tab: Tab::Models,
            servers: Vec::new(),
            server_selected: 0,
```

- [ ] **Step 4: Add `set_servers` / `selected_server` helpers**

Add as methods on `TuiState` (near `set_rows_preserving`):

```rust
    /// Replace the servers snapshot, keeping the cursor on the same pid when it
    /// survives the refresh, otherwise clamping to the new length.
    pub fn set_servers(&mut self, servers: Vec<ServingRecord>) {
        let selected_pid = self.servers.get(self.server_selected).map(|r| r.pid);
        self.servers = servers;
        self.server_selected = selected_pid
            .and_then(|pid| self.servers.iter().position(|r| r.pid == pid))
            .unwrap_or(0)
            .min(self.servers.len().saturating_sub(1));
    }

    pub fn selected_server(&self) -> Option<&ServingRecord> {
        self.servers.get(self.server_selected)
    }
```

- [ ] **Step 5: Add Tab toggle + servers navigation to `handle_key`**

In `handle_key`, the `Mode::List => match key.code { ... }` arm becomes tab-aware. Replace the whole `Mode::List` arm with:

```rust
            Mode::List => {
                // Tab toggles the top-level view from either tab.
                if key.code == K::Tab {
                    self.tab = match self.tab {
                        Tab::Models => Tab::Servers,
                        Tab::Servers => Tab::Models,
                    };
                    return Action::None;
                }
                match self.tab {
                    Tab::Models => match key.code {
                        K::Char('q') => return Action::Quit,
                        K::Up | K::Char('k') => self.selected = self.selected.saturating_sub(1),
                        K::Down | K::Char('j') => {
                            self.selected =
                                (self.selected + 1).min(self.rows.len().saturating_sub(1));
                        }
                        K::Enter => {
                            if !self.rows.is_empty() {
                                self.detail_plan = self.plan_for_selected();
                                self.detail_serve_plan = self.serve_plan_for_selected();
                                self.mode = Mode::Detail;
                            }
                        }
                        K::Char('x') => return self.run_selected(),
                        K::Char('s') => return self.serve_selected(),
                        K::Char('/') => {
                            self.mode = Mode::Search {
                                query: String::new(),
                            }
                        }
                        K::Char('g') => return self.set_use_case(UseCase::General),
                        K::Char('c') => return self.set_use_case(UseCase::Coding),
                        K::Char('r') => return self.set_use_case(UseCase::Reasoning),
                        K::Char('h') => return self.set_use_case(UseCase::Chat),
                        K::Char('R') => return Action::StartSync,
                        _ => {}
                    },
                    Tab::Servers => match key.code {
                        K::Char('q') => return Action::Quit,
                        K::Up | K::Char('k') => {
                            self.server_selected = self.server_selected.saturating_sub(1)
                        }
                        K::Down | K::Char('j') => {
                            self.server_selected = (self.server_selected + 1)
                                .min(self.servers.len().saturating_sub(1));
                        }
                        K::Char('x') => {
                            if let Some(r) = self.selected_server() {
                                return Action::StopServer(r.pid);
                            }
                        }
                        K::Char('c') => {
                            if let Some(r) = self.selected_server() {
                                return Action::CopyEndpoint(r.openai_url.clone());
                            }
                        }
                        _ => {}
                    },
                }
            }
```

NOTE: `Action::StopServer` and `Action::CopyEndpoint` are added in Task 2 — this task will not compile until Task 2's enum variants exist. To keep Task 1 self-contained and green, add the two variants now as part of Step 3 (they are inert until wired). Add to the `Action` enum:

```rust
    /// Stop the server with this pid (SIGTERM + unregister), then refresh.
    StopServer(u32),
    /// Copy this endpoint URL to the system clipboard.
    CopyEndpoint(String),
```

- [ ] **Step 6: Run tests, verify they pass**

Run: `cargo test -p paddock tab_toggles_between servers_tab_navigation set_servers_preserves`
Expected: PASS (3 tests). Also `cargo build` clean.

- [ ] **Step 7: Commit**

```bash
git add crates/paddock/src/tui/state.rs
git commit -m "feat(tui): Tab dimension, servers state, navigation"
```

---

## Task 2: State — stop/copy action plumbing tests

The `Action::StopServer`/`CopyEndpoint` variants were added in Task 1. This task verifies the key bindings produce them (behavioral lock, since the event loop relies on these exact actions).

**Files:**
- Test: `crates/paddock/src/tui/state.rs` (existing test module)

- [ ] **Step 1: Write the tests**

```rust
    #[test]
    fn x_on_servers_tab_returns_stop_action() {
        let mut s = state();
        s.set_servers(vec![srv(42, "qwen", 8080)]);
        s.tab = Tab::Servers;
        match s.handle_key(key(KeyCode::Char('x'))) {
            Action::StopServer(pid) => assert_eq!(pid, 42),
            other => panic!("expected StopServer, got {other:?}"),
        }
    }

    #[test]
    fn c_on_servers_tab_returns_copy_action() {
        let mut s = state();
        s.set_servers(vec![srv(42, "qwen", 8080)]);
        s.tab = Tab::Servers;
        match s.handle_key(key(KeyCode::Char('c'))) {
            Action::CopyEndpoint(url) => {
                assert_eq!(url, "http://127.0.0.1:8080/v1/chat/completions")
            }
            other => panic!("expected CopyEndpoint, got {other:?}"),
        }
    }

    #[test]
    fn x_on_empty_servers_tab_is_noop() {
        let mut s = state();
        s.tab = Tab::Servers;
        assert!(matches!(s.handle_key(key(KeyCode::Char('x'))), Action::None));
    }
```

NOTE: these tests `panic!("...{other:?}")`, so `Action` must derive `Debug`. It already does (`#[derive(Debug)]` on the enum). `srv` is the helper added in Task 1.

- [ ] **Step 2: Run, verify pass**

Run: `cargo test -p paddock _on_servers_tab`
Expected: PASS (3 tests). (They pass immediately because Task 1 implemented the bindings; this task is a behavioral guard.)

- [ ] **Step 3: Commit**

```bash
git add crates/paddock/src/tui/state.rs
git commit -m "test(tui): lock stop/copy actions on the servers tab"
```

---

## Task 3: Shared clipboard helper + tray refactor

**Files:**
- Create: `crates/paddock/src/clipboard.rs`
- Modify: `crates/paddock/src/main.rs` (declare the module)
- Modify: `crates/paddock/src/tray/mod.rs` (use the shared helper)

- [ ] **Step 1: Create the shared helper**

`crates/paddock/src/clipboard.rs`:

```rust
//! System clipboard write, shared by the tray and the TUI. arboard is already
//! a dependency (the tray used it inline).

/// Copy `text` to the system clipboard. Best-effort: a failure logs to stderr
/// and is otherwise ignored (no UI should crash because copy failed).
pub fn copy_to_clipboard(text: &str) {
    let res = arboard::Clipboard::new().and_then(|mut c| c.set_text(text.to_string()));
    if let Err(e) = res {
        eprintln!("could not copy to clipboard: {e}");
    }
}
```

- [ ] **Step 2: Declare the module**

In `crates/paddock/src/main.rs`, add to the module list at the top (alongside `mod app;` etc.):

```rust
mod clipboard;
```

- [ ] **Step 3: Point the tray at the shared helper**

In `crates/paddock/src/tray/mod.rs`, delete the private `fn copy_to_clipboard(text: &str) { ... }` (the 6-line fn at ~159-164) and replace its call sites. Find the call(s): `rg -n "copy_to_clipboard" crates/paddock/src/tray/mod.rs`. Replace `copy_to_clipboard(x)` / `Self::copy_to_clipboard(x)` with `crate::clipboard::copy_to_clipboard(x)`.

- [ ] **Step 4: Build + verify no behavior change**

Run: `cargo build && cargo clippy --workspace`
Expected: clean. The tray still compiles and copies via the shared helper.

- [ ] **Step 5: Commit**

```bash
git add crates/paddock/src/clipboard.rs crates/paddock/src/main.rs crates/paddock/src/tray/mod.rs
git commit -m "refactor(tui): shared clipboard helper for tray and TUI"
```

---

## Task 4: Background servers-refresh task

**Files:**
- Create: `crates/paddock/src/tui/servers_task.rs`
- Modify: `crates/paddock/src/tui/mod.rs` (declare `mod servers_task;` — wiring is Task 6)

- [ ] **Step 1: Create the task**

`crates/paddock/src/tui/servers_task.rs`:

```rust
//! Background servers refresh: a detached thread re-probes the serving
//! `Registry` on its own interval and sends each snapshot over an mpsc channel
//! the event loop drains. Kept off the UI thread because `list_live` makes
//! blocking HTTP readiness probes. The thread exits when the receiver is
//! dropped (TUI quit): the next `send` returns Err and the loop breaks.

use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

use paddock_core::hardware::RealSystemProbe;
use paddock_core::serving::{Registry, ServingRecord};

const REFRESH_EVERY: Duration = Duration::from_secs(2);

/// Spawn the periodic refresh. The receiver yields a fresh snapshot roughly
/// every 2s until the TUI drops it.
pub fn spawn_servers_refresh() -> Receiver<Vec<ServingRecord>> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        loop {
            let snapshot = Registry::open_default().list_live(&RealSystemProbe);
            if tx.send(snapshot).is_err() {
                break; // receiver dropped — TUI exited
            }
            std::thread::sleep(REFRESH_EVERY);
        }
    });
    rx
}
```

- [ ] **Step 2: Declare the module**

In `crates/paddock/src/tui/mod.rs`, add to the module list at the top (with `mod draw; mod state; mod sync_task;`):

```rust
mod servers_task;
```

- [ ] **Step 3: Build**

Run: `cargo build`
Expected: clean (the module compiles; `spawn_servers_refresh` is unused until Task 6 — a `dead_code` warning is acceptable here and resolved in Task 6. If clippy denies warnings in CI, that is fine because Task 6 lands in the same branch before merge.)

- [ ] **Step 4: Commit**

```bash
git add crates/paddock/src/tui/servers_task.rs crates/paddock/src/tui/mod.rs
git commit -m "feat(tui): background servers-refresh task"
```

---

## Task 5: Rendering — tab indicator, servers table, per-tab footer

**Files:**
- Modify: `crates/paddock/src/tui/draw.rs`

- [ ] **Step 1: Switch the body table by tab**

In `draw.rs`, the top-level `draw` fn currently always calls `draw_table` (models). Make it tab-aware. Replace the `draw_table(frame, table, state);` line with:

```rust
    match state.tab {
        crate::tui::state::Tab::Models => draw_table(frame, table, state),
        crate::tui::state::Tab::Servers => draw_servers(frame, table, state),
    }
```

Add `Tab` to the existing `use crate::tui::state::{...}` import line.

- [ ] **Step 2: Add the servers table renderer**

Add to `draw.rs` (near `draw_table`):

```rust
fn draw_servers(frame: &mut Frame, area: Rect, state: &TuiState) {
    if state.servers.is_empty() {
        let msg = Paragraph::new("no servers running — press s on a model to serve one")
            .style(Style::new().fg(Color::DarkGray));
        frame.render_widget(msg, area);
        return;
    }
    let header = Row::new(["MODEL", "RUNTIME", "ENDPOINT", "CTX", "UPTIME", "PID"])
        .style(Style::new().fg(Color::DarkGray).add_modifier(Modifier::BOLD));
    let rows = state.servers.iter().map(|r| {
        Row::new(vec![
            Cell::from(truncate_cell(&r.model_ref, 30)),
            Cell::from(runtime_label(r.runtime)),
            Cell::from(truncate_cell(&r.endpoint, 26)),
            Cell::from(r.ctx.to_string()),
            Cell::from(crate::output::humanize_since(uptime_secs(r.started_at))),
            Cell::from(r.pid.to_string()),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Min(20),
            Constraint::Length(10),
            Constraint::Length(24),
            Constraint::Length(7),
            Constraint::Length(8),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .row_highlight_style(Style::new().bg(ACCENT_DEEP).fg(Color::White));
    let mut ts = TableState::default().with_selected(Some(state.server_selected));
    frame.render_stateful_widget(table, area, &mut ts);
}

fn runtime_label(rt: RuntimeKind) -> &'static str {
    match rt {
        RuntimeKind::Ollama => "ollama",
        RuntimeKind::LlamaCpp => "llama.cpp",
        RuntimeKind::MlxLm => "mlx-lm",
    }
}

fn truncate_cell(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n - 1).collect();
        format!("{head}…")
    }
}

fn uptime_secs(started_at: i64) -> i64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    (now - started_at).max(0)
}
```

NOTE: `humanize_since` is `pub` in `crate::output` (used by the CLI `ps`). `RuntimeKind` is already imported at the top of draw.rs (`use paddock_core::catalog::RuntimeKind;`). `Cell`, `Modifier`, `ACCENT_DEEP` are already in scope.

- [ ] **Step 3: Tab indicator in the header**

The footer carries the key hints. Make `draw_footer` tab-aware. As the **very first statements** of `draw_footer` (before the `SPINNER` const / `sync_seg` computation), early-return a servers-specific hint line:

```rust
    if state.tab == Tab::Servers {
        let line = "j/k move · x stop · c copy endpoint · tab models · q quit";
        frame.render_widget(
            Paragraph::new(Span::styled(line, Style::new().fg(Color::DarkGray))),
            area,
        );
        return;
    }
```

`Tab` is in scope from the import added in Step 1; `Span`, `Style`, `Color`, `Paragraph` are already imported in draw.rs.

For the Models tab, make the tab discoverable by appending ` · tab servers` to the existing static hint string. It is the literal in the `(None, _) =>` arm of the `left` match. Change:

```rust
            "↑↓ move · enter detail · x run · s serve · / search · g/c/r/h use-case · R sync · q quit",
```

to:

```rust
            "↑↓ move · enter detail · x run · s serve · / search · g/c/r/h use-case · R sync · tab servers · q quit",
```

- [ ] **Step 4: Build + eyeball**

Run: `cargo build && cargo clippy --workspace`
Expected: clean. (Rendering is visually verified in Task 7's smoke test.)

- [ ] **Step 5: Commit**

```bash
git add crates/paddock/src/tui/draw.rs
git commit -m "feat(tui): render servers tab, indicator, and per-tab footer"
```

---

## Task 6: Event-loop wiring — spawn task, drain, stop/copy, inline detached serve

**Files:**
- Modify: `crates/paddock/src/tui/mod.rs`

- [ ] **Step 1: Spawn the refresh task and thread its receiver**

In `run`, after the sync setup and before `event_loop`, spawn the servers task:

```rust
    let servers_rx = servers_task::spawn_servers_refresh();
```

Change `event_loop`'s signature to accept it:

```rust
fn event_loop(
    terminal: &mut DefaultTerminal,
    state: &mut TuiState,
    app: &App,
    db: &Db,
    sync_rx: &mut Option<std::sync::mpsc::Receiver<sync_task::SyncMsg>>,
    servers_rx: &std::sync::mpsc::Receiver<Vec<paddock_core::serving::ServingRecord>>,
) -> Result<Option<Exit>> {
```

And pass it at the call site in `run`:

```rust
    let result = event_loop(&mut terminal, &mut state, &app, &db, &mut sync_rx, &servers_rx);
```

- [ ] **Step 2: Drain the servers channel each loop iteration**

In `event_loop`, right after the sync-channel drain block, add a drain that keeps only the latest snapshot:

```rust
        // Drain servers snapshots (keep the most recent).
        let mut latest = None;
        while let Ok(snapshot) = servers_rx.try_recv() {
            latest = Some(snapshot);
        }
        if let Some(snapshot) = latest {
            state.set_servers(snapshot);
        }
```

- [ ] **Step 3: Remove `Exit::Serve`; handle serve, stop, and copy inline**

Remove the `Serve(ServePlan)` variant from the `Exit` enum (keep `Run`). Update the post-loop `match result?` in `run` to drop the `Some(Exit::Serve(plan)) => ...` arm (only `Run` and `None` remain):

```rust
    match result? {
        Some(Exit::Run(plan)) => {
            println!("$ {}", plan.display());
            crate::launch(plan)
        }
        None => Ok(()),
    }
```

In the `match state.handle_key(key)` block, replace the `Action::Serve` arm and add the two new arms. The new `Action::Serve` arm suspends the terminal, serves detached, resumes, and switches to the servers tab with an immediate refresh:

```rust
            Action::Serve(plan) => {
                // Suspend the TUI so serve_with_plan can print load/install
                // progress (and prompt for install) on a clean terminal, then
                // resume. Detached: the server keeps running in the background.
                ratatui::restore();
                let res = crate::serve_with_plan(plan, false);
                *terminal = ratatui::init();
                if let Err(e) = res {
                    state.last_error = Some(e.to_string());
                }
                state.tab = state::Tab::Servers;
                // Immediate refresh so the new server shows without waiting for
                // the next background tick (it already passed readiness).
                let snapshot = paddock_core::serving::Registry::open_default()
                    .list_live(&paddock_core::hardware::RealSystemProbe);
                state.set_servers(snapshot);
            }
            Action::StopServer(pid) => {
                paddock_core::serving::terminate(pid);
                let _ = paddock_core::serving::Registry::open_default().unregister(pid);
                state.servers.retain(|r| r.pid != pid);
                state.server_selected = state
                    .server_selected
                    .min(state.servers.len().saturating_sub(1));
            }
            Action::CopyEndpoint(url) => crate::clipboard::copy_to_clipboard(&url),
```

Add the import for `Tab` use: the code references `state::Tab::Servers` — ensure `use state::{Action, SyncStatus, TuiState};` stays and `state::Tab` is reachable via the `state::` path (it is, since `Tab` is `pub` in the `state` module). No new `use` needed if you write `state::Tab::Servers`.

- [ ] **Step 4: Build + full tests**

Run: `cargo build && cargo test -p paddock && cargo test -p paddock-core && cargo clippy --workspace`
Expected: all green/clean. The `spawn_servers_refresh` dead_code warning from Task 4 is now resolved (it's used).

- [ ] **Step 5: Commit**

```bash
git add crates/paddock/src/tui/mod.rs
git commit -m "feat(tui): wire servers refresh, stop/copy, inline detached serve"
```

---

## Task 7: Docs + live smoke test

**Files:**
- Modify: `README.md` (TUI usage note)

- [ ] **Step 1: Document the TUI servers tab**

In `README.md`, find the Usage paragraph describing the TUI (search for "Running `paddock` with no arguments opens the interactive TUI"). Append a sentence:

```text
Press `Tab` to switch to the servers view: the models you have served show there with their endpoint, context, uptime and pid; `x` stops the selected server, `c` copies its OpenAI endpoint to the clipboard. Serving a model from the TUI (`s`) now runs it detached and lands you on the servers tab, so the TUI stays open.
```

(Match the README's prose style; no em-dashes per commit `517afa5`.)

- [ ] **Step 2: Commit the docs**

```bash
git add README.md
git commit -m "docs: document the TUI servers tab"
```

- [ ] **Step 3: Live smoke test (manual)**

Build and drive the TUI against a real llama.cpp-only model (e.g. `Qwen3.6-35B-A3B-MTP-GGUF`, which routes to llama-server):

```bash
cargo run -- # opens TUI
```

Verify:
1. `Tab` switches to an empty servers view showing "no servers running".
2. On the models tab, select a llama.cpp-only model and press `s`: the TUI suspends, shows load progress, resumes on the servers tab with the new server listed (model/runtime/endpoint/CTX/uptime/pid).
3. `c` copies the endpoint (paste elsewhere to confirm).
4. `x` stops the server; within ~2s (or immediately on the optimistic remove) it disappears from the list.
5. `q` quits cleanly; no orphaned process for a stopped server, but a still-running served model survives TUI exit (check `paddock ps`).

- [ ] **Step 4: Final verification**

Run: `cargo build && cargo test && cargo clippy --workspace`
Expected: all green/clean.

---

## Notes for the implementer

- The state machine is pure and fully unit-tested (Tasks 1-2). The background task, rendering, and event-loop IO are integration concerns verified by the live smoke test (Task 7) — keep them thin.
- Do NOT add an in-TUI log viewer, stop confirmation, multi-select, or Ollama-model actions (explicitly out of scope).
- `serve_with_plan(plan, false)` is the existing detached path from the CLI launcher; the TUI reuses it verbatim via suspend/resume.
