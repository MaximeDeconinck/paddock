# Background Catalog Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the catalog in a background thread while the TUI stays usable, per `docs/superpowers/specs/2026-06-20-background-sync-design.md`.

**Architecture:** SQLite in WAL mode lets a background sync thread write while the TUI reads. The thread runs the existing `catalog::sync()` on its own tokio runtime and Db connection, reporting terminal status over an `mpsc` channel the 250ms event loop drains. On completion the loop reloads + re-scores and swaps the rows atomically, preserving selection and search. A footer indicator shows sync state.

**Tech Stack:** Rust, std::thread + std::sync::mpsc, tokio (current-thread), rusqlite (WAL), ratatui.

**Plan deviation from spec (deliberate):** the spec said key `r` for manual sync, but `r` is already bound to `UseCase::Reasoning` (state.rs:145). Manual sync uses **`R`** (shift-r, "Refresh") instead.

---

### Task 1: WAL mode on the catalog DB

**Files:**
- Modify: `crates/paddock-core/src/catalog/db.rs` (`Db::open`, tests)

- [ ] **1.1 Write the failing test** (in db.rs `mod tests`):

```rust
    #[test]
    fn open_enables_wal_and_allows_concurrent_read_during_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.db");
        let writer = Db::open(&path).unwrap();

        // WAL is actually on.
        let mode: String = writer
            .conn
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal");

        // A second connection can read while the first holds data.
        writer.set_last_sync(1_700_000_000).unwrap();
        let reader = Db::open(&path).unwrap();
        assert_eq!(reader.last_sync().unwrap(), Some(1_700_000_000));
    }
```

- [ ] **1.2 Run, verify failure**: `cargo test -p paddock-core open_enables_wal` → fails (journal_mode is `delete`/`memory`, not `wal`). Note: `conn` is a private field; the test is in the same module so it can access it. If `conn` is not reachable from the test, add a `#[cfg(test)] pub(crate) fn journal_mode(&self) -> String` helper and assert on that instead.

- [ ] **1.3 Implement** in `Db::open`, right after the existing `conn.execute_batch("PRAGMA foreign_keys = ON;")?;`:

```rust
        // WAL + busy_timeout: the TUI reads while a background sync writes the
        // same file (docs/superpowers/specs/2026-06-20-background-sync-design.md).
        // query_row, not execute_batch: journal_mode returns the resulting mode.
        let _: String = conn.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))?;
        conn.execute_batch("PRAGMA busy_timeout = 5000;")?;
```

- [ ] **1.4 Run, verify pass**: `cargo test -p paddock-core` → all pass.

- [ ] **1.5 Commit**:
```bash
git add crates/paddock-core/src/catalog/db.rs
git commit -m "feat(core): WAL mode for concurrent catalog read/write"
```

---

### Task 2: `set_rows_preserving` on TuiState

**Files:**
- Modify: `crates/paddock/src/tui/state.rs` (new method + tests)

- [ ] **2.1 Write the failing tests** (in state.rs `mod tests`; reuse the existing test row helper — find how other tests build `ScoredModel`/`TuiState` in this module and match it):

```rust
    #[test]
    fn set_rows_preserving_keeps_selection_by_name() {
        let mut s = TuiState::new(rows3(), UseCase::General, runtimes());
        s.selected = 2; // third model
        let name = s.rows[2].model.name.clone();
        // New rows in a different order, same models present.
        let mut reordered = rows3();
        reordered.reverse();
        s.set_rows_preserving(reordered, UseCase::General);
        assert_eq!(s.rows[s.selected].model.name, name, "selection follows the model");
    }

    #[test]
    fn set_rows_preserving_clamps_when_model_gone() {
        let mut s = TuiState::new(rows3(), UseCase::General, runtimes());
        s.selected = 2;
        // Only the first model survives.
        s.set_rows_preserving(vec![rows3()[0].clone()], UseCase::General);
        assert!(s.selected < s.rows.len());
        assert_eq!(s.selected, 0);
    }

    #[test]
    fn set_rows_preserving_reapplies_active_query() {
        let mut s = TuiState::new(rows3(), UseCase::General, runtimes());
        let needle = s.rows[1].model.name[..3].to_lowercase();
        s.apply_search_pub(&needle); // expose apply_search to the test (see 2.3)
        let before = s.rows.len();
        s.set_rows_preserving(rows3(), UseCase::General);
        assert_eq!(s.rows.len(), before, "query still filters after swap");
        assert!(s.rows.iter().all(|r| r.model.name.to_lowercase().contains(&needle)));
    }
```

If the module has no `rows3()`/`runtimes()` helpers, add minimal ones in the test module that build 3 distinct `ScoredModel`s (distinct `model.name`) and a default `RuntimesStatus` — mirror whatever existing state.rs tests already do to construct these types.

- [ ] **2.2 Run, verify failure**: `cargo test -p paddock set_rows_preserving` → fails (method missing).

- [ ] **2.3 Implement** in `impl TuiState`, next to `set_rows`:

```rust
    /// Replace rows after a background sync. Like `set_rows` but preserves the
    /// selected model *by name* across the swap (cursor follows the model, not
    /// the index) and re-applies the active search filter.
    pub fn set_rows_preserving(&mut self, rows: Vec<ScoredModel>, use_case: UseCase) {
        let selected_name = self.rows.get(self.selected).map(|r| r.model.name.clone());
        self.all_rows = rows;
        self.use_case = use_case;
        let q = self.query.clone();
        self.apply_search(&q);
        self.selected = selected_name
            .and_then(|name| self.rows.iter().position(|r| r.model.name == name))
            .unwrap_or(0)
            .min(self.rows.len().saturating_sub(1));
    }
```

`apply_search` is private; the test in 2.1 needs it. Add a test-only shim in the `mod tests` block (not on the struct): since tests are in the same module, call `self.apply_search(&needle)` directly instead of `apply_search_pub` — replace that line in the test with `s.apply_search(&needle);`. (Private items are visible to the module's own `#[cfg(test)] mod tests` via `super::`.)

- [ ] **2.4 Run, verify pass**: `cargo test -p paddock set_rows_preserving` → 3 pass.

- [ ] **2.5 Commit**:
```bash
git add crates/paddock/src/tui/state.rs
git commit -m "feat(tui): set_rows_preserving keeps selection across a catalog swap"
```

---

### Task 3: SyncStatus state + message-to-status mapping

**Files:**
- Modify: `crates/paddock/src/tui/state.rs` (enum, field, pure mapping, tests)

- [ ] **3.1 Write the failing tests**:

```rust
    #[test]
    fn sync_status_default_is_idle() {
        let s = TuiState::new(rows3(), UseCase::General, runtimes());
        assert!(matches!(s.sync_status, SyncStatus::Idle));
    }

    #[test]
    fn running_then_done_and_failed() {
        assert!(matches!(SyncStatus::Idle.advance_running(), SyncStatus::Running));
        let done = SyncStatus::Running.on_done();
        assert!(matches!(done, SyncStatus::Done { .. }));
        let failed = SyncStatus::Running.on_failed("boom".into());
        assert!(matches!(failed, SyncStatus::Failed(m) if m == "boom"));
    }
```

- [ ] **3.2 Run, verify failure**: `cargo test -p paddock sync_status` → fails (type missing).

- [ ] **3.3 Implement** in state.rs (add near the top, after `Mode`):

```rust
/// Background-sync lifecycle as seen by the UI. Pure data; the event loop owns
/// the channel and drives the transitions.
#[derive(Debug)]
pub enum SyncStatus {
    /// No sync this session, or never triggered.
    Idle,
    /// A background sync is in flight (v1: spinner only, no count).
    Running,
    /// Finished at this instant (used to show "catalog updated" briefly).
    Done { at: std::time::Instant },
    /// Failed; the message is shown in the footer.
    Failed(String),
}

impl SyncStatus {
    pub fn advance_running(self) -> SyncStatus {
        SyncStatus::Running
    }
    pub fn on_done(self) -> SyncStatus {
        SyncStatus::Done {
            at: std::time::Instant::now(),
        }
    }
    pub fn on_failed(self, msg: String) -> SyncStatus {
        SyncStatus::Failed(msg)
    }
}
```

Add the field to `TuiState` (after `detail_serve_plan`):

```rust
    /// Background catalog-sync status, shown in the footer.
    pub sync_status: SyncStatus,
    /// Spinner animation frame, advanced once per event-loop tick.
    pub tick: u64,
```

Initialize both in `TuiState::new`:

```rust
            sync_status: SyncStatus::Idle,
            tick: 0,
```

- [ ] **3.4 Run, verify pass**: `cargo test -p paddock sync_status` → pass. Also `cargo build -p paddock` to confirm `TuiState::new` still compiles everywhere it is called (tui/mod.rs and state.rs tests).

- [ ] **3.5 Commit**:
```bash
git add crates/paddock/src/tui/state.rs
git commit -m "feat(tui): SyncStatus state on TuiState"
```

---

### Task 4: background sync task

**Files:**
- Create: `crates/paddock/src/tui/sync_task.rs`
- Modify: `crates/paddock/src/tui/mod.rs` (add `mod sync_task;`)

- [ ] **4.1 Create the module** (no unit test — it spawns a thread + network; verified end-to-end in Task 7). Content:

```rust
//! Background catalog refresh: a detached thread runs the same `catalog::sync`
//! the CLI uses, on its own tokio runtime and its own Db connection (never the
//! TUI's), reporting terminal status over an mpsc channel the event loop drains.

use std::sync::mpsc::{Receiver, channel};

use paddock_core::catalog::db::{Db, default_db_path};
use paddock_core::catalog::hf::ReqwestClient;
use paddock_core::catalog::{SyncOptions, SyncReport, sync};

/// Terminal-only in v1: `Progress` is reserved for a future per-source counter
/// (see the design's "Progress granularity"); it is never sent today.
pub enum SyncMsg {
    #[allow(dead_code)]
    Progress { source: &'static str, count: usize },
    Done(Box<SyncReport>),
    Failed(String),
}

/// Spawn the catalog sync on a background thread. The returned receiver yields
/// exactly one terminal message (`Done` or `Failed`) then disconnects.
pub fn spawn_sync(opts: SyncOptions) -> Receiver<SyncMsg> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let result = (|| -> Result<SyncReport, String> {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| format!("runtime: {e}"))?;
            let db = Db::open(default_db_path()).map_err(|e| e.to_string())?;
            let http = ReqwestClient::new().map_err(|e| e.to_string())?;
            rt.block_on(sync(&http, &db, &opts)).map_err(|e| e.to_string())
        })();
        let _ = match result {
            Ok(report) => tx.send(SyncMsg::Done(Box::new(report))),
            Err(e) => tx.send(SyncMsg::Failed(e)),
        };
    });
    rx
}
```

- [ ] **4.2 Register the module** in `crates/paddock/src/tui/mod.rs`, with the other `mod` lines at the top:

```rust
mod draw;
mod state;
mod sync_task;
```

- [ ] **4.3 Verify it compiles**: `cargo build -p paddock` → success. Confirm `SyncOptions`, `SyncReport`, `sync`, `ReqwestClient`, `Db`, `default_db_path` are all re-exported at those paths (they are used by `main.rs::Sync` already — match those import paths if any differ).

- [ ] **4.4 Commit**:
```bash
git add crates/paddock/src/tui/sync_task.rs crates/paddock/src/tui/mod.rs
git commit -m "feat(tui): background sync task over an mpsc channel"
```

---

### Task 5: event-loop integration (trigger, drain, refresh, `R` key, no-bail)

**Files:**
- Modify: `crates/paddock/src/tui/mod.rs` (run + event_loop)
- Modify: `crates/paddock/src/tui/state.rs` (Action::StartSync + `R` binding)

- [ ] **5.1 Add the Action variant + key binding** (state.rs). In `enum Action` add:

```rust
    /// Kick off a background catalog sync (the event loop owns the thread).
    StartSync,
```

In `handle_key`, `Mode::List` arm, add alongside the other char keys (after the `'h'` use-case line):

```rust
                K::Char('R') => return Action::StartSync,
```

- [ ] **5.2 Rewrite `tui::run`** (mod.rs) to never bail on an empty catalog and to trigger a stale/empty sync:

```rust
pub fn run(app: App) -> Result<()> {
    let db = app.open_db()?;
    let rows = app.scored_models(&db, UseCase::default(), false)?;
    let mut state = TuiState::new(rows, UseCase::default(), app.profile.runtimes.clone());

    // Stale (>24h) or empty catalog → kick off a background refresh. The TUI
    // opens immediately against whatever snapshot exists (possibly empty).
    const STALE_AFTER_SECS: i64 = 24 * 60 * 60;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let stale = match db.last_sync() {
        Ok(Some(ts)) => now - ts > STALE_AFTER_SECS,
        _ => true, // never synced
    };
    let mut sync_rx = if stale || state.all_rows.is_empty() {
        state.sync_status = state.sync_status.advance_running();
        Some(crate::tui::sync_task::spawn_sync(
            paddock_core::catalog::SyncOptions::default(),
        ))
    } else {
        None
    };

    let mut terminal = ratatui::init();
    let result = event_loop(&mut terminal, &mut state, &app, &db, &mut sync_rx);
    ratatui::restore();

    match result? {
        Some(Exit::Run(plan)) => {
            println!("$ {}", plan.display());
            crate::launch(plan)
        }
        Some(Exit::Serve(plan)) => crate::serve_with_plan(plan),
        None => Ok(()),
    }
}
```

Note: this drops the `anyhow::bail!("catalog is empty")`. Keep the `use` for `anyhow::Result` (still the return type).

- [ ] **5.3 Rewrite `event_loop`** to take the receiver, advance the spinner tick, drain the channel each iteration, and refresh on `Done`:

```rust
fn event_loop(
    terminal: &mut DefaultTerminal,
    state: &mut TuiState,
    app: &App,
    db: &Db,
    sync_rx: &mut Option<std::sync::mpsc::Receiver<crate::tui::sync_task::SyncMsg>>,
) -> Result<Option<Exit>> {
    use crate::tui::sync_task::SyncMsg;
    use std::sync::mpsc::TryRecvError;
    loop {
        state.tick = state.tick.wrapping_add(1);
        terminal.draw(|frame| draw::draw(frame, state, &app.profile))?;

        // Drain background-sync messages (non-blocking).
        if let Some(rx) = sync_rx.as_ref() {
            match rx.try_recv() {
                Ok(SyncMsg::Done(_)) => {
                    let rows = app.scored_models(db, state.use_case, false)?;
                    state.set_rows_preserving(rows, state.use_case);
                    state.sync_status = SyncStatus::Idle.on_done();
                    *sync_rx = None;
                }
                Ok(SyncMsg::Failed(e)) => {
                    state.sync_status = SyncStatus::Idle.on_failed(e);
                    *sync_rx = None;
                }
                Ok(SyncMsg::Progress { .. }) => {} // not emitted in v1
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    state.sync_status = SyncStatus::Idle.on_failed("sync thread died".into());
                    *sync_rx = None;
                }
            }
        }

        if !event::poll(Duration::from_millis(250))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        match state.handle_key(key) {
            Action::None => {}
            Action::Quit => return Ok(None),
            Action::Run(plan) => return Ok(Some(Exit::Run(plan))),
            Action::Serve(plan) => return Ok(Some(Exit::Serve(plan))),
            Action::Rescore(uc) => {
                let rows = app.scored_models(db, uc, false)?;
                state.set_rows(rows, uc);
            }
            Action::StartSync => {
                if sync_rx.is_none() {
                    state.sync_status = SyncStatus::Idle.advance_running();
                    *sync_rx = Some(crate::tui::sync_task::spawn_sync(
                        paddock_core::catalog::SyncOptions::default(),
                    ));
                }
            }
        }
    }
}
```

Add the needed imports at the top of mod.rs: `use state::{Action, SyncStatus, TuiState};` (extend the existing `use state::{...}` line).

- [ ] **5.4 Run**: `cargo build -p paddock` then `cargo test` (workspace) → all pass (existing tests unaffected; new state tests already green).

- [ ] **5.5 Commit**:
```bash
git add crates/paddock/src/tui/mod.rs crates/paddock/src/tui/state.rs
git commit -m "feat(tui): drive background sync from the event loop with atomic refresh"
```

---

### Task 6: footer sync indicator

**Files:**
- Modify: `crates/paddock/src/tui/draw.rs` (footer)

v1 shows the spinner while running and a brief `catalog updated` on completion;
the idle state shows nothing extra (no "synced X ago" — the footer has no DB
handle, and threading one in is out of scope). No humanizer needed.

- [ ] **6.1 Implement the footer indicator** (draw.rs). At the top of `draw_footer`, build a sync segment and prepend it to the right-aligned text. Replace the body of `draw_footer` with:

```rust
fn draw_footer(frame: &mut Frame, area: Rect, state: &TuiState) {
    const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let sync_seg = match &state.sync_status {
        SyncStatus::Running => {
            let frame = SPINNER[(state.tick as usize) % SPINNER.len()];
            Some(format!("{frame} syncing…"))
        }
        SyncStatus::Done { at } if at.elapsed().as_secs() < 5 => Some("catalog updated".into()),
        _ => None,
    };

    let left = match (&state.last_error, &state.sync_status) {
        (Some(err), _) => Line::from(Span::styled(
            format!("error: {err}"),
            Style::new().fg(ACCENT),
        )),
        (None, SyncStatus::Failed(msg)) => Line::from(Span::styled(
            format!("sync failed: {msg}"),
            Style::new().fg(ACCENT),
        )),
        (None, _) => Line::from(Span::styled(
            "↑↓ move · enter detail · x run · s serve · / search · g/c/r/h use-case · R sync · q quit",
            Style::new().fg(Color::DarkGray),
        )),
    };
    let uc = use_case_label(state.use_case);
    let base = match &state.mode {
        Mode::Search { query } => format!("{uc} · /{query}▌"),
        _ if !state.query.is_empty() => format!("{uc} · /{}", state.query),
        _ => uc.to_string(),
    };
    let right = match sync_seg {
        Some(seg) => format!("{seg} · {base}"),
        None => base,
    };
    frame.render_widget(Paragraph::new(left), area);
    frame.render_widget(
        Paragraph::new(Span::styled(right, Style::new().fg(ACCENT))).alignment(Alignment::Right),
        area,
    );
}
```

Add `SyncStatus` to the state import at the top of draw.rs: extend `use crate::tui::state::{Mode, TuiState, use_case_label};` to include `SyncStatus`.

- [ ] **6.2 Run**: `cargo build -p paddock` then `cargo test` → all pass.

- [ ] **6.3 Commit**:
```bash
git add crates/paddock/src/output.rs crates/paddock/src/tui/draw.rs
git commit -m "feat(tui): footer sync indicator (spinner / updated / failed)"
```

---

### Task 7: real verification + README

**Files:**
- Modify: `README.md` (tray/tui section: mention background sync + `R`)

- [ ] **7.1 Gates**: `cargo fmt --all` then `cargo clippy --all-targets` → zero warnings; `cargo test` → all pass.

- [ ] **7.2 Real run, stale path**: force a stale catalog and launch the TUI.
  - `cargo build --release`
  - Make the catalog look old: open the TUI (`./target/release/paddock`) — if it last synced <24h ago, press `R` to force a sync. Observe: the spinner appears in the footer right (`⠋ syncing…`), the list stays usable (move cursor, open detail) during the sync, and when it finishes the footer shows `catalog updated` and the row count grows. Paste what you observed (spinner seen, list usable, updated message).

- [ ] **7.3 Real run, empty path**: prove the empty-first-run no longer bails.
  - `PADDOCK_DB_PATH=/tmp/paddock-empty.db ./target/release/paddock`
  - Expect: the TUI opens (not a `catalog is empty` crash) with an empty list and `⠋ syncing…` in the footer, then fills once the sync completes. Paste the result.

- [ ] **7.4 Concurrency sanity**: while a `R`-triggered sync runs in the TUI, confirm no panic/lock error appears and the post-sync list is correct (WAL doing its job). Note the catalog count before/after.

- [ ] **7.5 README**: in the `paddock tray` / TUI area, add a short paragraph: the TUI refreshes the catalog in the background on launch when it is more than 24h stale (or empty), shows a spinner in the footer while it runs and `catalog updated` when done, and `R` forces a refresh on demand; the blocking `paddock sync` command is still available for scripts.

- [ ] **7.6 Commit + push**:
```bash
git add README.md
git commit -m "feat: background catalog sync in the TUI with footer indicator"
git push origin main
```

---

## Self-review notes

- **Spec coverage:** WAL (T1), background task (T4), SyncStatus (T3), event-loop trigger+drain+atomic refresh (T5), footer indicator (T6), empty-first-run no-bail (T5.2), selection/search preservation (T2), CLI unchanged (untouched), 24h-stale + manual key (T5, key is `R` not `r` — documented deviation).
- **Progress granularity:** v1 spinner-only; `SyncMsg::Progress` defined but never sent, `#[allow(dead_code)]` on the variant (T4). Matches the spec's resolved decision.
- **Type/signature consistency:** `set_rows_preserving(Vec<ScoredModel>, UseCase)` (T2) called in T5.3; `SyncStatus` variants `Idle/Running/Done{at}/Failed(String)` (T3) used in T5/T6; `spawn_sync(SyncOptions) -> Receiver<SyncMsg>` (T4) called in T5. Task 6 is footer-indicator only (spinner + `catalog updated`); no humanizer, no idle age text, so no unused code.
