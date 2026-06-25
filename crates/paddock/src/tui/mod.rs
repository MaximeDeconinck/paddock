//! Terminal lifecycle + event loop. All state transitions live in `state`,
//! all rendering in `draw` — this module only wires them to the terminal.

mod draw;
mod state;
mod sync_task;

use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyEventKind};
use paddock_core::catalog::db::Db;
use paddock_core::runtime::{RunPlan, ServePlan};
use paddock_core::score::UseCase;
use ratatui::DefaultTerminal;

use crate::app::App;
use state::{Action, SyncStatus, TuiState};

/// What to do after the terminal is restored.
enum Exit {
    Run(RunPlan),
    Serve(ServePlan),
}

pub fn run(app: App) -> Result<()> {
    let db = app.open_db()?;
    let rows = app.scored_models(&db, UseCase::default(), false)?;
    let mut state = TuiState::new(rows, UseCase::default(), app.profile.runtimes.clone());

    // Stale (>24h) or empty catalog -> kick off a background refresh. The TUI
    // opens immediately against whatever snapshot exists (possibly empty).
    const STALE_AFTER_SECS: i64 = 24 * 60 * 60;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    state.last_sync = db.last_sync().ok().flatten();
    let stale = match state.last_sync {
        Some(ts) => now - ts > STALE_AFTER_SECS,
        None => true, // never synced
    };
    let mut sync_rx = if stale || state.all_rows.is_empty() {
        state.sync_status = SyncStatus::Running;
        Some(sync_task::spawn_sync(
            paddock_core::catalog::SyncOptions::default(),
        ))
    } else {
        None
    };

    // ratatui 0.29 helpers: raw mode + alternate screen + panic hook.
    let mut terminal = ratatui::init();
    let result = event_loop(&mut terminal, &mut state, &app, &db, &mut sync_rx);
    ratatui::restore();

    // Launch AFTER restore so the child owns a clean tty. `launch` and
    // `serve_with_plan` are the same confirm-install paths as `paddock run` /
    // `paddock serve` — the never-auto-install guarantee lives in one place.
    match result? {
        Some(Exit::Run(plan)) => {
            println!("$ {}", plan.display());
            crate::launch(plan)
        }
        Some(Exit::Serve(plan)) => crate::serve_with_plan(plan, true),
        None => Ok(()),
    }
}

fn event_loop(
    terminal: &mut DefaultTerminal,
    state: &mut TuiState,
    app: &App,
    db: &Db,
    sync_rx: &mut Option<std::sync::mpsc::Receiver<sync_task::SyncMsg>>,
) -> Result<Option<Exit>> {
    use std::sync::mpsc::TryRecvError;
    use sync_task::SyncMsg;
    loop {
        state.tick = state.tick.wrapping_add(1);
        terminal.draw(|frame| draw::draw(frame, state, &app.profile))?;

        // Drain background-sync messages (non-blocking).
        if let Some(rx) = sync_rx.as_ref() {
            match rx.try_recv() {
                Ok(SyncMsg::Done(_)) => {
                    let rows = app.scored_models(db, state.use_case, false)?;
                    state.set_rows_preserving(rows, state.use_case);
                    state.last_sync = db.last_sync().ok().flatten();
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
        // macOS terminals also deliver Release/Repeat events — act on Press only.
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
                    state.sync_status = SyncStatus::Running;
                    *sync_rx = Some(sync_task::spawn_sync(
                        paddock_core::catalog::SyncOptions::default(),
                    ));
                }
            }
        }
    }
}
