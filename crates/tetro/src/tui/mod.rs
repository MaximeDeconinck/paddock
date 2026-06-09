//! Terminal lifecycle + event loop. All state transitions live in `state`,
//! all rendering in `draw` — this module only wires them to the terminal.

mod draw;
mod state;

use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyEventKind};
use ratatui::DefaultTerminal;
use tetro_core::catalog::db::Db;
use tetro_core::runtime::RunPlan;
use tetro_core::score::UseCase;

use crate::app::App;
use state::{Action, TuiState};

pub fn run(app: App) -> Result<()> {
    let db = app.open_db()?;
    let rows = app.scored_models(&db, UseCase::default(), false)?;
    if rows.is_empty() {
        anyhow::bail!("catalog is empty — run `tetro sync` first");
    }
    let mut state = TuiState::new(rows, UseCase::default(), app.profile.runtimes.clone());

    // ratatui 0.29 helpers: raw mode + alternate screen + panic hook.
    let mut terminal = ratatui::init();
    let result = event_loop(&mut terminal, &mut state, &app, &db);
    ratatui::restore();

    // Launch AFTER restore so the child owns a clean tty. `launch` is the
    // same confirm-install-then-exec path as `tetro run` — the never-auto-
    // install guarantee lives in one place.
    match result? {
        Some(plan) => {
            println!("$ {}", plan.display());
            crate::launch(plan)
        }
        None => Ok(()),
    }
}

fn event_loop(
    terminal: &mut DefaultTerminal,
    state: &mut TuiState,
    app: &App,
    db: &Db,
) -> Result<Option<RunPlan>> {
    loop {
        terminal.draw(|frame| draw::draw(frame, state, &app.profile))?;
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
            Action::Run(plan) => return Ok(Some(plan)),
            Action::Rescore(uc) => {
                let rows = app.scored_models(db, uc, false)?;
                state.set_rows(rows, uc);
            }
        }
    }
}
