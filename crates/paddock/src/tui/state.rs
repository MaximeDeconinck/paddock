//! Pure TUI state machine — no terminal IO, fully unit-testable.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use paddock_core::hardware::RuntimesStatus;
use paddock_core::runtime::{RunPlan, ServePlan, plan_run, plan_serve};
use paddock_core::score::UseCase;

use crate::app::ScoredModel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    List,
    Detail,
    Search { query: String },
}

/// Background-sync lifecycle as seen by the UI. Pure data; the event loop owns
/// the channel and drives the transitions.
// Variant payloads (`Done.at`, `Failed.0`) are read by the footer renderer.
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
    // The event loop assigns `SyncStatus::Running` directly (avoids moving out
    // of a borrowed field); this transition exists for symmetry and is tested.
    #[allow(dead_code)]
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

#[derive(Debug)]
pub enum Action {
    None,
    Quit,
    /// Quit the TUI and hand this plan to the launcher (after terminal restore).
    Run(RunPlan),
    /// Quit the TUI and hand this plan to the serve lifecycle (after restore).
    Serve(ServePlan),
    /// Re-score the catalog for a new use case (the event loop owns App + Db).
    Rescore(UseCase),
    /// Kick off a background catalog sync (the event loop owns the thread).
    StartSync,
}

pub struct TuiState {
    /// Rows currently displayed (filtered by the active search query).
    pub rows: Vec<ScoredModel>,
    /// Unfiltered rows for the current use case (search reset).
    pub all_rows: Vec<ScoredModel>,
    pub selected: usize,
    pub use_case: UseCase,
    pub mode: Mode,
    /// Applied search query (shown in the footer; empty = no filter).
    pub query: String,
    /// Runtime availability snapshot used to build run plans.
    pub runtimes: RuntimesStatus,
    /// Last plan_run failure, surfaced in the footer instead of crashing.
    pub last_error: Option<String>,
    /// Run plan for the selected row, computed once on Detail entry so the
    /// render path never calls plan_run. Cleared when Detail closes.
    pub detail_plan: Option<Result<RunPlan, String>>,
    /// Serve plan for the selected row, computed alongside `detail_plan` so
    /// the detail popup can show the endpoint without calling plan_serve.
    pub detail_serve_plan: Option<Result<ServePlan, String>>,
    /// Background catalog-sync status, shown in the footer.
    pub sync_status: SyncStatus,
    /// Spinner animation frame, advanced once per event-loop tick.
    // Advanced by the event loop; read by the footer renderer.
    pub tick: u64,
    /// Catalog `last_sync` epoch (seconds), read from the DB at launch and
    /// after each background sync; the footer shows it as "synced Xm ago".
    pub last_sync: Option<i64>,
}

impl TuiState {
    pub fn new(rows: Vec<ScoredModel>, use_case: UseCase, runtimes: RuntimesStatus) -> Self {
        Self {
            all_rows: rows.clone(),
            rows,
            selected: 0,
            use_case,
            mode: Mode::List,
            query: String::new(),
            runtimes,
            last_error: None,
            detail_plan: None,
            detail_serve_plan: None,
            sync_status: SyncStatus::Idle,
            tick: 0,
            last_sync: None,
        }
    }

    /// Replace rows after a re-score (use-case switch); keeps the active
    /// filter and the cursor position (clamped to the new row count).
    pub fn set_rows(&mut self, rows: Vec<ScoredModel>, use_case: UseCase) {
        self.all_rows = rows;
        self.use_case = use_case;
        let q = self.query.clone();
        let cursor = self.selected;
        self.apply_search(&q);
        self.selected = cursor.min(self.rows.len().saturating_sub(1));
    }

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

    pub fn handle_key(&mut self, key: KeyEvent) -> Action {
        use KeyCode as K;
        self.last_error = None;
        // Ctrl-C quits from any mode — never swallowed by search input.
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == K::Char('c') {
            return Action::Quit;
        }
        match &mut self.mode {
            Mode::Search { query } => match key.code {
                K::Esc => {
                    self.mode = Mode::List;
                    self.apply_search("");
                }
                K::Enter => {
                    let q = query.clone();
                    self.mode = Mode::List;
                    self.apply_search(&q);
                }
                K::Backspace => {
                    query.pop();
                    let q = query.clone();
                    self.apply_search(&q);
                }
                // Modified chars (ctrl/alt chords) are commands, not input.
                K::Char(c)
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                {
                    query.push(c);
                    let q = query.clone();
                    self.apply_search(&q);
                }
                _ => {}
            },
            Mode::Detail => match key.code {
                K::Esc | K::Char('q') | K::Enter => {
                    self.mode = Mode::List;
                    self.detail_plan = None;
                    self.detail_serve_plan = None;
                }
                K::Char('x') => return self.run_selected(),
                K::Char('s') => return self.serve_selected(),
                _ => {}
            },
            Mode::List => match key.code {
                K::Char('q') => return Action::Quit,
                K::Up | K::Char('k') => self.selected = self.selected.saturating_sub(1),
                K::Down | K::Char('j') => {
                    self.selected = (self.selected + 1).min(self.rows.len().saturating_sub(1));
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
        }
        Action::None
    }

    /// Case-insensitive contains filter on the model name; resets selection.
    fn apply_search(&mut self, q: &str) {
        self.query = q.to_string();
        let needle = q.to_lowercase();
        self.rows = if needle.is_empty() {
            self.all_rows.clone()
        } else {
            self.all_rows
                .iter()
                .filter(|r| r.model.name.to_lowercase().contains(&needle))
                .cloned()
                .collect()
        };
        self.selected = 0;
    }

    fn set_use_case(&mut self, uc: UseCase) -> Action {
        self.use_case = uc;
        Action::Rescore(uc)
    }

    /// Single source for run-plan computation (Detail entry and `x` both use
    /// it), so the render path never calls plan_run.
    fn plan_for_selected(&self) -> Option<Result<RunPlan, String>> {
        let row = self.rows.get(self.selected)?;
        let variant = &row.model.variants[row.variant_idx];
        // TUI has no flag surface, so ctx is always the default.
        Some(plan_run(&row.model, variant, &self.runtimes, None).map_err(|e| e.to_string()))
    }

    /// Single source for serve-plan computation (Detail entry and `s` both
    /// use it), so the render path never calls plan_serve. Default port: the
    /// TUI has no flag surface, so `port` is always None.
    fn serve_plan_for_selected(&self) -> Option<Result<ServePlan, String>> {
        let row = self.rows.get(self.selected)?;
        let variant = &row.model.variants[row.variant_idx];
        Some(plan_serve(&row.model, variant, &self.runtimes, None, None).map_err(|e| e.to_string()))
    }

    /// Build the run plan for the selected row. A plan_run failure must not
    /// crash the TUI: it is stored and rendered in the footer.
    fn run_selected(&mut self) -> Action {
        match self.plan_for_selected() {
            Some(Ok(plan)) => Action::Run(plan),
            Some(Err(e)) => {
                self.last_error = Some(e);
                Action::None
            }
            None => Action::None,
        }
    }

    /// Same error contract as `run_selected`: plan_serve failures go to the
    /// footer, never crash the TUI.
    fn serve_selected(&mut self) -> Action {
        match self.serve_plan_for_selected() {
            Some(Ok(plan)) => Action::Serve(plan),
            Some(Err(e)) => {
                self.last_error = Some(e);
                Action::None
            }
            None => Action::None,
        }
    }
}

pub fn use_case_label(uc: UseCase) -> &'static str {
    match uc {
        UseCase::General => "general",
        UseCase::Coding => "coding",
        UseCase::Chat => "chat",
        UseCase::Reasoning => "reasoning",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;
    use paddock_core::catalog::{CatalogModel, CatalogVariant, RuntimeKind, Source};
    use paddock_core::estimate::{DEFAULT_CONTEXT, MemoryBudget, estimate_memory, estimate_speed};
    use paddock_core::score::score_variant;

    fn fake_row(name: &str) -> ScoredModel {
        let model = CatalogModel {
            id: 0,
            name: name.to_string(),
            family: None,
            source: Source::Ollama,
            repo: None,
            params_total: 8_000_000_000,
            params_active: 8_000_000_000,
            architecture: None,
            context_max: 8192,
            released_at: None,
            released_approx: false,
            variants: vec![CatalogVariant {
                quant: "Q4_K_M".into(),
                bpw: 4.83,
                file_size_bytes: None,
                layers: 32,
                kv_heads: 8,
                head_dim: 128,
                embedding_dim: 4096,
                runtime_compat: vec![RuntimeKind::Ollama],
                source_tag: None,
            }],
        };
        let budget = MemoryBudget {
            gpu_effective_bytes: 24 * (1u64 << 30),
            ram_total_bytes: 32 * (1u64 << 30),
        };
        let mv = model.to_model_variant(&model.variants[0]);
        let memory = estimate_memory(&mv, DEFAULT_CONTEXT, &budget);
        let speed = estimate_speed(&mv, 400.0, memory.kv_cache_bytes);
        let score = score_variant(&mv, &memory, &speed, UseCase::General, None);
        ScoredModel {
            model,
            variant_idx: 0,
            memory,
            speed,
            score,
        }
    }

    fn state() -> TuiState {
        TuiState::new(
            vec![
                fake_row("Llama3-8B"),
                fake_row("Qwen2.5-Coder"),
                fake_row("Mistral-7B"),
            ],
            UseCase::General,
            RuntimesStatus::default(),
        )
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn j_k_move_selection_within_bounds() {
        let mut s = state();
        assert_eq!(s.selected, 0);
        s.handle_key(key(KeyCode::Char('k'))); // clamped at top
        assert_eq!(s.selected, 0);
        s.handle_key(key(KeyCode::Char('j')));
        assert_eq!(s.selected, 1);
        s.handle_key(key(KeyCode::Down));
        s.handle_key(key(KeyCode::Char('j'))); // clamped at bottom
        s.handle_key(key(KeyCode::Char('j')));
        assert_eq!(s.selected, 2);
        s.handle_key(key(KeyCode::Up));
        assert_eq!(s.selected, 1);
    }

    #[test]
    fn search_filters_incrementally_and_esc_clears() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/')));
        assert!(matches!(s.mode, Mode::Search { .. }));
        for c in "qwen".chars() {
            s.handle_key(key(KeyCode::Char(c)));
        }
        assert_eq!(s.rows.len(), 1);
        assert_eq!(s.rows[0].model.name, "Qwen2.5-Coder");
        assert_eq!(s.selected, 0);
        // Enter applies the filter and returns to the list
        s.handle_key(key(KeyCode::Enter));
        assert_eq!(s.mode, Mode::List);
        assert_eq!(s.rows.len(), 1);
        // '/' again then Esc clears everything
        s.handle_key(key(KeyCode::Char('/')));
        s.handle_key(key(KeyCode::Esc));
        assert_eq!(s.mode, Mode::List);
        assert_eq!(s.rows.len(), 3);
        assert!(s.query.is_empty());
    }

    #[test]
    fn x_returns_run_action_with_runtime_argv() {
        let mut s = state();
        let action = s.handle_key(key(KeyCode::Char('x')));
        match action {
            Action::Run(plan) => {
                assert!(!plan.argv.is_empty());
                assert_eq!(plan.argv[0], "ollama");
                // ollama not installed in the fake RuntimesStatus: the plan
                // must carry an install step, never auto-run it.
                assert!(plan.install.is_some());
            }
            other => panic!("expected Action::Run, got {other:?}"),
        }
    }

    #[test]
    fn s_returns_serve_action_with_endpoint() {
        // List mode
        let mut s = state();
        let action = s.handle_key(key(KeyCode::Char('s')));
        match action {
            Action::Serve(plan) => {
                assert!(!plan.endpoint.is_empty());
                // Ollama-shaped plan: fixed daemon port + `ollama pull` pre-step.
                assert!(plan.endpoint.contains("11434"));
                assert_eq!(plan.pre_steps[0][0], "ollama");
                // ollama not installed in the fake RuntimesStatus: the plan
                // must carry an install step, never auto-run it.
                assert!(plan.install.is_some());
            }
            other => panic!("expected Action::Serve, got {other:?}"),
        }
        // Detail mode: same key, same action.
        let mut s = state();
        s.handle_key(key(KeyCode::Enter));
        assert_eq!(s.mode, Mode::Detail);
        assert!(matches!(s.detail_serve_plan, Some(Ok(_))));
        assert!(matches!(
            s.handle_key(key(KeyCode::Char('s'))),
            Action::Serve(_)
        ));
    }

    #[test]
    fn ctrl_c_quits_from_all_modes() {
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        // List
        let mut s = state();
        assert!(matches!(s.handle_key(ctrl_c), Action::Quit));
        // Detail
        let mut s = state();
        s.handle_key(key(KeyCode::Enter));
        assert_eq!(s.mode, Mode::Detail);
        assert!(matches!(s.handle_key(ctrl_c), Action::Quit));
        // Search — and the 'c' must not land in the query
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/')));
        assert!(matches!(s.handle_key(ctrl_c), Action::Quit));
        assert!(matches!(&s.mode, Mode::Search { query } if query.is_empty()));
    }

    #[test]
    fn search_ignores_modified_chars() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/')));
        s.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::ALT));
        assert!(matches!(&s.mode, Mode::Search { query } if query.is_empty()));
        assert_eq!(s.rows.len(), 3);
    }

    #[test]
    fn detail_entry_computes_plan_and_exit_clears_it() {
        let mut s = state();
        assert!(s.detail_plan.is_none());
        s.handle_key(key(KeyCode::Enter));
        assert_eq!(s.mode, Mode::Detail);
        match &s.detail_plan {
            Some(Ok(plan)) => assert_eq!(plan.argv[0], "ollama"),
            other => panic!("expected Some(Ok(plan)), got {other:?}"),
        }
        s.handle_key(key(KeyCode::Esc));
        assert_eq!(s.mode, Mode::List);
        assert!(s.detail_plan.is_none());
        assert!(s.detail_serve_plan.is_none());
    }

    #[test]
    fn rescore_preserves_cursor_clamped() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('j')));
        s.handle_key(key(KeyCode::Char('j')));
        assert_eq!(s.selected, 2);
        // Same query, new rows: cursor kept.
        s.set_rows(
            vec![
                fake_row("Llama3-8B"),
                fake_row("Qwen2.5-Coder"),
                fake_row("Mistral-7B"),
            ],
            UseCase::Coding,
        );
        assert_eq!(s.selected, 2);
        // Fewer rows: cursor clamped to the last one.
        s.set_rows(vec![fake_row("Llama3-8B")], UseCase::Chat);
        assert_eq!(s.selected, 0);
    }

    #[test]
    fn set_rows_preserving_keeps_selection_by_name() {
        let mut s = state();
        s.selected = 2;
        let name = s.rows[2].model.name.clone();
        assert_eq!(name, "Mistral-7B");
        // Same three models, reversed order.
        s.set_rows_preserving(
            vec![
                fake_row("Mistral-7B"),
                fake_row("Qwen2.5-Coder"),
                fake_row("Llama3-8B"),
            ],
            UseCase::Coding,
        );
        assert_eq!(s.rows[s.selected].model.name, name);
        assert_eq!(s.selected, 0);
    }

    #[test]
    fn set_rows_preserving_clamps_when_model_gone() {
        let mut s = state();
        s.selected = 2;
        // The selected model is gone; only the first remains.
        s.set_rows_preserving(vec![fake_row("Llama3-8B")], UseCase::Coding);
        assert!(s.selected < s.rows.len());
        assert_eq!(s.selected, 0);
    }

    #[test]
    fn set_rows_preserving_reapplies_active_query() {
        let mut s = state();
        s.apply_search("qwen");
        assert_eq!(s.rows.len(), 1);
        // Fresh full rows arrive; the "qwen" filter must still apply.
        s.set_rows_preserving(
            vec![
                fake_row("Llama3-8B"),
                fake_row("Qwen2.5-Coder"),
                fake_row("Qwen2.5-7B"),
                fake_row("Mistral-7B"),
            ],
            UseCase::Coding,
        );
        assert_eq!(s.rows.len(), 2);
        assert!(
            s.rows
                .iter()
                .all(|r| r.model.name.to_lowercase().contains("qwen"))
        );
    }

    #[test]
    fn sync_status_default_is_idle() {
        let s = state();
        assert!(matches!(s.sync_status, SyncStatus::Idle));
    }

    #[test]
    fn running_then_done_and_failed() {
        assert!(matches!(
            SyncStatus::Idle.advance_running(),
            SyncStatus::Running
        ));
        let done = SyncStatus::Running.on_done();
        assert!(matches!(done, SyncStatus::Done { .. }));
        let failed = SyncStatus::Running.on_failed("boom".into());
        assert!(matches!(failed, SyncStatus::Failed(m) if m == "boom"));
    }

    #[test]
    fn q_quits_and_c_rescores_for_coding() {
        let mut s = state();
        assert!(matches!(
            s.handle_key(key(KeyCode::Char('c'))),
            Action::Rescore(UseCase::Coding)
        ));
        assert_eq!(s.use_case, UseCase::Coding);
        assert!(matches!(
            s.handle_key(key(KeyCode::Char('q'))),
            Action::Quit
        ));
    }
}
