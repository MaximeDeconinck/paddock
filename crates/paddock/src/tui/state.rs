//! Pure TUI state machine - no terminal IO, fully unit-testable.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use paddock_core::estimate::{MemoryBudget, resolve_ctx};
use paddock_core::hardware::RuntimesStatus;
use paddock_core::runtime::{RunPlan, ServePlan, plan_run, plan_serve};
use paddock_core::score::UseCase;
use paddock_core::serving::{AvailableRow, ServerRow, StopHandle};

use crate::app::ScoredModel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    List,
    Detail,
    Search { query: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Models,
    Servers,
}

enum SelectedRow<'a> {
    Running(&'a ServerRow),
    Available(&'a AvailableRow),
    None,
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
    /// Stop the selected server (SIGTERM + unregister, or `ollama stop`),
    /// then refresh.
    StopServer(StopHandle),
    /// Copy this endpoint URL to the system clipboard.
    CopyEndpoint(String),
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
    /// Memory budget for this machine; used to auto-size run/serve context.
    pub budget: MemoryBudget,
    /// Last plan_run failure, surfaced in the footer instead of crashing.
    pub last_error: Option<String>,
    /// Run plan for the selected row, computed once on Detail entry so the
    /// render path never calls plan_run. Cleared when Detail closes.
    pub detail_plan: Option<Result<RunPlan, String>>,
    /// Serve plan for the selected row, computed alongside `detail_plan` so
    /// the detail popup can show the endpoint without calling plan_serve.
    pub detail_serve_plan: Option<Result<ServePlan, String>>,
    /// Variant index (into the selected row's model.variants) chosen in the
    /// detail popup. Only meaningful in Mode::Detail.
    pub detail_variant: usize,
    /// Background catalog-sync status, shown in the footer.
    pub sync_status: SyncStatus,
    /// Spinner animation frame, advanced once per event-loop tick.
    // Advanced by the event loop; read by the footer renderer.
    pub tick: u64,
    /// Catalog `last_sync` epoch (seconds), read from the DB at launch and
    /// after each background sync; the footer shows it as "synced Xm ago".
    pub last_sync: Option<i64>,
    /// Active top-level tab. Search/Detail overlays only apply on `Models`.
    pub tab: Tab,
    /// Live servers shown on the Servers tab; refreshed by the background task.
    pub servers: Vec<ServerRow>,
    /// Locally-available (not running) models shown greyed below the running ones.
    pub available: Vec<AvailableRow>,
    /// Cursor within `servers`.
    pub server_selected: usize,
}

impl TuiState {
    pub fn new(
        rows: Vec<ScoredModel>,
        use_case: UseCase,
        runtimes: RuntimesStatus,
        budget: MemoryBudget,
    ) -> Self {
        Self {
            all_rows: rows.clone(),
            rows,
            selected: 0,
            use_case,
            mode: Mode::List,
            query: String::new(),
            runtimes,
            budget,
            last_error: None,
            detail_plan: None,
            detail_serve_plan: None,
            detail_variant: 0,
            sync_status: SyncStatus::Idle,
            tick: 0,
            last_sync: None,
            tab: Tab::Models,
            servers: Vec::new(),
            available: Vec::new(),
            server_selected: 0,
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

    /// Replace both groups. Cursor preserved by identity (model name, prefixed
    /// `r:`/`a:` per group) then clamped to the combined length. The model-name
    /// key is collision-free because `list_available` excludes running models
    /// from the available group, so no model appears in both at once.
    pub fn set_snapshot(&mut self, running: Vec<ServerRow>, available: Vec<AvailableRow>) {
        let prev = self.selected_combined_key();
        self.servers = running;
        self.available = available;
        let len = self.servers.len() + self.available.len();
        self.server_selected = prev
            .and_then(|k| self.combined_keys().position(|c| c == k))
            .unwrap_or(0)
            .min(len.saturating_sub(1));
    }

    /// Stable identity per combined row for cursor preservation across refreshes.
    fn combined_keys(&self) -> impl Iterator<Item = String> + '_ {
        let running = self.servers.iter().map(|r| format!("r:{}", r.model));
        let avail = self.available.iter().map(|a| format!("a:{}", a.model));
        running.chain(avail)
    }
    fn selected_combined_key(&self) -> Option<String> {
        self.combined_keys().nth(self.server_selected)
    }

    /// What the servers-tab cursor points at.
    fn selected_row(&self) -> SelectedRow<'_> {
        let n = self.servers.len();
        if self.server_selected < n {
            SelectedRow::Running(&self.servers[self.server_selected])
        } else if let Some(a) = self.available.get(self.server_selected - n) {
            SelectedRow::Available(a)
        } else {
            // Unreachable in practice: every mutator clamps server_selected to
            // the combined length. Kept as a total fallback.
            SelectedRow::None
        }
    }

    /// Drop the currently-selected server row (optimistic update after a stop;
    /// the next background refresh reconciles) and clamp the cursor.
    pub fn remove_selected(&mut self) {
        if self.server_selected < self.servers.len() {
            self.servers.remove(self.server_selected);
        }
        let len = self.servers.len() + self.available.len();
        self.server_selected = self.server_selected.min(len.saturating_sub(1));
    }

    #[cfg(test)]
    pub fn selected_server(&self) -> Option<&ServerRow> {
        self.servers.get(self.server_selected)
    }

    /// Close the Detail overlay back to the List view, clearing its cached
    /// plans. Used both by the Detail dismiss keys and when serving from Detail
    /// (the serve switches to the Servers tab, so the popup must not linger).
    pub fn close_detail(&mut self) {
        self.mode = Mode::List;
        self.detail_plan = None;
        self.detail_serve_plan = None;
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Action {
        use KeyCode as K;
        self.last_error = None;
        // Ctrl-C quits from any mode - never swallowed by search input.
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
                K::Esc | K::Char('q') | K::Enter => self.close_detail(),
                K::Up => self.move_detail_variant(-1),
                K::Down => self.move_detail_variant(1),
                K::Char('x') => return self.run_selected(),
                K::Char('s') => return self.serve_selected(),
                _ => {}
            },
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
                        K::Up => self.selected = self.selected.saturating_sub(1),
                        K::Down => {
                            self.selected =
                                (self.selected + 1).min(self.rows.len().saturating_sub(1));
                        }
                        K::Enter => {
                            if let Some(idx) =
                                self.rows.get(self.selected).map(|r| r.variant_idx)
                            {
                                self.detail_variant = idx;
                                self.mode = Mode::Detail;
                                self.detail_plan = self.plan_for_selected();
                                self.detail_serve_plan = self.serve_plan_for_selected();
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
                        K::Up => {
                            self.server_selected = self.server_selected.saturating_sub(1)
                        }
                        K::Down => {
                            let len = self.servers.len() + self.available.len();
                            self.server_selected =
                                (self.server_selected + 1).min(len.saturating_sub(1));
                        }
                        K::Enter => {
                            if let SelectedRow::Available(a) = self.selected_row() {
                                return Action::Serve(a.plan.clone());
                            }
                        }
                        K::Char('x') => {
                            if let SelectedRow::Running(r) = self.selected_row() {
                                return Action::StopServer(r.stop.clone());
                            }
                        }
                        K::Char('c') => {
                            if let SelectedRow::Running(r) = self.selected_row() {
                                return Action::CopyEndpoint(r.openai_url.clone());
                            }
                        }
                        _ => {}
                    },
                }
            }
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

    /// The variant index that run/serve should use: the chosen quant in Detail,
    /// otherwise the selected row's scored best.
    fn active_variant_idx(&self) -> usize {
        match self.mode {
            Mode::Detail => self.detail_variant,
            _ => self
                .rows
                .get(self.selected)
                .map(|r| r.variant_idx)
                .unwrap_or(0),
        }
    }

    /// Move the detail quant selection by `delta` within the quality order, then
    /// recompute the cached detail plans for the new quant.
    fn move_detail_variant(&mut self, delta: isize) {
        let order = match self.rows.get(self.selected) {
            Some(row) => {
                let mvs: Vec<_> = row
                    .model
                    .variants
                    .iter()
                    .map(|v| row.model.to_model_variant(v))
                    .collect();
                paddock_core::score::variants_by_quality(&mvs)
            }
            None => return,
        };
        if order.is_empty() {
            return;
        }
        let pos = order
            .iter()
            .position(|&i| i == self.detail_variant)
            .unwrap_or(0);
        let new_pos = (pos as isize + delta).clamp(0, order.len() as isize - 1) as usize;
        self.detail_variant = order[new_pos];
        self.detail_plan = self.plan_for_selected();
        self.detail_serve_plan = self.serve_plan_for_selected();
    }

    /// Single source for run-plan computation (Detail entry and `x` both use
    /// it), so the render path never calls plan_run.
    fn plan_for_selected(&self) -> Option<Result<RunPlan, String>> {
        let row = self.rows.get(self.selected)?;
        let idx = self
            .active_variant_idx()
            .min(row.model.variants.len().saturating_sub(1));
        let variant = &row.model.variants[idx];
        // TUI has no flag surface, so ctx is auto-sized against the memory budget.
        let mv = row.model.to_model_variant(variant);
        let ctx = resolve_ctx(None, &mv, &self.budget, row.model.context_max);
        Some(plan_run(&row.model, variant, &self.runtimes, Some(ctx)).map_err(|e| e.to_string()))
    }

    /// Single source for serve-plan computation (Detail entry and `s` both
    /// use it), so the render path never calls plan_serve. Default port: the
    /// TUI has no flag surface, so `port` is always None.
    fn serve_plan_for_selected(&self) -> Option<Result<ServePlan, String>> {
        let row = self.rows.get(self.selected)?;
        let idx = self
            .active_variant_idx()
            .min(row.model.variants.len().saturating_sub(1));
        let variant = &row.model.variants[idx];
        let mv = row.model.to_model_variant(variant);
        let ctx = resolve_ctx(None, &mv, &self.budget, row.model.context_max);
        Some(
            plan_serve(&row.model, variant, &self.runtimes, None, Some(ctx))
                .map_err(|e| e.to_string()),
        )
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
            Some(Ok(plan)) => {
                // Serving switches to the Servers tab, so a Detail popup opened
                // on this row must not linger over it.
                self.close_detail();
                Action::Serve(plan)
            }
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

    /// Same as `fake_row` but with a parameterized `context_max`, so tests can
    /// build a model whose auto-sized context exceeds the 8192 default.
    fn fake_row_ctx(name: &str, context_max: u32) -> ScoredModel {
        let mut row = fake_row(name);
        row.model.context_max = context_max;
        row
    }

    /// Multi-variant HuggingFace (GGUF) fixture: one CatalogVariant per quant,
    /// so detail-mode quant navigation has something to move across.
    fn fake_hf_row(name: &str, quants: &[&str]) -> ScoredModel {
        let variants: Vec<CatalogVariant> = quants
            .iter()
            .map(|q| CatalogVariant {
                quant: (*q).into(),
                bpw: paddock_core::catalog::quant_bpw(q).unwrap_or(4.0),
                file_size_bytes: None,
                layers: 32,
                kv_heads: 8,
                head_dim: 128,
                embedding_dim: 4096,
                runtime_compat: vec![RuntimeKind::LlamaCpp], // GGUF, not Ollama -> llama-cli/-server
                source_tag: None,
            })
            .collect();
        let model = CatalogModel {
            id: 0,
            name: name.to_string(),
            family: None,
            source: Source::HuggingFace,
            repo: Some(format!("bartowski/{name}")),
            params_total: 8_000_000_000,
            params_active: 8_000_000_000,
            architecture: None,
            context_max: 8192,
            released_at: None,
            released_approx: false,
            variants,
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

    use paddock_core::serving::{AvailableRow, ServerRow, StopHandle};

    fn avail(model: &str) -> AvailableRow {
        AvailableRow {
            model: model.into(),
            runtime: RuntimeKind::MlxLm,
            size_bytes: None,
            last_served_at: Some(0),
            plan: paddock_core::runtime::ServePlan {
                server_argv: Some(vec![
                    "mlx_lm.server".into(),
                    "--model".into(),
                    model.into(),
                    "--port".into(),
                    "8080".into(),
                ]),
                pre_steps: vec![],
                endpoint: "http://127.0.0.1:8080".into(),
                openai_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
                model_ref: model.into(),
                ready_path: "/v1/models".into(),
                install: None,
                port_ignored: false,
                runtime: RuntimeKind::MlxLm,
                ctx: 0,
                port: Some(8080),
            },
        }
    }

    fn srv(pid: u32, model: &str, port: u16) -> ServerRow {
        ServerRow {
            model: model.into(),
            runtime: RuntimeKind::LlamaCpp,
            endpoint: format!("http://127.0.0.1:{port}"),
            openai_url: format!("http://127.0.0.1:{port}/v1/chat/completions"),
            ctx: Some(8192),
            started_at: Some(0),
            stop: StopHandle::Pid(pid),
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
            MemoryBudget {
                gpu_effective_bytes: 24 * (1u64 << 30),
                ram_total_bytes: 32 * (1u64 << 30),
            },
        )
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn arrows_move_selection_within_bounds() {
        let mut s = state();
        assert_eq!(s.selected, 0);
        s.handle_key(key(KeyCode::Up)); // clamped at top
        assert_eq!(s.selected, 0);
        s.handle_key(key(KeyCode::Down));
        assert_eq!(s.selected, 1);
        s.handle_key(key(KeyCode::Down));
        s.handle_key(key(KeyCode::Down)); // clamped at bottom
        s.handle_key(key(KeyCode::Down));
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
        // Search - and the 'c' must not land in the query
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
        s.handle_key(key(KeyCode::Down));
        s.handle_key(key(KeyCode::Down));
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
        s.set_snapshot(vec![srv(1, "a", 8080), srv(2, "b", 8081)], vec![]);
        s.handle_key(key(KeyCode::Tab)); // -> Servers
        assert_eq!(s.server_selected, 0);
        s.handle_key(key(KeyCode::Up)); // clamped at top
        assert_eq!(s.server_selected, 0);
        s.handle_key(key(KeyCode::Down));
        assert_eq!(s.server_selected, 1);
        s.handle_key(key(KeyCode::Down)); // clamped at bottom
        assert_eq!(s.server_selected, 1);
    }

    #[test]
    fn set_snapshot_preserves_selection_by_model() {
        let mut s = state();
        s.set_snapshot(vec![srv(10, "a", 8080), srv(20, "b", 8081)], vec![]);
        s.tab = Tab::Servers;
        s.server_selected = 1; // model "b"
        s.set_snapshot(vec![srv(20, "b", 8081)], vec![]); // model "a" dropped
        // The cursor follows model "b" to its new index, keyed on the model name.
        assert_eq!(s.server_selected, 0);
        assert_eq!(s.selected_server().unwrap().model, "b");
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

    #[test]
    fn x_on_servers_tab_returns_stop_action() {
        let mut s = state();
        s.set_snapshot(vec![srv(42, "qwen", 8080)], vec![]);
        s.tab = Tab::Servers;
        match s.handle_key(key(KeyCode::Char('x'))) {
            Action::StopServer(h) => assert_eq!(h, StopHandle::Pid(42)),
            other => panic!("expected StopServer, got {other:?}"),
        }
    }

    #[test]
    fn c_on_servers_tab_returns_copy_action() {
        let mut s = state();
        s.set_snapshot(vec![srv(42, "qwen", 8080)], vec![]);
        s.tab = Tab::Servers;
        match s.handle_key(key(KeyCode::Char('c'))) {
            Action::CopyEndpoint(url) => {
                assert_eq!(url, "http://127.0.0.1:8080/v1/chat/completions")
            }
            other => panic!("expected CopyEndpoint, got {other:?}"),
        }
    }

    #[test]
    fn remove_selected_drops_and_clamps() {
        let mut s = state();
        s.set_snapshot(
            vec![srv(1, "a", 8080), srv(2, "b", 8081), srv(3, "c", 8082)],
            vec![],
        );
        s.tab = Tab::Servers;
        s.server_selected = 2; // last (model "c")
        s.remove_selected();
        assert_eq!(s.servers.len(), 2);
        assert_eq!(s.server_selected, 1); // clamped from 2 to last valid index
        // The dropped row is gone.
        assert!(s.servers.iter().all(|r| r.model != "c"));
    }

    #[test]
    fn remove_selected_on_empty_is_noop() {
        let mut s = state();
        s.remove_selected();
        assert_eq!(s.servers.len(), 0);
        assert_eq!(s.server_selected, 0);
    }

    #[test]
    fn tui_serve_plan_auto_sizes_context() {
        // A roomy budget + large context_max should auto-size well above the 8192 default.
        let mut s = state();
        s.rows = vec![fake_row_ctx("Big", 32_768)];
        s.all_rows = s.rows.clone();
        s.selected = 0;
        let plan = s.serve_plan_for_selected().unwrap().unwrap();
        assert!(plan.ctx > 8192, "expected auto-sized ctx, got {}", plan.ctx);
    }

    #[test]
    fn x_on_empty_servers_tab_is_noop() {
        let mut s = state();
        s.tab = Tab::Servers;
        assert!(matches!(s.handle_key(key(KeyCode::Char('x'))), Action::None));
    }

    #[test]
    fn serving_from_detail_closes_the_popup() {
        let mut s = state();
        s.handle_key(key(KeyCode::Enter)); // open Detail on the selected model
        assert_eq!(s.mode, Mode::Detail);
        let action = s.handle_key(key(KeyCode::Char('s'))); // serve from Detail
        assert!(matches!(action, Action::Serve(_)));
        assert_eq!(s.mode, Mode::List, "Detail popup must close when serving");
        assert!(s.detail_plan.is_none());
        assert!(s.detail_serve_plan.is_none());
    }

    #[test]
    fn navigation_spans_running_then_available() {
        let mut s = state();
        s.set_snapshot(vec![srv(1, "run-a", 8080)], vec![avail("avail-b"), avail("avail-c")]);
        s.tab = Tab::Servers;
        assert_eq!(s.server_selected, 0);
        s.handle_key(key(KeyCode::Down));
        s.handle_key(key(KeyCode::Down));
        assert_eq!(s.server_selected, 2);
        s.handle_key(key(KeyCode::Down)); // clamped
        assert_eq!(s.server_selected, 2);
    }

    #[test]
    fn enter_on_available_serves_its_plan() {
        let mut s = state();
        s.set_snapshot(vec![], vec![avail("avail-b")]);
        s.tab = Tab::Servers;
        match s.handle_key(key(KeyCode::Enter)) {
            Action::Serve(p) => assert_eq!(p.model_ref, "avail-b"),
            other => panic!("expected Serve, got {other:?}"),
        }
    }

    #[test]
    fn enter_on_running_is_noop() {
        let mut s = state();
        s.set_snapshot(vec![srv(1, "run-a", 8080)], vec![]);
        s.tab = Tab::Servers;
        assert!(matches!(s.handle_key(key(KeyCode::Enter)), Action::None));
    }

    #[test]
    fn x_on_available_is_noop() {
        let mut s = state();
        s.set_snapshot(vec![], vec![avail("avail-b")]);
        s.tab = Tab::Servers;
        assert!(matches!(s.handle_key(key(KeyCode::Char('x'))), Action::None));
    }

    #[test]
    fn entering_detail_sets_detail_variant_to_scored_best() {
        let mut s = state();
        s.rows = vec![fake_row("Llama3-8B")];
        s.all_rows = s.rows.clone();
        s.selected = 0;
        s.handle_key(key(KeyCode::Enter));
        assert_eq!(s.mode, Mode::Detail);
        assert_eq!(s.detail_variant, s.rows[0].variant_idx);
    }

    #[test]
    fn detail_arrows_move_quant_within_quality_order() {
        let mut s = state();
        s.rows = vec![fake_hf_row("M", &["Q8_0", "Q4_K_M", "Q2_K"])];
        s.all_rows = s.rows.clone();
        s.selected = 0;
        s.handle_key(key(KeyCode::Enter)); // detail_variant = variant_idx (0 = Q8_0)
        assert_eq!(s.rows[0].model.variants[s.detail_variant].quant, "Q8_0");
        s.handle_key(key(KeyCode::Down)); // -> Q4_K_M
        assert_eq!(s.rows[0].model.variants[s.detail_variant].quant, "Q4_K_M");
        s.handle_key(key(KeyCode::Down)); // -> Q2_K
        s.handle_key(key(KeyCode::Down)); // clamp at smallest
        assert_eq!(s.rows[0].model.variants[s.detail_variant].quant, "Q2_K");
        s.handle_key(key(KeyCode::Up)); // -> Q4_K_M
        assert_eq!(s.rows[0].model.variants[s.detail_variant].quant, "Q4_K_M");
    }

    #[test]
    fn serve_from_detail_uses_the_chosen_quant() {
        let mut s = state();
        s.rows = vec![fake_hf_row("M", &["Q8_0", "Q4_K_M", "Q2_K"])];
        s.all_rows = s.rows.clone();
        s.selected = 0;
        s.handle_key(key(KeyCode::Enter));
        s.handle_key(key(KeyCode::Down)); // choose Q4_K_M
        match s.handle_key(key(KeyCode::Char('s'))) {
            Action::Serve(p) => assert!(
                p.model_ref.contains("Q4_K_M"),
                "served plan should carry the chosen quant, got {}",
                p.model_ref
            ),
            other => panic!("expected Serve, got {other:?}"),
        }
    }
}
