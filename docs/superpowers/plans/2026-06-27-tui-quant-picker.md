# TUI Quant Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user pick the quantization in the TUI detail popup (list every quant with per-quant memory/tok-s/fit, arrow-key select, `x`/`s` launch the chosen one).

**Architecture:** A shared `variants_by_quality` ordering in core (reused by `best_variant`). The pure TUI state gains a `detail_variant` index, set on detail entry and moved by `↑`/`↓`; plan builders are parameterized by variant index so the active one (best in the list, chosen in detail) drives run/serve. The detail popup renders a per-quant table and the chart/hints for the selected quant.

**Tech Stack:** Rust (edition 2024), ratatui/crossterm, the existing `estimate_memory`/`estimate_speed` estimator.

---

## File structure

- `crates/paddock-core/src/score.rs` — `variants_by_quality`; `best_variant` refactored to reuse it. (Modify)
- `crates/paddock/src/tui/state.rs` — `detail_variant` field, detail entry/nav, variant-parameterized plan builders. (Modify)
- `crates/paddock/src/tui/draw.rs` — detail popup renders the quant table + chart/hints for the selected quant. (Modify)
- `README.md` — document the picker. (Modify)

---

## Task 1: `variants_by_quality` ordering in core

**Files:**
- Modify: `crates/paddock-core/src/score.rs`
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn variants_by_quality_orders_best_first() {
        let variants = vec![
            variant("Q2_K", 3.35, 8_000_000_000),
            variant("Q8_0", 8.5, 8_000_000_000),
            variant("Q4_K_M", 4.83, 8_000_000_000),
        ];
        let order = variants_by_quality(&variants);
        let labels: Vec<&str> = order.iter().map(|&i| variants[i].quant.as_str()).collect();
        assert_eq!(labels, vec!["Q8_0", "Q4_K_M", "Q2_K"]);
    }
```

(`variant(quant, bpw, params)` is the existing test helper in score.rs's test module.)

- [ ] **Step 2:** `cargo test -p paddock-core variants_by_quality_orders` → FAIL (not found).

- [ ] **Step 3: Implement `variants_by_quality`** in score.rs (above `best_variant`). Move the ranking `best_variant` currently inlines into this shared fn:

```rust
/// Variant indices ordered best-quality-first: by the QUANT_DESCENT ladder,
/// then higher bpw, then quant label. The order `best_variant` walks.
pub fn variants_by_quality(variants: &[ModelVariant]) -> Vec<usize> {
    let rank = |v: &ModelVariant| {
        QUANT_DESCENT
            .iter()
            .position(|q| *q == v.quant)
            .unwrap_or(QUANT_DESCENT.len())
    };
    let mut order: Vec<usize> = (0..variants.len()).collect();
    order.sort_by(|&a, &b| {
        rank(&variants[a])
            .cmp(&rank(&variants[b]))
            .then(
                variants[b]
                    .bpw
                    .partial_cmp(&variants[a].bpw)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then_with(|| variants[a].quant.cmp(&variants[b].quant))
    });
    order
}
```

- [ ] **Step 4: Refactor `best_variant` to reuse it.** Replace `best_variant`'s body (the inline `rank`/`ordered.sort_by(...)`) so it walks `variants_by_quality`:

```rust
pub fn best_variant<'a>(
    variants: &'a [ModelVariant],
    budget: &MemoryBudget,
) -> Option<&'a ModelVariant> {
    let order = variants_by_quality(variants);
    for accept in [
        FitVerdict::FitsGpu,
        FitVerdict::FitsWithSysctlTuning,
        FitVerdict::FitsRamOnly,
    ] {
        if let Some(&i) = order
            .iter()
            .find(|&&i| estimate_memory(&variants[i], DEFAULT_CONTEXT, budget).verdict == accept)
        {
            return Some(&variants[i]);
        }
    }
    None
}
```

(Behavior-preserving: same order, same fit-ladder walk. The existing `best_variant_*` tests are the safety net.)

- [ ] **Step 5:** `cargo test -p paddock-core score` → PASS (new test + the existing `best_variant_*`). `cargo build` clean.

- [ ] **Step 6: Commit**

```bash
git add crates/paddock-core/src/score.rs
git commit -m "feat(core): variants_by_quality ordering, reused by best_variant"
```

---

## Task 2: State — `detail_variant`, entry, nav, variant-parameterized plans

**Files:**
- Modify: `crates/paddock/src/tui/state.rs`
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Write the failing tests**

Add a multi-variant HuggingFace fixture to the test module (the existing `fake_row` is single-variant Ollama; the picker needs multiple GGUF quants whose `model_ref` carries the quant). Add near `fake_row`:

```rust
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
        ScoredModel { model, variant_idx: 0, memory, speed, score }
    }
```

Then the tests:

```rust
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
        // quality order: Q8_0(0), Q4_K_M(1), Q2_K(2)
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
        s.handle_key(key(KeyCode::Up));   // -> Q4_K_M
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
```

The test module already imports `CatalogModel, CatalogVariant, RuntimeKind, Source` (used by `fake_row`); confirm and add any missing (`quant_bpw` is referenced fully-qualified above).

- [ ] **Step 2:** `cargo test -p paddock detail_arrows_move_quant` → FAIL (no `detail_variant`).

- [ ] **Step 3: Add the field + helpers.**

Field on `TuiState` (after `last_sync` or near the other detail fields):

```rust
    /// Variant index (into the selected row's model.variants) chosen in the
    /// detail popup. Only meaningful in Mode::Detail.
    pub detail_variant: usize,
```

Init in `new`: `detail_variant: 0,`.

Add the active-variant resolver + a mover. Place near `plan_for_selected`:

```rust
    /// The variant index that run/serve should use: the chosen quant in Detail,
    /// otherwise the selected row's scored best.
    fn active_variant_idx(&self) -> usize {
        match self.mode {
            Mode::Detail => self.detail_variant,
            _ => self.rows.get(self.selected).map(|r| r.variant_idx).unwrap_or(0),
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
        let pos = order.iter().position(|&i| i == self.detail_variant).unwrap_or(0);
        let new_pos = (pos as isize + delta).clamp(0, order.len() as isize - 1) as usize;
        self.detail_variant = order[new_pos];
        self.detail_plan = self.plan_for_selected();
        self.detail_serve_plan = self.serve_plan_for_selected();
    }
```

- [ ] **Step 4: Make the plan builders use the active variant.** In `plan_for_selected` and `serve_plan_for_selected`, replace `let variant = &row.model.variants[row.variant_idx];` with:

```rust
        let idx = self.active_variant_idx().min(row.model.variants.len().saturating_sub(1));
        let variant = &row.model.variants[idx];
```

(Both functions; the rest of each is unchanged.)

- [ ] **Step 5: Set `detail_variant` on entry + wire arrows.** In the `Tab::Models` `K::Enter` arm, set the variant BEFORE computing plans, with `mode` already Detail so `active_variant_idx` resolves to it:

```rust
                        K::Enter => {
                            if let Some(row) = self.rows.get(self.selected) {
                                self.detail_variant = row.variant_idx;
                                self.mode = Mode::Detail;
                                self.detail_plan = self.plan_for_selected();
                                self.detail_serve_plan = self.serve_plan_for_selected();
                            }
                        }
```

In the `Mode::Detail` arm, add arrow handling:

```rust
            Mode::Detail => match key.code {
                K::Esc | K::Char('q') | K::Enter => self.close_detail(),
                K::Up => self.move_detail_variant(-1),
                K::Down => self.move_detail_variant(1),
                K::Char('x') => return self.run_selected(),
                K::Char('s') => return self.serve_selected(),
                _ => {}
            },
```

NOTE: `run_selected`/`serve_selected` call `plan_for_selected`/`serve_plan_for_selected`, which now use `active_variant_idx()` → `detail_variant` while in Detail. `serve_selected` builds the plan (Detail mode) THEN calls `close_detail()`, so the plan already captured the chosen quant. Good.

- [ ] **Step 6:** `cargo test -p paddock` → the 3 new tests PASS; existing tests still green. `cargo clippy --workspace` clean.

- [ ] **Step 7: Commit**

```bash
git add crates/paddock/src/tui/state.rs
git commit -m "feat(tui): detail_variant quant selection (entry, arrows, plans)"
```

---

## Task 3: Render the quant table in the detail popup

**Files:**
- Modify: `crates/paddock/src/tui/draw.rs`

- [ ] **Step 1: Make the detail render the selected variant + a quant table**

`draw_detail` and `draw_speed_chart` currently read `r.model.variants[r.variant_idx]`. Switch them to the chosen `state.detail_variant`, and replace the verbose score/memory/speed breakdown in `detail_lines` with a per-quant table.

Change `draw_detail` to pass the selected index + budget/bandwidth into `detail_lines` and the chart:

```rust
fn draw_detail(frame: &mut Frame, state: &TuiState, profile: &HardwareProfile) {
    let Some(r) = state.rows.get(state.selected) else {
        return;
    };
    let sel = state.detail_variant.min(r.model.variants.len().saturating_sub(1));
    let lines = detail_lines(
        r,
        sel,
        &state.budget,
        profile.bandwidth_gbps,
        state.detail_plan.as_ref(),
        state.detail_serve_plan.as_ref(),
    );
    let content_height = (lines.len() as u16)
        .saturating_add(SPEED_CHART_HEIGHT)
        .saturating_add(2);
    let area = centered(frame.area(), 70, content_height);
    frame.render_widget(Clear, area);
    let block = Block::bordered()
        .border_style(Style::new().fg(Color::DarkGray))
        .title(Span::styled(" detail ", Style::new().fg(ACCENT)));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [text_area, chart_area] =
        Layout::vertical([Constraint::Min(1), Constraint::Length(SPEED_CHART_HEIGHT)]).areas(inner);
    frame.render_widget(Paragraph::new(lines), text_area);
    draw_speed_chart(frame, chart_area, &r.model.to_model_variant(&r.model.variants[sel]), r.model.context_max, profile.bandwidth_gbps);
}
```

Change `draw_speed_chart`'s signature to take the chosen `ModelVariant` + `context_max` directly (it currently re-derives from `r.variant_idx`):

```rust
fn draw_speed_chart(frame: &mut Frame, area: Rect, v: &paddock_core::estimate::ModelVariant, context_max: u32, bandwidth_gbps: f64) {
    let max_ctx = context_max.clamp(8_192, 131_072);
    // ... rest unchanged, but it already uses `v` and `max_ctx` ...
}
```

(Remove the old first two lines of `draw_speed_chart` that built `v`/`max_ctx` from `r`; the body after that is unchanged.)

- [ ] **Step 2: Rewrite `detail_lines` to a quant table.** Replace the whole `detail_lines` fn with:

```rust
fn detail_lines<'a>(
    r: &'a ScoredModel,
    selected: usize,
    budget: &paddock_core::estimate::MemoryBudget,
    bandwidth_gbps: f64,
    plan: Option<&'a Result<RunPlan, String>>,
    serve_plan: Option<&'a Result<ServePlan, String>>,
) -> Vec<Line<'a>> {
    // draw.rs already imports estimate_speed + kv_cache_bytes at the top
    // (used by draw_speed_chart); only bring in the two not already in scope.
    use paddock_core::estimate::{DEFAULT_CONTEXT, estimate_memory};

    let mut lines = vec![
        Line::from(Span::styled(
            r.model.name.as_str(),
            Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
        )),
        Line::default(),
        Line::from(Span::styled(
            format!("  {:<14} {:>10} {:>7}  {}", "QUANT", "MEMORY", "TOK/S", "FIT"),
            Style::new().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
        )),
    ];

    for &i in &paddock_core::score::variants_by_quality(
        &r.model.variants.iter().map(|v| r.model.to_model_variant(v)).collect::<Vec<_>>(),
    ) {
        let v = r.model.to_model_variant(&r.model.variants[i]);
        let mem = estimate_memory(&v, DEFAULT_CONTEXT, budget);
        let tps = estimate_speed(&v, bandwidth_gbps, kv_cache_bytes(&v, DEFAULT_CONTEXT)).generation_tps;
        let marker = if i == selected { "> " } else { "  " };
        let row_style = if i == selected {
            Style::new().fg(Color::White).bg(ACCENT_DEEP)
        } else {
            Style::new().fg(Color::Gray)
        };
        lines.push(Line::from(Span::styled(
            format!(
                "{marker}{:<14} {:>10} {:>7}  {}",
                r.model.variants[i].quant,
                gib(mem.total_bytes),
                format!("{:.0}", tps),
                verdict_label(mem.verdict),
            ),
            row_style,
        )));
    }

    lines.push(Line::default());
    match plan {
        Some(Ok(p)) => lines.push(Line::from(vec![
            Span::styled("  x to run  ", Style::new().fg(Color::DarkGray)),
            Span::styled(p.display(), Style::new().fg(Color::Gray)),
        ])),
        Some(Err(e)) => lines.push(Line::from(Span::styled(
            format!("  run unavailable: {e}"),
            Style::new().fg(ACCENT),
        ))),
        None => {}
    }
    match serve_plan {
        Some(Ok(sp)) => lines.push(Line::from(vec![
            Span::styled("  s to serve on ", Style::new().fg(Color::DarkGray)),
            Span::styled(sp.endpoint.clone(), Style::new().fg(ACCENT)),
        ])),
        Some(Err(e)) => lines.push(Line::from(Span::styled(
            format!("  serve unavailable: {e}"),
            Style::new().fg(ACCENT),
        ))),
        None => {}
    }
    lines.push(Line::from(Span::styled(
        "  up/down pick quant · esc back",
        Style::new().fg(Color::DarkGray),
    )));
    lines
}
```

NOTE: the `budget` param is typed `&paddock_core::estimate::MemoryBudget` (fully qualified, no import needed). `gib`, `verdict_label`, `ACCENT_DEEP`, `RunPlan`, `ServePlan` and `estimate_speed`/`kv_cache_bytes` are already in scope in draw.rs; the inner `use` adds only `DEFAULT_CONTEXT`/`estimate_memory`. If the compiler reports any duplicate import, drop it from the inner `use`.

- [ ] **Step 3: Build + clippy**

Run: `cargo build && cargo test -p paddock && cargo clippy --workspace`
Expected: green/clean. Fix any unused imports the rewrite leaves (the old `detail_lines` used `r.score`/`r.memory`/`r.speed`/`section`/`kv` helpers now gone — ensure no leftover references; `r.speed`/`r.memory` may now be unused on `ScoredModel` but that struct is used elsewhere, so no warning). No em-dash (`—`); use `·`/`-`.

- [ ] **Step 4: Commit**

```bash
git add crates/paddock/src/tui/draw.rs
git commit -m "feat(tui): detail popup shows a per-quant table for the picker"
```

---

## Task 4: Docs + live smoke test

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the picker**

In the README TUI section, after the servers-tab paragraph (search "Press `Tab` to switch to the servers view"), add:

```text
Press `enter` on a model to open its detail popup, which lists every quantization it ships with their memory, tok/s and fit; use the arrow keys to pick a smaller quant for more speed and less memory, then `x` to run or `s` to serve the chosen one. Without opening the detail, `x`/`s` use the best quant that fits.
```

(No em-dash.)

- [ ] **Step 2: Commit docs**

```bash
git add README.md
git commit -m "docs: document the TUI quant picker in the detail popup"
```

- [ ] **Step 3: Live smoke test (manual)**

```bash
cargo run   # models tab
```
Verify:
1. `enter` on a model with multiple quants opens the detail; the quant table lists all quants with MEMORY/TOK/S/FIT, the best one highlighted.
2. `↑`/`↓` move the highlight; the chart and the `x`/`s` hint update to the selected quant.
3. `s` on a smaller quant serves THAT quant (check the spawned command / `paddock ps` shows the chosen quant's model_ref).
4. `esc` closes; from the list, `x`/`s` still use the best quant (quick path unchanged).
5. A single-quant model: the table shows one row; arrows do nothing.

- [ ] **Step 4: Final verification**

Run: `cargo build && cargo test && cargo clippy --workspace --all-targets`
Expected: all green/clean.

---

## Notes for the implementer

- Core ordering (Task 1) and the pure state machine (Task 2) are unit-tested. The detail rendering (Task 3) is visual, verified by the live smoke test.
- Reuse, don't reinvent: the chosen quant flows through the existing `plan_run`/`plan_serve` + `Action::Run`/`Action::Serve` paths via `active_variant_idx`.
- Do NOT add a CLI `--quant` flag, global-list quant cycling, or per-quant columns in the main table (out of scope).
- Em-dash `—` is banned project-wide; use `-` or `·`.
