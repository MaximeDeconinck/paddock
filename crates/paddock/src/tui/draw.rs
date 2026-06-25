//! Pure rendering — reads state, never mutates it, no IO.
//! Palette: DarkGray/Gray/White + a single deep-blue accent.

use paddock_core::estimate::{FitVerdict, estimate_speed, kv_cache_bytes};
use paddock_core::hardware::{HardwareProfile, RuntimeStatus};
use paddock_core::runtime::{RunPlan, ServePlan};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Cell, Chart, Clear, Dataset, GraphType};
use ratatui::widgets::{Paragraph, Row, Table, TableState};

use crate::app::ScoredModel;
use crate::output::{age_label, gib, verdict_label};
use crate::tui::state::{Mode, SyncStatus, Tab, TuiState, use_case_label};

/// Accent palette sampled from the paddock wordmark (deep indigo banner).
/// ACCENT for accented text on dark terminals (readable royal blue),
/// ACCENT_DEEP as the selection background (white foreground on top).
const ACCENT: Color = Color::Rgb(92, 102, 255);
const ACCENT_DEEP: Color = Color::Rgb(26, 26, 110);

pub fn draw(frame: &mut Frame, state: &TuiState, profile: &HardwareProfile) {
    // Breathing room against the terminal edges, like the box's own padding.
    let [header, table, footer] = Layout::vertical([
        Constraint::Length(7),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .horizontal_margin(2)
    .vertical_margin(1)
    .areas(frame.area());
    draw_header(frame, header, profile, state.tab);
    match state.tab {
        Tab::Models => draw_table(frame, table, state),
        Tab::Servers => draw_servers(frame, table, state),
    }
    draw_footer(frame, footer, state);
    if state.mode == Mode::Detail {
        draw_detail(frame, state, profile);
    }
}

/// Block-letter wordmark drawn in the header, llama.cpp-style.
const WORDMARK: [&str; 5] = [
    "█████▄   ▄███▄   ▄▄▄██   ▄▄▄██   ▄██▄    ▄███▄  ██  ██",
    "██  ██      ██  ██  ██  ██  ██  ██  ██  ██      ██ ██",
    "██  ██   ▄████  ██  ██  ██  ██  ██  ██  ██      ████",
    "█████▀  ██  ██  ██  ██  ██  ██  ██  ██  ██      ██ ██",
    "██       ▀▀▀██   ▀▀▀██   ▀▀▀██   ▀██▀    ▀███▀  ██  ██",
];
/// Display width of the widest wordmark row + a 2-col right margin.
const WORDMARK_WIDTH: u16 = 57;

fn draw_header(frame: &mut Frame, area: Rect, p: &HardwareProfile, tab: Tab) {
    // Active-tab indicator, top-right corner above the machine box.
    let (models_style, servers_style) = match tab {
        Tab::Models => (
            Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
            Style::new().fg(Color::DarkGray),
        ),
        Tab::Servers => (
            Style::new().fg(Color::DarkGray),
            Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
    };
    let tabs = Line::from(vec![
        Span::styled("models", models_style),
        Span::styled("  ", Style::new()),
        Span::styled("servers", servers_style),
    ]);
    let tab_row = Rect {
        x: area.x,
        y: area.y,
        width: area.width,
        height: 1,
    };
    frame.render_widget(Paragraph::new(tabs).alignment(Alignment::Right), tab_row);

    // Wide terminals: wordmark on the left, machine box to its right.
    // Narrow ones: machine box only (it carries the " paddock " title then).
    let min_box_width = 46;
    let (mark_area, box_area) = if area.width >= WORDMARK_WIDTH + min_box_width {
        let [m, b] = Layout::horizontal([Constraint::Length(WORDMARK_WIDTH), Constraint::Min(1)])
            .areas(area);
        (Some(m), b)
    } else {
        (None, area)
    };
    if let Some(m) = mark_area {
        // Top blank line: the 5-row wordmark sits level with the box's
        // content rows (the box spends its first row on the border).
        let mut mark = vec![Line::default()];
        mark.extend(
            WORDMARK
                .iter()
                .map(|l| Line::from(Span::styled(*l, Style::new().fg(ACCENT)))),
        );
        frame.render_widget(Paragraph::new(mark), m);
    }
    draw_machine_box(frame, box_area, p, mark_area.is_none());
}

fn draw_machine_box(frame: &mut Frame, area: Rect, p: &HardwareProfile, titled: bool) {
    let label = |s: &str| Span::styled(format!("{s:<11}"), Style::new().fg(Color::DarkGray));
    let value = |s: String| Span::styled(s, Style::new().fg(Color::Gray));
    let gpu_line = match p.gpu.metal_limit_bytes {
        Some(b) => format!("{} (Metal working set)", gib(b)),
        None => format!(
            "{} (fallback: 75% of RAM)",
            gib(p.gpu.effective_limit_bytes)
        ),
    };
    let bandwidth = format!(
        "{:.0} GB/s{}",
        p.bandwidth_gbps,
        if p.bandwidth_estimated {
            " (estimated)"
        } else {
            ""
        }
    );
    let rt = |s: &RuntimeStatus, name: &str| {
        if s.installed {
            format!(
                "{name} {}{}",
                s.version.as_deref().unwrap_or("?"),
                if s.running { " (running)" } else { "" }
            )
        } else {
            format!("{name} not installed")
        }
    };
    let runtimes = format!(
        "{} · {} · {}",
        rt(&p.runtimes.ollama, "ollama"),
        rt(&p.runtimes.llama_cpp, "llama.cpp"),
        rt(&p.runtimes.mlx, "mlx-lm")
    );
    let lines = vec![
        Line::from(vec![label("chip"), value(p.chip_name.clone())]),
        Line::from(vec![label("ram"), value(gib(p.ram_total_bytes))]),
        Line::from(vec![label("gpu limit"), value(gpu_line)]),
        Line::from(vec![label("bandwidth"), value(bandwidth)]),
        Line::from(vec![label("runtimes"), value(runtimes)]),
    ];
    let mut block = Block::bordered().border_style(Style::new().fg(Color::DarkGray));
    if titled {
        block = block.title(Span::styled(
            " paddock ",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    }
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn verdict_style(v: FitVerdict) -> Style {
    match v {
        FitVerdict::FitsGpu => Style::new().fg(Color::White),
        FitVerdict::FitsWithSysctlTuning => {
            Style::new().fg(Color::Gray).add_modifier(Modifier::ITALIC)
        }
        FitVerdict::FitsRamOnly | FitVerdict::DoesNotFit => Style::new().fg(Color::DarkGray),
    }
}

fn draw_table(frame: &mut Frame, area: Rect, state: &TuiState) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let header = Row::new(["MODEL", "AGE", "QUANT", "MEMORY", "TOK/S", "FIT", "SCORE"]).style(
        Style::new()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );
    let rows = state.rows.iter().map(|r| {
        let v = &r.model.variants[r.variant_idx];
        Row::new(vec![
            Cell::from(r.model.name.clone()),
            Cell::from(age_label(r.model.released_at, r.model.released_approx, now)),
            Cell::from(v.quant.clone()),
            Cell::from(gib(r.memory.total_bytes)),
            Cell::from(format!("{:.0}", r.speed.generation_tps)),
            Cell::from(Span::styled(
                verdict_label(r.memory.verdict),
                verdict_style(r.memory.verdict),
            )),
            Cell::from(format!("{:.0}", r.score.total)),
        ])
        .style(Style::new().fg(Color::Gray))
    });
    let table = Table::new(
        rows,
        [
            Constraint::Min(24),
            Constraint::Length(6),
            Constraint::Length(9),
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Length(12),
            Constraint::Length(6),
        ],
    )
    .header(header)
    .row_highlight_style(Style::new().fg(Color::White).bg(ACCENT_DEEP));
    // Fresh TableState each frame: draw stays pure, ratatui recomputes the
    // scroll offset so the selected row is always visible.
    let mut ts = TableState::default().with_selected(Some(state.selected));
    frame.render_stateful_widget(table, area, &mut ts);
}

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
            Cell::from(crate::output::truncate(&r.model_ref, 28)),
            Cell::from(crate::output::runtime_label(r.runtime)),
            Cell::from(crate::output::truncate(&r.endpoint, 26)),
            Cell::from(r.ctx.to_string()),
            Cell::from(crate::output::uptime_label(r.started_at)),
            Cell::from(r.pid.to_string()),
        ])
        .style(Style::new().fg(Color::Gray))
    });
    let table = Table::new(
        rows,
        [
            Constraint::Min(20),
            Constraint::Length(10),
            Constraint::Length(26),
            Constraint::Length(7),
            Constraint::Length(8),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .row_highlight_style(Style::new().fg(Color::White).bg(ACCENT_DEEP));
    let mut ts = TableState::default().with_selected(Some(state.server_selected));
    frame.render_stateful_widget(table, area, &mut ts);
}

fn draw_footer(frame: &mut Frame, area: Rect, state: &TuiState) {
    if state.tab == Tab::Servers {
        let line = "j/k move · x stop · c copy endpoint · tab models · q quit";
        frame.render_widget(
            Paragraph::new(Span::styled(line, Style::new().fg(Color::DarkGray))),
            area,
        );
        return;
    }
    const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let sync_seg = match &state.sync_status {
        SyncStatus::Running => {
            let frame = SPINNER[(state.tick as usize) % SPINNER.len()];
            Some(format!("{frame} syncing…"))
        }
        SyncStatus::Done { at } if at.elapsed().as_secs() < 5 => Some("catalog updated".into()),
        // Idle / flash expired / failed: show how old the catalog is, when known.
        // draw_table already reads the wall clock here for the AGE column.
        _ => state.last_sync.map(|ts| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            format!("synced {} ago", crate::output::humanize_since(now - ts))
        }),
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
            "↑↓ move · enter detail · x run · s serve · / search · g/c/r/h use-case · R sync · tab servers · q quit",
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

/// Rows reserved for the tok/s-vs-context chart inside the detail popup.
const SPEED_CHART_HEIGHT: u16 = 9;

fn draw_detail(frame: &mut Frame, state: &TuiState, profile: &HardwareProfile) {
    let Some(r) = state.rows.get(state.selected) else {
        return;
    };
    let lines = detail_lines(
        r,
        state.detail_plan.as_ref(),
        state.detail_serve_plan.as_ref(),
    );
    // +2 for the borders: the popup grows to fit its content (clipped only
    // when the terminal itself is too small).
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
    draw_speed_chart(frame, chart_area, r, profile.bandwidth_gbps);
}

/// Generation speed as a function of context depth — the KV cache is
/// re-streamed every token, so tok/s decays as the conversation grows.
/// Sampled from the same estimator the table uses (anchored at 8k there).
fn draw_speed_chart(frame: &mut Frame, area: Rect, r: &ScoredModel, bandwidth_gbps: f64) {
    let v = r.model.to_model_variant(&r.model.variants[r.variant_idx]);
    let max_ctx = r.model.context_max.clamp(8_192, 131_072);
    const SAMPLES: u32 = 64;
    let points: Vec<(f64, f64)> = (0..=SAMPLES)
        .map(|i| {
            let ctx = max_ctx as u64 * i as u64 / SAMPLES as u64;
            let tps =
                estimate_speed(&v, bandwidth_gbps, kv_cache_bytes(&v, ctx as u32)).generation_tps;
            (ctx as f64, tps)
        })
        .collect();
    let y_max = points.first().map(|p| p.1).unwrap_or(0.0).max(1.0);
    let ctx_label = |c: f64| {
        if c >= 1024.0 {
            format!("{}k", (c / 1024.0).round() as u64)
        } else {
            format!("{}", c as u64)
        }
    };
    let dataset = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(ACCENT))
        .data(&points);
    let x_axis = Axis::default()
        .style(Style::new().fg(Color::DarkGray))
        .bounds([0.0, max_ctx as f64])
        .labels(vec![
            Span::styled("0", Style::new().fg(Color::DarkGray)),
            Span::styled(
                ctx_label(max_ctx as f64 / 2.0),
                Style::new().fg(Color::DarkGray),
            ),
            Span::styled(ctx_label(max_ctx as f64), Style::new().fg(Color::DarkGray)),
        ]);
    let y_axis = Axis::default()
        .style(Style::new().fg(Color::DarkGray))
        .bounds([0.0, y_max])
        .labels(vec![
            Span::styled("0", Style::new().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}", y_max / 2.0),
                Style::new().fg(Color::DarkGray),
            ),
            Span::styled(format!("{y_max:.0}"), Style::new().fg(Color::DarkGray)),
        ]);
    let chart = Chart::new(vec![dataset])
        .block(
            Block::default().title(Span::styled(
                "tok/s by context depth",
                Style::new()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )),
        )
        .x_axis(x_axis)
        .y_axis(y_axis);
    frame.render_widget(chart, area);
}

fn detail_lines<'a>(
    r: &'a ScoredModel,
    plan: Option<&'a Result<RunPlan, String>>,
    serve_plan: Option<&'a Result<ServePlan, String>>,
) -> Vec<Line<'a>> {
    let v = &r.model.variants[r.variant_idx];
    let section = |s: &'a str| {
        Line::from(Span::styled(
            s,
            Style::new()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ))
    };
    let kv = |k: &'a str, val: String| {
        Line::from(vec![
            Span::styled(format!("  {k:<14}"), Style::new().fg(Color::DarkGray)),
            Span::styled(val, Style::new().fg(Color::Gray)),
        ])
    };

    let mut lines = vec![
        Line::from(vec![
            Span::styled(
                r.model.name.as_str(),
                Style::new().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  {}", v.quant), Style::new().fg(Color::Gray)),
        ]),
        Line::default(),
        section("score"),
        kv("total", format!("{:.0}/100", r.score.total)),
        kv(
            "breakdown",
            format!(
                "fit {:.0} · speed {:.0} · quality {:.0} · context {:.0}",
                r.score.fit, r.score.speed, r.score.quality, r.score.context
            ),
        ),
        Line::default(),
        section("memory"),
        kv("weights", gib(r.memory.weights_bytes)),
        kv("kv cache @8k", gib(r.memory.kv_cache_bytes)),
        kv("overhead", gib(r.memory.overhead_bytes)),
        kv(
            "total",
            format!(
                "{} vs GPU limit {} ({})",
                gib(r.memory.total_bytes),
                gib(r.memory.gpu_limit_bytes),
                verdict_label(r.memory.verdict)
            ),
        ),
        Line::default(),
        section("speed"),
        kv(
            "generation",
            format!(
                "~{:.0} tok/s ({})",
                r.speed.generation_tps,
                r.speed.tier.label()
            ),
        ),
        kv(
            "prompt",
            format!(
                "~{:.0}–{:.0} tok/s",
                r.speed.prompt_tps_range.0, r.speed.prompt_tps_range.1
            ),
        ),
        Line::default(),
        section("run"),
    ];
    match plan {
        Some(Ok(plan)) => lines.push(Line::from(vec![
            Span::styled("  $ ", Style::new().fg(Color::DarkGray)),
            Span::styled(plan.display(), Style::new().fg(ACCENT)),
        ])),
        Some(Err(e)) => lines.push(Line::from(Span::styled(
            format!("  {e}"),
            Style::new()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ))),
        None => {}
    }
    // One-line serve hint right under the run command, plan-derived so the
    // endpoint shown is exactly the one `s` would serve on.
    match serve_plan {
        Some(Ok(sp)) => {
            let runtime = crate::output::runtime_label(sp.runtime);
            lines.push(Line::from(vec![
                Span::styled("  s to serve on ", Style::new().fg(Color::DarkGray)),
                Span::styled(sp.endpoint.clone(), Style::new().fg(ACCENT)),
                Span::styled(format!(" ({runtime})"), Style::new().fg(Color::DarkGray)),
            ]));
        }
        Some(Err(e)) => lines.push(Line::from(Span::styled(
            format!("  {e}"),
            Style::new()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ))),
        None => {}
    }
    if r.memory.verdict == FitVerdict::FitsWithSysctlTuning {
        let mb = r.memory.total_bytes / (1024 * 1024) + 1024;
        lines.push(Line::default());
        lines.push(section("tuning"));
        lines.push(kv("hint", format!("sudo sysctl iogpu.wired_limit_mb={mb}")));
    }
    // Trailing gap before the speed chart rendered in the area below.
    lines.push(Line::default());
    lines
}

/// Centered popup: `pw`% of the width (min 40 cols) × exactly `height` rows,
/// both clamped to `area` so small terminals clip instead of panicking.
fn centered(area: Rect, pw: u16, height: u16) -> Rect {
    let width = (u32::from(area.width) * u32::from(pw) / 100) as u16;
    let width = width.max(40).min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}
