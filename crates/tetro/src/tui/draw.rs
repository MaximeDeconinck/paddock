//! Pure rendering — reads state, never mutates it, no IO.
//! Palette: DarkGray/Gray/White + a single Cyan accent.

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Cell, Clear, Paragraph, Row, Table, TableState};
use ratatui::Frame;
use tetro_core::estimate::FitVerdict;
use tetro_core::hardware::{HardwareProfile, RuntimeStatus};
use tetro_core::runtime::plan_run;

use crate::app::ScoredModel;
use crate::output::{gib, verdict_label};
use crate::tui::state::{use_case_label, Mode, TuiState};

const ACCENT: Color = Color::Cyan;

pub fn draw(frame: &mut Frame, state: &TuiState, profile: &HardwareProfile) {
    let [header, table, footer] = Layout::vertical([
        Constraint::Length(7),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    draw_header(frame, header, profile);
    draw_table(frame, table, state);
    draw_footer(frame, footer, state);
    if state.mode == Mode::Detail {
        draw_detail(frame, state);
    }
}

fn draw_header(frame: &mut Frame, area: Rect, p: &HardwareProfile) {
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
    let block = Block::bordered()
        .border_style(Style::new().fg(Color::DarkGray))
        .title(Span::styled(
            " tetro ",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
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
    let header = Row::new(["MODEL", "QUANT", "MEMORY", "TOK/S", "FIT", "SCORE"]).style(
        Style::new()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );
    let rows = state.rows.iter().map(|r| {
        let v = &r.model.variants[r.variant_idx];
        Row::new(vec![
            Cell::from(r.model.name.clone()),
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
            Constraint::Length(9),
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Length(12),
            Constraint::Length(6),
        ],
    )
    .header(header)
    .row_highlight_style(Style::new().fg(Color::Black).bg(ACCENT));
    // Fresh TableState each frame: draw stays pure, ratatui recomputes the
    // scroll offset so the selected row is always visible.
    let mut ts = TableState::default().with_selected(Some(state.selected));
    frame.render_stateful_widget(table, area, &mut ts);
}

fn draw_footer(frame: &mut Frame, area: Rect, state: &TuiState) {
    let left = match &state.last_error {
        Some(err) => Line::from(Span::styled(
            format!("error: {err}"),
            Style::new().fg(ACCENT),
        )),
        None => Line::from(Span::styled(
            "↑↓ move · enter detail · x run · / search · g/c/r/h use-case · q quit",
            Style::new().fg(Color::DarkGray),
        )),
    };
    let uc = use_case_label(state.use_case);
    let right = match &state.mode {
        Mode::Search { query } => format!("{uc} · /{query}▌"),
        _ if !state.query.is_empty() => format!("{uc} · /{}", state.query),
        _ => uc.to_string(),
    };
    frame.render_widget(Paragraph::new(left), area);
    frame.render_widget(
        Paragraph::new(Span::styled(right, Style::new().fg(ACCENT))).alignment(Alignment::Right),
        area,
    );
}

fn draw_detail(frame: &mut Frame, state: &TuiState) {
    let Some(r) = state.rows.get(state.selected) else {
        return;
    };
    let area = centered(frame.area(), 70, 70);
    frame.render_widget(Clear, area);
    let block = Block::bordered()
        .border_style(Style::new().fg(Color::DarkGray))
        .title(Span::styled(" detail ", Style::new().fg(ACCENT)));
    frame.render_widget(
        Paragraph::new(detail_lines(r, &state.runtimes)).block(block),
        area,
    );
}

fn detail_lines<'a>(
    r: &'a ScoredModel,
    runtimes: &tetro_core::hardware::RuntimesStatus,
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
    match plan_run(&r.model, v, runtimes) {
        Ok(plan) => lines.push(Line::from(vec![
            Span::styled("  $ ", Style::new().fg(Color::DarkGray)),
            Span::styled(plan.display(), Style::new().fg(ACCENT)),
        ])),
        Err(e) => lines.push(Line::from(Span::styled(
            format!("  {e}"),
            Style::new()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ))),
    }
    if r.memory.verdict == FitVerdict::FitsWithSysctlTuning {
        let mb = r.memory.total_bytes / (1024 * 1024) + 1024;
        lines.push(Line::default());
        lines.push(section("tuning"));
        lines.push(kv("hint", format!("sudo sysctl iogpu.wired_limit_mb={mb}")));
    }
    lines
}

/// Centered popup occupying `pw`% × `ph`% of `area`.
fn centered(area: Rect, pw: u16, ph: u16) -> Rect {
    let [_, mid, _] = Layout::vertical([
        Constraint::Percentage((100 - ph) / 2),
        Constraint::Percentage(ph),
        Constraint::Percentage((100 - ph) / 2),
    ])
    .areas(area);
    let [_, mid, _] = Layout::horizontal([
        Constraint::Percentage((100 - pw) / 2),
        Constraint::Percentage(pw),
        Constraint::Percentage((100 - pw) / 2),
    ])
    .areas(mid);
    mid
}
