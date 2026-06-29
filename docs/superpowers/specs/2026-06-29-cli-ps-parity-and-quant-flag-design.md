# CLI `ps` parity + `--quant` flag design

Date: 2026-06-29
Status: approved (design)

## Problem

Two divergences between the CLI and the TUI:

1. **`paddock ps` only shows paddock-spawned servers.** The handler
   (`main.rs:59`) calls `Registry::list_live`, so it misses Ollama-loaded
   models and the locally-available-but-not-running group. The TUI servers tab
   already shows all of it via `list_all_servers` + `list_available`. CLI users
   see a strictly poorer picture.

2. **No way to choose a quantization on the command line.** `paddock run` /
   `paddock serve` always launch `best_variant` (highest quality that fits).
   The TUI got a quant picker (PR #6); the CLI is stuck on the auto-pick. A user
   who wants a smaller, faster quant from a script or one-liner cannot.

## Goal

- `paddock ps` shows the same information as the TUI servers tab: running
  servers (paddock-spawned + Ollama-loaded) and an AVAILABLE group
  (Ollama-installed + previously-served history), in both text and JSON modes.
- `paddock run --quant <label>` / `paddock serve --quant <label>` launch the
  chosen quantization instead of the auto-picked best.

## Decisions (locked)

- **`ps` shows Running + AVAILABLE** (full parity with the TUI), not just
  running.
- **`--quant` launches even when the chosen quant does not fit** (consistent
  with the TUI quant picker where all quants are selectable, and with the CLI
  already serving `FitsRamOnly`). The fit verdict is informational only.
- **`--quant` with no matching variant errors** and lists the model's available
  quant labels (consistent with the existing ambiguous-name UX). No silent
  fallback to `best_variant`.
- **Multi-variant label collision:** two variants can share a quant string. A
  matching `--quant` picks the first in `variants_by_quality` order (best
  quality / highest bpw). Matching is case-insensitive.
- `--quant` operates on catalog variants (GGUF / MLX). Ollama quant lives in the
  tag name and is out of scope.

## Feature 1: `paddock ps` parity

### Core (`crates/paddock-core/src/serving.rs`)

Add `#[derive(Serialize)]` (alongside existing derives) to the types the CLI
will serialize for JSON mode:

- `ServerRow`
- `AvailableRow`
- `StopHandle`

`ServePlan` (embedded in `AvailableRow`) already derives `Deserialize`; add
`Serialize` to it too. No logic changes in core - `list_all_servers` and
`list_available` already exist and are used by the TUI.

### Handler (`crates/paddock/src/main.rs`)

Replace the `Command::Ps` arm body. Today:

```rust
Some(Command::Ps) => {
    let records = Registry::open_default().list_live(&RealSystemProbe);
    if cli.json {
        println!("{}", serde_json::to_string_pretty(&records)?);
    } else {
        output::print_ps_table(&records);
    }
}
```

New:

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

`History::open_default()` is the existing history loader the TUI's
`servers_task` uses (returns a `History`; `list_available` borrows it).

### Output (`crates/paddock/src/output.rs`)

Replace `print_ps_table(&[ServingRecord])` with:

```rust
/// `paddock ps` text view: a RUNNING section then an AVAILABLE section.
/// Empty sections are skipped; if both are empty, prints "no servers running".
pub fn print_servers(running: &[ServerRow], available: &[AvailableRow])
```

- If both slices empty: `println!("no servers running");` and return.
- RUNNING section (only if non-empty): header line `RUNNING`, then the table
  with columns `MODEL / RUNTIME / ENDPOINT / CTX / UPTIME / PID`. For each
  `ServerRow`:
  - CTX: `r.ctx.map(|c| c.to_string()).unwrap_or_else(|| "-".into())`
  - UPTIME: `r.started_at.map(uptime_label).unwrap_or_else(|| "-".into())`
  - PID: `match &r.stop { StopHandle::Pid(p) => p.to_string(), _ => "-".into() }`
- AVAILABLE section (only if non-empty): a blank line, header line `AVAILABLE`,
  then a table with columns `MODEL / RUNTIME / SIZE / LAST-SERVED`. For each
  `AvailableRow`:
  - SIZE: `r.size_bytes.map(gib).unwrap_or_else(|| "-".into())`
  - LAST-SERVED: `age_label(r.last_served_at, false, now)` (reuse the existing
    `age_label`; compute `now` once as unix seconds, same as `uptime_label`).

Keep the existing column widths/format strings from `print_ps_table` for the
RUNNING table so the layout is unchanged for the common case.

`StopHandle` and `AvailableRow` must be imported in output.rs (they come from
`paddock_core::serving`).

## Feature 2: `--quant` flag

### CLI (`crates/paddock/src/cli.rs`)

Add to both `Run` and `Serve` variants:

```rust
/// Quantization label to launch (e.g. Q4_K_M); default auto-picks the best fit
#[arg(long)]
quant: Option<String>,
```

### Quant resolution (`crates/paddock/src/main.rs`)

Add a pure helper:

```rust
/// Index into `variants` of the variant matching `label` (case-insensitive).
/// On a label shared by several variants, returns the best-quality one
/// (first in `variants_by_quality` order). Errors listing available quants
/// when nothing matches.
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

Confirm the variant field name during implementation (`ModelVariant.quant`) and
the `ModelVariant` import path; match the existing `mvs` construction in
`resolve_model`.

Change `resolve_model` to take the quant override:

```rust
fn resolve_model(app: &App, query: &str, quant: Option<&str>) -> Result<(CatalogModel, usize)>
```

After building `mvs` (the `Vec<ModelVariant>`), branch:

- `Some(label)`: `let idx = resolve_quant(&mvs, label)?; Ok((model, idx))`.
  Skips the `best_variant` fit gate entirely - the chosen quant launches even
  if it does not fit (decision locked).
- `None`: existing `best_variant` path (the `bail!` when nothing fits stays).

Thread the flag through the call sites:

```rust
Some(Command::Run { model, ctx, quant }) => run_model(&app, &model, ctx, quant, cli.json)?,
Some(Command::Serve { model, port, ctx, foreground, quant }) =>
    serve_model(&app, &model, port, ctx, foreground, quant, cli.json)?,
```

`run_model` / `serve_model` gain a `quant: Option<String>` param and pass
`quant.as_deref()` to `resolve_model`. Everything downstream (`resolved_ctx`,
`plan_run` / `plan_serve`) already takes the chosen index, so no further
changes.

## Error handling

- Unknown `--quant` label: `bail!` listing available quants (non-zero exit).
- Chosen quant that does not fit: launches anyway (per decision); the existing
  serve/run lifecycle handles `FitsWithSysctlTuning` / `FitsRamOnly` as it does
  for any model today.
- `ps` with an unreachable Ollama daemon: `list_all_servers` /
  `list_available` already return what they can (probe failures yield empty
  Ollama lists); the AVAILABLE/RUNNING sections degrade gracefully.

## Testing

Pure, unit-testable pieces:

- **`resolve_quant`** (in `main.rs` tests, or a small module):
  - exact label match returns its index;
  - case-insensitive (`q4_k_m` matches `Q4_K_M`);
  - a label on two variants returns the best-quality index (first in
    `variants_by_quality`);
  - no match returns an error whose message contains the available quant labels.
- **`print_servers`** formatting (in `output.rs` tests): build fixture
  `ServerRow` / `AvailableRow` values and assert the rendered string has the
  RUNNING / AVAILABLE headers, `-` for a missing PID (Ollama row) and missing
  CTX, and that both-empty prints `no servers running`. (Capture via a
  `format`-returning inner helper if `print_servers` writes to stdout - extract
  `fn servers_view(...) -> String` and have `print_servers` print it, so the
  test asserts on the string.)

JSON `ps` shape and the live Ollama/history merge are verified by a smoke test
(`cargo run -- ps` and `cargo run -- ps --json` with a server running and Ollama
models installed). `--quant` end-to-end is a smoke test: `paddock serve <model>
--quant Q2_K` launches the smaller quant; `--quant bogus` errors with the quant
list.

## Out of scope (YAGNI)

- Ollama tag-level quant selection via `--quant`.
- A `--quant` picker / interactive prompt (the TUI already has one).
- Per-model remembered quant preference.
- Changing `list_all_servers` / `list_available` core logic (reused as-is).
