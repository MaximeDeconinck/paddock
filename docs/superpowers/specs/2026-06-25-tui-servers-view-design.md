# TUI servers view design

Date: 2026-06-25
Status: approved (design)

## Problem

The CLI launcher (PR #1) lets you serve detached and manage servers with
`paddock ps`/`stop`/`logs`. But the TUI can't show or manage running servers, and
serving from the TUI tears down the TUI and runs the model **attached** in the
terminal (`tui/mod.rs` returns `Exit::Serve(plan)` → `serve_with_plan(plan, true)`).
So TUI users get neither the detached behavior nor a way to see what's running.

## Goal

Add an in-TUI servers view (list + stop + copy endpoint) and make serving from the
TUI run detached without leaving the TUI. Target audience: users who prefer the
TUI over the CLI.

This is workstream B, a follow-up to the CLI launcher (workstream #1). Same
positioning: paddock is the end-to-end Apple-Silicon launcher, now usable entirely
from the TUI.

## Decisions (locked)

- **v1 actions:** stop, copy endpoint, serve-detached-from-TUI. **No** in-TUI log
  viewer (deferred), no stop-confirmation, no Ollama-model actions.
- **Refresh:** background auto via an mpsc task (the `sync_task` pattern), so the
  blocking `list_live` probes never freeze the UI thread.
- **Serve from TUI:** suspend/resume the terminal around `serve_with_plan(plan,
  false)` (approach A). Readiness waits can run minutes (downloads); blocking the
  render loop is not an option, and suspend/resume shows load/install progress in
  the normal terminal.

## Components

### 1. Navigation: two tabs

A `models` tab (the existing list) and a new `servers` tab. `Tab` toggles between
them (ignored while the search input is active). A tab indicator renders in the
header.

```
 paddock                                      [ models ]  servers
 +- running servers ----------------------------------------------+
 | MODEL                     RUNTIME    ENDPOINT          CTX   UP |
 |> Qwen3.6-35B-A3B-MTP      llama.cpp  127.0.0.1:8080   49152 2m  |
 |  gpt-oss-20b              llama.cpp  127.0.0.1:8081   32768 9m  |
 +----------------------------------------------------------------+
  j/k move   x stop   c copy endpoint   tab models   q quit
```

Empty servers tab: `no servers running - press s on a model to serve one`.

### 2. State (`crates/paddock/src/tui/state.rs`)

- New field `tab: Tab` where `enum Tab { Models, Servers }`, orthogonal to the
  existing `mode: Mode` (which keeps the Search/Detail overlays of the models tab).
- New fields `servers: Vec<ServingRecord>` and `server_selected: usize`.
- `handle_key`: an active overlay (Search/Detail) is handled first as today;
  otherwise dispatch by `tab`. `Tab` toggles the tab (when not in search input).
  Servers tab keys: `j`/`k` (and Up/Down) navigate, `x` stop selected, `c` copy
  selected endpoint, `q` quit, `Tab` back to models.
- State stays **pure**: IO-bearing actions surface through `Action`. The tab toggle
  and server-cursor moves are pure state mutations (no `Action`).

### 3. Background refresh (`crates/paddock/src/tui/servers_task.rs`)

A thread modeled on `sync_task`: loop { `Registry::list_live(&RealSystemProbe)`;
send the `Vec<ServingRecord>` over an mpsc channel; sleep ~2s }. The event loop
drains the channel each iteration (exactly like `sync_rx`) and stores the snapshot
in `state.servers`, re-clamping `server_selected`. Because `list_live` makes
blocking HTTP probes, running it off the UI thread keeps rendering smooth. The
thread exits when the `Receiver` is dropped (TUI quit): the next `send` returns
`Err` and the loop breaks.

### 4. Actions (event loop performs IO; state pure)

- **Stop:** `Action::StopServer(pid)` -> event loop calls
  `paddock_core::serving::terminate(pid)` then `Registry::unregister(pid)`, and
  optimistically removes the record from `state.servers` (the background task
  reconciles). No confirmation (single selected server; `all` stays CLI-only).
- **Copy:** `Action::CopyEndpoint(String)` (the selected server's `openai_url`) ->
  event loop copies to the clipboard via a shared helper (see §5).
- **Serve detached (A):** `s` on the models tab returns `Action::Serve(plan)`,
  now handled **inline** by the event loop instead of returning `Exit::Serve`:
  `ratatui::restore()` -> `serve_with_plan(plan, false)` (load/install progress
  shows in the normal terminal) -> `*terminal = ratatui::init()` -> set `tab =
  Servers` and do one optimistic refresh. A serve error goes to `last_error` and
  the TUI still resumes.

### 5. Associated refactor

- Remove the `Exit::Serve` variant (serve is now inline). `Exit::Run` stays (run
  is an interactive chat that replaces the process, so it must exit the TUI).
- Extract the clipboard write currently in `tray/mod.rs` (`copy_to_clipboard`,
  `pbcopy`) into a small shared helper both the tray and the TUI call (DRY).

### 6. Error handling

- Background task dies -> the servers tab keeps showing the last snapshot with a
  discreet note (mirrors the footer sync-failure style). No crash.
- Stop on an already-dead pid -> no-op (`terminate` is best-effort); the record
  drops on the next refresh.
- `serve_with_plan` error on the inline path -> stored in `last_error`, TUI
  resumes normally.

### 7. Testing

State is pure and unit-testable: `Tab` toggle, servers-tab navigation/clamping,
`x` -> `Action::StopServer(pid)`, `c` -> `Action::CopyEndpoint(url)`, the serve
action, and the empty-state render path. The background task and the suspend/resume
serve path are IO/integration concerns (kept thin; verified by a manual live smoke
test: serve from TUI -> appears in servers tab -> copy endpoint -> stop -> drops).

## Out of scope (YAGNI)

In-TUI log viewer, stop confirmation, multi-select, actions on Ollama-loaded models
(managed by `ollama ps`/`ollama stop`), auto-refresh interval configuration.
