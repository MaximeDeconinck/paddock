# Background Catalog Sync — Design

**Date:** 2026-06-20
**Strategic context:** paddock competes with llmfit (28k stars) by being Apple-Silicon-first with a *live* catalog, where llmfit ships a static compile-time snapshot of ~206 models. A live catalog is only viable if refreshing it does not make the user wait. Today `paddock sync` is a blocking ~6-minute command; the TUI reads a static DB snapshot at launch and bails if the catalog is empty.

**This spec** makes catalog refresh happen in the background while the TUI is fully usable, with a small in-progress indicator. It is the prerequisite for later catalog-volume expansion (a separate spec): once sync is non-blocking, its duration stops mattering.

## Goals

1. The TUI launches instantly against the existing catalog snapshot (or an empty one).
2. A background task refreshes the catalog while the user browses, writing to the same SQLite file.
3. A small indicator shows sync state (running / just-updated / failed).
4. On completion, the list updates atomically, preserving the user's selection and active search.
5. The blocking `paddock sync` CLI command stays unchanged for scripting/cron.

## Non-goals

- No catalog-volume changes (more HF/MLX models) — that is Spec 2.
- No incremental/live list updates during sync — a single atomic swap on completion.
- No change to `catalog::sync()`'s own logic.

## Decisions (from brainstorming)

- **Trigger:** auto-sync on TUI launch when the catalog is stale (last sync > 24h, or never); plus a manual `r` key on demand. Only one sync runs at a time.
- **Refresh model:** atomic swap at completion (not incremental).
- **First run (empty catalog):** launch the TUI empty with a "building catalog…" message; populate on first completion. Replaces today's `bail!`.

## Architecture

### Component 1 — DB concurrency (WAL)

`Db::open` (db.rs) additionally runs `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`. WAL lets one writer (the background sync, on its own connection) and readers (the TUI) operate on the same file without blocking each other; `busy_timeout` absorbs the brief writer-lock windows. The CLI `paddock sync` benefits too. No other DB change.

### Component 2 — Background sync task (`crates/paddock/src/tui/sync_task.rs`, new)

```rust
pub enum SyncMsg {
    Progress { source: &'static str, count: usize },
    Done(Box<SyncReport>),
    Failed(String),
}

/// Spawn the catalog sync on a background thread. Returns the receiving end;
/// the caller drains it from the event loop. The thread owns its own tokio
/// runtime and its own Db connection (never shares the TUI's).
pub fn spawn_sync(opts: SyncOptions) -> std::sync::mpsc::Receiver<SyncMsg>;
```

Internals: `std::thread::spawn` → `tokio::runtime::Builder::new_current_thread().enable_all()` → `Db::open(default_db_path())` (a fresh connection) → `ReqwestClient::new()` → `catalog::sync(&http, &db, &opts)`. The thread sends `Done`/`Failed` and exits. `catalog::sync` is reused verbatim. (Progress reporting is best-effort; see "Progress granularity" below.)

### Component 3 — Sync status state

`TuiState` gains:

```rust
pub enum SyncStatus {
    Idle,                 // no sync this session, or never triggered
    Running,              // v1: spinner only, no mid-sync count (see Progress granularity)
    Done { at: std::time::Instant },
    Failed(String),
}
pub sync_status: SyncStatus,
```

State stays IO-free: the `Receiver` lives in the event loop, not in `TuiState`. The loop updates `sync_status` from drained messages.

### Component 4 — Event loop integration (`tui/mod.rs`)

- `run()`: read `db.last_sync()`. If stale (>24h) or empty catalog, call `spawn_sync` and hold the `Receiver`. Soften the empty-catalog `bail!`: launch with empty rows + a "building catalog…" status when a sync is starting; only error out if empty *and* no sync will run (e.g. offline manual-only) — and even then show a hint screen rather than crash if a sync is in flight.
- `event_loop`: each 250ms tick, before/after draw, drain `rx.try_recv()`:
  - `Progress{..}` (not emitted in v1) → would update a count; v1 keeps `sync_status = Running` set once at spawn.
  - `Done` → reload `app.scored_models(db, uc, false)`, `state.set_rows_preserving(rows, uc)`, `sync_status = Done { at: now }`.
  - `Failed(msg)` → `sync_status = Failed(msg)` (footer shows it; non-fatal).
  - channel disconnected with no terminal msg → treat as Failed("sync thread died").
- New key `r` in `state.handle_key` → returns `Action::StartSync`; the loop spawns a new task only if none is currently running (guard on whether a live `Receiver` exists).

### Component 5 — Indicator (`tui/draw.rs`)

A short status segment, accent-colored:
- `Running` → spinner glyph (frame advances per tick) + `syncing…` (no count in v1).
- `Done` (within ~5s of completion) → `catalog updated`; afterward `synced {age} ago` derived from `db.last_sync()` — already humanized like the AGE column.
- `Failed` → `sync failed: {msg}` in the footer error slot.
- `Idle` with a known last_sync → `synced {age} ago`.

Placement: footer right side (next to the use-case label) keeps the header box stable. Exact wording/spot finalized visually at implementation.

### Component 6 — Selection & search preservation

New `TuiState::set_rows_preserving(rows, uc)`:
1. Remember the currently-selected model name (if any) and the active search query.
2. Apply the new rows + use-case.
3. Re-apply the search filter.
4. Re-find the previously-selected model by name in the filtered set; if gone, clamp the index to range.

Pure transformation on `TuiState` fields → unit-testable without a terminal.

## Data flow

```
launch → read last_sync → (stale? spawn_sync → Receiver)
                              │
event loop tick (250ms): ─────┤
   draw(state)                │
   rx.try_recv() ─────────────┘
     (Progress not emitted in v1; status.Running set at spawn)
     Done     → reload rows + set_rows_preserving + status.Done
     Failed   → status.Failed
key 'r' → spawn_sync (if none running)
```

## Error handling

- Background sync failure is never fatal: the TUI keeps the existing snapshot and shows the error in the footer.
- Channel disconnect without a terminal message → treated as failure.
- WAL on a filesystem that rejects it (rare, e.g. some network mounts): `journal_mode=WAL` returns the actual mode; if it didn't switch, log a one-line warning and continue (SQLite falls back to the prior journal mode; concurrent read/write may then briefly block within `busy_timeout`, acceptable).
- A sync writing while the TUI reads: WAL + `busy_timeout` handle it; a transient `SQLITE_BUSY` past the timeout is surfaced as a non-fatal refresh error, the next tick retries.

## Progress granularity

`catalog::sync()` currently returns only a final `SyncReport`. Two options for the `seen` counter:
- **v1 (chosen):** no mid-sync progress plumbing. The indicator shows an animated spinner + `syncing…` (no count) while running, and the final counts on `Done`. Keeps `catalog::sync` untouched.
- Future: thread a callback / channel into `sync()` to emit per-source counts. Out of scope here.

This means `SyncMsg::Progress` is **not emitted in v1**; `Running` carries no meaningful count. Kept in the enum as the forward-compatible shape, but the v1 indicator is spinner-only. (Resolved to avoid an ambiguous half-feature.)

## Testing

- **db.rs:** after `Db::open`, `PRAGMA journal_mode` returns `wal`; a write on one connection followed by a read on a second connection of the same file succeeds (concurrent-access smoke test with two `Db::open` on a tempfile).
- **SyncStatus transitions:** a pure function mapping a drained `SyncMsg` to the next `SyncStatus` (Idle→Running→Done, →Failed, disconnect→Failed) — table test.
- **set_rows_preserving:** selection kept by name across a row swap; selection dropped when the model disappears (clamps); active query re-applied to the new rows.
- Not unit-tested: the thread + terminal wiring itself. All testable logic is extracted into pure functions (status transition, row preserve) and the DB pragma check.

## Sequencing

This spec ships first. Spec 2 (catalog volume: HF params fallback, factory authors, MLX expansion) builds on top — once sync is background, its longer runtime is invisible to the user.
