# TUI available-models design

Date: 2026-06-26
Status: approved (design)

## Problem

The TUI servers tab shows only what is *running* (paddock-spawned servers +
Ollama-loaded models). A user with many models installed locally has no way to
see them or launch one directly from the TUI: they must go back to the models
tab and search the catalog, and locally-cached non-catalog models (custom HF
repos) are not surfaced at all.

## Goal

Show locally-available-but-not-running models in the servers tab, greyed out, in
a second group below the running ones. Pressing `enter` on a grey model launches
it directly (reusing the existing detached-serve flow). Gives a one-screen view
of "everything on this machine" with one-key launch.

## Key decision: history, not a cache scan

Scanning `~/.cache/huggingface/hub/` to classify mlx vs gguf, detect complete
downloads, and reconstruct a launch ref is fuzzy and fragile (the incomplete
gpt-oss case showed why). Instead:

- **Ollama** installed models come from the daemon's `/api/tags` (authoritative,
  cheap, never stale).
- **llama.cpp / mlx** available models come from a **served-history** paddock
  maintains: every spawned serve is recorded, self-contained (stores the
  `ServePlan`), so it can be relaunched without the catalog. A model paddock has
  served once is by definition cached and known.

Tradeoff: an mlx/gguf model downloaded but never served via paddock will not
appear. Anything served once is remembered and relaunchable. Accepted.

## Components

### 1. Served-history store (`crates/paddock-core/src/serving.rs`)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub plan: ServePlan,      // self-contained: argv, runtime, model_ref, ctx, endpoint
    pub last_served_at: i64,
}
```

- Persisted as `serving/history.json` (a `Vec<HistoryEntry>`), next to the
  registry, via the same atomic tmp+rename write the `Registry` uses.
- `record_serve(plan: &ServePlan)`: upsert by `plan.model_ref` (replace the
  entry, refresh `last_served_at`). Called from `serve_with_plan` for spawned
  (llama.cpp/mlx) servers only — Ollama is covered by `/api/tags`, not recorded.
- `load_history() -> Vec<HistoryEntry>` (missing/corrupt file -> empty).
- `ServePlan` must gain `Deserialize` (it already derives `Serialize`); add
  `Deserialize` so history round-trips. `InstallPlan`/`RuntimeKind` already
  derive both as needed (verify and add where missing).

The stored `ServePlan` keeps the port it last used; relaunch reallocates a free
port via the existing `serve_with_plan` free-port step, so a stale port is fine.

### 2. Ollama installed models (`crates/paddock-core/src/serving.rs`)

```rust
const OLLAMA_TAGS_URL: &str = "http://127.0.0.1:11434/api/tags";

pub fn ollama_installed_models(probe: &dyn SystemProbe) -> Option<Vec<LoadedModel>>
```

Parses `/api/tags` (`{"models":[{"name","size",...}]}`) into the existing
`LoadedModel { name, size_bytes }`. None when the daemon is unreachable.

### 3. Available-list builder (`crates/paddock-core/src/serving.rs`)

```rust
#[derive(Debug, Clone)]
pub struct AvailableRow {
    pub model: String,
    pub runtime: RuntimeKind,
    pub detail: String,   // "17 GB (installed)" or "last served 2h ago"
    pub plan: ServePlan,  // what `enter` serves
}

/// Locally-available models NOT currently running, for the servers tab's grey
/// group. `running` is the live set (from `list_all_servers`) used to dedup.
pub fn list_available(
    registry: &Registry,
    probe: &dyn SystemProbe,
    running: &[ServerRow],
    now: i64,
) -> Vec<AvailableRow>
```

Build:
- **Ollama:** `ollama_installed_models` -> for each name not present in
  `running` (by model match) -> `AvailableRow { runtime: Ollama, detail:
  "{gib} (installed)", plan: <minimal ollama ServePlan for that name> }`. The
  minimal plan: `server_argv: None`, `runtime: Ollama`, `model_ref: name`,
  endpoint/openai_url = the Ollama base, `ready_path: "/api/version"`, `port:
  None`, `ctx: 0` (serve_with_plan warms it; it is not registered).
- **History (llama.cpp/mlx):** `load_history` -> for each entry whose
  `model_ref` is not in `running` AND whose HF cache dir exists (cheap per-entry
  check, see below) -> `AvailableRow { runtime, detail: "last served {ago}",
  plan: entry.plan }`.

Sort: Ollama installed first (alpha), then history by most-recent.

**Cheap cache-existence check** for a history entry: derive the HF cache dir
from the model_ref (`{org}/{name}[:quant]` -> `~/.cache/huggingface/hub/models--{org}--{name}`)
and `Path::exists()` it. One stat per entry, not a scan. Missing -> the entry is
hidden (and may be pruned from history opportunistically). Uses
`crate::paths` for the cache root (add a `hf_cache_dir()` helper there).

### 4. TUI state (`crates/paddock/src/tui/state.rs`)

- Add `pub available: Vec<AvailableRow>` alongside `servers: Vec<ServerRow>`.
- The servers tab now navigates a **combined** list: indices `0..servers.len()`
  are running rows, the rest are available rows. `server_selected` ranges over
  the combined length. A helper `selected_kind() -> Running(&ServerRow) |
  Available(&AvailableRow) | None` resolves the cursor.
- Keys on the servers tab:
  - `↑`/`↓`: move across both groups (clamp to combined length).
  - `enter`: if the cursor is on an **available** row -> `Action::Serve(row.plan.clone())`
    (reuses the existing detached-serve + suspend/resume + switch-to-servers +
    refresh). On a running row -> no-op (reserved).
  - `x` stop / `c` copy: act only on **running** rows (no-op on available).
- `set_servers` is joined by a `set_snapshot(running, available)` (or the
  background drain sets both); cursor preserved by identity (stop handle for
  running, model for available) then clamped to combined length.

### 5. Background refresh (`crates/paddock/src/tui/servers_task.rs`)

The task now produces both groups in one snapshot:

```rust
pub struct ServersSnapshot { pub running: Vec<ServerRow>, pub available: Vec<AvailableRow> }
```

Body: `let running = list_all_servers(&reg, &probe); let available =
list_available(&reg, &probe, &running, now);` then send `ServersSnapshot`. The
event loop drains it and calls `state.set_snapshot(running, available)`. (`now`
is a unix timestamp computed in the task.)

### 6. Rendering (`crates/paddock/src/tui/draw.rs`)

`draw_servers` renders two sections:
- `RUNNING` rows in normal colours (current table).
- A blank line, then `AVAILABLE` rows in `DarkGray`, columns MODEL / RUNTIME /
  DETAIL (the `detail` string). The selected row (running or available) is
  highlighted as today.
- Empty running + empty available -> the existing "no servers running" notice.
- Footer hint on the servers tab gains `enter launch`.

## Error handling

- `/api/tags` unreachable -> no Ollama available rows (daemon down); running
  Ollama rows already handle this.
- History file missing/corrupt -> empty history (no crash).
- A history entry whose cache dir is gone -> hidden; opportunistically pruned.
- `enter` launch failure -> goes through the existing serve error path
  (`last_error`, TUI resumes).

## Testing

State is pure and unit-testable: combined-index navigation across both groups,
`enter` on an available row returns `Action::Serve(plan)`, `enter` on a running
row is a no-op, `x`/`c` no-op on available rows, `set_snapshot` cursor preserve
+ clamp. Core: `ollama_installed_models` parses `/api/tags` (MockProbe fixture);
`list_available` dedups against running and maps both sources (MockProbe +
temp registry + a temp history file). History round-trip (`record_serve` ->
`load_history`). The cache-existence check and the background task are IO,
verified by a live smoke test (serve a model, stop it, confirm it appears in the
grey group, `enter` relaunches it; an installed-but-unloaded Ollama model
appears and launches).

## Out of scope (YAGNI)

Scanning the HF cache for never-served models, deleting models from disk
(`ollama rm` / cache eviction) from the TUI, a separate "models on disk" tab,
history size cap (entries are small; prune only on missing-cache).
