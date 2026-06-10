# tetro tray — Design

**Date:** 2026-06-10
**Status:** approved (brainstorm with user)
**Depends on:** `tetro serve` plan (docs/superpowers/plans/2026-06-10-tetro-serve.md), not yet implemented — the serving registry below is built INTO serve from the start, no retrofit.

## Goal

A macOS menu bar item showing which HTTP endpoints are currently serving which models, with one-click copy of the OpenAI-compatible URL.

## Decisions (user-validated)

| Question | Decision |
|---|---|
| Interactions | List + click-to-copy (copies the OpenAI chat-completions URL). No stop action in v1. |
| Lifecycle | Manual `tetro tray` subcommand. No auto-spawn from `serve`, no login item in v1. |
| Discovery | Hybrid: registry of tetro-launched servers + live Ollama `/api/ps` (sees models loaded outside tetro). |
| Tech | `tray-icon` crate (Tauri's tray lib) + `tao` event loop + `arboard` clipboard. Pure Rust, same workspace, aligns with the future Tauri app. |

## Components

### 1. Serving registry — tetro-core, new module `serving.rs`

- `ServingRecord { pid: u32, runtime: RuntimeKind, endpoint: String, openai_url: String, model_ref: String, ready_path: String, started_at: i64 }` (serde).
- Storage: one JSON file per server at `~/Library/Application Support/tetro/serving/{pid}.json`. Env override `TETRO_SERVING_DIR` (tests).
- API: `register(&record)`, `unregister(pid)`, `list_live(probe) -> Vec<ServingRecord>`.
- `list_live` reaps stale entries: record removed when the PID is dead (`kill(pid, 0)` fails) OR `GET {endpoint}{ready_path}` does not answer 2xx. IO kept in small functions testable with tempdir.
- Writers: `tetro serve` registers after readiness, unregisters on exit (Drop guard + best-effort SIGINT handler). The "Ollama already running" case writes NO record — the daemon is discovered via `/api/ps` anyway. Registry covers only tetro-spawned children (llama-server, mlx_lm.server, ollama serve booted by tetro).

### 2. Ollama discovery — tetro-core

- `ollama_loaded_models(probe: &dyn SystemProbe) -> Option<Vec<LoadedModel>>` with `LoadedModel { name: String, size_bytes: u64 }`, via `GET http://127.0.0.1:11434/api/ps` (`models[].name`, `models[].size`). `None` when unreachable. Reuses `http_get_local`; mock-tested on a JSON fixture.

### 3. `tetro tray` — binary subcommand

- `tao` event loop on the main thread, `tray-icon` with an embedded monochrome template icon, menu rebuilt every 5 s and on "Rafraîchir".
- Menu structure:
  - One section per server: disabled title line `Ollama — 127.0.0.1:11434`, then one clickable item per model (`● qwen3:30b-a3b — 18 GiB`). Click copies the server's OpenAI URL to the clipboard (arboard).
  - Sources merged: Ollama section from `/api/ps` (deduped against a registry record for an ollama tetro booted), one section per registry record for llama-server/mlx.
  - Empty states: no servers at all → disabled "Aucun serveur actif". All three runtimes uninstalled → additional disabled hint "Aucun runtime installé — lance `tetro run` pour installer".
  - Ollama not installed → no Ollama section at all. Ollama installed but `/api/ps` unreachable → disabled "Ollama — injoignable".
  - Footer: "Rafraîchir", "Quitter".
- New bin deps: `tray-icon`, `tao`, `arboard`. Subcommand gated `#[cfg(target_os = "macos")]`; on other platforms prints "tray is macOS-only" and exits 1.

## Error handling

- Stale registry files: reaped by every `list_live` call (tray and any future consumer).
- Two `tetro tray` instances: two icons, no lock in v1 (documented limitation).
- Clipboard failure: non-fatal, logged to stderr.

## Testing

- Core: registry roundtrip + stale-reap (dead PID, tempdir, mock probe for the HTTP liveness part); `/api/ps` parsing on canned JSON via MockProbe.
- Bin: menu construction extracted as pure `build_menu_model(records, ollama_ps, runtimes) -> MenuModel` (data structure, no tray-icon types) — unit-tested for all states above (servers present, empty, ollama uninstalled vs unreachable, no runtimes hint). The thin tray-icon rendering layer is manually smoke-tested (icon shows, click copies, Quitter exits).

## Out of scope (v1)

Stop-server action, login item, auto-spawn from serve, port scanning, non-macOS tray, single-instance lock.

## Sequencing

Extends the serve plan: Task 1bis (registry + /api/ps in core, consumed by `serve_model` from day one), Task 4 (tray subcommand). Implemented after serve Tasks 1-3, subagent-driven, Fable 5.
