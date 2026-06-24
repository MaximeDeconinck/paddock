# paddock launcher — background serving design

Date: 2026-06-24
Status: approved (design)

## Problem

`paddock serve` spawns llama.cpp / mlx-lm servers in the **foreground** and stays
attached: Ctrl-C kills the server. To run several models you keep one dedicated
terminal tab open per server, and tabs pile up fast. The serving `Registry`
already exists but is read only by the macOS tray — there is no terminal-side way
to see what is running or stop it. Two secondary frictions: `--ctx` must be set by
hand (the default 8k caused a real `request (N tokens) exceeds the available
context size` error against OpenWebUI), and wiring the endpoint into OpenWebUI is
manual copy-paste.

## Goal

Turn `serve` into a detached, manageable launcher so no dedicated terminal is
needed, with terminal-side lifecycle (`ps` / `stop` / `logs`) and a context
window that sizes itself to available memory.

## Competitive context

This is workstream #1 of differentiating paddock from **llmfit**. llmfit is a
cross-platform recommender that does **not** run or serve models and treats Apple
Silicon as a generic GPU. paddock's wedge is being the end-to-end Apple-Silicon
launcher: "llmfit tells you what fits; paddock runs it." Workstream #2
(Apple-calibrated benchmarking / accuracy) comes after and reuses the server
lifecycle built here.

## Decisions (locked)

- **Detached by default.** `serve` backgrounds spawned children; `--foreground`/`-f`
  keeps the current attached behavior.
- **ctx auto-size by default.** Largest context that fits memory; `--ctx` overrides.
- Command named `paddock ps` (Unix / `docker ps` / `ollama ps` convention).

## CLI surface

- `paddock serve <model> [--port N] [--ctx N] [--foreground|-f]`
- `paddock ps [--json]`
- `paddock stop <model|pid|all> [--yes|-y]`
- `paddock logs <model|pid> [-f]`

## Components

### 1. Detached serve (`crates/paddock/src/main.rs`)

Default path for **spawned-child runtimes** (llama.cpp, mlx-lm):

1. Resolve model, build `ServePlan` (unchanged).
2. Install confirmation, if a runtime is missing, happens **while still attached**
   (the prompt needs the terminal).
3. Spawn the child detached:
   - Open/create log file `<serving_dir>/logs/<pid>.log`, where `serving_dir` is
     `default_serving_dir()` (the dir the `Registry` already uses).
   - `Command` with `stdout(log)`, `stderr(log)`, `stdin(null)`.
   - Detach from the controlling terminal so it survives the tab closing:
     `unsafe { pre_exec(|| { libc::setsid(); Ok(()) }) }` (new session, no SIGHUP
     on terminal close). `libc` is already an indirect dep via the Metal/sysctl
     code; add it directly if needed.
4. Poll readiness (`/health` etc.) **before returning**, exactly as today, so the
   real endpoint and any startup failure are reported (servers may download via
   `-hf` first). No detach-and-pray.
5. Register a `ServingRecord` in the existing `Registry`. **Critically, the
   detached path does NOT install the RAII drop-guard** that unregisters on parent
   exit — the server must outlive the parent. (The `--foreground` path keeps the
   guard.)
6. Print the endpoint block plus a footer: `serving in background · pid N ·
   paddock logs <model>`. Exit 0.

`--foreground`/`-f`: current behavior verbatim — attached, streams to the tty,
Ctrl-C stops, drop-guard unregisters on exit.

**Ollama path** is already daemon-backed (no terminal blocked today): pull,
register the record, return. Nothing to detach. `stop` for it means
`ollama stop <model>` (unload), not killing the daemon.

### 2. Registry extension (`crates/paddock-core/src/serving.rs`)

`ServingRecord` gains:

```rust
#[serde(default)] pub ctx: u32,
#[serde(default)] pub log_path: Option<PathBuf>,
#[serde(default)] pub port: Option<u16>,
```

`#[serde(default)]` keeps old on-disk records deserializable. `list_live`
pruning is unchanged; when a record is pruned its `log_path` file is best-effort
removed.

### 3. `paddock ps`

`Registry::list_live(&probe)` → table `MODEL · RUNTIME · ENDPOINT · CTX · UPTIME ·
PID`. UPTIME derived from `started_at`. `--json` prints the records for scripting.
Empty → a short "no servers running" line.

### 4. `paddock stop <model|pid|all>`

Resolve target against live records:
- numeric → match by pid;
- `all` → every live record;
- otherwise → substring/name match against `model_ref`, with the same
  ambiguity-listing UX as `resolve_model` (list candidates, exit non-zero).

`all` prompts for confirmation before stopping (lists the servers it will stop,
requires explicit y/N); `--yes`/`-y` skips the prompt for scripting. Stopping a
single named/pid target does not prompt.

Per record: child runtimes → `SIGTERM` to pid (`libc::kill`) then unregister;
Ollama → `ollama stop <model_ref>` then unregister. Report what was stopped.

### 5. `paddock logs <model|pid> [-f]`

Resolve target as in `stop`. Read `log_path`; print it, `-f` follows (tail). Child
runtimes only — Ollama records have no `log_path`, so print a message pointing to
Ollama's own logs.

### 6. ctx auto-size (`crates/paddock-core/src/estimate.rs`)

```rust
/// Largest context whose weights + KV cache still fit the GPU budget,
/// rounded down to a 4k step, clamped to [4096, context_max].
pub fn auto_ctx(v: &ModelVariant, budget: &MemoryBudget, context_max: u32) -> u32
```

Walk 4k steps down from `context_max` (or up to it) picking the largest where
`weights + kv_cache_bytes(v, ctx) ≤ gpu budget`; clamp to `[4096, context_max]`.
Floor of 4096: a variant that cannot hold even 4k would not have been chosen by
`best_variant`.

`plan_run` / `plan_serve` resolve ctx as: explicit `--ctx` if given, else
`auto_ctx(...)`. The `ctx: Option<u32>` parameter is already wired; `None` now
means "auto" instead of the old fixed `DEFAULT_CONTEXT`. The resolved ctx is
stored in the `ServingRecord` so `ps` can show it.

### 7. OpenWebUI hookup

The `serve` output and `paddock ps` print a copy-paste-ready `openai` URL and
`model` ref. No deeper API auto-configuration (YAGNI).

## Error handling

- Detached spawn failure → report the error, write no registry entry, exit
  non-zero.
- `stop` / `logs` with no matching target → error listing the live servers.
- Pruned records best-effort delete their log file.
- `setsid`/`pre_exec` failure surfaces as a normal spawn error.

## Testing

- `auto_ctx`: fits mid-range, clamps to `context_max`, floors at 4096.
- `ServingRecord` serde round-trip **and** deserialization of an old record
  lacking the new fields (backward compat).
- `ps`/`stop` target resolution: pid, `all`, substring, ambiguous (lists
  candidates).
- Detach decision layer: which runtimes take the detached vs foreground path
  (keep the actual spawn thin; assert on the plan/branch, not a live process).

## Out of scope (YAGNI)

Auto-restart / supervision, launchd integration, multi-machine, OpenWebUI API
auto-config, log rotation.
