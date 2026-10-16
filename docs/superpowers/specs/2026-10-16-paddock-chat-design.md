# paddock_chat + MCP follow-ups + local-delegation demo

Date: 2026-10-16. Branch: `feat/paddock-chat`. Target release: v0.3.0 (after the demo video is approved).

## Goal

Let a remote orchestrator (Claude Code) hand sub-tasks to local models through `paddock mcp`, the way it hands tasks to its own subagents, without the raw data ever entering the orchestrator's context. Prove it with a real (vhs-recorded) demo: Claude Code categorizes a fictitious bank statement with local models and only ever sees the aggregate.

Success criteria:

- `paddock_chat` sends a prompt plus locally-read files to a running local model and returns only the model's answer.
- A server that is still loading is visible to `paddock_ps` and stoppable with `paddock_stop`.
- Calling `paddock_serve` twice for the same llama.cpp / mlx model reuses the running server.
- A demo video (`demo/paddock-demo.mp4` + `.gif`) recorded from a real run.

## 1. `paddock_chat`

### Input

| field | type | default | meaning |
|---|---|---|---|
| `model` | string | required | Model name substring or pid of a server listed by `paddock_ps` |
| `prompt` | string | required | Instruction for the local model |
| `files` | string[] | `[]` | Absolute paths, read by paddock, never echoed back |
| `system` | string | none | System message |
| `max_tokens` | u32 | 2048 | Completion cap |
| `temperature` | f32 | 0.2 | Low by default: extraction / classification tasks |

### Behaviour

- **No auto-serve.** `model` resolves over `list_all_servers` rows with `match_server_rows` (pid = paddock-spawned server, otherwise case-insensitive substring; covers Ollama-loaded models). Only rows with status `ready` are eligible.
  - nothing matches: error `not_serving` with `running: [model names]` and a hint to call `paddock_serve`.
  - several match: error `ambiguous_server` with candidates.
  - the match is `starting`: error `not_ready` (call again later).
- **Files.** Read locally as UTF-8. Total cap 1 MiB. A missing, unreadable or non-UTF-8 file is error `file_error` with the path and the reason, never the content. Each file is appended to the user message as:

  ````
  ### <file name>
  ```
  <content>
  ```
  ````

- **Context check.** Estimated prompt tokens = total bytes / 3.5 (conservative) + `max_tokens`. Compared with the record's ctx for llama.cpp. Unknown ctx (Ollama, mlx): no check. This is a fast pre-flight only: if the server still rejects the request (e.g. per-slot limits), its error message is surfaced through `chat_failed`. Overflow is error `context_overflow` with `estimated_tokens`, `limit` and a hint (split the files, or re-serve with a larger ctx).
- **Call.** Async `reqwest` `POST openai_url` with `{model: model_ref, messages, max_tokens, temperature, stream: false}`. Timeout 600 s. A non-2xx answer or a transport failure is error `chat_failed` with the HTTP status and the server's error message.
- **Output.** `{ text, model_ref, runtime, usage: { prompt_tokens, completion_tokens }, elapsed_ms, tok_per_s }`. A leading `<think>…</think>` block is stripped from `text`. `usage` fields are null when the server omits them; `tok_per_s` = completion_tokens / elapsed seconds, null without usage.

### Privacy contract

Stated in the tool description and in `INSTRUCTIONS`: file contents never appear in the tool result; only `text`, written by the local model, leaves. The orchestrator must therefore ask for aggregates, not for the rows. Parallelism comes from the orchestrator issuing several `paddock_chat` calls at once; servers queue them (llama.cpp slots, Ollama `OLLAMA_NUM_PARALLEL`, mlx serially).

## 2. Follow-up: starting servers are listed

`Registry::list_live` drops alive-but-not-ready records, so a loading server is invisible to `paddock_ps` and unstoppable.

- `paddock-core::serving`: new `Registry::list_alive(probe) -> Vec<LiveRecord>` where `LiveRecord { record: ServingRecord, ready: bool }`. Same reaping rules as today (unparseable or dead pid: file removed). `list_live` becomes `list_alive` filtered on `ready` (tray and existing callers unchanged).
- `ServerRow` gains `status: ServerStatus` (`ready` | `starting`, serialized lowercase). `list_all_servers` uses `list_alive`; Ollama rows are always `ready`.
- `lifecycle::resolve_servers` (stop) matches over `list_alive`, so `paddock_stop` and `paddock stop` reach a loading server.
- `paddock bench` keeps only `ready` rows before matching.
- `paddock ps` (CLI table) and the TUI servers view show starting rows; the TUI and CLI label them `starting`.

## 3. Follow-up: idempotent `serve` for spawned runtimes

In `lifecycle::serve`, `ServeMode::Detached`, for plans with `server_argv` (llama.cpp / mlx), before the port fallback: look for an alive record with the same `runtime` and `model_ref`.

- Found and ready: return `Ok(ServeOutcome)` built from the record (port, endpoint, pid, log_path, ctx) with `reused: true`. No spawn, no history write.
- Found and starting: wait for readiness within `ready_timeout` like a fresh serve would (same `wait_ready` path, no child handle), then `Ok` with `reused: true`, or `NotReady` for that pid.
- A different requested ctx or port does not force a second copy: the running server wins and the result reports its actual ctx and port. `INSTRUCTIONS` and the tool description say so (stop it first to change ctx).
- `ServeOutcome` gains `reused: bool`; the MCP result gains `"reused"`. CLI `paddock serve --detach` prints "already serving on <endpoint>" when reused.

Foreground serves are unchanged.

## 4. Demo

Directory `demo/` in the repo:

- `demo/data/`: fictitious September 2026 statements, French merchants, 3 files (`compte-courant.csv`, `compte-joint.csv`, `carte.csv`, ~60 rows each, `date;libelle;montant`). Generated by `demo/gen_data.py` (fixed seed), with a few planted findings: a forgotten subscription, a duplicate charge, one unusually large expense.
- `demo/claude/settings.json`: permissions deny `Read(./data/**)`, `Grep`, `Glob`, `Bash`; allow `mcp__paddock__*`. Claude can only reach the data through paddock.
- `demo/claude/mcp.json`: `paddock mcp` from the locally built binary.
- `demo/paddock-demo.tape`: vhs script. Launches `claude --settings … --mcp-config … --strict-mcp-config` in `demo/`, types the prompt, waits for the end, exits. Prompt (French): « Utilise paddock pour analyser mes relevés de septembre dans data/ avec un modèle local, un sous-agent par fichier, sans jamais lire les fichiers toi-même. Puis fais-moi le bilan : où part mon argent, et ce qui cloche. »
- Model: a small fast instruct model already served before recording (the video should not show a multi-GB download), picked with `paddock_fit` on this machine.
- Post-processing: `demo/render.sh` speeds up long idle stretches with ffmpeg and exports MP4 + GIF.

The video is reviewed locally; the PR and the v0.3.0 release follow its approval.

## Testing

- Unit: chat request building (file blocks, system message), `<think>` stripping, token estimate and overflow error, file errors, result shape; `list_alive` ready/starting split with a mock probe; `ServerRow.status`; serve reuse decision (pure function over records + plan).
- Integration: `paddock_chat` against a local fake OpenAI server (tokio TCP listener) for success, non-2xx, and timeout.
- Live: run against a real llama.cpp server once, then the demo itself.
