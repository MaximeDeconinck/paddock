<p align="center">
  <img src="docs/assets/paddock-wordmark.png" alt="paddock" width="480">
</p>

**Which LLMs fit your Apple Silicon Mac, how fast they'll run, and how to launch them, in one command.**

![demo](docs/demo.gif)

## Why paddock

Picking a local model today means answering three questions by hand:

1. **Does it fit?** You cross-reference parameter counts, quantization formats and your RAM, then discover at load time that macOS caps how much memory the GPU may actually wire.
2. **How fast will it be?** A model that "fits" at 3 tokens/s is a model you will never use.
3. **How do I run it?** Ollama tag? `llama-server` flag soup? An `mlx_lm` invocation against a Hugging Face repo?

paddock answers all three with one command, and it is **Apple Silicon-first** by design:

- **Unified memory:** there is no separate VRAM pool to reason about, but there *is* a Metal working-set limit (`recommendedMaxWorkingSetSize`, typically ~75% of RAM) that silently decides whether a model runs fully on the GPU. paddock reads it directly and budgets against it.
- **Per-chip bandwidth:** token generation on Apple Silicon is memory-bandwidth bound, and bandwidth varies 12× across the lineup (68 GB/s on a base M1 vs 819 GB/s on an M3 Ultra). paddock ships a per-chip table covering M1 through M5 (including the perf-core-dependent M3/M4/M5 Max variants), sourced from Apple newsroom specs. Unreleased chips (M4/M5 Ultra) fall back to conservative estimates and are flagged as such. Speed estimates model the per-token traffic as active weights plus the KV cache at 8k-deep context, so the TOK/S column reflects real long-context decoding, not the empty-context best case.
- **First-class MLX:** Apple's own framework is a peer of Ollama and llama.cpp, not an afterthought: the catalog indexes `mlx-community` quantizations and `paddock run` can plan an `mlx_lm` launch.

## Install

```bash
# from a checkout of this repository
cargo install --path crates/paddock
```

A Homebrew formula is coming. Requires macOS on Apple Silicon (M1 or later).

## Usage

Running `paddock` with no arguments opens the interactive TUI. It opens instantly against the last catalog snapshot and refreshes in the background when that snapshot is more than 24h old (or empty): a spinner shows in the footer while the refresh runs, the list stays fully usable, and it swaps in the new catalog atomically when done (keeping your selection and search). Press `R` to force a refresh on demand. The blocking `paddock sync` command below is still there for scripts and cron.

Press `Tab` to switch to the servers view: everything currently running shows there, both the llama.cpp/mlx servers paddock spawned and the models loaded in the Ollama daemon, with their endpoint, context, uptime and pid. Navigate with the arrow keys; `x` stops the selected server, `c` copies its OpenAI endpoint to the clipboard. Serving a model from the TUI (`s`) now runs it detached and lands you on the servers tab, so the TUI stays open. Below the running servers, the tab lists models available locally but not loaded, greyed out: your installed Ollama models and any llama.cpp/mlx model paddock has served before. Press `enter` on a greyed model to launch it.

On the models tab, `enter` opens a model's detail popup, which lists every quantization it ships with their memory, tok/s and fit; use the arrow keys to pick a smaller quant for more speed and less memory, then `x` to run or `s` to serve the chosen one. Without opening the detail, `x`/`s` use the best quant that fits.

Nine subcommands cover everything scriptable:

### `paddock scan`: what is this machine?

```text
$ paddock scan
chip        Apple M5
ram         32.0 GiB
cores       4P + 6E
bandwidth   154 GB/s
gpu limit   25.0 GiB (Metal recommendedMaxWorkingSetSize)
runtimes    ollama 0.30.6 (running)
            llama.cpp not installed
            mlx-lm not installed
```

### `paddock sync`: refresh the model catalog

```text
$ paddock sync
synced: 78 curated (461 ollama tags), 44 discovered, 84 huggingface, 20 mlx
```

Four sources:

1. **Curated Ollama models:** an embedded list of ~78 popular text-generation models with hand-checked architecture parameters (layers, KV heads, context). Always works offline.
2. **Discovered Ollama models:** the top of the Ollama library *beyond* the curated list, fully automatic: for each model, sync reads the tag list, fetches one OCI manifest from `registry.ollama.ai`, and Range-reads the first 256 KiB of the weights blob to parse the real GGUF header (architecture, layers, KV heads, parameter count, context). No curation needed. Brand-new architectures show up with correct fit estimates. Cloud-only builds (`*-cloud`) and embedding/vision-encoder architectures are skipped.
3. **Hugging Face GGUF:** the most-downloaded GGUF repos, with architecture details read from GGUF headers via HTTP Range requests (no full downloads).
4. **mlx-community:** Apple-native MLX quantizations.

On top of the curated list, sync enriches each model with the **live tag list of the Ollama library** (one request per model family): it extracts tag names from the URL pattern on `ollama.com/library/{model}/tags` and keeps one tag per known quantization (`8b-instruct-q4_K_M`, `8b-instruct-q8_0`, …), so `paddock run` launches the exact tag instead of the library default. This step is best-effort: registry errors degrade to warnings and the curated data stands.

Flags: `--hf-limit N` (default 100) and `--mlx-limit N` (default 60) bound the network sources; `--discover-limit N` (default 60) bounds library auto-discovery and `--no-discover` turns it off; `--no-ollama-registry` skips the live tag enrichment for a fully offline-friendly sync.

### `paddock fit`: what fits, ranked

```text
$ paddock fit -n 8
MODEL                               AGE QUANT        MEMORY     TOK/S  FIT          SCORE
nemotron3:33b                         ? Q4_K_M     21.2 GiB        49  fits            91
nemotron-cascade-2:30b                ? Q4_K_M     23.4 GiB        52  fits            90
glm-4.7-flash:q4_K_M                ~5d Q4_K_M     21.1 GiB        41  fits            89
qwen3-coder:30b                    ~7mo Q4_K_M     18.9 GiB        40  fits            89
Qwen3.6-35B-A3B-Uncensored-Hauh…     7w Q4_K_M     21.3 GiB        25  fits            86
Qwen3.6-35B-A3B-GGUF                 7w Q4_K_M     21.3 GiB        25  fits            86
Qwen3.6-35B-A3B-MTP-GGUF             4w UD-Q4_K_XL  23.3 GiB        24  fits            85
gpt-oss:20b                        ~7mo Q4_K_M     14.2 GiB        27  fits            85
```

For each model paddock picks the best quantization that fits (starting from the highest-quality quant and walking down toward Q2 only as memory demands), then estimates memory and speed and ranks. `--all` includes models that don't fit, `--use-case coding|chat|reasoning|general` changes the scoring weights, `-n` limits rows.

Ranking is age-aware: the quality sub-score takes a progressive malus once a model is older than six months (−10 points per year, capped at −20), so a year-old model sinks below a fresh one of comparable size without disappearing from the list. The AGE column shows how old each model is: `~` marks approximate dates (inferred from the Ollama tags page), `?` means no release date is known.

### `paddock recommend`: top 5 with reasons

```text
$ paddock recommend --use-case coding
Qwen3.6-35B-A3B-MTP-GGUF Q4_K_M - 78/100: fits GPU with 2.6 GiB to spare, ~64 tok/s (instant), 256k context
Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive Q4_K_M - 78/100: fits GPU with 3.1 GiB to spare, ~64 tok/s (instant), 256k context
Qwen3.6-35B-A3B-GGUF Q4_K_M - 78/100: fits GPU with 3.1 GiB to spare, ~64 tok/s (instant), 256k context
gpt-oss-20b-MXFP4-Q8 MLX_4BIT - 77/100: fits GPU with 13.2 GiB to spare, ~57 tok/s (instant), 128k context
Qwen3-30B-A3B-Instruct-2507-GGUF Q5_K_M - 75/100: fits GPU with 2.9 GiB to spare, ~49 tok/s (instant), 256k context
```

### `paddock run <model>`: launch it

```text
$ paddock run llama3.2:3b
$ ollama run llama3.2:3b
>>>
```

paddock picks the runtime based on the variant's source: GGUF models (Hugging Face or the Ollama library) launch via Ollama, MLX models via mlx-lm (llama.cpp is detected and displayed but never auto-selected in v0.1). It prints the exact command, and execs it. If no capable runtime is installed it proposes the install command and asks for confirmation before doing anything. Unknown names produce a real error, not a shrug:

```text
$ paddock run definitely-not-a-model
Error: model `definitely-not-a-model` not found in catalog. Run `paddock sync` or check `paddock fit` for names.
```

`--quant <label>` (for example `--quant Q4_K_M`) launches a specific quantization instead of the auto-picked best fit. Any quant the model offers is selectable, even one that does not fit this machine: it launches anyway and the fit verdict is informational. An unknown label errors and lists the model's available quants. `run` and `serve` both take it.

### `paddock serve <model>`: OpenAI-compatible endpoint

Same model resolution as `run`, but instead of an interactive chat you get an HTTP endpoint any OpenAI-compatible client can talk to:

```text
$ paddock serve llama3.2:1b
$ ollama pull llama3.2:1b
success
endpoint    http://127.0.0.1:11434
openai      http://127.0.0.1:11434/v1/chat/completions
model       llama3.2:1b
try it      curl -s http://127.0.0.1:11434/v1/chat/completions \
              -d '{"model":"llama3.2:1b","messages":[{"role":"user","content":"hello"}]}'
```

By default `serve` runs detached: paddock spawns the server in the background, waits until it answers its readiness check, prints the endpoint, and returns to your shell. No dedicated terminal stays open. The server keeps running, and paddock records it in a registry so `paddock ps`, `paddock stop`, and `paddock logs` (below) can find it later:

```text
$ paddock serve qwen3-8b
…
endpoint    http://127.0.0.1:8080
serving in background · pid 51234 · paddock logs qwen3-8b
```

Pass `--foreground` (`-f`) to keep the old attached behavior instead: paddock stays in the foreground, streams the server's logs to your terminal, and **Ctrl-C stops the server**.

The lifecycle depends on the runtime:

- **Ollama models** reuse the daemon: paddock pulls the model, prints the endpoint, and the daemon keeps serving in the background on its fixed port 11434. Ollama-loaded models are managed by `ollama ps` / `ollama stop`, not paddock's lifecycle commands. (In `--foreground`, if paddock had to cold-start `ollama serve`, Ctrl-C stops that daemon.)
- **llama.cpp / mlx-lm models** spawn a server (`llama-server -hf …` / `mlx_lm.server --model …`) that paddock owns. By default it runs detached (logs go to a file you can read with `paddock logs`); with `--foreground` it stays attached and Ctrl-C stops it. llama-server may download the model first; paddock waits through that. llama.cpp commands skip vision projectors with `--no-mmproj` (text-only v0.1).

Context auto-sizes by default. For llama.cpp, paddock picks the largest context window that fits this machine's memory budget rather than the model default (which can be far larger, 262k on some models, and OOM), which is what prevents `request (N tokens) exceeds the available context size` errors. Override it with `--ctx <tokens>`. Ollama and MLX manage their own context, so `--ctx` is llama.cpp-only.

`--port <N>` picks the port for llama.cpp / mlx servers (default 8080). When the chosen port is already taken, paddock serves on the next free one (so serving several models never collides), printing where it landed. The Ollama daemon's port is fixed, so `--port` is ignored there with a warning. `--json` prints the full serve plan (argv, endpoint, pre-steps) without spawning or pulling anything.

### `paddock ps`, `paddock stop`, `paddock logs`: manage running servers

These cover the llama.cpp / mlx servers paddock spawned (Ollama-loaded models are managed by `ollama ps` / `ollama stop`).

`paddock ps` groups what it shows into two sections, matching the TUI servers tab: a `RUNNING` section (the llama.cpp / mlx servers paddock spawned plus the models loaded in the Ollama daemon) and an `AVAILABLE` section (your installed Ollama models and previously served catalog entries that aren't currently running, ready to relaunch):

```text
$ paddock ps
RUNNING
MODEL          RUNTIME    ENDPOINT                CTX   UPTIME   PID
qwen3-8b       llama.cpp  http://127.0.0.1:8080  32768     2m    51234

AVAILABLE
MODEL          RUNTIME
llama3.2:3b    ollama
```

`--json` emits the same data machine-readable for scripts, as `{ "running": [...], "available": [...] }`.

`paddock stop <target>` stops a server by model name, pid, or `all`:

```text
$ paddock stop qwen3-8b
stopped qwen3-8b (pid 51234)
```

`stop all` asks for confirmation before stopping everything; `-y` (`--yes`) skips the prompt.

`paddock logs <target>` prints a detached server's log (by model name or pid). Pass `-f` (`--follow`) to follow it live, like `tail -f`.

Some Hugging Face repos ship their vision projector as a separate `mmproj-*.gguf` file (Unsloth's Qwen3.6 uploads, for example). Ollama cannot import those repos via `hf.co/…` ([ollama/ollama#15447](https://github.com/ollama/ollama/issues/15447)); it would download the full weights and then fail. paddock detects the `mmproj` file at sync time, marks every variant of the repo llama.cpp-only, and serves it with `llama-server` (or proposes `brew install llama.cpp`) even when Ollama is installed and running. Run `paddock sync` to refresh these compatibility flags.

### `paddock tray`: menu bar (macOS)

```sh
paddock tray
```

Puts a small "P" in the macOS menu bar listing every active serve endpoint. Discovery is hybrid: servers launched by `paddock serve` (llama-server, mlx-lm, an `ollama serve` booted by paddock) plus whatever the local Ollama daemon reports via `/api/ps`, so models loaded outside paddock show up too. Each section is `runtime - host:port` with one row per model; **clicking a model row copies its OpenAI-compatible URL** to the clipboard. "Refresh" re-scans immediately; otherwise the menu refreshes every 5 seconds.

v1 limits: macOS only, no stop-server action, no login item (launch it manually), and running two instances gives you two icons.

### `--json` everywhere

Every subcommand takes `--json` for machine-readable output; `paddock run x --json` prints the planned `argv` without launching anything:

```json
{
  "argv": ["ollama", "run", "llama3.2:3b"],
  "install": null
}
```

`--cli` forces plain tables when you want to skip the TUI in a terminal.

### TUI keys

Two tabs: **models** (the ranking) and **servers** (what is running, plus an *available* group of installed Ollama models and previously served catalog entries you can relaunch). `tab` switches between them.

**Models tab:**

| Key | Action |
|-----|--------|
| `↑`/`↓` | move selection |
| `Enter` | open detail view (memory breakdown, run plan) |
| `x` | run the selected model |
| `s` | serve the selected model (OpenAI-compatible endpoint) |
| `/` | search (type to filter, `Enter` apply, `Esc` clear) |
| `g` / `c` / `r` / `h` | re-score for general / coding / reasoning / chat |
| `tab` | switch to the servers tab |
| `q`, `Ctrl-C` | quit |

**Servers tab:**

| Key | Action |
|-----|--------|
| `↑`/`↓` | move selection |
| `Enter` | relaunch the selected available model |
| `x` | stop the selected running server |
| `c` | copy the endpoint URL |
| `tab` | switch to the models tab |
| `q`, `Ctrl-C` | quit |

## How the estimates work

This is the product, so it is fully transparent. Every number paddock prints comes from the formulas below: no magic, and no fudge factors beyond the ones documented here.

### Memory

Three components, summed:

```text
weights  = params_total × bpw / 8
kv_cache = 2 × layers × kv_heads × head_dim × context × 2 bytes    (fp16 K+V, GQA-aware)
overhead = 500 MiB + 5% × weights                                  (Metal heaps, buffers, tokenizer)
```

The KV cache uses `kv_heads`, not the full attention head count: grouped-query-attention models (nearly everything modern) need 4–8× less cache than the naive formula suggests, which is often the difference between "fits" and "doesn't". For Llama 3.1 8B at 8k context this gives 1.0 GiB, matching what llama.cpp actually allocates.

### Speed

Token generation on Apple Silicon is memory-bandwidth bound: every generated token streams all active weights through the memory bus once. So:

```text
tps = bandwidth / (params_active × bpw / 8) × efficiency
```

`params_active` equals total parameters for dense models and the active-expert count for MoE models, which is why a 30B-A3B MoE generates faster than a dense 8B. The efficiency factor is the fraction of theoretical bandwidth that llama.cpp/MLX kernels actually sustain, and it differs by architecture: **0.75 for dense models** (calibrated against community benchmarks, e.g. Llama 3.1 8B Q4_K_M on an M2 Max: predicted ~62 tok/s, field reports 55–70) and **0.3 for MoE models** (calibrated on a real measurement: Qwen3.6-35B-A3B UD-Q4_K_XL on an M5 measured 22.6 tok/s vs ~23.7 predicted; community Qwen3-30B-A3B numbers on M3 Max imply the same ≈0.3). Expert routing scatters weight reads across memory, so MoE kernels get nowhere near the dense streaming case, and MoE estimates carry more variance than dense ones. Prompt processing is compute-bound, not bandwidth-bound; paddock reports it as a rough 5–10× multiple of generation speed.

**Stated plainly:** these are bandwidth-bound theoretical estimates at fixed efficiency factors. They are good enough to rank models and avoid wasted downloads; they are not a benchmark. A real `paddock bench` module that measures *your* machine and recalibrates the factors per-device is on the roadmap.

### Bits per weight

Quantization labels hide metadata overhead, so paddock uses effective bpw, not nominal:

| Quant | bpw | Quant | bpw |
|-------|------|-------|------|
| F16 / BF16 | 16.0 | Q4_K_M | 4.83 |
| Q8_0 | 8.5 | Q4_0 | 4.55 |
| MLX 8-bit | 8.5 | MLX 4-bit | 4.5 |
| Q6_K | 6.59 | IQ4_XS | 4.25 |
| Q5_K_M | 5.69 | Q3_K_M | 3.91 |
| | | Q2_K | 3.35 |

### Fit verdicts

Total memory is checked against a ladder, because "fits" is not binary on macOS:

1. **`fits`**: total ≤ the Metal working-set limit. Fully GPU-resident, full speed.
2. **`tune sysctl`**: exceeds the default GPU limit but fits within `RAM − 4 GiB`. macOS lets you raise the GPU wiring cap with `sudo sysctl iogpu.wired_limit_mb=<MB>`; paddock keeps a 4 GiB system reserve when judging this, and the detail view shows the exact command.
3. **`ram only`**: fits in total RAM only, with partial CPU offload. It will run, degraded. (v0.1 does not model the offload speed penalty.)
4. **`no fit`**: don't bother downloading.

### Experience tiers

| tok/s | Tier |
|-------|------|
| > 30 | instant |
| 15–30 | smooth |
| 5–15 | usable |
| < 5 | slow |

### Scoring

Four sub-scores (fit, speed, quality, context), each 0–100, combined as a **weighted geometric mean**, not an arithmetic average:

```text
total = 100 × Π (max(subscore, 1) / 100)^(weight / 100)
```

Geometric, because on a capable machine fit/speed/context all saturate near 100 for small models; with an arithmetic average a terrible quality score could only subtract its own weight, flooring totals around 75. With the geometric mean, one bad component drags the whole total down and saturated components can't compensate. The quality proxy is `log10(params)` normalized over the local-model range (1B → 0, ~70B → 100, clamped outside the 1B–70B range), minus a malus for sub-Q4/sub-Q3 quants. Fit is a gate, not a reward: fitting the GPU scores 85 plus up to 15 for headroom.

## Roadmap

- **Tauri desktop app** on top of `paddock-core`
- **`paddock bench`**: real measured tok/s, per-machine recalibration of the efficiency factor, offload-penalty modeling
- **MCP server**: let coding agents ask "what can this machine run?"
- **Linux / Windows**: same idea, different memory model (discrete VRAM, CUDA/ROCm)

## License

Apache-2.0. See [LICENSE](LICENSE).
