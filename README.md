# tetro

**Which LLMs fit your Apple Silicon Mac, how fast they'll run, and how to launch them — in one command.**

![demo](docs/demo.gif)
<!-- TODO: record with vhs -->

## Why tetro

Picking a local model today means answering three questions by hand:

1. **Does it fit?** You cross-reference parameter counts, quantization formats and your RAM, then discover at load time that macOS caps how much memory the GPU may actually wire.
2. **How fast will it be?** A model that "fits" at 3 tokens/s is a model you will never use.
3. **How do I run it?** Ollama tag? `llama-server` flag soup? An `mlx_lm` invocation against a Hugging Face repo?

tetro answers all three with one command, and it is **Apple Silicon-first** by design:

- **Unified memory** — there is no separate VRAM pool to reason about, but there *is* a Metal working-set limit (`recommendedMaxWorkingSetSize`, typically ~75% of RAM) that silently decides whether a model runs fully on the GPU. tetro reads it directly and budgets against it.
- **Per-chip bandwidth** — token generation on Apple Silicon is memory-bandwidth bound, and bandwidth varies 12× across the lineup (68 GB/s on a base M1 vs 819 GB/s on an M3 Ultra). tetro ships a per-chip table covering M1 through M5 — including the perf-core-dependent M3/M4/M5 Max variants — sourced from Apple newsroom specs. Unreleased chips (M4/M5 Ultra) fall back to conservative estimates and are flagged as such.
- **First-class MLX** — Apple's own framework is a peer of Ollama and llama.cpp, not an afterthought: the catalog indexes `mlx-community` quantizations and `tetro run` can plan an `mlx_lm` launch.

## Install

```bash
# from a checkout of this repository
cargo install --path crates/tetro
```

A Homebrew formula is coming. Requires macOS on Apple Silicon (M1 or later).

## Usage

Running `tetro` with no arguments opens the interactive TUI. Five subcommands cover everything scriptable:

### `tetro scan` — what is this machine?

```text
$ tetro scan
chip        Apple M5
ram         32.0 GiB
cores       4P + 6E
bandwidth   154 GB/s
gpu limit   25.0 GiB (Metal recommendedMaxWorkingSetSize)
runtimes    ollama 0.30.6 (running)
            llama.cpp not installed
            mlx-lm not installed
```

### `tetro sync` — refresh the model catalog

```text
$ tetro sync
synced: 40 curated, 26 huggingface, 5 mlx
```

Three sources: an embedded curated list of popular Ollama models, the most-downloaded GGUF repos on Hugging Face (architecture details read from GGUF headers via HTTP Range requests — no full downloads), and `mlx-community` quantizations. Network failures degrade to warnings; the curated list always works offline.

### `tetro fit` — what fits, ranked

```text
$ tetro fit -n 8
MODEL                            QUANT        MEMORY     TOK/S  FIT          SCORE
gpt-oss-20b-MXFP4-Q8             MLX_4BIT   11.8 GiB        57  fits            81
Qwen3.6-35B-A3B-Uncensored-Hauh… Q4_K_M     21.9 GiB        64  fits            79
Qwen3.6-35B-A3B-GGUF             Q4_K_M     21.9 GiB        64  fits            79
Qwen3.6-35B-A3B-MTP-GGUF         Q4_K_M     22.4 GiB        64  fits            79
llama3.2:3b                      Q4_K_M      3.3 GiB        59  fits            78
Qwen3-30B-A3B-Instruct-2507-GGUF Q5_K_M     22.1 GiB        49  fits            77
gemma3:4b                        Q4_K_M      4.1 GiB        44  fits            77
phi3:3.8b                        Q4_K_M      5.7 GiB        50  fits            76
```

For each model tetro picks the best quantization that fits — starting from the highest-quality quant and walking down toward Q2 only as memory demands — then estimates memory and speed and ranks. `--all` includes models that don't fit, `--use-case coding|chat|reasoning|general` changes the scoring weights, `-n` limits rows.

### `tetro recommend` — top 5 with reasons

```text
$ tetro recommend --use-case coding
Qwen3.6-35B-A3B-MTP-GGUF Q4_K_M — 78/100: fits GPU with 2.6 GiB to spare, ~64 tok/s (instant), 256k context
Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive Q4_K_M — 78/100: fits GPU with 3.1 GiB to spare, ~64 tok/s (instant), 256k context
Qwen3.6-35B-A3B-GGUF Q4_K_M — 78/100: fits GPU with 3.1 GiB to spare, ~64 tok/s (instant), 256k context
gpt-oss-20b-MXFP4-Q8 MLX_4BIT — 77/100: fits GPU with 13.2 GiB to spare, ~57 tok/s (instant), 128k context
Qwen3-30B-A3B-Instruct-2507-GGUF Q5_K_M — 75/100: fits GPU with 2.9 GiB to spare, ~49 tok/s (instant), 256k context
```

### `tetro run <model>` — launch it

```text
$ tetro run llama3.2:3b
$ ollama run llama3.2:3b
>>>
```

tetro picks the runtime based on the variant's source — GGUF models (Hugging Face or the Ollama library) launch via Ollama, MLX models via mlx-lm (llama.cpp is detected and displayed but never auto-selected in v0.1) — prints the exact command, and execs it. If no capable runtime is installed it proposes the install command and asks for confirmation before doing anything. Unknown names produce a real error, not a shrug:

```text
$ tetro run definitely-not-a-model
Error: model `definitely-not-a-model` not found in catalog. Run `tetro sync` or check `tetro fit` for names.
```

### `--json` everywhere

Every subcommand takes `--json` for machine-readable output; `tetro run x --json` prints the planned `argv` without launching anything:

```json
{
  "argv": ["ollama", "run", "llama3.2:3b"],
  "install": null
}
```

`--cli` forces plain tables when you want to skip the TUI in a terminal.

### TUI keys

| Key | Action |
|-----|--------|
| `↑`/`↓` or `j`/`k` | move selection |
| `Enter` | open detail view (memory breakdown, run plan) |
| `x` | run the selected model |
| `/` | search (type to filter, `Enter` apply, `Esc` clear) |
| `g` / `c` / `r` / `h` | re-score for general / coding / reasoning / chat |
| `q`, `Ctrl-C` | quit |

## How the estimates work

This is the product, so it is fully transparent. Every number tetro prints comes from the formulas below — no magic, and no fudge factors beyond the ones documented here.

### Memory

Three components, summed:

```text
weights  = params_total × bpw / 8
kv_cache = 2 × layers × kv_heads × head_dim × context × 2 bytes    (fp16 K+V, GQA-aware)
overhead = 500 MiB + 5% × weights                                  (Metal heaps, buffers, tokenizer)
```

The KV cache uses `kv_heads`, not the full attention head count: grouped-query-attention models (nearly everything modern) need 4–8× less cache than the naive formula suggests, which is often the difference between "fits" and "doesn't". For Llama 3.1 8B at 8k context this gives 1.0 GiB — matching what llama.cpp actually allocates.

### Speed

Token generation on Apple Silicon is memory-bandwidth bound: every generated token streams all active weights through the memory bus once. So:

```text
tps = bandwidth / (params_active × bpw / 8) × 0.75
```

`params_active` equals total parameters for dense models and the active-expert count for MoE models — which is why a 30B-A3B MoE generates faster than a dense 8B. The **0.75 efficiency factor** is the fraction of theoretical bandwidth that llama.cpp/MLX kernels actually sustain, calibrated against community benchmarks (e.g. Llama 3.1 8B Q4_K_M on an M2 Max: predicted ~62 tok/s, field reports 55–70). Prompt processing is compute-bound, not bandwidth-bound; tetro reports it as a rough 5–10× multiple of generation speed.

**Stated plainly:** these are bandwidth-bound theoretical estimates at a fixed 75% efficiency. They are good enough to rank models and avoid wasted downloads; they are not a benchmark. A real `tetro bench` module that measures *your* machine and recalibrates the factor per-device is on the roadmap.

### Bits per weight

Quantization labels hide metadata overhead, so tetro uses effective bpw, not nominal:

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

1. **`fits`** — total ≤ the Metal working-set limit. Fully GPU-resident, full speed.
2. **`tune sysctl`** — exceeds the default GPU limit but fits within `RAM − 4 GiB`. macOS lets you raise the GPU wiring cap with `sudo sysctl iogpu.wired_limit_mb=<MB>`; tetro keeps a 4 GiB system reserve when judging this, and the detail view shows the exact command.
3. **`ram only`** — fits in total RAM only, with partial CPU offload. It will run, degraded. (v0.1 does not model the offload speed penalty.)
4. **`no fit`** — don't bother downloading.

### Experience tiers

| tok/s | Tier |
|-------|------|
| > 30 | instant |
| 15–30 | smooth |
| 5–15 | usable |
| < 5 | slow |

## Roadmap

- **Tauri desktop app** on top of `tetro-core`
- **`tetro bench`** — real measured tok/s, per-machine recalibration of the efficiency factor, offload-penalty modeling
- **MCP server** — let coding agents ask "what can this machine run?"
- **Linux / Windows** — same idea, different memory model (discrete VRAM, CUDA/ROCm)

## License

Apache-2.0. See [LICENSE](LICENSE).
