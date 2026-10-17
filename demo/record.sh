#!/usr/bin/env bash
# Record the paddock demo video from a real run.
#
#   demo/record.sh            build paddock, stage ~/releves, record, render
#
# Needs: vhs, ffmpeg, Claude Code (logged in), and the demo model already
# pulled so the video shows no multi-GB download:
#   paddock serve Qwen3-30B-A3B-Instruct-2507-GGUF --quant Q4_K_M
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
stage="$HOME/releves"
out="$repo/demo/out"

cargo build --release --manifest-path "$repo/Cargo.toml" -q
python3 -I "$repo/demo/gen_data.py" >/dev/null

# The recorded folder: data + project settings, nothing else.
rm -rf "$stage"
mkdir -p "$stage" "$out"
cp -R "$repo/demo/workspace/." "$stage/"
cp "$repo/demo/mcp.json" "$stage/.mcp-demo.json"

# A clean Claude Code: no variables inherited from a parent Claude session.
for v in $(env | grep -E '^(CLAUDE|CLAUDECODE)' | cut -d= -f1); do unset "$v"; done
export PATH="$repo/target/release:$PATH"

cd "$stage"
vhs "$repo/demo/paddock-demo.tape"
mv "$stage/.raw.mp4" "$out/raw.mp4"
"$repo/demo/render.sh" "$out/raw.mp4"
