#!/usr/bin/env bash
# Turn the raw vhs recording into the shareable demo.
#
#   demo/render.sh demo/out/raw.mp4
#
# Near-identical frames (waiting on the models, spinner only) are dropped,
# at most $SKIP in a row, so idle stretches play ~$SKIP x faster while text
# appearing on screen keeps its pace. Writes paddock-demo.mp4 and .gif next
# to the input.
set -euo pipefail

raw="$1"
dir="$(dirname "$raw")"
skip="${SKIP:-12}"

ffmpeg -y -loglevel error -i "$raw" \
  -vf "mpdecimate=max=$skip,setpts=N/FRAME_RATE/TB" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart -an \
  "$dir/paddock-demo.mp4"

ffmpeg -y -loglevel error -i "$dir/paddock-demo.mp4" \
  -vf "fps=12,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
  "$dir/paddock-demo.gif"

for f in "$raw" "$dir/paddock-demo.mp4"; do
  printf '%s: %ss\n' "$(basename "$f")" \
    "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" | cut -d. -f1)"
done
ls -lh "$dir/paddock-demo.mp4" "$dir/paddock-demo.gif" | awk '{print $5, $9}'
