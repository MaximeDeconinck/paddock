#!/usr/bin/env bash
# Turn the raw vhs recording into the shareable demo.
#
#   demo/render.sh demo/out/raw.mp4
#
# 1. Of frames where only the spinner / timer moved, one in $SKIP is kept, so
#    waiting on the models plays fast while text appearing keeps its pace.
# 2. The raw is cut where the screen last changed; the result is sped up further if needed to fit $TARGET seconds, then the
#    last frame is held $HOLD seconds so the final answer can be read.
# 3. The terminal is composited into a macOS window (demo/frame.py).
# Writes paddock-demo.mp4 and .gif next to the input.
set -euo pipefail

raw="$1"
dir="$(dirname "$raw")"
here="$(cd "$(dirname "$0")" && pwd)"
target="${TARGET:-24}"
hold="${HOLD:-10}"
skip="${SKIP:-15}"
# Window content origin, from demo/frame.py (CONTENT_X, CONTENT_Y).
cx=150
cy=116

dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }

python3 -I "$here/frame.py" "$dir/frame.png"

# The screen stops changing once the answer is out: cut the raw there.
decimate="mpdecimate=hi=6400:lo=1600:frac=0.01"
# (scene score, not mpdecimate: the blinking cursor never stops "changing").
last="$(ffmpeg -i "$raw" -vf "select='gt(scene,0.001)',showinfo" -f null - 2>&1 |
  sed -n 's/.*pts_time:\([0-9.]*\).*/\1/p' | tail -1)"
end="$(python3 -c "print($last + 1.0)")"

ffmpeg -y -loglevel error -t "$end" -i "$raw" \
  -vf "$decimate:max=$skip,setpts=N/FRAME_RATE/TB" -an \
  -c:v libx264 -crf 12 "$dir/.decimated.mp4"

speed="$(python3 -c "print(max(1.0, $(dur "$dir/.decimated.mp4") / $target))")"

ffmpeg -y -loglevel error -i "$dir/.decimated.mp4" -loop 1 -i "$dir/frame.png" \
  -filter_complex "\
[0:v]setpts=PTS/$speed,fps=30,tpad=stop_mode=clone:stop_duration=$hold[term];\
color=c=black:s=1920x1080:r=30[bg];\
[bg][term]overlay=$cx:$cy:shortest=1[mid];\
[mid][1:v]overlay=0:0:shortest=1,format=yuv420p" \
  -c:v libx264 -crf 18 -preset slow -movflags +faststart -an \
  "$dir/paddock-demo.mp4"

ffmpeg -y -loglevel error -i "$dir/paddock-demo.mp4" \
  -vf "fps=15,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
  "$dir/paddock-demo.gif"

rm -f "$dir/.decimated.mp4"
printf 'raw %ss -> demo %ss (x%s)\n' "$(dur "$raw" | cut -d. -f1)" "$(dur "$dir/paddock-demo.mp4" | cut -d. -f1)" "$speed"
ls -lh "$dir/paddock-demo.mp4" "$dir/paddock-demo.gif" | awk '{print $5, $9}'
