#!/usr/bin/env bash
# Renders the README animation (scripted workload from rest until settled,
# so the GIF loops).
#
#   scripts/gif.sh [out.gif] [situ options...]
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-assets/demo.gif}
shift || true
fps=25
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cargo build --release --quiet --example preview
target/release/examples/preview "$tmp/rest.ppm" --size 960x600 --fps $fps \
    --stir --frames "$tmp" --rows 30 "$@"

# Hold the rest image at both ends; 2-pass palette.
ffmpeg -loglevel error -y -framerate $fps -i "$tmp/%04d.ppm" \
    -vf "tpad=start_mode=clone:start_duration=0.6:stop_mode=clone:stop_duration=0.8,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" \
    -loop 0 "$out"
echo "$out: $(du -h "$out" | cut -f1)"
