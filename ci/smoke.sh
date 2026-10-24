#!/usr/bin/env bash
# Smoke test on headless sway at scale 1.5: wallpaper appears, a second
# instance replaces the first, and the process goes idle once settled.
set -euo pipefail
bin=$(realpath "$1")
out=target/smoke
mkdir -p "$out"

export XDG_RUNTIME_DIR=$(mktemp -d)
chmod 700 "$XDG_RUNTIME_DIR"
cat >"$out/sway.conf" <<'EOF'
output HEADLESS-1 resolution 1280x720 scale 1.5
EOF
WLR_BACKENDS=headless WLR_RENDERER=pixman WLR_LIBINPUT_NO_DEVICES=1 \
    sway -c "$out/sway.conf" >"$out/sway.log" 2>&1 &
sway_pid=$!
trap 'kill $sway_pid 2>/dev/null || true' EXIT

for _ in $(seq 50); do
    sock=$(find "$XDG_RUNTIME_DIR" -maxdepth 1 -name 'wayland-*' ! -name '*.lock' | head -1)
    [ -n "$sock" ] && break
    sleep 0.1
done
[ -n "${sock:-}" ] || { cat "$out/sway.log"; echo "sway did not start"; exit 1; }
export WAYLAND_DISPLAY=$(basename "$sock")

# Fraction of pixels that differ from the most common (background) color.
coverage() {
    python3 - "$1" <<'EOF'
import sys
from collections import Counter
data = open(sys.argv[1], 'rb').read()
parts = data.split(b'\n', 3)  # P6, "w h", maxval, pixels
w, h = map(int, parts[1].split())
px = parts[3]
counts = Counter(px[i:i + 3] for i in range(0, w * h * 3, 3))
bg, n = counts.most_common(1)[0]
print(f"{w}x{h} colors={len(counts)} lit={1 - n / (w * h):.3f}")
sys.exit(0 if len(counts) > 3 and 0.02 < 1 - n / (w * h) < 0.9 else 1)
EOF
}

"$bin" --color e8be7c &
first=$!
sleep 2
"$bin" --color 2897a1 --rows 40 &
second=$!
sleep 2
if kill -0 "$first" 2>/dev/null; then
    echo "first instance was not replaced"
    exit 1
fi
kill -0 "$second" || { echo "second instance died"; exit 1; }

grim -t ppm "$out/screen.ppm"
coverage "$out/screen.ppm"

ticks() { awk '{print $14 + $15}' "/proc/$second/stat"; }
t0=$(ticks)
sleep 3
t1=$(ticks)
echo "idle CPU ticks over 3s: $((t1 - t0))"
[ $((t1 - t0)) -le 1 ] || { echo "not idle"; exit 1; }

# Settled for >3s: buffers the compositor released are freed (how many it
# keeps is its choice, so this is informational), and the screen must not
# change: the compositor keeps showing what it was given.
shm=$(awk '/^RssShmem:/ {print $2}' "/proc/$second/status")
echo "settled shm RSS: ${shm} kB"
grim -t ppm "$out/trimmed.ppm"
coverage "$out/trimmed.ppm"
cmp -s "$out/screen.ppm" "$out/trimmed.ppm" || { echo "screen changed after trim"; exit 1; }

kill -TSTP "$second"
sleep 0.5
if kill -0 "$second" 2>/dev/null; then
    echo "Ctrl+Z did not exit"
    exit 1
fi
echo "smoke test passed"
