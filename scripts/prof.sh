#!/usr/bin/env bash
# Measuring situ. Only `flame` needs perf_event_paranoid<=1.
#
#   scripts/prof.sh bench [FILTER]         time + instructions per case
#   scripts/prof.sh ab REF [FILTER]        git REF vs working tree, interleaved
#   scripts/prof.sh simd [FILTER]          every SIMD level vs the weakest
#   scripts/prof.sh flame [PREVIEW ARGS]   sampled profile (samply or perf)
#   scripts/prof.sh asm [SYMBOL|INDEX]     generated code (cargo-show-asm)
#   scripts/prof.sh live [SECS] [ARGS]     real session: size, RSS, idle cost
#
# Prefer instruction deltas over wall time. FILTER selects bench cases by
# substring, e.g. "frame" or "1600p".
set -euo pipefail
cd "$(dirname "$0")/.."
export BENCH_DIR=$PWD/target/bench
mkdir -p target/prof "$BENCH_DIR"

bench() { cargo bench --quiet --bench engine -- "$@"; }

cmd=${1:-help}
shift || true
case $cmd in
bench)
    bench "$@"
    ;;
ab)
    ref=${1:?usage: prof.sh ab REF [FILTER]}
    filter=${2:-}
    sha=$(git rev-parse --verify "$ref^{commit}")
    wt=target/ab/${sha:0:12}
    [ -d "$wt" ] || git worktree add --detach "$wt" "$sha" >/dev/null
    rm -f "$BENCH_DIR"/ab-{ref,new}.tsv
    for round in $(seq "${ROUNDS:-3}"); do
        echo "round $round/${ROUNDS:-3}" >&2
        (cd "$wt" && cargo bench --quiet --bench engine -- --merge ab-ref $filter >/dev/null 2>&1)
        bench --merge ab-new $filter >/dev/null 2>&1
    done
    echo "== $ref (${sha:0:12}) -> working tree"
    bench --compare ab-ref ab-new
    ;;
simd)
    levels=$(bench --levels)
    for l in $levels; do
        SITU_SIMD=$l bench --save "simd-$l" "$@" >/dev/null 2>&1
    done
    weakest=${levels##* }
    for l in $levels; do
        echo "== $l vs $weakest"
        bench --compare "simd-$weakest" "simd-$l"
    done
    ;;
flame)
    if [ "$(cat /proc/sys/kernel/perf_event_paranoid)" -gt 1 ]; then
        echo "sampling needs: sudo sysctl kernel.perf_event_paranoid=1" >&2
        exit 1
    fi
    # Frame pointers for stacks; separate target dir keeps the main cache.
    RUSTFLAGS="-C force-frame-pointers=yes" CARGO_TARGET_DIR=target/fp \
        cargo build --quiet --profile profiling --example preview
    run=(target/fp/profiling/examples/preview target/prof/frame.ppm
         --size 2560x1600 --stir --repeat 5 "$@")
    if command -v samply >/dev/null; then
        samply record -- "${run[@]}"
    else
        perf record -q -e cycles:u --call-graph fp -o target/prof/perf.data -- "${run[@]}"
        perf report -i target/prof/perf.data --stdio --no-children 2>/dev/null |
            grep -Ev '^(#|$)' | head -40
    fi
    ;;
asm)
    command -v cargo-asm >/dev/null || {
        echo "needs: cargo install cargo-show-asm" >&2
        exit 1
    }
    if [ $# -eq 0 ]; then
        # Hot kernels, one entry per SIMD level; pass the index to view one.
        cargo asm --lib --profile profiling 2>/dev/null |
            grep -E 'sim::|render::' | grep -vE 'closure|drop_in_place' || true
    else
        cargo asm --lib --profile profiling "$@"
    fi
    ;;
live)
    secs=${1:-10}
    shift || true
    cargo build --release --quiet
    bin=target/release/situ
    "$bin" "$@" &
    pid=$!
    trap 'kill $pid 2>/dev/null' EXIT
    sleep 2 # don't touch the desktop while measuring
    ctx() { awk '/^voluntary_ctxt_switches/{print $2}' "/proc/$pid/status"; }
    cpu() { awk '{print $14 + $15}' "/proc/$pid/stat"; }
    w0=$(ctx) c0=$(cpu)
    sleep "$secs"
    w1=$(ctx) c1=$(cpu)
    printf 'binary      %s KiB\n' $(($(stat -c %s "$bin") / 1024))
    awk '/^(VmRSS|RssAnon|RssShmem)/{printf "%-11s %s KiB\n", $1, $2}' "/proc/$pid/status"
    printf 'idle %ss    %s wakeups, %s ms CPU\n' "$secs" $((w1 - w0)) \
        $(((c1 - c0) * 1000 / $(getconf CLK_TCK)))
    ;;
*)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
