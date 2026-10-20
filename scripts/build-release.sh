#!/usr/bin/env bash
# Builds a release binary with a pinned nightly: std is rebuilt optimized for
# size and panic formatting is dropped. Development uses the beta toolchain.
#
#   scripts/build-release.sh [TARGET]    # prints the binary path
set -euo pipefail
cd "$(dirname "$0")/.."

nightly=nightly-2026-09-25
target=${1:-$(rustc -vV | sed -n 's/^host: //p')}
rustup toolchain install "$nightly" --profile minimal --component rust-src >&2

# --config adds to .cargo/config.toml's rustflags; RUSTFLAGS would replace them.
cargo +"$nightly" build --locked --release --target "$target" \
    -Zbuild-std=std,panic_abort -Zbuild-std-features=optimize_for_size \
    --config "target.'cfg(all())'.rustflags=['-Zlocation-detail=none', '-Zfmt-debug=none', '-Zunstable-options', '-Cpanic=immediate-abort']" >&2
echo "target/$target/release/situ"
