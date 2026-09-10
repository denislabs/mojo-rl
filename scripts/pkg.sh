#!/usr/bin/env bash
# The precompiled package: build it, and build or run a driver against it.
#
#   pixi run build-pkg                       # mojo precompile mojo_rl -> build/pkg/mojo_rl.mojoc
#   pixi run run-pkg examples/x.mojo [args]  # compile x against the package, then run it
#   pixi run compile-pkg examples/x.mojo -o out [mojo build flags]
#
# Why: `Import Mojo` is an uncached per-build floor that scales with the
# symbols a driver references (docs/COMPILE_TIME_PROFILING.md §3.1). The
# SAC HalfCheetah driver pays 32 s of it on EVERY rebuild from source and
# 2.5 s from the package: warm rebuild 38 s -> 8 s on Mojo 1.0, identical
# binary. The package costs ~3 min and the whole box to build, so this is
# for iterating on a driver, config or example, not on mojo_rl/ itself.
#
# Two things this script exists to get right:
#   1. The source tree SHADOWS the package whenever the working directory
#      holds `mojo_rl/`, so the compile runs from build/pkg/ with the
#      driver's absolute path (measured: 128 s from the root, 8 s from here).
#   2. Drivers find their C shims and assets relative to the project root,
#      so the BINARY is run from the root, not from build/pkg/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$ROOT/build/pkg"
MOJOC="$PKG/mojo_rl.mojoc"

usage() { sed -n 2,7p "$0" | sed 's/^# \{0,1\}//'; exit 2; }

stale_check() {
    [ -f "$MOJOC" ] || { echo "pkg.sh: no $MOJOC — run \`pixi run build-pkg\` first" >&2; exit 1; }
    local newer
    newer=$(find "$ROOT/mojo_rl" -name '*.mojo' -newer "$MOJOC" | wc -l | tr -d ' ')
    if [ "$newer" != 0 ]; then
        echo "pkg.sh: WARNING the package is STALE: $newer file(s) under mojo_rl/ are newer than it." >&2
        echo "        The driver is built against the OLD library. \`pixi run build-pkg\` to refresh." >&2
    fi
}

cmd="${1:-}"; shift || usage
case "$cmd" in
build)
    mkdir -p "$PKG"
    echo "pkg.sh: precompiling mojo_rl -> $MOJOC (about 3 min; it wants the whole box)"
    t0=$(date +%s)
    ( cd "$ROOT" && mojo precompile mojo_rl -o "$MOJOC" )
    echo "pkg.sh: done in $(( $(date +%s) - t0 )) s, $(( $(stat -f %z "$MOJOC" 2>/dev/null || stat -c %s "$MOJOC") / 1048576 )) MB"
    ;;
compile)
    # compile FILE [mojo build flags...]. The compile runs from build/pkg/, so a
    # relative `-o` is re-rooted at the project root before it gets there.
    [ $# -ge 1 ] || usage
    stale_check
    src="$(cd "$ROOT" && realpath "$1")"; shift
    args=(); prev=""
    for a in "$@"; do
        if [ "$prev" = "-o" ] && [ "${a#/}" = "$a" ]; then a="$ROOT/$a"; fi
        args+=("$a"); prev="$a"
    done
    ( cd "$PKG" && mojo build -I . "${args[@]}" "$src" )
    ;;
run)
    # run FILE [program args...]: compile into build/pkg/bin/<name>, then exec it from the root.
    [ $# -ge 1 ] || usage
    stale_check
    src="$(cd "$ROOT" && realpath "$1")"; shift
    name="$(basename "${src%.mojo}")"
    mkdir -p "$PKG/bin"
    extra=()
    # ACT-sized drivers trip the macOS linker's symbol-length assertion; -ld_classic is the known fix.
    [ "$(uname)" = Darwin ] && extra=(-Xlinker -ld_classic)
    ( cd "$PKG" && mojo build -I . "${extra[@]}" -o "$PKG/bin/$name" "$src" )
    cd "$ROOT" && exec "$PKG/bin/$name" "$@"
    ;;
*) usage ;;
esac
