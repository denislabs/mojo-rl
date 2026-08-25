#!/usr/bin/env bash
# Build the qhull shim for `mojo_rl/physics3d/collision/`.
#
# The dylib is what the Mojo side dlopen's (`collision/qhull_native.mojo`).
# ⚠ A DYLIB AND NOT AN OBJECT, for the reason `mojo_rl/io/serial/native.mojo`
# records: `mojo run`'s JIT does not honour `-Xlinker` at all, and every test
# in this repo runs under `mojo run`. Resolving through `_get_dylib_function`
# (the stdlib's own `dlsym`) is the only route that works in both.
#
# Not tracked in git. Re-run after editing native/mrl_qhull.c.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/mojo_rl/physics3d/collision/native/mrl_qhull.c"
OUTDIR="$ROOT/mojo_rl/physics3d/collision"
# ⚠ FIND THE PREFIX THAT ACTUALLY HAS THE HEADER, do not trust $CONDA_PREFIX.
# Outside `pixi run` it points at the user's own miniforge, which has no
# libqhull_r — the first version of this script took it and failed with a
# "file not found" that looked like a missing dependency rather than a wrong
# prefix. Candidates in priority order; QHULL_PREFIX overrides everything.
PREFIX=""
for c in "${QHULL_PREFIX:-}" "$ROOT/.pixi/envs/default" "${CONDA_PREFIX:-}" \
         "$ROOT/.pixi/envs/apple" "$ROOT/.pixi/envs/nvidia" /usr/local /usr; do
    [[ -n "$c" && -f "$c/include/libqhull_r/qhull_ra.h" ]] && { PREFIX="$c"; break; }
done
if [[ -z "$PREFIX" ]]; then
    echo "build_qhull.sh: libqhull_r headers not found." >&2
    echo "  looked for include/libqhull_r/qhull_ra.h under:" >&2
    echo "  \$QHULL_PREFIX, .pixi/envs/{default,apple,nvidia}, \$CONDA_PREFIX, /usr/local, /usr" >&2
    echo "  qhull is a declared dependency in pixi.toml; try \`pixi install\`." >&2
    exit 1
fi
echo "  qhull prefix: $PREFIX"

case "$(uname -s)" in
Darwin) LIB="$OUTDIR/libmrl_qhull.dylib" ;;
Linux)  LIB="$OUTDIR/libmrl_qhull.so" ;;
*) echo "build_qhull.sh: unsupported OS $(uname -s)" >&2; exit 1 ;;
esac

if [[ "$SRC" -ot "$LIB" && -f "$LIB" ]]; then
    echo "qhull shim up to date: $LIB"
    exit 0
fi

CC="${CC:-cc}"
echo "building qhull shim from $SRC"
# ⚠ RPATH TO THE ENV, so the dylib finds libqhull_r wherever pixi put it.
"$CC" -O2 -fPIC -shared -I"$PREFIX/include" -o "$LIB" "$SRC" \
      -L"$PREFIX/lib" -lqhull_r -Wl,-rpath,"$PREFIX/lib"
echo "  $LIB"
