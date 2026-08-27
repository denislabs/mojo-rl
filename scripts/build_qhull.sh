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
# ⚠ A SECOND TRANSLATION UNIT, AND IT IS C++ ON PURPOSE. `mrl_poly_order`
# reproduces `MakePolygons`' emission order by calling the SAME
# `std::unordered_map` MuJoCo does — see native/mrl_polyorder.cc for why that
# is a call rather than a reimplementation. It rides in this dylib because the
# Mojo side already dlopen's exactly one library for mesh topology.
SRC_CXX="$ROOT/mojo_rl/physics3d/collision/native/mrl_polyorder.cc"
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

if [[ "$SRC" -ot "$LIB" && "$SRC_CXX" -ot "$LIB" && -f "$LIB" ]]; then
    echo "qhull shim up to date: $LIB"
    exit 0
fi

CC="${CC:-cc}"
CXX="${CXX:-c++}"
echo "building qhull shim from $SRC + $SRC_CXX"
TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT
"$CC"  -O2 -fPIC -c -I"$PREFIX/include" -o "$TMPD/mrl_qhull.o"     "$SRC"
"$CXX" -O2 -fPIC -c -std=c++17          -o "$TMPD/mrl_polyorder.o" "$SRC_CXX"
# ⚠ LINK WITH THE C++ DRIVER — one object needs libc++, and it is the whole
# point of that object that it is the SAME libc++ MuJoCo's map ran on.
# ⚠ RPATH TO THE ENV, so the dylib finds libqhull_r wherever pixi put it.
"$CXX" -O2 -fPIC -shared -o "$LIB" "$TMPD/mrl_qhull.o" "$TMPD/mrl_polyorder.o" \
      -L"$PREFIX/lib" -lqhull_r -Wl,-rpath,"$PREFIX/lib"
echo "  $LIB"
