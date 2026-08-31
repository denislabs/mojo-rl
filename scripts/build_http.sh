#!/usr/bin/env bash
# Build the HTTP shim for `mojo_rl/io/http.mojo`.
#
# The dylib is what the Mojo side dlopen's; it links against the pixi env's
# libcurl. Not tracked in git. Re-run after editing native/mrl_http.c.
#
#   pixi run build-http
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/mojo_rl/io/native/mrl_http.c"
OUTDIR="$ROOT/mojo_rl/io"

case "$(uname -s)" in
Darwin) LIB="$OUTDIR/libmrl_http.dylib" ;;
Linux)  LIB="$OUTDIR/libmrl_http.so" ;;
*) echo "build_http.sh: unsupported OS $(uname -s)" >&2; exit 1 ;;
esac

FORCE="${1:-}"
if [ "$FORCE" != "-f" ] && [ -f "$LIB" ] && [ "$SRC" -ot "$LIB" ]; then
    echo "http shim up to date: $LIB"
    exit 0
fi

# ⚠ CONDA_PREFIX IS THE PIXI ENV, AND IT IS THE ONLY CORRECT ANSWER HERE.
# macOS ships its own libcurl (built against Secure Transport, with a
# different CA story); linking that one and then running inside pixi gives a
# binary whose TLS trust store depends on which curl won the link — the
# `build_opencv.sh` hazard, one layer down. Fail loudly instead.
PREFIX="${CONDA_PREFIX:-$ROOT/.pixi/envs/default}"
if [ ! -f "$PREFIX/include/curl/curl.h" ]; then
    echo "build_http.sh: no libcurl headers under $PREFIX." >&2
    echo "  Run this through pixi:  pixi run build-http" >&2
    exit 1
fi

CC="${CC:-cc}"
echo "building http shim from $SRC"
echo "  against $PREFIX ($("$PREFIX/bin/curl" --version 2>/dev/null | head -1 || echo 'libcurl'))"

# ⚠ -rpath IS NOT OPTIONAL. Without it the dylib links and then fails to find
# libcurl.4 at the FIRST CALL, as a dlopen abort with no useful message.
"$CC" -O2 -fPIC -shared \
    -I "$PREFIX/include" \
    -L "$PREFIX/lib" -lcurl -lz -lzstd \
    -Wl,-rpath,"$PREFIX/lib" \
    -o "$LIB" "$SRC"

echo "  $LIB"
