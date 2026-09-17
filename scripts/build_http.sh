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
# ⚠ THE SOURCE GOES BEFORE THE -l FLAGS. Ubuntu's toolchain (Jetson) links
# `--as-needed` by default: a library named before anything references it is
# DROPPED, and a shared object may leave symbols undefined, so the link
# succeeds with no NEEDED libcurl and dlopen fails at runtime. macOS ld never
# drops, so the Mac cannot show this. `--no-undefined` makes it a link error.
# ⚠ -pthread IS REQUIRED, AND WHICH MACHINES NEED IT IS NOT OBVIOUS.
# `mrl_http.c` guards its `curl_global_init` with `pthread_once`. glibc 2.34
# MERGED libpthread into libc, so on a modern distro the symbol resolves with
# no flag at all — Ubuntu 22.04 on the Jetson (glibc 2.35) links this happily,
# and macOS always has pthread in libSystem. A conda toolchain is the case
# that breaks: it links through its OWN sysroot, whose glibc predates the
# merge, so `pthread_once` is still in libpthread.so and must be asked for:
#
#   x86_64-conda-linux-gnu-ld: undefined reference to `pthread_once'
#
# ⚠ THAT ERROR IS `--no-undefined` DOING ITS JOB, not a regression. Without it
# a shared object is allowed to leave symbols unresolved, so this would have
# linked "successfully" and then aborted at the first dlopen on the one box
# that could not resolve it. Keep both flags.
NO_UNDEF=""
[ "$(uname -s)" = Linux ] && NO_UNDEF="-Wl,--no-undefined"
"$CC" -O2 -fPIC -shared -pthread \
    -I "$PREFIX/include" \
    -o "$LIB" "$SRC" \
    -L "$PREFIX/lib" -lcurl -lz -lzstd \
    -Wl,-rpath,"$PREFIX/lib" $NO_UNDEF

echo "  $LIB"
