#!/usr/bin/env bash
# Build the serial shim for `mojo_rl/io/serial/`.
#
# The dylib is what the Mojo side dlopen's (`mojo_rl/io/serial/native.mojo`);
# the object file is kept because it costs one extra compiler invocation and
# is what a future self-contained build would need if Mojo ever grows a way to
# express it — today it is unused. Neither is tracked in git. Re-run after
# editing native/mrl_serial.c.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/mojo_rl/io/serial/native/mrl_serial.c"
OUTDIR="$ROOT/mojo_rl/io/serial"

case "$(uname -s)" in
Darwin) LIB="$OUTDIR/libmrl_serial.dylib" ;;
Linux)  LIB="$OUTDIR/libmrl_serial.so" ;;
*) echo "build_serial.sh: unsupported OS $(uname -s)" >&2; exit 1 ;;
esac
OBJ="$OUTDIR/mrl_serial.o"

if [[ "$SRC" -ot "$LIB" && "$SRC" -ot "$OBJ" && -f "$LIB" && -f "$OBJ" ]]; then
    echo "serial shim up to date: $LIB"
    exit 0
fi

# ⚠ `--no-undefined` FOR THE SAME REASON build_http.sh HAS IT. A shared object
# is allowed to leave symbols unresolved, so a missing library links
# "successfully" here and aborts at the first dlopen instead — which for this
# shim means in front of an energised arm, since `SerialPort.__init__` calls
# into it before every port open. The http shim found exactly this on a conda
# toolchain (`pthread_once` in libpthread, not libc, below glibc 2.34): make
# it a link error on the machine that builds it.
NO_UNDEF=""
[ "$(uname -s)" = Linux ] && NO_UNDEF="-Wl,--no-undefined"

CC="${CC:-cc}"
echo "building serial shim from $SRC"
"$CC" -O2 -fPIC -c   -o "$OBJ" "$SRC"
"$CC" -O2 -shared    -o "$LIB" "$SRC" $NO_UNDEF
echo "  $LIB"
echo "  $OBJ"
