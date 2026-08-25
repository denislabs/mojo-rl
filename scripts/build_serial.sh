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

CC="${CC:-cc}"
echo "building serial shim from $SRC"
"$CC" -O2 -fPIC -c   -o "$OBJ" "$SRC"
"$CC" -O2 -shared    -o "$LIB" "$SRC"
echo "  $LIB"
echo "  $OBJ"
