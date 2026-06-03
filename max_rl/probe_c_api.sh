#!/usr/bin/env bash
# Probe whether the MAX C API is *linkable* in the active environment — the
# precondition for Path B (Mojo -> MAX C API on a precompiled MEF, no Python in
# the hot loop). On Apple the C API is header-only (no implementing library);
# the real question is whether the linux/NVIDIA MAX package ships it.
#
# Usage:
#   pixi run -e nvidia bash max_rl/probe_c_api.sh      # the run that matters
#   pixi run -e apple  bash max_rl/probe_c_api.sh      # for contrast (expected: blocked)
#
# Verdict at the end is GO / NO-GO for building the Mojo FFI binding.

set -uo pipefail

PREFIX="${CONDA_PREFIX:-${1:-}}"
if [[ -z "$PREFIX" || ! -d "$PREFIX" ]]; then
  echo "Could not determine env prefix. Run via 'pixi run -e <env> bash $0' or pass the prefix as arg 1."
  exit 2
fi
echo "env prefix: $PREFIX"
echo "platform:   $(uname -s) $(uname -m)"
echo

# 1) Headers present?
HDR="$PREFIX/include/max/c/model.h"
if [[ -f "$HDR" ]]; then
  echo "[1] C API headers:        PRESENT ($PREFIX/include/max/c/)"
else
  echo "[1] C API headers:        MISSING"
fi

# 2) Which library (if any) EXPORTS the key C-ABI symbols?
KEY="M_compileModel|M_executeModelSync|M_newRuntimeContext|M_initModel|M_borrowTensorInto"
echo "[2] Scanning libraries for exported C-ABI symbols ($KEY)..."
FOUND_LIB=""
while IFS= read -r lib; do
  # Try dynamic-symbol table first (linux), then default (mac).
  syms="$( { nm -D "$lib" 2>/dev/null; nm -g "$lib" 2>/dev/null; } | grep -E "$KEY")"
  # Only count DEFINED/exported symbols: linux 'T'/'t'/'W', mac 'T'/'S'. Exclude 'U' (undefined import).
  defined="$(echo "$syms" | grep -E " [TtWSs] " )"
  if [[ -n "$defined" ]]; then
    echo "    EXPORTS in: $lib"
    echo "$defined" | sed 's/^/        /' | head -6
    FOUND_LIB="$lib"
  fi
done < <(find "$PREFIX" \( -name '*.so' -o -name '*.so.*' -o -name '*.dylib' -o -name '*.a' \) 2>/dev/null)
if [[ -z "$FOUND_LIB" ]]; then
  echo "    NONE — no library exports the C API (engine is only inside the Python ext)."
fi

# 3) Can Python emit a .mef artifact the C API could load? (best-effort)
echo "[3] Checking for a Python MEF-export path..."
PY="$PREFIX/bin/python"
[[ -x "$PY" ]] || PY="python"
"$PY" - <<'PYEOF' 2>/dev/null
ok = False
try:
    from max.engine import CompiledModel
    meths = [m for m in dir(CompiledModel) if any(k in m.lower() for k in ("save","export","serial","mef","write","dump"))]
    print("    CompiledModel export-ish methods:", meths or "NONE")
    ok = bool(meths)
except Exception as e:
    print("    (could not import CompiledModel:", e, ")")
try:
    import max._core as c
    cands = [n for n in dir(c) if any(k in n.lower() for k in ("mef","serial","save_model","export"))]
    print("    max._core MEF-ish symbols:", cands or "NONE")
    ok = ok or bool(cands)
except Exception:
    pass
import sys
sys.exit(0 if ok else 1)
PYEOF
MEF_OK=$?

echo
echo "================ VERDICT ================"
if [[ -n "$FOUND_LIB" ]]; then
  echo "C API library:  AVAILABLE  -> Mojo FFI binding is buildable."
  if [[ $MEF_OK -eq 0 ]]; then
    echo "MEF export:     AVAILABLE  -> Path B is GO (compile-from-MEF, no Python in loop)."
  else
    echo "MEF export:     NOT FOUND  -> Path B partial: C API can compile a *model file*"
    echo "                (M_setModelPath) but you must serialize the graph from Python"
    echo "                (Graph.module MLIR) first. Verify M_setModelPath accepts it."
  fi
else
  echo "C API library:  ABSENT     -> Path B NO-GO here. Engine is only reachable via the"
  echo "                Python bindings (max._core). No linkable/dlopen-able C-ABI surface."
fi
echo "========================================"
