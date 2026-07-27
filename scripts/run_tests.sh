#!/usr/bin/env bash
# run_tests.sh — minimal mojo test runner (manifest or discovery).
#
# Usage (always through pixi so `mojo` is on PATH):
#   pixi run        test-smoke          # curated fast CPU gates (CI tier)
#   pixi run        test-cpu            # discover all non-GPU tests (hours)
#   pixi run -e apple  test-gpu         # discover *_gpu.mojo tests
#   pixi run -e nvidia test-gpu
#
# Direct:
#   bash scripts/run_tests.sh tests/manifests/smoke.txt
#   bash scripts/run_tests.sh --compile-only tests/manifests/examples-compile.txt
#   bash scripts/run_tests.sh --discover tests --cpu-only
#   bash scripts/run_tests.sh --discover tests --gpu-only
#
# Each entry is one `mojo run -I . <file>` (compile + run), or with
# `--compile-only` one `mojo build -I . <file>` (compile probe — for
# examples that would train for hours if executed). Lines starting with `#`
# and blank lines in a manifest are skipped. Output of failing tests is
# echoed (last 40 lines); a summary table always prints. Exit code is the
# number of failures (0 = green).

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="manifest"
COMPILE_ONLY=0
if [[ "${1:-}" == "--compile-only" ]]; then
    COMPILE_ONLY=1
    shift
fi
TARGET="${1:-tests/manifests/smoke.txt}"
FILTER="all"

if [[ "${1:-}" == "--discover" ]]; then
    MODE="discover"
    TARGET="${2:-tests}"
    FILTER="${3:-all}"
fi

declare -a FILES=()
if [[ "$MODE" == "manifest" ]]; then
    if [[ ! -f "$TARGET" ]]; then
        echo "manifest not found: $TARGET" >&2
        exit 2
    fi
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | xargs)"  # trim
        [[ -z "$line" ]] && continue
        FILES+=("$line")
    done < "$TARGET"
else
    while IFS= read -r f; do
        case "$FILTER" in
            --gpu-only) [[ "$f" == *_gpu.mojo ]] || continue ;;
            --cpu-only) [[ "$f" == *_gpu.mojo ]] && continue ;;
        esac
        FILES+=("$f")
    done < <(find "$TARGET" -name "test_*.mojo" | sort)
fi

TOTAL=${#FILES[@]}
if [[ "$TOTAL" -eq 0 ]]; then
    echo "no tests selected" >&2
    exit 2
fi

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mojo-rl-tests.XXXXXX")"
echo "running $TOTAL test file(s)  (logs: $LOG_DIR)"
echo "----------------------------------------------------------------------"

PASS=0
FAIL=0
declare -a FAILED=()
SUITE_T0=$SECONDS

for f in "${FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        printf "MISSING  %-60s\n" "$f"
        FAIL=$((FAIL + 1))
        FAILED+=("$f (file not found)")
        continue
    fi
    log="$LOG_DIR/$(echo "$f" | tr '/' '_').log"
    t0=$SECONDS
    if [[ "$COMPILE_ONLY" == 1 ]]; then
        cmd=(mojo build -I . "$f" -o "$LOG_DIR/probe.bin")
    else
        cmd=(mojo run -I . "$f")
    fi
    if "${cmd[@]}" >"$log" 2>&1; then
        printf "PASS  %4ss  %s\n" "$((SECONDS - t0))" "$f"
        PASS=$((PASS + 1))
    else
        printf "FAIL  %4ss  %s\n" "$((SECONDS - t0))" "$f"
        FAIL=$((FAIL + 1))
        FAILED+=("$f")
        echo "──── last 40 lines of $f ────"
        tail -40 "$log"
        echo "─────────────────────────────"
    fi
done

echo "----------------------------------------------------------------------"
echo "total: $TOTAL  pass: $PASS  fail: $FAIL  elapsed: $((SECONDS - SUITE_T0))s"
if [[ "$FAIL" -gt 0 ]]; then
    echo "failed:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi
exit "$FAIL"
