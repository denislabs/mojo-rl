#!/usr/bin/env bash
# LIBERO COLLISION ARMS — PERFORMANCE.md §13.55. Builds of
# benchmarks/physics3d_gpu/bench_libero_collision.mojo, one knob set each,
# run INTERLEAVED, MIN over rounds.
#
#   pixi run -e nvidia bash scripts/libero_collision_arms.sh build   # ~8 compiles
#   pixi run -e nvidia bash scripts/libero_collision_arms.sh run
#   pixi run python scripts/libero_collision_arms.py                 # the table
#
#   OUT=libero_coll  LANES=256  ROUNDS=3  WINDOWS=5,30,45  SNAPS=8
#   ARMS="base nofb nopre stop1 stop2 stop3 tpb64 report"
#   are the knobs (defaults shown).
#
# ⚠ NVIDIA ARMS. Production `base` carries `COLL_PREFILTER` on NVIDIA only
# (`ccd_workspace.mojo`, 2026-09-15): on Metal `base` and `nopre` are the SAME
# binary and `build` refuses them, which is the md5 guard doing its job.
#
# THE ARMS, and the question each answers:
#
#   base   production                     the reference time
#   nofb   COLL_NO_FALLBACK               `flagged` must stay 0; base - nofb =
#          the flagged-only serial launch that runs every substep
#   nopre  COLL_PREFILTER=False           the production before §13.55: every
#          LIBERO lane overflows into the serial kernel; nopre / base is the
#          prefilter's win, and base's CPU check must equal nopre's
#   stop1  nofb + COLL_STOP_AFTER=1       phase 0: poses + AABBs (+ launch)
#   stop2  nofb + COLL_STOP_AFTER=2       + phase 1: thread 0's listing
#   stop3  nofb + COLL_STOP_AFTER=3       + phase 2: the narrow phase
#          (nofb - stop3 = phase 3, compaction + contact sort)
#   tpb64  COLL_TPB=64                    fewer rounds, twice the CCD rows
#   report COLL_CAND_REPORT               per lane: uncapped AABB passes,
#          survivors, listed candidates, rounds, kinds (a counting arm, run
#          once — its times are not the kernel's)
#
# optional:  nopre_nofb (the old production's flagged count),
#            report_nopre (the overflow, as the old production listed it),
#            c1024 (COLL_NCAND_CAP=1024 without the prefilter)
#
set -euo pipefail

CMD=${1:?usage: libero_collision_arms.sh build|run}
OUT=${OUT:-libero_coll}
LANES=${LANES:-256}
ARMS=${ARMS:-base nofb nopre stop1 stop2 stop3 tpb64 report}
ROUNDS=${ROUNDS:-3}
WINDOWS=${WINDOWS:-5,30,45}
SNAPS=${SNAPS:-8}

BENCH=benchmarks/physics3d_gpu/bench_libero_collision.mojo
CCD=mojo_rl/physics3d/collision/ccd_workspace.mojo
SAP=mojo_rl/physics3d/collision/broadphase_sap.mojo

[ -f "$BENCH" ] || { echo "!! run from the repository root ($BENCH not found)"; exit 1; }
mkdir -p "$OUT"

_md5() { (md5sum "$1" 2>/dev/null || md5 -q "$1") | awk '{print $1}'; }

# "file knob value" lines for an arm
_knobs() {
  case "$1" in
    base)         ;;
    nofb)         echo "$CCD COLL_NO_FALLBACK True" ;;
    nopre)        echo "$CCD COLL_PREFILTER False" ;;
    nopre_nofb)   echo "$CCD COLL_PREFILTER False"; echo "$CCD COLL_NO_FALLBACK True" ;;
    stop1)        echo "$CCD COLL_NO_FALLBACK True"; echo "$SAP COLL_STOP_AFTER 1" ;;
    stop2)        echo "$CCD COLL_NO_FALLBACK True"; echo "$SAP COLL_STOP_AFTER 2" ;;
    stop3)        echo "$CCD COLL_NO_FALLBACK True"; echo "$SAP COLL_STOP_AFTER 3" ;;
    tpb64)        echo "$CCD COLL_TPB 64" ;;
    report)       echo "$CCD COLL_CAND_REPORT True" ;;
    report_nopre) echo "$CCD COLL_CAND_REPORT True"; echo "$CCD COLL_PREFILTER False" ;;
    c1024)        echo "$CCD COLL_PREFILTER False"; echo "$CCD COLL_NCAND_CAP 1024" ;;
    *) echo "!! unknown arm '$1'" >&2; exit 1 ;;
  esac
}

# Set `comptime KNOB: T = ...` to VALUE; exactly one such line must exist.
_set_knob() {
  python - "$1" "$2" "$3" <<'PY'
import re, sys
path, knob, value = sys.argv[1:4]
src = open(path).read()
pat = re.compile(r'^(comptime %s: \w+ = ).*$' % re.escape(knob), re.M)
hits = pat.findall(src)
if len(hits) != 1:
    sys.exit(f"!! {path}: {len(hits)} lines define {knob}, expected 1")
open(path, "w").write(pat.sub(lambda m: m.group(1) + value, src))
print(f"   {knob} = {value}  ({path})")
PY
}

build() {
  for arm in $ARMS; do _knobs "$arm" > /dev/null; done   # refuse a typo before compiling
  if ! git diff --quiet -- "$CCD" "$SAP"; then
    echo "!! $CCD or $SAP has uncommitted edits. This script flips knobs in"
    echo "   those files and restores them from a copy; commit or stash first."
    exit 1
  fi
  command -v mojo >/dev/null || { echo "!! mojo not on PATH — run under pixi run -e nvidia"; exit 1; }
  echo "mojo: $(mojo --version 2>&1 | head -1)"
  mkdir -p "$OUT/.orig"
  cp "$CCD" "$OUT/.orig/ccd_workspace.mojo"
  cp "$SAP" "$OUT/.orig/broadphase_sap.mojo"
  _restore() {
    cp "$OUT/.orig/ccd_workspace.mojo" "$CCD"
    cp "$OUT/.orig/broadphase_sap.mojo" "$SAP"
  }
  trap _restore EXIT INT TERM

  local src="$BENCH"
  if [ "$LANES" != "256" ]; then
    src="$OUT/bench_libero_collision_${LANES}.mojo"
    sed "s/^comptime LANES: Int = .*/comptime LANES: Int = $LANES/" "$BENCH" > "$src"
    grep -q "^comptime LANES: Int = $LANES$" "$src" || { echo "!! LANES sed failed"; exit 1; }
  fi

  : > "$OUT/ARMS.txt"
  for arm in $ARMS; do
    echo "=== build $arm ==="
    _restore
    while read -r file knob value; do
      [ -n "${file:-}" ] && _set_knob "$file" "$knob" "$value"
    done < <(_knobs "$arm")
    local t0=$SECONDS
    mojo build -I . -o "$OUT/bench_$arm" "$src"
    _restore
    cmp -s "$CCD" "$OUT/.orig/ccd_workspace.mojo" && cmp -s "$SAP" "$OUT/.orig/broadphase_sap.mojo" \
      || { echo "!! the knob files did not restore"; exit 1; }
    local m; m=$(_md5 "$OUT/bench_$arm")
    echo "$arm $m $((SECONDS - t0))s" | tee -a "$OUT/ARMS.txt"
  done
  trap - EXIT INT TERM
  _restore
  git diff --quiet -- "$CCD" "$SAP" || { echo "!! knob files differ from HEAD after build"; exit 1; }
  local dups; dups=$(awk '{print $2}' "$OUT/ARMS.txt" | sort | uniq -d)
  if [ -n "$dups" ]; then
    echo "!! two arms produced the SAME binary ($dups) — a knob did not take:"
    cat "$OUT/ARMS.txt"; exit 1
  fi
  echo "built. now:  OUT=$OUT pixi run -e nvidia bash scripts/libero_collision_arms.sh run"
}

run() {
  for arm in $ARMS; do
    [ -x "$OUT/bench_$arm" ] || { echo "!! $OUT/bench_$arm missing — build first"; exit 1; }
  done
  local arms=($ARMS)
  for r in $(seq 1 "$ROUNDS"); do
    # rotate the order each round so a drift does not always land on one arm
    local n=${#arms[@]} k=$(( (r - 1) % ${#arms[@]} ))
    local order=("${arms[@]:$k}" "${arms[@]:0:$k}")
    for arm in "${order[@]}"; do
      if [[ "$arm" == report* ]] && [ "$r" -gt 1 ]; then
        continue   # a counting arm, not a timing one
      fi
      local cpu=0; [ "$r" -eq 1 ] && cpu=1
      local log="$OUT/$arm.r$r.log"
      echo "=== r$r $arm ==="
      if ! "$OUT/bench_$arm" --windows "$WINDOWS" --snaps "$SNAPS" --rounds 2 \
            --cpu-check "$cpu" > "$log" 2>&1; then
        echo "!! $arm r$r failed:"; tail -20 "$log"; exit 1
      fi
      grep "^RESULT" "$log" | sed 's/^/   /'
    done
  done
  echo "collected. now:  OUT=$OUT pixi run python scripts/libero_collision_arms.py"
}

case "$CMD" in
  build) build ;;
  run) run ;;
  *) echo "usage: libero_collision_arms.sh build|run"; exit 1 ;;
esac
