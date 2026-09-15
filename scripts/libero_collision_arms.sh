#!/usr/bin/env bash
# LIBERO COLLISION ARMS — PERFORMANCE.md §13.55. Nine builds of
# benchmarks/physics3d_gpu/bench_libero_collision.mojo, one knob set each,
# run INTERLEAVED, MIN over rounds.
#
#   pixi run -e nvidia bash scripts/libero_collision_arms.sh build   # ~9 compiles
#   pixi run -e nvidia bash scripts/libero_collision_arms.sh run
#   pixi run python scripts/libero_collision_arms.py                 # the table
#
#   OUT=libero_coll  LANES=256  ROUNDS=3  WINDOWS=5,30,45  SNAPS=8
#   ARMS="base nofb pre pre_nofb pre_stop1 pre_stop2 pre_stop3 pre_tpb64 report"
#   are the knobs (defaults shown).
#
# WHAT THE MAC ALREADY SAID (4 lanes, the report): EVERY lane's sweep passes
# 377-500 AABB pairs against the 256-candidate cap, so every lane overflows
# and the block kernel hands it to the serial kernel — while only 25-127 of
# those pairs survive the narrow phase's first rejects. So the arms are
# about the PREFILTER (`COLL_PREFILTER`), which drops those pairs at listing.
#
# THE ARMS, and the question each answers:
#
#   base       production                    the reference time
#   nofb       COLL_NO_FALLBACK              the block launch ALONE; `flagged`
#              = lanes that overflowed (expect all of them), base - nofb =
#              the serial kernel doing every lane's collision
#   pre        COLL_PREFILTER                THE CANDIDATE FIX; base / pre is
#              its speedup, and its CPU check must match base's
#   pre_nofb   pre + COLL_NO_FALLBACK        flagged must be 0; pre - pre_nofb
#              = the flagged-only serial launch paid for nothing
#   pre_stop1  pre_nofb + COLL_STOP_AFTER=1  phase 0: poses + AABBs (+ launch)
#   pre_stop2  pre_nofb + COLL_STOP_AFTER=2  + phase 1: thread 0's listing
#   pre_stop3  pre_nofb + COLL_STOP_AFTER=3  + phase 2: the narrow phase
#              (pre_nofb - pre_stop3 = phase 3, compaction + contact sort)
#   pre_tpb64  pre + COLL_TPB=64             fewer rounds, twice the CCD rows
#   report     pre + COLL_CAND_REPORT        per lane: uncapped AABB passes,
#              survivors, listed candidates, rounds, kinds (a counting arm,
#              run once — its times are not the kernel's)
#
# optional:  report_nopre (the overflow, as production lists it),
#            c1024 (COLL_NCAND_CAP=1024 without the prefilter)
#
# ⚠ THE KNOBS ARE FLIPPED IN THE TREE, ONE BUILD AT A TIME, AND PUT BACK FROM
# THIS SCRIPT'S OWN COPY — never `git checkout`. The script refuses to start
# if either knob file has uncommitted edits, restores on EXIT/INT/TERM, and
# checks the restored bytes against the copy after every build. Do not edit
# those two files, or run a second build, while `build` runs.
#
# ⚠ SAME RULES AS p0_ab.sh: nothing else on the GPU during `run`; the arms are
# BINARIES, and two arms with the same md5 are refused — a delta of zero is
# the shape of an unchanged binary, not of a null result.
set -euo pipefail

CMD=${1:?usage: libero_collision_arms.sh build|run}
OUT=${OUT:-libero_coll}
LANES=${LANES:-256}
ARMS=${ARMS:-base nofb pre pre_nofb pre_stop1 pre_stop2 pre_stop3 pre_tpb64 report}
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
    base)      ;;
    nofb)      echo "$CCD COLL_NO_FALLBACK True" ;;
    pre)       echo "$CCD COLL_PREFILTER True" ;;
    pre_nofb)  echo "$CCD COLL_PREFILTER True"; echo "$CCD COLL_NO_FALLBACK True" ;;
    pre_stop1) echo "$CCD COLL_PREFILTER True"; echo "$CCD COLL_NO_FALLBACK True"; echo "$SAP COLL_STOP_AFTER 1" ;;
    pre_stop2) echo "$CCD COLL_PREFILTER True"; echo "$CCD COLL_NO_FALLBACK True"; echo "$SAP COLL_STOP_AFTER 2" ;;
    pre_stop3) echo "$CCD COLL_PREFILTER True"; echo "$CCD COLL_NO_FALLBACK True"; echo "$SAP COLL_STOP_AFTER 3" ;;
    pre_tpb64) echo "$CCD COLL_PREFILTER True"; echo "$CCD COLL_TPB 64" ;;
    report)    echo "$CCD COLL_PREFILTER True"; echo "$CCD COLL_CAND_REPORT True" ;;
    report_nopre) echo "$CCD COLL_CAND_REPORT True" ;;
    c1024)     echo "$CCD COLL_NCAND_CAP 1024" ;;
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
