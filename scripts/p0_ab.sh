#!/usr/bin/env bash
# P0 A/B — TWO PROBE BINARIES, INTERLEAVED, nsys PER LEG, MIN OVER ROUNDS.
#
#   pixi run -e nvidia bash scripts/p0_ab.sh <A_BIN> <B_BIN>
#   pixi run python scripts/p0_ab.py            # the table
#
#   KS="3 13" ROUNDS=3 OUT=p0_ab   are the knobs (defaults shown).
#
# WHAT IT ANSWERS that scripts/p0_attrib.sh cannot: "did THIS kernel move
# between these two builds?" A single sweep per arm cannot say — the 5090's
# run-to-run noise is 0.5% median and 5.8% max on one row, and one perturbed
# process (a k=3 leg that came back 60% slow with a +3.5 ms host residual) is
# indistinguishable from a regression at n=1. Interleaving A and B inside
# every round, alternating which goes first, and taking the MIN per kernel
# over rounds is the same discipline the CPU boards use (PERFORMANCE.md
# §13.36-37: interleaved, MIN of three).
#
# The arms are BINARIES, not commits: build each with scripts/p0_attrib.sh
# (its `OUT/park_attrib_probe`) or `mojo build -I . -o <bin>
# examples/so101/so101_park_attrib_probe.mojo`, and keep the old one. The
# script refuses two arms with the same md5 — a ratio of 1.00 at every
# kernel is the shape of an unchanged binary, not of a null result.
#
# ⚠ SAME RULES AS p0_attrib.sh: run INSIDE the nvidia env (the CUDA
# interceptor arrives through LD_PRELOAD from the environment's activation);
# nothing else on the GPU during the run; do not build while it runs.
set -uo pipefail

A=${1:?usage: p0_ab.sh <A_BIN> <B_BIN>}
B=${2:?usage: p0_ab.sh <A_BIN> <B_BIN>}
OUT=${OUT:-p0_ab}
KS=${KS:-3 13}
ROUNDS=${ROUNDS:-3}

command -v nsys >/dev/null || { echo "!! nsys not on PATH"; exit 1; }
if [ -z "${LD_PRELOAD:-}" ]; then
  echo "!! LD_PRELOAD is empty — run as: pixi run -e nvidia bash scripts/p0_ab.sh ..."
  exit 1
fi
for bin in "$A" "$B"; do
  [ -x "$bin" ] || { echo "!! not an executable: $bin"; exit 1; }
done
_md5() { (md5sum "$1" 2>/dev/null || md5 -q "$1") | awk '{print $1}'; }
MA=$(_md5 "$A"); MB=$(_md5 "$B")
if [ "$MA" = "$MB" ]; then
  echo "!! A and B are the SAME binary ($MA). The delta would be noise by"
  echo "   construction. Rebuild one arm from the other tree first."
  exit 1
fi

mkdir -p "$OUT/A" "$OUT/B"
{
  echo "A  $MA  $A"
  echo "B  $MB  $B"
  echo "ks $KS"
  echo "rounds $ROUNDS"
} > "$OUT/ARMS.txt"
cat "$OUT/ARMS.txt"

_leg() {  # arm bin round k
  local arm=$1 bin=$2 r=$3 k=$4
  local d="$OUT/$arm/r$r"; mkdir -p "$d"
  local rep="$d/k$k"
  rm -f "$rep.nsys-rep" "$rep.sqlite"
  nsys profile --force-overwrite=true -o "$rep" --stats=false \
      "$bin" "$k" 2>&1 | tee "$rep.probe.txt" | grep -E "^\s+(nv|ms_per_step)" || true
  if [ ! -f "$rep.nsys-rep" ]; then
    echo "!! no report for $arm r$r k=$k — aborting rather than tabulating a missing arm."
    exit 1
  fi
  nsys stats --report cuda_gpu_kern_sum --format table "$rep.nsys-rep" \
      > "$rep.kern.txt" 2>&1
  rm -f "$rep.nsys-rep" "$rep.sqlite"   # the summary is what the table reads
}

for r in $(seq 1 "$ROUNDS"); do
  for k in $KS; do
    # Alternate the order so a drift within a round (clocks, thermals) does
    # not always land on the same arm.
    if [ $((r % 2)) -eq 1 ]; then
      echo "=== r$r k=$k  A then B ==="; _leg A "$A" "$r" "$k"; _leg B "$B" "$r" "$k"
    else
      echo "=== r$r k=$k  B then A ==="; _leg B "$B" "$r" "$k"; _leg A "$A" "$r" "$k"
    fi
  done
done

echo
echo "collected. now:  OUT=$OUT pixi run python scripts/p0_ab.py"
