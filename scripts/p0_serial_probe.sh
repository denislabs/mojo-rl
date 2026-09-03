#!/usr/bin/env bash
# Serial-term probe — WHERE is the blocked Newton kernel's tid-0 floor?
#
#   pixi run -e nvidia bash scripts/p0_serial_probe.sh
#
# `NEWTON_COOP_DIV` proved the floor is ~90% of newton's excess. This finds
# WHICH TERM it is, by MEASUREMENT rather than by counting ops — op counts have
# now over-predicted time-saved four times running (PN2c, P2, F2, F3b), because
# serial tid-0 GPU code is latency-bound on dependent shared loads.
#
# Each term is run `SERIAL_PROBE_REPEAT - 1` EXTRA times before its real,
# untouched instance; `t(probe) - t(baseline)` over `REPEAT-1` is its marginal
# cost. Bit-exact at every setting (see the long note at the constant), and the
# `T=0` leg is codegen-identical to production.
#
# ⚠⚠ THE MAGNITUDE IS THE NON-VACUITY CHECK. At REPEAT=10 a term worth 5% of
# newton must move newton by ~45%. A term that comes back INSIDE THE NOISE
# FLOOR (~1.7% on this box for kernels over 1 ms/step) has either been
# optimised away or is genuinely free, and THOSE ARE NOT THE SAME ANSWER —
# re-run that one at REPEAT=100 before believing it is free.
#
# ⚠ Restores the constant on ANY exit, including Ctrl-C. Check `git diff` after
# a crash anyway: a probe build left in the tree is a wrong production binary.
set -uo pipefail

SRC=mojo_rl/physics3d/solver/newton_solve.mojo
KEY='comptime NEWTON_SERIAL_PROBE: Int'
TERMS=${TERMS:-0 1 2 3 4}
export KS=${KS:-"0 13"}

command -v mojo >/dev/null || {
  echo "!! run inside the pixi env: pixi run -e nvidia bash $0"; exit 1; }

_set() { sed -i "s/^${KEY} = .*/${KEY} = $1/" "$SRC"; }
_restore() { _set 0; echo "-- restored NEWTON_SERIAL_PROBE = 0"; }
trap _restore EXIT INT TERM

grep -q "^${KEY} = " "$SRC" || { echo "!! cannot find '$KEY' in $SRC"; exit 1; }

for T in $TERMS; do
  echo "############## NEWTON_SERIAL_PROBE = $T"
  _set "$T"
  grep -n "^${KEY} = " "$SRC"
  OUT="p0_sp$T" bash scripts/p0_attrib.sh || { echo "!! leg T=$T failed"; exit 1; }
done

echo
echo "=== build fingerprints — EVERY ONE MUST DIFFER ==="
echo "    (equal hashes mean two arms are the same program; see BUILD.txt)"
for T in $TERMS; do
  printf "  T=%s  " "$T"
  sed -n 's/^binary_md5 *//p' "p0_sp$T/BUILD.txt" 2>/dev/null || echo "(missing)"
done
