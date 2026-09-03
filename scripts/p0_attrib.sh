#!/usr/bin/env bash
# P0 — collect per-kernel attribution for k in {0,3,6,9}, ONE PROCESS PER k.
#
#   pixi run -e nvidia bash scripts/p0_attrib.sh
#   pixi run python scripts/p0_attrib.py
#
# ⚠⚠ ONE PROCESS PER k IS THE WHOLE POINT. Mojo mangles a kernel's comptime
# Int parameters into a hash, so a single process running every leg gives four
# indistinguishable `_ldl_factor_*` entries. See the probe's header.
#
# ⚠ NVIDIA ONLY. Apple cannot build this scene past k=0.
#
# Collects three things per k and DELETES NOTHING:
#   *.probe.txt   the probe's own header (nv, step counts, ms/step)
#   *.kern.txt    cuda_gpu_kern_sum   — the per-kernel summary
#   *.trace.txt   cuda_gpu_trace      — every launch in time order, which is
#                 what disambiguates two kernels that share a mangled prefix
#                 (ldl_factor is enqueued before compute_m_inv, rk4.mojo:565-566)
set -uo pipefail

OUT=${OUT:-p0_attrib}
PROBE=examples/so101/so101_park_attrib_probe.mojo
mkdir -p "$OUT"

command -v nsys >/dev/null || { echo "!! nsys not on PATH"; exit 1; }

# ⚠⚠ PREFLIGHT, AND THE SECOND TEST MATTERS MORE THAN THE FIRST. A missing
# `mojo` fails loudly two lines below; a missing LD_PRELOAD does NOT — it
# produces a complete, plausible, WRONG measurement, because the CUDA
# interceptor arrives through the nvidia environment's ACTIVATION and not
# through the binary. Both have the same cause and the same fix.
if ! command -v mojo >/dev/null; then
  echo "!! \`mojo\` is not on PATH -- this script must run INSIDE the pixi env:"
  echo "       pixi run -e nvidia bash scripts/p0_attrib.sh"
  exit 1
fi
if [ -z "${LD_PRELOAD:-}" ]; then
  echo "!! LD_PRELOAD is empty. The CUDA interceptor comes from the nvidia"
  echo "   environment's activation, so this run would measure the wrong thing"
  echo "   SILENTLY. Re-run as:"
  echo "       pixi run -e nvidia bash scripts/p0_attrib.sh"
  exit 1
fi

# ⚠ BUILD ONCE, RUN SIX TIMES. `mojo run` compiles, and the probe carries every
# leg in one binary (a runtime switch over comptime instantiations). Measured on
# Apple: a cold `mojo run` is 136 s and a warm one 30 s — so `mojo run` DOES
# cache and the six invocations were NOT six full compiles, only one plus five
# cache hits. Building once still removes those five (~150 s here) and, more
# usefully, makes the sweep cost one predictable compile instead of a cache
# behaviour.
#
# ⚠ WHERE THE TIME ACTUALLY WENT WHEN k=12/13 WERE ADDED, since it is worth
# knowing before blaming the wrong change. The block work is compile-NEUTRAL:
# `test_newton_solve_fields` (one model) builds in 81.9 s before the campaign
# and 74.8 s after. The cost is the two new legs — 155 s for four, 195 s for
# six — and they are the two LARGEST models AND they cross the `Je` spill
# boundary, so `JE_IN_SHARED` flips and the blocked Newton kernel is
# instantiated in BOTH address-space variants.
#
# ⚠ IF THAT FIRST COMPILE IS STILL TOO SLOW, the fix is fewer legs per binary,
# not fewer runs: `KS="0 3 6 9"` skips the wide pair at RUNTIME but still
# compiles it. Splitting the probe into a narrow and a wide file is the real
# answer and has not been done.
#
# ⚠ RUN IT FROM INSIDE THE PIXI ENV, WHICH THIS SCRIPT ALREADY IS. The CUDA
# interceptor arrives through `LD_PRELOAD`, set by the nvidia environment's
# ACTIVATION and not by the binary — running the built executable from a plain
# shell silently loses it. The tell is the `[intercept] ... loaded` banner: if
# it is missing from the probe output, the environment is wrong.
BIN="$OUT/park_attrib_probe"
echo "=== building once ($PROBE) ==="
mojo build -I . -o "$BIN" "$PROBE" || { echo "!! build failed"; exit 1; }
echo "  -> $BIN"

# ⚠ 12 AND 13 EXIST BECAUSE P4 MADE THEM COMPILE, AND THEY CROSS THE `Je`
# SPILL BOUNDARY: k<=9 keeps `Je` in threadgroup memory, k>=10 re-reads it from
# global every Newton iteration. Do not draw one `x k=0` curve across that —
# report 0..9 and 12..13 separately, or the slot count gets charged for a
# change of code path.
for k in ${KS:-0 3 6 9 12 13}; do
  echo "=== k=$k ==="
  rep="$OUT/k$k"
  rm -f "$rep.nsys-rep" "$rep.sqlite"
  # --stats=false: the summaries are exported explicitly below, so a failure
  # to export is a visible error rather than a missing tail of stdout.
  nsys profile --force-overwrite=true -o "$rep" --stats=false \
      "$BIN" "$k" 2>&1 | tee "$OUT/k$k.probe.txt"
  if [ ! -f "$rep.nsys-rep" ]; then
    echo "!! no report for k=$k — the leg did not run; NOT continuing to the"
    echo "   table, which would be computed from a missing arm."
    exit 1
  fi
  nsys stats --report cuda_gpu_kern_sum --format table "$rep.nsys-rep" \
      > "$OUT/k$k.kern.txt" 2>&1
  # ⚠ THE REPORT NAME VARIES BY nsys VERSION. If this export fails the table
  # is still produced — the ldl_factor/compute_m_inv pair simply stays POOLED
  # instead of split — so warn loudly rather than aborting a run that took
  # four profiled legs to collect.
  if ! nsys stats --report cuda_gpu_trace --format csv "$rep.nsys-rep" \
        > "$OUT/k$k.trace.csv" 2>"$OUT/k$k.trace.err"; then
    echo "  !! cuda_gpu_trace export failed (see $OUT/k$k.trace.err)."
    echo "     The term table is unaffected; ldl_factor and compute_m_inv"
    echo "     will stay POOLED. Try: nsys stats --help-reports | grep trace"
  fi
  if ! grep -qi 'name' "$OUT/k$k.trace.csv" 2>/dev/null; then
    echo "  !! $OUT/k$k.trace.csv has no Name column — pair stays pooled."
  fi
  echo "  -> $OUT/k$k.{probe.txt,kern.txt,trace.csv}"
done

echo
echo "collected. now:  pixi run python scripts/p0_attrib.py"
