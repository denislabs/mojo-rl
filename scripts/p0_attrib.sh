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
      mojo run -I . "$PROBE" "$k" 2>&1 | tee "$OUT/k$k.probe.txt"
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
