"""Is the probe's workload STATIONARY along its trajectory? Bin one kernel's
per-launch duration by step index from a `cuda_gpu_trace` export.

    python scripts/p0_drift.py OUT/k13.trace.csv [bins] [name-substring]

Reads the Start/Duration/Name columns that scripts/p0_attrib.sh already
exports, groups the matching launches (default `solver_newt`) into `bins`
equal spans of launches, and prints avg/max per span with the step range
each span covers (8 launches per step on the RK4 park scene — printed as
launches too, so the divisor is visible).

WHY. 2026-09-07: the same Newton kernel (identical hash) read 2181 us per
launch over a 500-step run and 6095 us over a 1700-step run. The Sep 4
1000-step k=3 trace shows the jump: flat at ~178 us to step 700, then
10x spikes. A per-launch AVERAGE over a trajectory that changes character
is not a property of the kernel, and two sweeps of different lengths are
not comparable. Run this on every sweep's k=13 trace before reading its
table; a flat profile is what makes the average a number.
"""

import csv
import io
import os
import re
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 2
    path = sys.argv[1]
    bins = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    pat = sys.argv[3] if len(sys.argv) > 3 else "solver_newt"
    lines = open(path, errors="replace").read().splitlines()
    hdr = [i for i, l in enumerate(lines) if l.startswith("Start (ns)")]
    if not hdr:
        print(f"!! no 'Start (ns)' header in {path} — is it a cuda_gpu_trace csv?")
        return 1
    rows = csv.DictReader(io.StringIO("\n".join(lines[hdr[0]:])))
    d = [float(r["Duration (ns)"]) / 1e3 for r in rows if pat in r.get("Name", "")]
    if not d:
        print(f"!! no launch matches '{pat}'"); return 1
    n = len(d); per = max(1, n // bins)
    # Launches per step are DERIVED from the probe header next to the trace
    # (`k13.probe.txt`: `total_steps`), not assumed: RK4 at frame skip 2 made
    # 8, Euler makes 2, and a constant here is how a column drifts.
    lps = 8.0
    ph = path.replace(".trace.csv", ".probe.txt")
    if os.path.exists(ph):
        for line in open(ph, errors="replace"):
            m = re.match(r"\s+total_steps\s+(\d+)", line)
            if m:
                lps = n / float(m.group(1))
    print(f"{path}: {n} launches of '{pat}'  ({lps:.1f}/step => {n/lps:.0f} steps)")
    print(f"  {'launches':>14}  {'~steps':>11}  {'avg us':>9}  {'max us':>9}")
    for b in range(bins):
        seg = d[b * per:(b + 1) * per] if b < bins - 1 else d[b * per:]
        if not seg: break
        lo, hi = b * per, b * per + len(seg)
        print(f"  {lo:6d}-{hi:6d}  {lo/lps:5.0f}-{hi/lps:5.0f}  {sum(seg)/len(seg):9.1f}  {max(seg):9.1f}")
    first, last = d[:per], d[-per:]
    print(f"  last/first span ratio: {(sum(last)/len(last))/(sum(first)/len(first)):.2f}"
          f"   (a flat workload reads ~1.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
