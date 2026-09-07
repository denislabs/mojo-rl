"""Launch shape, registers and threadgroup memory per kernel, from a
`cuda_gpu_trace` csv — the columns that decide blocks per SM.

    python scripts/p0_kernel_shape.py OUT/k13.trace.csv [name-substring]

One row per distinct kernel name (first launch seen): grid, block,
registers per thread, static + dynamic shared KB, one launch's duration,
and the blocks-per-SM bounds a 128 KB / 64K-register SM implies. It is
what read the Newton kernel's 96 KB at k=13 off the baseline trace
(BLOCK_DIAGONAL_..., 2026-09-07); typed ad hoc twice, so it lives here.
"""

import csv
import io
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 2
    path = sys.argv[1]
    pat = sys.argv[2] if len(sys.argv) > 2 else ""
    lines = open(path, errors="replace").read().splitlines()
    hdr = [i for i, l in enumerate(lines) if l.startswith("Start (ns)")]
    if not hdr:
        print(f"!! no 'Start (ns)' header in {path}"); return 1
    seen = set()
    print(f"{'kernel':40} {'grid':>6} {'block':>5} {'regs':>4} {'shared KB':>9} "
          f"{'us/launch':>9}  blocks/SM by shared, by regs")
    for r in csv.DictReader(io.StringIO("\n".join(lines[hdr[0]:]))):
        n = r.get("Name", "")
        if pat and pat not in n:
            continue
        key = n.replace("mojo_rl_physics3d_", "")[:40]
        if key in seen or not r.get("BlkX"):
            continue
        seen.add(key)
        blk = int(r["BlkX"]); regs = int(r["Reg/Trd"] or 0)
        sm = (float(r["StcSMem (MB)"] or 0) + float(r["DymSMem (MB)"] or 0)) * 1024
        by_sh = int(128 / sm) if sm > 0 else 99
        by_rg = 65536 // (regs * blk) if regs and blk else 99
        print(f"{key:40} {r['GrdX']:>6} {blk:>5} {regs:>4} {sm:>9.1f} "
              f"{float(r['Duration (ns)'])/1e3:>9.1f}  {min(by_sh,32):>3}, {min(by_rg,32):>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
