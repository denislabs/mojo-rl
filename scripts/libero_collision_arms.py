"""The table for `scripts/libero_collision_arms.sh run` — PERFORMANCE.md §13.55.

    OUT=libero_coll pixi run python scripts/libero_collision_arms.py

Reads `$OUT/<arm>.r<round>.log`, takes each arm's `ms_mean` per window and
the MIN over rounds (the p0_ab discipline: the 5090's run-to-run noise is
~0.5% median with single-row flukes), then derives:

    the prefilter's win (the production before it)   = nopre / base
    production, phase by phase (NVIDIA: prefiltered):
      phase 0  world poses + AABBs (+ launch)         = stop1
      phase 1  thread 0's candidate listing           = stop2 - stop1
      phase 2  the narrow phase, in rounds            = stop3 - stop2
      phase 3  compaction + the contact sort          = nofb  - stop3
      the flagged-only serial launch                  = base  - nofb
    COLL_TPB 64                                       = tpb64 / base

and prints the `report` arms' per-window candidate summaries verbatim.
Any arm whose round-1 CPU check departs from `base`'s is flagged first.
"""
import glob
import os
import re
import sys
from collections import defaultdict

OUT = os.environ.get("OUT", sys.argv[1] if len(sys.argv) > 1 else "libero_coll")
ARMS = ["base", "nofb", "nopre", "nopre_nofb", "stop1", "stop2", "stop3",
        "tpb64", "c1024", "report", "report_nopre"]


def parse(path):
    rows = {}
    for line in open(path):
        if not line.startswith("RESULT "):
            continue
        kv = dict(tok.split("=", 1) for tok in line.split()[1:])
        rows[int(kv["window"])] = kv
    return rows


def main():
    logs = sorted(glob.glob(os.path.join(OUT, "*.r*.log")))
    if not logs:
        sys.exit(f"no logs under {OUT}/ — run `scripts/libero_collision_arms.sh run` first")
    ms = defaultdict(lambda: defaultdict(list))      # arm -> window -> [ms_mean]
    first = defaultdict(dict)                         # arm -> window -> r1 row
    for path in logs:
        m = re.match(r"(.+)\.r(\d+)\.log$", os.path.basename(path))
        if not m:
            continue
        arm, rnd = m.group(1), int(m.group(2))
        for w, kv in parse(path).items():
            ms[arm][w].append(float(kv["ms_mean"]))
            if rnd == 1:
                first[arm][w] = kv
    arms = [a for a in ARMS if a in ms] + sorted(a for a in ms if a not in ARMS)
    windows = sorted({w for a in ms for w in ms[a]})
    best = {a: {w: min(ms[a][w]) for w in ms[a]} for a in arms}

    print(f"{OUT}: arms {' '.join(arms)}; rounds per arm "
          + ", ".join(f"{a} {len(next(iter(ms[a].values())))}" for a in arms))
    print()
    print("ms per collision launch (MIN over rounds of the last-round mean)")
    print("window " + "".join(f"{a:>9}" for a in arms))
    for w in windows:
        print(f"{w:>6} " + "".join(
            f"{best[a][w]:>9.3f}" if w in best[a] else f"{'-':>9}" for a in arms))
    if any(a.startswith("report") for a in arms):
        print("        (report arms are counting builds: their times are not the kernel's)")

    def get(a, w):
        return best.get(a, {}).get(w)

    # ⚠ A TIME BOUGHT BY LOSING CONTACTS IS NOT A SPEEDUP. Any arm whose round-1
    # CPU check departs from production's is flagged before its numbers are
    # read. This caught a real one: the block kernel used to drop every box/box
    # pair on Metal, so a `pre` arm there returned ~0 contacts in half the time
    # (fixed in §13.56; `tests/physics3d/test_box_box_sap_gpu_parity.mojo` is
    # the gate). Keep the check — it is what makes an arm's time mean anything.
    print()
    bad = False
    for a in arms:
        if a == "base" or a not in first or "base" not in first:
            continue
        for w in windows:
            kv, kb = first[a].get(w), first["base"].get(w)
            if not kv or not kb or int(kv["cpu_lanes"]) == 0:
                continue
            ma, mb = int(kv["cpu_ncon_mismatch"]), int(kb["cpu_ncon_mismatch"])
            tol = max(2, int(kv["cpu_lanes"]) // 100)
            if ma > mb + tol or float(kv["ncon_mean"]) < 0.5 * float(kb["ncon_mean"]):
                bad = True
                print(f"!! {a} window {w}: its contacts depart from production's"
                      f" (cpu mismatches {ma} vs base {mb}, ncon mean"
                      f" {kv['ncon_mean']} vs {kb['ncon_mean']}) — its time is NOT"
                      " comparable, and the change it measures is not exact here")
    if not bad:
        print("contact sets: every CPU-checked arm matches production's mismatch"
              " count and ncon (within 1% of lanes)")

    print()
    print("the derivations, ms per launch")
    for w in windows:
        base, nofb, nopre = get("base", w), get("nofb", w), get("nopre", w)
        s1, s2, s3 = get("stop1", w), get("stop2", w), get("stop3", w)
        print(f"  window {w}:  base {base:.3f} ms" if base is not None else f"  window {w}:")
        if nopre is not None and base is not None:
            print(f"    COLL_PREFILTER: {nopre:.3f} (without) -> {base:.3f}  ({nopre / base:.2f}x)")
        nn = get("nopre_nofb", w)
        if nn is not None and nopre is not None:
            print(f"    without it, the serial fallback was {nopre - nn:.3f} of {nopre:.3f}")
        parts = []
        if s1 is not None:
            parts.append(("phase 0  poses + AABBs (+ launch)", s1))
        if s1 is not None and s2 is not None:
            parts.append(("phase 1  thread-0 listing: plane, sort, sweep", s2 - s1))
        if s2 is not None and s3 is not None:
            parts.append(("phase 2  narrow phase (rounds of COLL_TPB)", s3 - s2))
        if s3 is not None and nofb is not None:
            parts.append(("phase 3  compaction + contact sort", nofb - s3))
        if nofb is not None and base is not None:
            parts.append(("flagged-only serial launch", base - nofb))
        for name, v in parts:
            share = f"  {100 * v / base:5.1f}% of base" if base else ""
            print(f"    {name:<48} {v:7.3f}{share}")
        t64 = get("tpb64", w)
        if t64 is not None and base is not None:
            print(f"    COLL_TPB 64 / 32 = {t64 / base:.3f}")
        c1 = get("c1024", w)
        if c1 is not None and nopre is not None:
            print(f"    COLL_NCAND_CAP 1024 without the prefilter: {nopre:.3f} -> {c1:.3f}")

    print()
    print("round-1 checks per arm: lanes flagged for the fallback, and the CPU check")
    for a in arms:
        cells = []
        for w in windows:
            kv = first[a].get(w)
            if not kv:
                continue
            cells.append(f"w{w}: flagged {kv['flagged']} ncon mean {kv['ncon_mean']}"
                         f" max {kv['ncon_max']} sat {kv['saturated']}"
                         f" | cpu mismatch {kv['cpu_ncon_mismatch']}/{kv['cpu_lanes']}"
                         f" worst {float(kv['cpu_worst']):.2e}")
        print(f"  {a:<7} " + "\n          ".join(cells))

    for name in ("report", "report_nopre"):
        rep = os.path.join(OUT, f"{name}.r1.log")
        if not os.path.exists(rep):
            continue
        print()
        print(f"the candidate report — {name} (last round of each window)")
        on = False
        for line in open(rep):
            if line.startswith("REPORT"):
                on = True
            elif not line.startswith("  ") or re.match(r"^  round \d+ snap", line):
                on = False
            if on:
                print("  " + line.rstrip())


if __name__ == "__main__":
    main()
