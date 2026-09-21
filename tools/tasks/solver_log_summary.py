"""The distributions behind a `libero_demo_batched --solver-log` CSV.

    pixi run python tools/tasks/solver_log_summary.py /tmp/libero_256_solver.csv
    pixi run python tools/tasks/solver_log_summary.py /tmp/x.csv --every 5

PERFORMANCE.md §13.53 asks one question of this file: is the Newton kernel
slow because each solve is EXPENSIVE (few iterations, heavy per iteration —
the blocked-kernel lever) or because solves run LONG (many iterations, a
cap hit often — the budget lever)? The driver prints the means; means hide
exactly the shape that decides it ("a solver that takes 3 iterations on
most steps and 100 on a few has a different problem from one that always
takes 50" — `newton_solve.mojo`'s own note on `NEWTON_ITER_REPORT`). So
this prints the distributions.

Columns (one row per control step and lane):

    t, lane, task, in_demo        the step, the lane, its task index, and
                                  whether the lane's demonstration is still
                                  running (past its end the action is zero)
    ncon_last, it_last, lsev_last the LAST substep's contact count, Newton
                                  iterations and line-search evaluations
    it_sum, lsev_sum, ncon_sum    the same three summed over the step's
                                  substeps (25 on LIBERO)
    capped_sum                    solves in the step that hit `iterations`

`it_last` is one solve in 25 — the per-solve distribution below is that
sample, not every solve. The sums are every solve.
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict


def _pct(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    k = int(round((len(xs) - 1) * q))
    return xs[k]


def _hist(xs, edges):
    counts = [0] * (len(edges) + 1)
    for x in xs:
        i = 0
        while i < len(edges) and x > edges[i]:
            i += 1
        counts[i] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--every", type=int, default=5,
                    help="print the per-step table every N steps")
    ap.add_argument("--substeps", type=int, default=25,
                    help="physics substeps per control step (the driver's"
                         " SUBSTEPS; LIBERO's frame skip is 25)")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    if not rows:
        sys.exit("empty log")
    for r in rows:
        for k in r:
            r[k] = int(r[k])
    S = a.substeps
    steps = sorted({r["t"] for r in rows})
    lanes = sorted({r["lane"] for r in rows})
    print(f"{len(rows)} rows: {len(steps)} control steps x {len(lanes)} lanes,"
          f" {S} substeps each -> {len(rows) * S} solves")

    # ── per-step table ─────────────────────────────────────────────────
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["t"]].append(r)
    print()
    print("per control step (means are per SOLVE, over every lane and substep)")
    print(f"{'t':>4} {'live':>5} {'ncon':>6} {'ncon_lastmax':>12} {'iters':>6}"
          f" {'it_lastmax':>10} {'lsev/it':>8} {'capped':>7}")
    for t in steps:
        if t % a.every and t != steps[-1]:
            continue
        rs = by_t[t]
        n = len(rs) * S
        it = sum(r["it_sum"] for r in rs)
        ls = sum(r["lsev_sum"] for r in rs)
        nc = sum(r["ncon_sum"] for r in rs)
        cap = sum(r["capped_sum"] for r in rs)
        print(f"{t:>4} {sum(r['in_demo'] for r in rs):>5} {nc / n:>6.2f}"
              f" {max(r['ncon_last'] for r in rs):>12} {it / n:>6.2f}"
              f" {max(r['it_last'] for r in rs):>10}"
              f" {(ls / it if it else 0):>8.2f} {cap:>7}")

    # ── the per-solve distributions (the 1-in-S sample) ────────────────
    it_last = [r["it_last"] for r in rows]
    nc_last = [r["ncon_last"] for r in rows]
    print()
    print("Newton iterations per solve — the last-substep sample"
          f" ({len(it_last)} solves)")
    print(f"  mean {statistics.mean(it_last):.2f}  p50 {_pct(it_last, .5)}"
          f"  p90 {_pct(it_last, .9)}  p99 {_pct(it_last, .99)}"
          f"  max {max(it_last)}")
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 99]
    labels = ["0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64",
              "65-99", ">=100"]
    counts = _hist(it_last, edges)
    for lab, c in zip(labels, counts):
        if c:
            bar = "#" * int(60 * c / len(it_last))
            print(f"  {lab:>6} {c:>7} {100 * c / len(it_last):5.1f}% {bar}")

    print()
    print("contacts handed to the solve — the last-substep sample")
    print(f"  mean {statistics.mean(nc_last):.2f}  p50 {_pct(nc_last, .5)}"
          f"  p90 {_pct(nc_last, .9)}  max {max(nc_last)}")

    # ── iterations against contacts: is the cost the SCENE or the SOLVER ─
    print()
    print("iterations per solve by contact count (last-substep sample)")
    by_nc = defaultdict(list)
    for r in rows:
        by_nc[min(r["ncon_last"] // 8 * 8, 64)].append(r["it_last"])
    for lo in sorted(by_nc):
        xs = by_nc[lo]
        print(f"  ncon {lo:>2}-{lo + 7:<2} n={len(xs):>6} iters mean"
              f" {statistics.mean(xs):6.2f} p90 {_pct(xs, .9):>4}"
              f" max {max(xs):>4}")

    # ── the whole run: every solve, through the sums ───────────────────
    n = len(rows) * S
    it = sum(r["it_sum"] for r in rows)
    ls = sum(r["lsev_sum"] for r in rows)
    nc = sum(r["ncon_sum"] for r in rows)
    cap = sum(r["capped_sum"] for r in rows)
    print()
    print(f"every solve ({n}): iters/solve {it / n:.2f}, lsev/iter"
          f" {(ls / it if it else 0):.2f}, ncon/solve {nc / n:.2f},"
          f" at the cap {cap} ({100 * cap / n:.2f}%)")

    # ── per task ───────────────────────────────────────────────────────
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    print()
    print("per task (every solve)")
    for k in sorted(by_task):
        rs = by_task[k]
        n = len(rs) * S
        it = sum(r["it_sum"] for r in rs)
        nc = sum(r["ncon_sum"] for r in rs)
        cap = sum(r["capped_sum"] for r in rs)
        print(f"  task {k:>2}: iters/solve {it / n:6.2f}  ncon/solve"
              f" {nc / n:6.2f}  capped {cap:>6}  it_last max"
              f" {max(r['it_last'] for r in rs):>4}")

    # ── the verdict the section asked for ──────────────────────────────
    print()
    n = len(rows) * S
    it = sum(r["it_sum"] for r in rows)
    cap = sum(r["capped_sum"] for r in rows)
    mean_it = it / n
    frac_cap = cap / n
    if frac_cap > 0.05 or _pct(it_last, .9) >= 50:
        print("READ: solves run LONG (p90 >= 50 iterations or >5% at the cap)"
              " — the budget and the float32 convergence floor are in play"
              " (§13.53 lever 4, §13.7); price the cap before the kernel.")
    else:
        print(f"READ: solves are SHORT (mean {mean_it:.1f} iterations, p90"
              f" {_pct(it_last, .9)}, {100 * frac_cap:.2f}% at the cap) —"
              " the cost is per iteration, i.e. the per-env kernel's serial"
              " chain (§13.53 lever 3, the blocked elliptic kernel).")


if __name__ == "__main__":
    main()
