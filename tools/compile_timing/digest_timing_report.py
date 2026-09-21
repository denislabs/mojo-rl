#!/usr/bin/env python3
"""Digest a `mojo build --mlir-timing --llvm-timing` text report.

Prints: the MLIR top-level rows (the compile total is the MLIR root),
the ElaborateGenerators subtree with the offload scope, and the LLVM
pass totals per pipeline aggregated by pass name (the `#N` instance
suffix stripped), so a pass that runs once per module is one row.
"""
import re
import sys
from collections import defaultdict

# Single-threaded reports print one column (wall); default-thread reports print
# two (user, wall). Either way the LAST duration column is the wall time.
ROW = re.compile(
    r"^( +)(?:[0-9]+\.[0-9]+ \( *-?[0-9.]+%\) +)?([0-9]+\.[0-9]+) \( *(-?[0-9.]+)%\)( +)(.*)$"
)
LLVM_ROW = re.compile(
    r"^ +([0-9.]+) \( *[0-9.]+%\) +([0-9.]+) \( *[0-9.]+%\) +([0-9.]+) \( *[0-9.]+%\)"
    r" +([0-9.]+) \( *[0-9.]+%\) +([0-9]+) +(.*)$"
)


def parse(path):
    text = open(path).read().splitlines()
    sections = []
    cur = None
    for line in text:
        if line.startswith("===--- "):
            cur = {"title": line.strip("= -"), "lines": []}
            sections.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
    return sections


def mlir_rows(lines):
    rows = []
    for line in lines:
        m = ROW.match(line)
        if not m:
            continue
        indent = len(m.group(4))
        rows.append((indent, float(m.group(2)), float(m.group(3)), m.group(5).strip()))
    return rows


def digest_mlir(sec, top):
    rows = mlir_rows(sec["lines"])
    if not rows:
        return
    base = min(r[0] for r in rows)
    total = next((r for r in rows if r[3] == "Total"), None)
    print(f"MLIR root (whole compile): {total[1]:.2f}s" if total else "no Total row")
    print("\nTop-level rows (>= 0.5% of the compile):")
    top_rows = [r for r in rows if r[0] == base and r[3] not in ("Total",)]
    for r in sorted(top_rows, key=lambda r: -abs(r[1]))[: top]:
        tag = "  <- scope, counted inside ElaborateGenerators" if "also in host" in r[3] else ""
        print(f"  {r[1]:9.2f}s ({r[2]:5.1f}%)  {r[3][:70]}{tag}")
    # ElaborateGenerators subtree: rows after it until the next base-indent row.
    print("\nElaborateGenerators subtree (children, > 0.1s):")
    inside = False
    for r in rows:
        if r[0] == base and r[3] == "ElaborateGenerators":
            inside = True
            continue
        if inside and r[0] == base and "also in host" not in r[3]:
            break
        if inside and r[1] > 0.1:
            print(f"  {' ' * (r[0] - base)}{r[1]:8.2f}s ({r[2]:5.1f}%)  {r[3][:70]}")
    # Pipelines by name, summed across instances.
    agg = defaultdict(float)
    for r in rows:
        if r[3] not in ("Total", "Rest") and "also in host" not in r[3]:
            agg[r[3]] += r[1]
    print("\nMLIR pass names summed over every instance and depth (top):")
    for name, t in sorted(agg.items(), key=lambda kv: -kv[1])[: top]:
        print(f"  {t:9.2f}s  {name[:70]}")


def digest_llvm(sec, top):
    print(f"\n{sec['title']}")
    group = None
    totals = {}
    agg = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    for line in sec["lines"]:
        s = line.strip()
        if s in (
            "Pass execution timing report",
            "Analysis execution timing report",
            "Instruction Selection and Scheduling",
            "Register Allocation",
        ):
            group = s
            continue
        if s.startswith("Total Execution Time:") and group:
            totals[group] = float(s.split()[3])
            continue
        m = LLVM_ROW.match(line)
        if m and group:
            name = re.sub(r" #\d+$", "", m.group(6).strip())
            if name == "Total":
                continue
            agg[group][name] += float(m.group(4))
            counts[group][name] += 1
    for g in ("Pass execution timing report", "Analysis execution timing report"):
        if g in totals:
            print(f"  {g}: {totals[g]:.2f}s wall")
            for name, t in sorted(agg[g].items(), key=lambda kv: -kv[1])[: top]:
                print(f"    {t:8.2f}s  x{counts[g][name]:<5d} {name[:60]}")
    for g in ("Instruction Selection and Scheduling", "Register Allocation"):
        if g in totals:
            print(f"  {g}: {totals[g]:.2f}s (inside the pass report, not additional)")


def main():
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    for sec in parse(path):
        if sec["title"].startswith("MLIR pass timing"):
            digest_mlir(sec, top)
        elif sec["title"].startswith("LLVM pass timing"):
            digest_llvm(sec, top)


if __name__ == "__main__":
    main()
