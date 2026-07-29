#!/usr/bin/env python3
"""Flag `while` loops matching the Mojo compile-time-explosion shape.

Found 2026-07-29 while adding `physics3d/sensors/subtree.mojo`. A single call
site took a build from ~2 s to >150 s (never finished). Bisected to a precise
shape — all three conditions are required:

  1. the `while` condition (or a break guard) is DATA-DEPENDENT, i.e. reads an
     array/field rather than a plain counter;
  2. the loop body contains an EARLY EXIT (`break` / `return`);
  3. it sits at loop-nest depth >= 3, counting enclosing loops ACROSS call
     boundaries (a callee's loop counts).

Verified NOT to matter, each tested individually: `@always_inline`,
`@no_inline`, `Tuple` returns, `mut` out-params, `List` vs `InlineArray`,
parameter vs local, package-`__init__` imports, genericity over `DType`,
`continue` on its own, and compile-time-constant trip counts.

Dropping ANY of the three makes it compile in ~2 s. In particular a
data-dependent `while` that runs to a fixpoint with no `break` is fine — which
is why the union-find loops in `physics3d/solver/island_pgs_solve.mojo` are
safe.

FIX: bound the loop — `for _ in range(max)` + `break`. For tree/parent walks
the bound is exact (a chain cannot revisit a node), so nothing is lost. See
`physics3d/sensors/subtree.mojo::walk_to_root`.

Usage:
    python3 scripts/audit_while_compile_risk.py [roots...]     # default: mojo_rl tests

Exit code is the number of RISK findings, so it can gate CI if wanted.

CAVEATS — this is a SHORTLIST, not a verdict:

  * Depth is counted within a single function, so a depth-0 finding can still
    be at risk when its function is called from nested loops (that was the
    original bug), and a flagged RISK may be harmless.
  * The three conditions are necessary but NOT sufficient. As of 2026-07-29
    every site this flags was measured and none explodes: `mcts_cpu.search`
    (depth 4) builds+runs via the AlphaZero arena tests in 22-46 s, the arcade
    `while True` game loops are 1-2 s in the smoke tier, and the parser hits
    are comptime-interpreted (a different mechanism) with every model building
    fine.
  * A likely missing fourth ingredient is a SELF-REFERENTIAL index chase --
    the next index loaded from the very array being indexed (`b = xs[b] - 1`),
    which is what the original bug did. `island_pgs_solve`'s union-find chases
    that way too but has no `break`, so it stays fast.

MEASURE before acting: bisect with a hard time cap (subprocess timeout that
prints SECONDS or TIMEOUT) rather than waiting on a build.
"""
import re
import pathlib
import sys

SKIP = ("references/", "docs-site/", ".pixi/")
DATA_DEP = re.compile(r"\[|\.data|rebind|Int\(")


def scan(paths):
    findings = []
    for root in paths:
        for p in pathlib.Path(root).rglob("*.mojo"):
            if any(s in str(p) for s in SKIP):
                continue
            lines = p.read_text().splitlines()
            fn = None
            for i, ln in enumerate(lines):
                m = re.match(r"^\s*def\s+(\w+)", ln)
                if m:
                    fn = m.group(1)
                w = re.match(r"^(\s*)while\s+(.+?):\s*$", ln)
                if not w:
                    continue
                indent, cond = len(w.group(1)), w.group(2).strip()

                # Body: early exit, and (for `while True`) a data-dependent guard.
                body, j = [], i + 1
                while j < len(lines):
                    l2 = lines[j]
                    if l2.strip() and (len(l2) - len(l2.lstrip())) <= indent:
                        break
                    body.append(l2)
                    j += 1
                body_txt = "\n".join(body)
                has_exit = bool(re.search(r"^\s*(break|return)\b", body_txt, re.M))
                data_dep = bool(DATA_DEP.search(cond)) or (
                    cond in ("True", "1") and bool(DATA_DEP.search(body_txt))
                )
                if not (has_exit and data_dep):
                    continue

                depth = 0
                for k in range(i - 1, -1, -1):
                    l3 = lines[k]
                    if not l3.strip():
                        continue
                    ind3 = len(l3) - len(l3.lstrip())
                    if ind3 >= indent:
                        continue
                    if re.match(r"^\s*(for|while)\b", l3):
                        depth += 1
                    if re.match(r"^\s*def\b", l3):
                        break
                findings.append((str(p), i + 1, fn or "?", cond[:56], depth))
    return findings


def main():
    roots = sys.argv[1:] or ["mojo_rl", "tests"]
    findings = scan(roots)
    risk = [f for f in findings if f[4] >= 2]
    watch = [f for f in findings if f[4] < 2]
    print(f"data-dependent `while` with an early exit: {len(findings)}")
    for label, group in (("RISK (>=2 enclosing loops)", risk),
                         ("watch (check the callers' nesting)", watch)):
        if not group:
            continue
        print(f"\n{label}:")
        for f, ln, fn, cond, d in sorted(group, key=lambda r: -r[4]):
            print(f"  depth {d}  {f}:{ln}  {fn}()  while {cond}")
    return len(risk)


if __name__ == "__main__":
    sys.exit(main())
