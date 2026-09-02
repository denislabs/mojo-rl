"""P0 — WHICH TERM carries the quadratic? Merge the four per-k nsys runs.

    pixi run -e nvidia bash scripts/p0_attrib.sh
    pixi run python scripts/p0_attrib.py

Reads `p0_attrib/k{0,3,6,9}.{probe.txt,kern.txt}` and prints, per k, every
kernel's per-step cost — then the same rows grouped into the terms
`docs/BLOCK_DIAGONAL_MASS_MATRIX_PLAN.md` §2.2 is choosing between.

⚠⚠ IT LABELS WHAT IT CAN PROVE AND PRINTS THE REST AS `unlabelled`. It does
NOT sort kernels by duration and assign the biggest to the biggest term: the
whole question is which term is biggest, and a heuristic that assumes an
ordering would answer it by construction. Everything unmatched keeps its
numbers and appears by name, so a first run is informative even if every
pattern misses.

⚠ WHY LABELLING IS NOT TRIVIAL. Mojo emits two symbol styles, both present in
this tree's own captured profiles:

    naive_batched_matmul_kernel_float32_float32_float32_True_79987f6b
    mojo_rl_nn_primitives_conv2d6A6A6A6A6A6A6A_5bd29d73087ee488

The first keeps the function name; the second truncates a long one to a fixed
width, pads with `6A`, and appends a hash. If our kernels take the second form
then `ldl_factor` and `compute_m_inv` — which share `dynamics/ldl.mojo` —
collide on one prefix and differ only by hash. `disambiguate_by_launch_order`
below resolves exactly that case from `cuda_gpu_trace`, using the fact that
`rk4.mojo:565-566` enqueues the factorisation BEFORE the inverse, and it
reports how many cycles agreed rather than asserting it.

⚠ PER-STEP COST IS `avg_ns * launches_per_step`, AND `launches_per_step` IS
DERIVED, NOT ASSUMED. This model is RK4 with FRAME_SKIP=2, so the dynamics
chain runs 4 stages x 2 = 8 times per env step (rk4.mojo:565-566 sit inside
`_stage_dynamics`) — but that is a fact about today's source, and a launch
count read off a comment is exactly the constant that drifts. It is computed
here as `instances / total_steps`, with `total_steps` taken from the probe's
own header, and printed so it can be checked against the source.

⚠ `total_steps` IS WARMUP + TIMED. The warmup launches are in the nsys
instance counts and cannot be separated from the timed ones, which is why this
works from per-launch averages. `ms/step` from the probe's header divides by
TIMED steps only. Mixing the two rescales every term at once, so both numbers
are printed side by side and the residual below is what catches it.
"""

import os
import re
import sys
import csv
from collections import defaultdict

OUT = os.environ.get("OUT", "p0_attrib")
KS = [0, 3, 6, 9]

# ⚠⚠ THESE ARE TRUNCATED MODULE PATHS, AND THE TRUNCATION IS THE POINT.
# Mojo emits a long kernel symbol as the module path CUT TO 29 CHARACTERS, then
# `6A` padding, then a hash of the comptime parameters:
#
#     mojo_rl/nn/primitives/conv2d          (28) -> mojo_rl_nn_primitives_conv2d
#     mojo_rl/nn/primitives/batch_norm      (32) -> mojo_rl_nn_primitives_batch_n
#     mojo_rl/physics3d/solver/newton_solve (37) -> mojo_rl_physics3d_solver_newt
#
# The first two are read off this tree's own captured profile
# (`act_so101_profile_gpu_baseline.txt`); the third off the ptxas error quoted
# in `so101_park_budget_probe.mojo`'s header. Three independent confirmations of
# the same rule.
#
# ⚠ SO A PATTERN MUST NOT SPELL MORE THAN SURVIVES. `physics3d_dynamics_mass`
# never appears — `dynamics/mass_matrix` is 38 characters and arrives as
# `mojo_rl_physics3d_dynamics_ma`. The first draft of this table spelled the
# full names and would have labelled EVERY physics kernel `unlabelled` while
# looking perfectly reasonable. `self_test()` below pins each one.
#
# ⚠ AND ONE COLLISION IS UNFIXABLE HERE: `integrator/rk4` and
# `integrator/euler` both truncate to `mojo_rl_physics3d_integrator_`. On an
# RK4 model only RK4's kernels launch, so the bucket is honestly named
# `integrator` rather than guessing which file it came from.
#
# Ordered; first match wins.
TERMS = [
    ("crba",       r"physics3d_dynamics_ma|mass_matrix|_crba"),
    ("ldl_pair",   r"physics3d_dynamics_ld|_ldl_factor|_m_inv"),
    ("lu",         r"physics3d_dynamics_lu"),
    ("newton",     r"physics3d_solver_newt|newton_solve|_chol"),
    ("rne",        r"physics3d_dynamics_rn|bias_forces"),
    ("cdof",       r"physics3d_dynamics_cd"),
    ("integrator", r"physics3d_integrator_|_finalize_kernel|_rk4_"),
    ("fk",         r"physics3d_kinematics_|forward_kinematics"),
    ("collision",  r"physics3d_collision_|broadphase|contact_detect"),
    ("constraint", r"physics3d_constraints"),
    ("solver_etc", r"physics3d_solver_"),
]


def self_test():
    """Pin every pattern to the symbol Mojo will actually emit.

    ⚠ THIS IS A GATE, NOT A DEMO. The failure it exists for is silent: a
    pattern that spells more of the module path than survives truncation files
    its kernel under `unlabelled`, and the report still prints a full,
    plausible, complete-looking table with the term missing.
    """
    def trunc(mod):
        return mod.replace("/", "_")[:29]

    cases = [
        ("mojo_rl/physics3d/dynamics/mass_matrix", "crba"),
        ("mojo_rl/physics3d/dynamics/ldl", "ldl_pair"),
        ("mojo_rl/physics3d/dynamics/lu", "lu"),
        ("mojo_rl/physics3d/dynamics/rne", "rne"),
        ("mojo_rl/physics3d/dynamics/cdof", "cdof"),
        ("mojo_rl/physics3d/solver/newton_solve", "newton"),
        ("mojo_rl/physics3d/integrator/rk4", "integrator"),
        ("mojo_rl/physics3d/integrator/euler", "integrator"),
        ("mojo_rl/physics3d/kinematics/forward_kinematics", "fk"),
        ("mojo_rl/physics3d/collision/contact_detection", "collision"),
        ("mojo_rl/physics3d/constraints/contact_solve", "constraint"),
    ]
    bad = 0
    print("=== labeller self-test (truncate-to-29 + 6A padding + hash) ===")
    for mod, want in cases:
        sym = trunc(mod) + "6A6A6A6A_deadbeefdeadbeef"
        got = label(sym)
        ok = got == want
        bad += not ok
        print(f"  {'ok ' if ok else 'FAIL'} {sym[:40]:<42} -> {got:<11}"
              f"{'' if ok else '  want ' + want}")
    # Negative control: an nn kernel must NOT be claimed by a physics term.
    for sym in ("mojo_rl_nn_primitives_conv2d6A6A_x",
                "naive_batched_matmul_kernel_float32_float32_float32_True_1",
                "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm>(P)"):
        got = label(sym)
        ok = got == "unlabelled"
        bad += not ok
        print(f"  {'ok ' if ok else 'FAIL'} {sym[:40]:<42} -> {got:<11}"
              f"{'' if ok else '  want unlabelled'}")
    print(f"=== {len(cases)+3-bad} / {len(cases)+3} ===")
    return bad


def label(name):
    for term, pat in TERMS:
        if re.search(pat, name):
            return term
    return "unlabelled"


def read_probe(k):
    """The probe's own header — nv, step counts, ms/step."""
    p = f"{OUT}/k{k}.probe.txt"
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, errors="replace"):
        m = re.match(r"\s{2}(\w+)\s+(-?[\d.]+)\s*$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    need = {"nv", "total_steps", "timed_steps", "ms_per_step"}
    if not need <= out.keys():
        return None
    return out


def read_kern(k, diagnose=False):
    """cuda_gpu_kern_sum -> [(total_ns, instances, avg_ns, name)].

    ⚠⚠ KEYED ON THE `Instances` COLUMN HEADER, NOT ON THE `**` BANNER. The
    banner text and the surrounding chatter move between nsys versions, and a
    reader anchored on them returns ZERO ROWS on a perfectly good report while
    saying only "unparsable". The column header is the stable thing, and it is
    also what distinguishes this table from `cuda_api_sum` — which has the same
    shape but counts `Num Calls`, not `Instances`. Host API time is not kernel
    time, and an unscoped read puts `cuMemFree_v2` on top at 30% of "GPU time".

    ⚠⚠ TWO TABLE DIALECTS, AND THE PIPED ONE IS WHAT AN RTX 5090 BOX EMITTED.
    `nsys stats --format table` renders either whitespace-aligned columns or a
    boxed, PIPE-DELIMITED table depending on version:

        |     69.0 |     11321122872 |      2800 | 4043258.2 | ... | mojo_... |

    A whitespace split on that yields `['|', '69.0', '|', ...]` — every field
    off by one and every row rejected. The delimiter is detected from the
    header rather than assumed, and CSV is accepted too so the driver can move
    to `--format csv` without this needing to change again.

    ⚠ THE NAME IS THE LAST FIELD, and in the whitespace dialect it is the REST
    of the line: vendor kernels carry spaces
    (`void cutlass::Kernel2<...>(T1::Params)`) and splitting on whitespace
    files each fragment as its own kernel.
    """
    p = f"{OUT}/k{k}.kern.txt"
    if not os.path.exists(p):
        if diagnose:
            print(f"    (no such file: {p})")
        return []
    raw = open(p, errors="replace").read().replace("\r", "\n")
    lines = raw.split("\n")

    # Find the header row: the kernel summary's own columns.
    hdr = None
    for i, line in enumerate(lines):
        # ⚠ `Instances` OR `Count` — nsys versions differ. NOT `Calls`:
        # `cuda_api_sum` spells its column `Num Calls` and would be admitted.
        if (("Instances" in line or "Count" in line)
                and ("Total Time" in line or "Time (%)" in line)):
            hdr = i
            break
    if hdr is None:
        if diagnose:
            print(f"    !! no 'Instances' header in {p}. First lines seen:")
            for line in [x for x in lines if x.strip()][:14]:
                print(f"       | {line[:120]}")
        return []

    # Dialect first, then columns. `split_row` is used for the header and for
    # every data row, so the two can never disagree about where a field ends.
    head = lines[hdr]
    if head.count("|") >= 3:
        def split_head(line):
            return [c.strip() for c in line.strip().strip("|").split("|")]
        def split_row(line, _n):
            return [c.strip() for c in line.strip().strip("|").split("|")]
    elif head.count(",") >= 3 and '"' not in head:
        def split_head(line):
            return [c.strip() for c in line.strip().split(",")]
        def split_row(line, _n):
            return [c.strip() for c in line.strip().split(",")]
    else:
        # ⚠ THE HEADER AND THE DATA SPLIT DIFFERENTLY IN THIS DIALECT, and
        # conflating them is a real trap: column TITLES contain single spaces
        # (`Total Time (ns)`), so the header must split on RUNS of whitespace,
        # while a data row splits on any whitespace with the NAME taking the
        # remainder. Using one rule for both turns `Time (%)` into two columns
        # and every field index is then wrong.
        def split_head(line):
            return re.split(r"\s{2,}", line.strip())
        def split_row(line, n):
            return [c.strip() for c in line.split(None, n - 1)]

    cols = split_head(head)
    def idx(*want):
        for j, c in enumerate(cols):
            if any(w in c for w in want):
                return j
        return None
    i_tot = idx("Total Time")
    i_inst = idx("Instances", "Count")
    i_avg = idx("Avg")
    if None in (i_tot, i_inst, i_avg):
        if diagnose:
            print(f"    !! header found but columns not: {cols}")
        return []
    ncol = len(cols)

    rows = []
    for line in lines[hdr + 1:]:
        if not line.strip():
            continue
        if re.match(r"\s*\*\*\s", line):
            break                       # next report section
        # Box rules (`+----+----+`) and dashed separators carry no data.
        if set(line.strip()) <= set("+-| "):
            continue
        f = split_row(line, ncol)
        if len(f) < ncol:
            continue
        try:
            total = float(f[i_tot].replace(",", ""))
            inst = int(f[i_inst].replace(",", ""))
            avg = float(f[i_avg].replace(",", ""))
        except (ValueError, IndexError):
            continue                    # separator / continuation rows
        name = f[ncol - 1].strip()
        # Belt and braces: a CUDA driver entry point is not a kernel.
        if re.match(r"cu[A-Z]\w*(_v\d)?$", name):
            continue
        rows.append((total, inst, avg, name))
    if not rows and diagnose:
        print(f"    !! header at line {hdr} but no data rows parsed. "
              f"cols={cols}")
        for line in lines[hdr:hdr + 6]:
            print(f"       | {line[:120]}")
    return rows


# The `dynamics/ldl.mojo` kernels, in the order RK4 enqueues them
# (rk4.mojo:565, :566, :611). All three share one truncated module prefix and
# differ only by hash, so this order is the only thing that identifies them.
LDL_ROLES = ["ldl_factor", "compute_m_inv", "ldl_solve"]


def disambiguate_by_launch_order(k, names):
    """Identify same-prefix kernels by their ORDER after an anchor launch.

    Returns ({name: role}, why). `dynamics/ldl.mojo` contributes three kernels
    under one truncated prefix — `ldl_factor`, `compute_m_inv` and `ldl_solve`
    — and CRBA (`dynamics/mass_matrix.mojo`, a different prefix) is enqueued
    before all of them at rk4.mojo:536. So within each stage the order after a
    CRBA launch is exactly `LDL_ROLES`.

    ⚠⚠ ADJACENCY BETWEEN THE TARGETS ALONE PROVES NOTHING, and the first
    version of this used exactly that. Filter a trace to kernels that cycle in
    a fixed order and the pairwise counts come out even whichever rotation is
    the real one — a fixture with a KNOWN order returned "undecided", which is how it was
    caught. The anchor is what carries the information.

    ⚠ IT REPORTS AGREEMENT RATHER THAN ASSERTING IT. No anchor, a ragged
    cycle, or a count mismatch means "undecided": the kernels stay pooled. A
    pooled honest number beats a split invented one.
    """
    p = f"{OUT}/k{k}.trace.csv"
    if not os.path.exists(p):
        return {}, "no trace file"
    if not 2 <= len(names) <= len(LDL_ROLES):
        return {}, (f"{len(names)} kernels under one prefix; "
                    f"only 2..{len(LDL_ROLES)} are named in LDL_ROLES")
    tgt = set(names)
    seq = []
    try:
        with open(p, newline="", errors="replace") as fh:
            for row in csv.DictReader(fh):
                nm = (row.get("Name") or row.get("name") or "").strip()
                if not nm:
                    continue
                if nm in tgt:
                    seq.append(nm)
                elif label(nm) == "crba":
                    seq.append(None)          # the anchor
    except Exception as e:                       # noqa: BLE001
        return {}, f"trace unreadable: {e}"
    if None not in seq:
        return {}, "no CRBA launch in the trace to anchor on"

    # Each cycle: the distinct targets seen after an anchor, in first-seen
    # order, up to the next anchor.
    votes, cycles, ragged = {}, 0, 0
    cur = []
    for tok in seq + [None]:
        if tok is None:
            if cur:
                cycles += 1
                if len(cur) == len(names):
                    votes[tuple(cur)] = votes.get(tuple(cur), 0) + 1
                else:
                    ragged += 1
            cur = []
        elif tok not in cur:
            cur.append(tok)
    if not votes:
        return {}, f"no complete cycle ({cycles} cycles, {ragged} ragged)"
    order, n = max(votes.items(), key=lambda kv: kv[1])
    total = sum(votes.values())
    if n < 0.9 * total:
        return {}, (f"order not stable ({n}/{total} cycles agree; "
                    f"{len(votes)} distinct orders)")
    return ({nm: LDL_ROLES[i] for i, nm in enumerate(order)},
            f"{n}/{total} cycles"
            + (f", {ragged} ragged ignored" if ragged else ""))


def main():
    probes, kerns = {}, {}
    for k in KS:
        pr, kr = read_probe(k), read_kern(k)
        if pr is None or not kr:
            # ⚠ SAY WHAT WAS SEEN. "unparsable" with no evidence is a dead end;
            # re-read with diagnostics so the next move is obvious.
            print(f"!! k={k}: missing or unparsable"
                  f"{' [probe]' if pr is None else ''}"
                  f"{' [kern]' if not kr else ''}")
            if pr is None and os.path.exists(f"{OUT}/k{k}.probe.txt"):
                print(f"    !! {OUT}/k{k}.probe.txt has no "
                      f"'nv/total_steps/timed_steps/ms_per_step' header lines")
            read_kern(k, diagnose=True)
        else:
            probes[k], kerns[k] = pr, kr
    if not probes:
        print("nothing to report."); sys.exit(1)

    # ── per-kernel, per-k ─────────────────────────────────────────────────
    for k in sorted(probes):
        pr, kr = probes[k], kerns[k]
        steps = pr["total_steps"]
        kr = sorted(kr, key=lambda r: -r[0])
        gpu_ns = sum(r[0] for r in kr)
        print(f"\n=== k={k}  nv={int(pr['nv'])}  "
              f"probe {pr['ms_per_step']:.2f} ms/step  "
              f"(wall over {int(pr['timed_steps'])} timed steps) ===")
        print(f"  {'term':<11}{'launch/step':>12}{'avg us':>10}"
              f"{'ms/step':>10}{'% gpu':>8}  kernel")
        print("  " + "-" * 86)
        for total, inst, avg, name in kr[:18]:
            per_step = inst / steps
            ms = avg * per_step / 1e6
            print(f"  {label(name):<11}{per_step:>12.2f}{avg/1e3:>10.1f}"
                  f"{ms:>10.3f}{100*total/gpu_ns:>8.1f}  {name[:70]}")
        if len(kr) > 18:
            print(f"  ... {len(kr)-18} more kernels "
                  f"({100*sum(r[0] for r in kr[18:])/gpu_ns:.1f}% of gpu time)")

    # ── the five-term table, which is the actual P0 deliverable ───────────
    print("\n\n=== P0: per-step ms by TERM ===")
    print("⚠ per-step = avg_ns * (instances/total_steps); both derived, "
          "neither assumed.\n")
    terms = sorted({label(n) for k in kerns for *_, n in kerns[k]})
    terms = [t for t in terms if t != "unlabelled"] + (
        ["unlabelled"] if any(label(n) == "unlabelled"
                              for k in kerns for *_, n in kerns[k]) else [])
    ks = sorted(probes)
    print(f"  {'term':<12}" + "".join(f"{'k='+str(k):>12}" for k in ks)
          + f"{'d/dnv^2':>12}")
    print("  " + "-" * (12 + 12 * len(ks) + 12))
    per = defaultdict(dict)
    for k in ks:
        steps = probes[k]["total_steps"]
        for total, inst, avg, name in kerns[k]:
            per[label(name)][k] = per[label(name)].get(k, 0.0) + \
                avg * (inst / steps) / 1e6
    nv = {k: probes[k]["nv"] for k in ks}
    for t in terms:
        row = "".join(f"{per[t].get(k, 0.0):>12.3f}" for k in ks)
        # The quadratic test: excess over k=0 divided by the added dofs SQUARED.
        # Constant down a row = that term is the quadratic. The scope doc's own
        # curve gives 0.0173 / 0.0163 / 0.0187 for the whole step.
        if len(ks) > 1 and ks[0] in per[t]:
            k9 = ks[-1]
            d = per[t].get(k9, 0.0) - per[t][ks[0]]
            dn = nv[k9] - nv[ks[0]]
            q = f"{d/(dn*dn):>12.5f}" if dn else f"{'-':>12}"
        else:
            q = f"{'-':>12}"
        print(f"  {t:<12}{row}{q}")
    tot = {k: sum(per[t].get(k, 0.0) for t in terms) for k in ks}
    print("  " + "-" * (12 + 12 * len(ks) + 12))
    print(f"  {'GPU TOTAL':<12}" + "".join(f"{tot[k]:>12.3f}" for k in ks))
    print(f"  {'probe wall':<12}"
          + "".join(f"{probes[k]['ms_per_step']:>12.3f}" for k in ks))
    # ⚠ THE RESIDUAL IS THE HONESTY CHECK. GPU kernel time under the wall time
    # is gaps — launch overhead, host work, sync. Far OVER it means the
    # launches/step divisor is wrong, and every term above is wrong with it.
    print(f"  {'residual':<12}"
          + "".join(f"{probes[k]['ms_per_step']-tot[k]:>12.3f}" for k in ks)
          + "   <- gaps/host if +, a WRONG DIVISOR if -")

    # ── split the ldl pair, if it collided ────────────────────────────────
    print("\n=== splitting dynamics/ldl.mojo by launch order ===")
    print("  (ldl_factor, compute_m_inv and ldl_solve share one "
          "truncated prefix; rk4.mojo:565/566/611 fixes the order)")
    for k in ks:
        names = sorted({n for *_, n in kerns[k] if label(n) == "ldl_pair"})
        if len(names) <= 1:
            print(f"  k={k}: {len(names)} kernel(s) under `ldl_pair` — "
                  f"{'nothing to split' if names else 'NONE FOUND, check the patterns'}")
            for n in names:
                print(f"        {n}")
            continue
        roles, why = disambiguate_by_launch_order(k, names)
        if roles:
            avg = {n: (a, i) for _, i, a, n in
                   [(t, i, a, n) for t, i, a, n in kerns[k]] if n in roles}
            steps = probes[k]["total_steps"]
            print(f"  k={k}: split by launch order ({why})")
            for n, role in roles.items():
                a, i = avg[n]
                print(f"        {role:<14}{a*(i/steps)/1e6:>8.3f} ms/step   {n[:60]}")
        else:
            print(f"  k={k}: NOT SPLIT — {why}. The two stay pooled above;")
            print(f"        that is a pooled honest number, not a split guess.")
            for n in names:
                print(f"        {n}")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(1 if self_test() else 0)
    main()
