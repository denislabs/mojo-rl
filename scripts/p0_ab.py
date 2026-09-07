"""The table for scripts/p0_ab.sh — per term and per kernel, A vs B, MIN over
rounds, with the sign counted across rounds so a one-round fluke shows as one.

    OUT=p0_ab python scripts/p0_ab.py

Reads `OUT/{A,B}/r*/k*.{probe,kern}.txt`. Per-step cost is
`avg_ns * instances / total_steps` exactly as scripts/p0_attrib.py derives it
(the warmup launches are in the instance count; the two arms share the bias).
The per-launch average of the single largest kernel of each term is printed
too, because the term sum hides a kernel that appeared or vanished — the
Sep 4 -> Sep 7 comparison lost one of three LDL launches, and a term-only
table would have read that as "ldl 2x faster", which is true but not why.
"""

import glob
import os
import re
from collections import defaultdict

OUT = os.environ.get("OUT", "p0_ab")

# The truncated-module-path labeller from scripts/p0_attrib.py; first match
# wins. Mojo cuts the module path to 29 characters before the `6A` padding.
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


def label(name):
    for term, pat in TERMS:
        if re.search(pat, name):
            return term
    return "unlabelled"


# One regex for both nsys dialects (pipe-boxed and whitespace-aligned): the
# eight numeric columns of cuda_gpu_kern_sum, then the name as the rest.
_ROW = re.compile(
    r"^\s*\|?\s*([\d.]+)\s*\|?\s*(\d+)\s*\|?\s*(\d+)\s*\|?\s*([\d.]+)\s*\|?\s*"
    r"([\d.]+)\s*\|?\s*(\d+)\s*\|?\s*(\d+)\s*\|?\s*([\d.]+)\s*\|?\s*(\S.*?)\s*\|?\s*$"
)


def read_kern(path):
    """-> {name: (instances, avg_ns)}."""
    out = {}
    for line in open(path, errors="replace"):
        m = _ROW.match(line)
        if m:
            out[m.group(9)] = (int(m.group(3)), float(m.group(4)))
    return out


def read_probe(path):
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"\s{2}(\w+)\s+(-?[\d.]+)\s*$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def rounds(arm):
    """-> {k: [(wall_ms, {term: ms}, {term: (avg_us, name)})...]} per round."""
    res = defaultdict(list)
    for kern in sorted(glob.glob(f"{OUT}/{arm}/r*/k*.kern.txt")):
        probe = kern.replace(".kern.txt", ".probe.txt")
        k = int(re.search(r"/k(\d+)\.kern\.txt$", kern).group(1))
        pr = read_probe(probe)
        if "total_steps" not in pr or "ms_per_step" not in pr:
            print(f"  !! {probe}: no total_steps/ms_per_step header — leg skipped")
            continue
        steps = pr["total_steps"]
        terms = defaultdict(float)
        biggest = {}
        for name, (inst, avg) in read_kern(kern).items():
            t = label(name)
            terms[t] += avg * inst / steps / 1e6
            if t not in biggest or avg / 1e3 > biggest[t][0]:
                biggest[t] = (avg / 1e3, name)
        res[k].append((pr["ms_per_step"], dict(terms), biggest))
    return res


def main():
    if os.path.exists(f"{OUT}/ARMS.txt"):
        print(open(f"{OUT}/ARMS.txt").read().rstrip())
    ra, rb = rounds("A"), rounds("B")
    ks = sorted(set(ra) & set(rb))
    if not ks:
        print(f"nothing to compare under {OUT}/ (A: {sorted(ra)}, B: {sorted(rb)})")
        return 1
    for k in ks:
        A, B = ra[k], rb[k]
        n = min(len(A), len(B))
        print(f"\n=== k={k}   rounds A={len(A)} B={len(B)}   "
              f"per-step ms, MIN over rounds; ratio = B/A ===")
        wa, wb = min(r[0] for r in A), min(r[0] for r in B)
        print(f"  {'wall':<11}{wa:>10.3f}{wb:>10.3f}{wb/wa:>8.3f}")
        terms = sorted(set().union(*[r[1] for r in A + B]),
                       key=lambda t: -max(r[1].get(t, 0) for r in A))
        print(f"  {'term':<11}{'A':>10}{'B':>10}{'B/A':>8}  B<A in rounds")
        for t in terms:
            ma = min(r[1].get(t, 0.0) for r in A)
            mb = min(r[1].get(t, 0.0) for r in B)
            # Sign across PAIRED rounds: a real move has one sign in every
            # round; a fluke has it in one.
            sign = sum(1 for i in range(n)
                       if B[i][1].get(t, 0.0) < A[i][1].get(t, 0.0))
            ratio = mb / ma if ma > 0 else float("nan")
            print(f"  {t:<11}{ma:>10.3f}{mb:>10.3f}{ratio:>8.3f}  {sign}/{n}")
        print(f"  -- largest kernel per term, avg us/launch, MIN over rounds")
        for t in terms:
            ka = [r[2][t] for r in A if t in r[2]]
            kb = [r[2][t] for r in B if t in r[2]]
            if not ka or not kb:
                continue
            # MIN over rounds of the per-launch average, of the kernel that
            # was the term's largest in that round.
            ua, na = min(ka)
            ub, nb = min(kb)
            same = "" if na == nb else "  (different kernel hash: code path changed)"
            print(f"  {t:<11}{ua:>10.1f}{ub:>10.1f}{ub/ua:>8.3f}"
                  f"  {nb.replace('mojo_rl_physics3d_', '')[:44]}{same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
