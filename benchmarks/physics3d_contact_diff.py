"""Diff two contact-probe logs (ours, MuJoCo): the first step where the body-pair
multisets differ, with the pairs each side has that the other lacks, and a
summary of which pairs carry the count gap over the whole run.

    python3 benchmarks/physics3d_contact_diff.py ours.txt mujoco.txt [xml]
"""
import sys
from collections import Counter

def load(p):
    rows = []
    for ln in open(p):
        if not ln.startswith("STEP "):
            continue
        toks = ln.split()
        n = int(toks[1]); ncon = int(toks[2].split("=")[1]); qsum = float(toks[3].split("=")[1])
        pairs = toks[4][len("pairs="):]
        # our record calls the world body -1; MuJoCo calls it 0
        norm = [p.replace("-1-", "0-", 1) if p.startswith("-1-") else p for p in pairs.split(",")] if pairs else []
        rows.append((n, ncon, qsum, Counter(norm)))
    return rows

ours, mj = load(sys.argv[1]), load(sys.argv[2])
names = {}
if len(sys.argv) > 3:
    import mujoco
    m = mujoco.MjModel.from_xml_path(sys.argv[3])
    names = {i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or str(i) for i in range(m.nbody)}
def nm(pair):
    a, b = pair.rsplit("-", 1); return f"{names.get(int(a), a)}~{names.get(int(b), b)}"

first = None
gap = Counter(); gap_mj = Counter(); tot_o = tot_m = 0
for (n, no, qo, po), (_, nmj, qm, pm) in zip(ours, mj):
    tot_o += no; tot_m += nmj
    if po != pm and first is None:
        first = n
        print(f"first divergence at step {n}: ours ncon={no} MuJoCo ncon={nmj}  |dqsum|={abs(qo-qm):.3e}")
        print("  only ours  :", ", ".join(f"{nm(k)} x{v}" for k, v in sorted((po - pm).items())))
        print("  only MuJoCo:", ", ".join(f"{nm(k)} x{v}" for k, v in sorted((pm - po).items())))
    gap += (pm - po); gap_mj += pm
n = min(len(ours), len(mj))
print(f"\n{n} steps: mean ncon ours {tot_o/n:.2f}  MuJoCo {tot_m/n:.2f}")
print("pairs MuJoCo has more of, summed over the run (extra contacts / MuJoCo's total for that pair):")
for k, v in gap.most_common(12):
    print(f"  {nm(k):40s} +{v:6d}  of {gap_mj[k]:6d}")
extra = Counter()
for (n, no, qo, po), (_, nmj, qm, pm) in zip(ours, mj):
    extra += (po - pm)
if extra:
    print("pairs OURS has more of:")
    for k, v in extra.most_common(6):
        print(f"  {nm(k):40s} +{v:6d}")
# trajectory drift
for step in (0, 10, 100, 500, 1000, 2000):
    if step < n:
        print(f"  step {step}: |dqsum| = {abs(ours[step][2]-mj[step][2]):.3e}  ncon {ours[step][1]} vs {mj[step][1]}")
