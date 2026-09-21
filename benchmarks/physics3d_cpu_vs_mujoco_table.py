"""Fold the `RESULT` lines of `scripts/physics3d_cpu_vs_mujoco.sh` into a table.

    pixi run python benchmarks/physics3d_cpu_vs_mujoco_table.py <results.txt>

Per model and side: the MIN over rounds (the rounds were interleaved, so a
slow round is the machine, not the code), the spread `max/min - 1` so a row
whose rounds disagree is visible, and the work counters. A row whose contact
counts disagree between the two sides is flagged: that row is two different
problems, not a ratio.
"""
import sys
from collections import defaultdict

rows = defaultdict(list)
for ln in open(sys.argv[1]):
    if not ln.startswith("RESULT "):
        continue
    kv = dict(tok.split("=", 1) for tok in ln.split()[1:])
    rows[(kv["model"], kv["side"])].append(kv)

models = []
for (model, _side) in rows:
    if model not in models:
        models.append(model)

print(f"{'model':<14} {'nv':>3} {'integ':<6} {'ours us':>9} {'spread':>7} "
      f"{'mujoco us':>9} {'spread':>7} {'ratio':>6}  "
      f"{'ncon ours/mj':>13} {'nefc':>5} {'niter':>5} {'rounds':>6}")
for model in models:
    o = rows.get((model, "ours"), [])
    mj = rows.get((model, "mujoco"), [])
    if not o or not mj:
        print(f"{model:<14}  MISSING ours={len(o)} mujoco={len(mj)}")
        continue
    ot = [float(r["us_per_step"]) for r in o]
    mt = [float(r["us_per_step"]) for r in mj]
    o_min, m_min = min(ot), min(mt)
    o_spr = max(ot) / o_min - 1
    m_spr = max(mt) / m_min - 1
    onc = float(o[0]["ncon_mean"])
    mnc = float(mj[0]["ncon_mean"])
    flag = "" if abs(onc - mnc) <= max(0.5, 0.1 * max(onc, mnc)) else "  !! ncon differs"
    integ = o[0]["integ"] + ("" if o[0]["integ"] == mj[0]["integ"] else "/" + mj[0]["integ"])
    print(f"{model:<14} {o[0]['nv']:>3} {integ:<6} {o_min:9.3f} {100*o_spr:6.1f}% "
          f"{m_min:9.3f} {100*m_spr:6.1f}% {o_min/m_min:6.2f}x  "
          f"{onc:6.2f}/{mnc:<6.2f} {float(mj[0]['nefc_mean']):5.1f} "
          f"{float(mj[0]['niter_mean']):5.2f} {min(len(o), len(mj)):>6}{flag}")
if models:
    print(f"\nours = {o[0]['dtype']} single env, MuJoCo {mj[0].get('version','?')} f64, "
            f"{o[0]['steps']} timed steps/round, MIN of interleaved rounds, bvactive=0.")
