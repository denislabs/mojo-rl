#!/usr/bin/env python3
"""MuJoCo's contact count per BODY PAIR at a dumped device pose — the third leg.

    pixi run python tools/tasks/contact_pairs_mujoco.py \
        libero_living_room_scene3 build/diag/lr3_dev_states.txt

`examples/tasks/libero_family_batched.mojo --dump-state PATH` writes the DEVICE's
own `qpos` at the first step where its contact count differs from the CPU leg's.
`tools/tasks/contact_pairs_at_state.mojo` runs OUR CPU detector on those words;
this runs MuJoCo on the same words, printing the same per-pair shape so the
three can be read side by side.

⚠⚠ THE DEVICE COLUMN IN THE TABLE BELOW IS A LAGGED LIST — see
`tools/tasks/contact_pairs_at_state.mojo`. The batched env does not refresh
`d.contacts` after a step, so it describes the state before the last
integration. At EQUAL poses (`tools/tasks/collision_at_pose.mojo`) the GPU
agrees with our CPU and with MuJoCo on these very pairs; the earlier reading of
these rows as a GPU defect is withdrawn.

⚠⚠ WHY THE POSE MATTERS MORE THAN THE COUNT. Two engines stepping the same
scene diverge, and by the time their contact counts differ their STATES already
differ — so a count mismatch says nothing about which one is wrong. Feeding one
recorded pose to all three is what makes it decidable. Measured on
`libero_living_room_scene3` (Metal, 2026-09-16, at 8fea21fb9 + 26318ac16):

    lane 0 step 8   table x alphabet_soup_1    device 6   our CPU 4   MuJoCo 4
    lane 1 step 4   table x ketchup_1          device 2   our CPU 4   MuJoCo 4
    lane 2 step 2   table x wooden_tray_1      device 3   our CPU 3   MuJoCo 3

⚠ `mj_forward` ONLY — no stepping. The dumped words are a state, and the
question is what the collider says about it, not where it goes next.

⚠ THE ARM IS NOT PINNED AND DOES NOT NEED TO BE, for the same reason: nothing
is integrated here.
"""
from __future__ import annotations

import collections
import sys


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit(
            "usage: contact_pairs_mujoco.py <family> <dumped-states.txt>"
            " [substring]"
        )
    family, path = sys.argv[1], sys.argv[2]
    only = sys.argv[3] if len(sys.argv) > 3 else ""

    import mujoco

    xml = f"noeira/tasks/scenes/{family}.xml"
    m = mujoco.MjModel.from_xml_path(xml)
    name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)

    rows = 0
    for line in open(path):
        t = line.split()
        if not t or t[0] != "QPOS":
            continue
        lane, step = int(t[2]), int(t[4])
        q = [float(x) for x in t[5:]]
        if len(q) != m.nq:
            sys.exit(
                f"{path}: lane {lane} step {step} has {len(q)} words, "
                f"{xml} has nq {m.nq} — the dump is not this family's"
            )
        d = mujoco.MjData(m)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        counts: collections.Counter = collections.Counter()
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = name(m.geom_bodyid[c.geom1]), name(m.geom_bodyid[c.geom2])
            counts[tuple(sorted((a, b)))] += 1
        print(f"lane {lane} step {step} : mujoco ncon {d.ncon}")
        for (a, b), n in sorted(counts.items()):
            if only and only not in a and only not in b:
                continue
            print(f"   {n:>3}  {a} x {b}")
        rows += 1
    if rows == 0:
        # ⚠ AN EMPTY DUMP IS A VACUOUS PASS: the driver writes nothing when the
        # counts never differed, which is the GOOD outcome and must not read as
        # "MuJoCo agrees".
        print(f"no QPOS rows in {path} — the run found no count mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
