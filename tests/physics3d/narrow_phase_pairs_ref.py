"""Synthetic narrow-phase coverage model: every primitive pair type, BOTH
geom-index orderings.

The engine picks its narrow-phase branch from the ORDER the two geoms arrive
in, and the reversed-order branches are the ones that carried bug 35. So each
pair type appears twice: once as (A then B) in XML/geom order and once as
(B then A). Groups are spaced 1 m apart in x so no group can touch another.

Each body carries one slide joint — without a joint every body would be welded
to the world and MuJoCo would exclude every pair.
"""

# half-extent along x for each type, used to set a ~5 mm penetration
HALF_X = {"sphere": 0.05, "capsule": 0.04, "box": 0.05, "cylinder": 0.05}

GEOM = {
    "sphere":   'type="sphere" size=".05"',
    # axis along y, so its x half-extent is the radius
    "capsule":  'type="capsule" size=".04" fromto="0 -.06 0 0 .06 0"',
    "box":      'type="box" size=".05 .05 .05"',
    "cylinder": 'type="cylinder" size=".05 .05"',
}

PAIRS = [
    ("sphere", "capsule"),
    ("sphere", "box"),
    ("sphere", "cylinder"),
    ("capsule", "box"),
    ("capsule", "cylinder"),
    ("box", "cylinder"),
]

PENETRATION = 0.005


def groups():
    """(group_index, typeA, typeB) with both orderings of every pair."""
    out = []
    for a, b in PAIRS:
        out.append((a, b))
        out.append((b, a))
    return out


def make_xml():
    body = []
    for g, (ta, tb) in enumerate(groups()):
        x = g * 1.0
        dx = HALF_X[ta] + HALF_X[tb] - PENETRATION
        body.append(f'''
    <body name="g{g}a" pos="{x} 0 0.5">
      <joint name="j{g}a" type="slide" axis="1 0 0"/>
      <geom name="c{g}a" {GEOM[ta]}/>
    </body>
    <body name="g{g}b" pos="{x + dx} 0 0.5">
      <joint name="j{g}b" type="slide" axis="1 0 0"/>
      <geom name="c{g}b" {GEOM[tb]}/>
    </body>''')
    return f'''<mujoco model="pairs">
  <option timestep="0.002" gravity="0 0 0"/>
  <default>
    <geom friction="1 0.005 0.0001" solimp="0.9 0.95 0.001" solref="0.02 1"/>
  </default>
  <worldbody>{''.join(body)}
  </worldbody>
</mujoco>
'''


def model():
    import mujoco
    return mujoco.MjModel.from_xml_string(make_xml())


if __name__ == "__main__":
    import mujoco, numpy as np
    m = model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    gs = groups()
    print(f"ngeom={m.ngeom} nbody={m.nbody} nv={m.nv} ncon={d.ncon}")
    print(f"{'group':>5} {'types':22} {'ncon':>4}  geom1/geom2  bodies      dist        normal")
    per = {}
    for k in range(d.ncon):
        c = d.contact[k]
        g1, g2 = int(c.geom1), int(c.geom2)
        grp = g1 // 2
        per.setdefault(grp, []).append((g1, g2, c))
    for grp, (ta, tb) in enumerate(gs):
        cs = per.get(grp, [])
        head = f"{grp:>5} {ta+'/'+tb:22} {len(cs):>4}"
        if not cs:
            print(head + "   *** NO CONTACT ***")
            continue
        for (g1, g2, c) in cs:
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            print(head + f"  {g1:>2}/{g2:<2}      {b1}/{b2}   {c.dist:+.6f}  "
                  f"[{c.frame[0]:+.4f} {c.frame[1]:+.4f} {c.frame[2]:+.4f}]")
            head = " " * 33
