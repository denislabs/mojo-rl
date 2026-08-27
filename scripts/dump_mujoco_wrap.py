"""Generate `tests/physics3d/wrap_goldens.mojo` from MuJoCo 3.10.0.

`mju_wrap` is NOT exposed to the Python bindings (`dir(mujoco)` has only
`MjsWrap`/`mjtWrap`), so the oracle has to be a real model: build a one-tendon
`site -> geom -> site` MJCF, sweep the moving site through poses that wrap and
poses that do not, and read the answer off `mjData`.

    d.ten_length[t]                     total tendon length
    d.ten_wrapnum[t]                    2 = straight, 4 = wrapped
    d.wrap_xpos[adr .. adr+3*num]       the waypoints, wrap points included

The arc length MuJoCo used is then `ten_length - |x0-w0| - |w1-x1|`, which is
what `mju_wrap` returns. That is a DERIVED golden and it is the only way to
get one; the wrap POINTS are read directly and are the stronger arm.

⚠ THE POSES ARE CHOSEN TO COVER THE BRANCHES, not to look like a robot: no
wrap, wrap either side, a sidesite forcing the LONG way round, a sidesite
INSIDE the object (`wrap_inside`, a different solver), a cylinder with the
endpoints at different heights (the helix correction), and collinear
endpoints through a sphere (the degenerate-plane fallback).

Run:  pixi run python scripts/dump_mujoco_wrap.py
"""

import numpy as np
import mujoco

MODEL = """
<mujoco model="wrap_probe">
  <compiler angle="radian"/>
  <worldbody>
    <body name="anchor0" pos="0 0 0">
      <joint name="a0x" type="slide" axis="1 0 0" limited="false"/>
      <joint name="a0y" type="slide" axis="0 1 0" limited="false"/>
      <joint name="a0z" type="slide" axis="0 0 1" limited="false"/>
      <geom name="g0" type="sphere" size="0.005" contype="0" conaffinity="0"/>
      <site name="s0" pos="0 0 0" size="0.005"/>
    </body>
    <body name="anchor1" pos="0 0 0">
      <joint name="a1x" type="slide" axis="1 0 0" limited="false"/>
      <joint name="a1y" type="slide" axis="0 1 0" limited="false"/>
      <joint name="a1z" type="slide" axis="0 0 1" limited="false"/>
      <geom name="g1" type="sphere" size="0.005" contype="0" conaffinity="0"/>
      <site name="s1" pos="0 0 0" size="0.005"/>
    </body>
    <body name="obj" pos="{OPOS}" quat="{OQUAT}">
      <geom name="wrapper" type="{OTYPE}" size="{OSIZE}"
            contype="0" conaffinity="0"/>
      <site name="side" pos="{SIDE}" size="0.005"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t">
      <site site="s0"/>
      <geom geom="wrapper" {SIDEATTR}/>
      <site site="s1"/>
    </spatial>
  </tendon>
</mujoco>
"""


def build(otype, osize, opos, oquat, side, use_side):
    xml = (MODEL
           .replace("{OTYPE}", otype)
           .replace("{OSIZE}", " ".join(str(v) for v in osize))
           .replace("{OPOS}", " ".join(str(v) for v in opos))
           .replace("{OQUAT}", " ".join(str(v) for v in oquat))
           .replace("{SIDE}", " ".join(str(v) for v in side))
           .replace("{SIDEATTR}", 'sidesite="side"' if use_side else ""))
    return mujoco.MjModel.from_xml_string(xml)


def sample(m, p0, p1):
    d = mujoco.MjData(m)
    d.qpos[:3] = p0
    d.qpos[3:6] = p1
    mujoco.mj_forward(m, d)

    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wrapper")
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "side")
    adr = m.tendon_adr[0] if hasattr(m, "tendon_adr") else 0
    wadr = d.ten_wrapadr[0]
    wnum = d.ten_wrapnum[0]
    pts = d.wrap_xpos[3 * wadr: 3 * (wadr + wnum)].reshape(-1, 3)

    x0 = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "s0")].copy()
    x1 = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "s1")].copy()

    if wnum == 4:
        w0, w1 = pts[1], pts[2]
        wlen = d.ten_length[0] - np.linalg.norm(x0 - w0) - np.linalg.norm(w1 - x1)
    else:
        w0 = np.zeros(3)
        w1 = np.zeros(3)
        wlen = -1.0

    return dict(
        x0=x0, x1=x1,
        gpos=d.geom_xpos[gid].copy(),
        gmat=d.geom_xmat[gid].copy(),
        radius=float(m.geom_size[gid][0]),
        side=d.site_xpos[sid].copy(),
        wlen=float(wlen), w0=w0, w1=w1,
        wrapnum=int(wnum),
    )


CASES = []


def add(label, otype, osize, opos, oquat, side, use_side, p0, p1):
    m = build(otype, osize, opos, oquat, side, use_side)
    r = sample(m, p0, p1)
    r["label"] = label
    r["wtype"] = 1 if otype == "sphere" else 2
    r["use_side"] = use_side
    CASES.append(r)


I = (1, 0, 0, 0)
Q45 = (0.92387953, 0.0, 0.0, 0.38268343)      # 45 deg about z
QX = (0.70710678, 0.70710678, 0.0, 0.0)       # 90 deg about x

# ── sphere, no sidesite ──────────────────────────────────────────────────
add("sphere clear (no wrap)", "sphere", [0.1], (0, 0, 0), I, (0, 0.2, 0),
    False, (-0.5, 0.4, 0.0), (0.5, 0.4, 0.0))
add("sphere blocked", "sphere", [0.1], (0, 0, 0), I, (0, 0.2, 0),
    False, (-0.5, 0.02, 0.0), (0.5, 0.02, 0.0))
add("sphere blocked, asymmetric", "sphere", [0.1], (0, 0, 0), I, (0, 0.2, 0),
    False, (-0.4, 0.05, 0.03), (0.6, -0.02, -0.05))
add("sphere collinear through centre", "sphere", [0.1], (0, 0, 0), I,
    (0, 0.2, 0), False, (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0))
add("sphere offset centre", "sphere", [0.08], (0.1, 0.05, -0.02), Q45,
    (0, 0.2, 0), False, (-0.5, 0.05, 0.0), (0.5, 0.05, 0.0))

# ── sphere, sidesite outside ─────────────────────────────────────────────
add("sphere sidesite +y", "sphere", [0.1], (0, 0, 0), I, (0, 0.3, 0),
    True, (-0.5, 0.02, 0.0), (0.5, 0.02, 0.0))
add("sphere sidesite -y (the long way)", "sphere", [0.1], (0, 0, 0), I,
    (0, -0.3, 0), True, (-0.5, 0.02, 0.0), (0.5, 0.02, 0.0))
add("sphere sidesite, clear segment", "sphere", [0.1], (0, 0, 0), I,
    (0, -0.3, 0), True, (-0.5, 0.4, 0.0), (0.5, 0.4, 0.0))

# ── sphere, sidesite INSIDE (wrap_inside) ────────────────────────────────
add("sphere sidesite inside", "sphere", [0.2], (0, 0, 0), I, (0, 0.05, 0),
    True, (-0.5, 0.3, 0.0), (0.5, 0.3, 0.0))
add("sphere sidesite inside, asymmetric", "sphere", [0.2], (0, 0, 0), I,
    (0.02, 0.05, 0.01), True, (-0.6, 0.35, 0.05), (0.45, 0.28, -0.04))

# ── cylinder (axis = local z) ────────────────────────────────────────────
add("cylinder clear", "cylinder", [0.1, 0.3], (0, 0, 0), I, (0, 0.2, 0),
    False, (-0.5, 0.4, 0.0), (0.5, 0.4, 0.0))
add("cylinder blocked, flat", "cylinder", [0.1, 0.3], (0, 0, 0), I,
    (0, 0.2, 0), False, (-0.5, 0.02, 0.0), (0.5, 0.02, 0.0))
add("cylinder blocked, helix", "cylinder", [0.1, 0.3], (0, 0, 0), I,
    (0, 0.2, 0), False, (-0.5, 0.02, -0.2), (0.5, 0.02, 0.25))
add("cylinder rotated 90 about x", "cylinder", [0.1, 0.3], (0, 0, 0), QX,
    (0, 0, 0.2), False, (-0.5, 0.0, 0.02), (0.5, 0.0, 0.02))
add("cylinder sidesite +y", "cylinder", [0.1, 0.3], (0, 0, 0), I,
    (0, 0.3, 0), True, (-0.5, 0.02, 0.05), (0.5, 0.02, -0.05))
add("cylinder sidesite -y (long way)", "cylinder", [0.1, 0.3], (0, 0, 0), I,
    (0, -0.3, 0), True, (-0.5, 0.02, 0.05), (0.5, 0.02, -0.05))
add("cylinder sidesite inside", "cylinder", [0.25, 0.3], (0, 0, 0), I,
    (0, 0.05, 0), True, (-0.6, 0.4, 0.0), (0.6, 0.4, 0.0))
add("cylinder offset + rotated", "cylinder", [0.07, 0.3], (0.05, -0.03, 0.01),
    Q45, (0, 0.2, 0), False, (-0.5, -0.02, 0.04), (0.5, 0.01, -0.03))


def fmt(v):
    return repr(float(v))


ARM_XML = "tests/physics3d/assets/wrap_arm.xml"
SOFTFOOT = "references/mujoco_menagerie-main/iit_softfoot/scene.xml"


def arm_rows():
    """The hinge sweep: 12 angles x 2 tendons, crossing BOTH transitions.

    A fixed pose cannot tell a wrap that never engages from one that always
    does; the sweep is what makes `mju_wrap`'s early returns and the routing
    loop's `j += 2` observable in the same test.
    """
    import mujoco as mj
    m = mj.MjModel.from_xml_path(ARM_XML)
    d = mj.MjData(m)
    rows = []
    for k in range(12):
        a = -1.5 + k * 0.3
        mj.mj_resetData(m, d)
        d.qpos[0] = a
        mj.mj_forward(m, d)
        rows.append([a,
                     float(d.ten_length[0]), float(d.ten_wrapnum[0]),
                     float(d.ten_length[1]), float(d.ten_wrapnum[1])])
    return rows


def softfoot_rows():
    """iit_softfoot: 39-waypoint tendons, 18 wrap cylinders each.

    ⚠ THE OFFSET IS UNIFORM ON PURPOSE. An index-dependent perturbation would
    make the goldens depend on our joint ORDER matching MuJoCo's, and this
    test is about tendon routing, not about joint numbering — a mismatch
    there would show up here as a wrap bug.
    """
    import mujoco as mj
    import os
    cwd = os.getcwd()
    os.chdir(os.path.dirname(SOFTFOOT))
    try:
        m = mj.MjModel.from_xml_path("scene.xml")
        d = mj.MjData(m)
        rows = []
        for amp in [0.0, 0.2, 0.5]:
            mj.mj_resetData(m, d)
            d.qpos[:] = m.qpos0 + amp
            mj.mj_forward(m, d)
            rows.append([amp] + [float(v) for v in d.ten_length])
        return rows
    finally:
        os.chdir(cwd)


def main():
    rows = []
    for c in CASES:
        vals = []
        vals += list(c["x0"]) + list(c["x1"]) + list(c["gpos"])
        vals += list(c["gmat"])
        vals += [c["radius"], float(c["wtype"]), 1.0 if c["use_side"] else 0.0]
        vals += list(c["side"])
        vals += [c["wlen"]] + list(c["w0"]) + list(c["w1"])
        assert len(vals) == 31, len(vals)
        rows.append((c["label"], c["wrapnum"], vals))

    out = []
    out.append('"""MuJoCo 3.10.0 goldens for `mju_wrap` — GENERATED, do not edit.')
    out.append("")
    out.append("Regenerate: pixi run python scripts/dump_mujoco_wrap.py")
    out.append("")
    out.append("`mju_wrap` is not exposed to the Python bindings, so each row is read")
    out.append("off a real one-tendon model: the wrap POINTS come straight from")
    out.append("`d.wrap_xpos`, and the arc length is `ten_length` minus the two")
    out.append("straight runs. See scripts/dump_mujoco_wrap.py for the poses and why")
    out.append("each one is there.")
    out.append("")
    out.append("Columns (31 per case):")
    out.append("   0- 2  x0            6- 8  geom pos      18     radius")
    out.append("   3- 5  x1            9-17  geom mat      19     wtype 1=sph 2=cyl")
    out.append("  20     has_side     21-23  sidesite      24     wlen (-1 = no wrap)")
    out.append("  25-27  wrap point 0 28-30  wrap point 1")
    out.append('"""')
    out.append("")
    out.append("comptime WRAP_COLS: Int = 31")
    out.append("")
    out.append("")
    out.append("def wrap_case_labels() -> List[String]:")
    out.append("    var v = List[String]()")
    for label, wn, _ in rows:
        out.append('    v.append(String("%s [wrapnum %d]"))' % (label, wn))
    out.append("    return v^")
    out.append("")
    out.append("")
    out.append("def wrap_goldens() -> List[Float64]:")
    out.append("    var v = List[Float64]()")
    for label, _, vals in rows:
        out.append("    # %s" % label)
        for i in range(0, 31, 4):
            chunk = vals[i:i + 4]
            out.append("    " + "".join("v.append(%s); " % fmt(x) for x in chunk).rstrip())
    out.append("    return v^")
    out.append("")

    arm = arm_rows()
    sf = softfoot_rows()
    out.append("")
    out.append("comptime ARM_COLS: Int = 5   # angle, len_cyl, wrapnum_cyl, len_sph, wrapnum_sph")
    out.append("comptime SOFTFOOT_COLS: Int = 6  # uniform qpos offset, then 5 tendon lengths")
    out.append("")
    out.append("")
    out.append("def arm_sweep_goldens() -> List[Float64]:")
    out.append('    """`tests/physics3d/assets/wrap_arm.xml`, hinge swept -1.5 .. +1.8 rad."""')
    out.append("    var v = List[Float64]()")
    for r in arm:
        out.append("    " + "".join("v.append(%s); " % fmt(x) for x in r).rstrip())
    out.append("    return v^")
    out.append("")
    out.append("")
    out.append("def softfoot_goldens() -> List[Float64]:")
    out.append('    """iit_softfoot at qpos0 + a uniform offset, 5 tendons each."""')
    out.append("    var v = List[Float64]()")
    for r in sf:
        out.append("    " + "".join("v.append(%s); " % fmt(x) for x in r).rstrip())
    out.append("    return v^")
    out.append("")

    path = "tests/physics3d/wrap_goldens.mojo"
    with open(path, "w") as f:
        f.write("\n".join(out))
    wrapped = sum(1 for _, wn, _ in rows if wn == 4)
    arm_wrapped = sum(1 for r in arm for wn in (r[2], r[4]) if wn == 4)
    print("wrote %s: %d mju_wrap cases (%d wrap, %d straight); "
          "arm sweep %d poses (%d/%d tendon-poses wrap); softfoot %d poses"
          % (path, len(rows), wrapped, len(rows) - wrapped,
             len(arm), arm_wrapped, 2 * len(arm), len(sf)))


if __name__ == "__main__":
    main()
