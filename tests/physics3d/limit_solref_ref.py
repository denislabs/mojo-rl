"""A model whose limit solver params are NOT uniform across joints.

Defect 22: `fields_build` broadcast JOINT 0's `solreflimit`/`solimplimit` to
model meta and `constraints/limits.mojo` read them from there, so every limit
row in the model used joint 0's parameters. On dog that meant reading the
model defaults off the FREE ROOT — the one joint that can never own a limit
row — and every limit came out 3.68x too soft.

`test_humanoid_limits_fields_vs_mujoco` cannot catch this: humanoid's joints
all carry the same limit params, so joint 0's values ARE the right ones and the
broadcast is invisible. A gate for this needs a model where joint 0 DIFFERS
from a limited joint, which is what this is:

    j0  hinge, UNLIMITED          -> keeps the global default solreflimit
                                     [0.02 1], solimplimit [0.9 0.95 0.001 ...]
    j1  hinge, LIMITED, driven past its range
        solreflimit  "0.04 1"        4x SOFTER than the default
        solimplimit  "0.9 0.99 0.01"

⚠ j0 IS UNLIMITED ON PURPOSE. It mirrors dog's free root: the joint whose
params were broadcast is one that never forms a row, so a fix that merely made
the broadcast "use the first LIMITED joint" would still be wrong in general and
must still fail here.

⚠ AND j1 IS NOT JOINT 0. If the only limited joint were joint 0 the broadcast
would accidentally be correct and this file would pass either way — the same
way humanoid does.

K = 1/(dmax^2 * timeconst^2 * dampratio^2):
    joint 0's params  ->  1/(0.95^2 * 0.02^2)  =  2770.08
    j1's own params   ->  1/(0.99^2 * 0.04^2)  =   637.69      4.3x apart

⚠ 0.04 IS CHOSEN TO STAY ABOVE 2*timestep, AND THAT IS NOT COSMETIC. MuJoCo
applies `solref[0] = max(solref[0], 2*timestep)` for the standard format
(engine_core_constraint.c:2028, "integrator safety", active unless
mjDSBL_REFSAFE). This engine does NOT implement that clamp — a separate defect —
so a fixture below 2*dt (0.01 here) would fail for TWO reasons at once and could
not verify either fix. Measured: solreflimit 0.005 makes MuJoCo report
K = 10203.04, exactly 1/(0.99^2 * 0.01^2), i.e. the clamped value.
"""

XML = """<mujoco model="limit_solref">
  <compiler angle="degree"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b0" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      <body name="b1" pos="0.2 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" limited="true"
               range="-30 30" solreflimit="0.04 1"
               solimplimit="0.9 0.99 0.01"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# j1 driven 15 degrees past its +30 limit, with a velocity that keeps pushing
# into it so the damping term is live too (a pure position violation would
# gate K and leave B untested).
QPOS = (0.3, 0.785398163397448)      # j0 free-ish, j1 = 45 deg = 0.7854 rad
QVEL = (0.0, 1.5)


def model():
    import mujoco
    return mujoco.MjModel.from_xml_string(XML)


if __name__ == "__main__":
    import mujoco, numpy as np
    m = model()
    d = mujoco.MjData(m)
    print("njnt", m.njnt, " nv", m.nv)
    for j in range(m.njnt):
        print(f"  jnt{j} {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)}"
              f" limited={int(m.jnt_limited[j])}"
              f" solref={np.array(m.jnt_solref[j])}"
              f" solimp={np.array(m.jnt_solimp[j])}")
    d.qpos[:] = QPOS
    d.qvel[:] = QVEL
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    mujoco.mj_forward(m, d)
    print("nefc", d.nefc)
    for i in range(d.nefc):
        print(f"  efc[{i}] type={int(d.efc_type[i])} id={int(d.efc_id[i])}"
              f" pos={d.efc_pos[i]:.6g} KBIP={np.array(d.efc_KBIP[i])}")
    print("qacc", np.array(d.qacc))
