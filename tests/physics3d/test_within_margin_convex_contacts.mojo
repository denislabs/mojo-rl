"""Convex pairs SEPARATED BY LESS THAN `margin` are still contacts.

`gjk_epa` used to return `1e30` whenever `_gjk_intersect` proved the pair was
outside the Minkowski difference — "no collision, so report a separation the
caller's `dist < margin` test will reject". That is right only when `margin` is
0. With a margin, a pair separated by less than it IS a contact, MuJoCo reports
one, and we returned nothing.

⚠⚠ IT FIRED EXACTLY WHERE IT MATTERED AND NOWHERE ELSE. The separation
certificate needs a 4-simplex, which only forms at SMALL separations, so
distant pairs never reached the branch — the failure band was the millimetre
either side of touching. Cylinder over a box, `margin=0.01` on each
(`includemargin` 0.02), before the fix:

    gap        ours   MuJoCo
    -0.0005      5      5
     0.0         0      5     <- all contacts lost
     0.0005      0      5     <- lost
     0.001       0      5     <- lost
     0.002       1      5     <- 4 of 5 manifold rows lost
     0.012       1      5
     0.02        0      0

The 1-instead-of-5 rows are the same cause: `mjc_Convex`'s perturbation loop
runs `gjk_epa` several more times, and each perturbed query hit the same early
return.

⚠ MUJOCO NEVER NEEDS A SEPARATION DISTANCE, which is why the port could look
finished without this. `mjc_penetration` sets `dist_cutoff = 0` and INFLATES
both geoms by `margin` (`mjc_initCCDObj(&obj, m, d, g, margin)`), so a
within-margin pair reads as penetrating and `con->dist = margin + dist`
un-inflates it. We do not inflate — our caller compares a real distance against
`margin` — so our GJK MUST produce one where MuJoCo's need not. Copying the
reference's control flow without copying its margin inflation is what left the
hole.

⚠ NON-ZERO MARGINS ARE REAL IN THIS TREE: sawyer's meshes carry
`margin="0.001"` and go through the convex path; hopper, humanoid and
humanoid_standup use `margin="0.001"` on primitives.

Run with:
    pixi run mojo run -I . tests/physics3d/test_within_margin_convex_contacts.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.fields import Model, Data, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE, CONTACT_IDX_DIST, META_IDX_NUM_CONTACTS,
)
from max.gpu.host import DeviceContext
from mojo_rl.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float64
comptime NMV: Int = 64

# Cylinder over a box: a SMOOTH pair, so it takes the `mjc_Convex` GJK/EPA
# route rather than a primitive routine. `margin='0.01'` on each gives
# `includemargin` 0.02, wide enough to sweep the whole band in one model.
comptime XML = String(
    """<mujoco><option gravity='0 0 0'/>
  <worldbody>
    <geom name='g_box' type='box' size='0.05 0.04 0.03' margin='0.01'/>
    <body name='b1' pos='0 0 0'>
      <freejoint/>
      <geom name='g_cyl' type='cylinder' size='0.03 0.02' margin='0.01' mass='1'/>
    </body>
  </worldbody>
</mujoco>"""
)

comptime PM = parse_xml(XML)
comptime MD = ModelDefFromXML[
    xml=XML, nbody=PM.NBODY, njoint=PM.NJOINT, nq=PM.NQ, nv=PM.NV,
    ngeom=PM.NGEOM, nact=PM.NACT, ntex=PM.NTEX, nmat=PM.NMAT,
    nlight=PM.NLIGHT, ncam=PM.NCAM, nsite=PM.NSITE, neq=PM.NEQ,
    nexclude=PM.NEXCLUDE, npair=PM.NPAIR, max_tendon=PM.NTENDON,
    max_condim=PM.MAX_CONDIM, max_equality=1, max_contacts=16,
    timestep=PM.TIMESTEP,
]
comptime MD_2 = ModelDims[MD, 64]

# Distances agree with MuJoCo to ~2.8e-8 across the band; MuJoCo's own values
# carry that much because its support functions are margin-inflated and
# un-inflated again.
comptime TOL_DIST: Float64 = 1e-6


def test_contacts_survive_the_whole_margin_band() raises:
    """Counts and distances across the gap, from penetrating to beyond margin.

    ⚠ THE PENETRATING AND BEYOND-MARGIN ENDS ARE PART OF THE GATE, not padding.
    "Emit a contact whenever anything is nearby" would pass a band-only check;
    the far end is what says we still stop, and the near end is what says the
    fix did not disturb the penetrating path that `_gjk_intersect`'s
    certificate exists to protect.
    """
    var sf = MD.make_spec_fields[DTYPE]()
    print("=== convex contacts across the margin band ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")

    var m = mujoco.MjModel.from_xml_string(XML)
    var md = mujoco.MjData(m)

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_2]()
    MD.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD_2, 1]()

    # Box top face z = 0.03; cylinder half-length 0.02 on a +z axis, so its
    # bottom face is at qz - 0.02 and the gap is qz - 0.05.
    var gaps = [
        -0.002, -0.0005, 0.0, 0.0002, 0.0005, 0.001, 0.002, 0.004,
        0.006, 0.008, 0.0095, 0.012, 0.02, 0.05, 0.2,
    ]
    var n_band = 0
    var n_far = 0
    var worst_dist = Float64(0)
    var mismatches = 0

    for gi in range(len(gaps)):
        var gap = gaps[gi]
        var qz = 0.05 + gap
        MD.reset_data[DTYPE](sf, d)
        for i in range(MD.NQ):
            md.qpos[i] = 0.0
            d.qpos.data[i] = 0
        md.qpos[2] = qz
        d.qpos.data[2] = Scalar[DTYPE](qz)
        md.qpos[3] = 1.0
        d.qpos.data[3] = Scalar[DTYPE](1.0)
        for i in range(MD.NV):
            md.qvel[i] = 0.0
            d.qvel.data[i] = 0
        mujoco.mj_forward(m, md)
        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)

        var our_n = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
        var mj_n = Int(py=md.ncon)
        var e = Float64(0)
        if our_n > 0 and mj_n > 0:
            e = abs(
                Float64(d.contacts.data[CONTACT_IDX_DIST])
                - Float64(py=md.contact[0].dist)
            )
            if e > worst_dist:
                worst_dist = e
        if our_n != mj_n:
            mismatches += 1
        if mj_n > 0:
            n_band += 1
        else:
            n_far += 1
        print("   gap", gap, " ours", our_n, " MuJoCo", mj_n,
              "  |d(dist)|", e)

    print("  poses with a contact:", n_band, "  beyond margin:", n_far)
    print("  count mismatches:", mismatches, "  worst |d(dist)|", worst_dist)

    # Vacuity: the sweep has to actually cross the boundary in both directions.
    assert_true(
        n_band >= 8 and n_far >= 2,
        "the sweep did not straddle the margin boundary (" + String(n_band)
        + " contacting, " + String(n_far) + " beyond), so it gates nothing",
    )
    assert_true(
        mismatches == 0,
        String(mismatches) + " of " + String(len(gaps)) + " gaps disagree with"
        " MuJoCo on the contact COUNT. ⚠ the historical failure was ZERO"
        " contacts for gaps in [0, 0.001] and 1-instead-of-5 above that, so"
        " check WHICH gaps before assuming this is a manifold-count nuance",
    )
    assert_true(
        worst_dist <= TOL_DIST,
        "contact distance diverges from MuJoCo by " + String(worst_dist)
        + " across the margin band",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
