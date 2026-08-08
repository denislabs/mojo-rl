"""Walker2D contact detection against MuJoCo, at the golden gate's own poses.

WHY THIS EXISTS. `test_contact_detection_fields.mojo` freezes an
ORDER-SENSITIVE checksum of the contact records, originally validated against
the legacy kernels — which have since been deleted. A golden with no live
reference can tell you the records CHANGED; it cannot tell you whether they
changed for a good reason. When the element-order fix (2026-08-03) made
`full_parser` group geoms by body like MuJoCo, geom indices moved, the emission
order of the 10 contacts moved with them, and that checksum went from
3390.603547532484 to 9501.98150853524 with `ncon` unchanged at 10.

"The count is the same and the order changed" is a hypothesis. This file is the
measurement: the same two poses, compared against MuJoCo contact by contact on
the quantities that are physics rather than bookkeeping — body pair, penetration
depth, contact point and normal.

⚠ MATCHED BY POSITION, DELIBERATELY. Sorting both sides first would hide the
very thing in question. If our order is MuJoCo's, an elementwise comparison
passes; if it is merely a permutation of the right SET, this fails and says so,
which is the outcome that would mean the golden should NOT simply be refreshed.

⚠ float32. This model runs `DType.float32` (as the golden gate does), so the
budget is single-precision FK round-off on positions of order 1, not the 1e-15
the float64 dm_control gates live at.

Run with:
    pixi run mojo run -I . tests/physics3d/test_walker2d_contacts_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    META_IDX_NUM_CONTACTS,
    METADATA_SIZE,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel, walker2d_xml

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.NEXCLUDE

# float32 FK on positions of order 1. Measured, then budgeted an order above.
comptime POS_TOL: Float64 = 1e-5
comptime DIST_TOL: Float64 = 1e-5


def _poses() -> List[List[Float64]]:
    """The golden gate's own two poses, so this measures the same thing it
    does: env0 a slight floor penetration, env1 heavy penetration with bent
    legs."""
    var out = List[List[Float64]]()
    var q0 = List[Float64](length=NQ, fill=0.0)
    q0[1] = 1.18
    out.append(q0^)
    var q1 = List[Float64](length=NQ, fill=0.0)
    q1[1] = 0.85
    q1[3] = 0.6
    q1[4] = -1.1
    q1[6] = -0.4
    q1[7] = -0.9
    out.append(q1^)
    return out^


def test_walker2d_contacts_match_mujoco() raises:
    print("--- walker2d contact detection vs MuJoCo ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[walker2d_xml]())
    var md = mujoco.MjData(m)

    var ctx = DeviceContext()
    var mf = Model[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0]()
    Walker2dModel.init_fields[DTYPE, 0](ctx, mf)

    var poses = _poses()
    var total_ours = 0
    var total_mj = 0
    var worst_dist = 0.0
    var worst_pos = 0.0
    var worst_normal = 0.0
    var pair_mismatch = 0

    for e in range(len(poses)):
        var d = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, 1]()
        Walker2dModel.reset_data[DTYPE](d)
        for i in range(NQ):
            d.qpos.data[i] = Scalar[DTYPE](poses[e][i])
        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)

        mujoco.mj_resetData(m, md)
        for i in range(NQ):
            md.qpos[i] = poses[e][i]
        mujoco.mj_forward(m, md)

        var nc = Int(d.meta.data[META_IDX_NUM_CONTACTS])
        var nmj = Int(py=md.ncon)
        total_ours += nc
        total_mj += nmj
        print("  pose", e, ": ours ncon", nc, " MuJoCo ncon", nmj)
        assert_true(
            nc == nmj,
            String("contact COUNT differs at pose ") + String(e)
            + " — this is not an ordering question at all",
        )

        for c in range(nc):
            var o = c * CONTACT_SIZE
            var ba = Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_A]))
            var bb = Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_B]))
            var mc_ = md.contact[c]
            var mb1 = Int(py=m.geom_bodyid[Int(py=mc_.geom1)])
            var mb2 = Int(py=m.geom_bodyid[Int(py=mc_.geom2)])
            # Our convention orders the pair by which geom was `gi`; compare as
            # an unordered pair, since the ORDER within a contact is a separate
            # (already-gated) convention from the order OF contacts.
            var same_pair = (ba == mb1 and bb == mb2) or (
                ba == mb2 and bb == mb1
            )
            if not same_pair:
                pair_mismatch += 1
                print("    contact", c, "body pair ours", ba, bb,
                      " MuJoCo", mb1, mb2)

            var dd = abs(
                Float64(d.contacts.data[o + CONTACT_IDX_DIST])
                - Float64(py=mc_.dist)
            )
            if dd > worst_dist:
                worst_dist = dd
            var dp = abs(
                Float64(d.contacts.data[o + CONTACT_IDX_POS_X])
                - Float64(py=mc_.pos[0])
            )
            var dpz = abs(
                Float64(d.contacts.data[o + CONTACT_IDX_POS_Z])
                - Float64(py=mc_.pos[2])
            )
            if dp > worst_pos:
                worst_pos = dp
            if dpz > worst_pos:
                worst_pos = dpz
            # The normal may point either way depending on which geom is `gi`;
            # compare the AXIS, not the arrow.
            var dn = abs(
                abs(Float64(d.contacts.data[o + CONTACT_IDX_NZ]))
                - abs(Float64(py=mc_.frame[2]))
            )
            if dn > worst_normal:
                worst_normal = dn

    print("  total contacts: ours", total_ours, " MuJoCo", total_mj)
    print("  worst |d(dist)| =", worst_dist,
          "  |d(pos)| =", worst_pos, "  |d(normal_z)| =", worst_normal)
    print("  body-pair mismatches (position-matched):", pair_mismatch)

    # NON-VACUITY: the golden gate raises when these poses produce no contacts,
    # and so does this — a pose pair that floated free would post perfect zeros.
    assert_true(
        total_mj >= 8,
        "MuJoCo finds almost no contacts at these poses — the fixture is not"
        " the penetrating pair the golden gate believes it is",
    )
    assert_true(
        pair_mismatch == 0,
        "our contacts are not in MuJoCo's ORDER (the sets may still agree) —"
        " if this is the only failure, the frozen checksum in"
        " test_contact_detection_fields.mojo must NOT simply be refreshed",
    )
    assert_true(worst_dist <= DIST_TOL, "penetration depth differs from MuJoCo")
    assert_true(worst_pos <= POS_TOL, "contact point differs from MuJoCo")
    assert_true(
        worst_normal <= POS_TOL, "contact normal differs from MuJoCo"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
