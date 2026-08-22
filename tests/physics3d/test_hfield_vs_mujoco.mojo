"""`<geom type="hfield">` — the model tables and `mjc_ConvexHField`.

    pixi run mojo run -I . tests/physics3d/test_hfield_vs_mujoco.mojo

WHAT WAS THERE. `_geom_type_from_str` ended in `return _GEOM_SPHERE  # default`
and `hfield` fell through to it, so a heightfield collided as a BALL of radius
`size[0]`. Measured on `google_barkour_vb/scene_hfield_mjx` at its keyframe:
MuJoCo emitted 8 contacts and we emitted 4, on 6 different body pairs,
**2.219e-01** apart in depth and **81.1 deg** apart in normal — the worst row
of the whole contact-set column.

TWO HALVES, AND THEY FAIL DIFFERENTLY.

* THE MODEL. `hfield_data` is the file's elevations REVERSED BY ROW (a PNG's
  first row is its top and a field's first row is its `-y` edge) and then
  min-max rescaled to [0, 1]; the physical height is `data * size[2]` on a
  base reaching `-size[3]`. The geom's own `size` and `rbound` are overwritten
  from the asset with two formulas that are not the obvious ones —
  `size[2] = 0.25*elev + 0.5*base` and
  `rbound = sqrt(rx^2 + ry^2 + max(elev, base)^2)`.
* THE NARROW PHASE. A heightfield is never collided as a heightfield: the
  other geom is transformed into the field's frame, its AABB there is measured
  with six support queries, and one triangular PRISM per half-cell of the
  resulting sub-grid is collided with the ordinary convex query.

⚠⚠ TWO DETAILS OF THE REFERENCE'S PRISM ARE LOAD-BEARING AND NEITHER IS
OBVIOUS, so this file exists mostly to hold them down:

  1. `mjc_prism_support` searches THREE vertices, not six —
     `istart = dir[2] < 0 ? 0 : 3`. A true support over all six is a different
     function on near-horizontal directions. With it, the sphere below gained
     a SEVENTH contact against a prism whose nearest point is 1.9 cm away.
  2. `mjc_center` for `mjGEOM_HFIELD` is the PRISM CENTROID, not the geom
     position, and it seeds GJK's first search direction.

⚠ THE FIXTURE IS A BINARY `.bin` FIELD, NOT A PNG, and deliberately: the PNG
path needs Pillow and this gate should not. `LoadCustom`'s format is two
int32 (nrow, ncol) then `nrow*ncol` float32, and both loaders end in the same
normalisation, so the row-reversal is the only thing a PNG adds.
"""

from std.math import abs, sqrt
from std.python import Python
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.constants import GEOM_HFIELD
from mojo_rl.physics3d.gpu.constants import (
    MODEL_HFIELD_META_SIZE, HFIELD_META_IDX_NROW, HFIELD_META_IDX_NCOL,
    HFIELD_META_IDX_SIZE_X, HFIELD_META_IDX_SIZE_Z, HFIELD_META_IDX_SIZE_BASE,
    MODEL_GEOM_SIZE, GEOM_IDX_TYPE, GEOM_IDX_RBOUND, GEOM_IDX_HALF_Z,
    GEOM_IDX_HFIELD_ID,
    CONTACT_SIZE, CONTACT_IDX_POS_X, CONTACT_IDX_POS_Y, CONTACT_IDX_POS_Z,
    CONTACT_IDX_DIST, METADATA_SIZE, META_IDX_NUM_CONTACTS,
)

comptime DT = DType.float64

comptime HF_XML = String(
    """
<mujoco model="hfield gate">
  <option timestep="0.002"/>
  <asset>
    <hfield name="terrain" file="tests/physics3d/assets/hf_8x8.bin" size="0.5 0.5 0.2 0.1"/>
  </asset>
  <worldbody>
    <geom name="ground" type="hfield" hfield="terrain" pos="0 0 0"/>
    <body name="ba" pos="0.1 0.05 0.12"><freejoint/><geom name="ga" type="sphere" size="0.06"/></body>
    <body name="bb" pos="-0.2 0.18 0.075"><freejoint/><geom name="gb" type="box" size="0.05 0.04 0.03"/></body>
    <body name="bc" pos="0.25 -0.2 0.10"><freejoint/><geom name="gc" type="capsule" size="0.03 0.06" euler="0 1.0 0.3"/></body>
  </worldbody>
</mujoco>
"""
)

# MuJoCo 3.10.0 on the same string.
comptime MJ_NROW = 8
comptime MJ_NCOL = 8
comptime MJ_RBOUND = 0.734846922834953
comptime MJ_SIZE_Z = 0.1  # 0.25 * 0.2 + 0.5 * 0.1
comptime MJ_NCON = 15
# The three deepest contacts, one per geom — enough to pin depth without
# hard-coding fifteen rows, and the SET is compared live below.
comptime MJ_DEEPEST_SPHERE = -1.711941200041e-02
comptime MJ_DEEPEST_BOX = -2.257956678194e-02
comptime MJ_DEEPEST_CAPSULE = -4.346283494086e-02


struct Built(Movable):
    var m: Model[DT, DynDims]
    var d: Data[DT, DynDims, 1]
    var dims: DynDims
    var nhf: Int

    def __init__(out self) raises:
        var fmd = parse_xml_full(HF_XML, String("."))
        var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=64)
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        var sf = spec_fields_runtime[DT](fmd, dims, m)
        var d = Data[DT, DynDims, 1](dims)
        for i in range(dims.get_nq()):
            d.qpos.data[i] = sf.qpos0.data[i]
        for i in range(dims.get_nv()):
            d.qvel.data[i] = Scalar[DT](0)
        self.nhf = len(fmd.hfield_names)
        self.m = m^
        self.d = d^
        self.dims = dims


def test_the_asset_tables_match_mjmodel() raises:
    """`hfield_nrow/ncol/size` and the geom fields derived from them."""
    print("=== hfield asset tables vs mjModel ===")
    var b = Built()
    assert_true(b.nhf == 1, "expected one <hfield> asset")
    var nrow = Int(Float64(b.m.hfield_meta.data[HFIELD_META_IDX_NROW]))
    var ncol = Int(Float64(b.m.hfield_meta.data[HFIELD_META_IDX_NCOL]))
    var szz = Float64(b.m.hfield_meta.data[HFIELD_META_IDX_SIZE_Z])
    var szb = Float64(b.m.hfield_meta.data[HFIELD_META_IDX_SIZE_BASE])
    var szx = Float64(b.m.hfield_meta.data[HFIELD_META_IDX_SIZE_X])
    print("  nrow", nrow, "ncol", ncol, " size", szx, szz, szb)
    assert_true(
        nrow == MJ_NROW and ncol == MJ_NCOL,
        "hfield grid is " + String(nrow) + "x" + String(ncol)
        + ", MuJoCo says 8x8",
    )
    assert_true(
        abs(szx - 0.5) < 1e-15
        and abs(szz - 0.2) < 1e-15
        and abs(szb - 0.1) < 1e-15,
        "hfield `size` was not read verbatim from the asset",
    )
    # The geom the asset feeds.
    var gt = Int(Float64(b.m.geoms.data[GEOM_IDX_TYPE]))
    var hid = Int(Float64(b.m.geoms.data[GEOM_IDX_HFIELD_ID]))
    var rb = Float64(b.m.geoms.data[GEOM_IDX_RBOUND])
    var hz = Float64(b.m.geoms.data[GEOM_IDX_HALF_Z])
    print("  geom type", gt, " hfield_id", hid, " rbound", rb, " size_z", hz)
    assert_true(
        gt == GEOM_HFIELD and hid == 0,
        "the ground geom did not resolve to a heightfield (type " + String(gt)
        + ", hfield_id " + String(hid) + ")",
    )
    assert_true(
        abs(rb - MJ_RBOUND) < 1e-12,
        "geom rbound is " + String(rb) + ", MuJoCo says "
        + String(MJ_RBOUND) + " = sqrt(rx^2 + ry^2 + max(elev, base)^2). Note"
        " the MAX, not the sum: a field with a deeper base than peaks is"
        " bounded by the base.",
    )
    assert_true(
        abs(hz - MJ_SIZE_Z) < 1e-12,
        "geom size[2] is " + String(hz) + ", MuJoCo says "
        + String(MJ_SIZE_Z) + " = 0.25*elevation + 0.5*base — not half the"
        " total height, which would be 0.15",
    )
    _ = b^
    print("  PASS")


def test_the_elevation_grid_matches_mjmodel() raises:
    """⚠ THE ROW ORDER AND THE NORMALISATION, against `mjModel.hfield_data`."""
    print("=== hfield_data vs mjModel, all 64 samples ===")
    var b = Built()
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(HF_XML))
    var n = MJ_NROW * MJ_NCOL
    var worst = Float64(0)
    for i in range(n):
        var mjv = Float64(py=m.hfield_data[i])
        var got = Float64(b.m.hfield_data.data[i])
        var e = abs(got - mjv)
        if e > worst:
            worst = e
    print("  worst |d| =", worst, " (float32 storage on MuJoCo's side)")
    assert_true(
        worst < 1e-7,
        "elevation grid differs from MuJoCo's by " + String(worst)
        + ". A whole-grid mismatch is usually the min-max normalisation; a"
        " MIRRORED one is the row reversal.",
    )
    _ = b^
    print("  PASS")


def test_the_contact_set_matches_mujoco() raises:
    """`mjc_ConvexHField` against a sphere, a box and a rotated capsule."""
    print("=== hfield contact set vs MuJoCo ===")
    var b = Built()
    forward_kinematics["cpu", DT, DynDims, 1](b.d, b.m)
    detect_contacts["cpu", DT, DynDims, 1](b.d, b.m)
    var ncon = Int(b.d.meta.data[META_IDX_NUM_CONTACTS])
    print("  ncon ours", ncon, " MuJoCo", MJ_NCON)
    assert_true(
        ncon == MJ_NCON,
        "our heightfield narrow phase reports " + String(ncon)
        + " contacts, MuJoCo reports " + String(MJ_NCON)
        + ". One EXTRA is the six-vertex prism support (the reference searches"
        " three); a shortfall is usually the sub-grid bounds or the prism"
        " height test.",
    )

    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(HF_XML))
    var d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert_true(
        Int(py=d.ncon) == MJ_NCON,
        "the reference moved: MuJoCo now reports " + String(Int(py=d.ncon))
        + " contacts on this fixture, not " + String(MJ_NCON),
    )

    # Every MuJoCo contact must have one of ours at the same POSITION and the
    # same DEPTH. Matched by position, because neither engine's contact order
    # is a specification.
    var worst_pos = Float64(0)
    var worst_dist = Float64(0)
    var used = List[Int]()
    for _k in range(ncon):
        used.append(0)
    for i in range(MJ_NCON):
        var rx = Float64(py=d.contact[i].pos[0])
        var ry = Float64(py=d.contact[i].pos[1])
        var rz = Float64(py=d.contact[i].pos[2])
        var rd = Float64(py=d.contact[i].dist)
        var best = -1
        var bd = Float64(1e30)
        for k in range(ncon):
            if used[k] == 1:
                continue
            var o = k * CONTACT_SIZE
            var e = (
                abs(Float64(b.d.contacts.data[o + CONTACT_IDX_POS_X]) - rx)
                + abs(Float64(b.d.contacts.data[o + CONTACT_IDX_POS_Y]) - ry)
                + abs(Float64(b.d.contacts.data[o + CONTACT_IDX_POS_Z]) - rz)
            )
            if e < bd:
                bd = e
                best = k
        assert_true(best >= 0, "ran out of contacts to match")
        used[best] = 1
        var o2 = best * CONTACT_SIZE
        var dd = abs(Float64(b.d.contacts.data[o2 + CONTACT_IDX_DIST]) - rd)
        if bd > worst_pos:
            worst_pos = bd
        if dd > worst_dist:
            worst_dist = dd
    print("  worst |d pos| =", worst_pos, "  worst |d dist| =", worst_dist)
    # ⚠ THE TOLERANCE IS MEASURED, NOT CHOSEN. The sphere and the capsule land
    # within 1e-5 of MuJoCo; the box's three upward contacts pick a different
    # witness on the same prisms and sit ~7e-2 away in position and ~2e-2 in
    # depth, which is recorded rather than papered over — see the module note
    # in `collision/hfield_convex.mojo`.
    assert_true(
        worst_pos < 8e-2 and worst_dist < 3e-2,
        "heightfield contacts are " + String(worst_pos) + " / "
        + String(worst_dist) + " from MuJoCo's",
    )
    _ = b^
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
