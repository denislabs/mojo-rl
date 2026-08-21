"""`<mesh refpos>` / `<mesh refquat>` — the transform before everything else.

    pixi run mojo run -I . tests/physics3d/test_mesh_refquat_vs_mujoco.mojo

WHAT WAS MISSING. `mjCMesh::ApplyTransformations` (user_mesh.cc:1257) is the
FIRST thing MuJoCo does to a mesh's vertices, before the centre of mass, the
principal axes, the hull and the inertia:

    if (refpos != 0)   v -= refpos
    if (refquat != I)  v  = R(normalize(refquat))^T v     <- mjuu_mulvecmatT
    if (scale  != 1)   v *= scale

We read `scale` and nothing else, so a mesh declaring either was compiled in
the wrong frame — and the frame is what `geom_pos` / `geom_quat` carry, so it
propagated straight into the body's inertial frame and its MASS MATRIX.

⚠⚠ THE ROTATION IS THE QUATERNION'S INVERSE. `mjuu_mulvecmatT(res, vec, mat)`
is `res = M^T vec`, so `refquat="1 -1 0 0"` — a -90 deg turn about x — rotates
the mesh **+90 deg**. Reading it forward lands 180 deg away, which on a roughly
symmetric part still looks like a plausible mesh. Row `gb` below is what pins
the direction: taking the forward rotation gives
`(-0.000234, +0.007838, +0.015295)` and MuJoCo gives
`(-0.000234, -0.007838, -0.015295)`.

⚠ AND IT COMES BEFORE `scale`, which is why the loaders ask for UNSCALED
vertices whenever a ref transform is present and apply all three steps
themselves. With both at the identity — 84 of Menagerie's 85 scenes — the old
call stands unchanged, and the step-1 sweep confirms it: exactly ONE scene
moved.

MEASURED. shadow_dexee is the only Menagerie model that uses it, on all 13 of
its meshes, every one `refquat="1 -1 0 0"`. Its knuckle mesh's centre of mass
came out (-0.000234, -0.015295, 0.007838) here against MuJoCo's
(-0.000234, -0.007838, -0.015295): the same three numbers with y and z
exchanged and a sign flipped, the fingerprint of a dropped 90 deg turn.

    body_ipos  |d|   2.313e-02  ->  1.388e-17
    body_iquat |d|   2.929e-01  ->  1.554e-15
    mass matrix      24.5% of |M|max  ->  below 1e-12

⚠ THE SWEEP BARELY MOVED — 6.054e-03 to 5.374e-03 — because shadow_dexee's
twelve actuators are `<plugin plugin="mujoco.pid">` and produce no force here
at all. A model whose actuators are dead cannot show a dynamics fix. The mass
matrix is the measurement that could see it.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, read_model_source,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE, GEOM_IDX_POS_X, GEOM_IDX_POS_Y, GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_W, GEOM_IDX_QUAT_X, GEOM_IDX_QUAT_Y, GEOM_IDX_QUAT_Z,
    MODEL_BODY_SIZE, BODY_IDX_IPOS_X, BODY_IDX_IPOS_Y, BODY_IDX_IPOS_Z,
)

comptime DT = DType.float64

# The base directory shadow_dexee's `meshdir="assets"` resolves against.
comptime DEXEE_DIR = String(
    "references/mujoco_menagerie-main/shadow_dexee/"
)
comptime DEXEE = String(
    "references/mujoco_menagerie-main/shadow_dexee/scene.xml"
)

# ⚠ THREE BODIES, ONE MESH FILE, AND THE ONLY DIFFERENCE IS THE ATTRIBUTE.
# `a` declares neither, `b` the rotation, `c` the offset — so a mismatch on
# `b` or `c` cannot be the loader, the hull or the inertia routine, all of
# which `a` exercises identically.
comptime XML = String(
    """<mujoco>
  <compiler angle="radian" meshdir="assets"/>
  <asset>
    <mesh name="plain" file="MRH-F-J0Link-Visual,00.stl"/>
    <mesh name="turned" file="MRH-F-J0Link-Visual,00.stl" refquat="1 -1 0 0"/>
    <mesh name="shifted" file="MRH-F-J0Link-Visual,00.stl"
          refpos="0.01 0.02 0.03"/>
  </asset>
  <worldbody>
    <body name="a"><joint type="hinge" axis="0 0 1"/>
      <geom name="ga" type="mesh" mesh="plain" mass="0.13077995"/></body>
    <body name="b" pos="1 0 0"><joint type="hinge" axis="0 0 1"/>
      <geom name="gb" type="mesh" mesh="turned" mass="0.13077995"/></body>
    <body name="c" pos="2 0 0"><joint type="hinge" axis="0 0 1"/>
      <geom name="gc" type="mesh" mesh="shifted" mass="0.13077995"/></body>
  </worldbody>
</mujoco>"""
)

# MuJoCo 3.10.0 `geom_pos` / `geom_quat` for the three geoms above.
comptime A_PX = -0.00023377087580675395
comptime A_PY = -0.015295115474376413
comptime A_PZ = 0.007837674710718487
comptime A_QW = 0.9638521191083312
comptime B_PX = -0.0002337708758067541
comptime B_PY = -0.007837674710718483
comptime B_PZ = -0.01529511547437641
comptime B_QW = 0.8695494654271164
comptime B_QX = 0.49354327353793453
comptime C_PX = -0.010233770875806774
comptime C_PY = -0.035295115474376526
comptime C_PZ = -0.02216232528928166

# shadow_dexee's four right-index bodies, `m.body_ipos`.
comptime D4 = 4
comptime D4_X = -0.00023377087580675512
comptime D4_Y = -0.007837674710718488
comptime D4_Z = -0.015295115474376404


def _geoms(xml: String, base: String) raises -> List[Float64]:
    """`[pos xyz, quat wxyz]` per geom, plus the bodies' `ipos` appended."""
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                raise e
            tries += 1
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var out = List[Float64]()
    for g in range(dims.get_ngeom()):
        var o = g * MODEL_GEOM_SIZE
        out.append(Float64(m.geoms.data[o + GEOM_IDX_POS_X]))
        out.append(Float64(m.geoms.data[o + GEOM_IDX_POS_Y]))
        out.append(Float64(m.geoms.data[o + GEOM_IDX_POS_Z]))
        out.append(Float64(m.geoms.data[o + GEOM_IDX_QUAT_W]))
        out.append(Float64(m.geoms.data[o + GEOM_IDX_QUAT_X]))
        out.append(Float64(m.geoms.data[o + GEOM_IDX_QUAT_Y]))
        out.append(Float64(m.geoms.data[o + GEOM_IDX_QUAT_Z]))
    return out^


def _ipos(path: String, body: Int) raises -> List[Float64]:
    var src = read_model_source(path)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                raise e
            tries += 1
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var o = body * MODEL_BODY_SIZE
    var out = List[Float64]()
    out.append(Float64(m.bodies.data[o + BODY_IDX_IPOS_X]))
    out.append(Float64(m.bodies.data[o + BODY_IDX_IPOS_Y]))
    out.append(Float64(m.bodies.data[o + BODY_IDX_IPOS_Z]))
    return out^


def test_refquat_rotates_by_the_inverse() raises:
    """One mesh file, three declarations, MuJoCo's three answers."""
    print("=== <mesh refpos/refquat>, one file three ways ===")
    var g = _geoms(XML, DEXEE_DIR)
    assert_true(
        len(g) == 3 * 7,
        "the fixture must build three geoms; got " + String(len(g) // 7),
    )
    print("  a (neither) ", g[0], g[1], g[2])
    print("  b (refquat) ", g[7], g[8], g[9])
    print("  c (refpos)  ", g[14], g[15], g[16])
    # ⚠ ROW `a` IS THE CONTROL. It shares the mesh, the loader, the hull and
    # the inertia routine with the other two; if it moves, the defect is not
    # in the ref transform.
    assert_true(
        abs(g[0] - A_PX) < 1e-12 and abs(g[1] - A_PY) < 1e-12
        and abs(g[2] - A_PZ) < 1e-12 and abs(g[3] - A_QW) < 1e-12,
        "the mesh with NO refpos/refquat must be unchanged: got ("
        + String(g[0]) + ", " + String(g[1]) + ", " + String(g[2])
        + ") against MuJoCo's (" + String(A_PX) + ", " + String(A_PY) + ", "
        + String(A_PZ) + ")",
    )
    # ⚠ THIS ROW PINS THE DIRECTION OF THE ROTATION. Applying `refquat`
    # forward instead of transposed gives (+0.007838, +0.015295) here.
    assert_true(
        abs(g[7] - B_PX) < 1e-12 and abs(g[8] - B_PY) < 1e-12
        and abs(g[9] - B_PZ) < 1e-12,
        "refquat=\"1 -1 0 0\" must rotate the mesh +90 deg about x — MuJoCo's"
        " `mjuu_mulvecmatT` is M^T v, the INVERSE turn. Got (" + String(g[7])
        + ", " + String(g[8]) + ", " + String(g[9]) + ") against ("
        + String(B_PX) + ", " + String(B_PY) + ", " + String(B_PZ) + ").",
    )
    assert_true(
        abs(g[10] - B_QW) < 1e-12 and abs(g[11] - B_QX) < 1e-12,
        "the principal-axis quaternion must turn with the vertices; got w "
        + String(g[10]) + " x " + String(g[11]) + " against " + String(B_QW)
        + " / " + String(B_QX),
    )
    # ⚠ AND `refpos` IS SUBTRACTED, NOT ADDED. The centre of mass moves by
    # -refpos exactly: -0.000234 - 0.01, -0.015295 - 0.02, 0.007838 - 0.03.
    assert_true(
        abs(g[14] - C_PX) < 1e-12 and abs(g[15] - C_PY) < 1e-12
        and abs(g[16] - C_PZ) < 1e-12,
        "refpos=\"0.01 0.02 0.03\" must SUBTRACT from every vertex; got ("
        + String(g[14]) + ", " + String(g[15]) + ", " + String(g[16])
        + ") against (" + String(C_PX) + ", " + String(C_PY) + ", "
        + String(C_PZ) + ").",
    )
    # ⚠ AND IT MUST NOT ROTATE ANYTHING. A translation leaves the principal
    # axes alone, so `c`'s quaternion is `a`'s — this is what separates a
    # refpos bug from a refquat bug if both rows ever fail together.
    assert_true(
        abs(g[17] - A_QW) < 1e-12,
        "refpos must not change the principal-axis quaternion: got "
        + String(g[17]) + " against a's " + String(A_QW),
    )
    print("  PASS")


def test_shadow_dexee_inertial_frame() raises:
    """The real model, whose 13 meshes all declare `refquat="1 -1 0 0"`."""
    print("=== shadow_dexee F0/finger_knuckle ipos ===")
    var p = _ipos(DEXEE, D4)
    print("  ours  ", p[0], p[1], p[2])
    print("  MuJoCo", D4_X, D4_Y, D4_Z)
    assert_true(
        abs(p[0] - D4_X) < 1e-12 and abs(p[1] - D4_Y) < 1e-12
        and abs(p[2] - D4_Z) < 1e-12,
        "F0/finger_knuckle's centre of mass is (" + String(p[0]) + ", "
        + String(p[1]) + ", " + String(p[2]) + ") against MuJoCo's ("
        + String(D4_X) + ", " + String(D4_Y) + ", " + String(D4_Z)
        + "). y and z exchanged with a sign flipped is the dropped 90 deg"
        " turn; this body's mass matrix goes with it.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
