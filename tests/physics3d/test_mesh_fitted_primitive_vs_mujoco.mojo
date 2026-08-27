"""`<geom mesh="...">` on a PRIMITIVE type is FITTED to that mesh.

    pixi run mojo run -I . tests/physics3d/test_mesh_fitted_primitive_vs_mujoco.mojo

WHAT WAS MISSING. A geom that names a mesh while its `type` is a primitive is
neither an error nor a mesh: MuJoCo fits the primitive to that mesh and then
CLEARS the mesh reference (`mjCGeom::Compile` -> `mjCMesh::FitGeom`,
user_objects.cc:4038). We resolved `mesh=` ONLY when the type was already
`mesh`, so such a geom got no mesh at all and fell back to the default sphere
radius of 0.5.

THE FIT (default `fitaabb=false`, so the INERTIA BOX and not the AABB —
user_mesh.cc:944):

    boxsz[k] = 0.5*sqrt(6*(the OTHER two eigvals - eigval[k]) / volume)
    SPHERE            size[0] = (bx + by + bz)/3
    CAPSULE           size[0] = (bx + by)/2, size[1] = max(0, bz - size[0]/2)
    CYLINDER          size[0] = (bx + by)/2, size[1] = bz
    BOX / ELLIPSOID   size    = (bx, by, bz)

⚠ THE EIGENVALUES ARE UNITLESS ON BOTH SIDES — MuJoCo divides by the mesh
VOLUME here, so no mass or density enters. Scaling them by `mass/volume`
first, which is what the body-inertia path does, would make a geom's size
depend on how heavy its body is.

⚠⚠ IT IS NOT A ROUNDING-SIZE DETAIL, IT IS A DIFFERENT ROBOT. rainbow_robotics
rby1's `<default class="in-model-collision">` sets contype/conaffinity and NO
type, so all 49 of its collidable arm and finger geoms are fitted spheres.
Without the fit they were spheres of radius 0.5 — 33x too large for a finger —
and the model self-collided everywhere:

    ncon at qpos0    ours 128 (the buffer cap)   MuJoCo 0

including a reported penetration of 0.56 m between two grippers 0.44 m apart,
which is exactly 0.5 + 0.5 - 0.44 and is the fingerprint of a default radius
rather than a wrong one.

⚠ THE FRAME COMES WITH IT. `mjuu_frameaccum` sits OUTSIDE MuJoCo's fitting
branch (user_objects.cc:4053), so a fitted primitive is offset by the mesh's
CoM exactly as a mesh geom is. Fitting the SIZE but not the POSE left two of
rby1's arm spheres CONCENTRIC — a reported `dist` of exactly `-(r1 + r2)`,
which is what a zero centre-distance gives and is a different fingerprint from
a wrong radius. Both are needed; the first fix alone took ncon 128 -> 18, the
second 18 -> 2.

⚠ BUT NOT THE INERTIA. MuJoCo has cleared the mesh pointer by the time it
computes a fitted geom's mass properties, so those come from the PRIMITIVE.
Feeding the mesh's volume and eigenvalues to the body-inertia path would give
a sphere the tensor of the mesh it was fitted to.

SCOPE: 7 models, all Menagerie — rby1 (49 fitted geoms across five scenes),
agilex_piper (7) and arx_l5 (8). Zero in-repo environment assets use it, so
no training changes.

MEASURED, `geom_rbound` against MuJoCo 3.10.0 over every COLLIDABLE non-plane
geom (72 on rby1): worst |d| 1.8e-15, from 0.485 before. rby1's worst
|d(qpos)| under a fixed random control sequence, 100 steps from qpos0:
1.402 -> 0.0997.

⚠ WHAT IS STILL WRONG, because the number above is not zero: rby1's two WHEEL
geoms are ordinary `type="mesh"` geoms whose mesh MuJoCo reorients into its
PRINCIPAL AXIS frame (`mesh_quat[WHEEL] = (0.5, 0.5, 0.5, 0.5)`, a 120-degree
axis permutation). MuJoCo's compiled z half-extent is 0.099965 where the raw
file's is 0.1, so its wheels clear the floor by 35 um and it reports ncon 0 at
qpos0; ours rest exactly on it and report 2 contacts carrying 178 N. That is a
separate, pre-existing defect in how the hull frame and `mesh_quat` are
composed — it predates this fix and this fix does not touch it.
"""

from std.math import abs, sqrt
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, read_model_source,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE, GEOM_IDX_RBOUND, GEOM_IDX_RADIUS, GEOM_IDX_TYPE,
    GEOM_IDX_CONTYPE, GEOM_IDX_CONAFFINITY,
)

comptime DT = DType.float64

comptime RBY1 = String(
    "references/mujoco_menagerie-main/rainbow_robotics_rby1/"
    + "scene_rby1a_1.2.xml"
)
comptime PIPER = String(
    "references/mujoco_menagerie-main/agilex_piper/scene.xml"
)

# MuJoCo 3.10.0's `geom_rbound` for rby1's two finger collision geoms, which
# are `<geom mesh="EE_FINGER_0"/>` with no type at all. Cross-checked against
# the formula by hand from the raw OBJ: boxsz (0.0295045, 0.0139041,
# 0.0015000) -> (bx+by+bz)/3 = 0.0149700.
comptime MJ_FINGER0_R = 0.014969515740122863
comptime MJ_FINGER1_R = 0.015776957368352922


def _load(path: String) raises -> Tuple[Model[DT, DynDims], DynDims]:
    var src = read_model_source(path)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
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
            dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    return (m^, dims)


def test_rby1_fitted_spheres_match_mujoco() raises:
    """The two finger geoms, against MuJoCo's own `geom_rbound`.

    ⚠ THE VALUES ARE MUJOCO'S, and they are ALSO reproduced from the mesh by
    the documented formula — a fit that happened to land on our own default
    would otherwise be indistinguishable from a correct one.
    """
    print("=== rby1 fitted spheres ===")
    var r = _load(RBY1)
    ref m = r[0]
    var dims = r[1]
    var g84 = Float64(m.geoms.data[84 * MODEL_GEOM_SIZE + GEOM_IDX_RADIUS])
    var g85 = Float64(m.geoms.data[85 * MODEL_GEOM_SIZE + GEOM_IDX_RADIUS])
    print("  geom 84 radius", g84, " (MuJoCo", MJ_FINGER0_R, ")")
    print("  geom 85 radius", g85, " (MuJoCo", MJ_FINGER1_R, ")")
    assert_true(
        abs(g84 - MJ_FINGER0_R) < 1e-9 and abs(g85 - MJ_FINGER1_R) < 1e-9,
        "rby1's finger geoms are `<geom mesh=... />` with NO type, so MuJoCo"
        " fits a sphere to the mesh's inertia box: radii "
        + String(MJ_FINGER0_R) + " and " + String(MJ_FINGER1_R) + ". Got "
        + String(g84) + " and " + String(g85)
        + ". A value of 0.5 means the mesh was never resolved and the geom"
        " kept the default sphere size.",
    )
    # ⚠ THE NEGATIVE CONTROL. If the fit were applied to geoms that already
    # declare a size, this would move — geom 0 is the scene's floor plane and
    # geom 5 is a genuine `type="mesh"` geom whose rbound comes from its hull.
    var g5 = Float64(m.geoms.data[5 * MODEL_GEOM_SIZE + GEOM_IDX_RBOUND])
    print("  geom 5 (a real type='mesh' geom) rbound", g5, " (MuJoCo"
          " 0.08765793)")
    # ⚠ 1e-5, NOT 1e-9. This is a HULL rbound — the largest vertex norm of a
    # mesh MuJoCo stores as FLOAT32 (`mjModel.mesh_vert` is `float*`), so the
    # two engines cannot agree past ~1e-7 relative however correct both are.
    # The fitted radii above ARE asserted at 1e-9 because they come from the
    # mesh's inertia integral in float64 on both sides. Pinning this row at
    # 1e-9 would be a gate that fails for the float32 noise floor rather than
    # for a defect — this tree has paid for that three times.
    assert_true(
        abs(g5 - 0.0876579330731085) < 1e-5,
        "a geom that really IS `type='mesh'` must keep its hull rbound, not"
        " be refitted; got " + String(g5),
    )
    print("  PASS")


def test_no_collidable_geom_keeps_the_default_radius() raises:
    """The whole-model check, and the one that names the failure mode.

    ⚠ THE OBSERVABLE IS "IS ANY COLLIDABLE GEOM STILL 0.5", not a per-geom
    golden. The defect gave EVERY unfitted geom the same default radius, so
    counting them is both the symptom and the measure — 49 on rby1 before, 0
    after. A per-geom list would rot the first time Menagerie edits a mesh.
    """
    print("=== no collidable geom is left at the default radius ===")
    for which in range(2):
        var path = RBY1 if which == 0 else PIPER
        var r = _load(path)
        ref m = r[0]
        var dims = r[1]
        var n_default = 0
        var n_coll = 0
        for g in range(dims.get_ngeom()):
            var o = g * MODEL_GEOM_SIZE
            var ct = Int(Float64(m.geoms.data[o + GEOM_IDX_CONTYPE]))
            var ca = Int(Float64(m.geoms.data[o + GEOM_IDX_CONAFFINITY]))
            if ct == 0 and ca == 0:
                continue
            n_coll += 1
            var rb = Float64(m.geoms.data[o + GEOM_IDX_RBOUND])
            if abs(rb - 0.5) < 1e-12:
                n_default += 1
        print("  ", "rby1" if which == 0 else "agilex_piper",
              " collidable", n_coll, " still at the 0.5 default", n_default)
        # ⚠ VACUITY. A model that loaded no collidable geoms would pass the
        # count below for the wrong reason.
        # ⚠ 10, sized to the SMALLER of the two models. rby1 has 73
        # collidable geoms and agilex_piper 12; a bound of 20 would have made
        # this row fail on piper for a reason that has nothing to do with the
        # fit, which is what it did on the first run.
        assert_true(
            n_coll >= 10,
            "expected a model with many collidable geoms — got "
            + String(n_coll),
        )
        assert_true(
            n_default == 0,
            String(n_default) + " collidable geoms are still spheres of"
            " radius 0.5 — the default. Every one of them names a mesh that"
            " MuJoCo would have fitted the primitive to; at 0.5 they collide"
            " with most of the robot.",
        )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
