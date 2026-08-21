"""`mjModel.mesh_vert` is `float*`, and one ulp decides a tangent contact.

    pixi run mojo run -I . tests/physics3d/test_mesh_verts_are_float32_vs_mujoco.mojo

WHAT WAS DIFFERENT. MuJoCo's compiler does every mesh step in double — scale,
recentre, convex hull, inertia — and then copies the vertices into a FLOAT
array. That float array is what collision reads: `mjc_initCCDObj` hands
`m->mesh_vert` straight to the support function, and `mjc_PlaneConvex` walks
the same floats. We kept our copy in double, so our hull sat a few hundred
picometres away from the one the reference actually collides with.

⚠⚠ THAT IS ENOUGH TO CREATE A CONTACT OUT OF NOTHING. rby1's two drive wheels
are modelled EXACTLY tangent to the floor. MuJoCo's lowest hull vertex lands at
world z = +6.372438687840543e-11 and ours landed at -7.450580430390374e-10, so
we opened two contacts MuJoCo does not have and the robot diverged 4.99e-03 in
a single step. The two hulls differ by ONE float32 ulp: rounding ours gives
MuJoCo's value BIT FOR BIT on every coordinate.

    rainbow_robotics_rby1/scene_rby1a_1.2            4.986e-03 -> 2.860e-05
    rainbow_robotics_rby1/scene_rby1a_1.2_no_gripper 4.989e-03 -> 2.860e-05
    rainbow_robotics_rby1/scene_rby1m_1.2            1.287e-04 -> 2.095e-05
    google_barkour_v0/scene_barkour                  1.386e-10 -> 5.551e-17

⚠ AFTER THE HULL, NOT BEFORE. MuJoCo picks hull membership from the double
vertices; only the STORED coordinates are float. Rounding earlier could change
which vertices survive, which is a different model, not a closer one.

⚠ AND ONLY THE VERTICES. Measured on the 3.10.0 runtime: `mesh_vert` and
`mesh_normal` are float32, `mesh_polynormal` is float64. Rounding the polygon
normals too would be a second wrong answer, not a more consistent one.

⚠ NINE OF 85 MENAGERIE SCENES MOVED and nothing regressed above the float
noise floor — the one that rose, trossen_wxai, went 6.7e-11 to 3.0e-10, both
two orders below the 1e-9 the sweep calls exact.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.integrator.implicit import ImplicitIntegrator
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, MODEL_MESH_META_SIZE, MODEL_GEOM_SIZE,
    GEOM_IDX_MESH_ID,
)

comptime DT = DType.float64

comptime RBY1 = String(
    "references/mujoco_menagerie-main/rainbow_robotics_rby1/scene_rby1a_1.2.xml"
)
# ⚠ THE NON-VACUITY MODEL. Its hulls really do interpenetrate at reset, so a
# "no contacts anywhere" regression cannot pass this file.
comptime KINOVA = String(
    "references/mujoco_menagerie-main/kinova_gen3/scene.xml"
)

# MuJoCo 3.10.0, `m.mesh_vert` for rby1's right drive wheel (geom 10).
comptime MJ_WHEEL_ZMIN = -0.09996499866247177
comptime MJ_WHEEL_ZMAX = 0.09996499866247177

comptime _IMPFAST = ImplicitIntegrator[
    DT, DynDims, ConeType.PYRAMIDAL, 1, "newton", SKIP_RNE_DERIV=True,
    MAX_CONDIM=6,
]




def test_every_stored_vertex_survives_a_float32_round_trip() raises:
    """The property, over every collidable vertex of a real model.

    ⚠ THIS IS THE ONE THAT GENERALISES. The rby1 numbers below pin one mesh;
    this says the whole array is what MuJoCo would store, so a future mesh
    step that reintroduces double precision fails here rather than waiting for
    a model whose geometry happens to be tangent.
    """
    print("=== every stored hull vertex is a float32 ===")
    var src = read_model_source(RBY1)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
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
            dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var checked = 0
    var bad = 0
    var worst = 0.0
    for i in range(dims.get_nmesh_verts() * 3):
        var v = Float64(m.mesh_verts.data[i])
        if v == 0.0:
            continue
        var r = Float64(Scalar[DT](v).cast[DType.float32]())
        checked += 1
        if r != v:
            bad += 1
            if abs(r - v) > worst:
                worst = abs(r - v)
    print("  nonzero coordinates checked", checked, " not float32", bad,
          " worst |d|", worst)
    # ⚠ VACUITY GUARD, AND THE NUMBER IS MEASURED. rby1's collidable hulls
    # come to 5121 nonzero coordinates; the bound is set below that with room
    # for the meshes to change, but far enough above zero that an empty or
    # unbuilt array cannot pass. The first draft guessed "tens of thousands"
    # and failed on the real model — a bound from imagination gates nothing.
    assert_true(
        checked > 3000,
        "only " + String(checked) + " nonzero mesh coordinates were checked —"
        " rby1 carries about 5121 and the loop is measuring nothing",
    )
    assert_true(
        bad == 0,
        String(bad) + " of " + String(checked) + " stored hull coordinates"
        " are not exactly representable in float32 (worst |d| "
        + String(worst) + "). MuJoCo's `mesh_vert` is `float*` and every"
        " collision routine reads it, so a double here is a hull the"
        " reference never collides with.",
    )
    print("  PASS")


def test_rby1_wheels_do_not_touch_the_floor() raises:
    """The consequence, on the model that found it.

    rby1's drive wheels are modelled exactly tangent; MuJoCo reports ncon 0 at
    the reset pose. One float32 ulp on the hull put our lowest vertex 745 pm
    BELOW the floor instead of 64 pm above it, and two contacts appeared.
    """
    print("=== rby1 at reset: MuJoCo ncon 0 ===")
    var src = read_model_source(RBY1)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
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
            dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)

    # The wheel's own local z extent, against MuJoCo's stored floats.
    var mid = Int(Float64(
        m.geoms.data[10 * MODEL_GEOM_SIZE + GEOM_IDX_MESH_ID]
    ))
    assert_true(
        mid >= 0,
        "geom 10 must be rby1's right drive wheel mesh; got mesh_id "
        + String(mid),
    )
    var o = mid * MODEL_MESH_META_SIZE
    var va = Int(Float64(m.mesh_meta.data[o + 0]))
    var nv = Int(Float64(m.mesh_meta.data[o + 1]))
    var lo = 1e30
    var hi = -1e30
    for i in range(nv):
        var z = Float64(m.mesh_verts.data[(va + i) * 3 + 2])
        if z < lo:
            lo = z
        if z > hi:
            hi = z
    print("  wheel local z [", lo, ",", hi, "]")
    print("  MuJoCo        [", MJ_WHEEL_ZMIN, ",", MJ_WHEEL_ZMAX, "]")
    # ⚠ EXACT. Both sides are float32 values held in a double, so the only
    # right answer is bit equality — a tolerance here would hide the ulp.
    assert_true(
        lo == MJ_WHEEL_ZMIN and hi == MJ_WHEEL_ZMAX,
        "the wheel hull's local z extent is [" + String(lo) + ", "
        + String(hi) + "] against MuJoCo's [" + String(MJ_WHEEL_ZMIN) + ", "
        + String(MJ_WHEEL_ZMAX) + "] — one float32 ulp apart is the defect.",
    )

    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
        d.qfrc.data[i] = Scalar[DT](0)
    var integ = _IMPFAST(dims)
    integ.step["cpu"](d, m)
    var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    print("  ncon", nc, "  (MuJoCo 0)")
    assert_true(
        nc == 0,
        "rby1 reports " + String(nc) + " contacts at its reset pose where"
        " MuJoCo reports 0. Its wheels are exactly tangent to the floor, so a"
        " hull one ulp too large pushes them through it.",
    )
    print("  PASS")


def test_a_model_that_really_does_touch_still_does() raises:
    """The negative control: rounding must not delete real contacts.

    ⚠ WITHOUT THIS ROW the file above passes against a narrow phase that
    reports nothing at all. kinova_gen3's base and shoulder hulls start 12 mm
    interpenetrated at its own keyframe and MuJoCo finds four contacts there.
    """
    print("=== kinova still finds its four ===")
    var src = read_model_source(KINOVA)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
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
            dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
        d.qfrc.data[i] = Scalar[DT](0)
    var integ = _IMPFAST(dims)
    integ.step["cpu"](d, m)
    var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    print("  ncon", nc, "  (MuJoCo 4)")
    assert_true(
        nc == 4,
        "kinova_gen3 reports " + String(nc) + " contacts at its keyframe"
        " where MuJoCo reports 4 — rounding the hull must not cost a real"
        " contact.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
