"""A face polygon wider than the cap silently lost its whole manifold.

    pixi run mojo run -I . tests/physics3d/test_multicontact_polygon_caps_vs_mujoco.mojo

WHAT WAS THERE. `native_multicontact` sizes its working polygons with two
comptime constants, and `_mesh_face` opens with

    if num > MC_MAX_POLYVERT:
        return 0

A return of 0 is the routine's own "these features do not line up" answer, so
the caller did exactly what the reference does when no faces align: it emitted
the single EPA point. The cap therefore did not overflow, crash or warn — it
DOWNGRADED the pair, and the file's own comment claimed the opposite ("THESE
CAPS ARE CHECKED AT MODEL BUILD, NOT SILENTLY OBEYED HERE").

⚠⚠ THE OLD VALUES WERE 16 AND 16, AND MENAGERIE IS NOT NEAR THEM. Measured on
the 3.10.0 runtime over the 59 mesh-bearing scenes, from `mesh_polyvertnum`
and `mesh_polymapnum`:

    widest face polygon         144 vertices (robotiq_2f85)   47/59 over 16
    most polygons at one vertex  47          (flexiv_rizon4)  39/59 over 16

MuJoCo has no such cap at all: `npolygonmax` / `nmeshdegmax` are runtime model
fields, sized per model. Ours must be comptime — a Metal kernel cannot size a
local array from a model field.

⚠⚠ AND METAL SETS THE CEILING, NOT TASTE. These arrays are function-local, so
they are per-thread stack in the collision kernel. Measured by bisection on
this machine, `test_plane_mesh_fields` compiles at (56, 48) and fails at
(64, 48) and at (56, 64) with

    Failed to create compute pipeline state: Compute function exceeds
    available stack space

so 56 and 48 are the largest values that keep the GPU path alive, and the CPU
takes the same numbers rather than quietly seeing a different manifold. That
covers EVERY vertex degree in Menagerie (the worst is 47) but not every
polygon width (the worst is robotiq_2f85's 144): 36 of the 59 mesh-bearing
scenes are fully covered, against 12 before. The remaining 23 keep the single
point on their widest faces, and now say so at build. Closing that gap means
moving the buffers off the stack into a scratch tensor, which is a change to
the routine's signature, not to a constant.

MEASURED CONSEQUENCE. kinova_gen3 ships `home` with its base and shoulder
hulls 12 mm interpenetrated across two faces of 31 and 29 vertices. MuJoCo
clips them to a four-point manifold; we returned one point.

    kinova_gen3   4.351e-02 -> 5.709e-12   (worst |d(qpos)|, one step)

⚠ EXACTLY TWO SCENES IN THE SWEEP MOVED, and that understates it. The cap only
bites on a pair that is actually touching, and at step one out of a keyframe
most of those 47 scenes are not touching on their widest face. The other one
that moved is `hello_robot_stretch_3`, 1.733e-02 -> 5.626e-02: its manifold is
now GENERATED and lands on a different face pair from MuJoCo's, a separate
narrow-phase defect this cap was masking behind a single point.

⚠ THE COST IS THE WORK, NOT THE ARRAYS. Interleaved, min of 3 rounds, 200
steps: barkour 106.1 -> 105.7 us/step and spot 115.7 -> 115.5 — unchanged,
because a pair that never reached the manifold path never touches the bigger
buffers. kinova pays 55.4 -> 70.8 us/step (+28%), which is the price of
clipping a manifold and solving four contact rows instead of one.
"""

from std.math import abs, sqrt
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
from mojo_rl.physics3d.collision.native_multicontact import (
    MC_MAX_POLYVERT, MC_MAX_DEG,
)
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, CONTACT_SIZE,
    CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X, CONTACT_IDX_POS_Y, CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX, CONTACT_IDX_NY, CONTACT_IDX_NZ, CONTACT_IDX_DIST,
)

comptime DT = DType.float64

comptime KINOVA = String(
    "references/mujoco_menagerie-main/kinova_gen3/scene.xml"
)

# MuJoCo 3.10.0 at kinova's keyframe 0, `mj_forward`. Four contacts, all on
# one plane, all at one depth, normal +z (geom1 -> geom2; our record stores
# `body_b -> body_a`, which is the opposite sign).
comptime MJ_NCON = 4
comptime MJ_DIST = -0.012044999767961329
comptime MJ_PLANE_Z = 0.164827498456591
# The centroid of MuJoCo's four points, and the radii they sit at around it.
comptime MJ_CX = -0.0014811771189564577
comptime MJ_CY = 0.0008911941017506531
comptime MJ_RMIN = 0.042
comptime MJ_RMAX = 0.046

# The largest values that still compile a Metal collision kernel on this
# machine, found by bisection — see the module docstring. Menagerie's own worst
# case is 144 (robotiq_2f85) and 47 (flexiv_rizon4), so the degree axis is
# fully covered and the width axis is not.
comptime REQUIRED_POLYVERT = 56
comptime REQUIRED_DEG = 48
# Menagerie's worst, measured on the runtime over all 59 mesh-bearing scenes.
comptime MENAGERIE_MAX_POLYVERT = 144  # robotiq_2f85
comptime MENAGERIE_MAX_DEG = 47        # flexiv_rizon4

comptime _IMPFAST = ImplicitIntegrator[
    DT, DynDims, ConeType.PYRAMIDAL, 1, "newton", SKIP_RNE_DERIV=True,
    MAX_CONDIM=6,
]


def test_caps_are_at_the_measured_ceiling() raises:
    """The constants are a REQUIREMENT, not a convenience.

    ⚠ THIS IS THE ROW THAT STOPS THEM SHRINKING BACK. The kinova assertion
    below needs only 38 and 21; every other scene that would lose a manifold
    is invisible to it, because a cap only bites on a pair that is actually
    touching on its widest face. Stating the ceiling here is what keeps the
    rest of Menagerie in the gate.

    ⚠ AND IT IS A FLOOR, NOT AN EQUALITY. If the buffers ever move off the
    stack the caps SHOULD rise past this, ideally to Menagerie's own 144 and
    47, and this file must not stand in the way of that.
    """
    print("=== the caps against the Metal ceiling and Menagerie's worst ===")
    print("  MC_MAX_POLYVERT", MC_MAX_POLYVERT, " ceiling",
          REQUIRED_POLYVERT, " Menagerie needs", MENAGERIE_MAX_POLYVERT)
    print("  MC_MAX_DEG     ", MC_MAX_DEG, " ceiling", REQUIRED_DEG,
          " Menagerie needs", MENAGERIE_MAX_DEG)
    assert_true(
        MC_MAX_POLYVERT >= REQUIRED_POLYVERT,
        "MC_MAX_POLYVERT is " + String(MC_MAX_POLYVERT) + ", below the "
        + String(REQUIRED_POLYVERT) + " a Metal collision kernel still"
        " compiles with. Every pair touching on a wider face falls back to a"
        " SINGLE point where MuJoCo clips a four-point manifold.",
    )
    assert_true(
        MC_MAX_DEG >= MENAGERIE_MAX_DEG,
        "MC_MAX_DEG is " + String(MC_MAX_DEG) + " but flexiv_rizon4 has a"
        " vertex where " + String(MENAGERIE_MAX_DEG) + " polygons meet."
        " `_mesh_normals` stops collecting candidate face normals at the cap,"
        " so the matching face may never be offered to `alignedFaces`."
        " Unlike the width axis this one IS fully reachable — do not give it"
        " up.",
    )
    print("  PASS")


def test_kinova_manifold_is_four_points() raises:
    """The model it was found on, against MuJoCo's own four.

    ⚠ THE POSITIONS ARE NOT PINNED, ON PURPOSE. MuJoCo keeps a different four
    of the same clipped ring (a pruner tie-break documented in
    `native_multicontact`'s `MC_DEBUG_RING`), so pinning them would gate a
    known-open residual instead of the cap. What IS pinned is everything the
    manifold must agree on: how many points, their common depth, the plane
    they lie in, the normal, and the ring radius they sit at around MuJoCo's
    centroid — which together say "we clipped the same two faces".
    """
    print("=== kinova_gen3 base/shoulder manifold ===")
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
    var nq = dims.get_nq()
    assert_true(
        dims.get_nkey() > 0,
        "kinova_gen3 must carry a keyframe — qpos0 does not interpenetrate"
        " and this gate would measure nothing",
    )
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
        d.qfrc.data[i] = Scalar[DT](0)
    var integ = _IMPFAST(dims)
    integ.step["cpu"](d, m)

    var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    print("  ncon", nc, "  (MuJoCo", MJ_NCON, ")")
    for k in range(nc):
        var o = k * CONTACT_SIZE
        print(
            "   ", k, " bodies",
            Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_A])),
            Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_B])),
            " pos", Float64(d.contacts.data[o + CONTACT_IDX_POS_X]),
            Float64(d.contacts.data[o + CONTACT_IDX_POS_Y]),
            Float64(d.contacts.data[o + CONTACT_IDX_POS_Z]),
            " dist", Float64(d.contacts.data[o + CONTACT_IDX_DIST]),
        )
    assert_true(
        nc == MJ_NCON,
        "ncon is " + String(nc) + " where MuJoCo has " + String(MJ_NCON)
        + ". A 1 here is the single EPA point, i.e. the manifold was dropped;"
        " anything above 4 means the clip is emitting points MuJoCo prunes.",
    )
    var worst_d = 0.0
    var worst_z = 0.0
    var worst_n = 0.0
    var rmin = 1e30
    var rmax = -1e30
    for k in range(nc):
        var o = k * CONTACT_SIZE
        var px = Float64(d.contacts.data[o + CONTACT_IDX_POS_X])
        var py = Float64(d.contacts.data[o + CONTACT_IDX_POS_Y])
        var pz = Float64(d.contacts.data[o + CONTACT_IDX_POS_Z])
        var ed = abs(Float64(d.contacts.data[o + CONTACT_IDX_DIST]) - MJ_DIST)
        if ed > worst_d:
            worst_d = ed
        var ez = abs(pz - MJ_PLANE_Z)
        if ez > worst_z:
            worst_z = ez
        # The record stores `body_b -> body_a`, the opposite of MuJoCo's
        # frame, so only the AXIS is compared: |nz| must be 1 and nx, ny 0.
        var en = abs(abs(Float64(d.contacts.data[o + CONTACT_IDX_NZ])) - 1.0)
        var ex = abs(Float64(d.contacts.data[o + CONTACT_IDX_NX]))
        var ey = abs(Float64(d.contacts.data[o + CONTACT_IDX_NY]))
        if ex > en:
            en = ex
        if ey > en:
            en = ey
        if en > worst_n:
            worst_n = en
        var dx = px - MJ_CX
        var dy = py - MJ_CY
        var r = sqrt(dx * dx + dy * dy)
        if r < rmin:
            rmin = r
        if r > rmax:
            rmax = r
    print("  worst |d dist|", worst_d, " |d plane z|", worst_z,
          " |d normal|", worst_n)
    print("  ring radius about MuJoCo's centroid: [", rmin, ",", rmax, "]")
    assert_true(
        worst_d < 1e-8,
        "the four points must share MuJoCo's depth " + String(MJ_DIST)
        + "; worst |d| = " + String(worst_d),
    )
    assert_true(
        worst_z < 1e-8,
        "the four points must lie in MuJoCo's contact plane z = "
        + String(MJ_PLANE_Z) + "; worst |d| = " + String(worst_z),
    )
    assert_true(
        worst_n < 1e-12,
        "the manifold normal must be the face normal, +-z here; worst"
        " deviation " + String(worst_n),
    )
    # ⚠ THE ROW THAT SAYS "THE SAME TWO FACES". Four points at the right depth
    # could still come from the wrong feature; MuJoCo's sit on a ring of
    # radius 0.0427-0.0451 about its centroid, and a manifold clipped from a
    # different face pair does not land there.
    assert_true(
        rmin > MJ_RMIN and rmax < MJ_RMAX,
        "the four points sit at radius [" + String(rmin) + ", "
        + String(rmax) + "] about MuJoCo's centroid, outside its own ["
        + String(MJ_RMIN) + ", " + String(MJ_RMAX) + "] — the clip found a"
        " different face pair, not a different four points of the same one.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
