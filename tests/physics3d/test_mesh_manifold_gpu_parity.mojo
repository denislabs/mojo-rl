"""The multicontact CLIPPER on the GPU — the path that had never actually RUN.

WHY THIS FILE EXISTS. `native_multicontact` was ported, gated and landed against
MuJoCo entirely on the CPU. The Metal kernel LINKED with the clipper compiled
into it and `test_mesh_detection_fields` passed its GPU leg — but that fixture's
geom pair never reaches `multicontact()` (only one of its geoms is box-or-mesh,
so `maxContacts` returns 1), so the clipper had never executed on a GPU even
once. A green build is not a runnable kernel, and a green CPU/GPU comparison
over a pair that never enters the branch gates nothing.

⚠ THIS IS DEFECT 27'S FAMILY. `_polygon_clip` keeps its ring, its per-edge plane
normals and its distances in RUNTIME-INDEXED PER-THREAD `InlineArray`s. A Metal
miscompute of exactly that shape cost 0.275 in contact position and was
invisible to every green build until someone compared CPU against GPU
numerically; see `feedback_metal_wide_per_thread_inlinearray_miscompute`.

⚠ FLOAT32, NOT FLOAT64. The sibling MuJoCo-parity file
`test_mesh_manifold_vs_mujoco.mojo` is float64 because it compares against a
float64 reference. Float64 is BANNED on the GPU path here, so this file
re-instantiates the SAME fixture at float32. That is also why the tolerance
below is 1e-4 and not 1e-15: this gate is about CPU-vs-GPU agreement, not about
MuJoCo parity, which the sibling already covers.

⚠ THE POSE SCHEDULE IS THE MEASUREMENT. Regimes 0 and 1 (exactly aligned, and a
sub-degree tilt) are what drive `alignedFaces` into the face/face branch and
produce the multi-point manifolds that run the clipper at all. A uniform random
sweep would emit one point per pair and pass this file vacuously, which is why
`multi_pairs` below is asserted rather than printed.

Run: pixi run mojo run -I . tests/physics3d/test_mesh_manifold_gpu_parity.mojo
"""

from std.math import abs, sqrt, sin, cos
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap
from mojo_rl.physics3d.collision.native_multicontact import MC_ENABLED
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    METADATA_SIZE,
    META_IDX_NUM_CONTACTS,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
)


comptime DTYPE = DType.float32

# The same five groups as the MuJoCo-parity sibling, kept verbatim so a failure
# here can be read against that file's numbers. Both orderings of mesh/box are
# present because the narrow-phase branch is chosen from the order the geoms
# arrive in.
comptime MM_XML = """
<mujoco model="mesh manifold gpu">
  <option timestep="0.002"/>
  <asset>
    <mesh name="cube" file="tests/physics3d/assets/mc_cube.stl"/>
    <mesh name="hex" file="tests/physics3d/assets/mc_hex.stl"/>
  </asset>
  <worldbody>
    <body name="a0" pos="0 0 0.5">
      <geom name="g0a" type="mesh" mesh="cube"/>
    </body>
    <body name="b0" pos="0 0 0.5">
      <joint name="j0" type="free"/>
      <geom name="g0b" type="box" size=".05 .04 .06"/>
    </body>

    <body name="a1" pos="2 0 0.5">
      <geom name="g1a" type="box" size=".05 .04 .06"/>
    </body>
    <body name="b1" pos="2 0 0.5">
      <joint name="j1" type="free"/>
      <geom name="g1b" type="mesh" mesh="cube"/>
    </body>

    <body name="a2" pos="4 0 0.5">
      <geom name="g2a" type="mesh" mesh="cube"/>
    </body>
    <body name="b2" pos="4 0 0.5">
      <joint name="j2" type="free"/>
      <geom name="g2b" type="mesh" mesh="cube"/>
    </body>

    <body name="a3" pos="6 0 0.5">
      <geom name="g3a" type="mesh" mesh="hex"/>
    </body>
    <body name="b3" pos="6 0 0.5">
      <joint name="j3" type="free"/>
      <geom name="g3b" type="box" size=".05 .04 .06"/>
    </body>

    <body name="a4" pos="8 0 0.5">
      <geom name="g4a" type="mesh" mesh="hex"/>
    </body>
    <body name="b4" pos="8 0 0.5">
      <joint name="j4" type="free"/>
      <geom name="g4b" type="mesh" mesh="cube"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime mm = parse_xml(MM_XML)
comptime MMM = ModelDefFromXML[
    xml=MM_XML,
    nbody=mm.NBODY, njoint=mm.NJOINT, nq=mm.NQ, nv=mm.NV,
    ngeom=mm.NGEOM, nact=mm.NACT, ntex=mm.NTEX, nmat=mm.NMAT,
    nlight=mm.NLIGHT, ncam=mm.NCAM, nsite=mm.NSITE,
    max_tendon=mm.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=64,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=mm.TIMESTEP,
]

comptime NQ: Int = MMM.NQ
comptime NV: Int = MMM.NV
comptime NBODY: Int = MMM.NBODY
comptime NJOINT: Int = MMM.NJOINT
comptime NGEOM: Int = MMM.NGEOM
comptime NEQ: Int = MMM.MAX_EQUALITY
comptime NTD: Int = MMM.MAX_TENDON
comptime NSITE: Int = MMM.NSITE
comptime NEXCL: Int = MMM.NEXCLUDE
comptime MC: Int = MMM.MAX_CONTACTS
comptime NMESHV: Int = 64
comptime BATCH: Int = 1

comptime NGROUP: Int = 5
comptime NPOSE: Int = 24

# float32 through FK, the hull transform, GJK/EPA and the clip. The sibling's
# float64 residuals are ~1e-16, so anything here is float32 noise, not physics.
comptime TOL: Float64 = 1e-4

comptime Dat = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]
comptime Mod = Model[
    DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, NMESHV,
]


def _stack_z(g: Int) -> Float64:
    """Face-to-face separation of group `g`'s two geoms when aligned."""
    if g == 0:
        return 0.05 + 0.06
    if g == 1:
        return 0.06 + 0.05
    if g == 2:
        return 0.05 + 0.05
    if g == 3:
        return 0.08 + 0.06
    return 0.08 + 0.05


struct Lcg(Copyable, Movable):
    """A fixed 64-bit LCG, so a failing pose index is reproducible."""

    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Float64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float64((self.s >> 11) & 0x1FFFFFFFFFFFFF) / 9007199254740992.0

    def sym(mut self, a: Float64) -> Float64:
        return (self.next() * 2.0 - 1.0) * a


def test_mesh_manifold_cpu_vs_gpu() raises:
    """Same fixture, same poses, CPU vs GPU — records must agree."""
    print("--- mesh manifold CPU vs GPU:", NPOSE, "poses x", NGROUP, "groups")
    assert_true(
        MC_ENABLED,
        "MC_ENABLED is False — the clipper is compiled out and this file"
        " would pass without ever exercising it",
    )

    var ctx = DeviceContext()
    var mf = Mod()
    MMM.init_fields[DTYPE, NMESHV](ctx, mf)
    var d = Dat()
    var rng = Lcg(0x9E3779B97F4A7C15)

    var worst_pos = Float64(0)
    var worst_dist = Float64(0)
    var worst_dir = Float64(0)
    var worst_pose = -1
    var cnt_bad = 0
    var total_cpu = 0
    var total_gpu = 0
    # Poses where SOME pair produced more than one point, i.e. where the
    # clipper genuinely ran. Asserted below — without it a fixture that emits
    # one point per pair passes this file while testing nothing.
    var multi_poses = 0

    for p in range(NPOSE):
        MMM.reset_data(d)
        for g in range(NGROUP):
            var qo = g * 7
            # Regimes 0 and 1 dominate on purpose: they are the ones that reach
            # the face/face branch and so the clipper.
            var regime = p % 3
            var ang: Float64
            if regime == 0:
                ang = 0.0
            elif regime == 1:
                ang = rng.sym(0.008)
            else:
                ang = rng.sym(0.09)

            var pen = 0.002 + 0.003 * rng.next()
            var px = Float64(g) * 2.0 + rng.sym(0.01)
            var py = rng.sym(0.01)
            var pz = 0.5 + _stack_z(g) - pen

            var ax = rng.sym(1.0)
            var ay = rng.sym(1.0)
            var az = rng.sym(1.0)
            var an = sqrt(ax * ax + ay * ay + az * az)
            if an < 1e-9:
                ax = 1.0
                ay = 0.0
                az = 0.0
                an = 1.0
            var s = sin(0.5 * ang) / an
            var qw = cos(0.5 * ang)
            var qx = ax * s
            var qy = ay * s
            var qz = az * s

            # ⚠ Free-joint qpos is (x, y, z, qw, qx, qy, qz).
            d.qpos.data[qo + 0] = Scalar[DTYPE](px)
            d.qpos.data[qo + 1] = Scalar[DTYPE](py)
            d.qpos.data[qo + 2] = Scalar[DTYPE](pz)
            d.qpos.data[qo + 3] = Scalar[DTYPE](qw)
            d.qpos.data[qo + 4] = Scalar[DTYPE](qx)
            d.qpos.data[qo + 5] = Scalar[DTYPE](qy)
            d.qpos.data[qo + 6] = Scalar[DTYPE](qz)

        # ---- CPU
        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)
        var ncc = Int(d.meta.data[META_IDX_NUM_CONTACTS])
        var cpu = List[Float64]()
        for c in range(ncc):
            for k in range(CONTACT_SIZE):
                cpu.append(Float64(d.contacts.data[c * CONTACT_SIZE + k]))
        if ncc > NGROUP:
            multi_poses += 1

        # ---- GPU, same qpos. `upload_all` pushes the pose that the CPU leg
        # just ran on, so the two legs cannot silently diverge on their input.
        d.upload_all(ctx)
        forward_kinematics[
            "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
            NEXCL, NMESHV, BATCH,
        ](d, mf, ctx)
        detect_contacts[
            "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
            NEXCL, NMESHV, BATCH,
        ](d, mf, ctx)
        d.contacts.download(ctx)
        d.meta.download(ctx)
        var ncg = Int(d.meta.data[META_IDX_NUM_CONTACTS])

        total_cpu += ncc
        total_gpu += ncg
        if ncc != ncg:
            cnt_bad += 1
            print("  pose", p, " count CPU", ncc, " GPU", ncg)
            continue

        # ⚠ MATCH AS A SET, NOT POSITIONALLY. The two phases can order a pair
        # the opposite way round, which negates the normal and swaps the body
        # ids; comparing row i to row i reports that as a 2.0 normal error.
        # That artefact already cost this arc one false alarm.
        for i in range(ncc):
            var bj = -1
            var bd = Float64(1e30)
            for j in range(ncg):
                var jo = j * CONTACT_SIZE
                var ex = Float64(d.contacts.data[jo + CONTACT_IDX_POS_X]) \
                    - cpu[i * CONTACT_SIZE + CONTACT_IDX_POS_X]
                var ey = Float64(d.contacts.data[jo + CONTACT_IDX_POS_Y]) \
                    - cpu[i * CONTACT_SIZE + CONTACT_IDX_POS_Y]
                var ez = Float64(d.contacts.data[jo + CONTACT_IDX_POS_Z]) \
                    - cpu[i * CONTACT_SIZE + CONTACT_IDX_POS_Z]
                var dd = sqrt(ex * ex + ey * ey + ez * ez)
                if dd < bd:
                    bd = dd
                    bj = j
            if bj < 0:
                continue
            if bd > worst_pos:
                worst_pos = bd
                worst_pose = p

            var jo = bj * CONTACT_SIZE
            var ed = abs(
                Float64(d.contacts.data[jo + CONTACT_IDX_DIST])
                - cpu[i * CONTACT_SIZE + CONTACT_IDX_DIST]
            )
            if ed > worst_dist:
                worst_dist = ed

            # The normal may be recorded in either body order; both legs run
            # the same phase, so take the better of n and -n rather than
            # asserting an ordering this file does not control.
            var an0 = Float64(0)
            var an1 = Float64(0)
            for k in range(3):
                var kk = CONTACT_IDX_NX + k
                var gv = Float64(d.contacts.data[jo + kk])
                var cv = cpu[i * CONTACT_SIZE + kk]
                if abs(gv - cv) > an0:
                    an0 = abs(gv - cv)
                if abs(gv + cv) > an1:
                    an1 = abs(gv + cv)
            var ndiff = an0 if an0 < an1 else an1
            if ndiff > worst_dir:
                worst_dir = ndiff

    print("  contacts CPU", total_cpu, " GPU", total_gpu)
    print("  poses with a multi-point manifold:", multi_poses, "/", NPOSE)
    print("  worst |dpos|", worst_pos, " at pose", worst_pose)
    print("  worst |ddist|", worst_dist, "  worst |dn|", worst_dir)

    assert_true(
        multi_poses > 0,
        "no pose produced more than one contact per pair, so the clipper never"
        " ran and this comparison is vacuous — the pose schedule is broken,"
        " not the engine",
    )
    assert_true(
        cnt_bad == 0,
        String("CPU and GPU disagree on the contact COUNT on ")
        + String(cnt_bad) + " of " + String(NPOSE) + " poses",
    )
    assert_true(
        worst_pos <= TOL,
        String("worst CPU-vs-GPU contact position error ") + String(worst_pos)
        + " m at pose " + String(worst_pose) + " — the clipper's runtime-indexed"
        " per-thread arrays are defect 27's shape, check there first",
    )
    assert_true(
        worst_dist <= TOL and worst_dir <= TOL,
        String("worst CPU-vs-GPU depth ") + String(worst_dist)
        + " / normal " + String(worst_dir),
    )


def test_sap_matches_on2_on_mesh_manifolds() raises:
    """SAP and the O(N^2) phase must agree — both now dispatch the clipper.

    The multicontact dispatch is written TWICE, once in `contact_detection` and
    once in `broadphase_sap`, so the two can drift. This runs both phases over
    the same poses and compares their contact SETS.

    ⚠ AS SETS, NOT POSITIONALLY. SAP can order a pair the opposite way round to
    the O(N^2) phase, which swaps body_a/body_b and negates the normal. An
    earlier positional version of this probe reported `worst |d record| = 2.0`
    and sent me looking for a sign bug in the clipper that did not exist.
    """
    print("--- SAP vs O(N^2) on mesh manifolds:", NPOSE, "poses")
    var ctx = DeviceContext()
    var mf = Mod()
    MMM.init_fields[DTYPE, NMESHV](ctx, mf)
    var d = Dat()
    var rng = Lcg(0x9E3779B97F4A7C15)

    var worst_pos = Float64(0)
    var worst_pose = -1
    var cnt_bad = 0
    var total = 0
    var multi_poses = 0

    for p in range(NPOSE):
        MMM.reset_data(d)
        for g in range(NGROUP):
            var qo = g * 7
            var regime = p % 3
            var ang: Float64
            if regime == 0:
                ang = 0.0
            elif regime == 1:
                ang = rng.sym(0.008)
            else:
                ang = rng.sym(0.09)

            var pen = 0.002 + 0.003 * rng.next()
            var px = Float64(g) * 2.0 + rng.sym(0.01)
            var py = rng.sym(0.01)
            var pz = 0.5 + _stack_z(g) - pen

            var ax = rng.sym(1.0)
            var ay = rng.sym(1.0)
            var az = rng.sym(1.0)
            var an = sqrt(ax * ax + ay * ay + az * az)
            if an < 1e-9:
                ax = 1.0
                ay = 0.0
                az = 0.0
                an = 1.0
            var s = sin(0.5 * ang) / an
            var qw = cos(0.5 * ang)
            var qx = ax * s
            var qy = ay * s
            var qz = az * s

            d.qpos.data[qo + 0] = Scalar[DTYPE](px)
            d.qpos.data[qo + 1] = Scalar[DTYPE](py)
            d.qpos.data[qo + 2] = Scalar[DTYPE](pz)
            d.qpos.data[qo + 3] = Scalar[DTYPE](qw)
            d.qpos.data[qo + 4] = Scalar[DTYPE](qx)
            d.qpos.data[qo + 5] = Scalar[DTYPE](qy)
            d.qpos.data[qo + 6] = Scalar[DTYPE](qz)

        forward_kinematics["cpu"](d, mf)

        detect_contacts["cpu"](d, mf)
        var n2 = Int(d.meta.data[META_IDX_NUM_CONTACTS])
        var n2_rows = List[Float64]()
        for c in range(n2):
            for k in range(CONTACT_SIZE):
                n2_rows.append(Float64(d.contacts.data[c * CONTACT_SIZE + k]))
        if n2 > NGROUP:
            multi_poses += 1

        # Same `Data`, same poses — only the broadphase differs.
        detect_contacts_sap["cpu"](d, mf)
        var ns = Int(d.meta.data[META_IDX_NUM_CONTACTS])
        total += n2
        if n2 != ns:
            cnt_bad += 1
            print("  pose", p, " count O(N^2)", n2, " SAP", ns)
            continue

        for i in range(n2):
            var bd = Float64(1e30)
            for j in range(ns):
                var jo = j * CONTACT_SIZE
                var ex = Float64(d.contacts.data[jo + CONTACT_IDX_POS_X]) \
                    - n2_rows[i * CONTACT_SIZE + CONTACT_IDX_POS_X]
                var ey = Float64(d.contacts.data[jo + CONTACT_IDX_POS_Y]) \
                    - n2_rows[i * CONTACT_SIZE + CONTACT_IDX_POS_Y]
                var ez = Float64(d.contacts.data[jo + CONTACT_IDX_POS_Z]) \
                    - n2_rows[i * CONTACT_SIZE + CONTACT_IDX_POS_Z]
                var dd = sqrt(ex * ex + ey * ey + ez * ez)
                if dd < bd:
                    bd = dd
            if bd > worst_pos:
                worst_pos = bd
                worst_pose = p

    print("  contacts O(N^2)", total, " count mismatches", cnt_bad, "/", NPOSE)
    print("  poses with a multi-point manifold:", multi_poses, "/", NPOSE)
    print("  worst |dpos|", worst_pos, " at pose", worst_pose)

    assert_true(
        multi_poses > 0,
        "no multi-point manifold, so the clipper never ran in either phase and"
        " this comparison is vacuous",
    )
    assert_true(
        cnt_bad == 0,
        String("SAP and the O(N^2) phase disagree on the contact COUNT on ")
        + String(cnt_bad) + " of " + String(NPOSE) + " poses — the multicontact"
        " dispatch is written twice and the two have drifted",
    )
    assert_true(
        worst_pos <= TOL,
        String("worst SAP-vs-O(N^2) contact position error ")
        + String(worst_pos) + " m at pose " + String(worst_pose),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
