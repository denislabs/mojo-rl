"""G4 gate: spec-direct fields model build vs the legacy two-hop build.

Compares every ModelFields record tensor produced by

  OLD: init_fields     (setup_model_and_data -> CPU Model -> load_from_model
                        -> fields invweight0)
  NEW: init_fields_v2  (parse_xml_full -> build_model_fields_from_flat
                        -> fields invweight0)

BIT-EXACTLY, for Hopper (ifg, hinge+slide), Ant (free joint), Reacher
(sites), and Humanoid (27-DOF, tendon capacity). The ONE intentional
difference: `sites` — the legacy build left the site records all-zero
(latent gap); the new build fills them, so sites are checked as
new-nonzero-where-old-zero instead of equal.

Run: pixi run mojo run -I . tests/physics3d/test_fields_build_ab.mojo
(Sawyer — mesh hulls + equality + mocap — is gated separately in
test_fields_build_ab_sawyer.mojo; its comptime parse is slow.)
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import ModelFields
from mojo_rl.envs.hopper.hopper_xml import HopperModel
from mojo_rl.envs.ant.ant_xml import AntModel
from mojo_rl.envs.reacher.reacher_xml import ReacherModel
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel


def _cmp[
    DTYPE: DType
](name: String, a: TensorImpl[DTYPE], b: TensorImpl[DTYPE]) raises -> Int:
    """Bit-exact tensor compare; prints the first few mismatches."""
    if len(a.data) != len(b.data):
        print("    MISMATCH", name, ": len", len(a.data), "vs", len(b.data))
        return 1
    var bad = 0
    for i in range(len(a.data)):
        if a.data[i] != b.data[i]:
            # NaN != NaN is fine here: the build never writes NaN; inf==inf.
            if bad < 5:
                print(
                    "    MISMATCH",
                    name,
                    "[",
                    i,
                    "]: old=",
                    Float64(a.data[i]),
                    " new=",
                    Float64(b.data[i]),
                )
            bad += 1
    return bad


def _run_hopper() raises -> Int:
    comptime M = HopperModel
    var ctx = DeviceContext()
    var mf_old = ModelFields[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DType.float64, 0](ctx, mf_old)
    var mf_new = ModelFields[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields_v2[DType.float64, 0](ctx, mf_new)
    return _compare_all(
        "Hopper", mf_old, mf_new, expect_sites=M.NSITE > 0
    )


def _run_ant() raises -> Int:
    comptime M = AntModel
    var ctx = DeviceContext()
    var mf_old = ModelFields[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DType.float64, 0](ctx, mf_old)
    var mf_new = ModelFields[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields_v2[DType.float64, 0](ctx, mf_new)
    return _compare_all("Ant", mf_old, mf_new, expect_sites=M.NSITE > 0)


def _run_reacher() raises -> Int:
    comptime M = ReacherModel
    var ctx = DeviceContext()
    var mf_old = ModelFields[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DType.float64, 0](ctx, mf_old)
    var mf_new = ModelFields[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields_v2[DType.float64, 0](ctx, mf_new)
    return _compare_all(
        "Reacher", mf_old, mf_new, expect_sites=M.NSITE > 0
    )


def _run_humanoid() raises -> Int:
    comptime M = HumanoidModel
    var ctx = DeviceContext()
    var mf_old = ModelFields[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DType.float64, 0](ctx, mf_old)
    var mf_new = ModelFields[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields_v2[DType.float64, 0](ctx, mf_new)
    return _compare_all(
        "Humanoid", mf_old, mf_new, expect_sites=M.NSITE > 0
    )


def _compare_all[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
    NEQUALITY: Int,
    NTENDON: Int,
    NSITE: Int,
    NEXCLUDE: Int,
    NMV: Int,
](
    name: String,
    mf_old: ModelFields[
        DTYPE, NV, NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON, NSITE, NEXCLUDE,
        NMV,
    ],
    mf_new: ModelFields[
        DTYPE, NV, NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON, NSITE, NEXCLUDE,
        NMV,
    ],
    expect_sites: Bool,
) raises -> Int:
    print("---", name, "---")
    var bad = 0
    bad += _cmp("bodies", mf_old.bodies, mf_new.bodies)
    bad += _cmp("joints", mf_old.joints, mf_new.joints)
    bad += _cmp("meta", mf_old.meta, mf_new.meta)
    bad += _cmp("curriculum", mf_old.curriculum, mf_new.curriculum)
    bad += _cmp("geoms", mf_old.geoms, mf_new.geoms)
    bad += _cmp("equality", mf_old.equality, mf_new.equality)
    bad += _cmp("tendons", mf_old.tendons, mf_new.tendons)
    bad += _cmp(
        "body_invweight0", mf_old.body_invweight0, mf_new.body_invweight0
    )
    bad += _cmp("dof_invweight0", mf_old.dof_invweight0, mf_new.dof_invweight0)
    bad += _cmp("excludes", mf_old.excludes, mf_new.excludes)
    bad += _cmp("mesh_meta", mf_old.mesh_meta, mf_new.mesh_meta)
    bad += _cmp("mesh_verts", mf_old.mesh_verts, mf_new.mesh_verts)

    # sites: legacy build left them zero (latent gap); the new build fills
    # them. Old must be all-zero; new must be nonzero for NSITE > 0.
    var old_nonzero = 0
    var new_nonzero = 0
    for i in range(len(mf_old.sites.data)):
        if mf_old.sites.data[i] != 0:
            old_nonzero += 1
        if mf_new.sites.data[i] != 0:
            new_nonzero += 1
    if old_nonzero > 0:
        print("    UNEXPECTED: legacy sites nonzero (", old_nonzero, ")")
        bad += 1
    if expect_sites and new_nonzero == 0:
        print("    MISSING: new build sites all-zero despite NSITE > 0")
        bad += 1
    print(
        "    sites: legacy zero (as before), new nonzero elems =", new_nonzero
    )

    if bad == 0:
        print("    OK — all record tensors bit-exact")
    return bad


def main() raises:
    var bad = 0
    bad += _run_hopper()
    bad += _run_ant()
    bad += _run_reacher()
    bad += _run_humanoid()
    if bad > 0:
        raise Error("fields build A/B mismatches: " + String(bad))
    print("test_fields_build_ab: ALL PASS")
