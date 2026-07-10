"""G4 gate (Sawyer leg): spec-direct fields build vs legacy two-hop build.

SawyerReach is the feature-complete acid test: mesh convex hulls (STL load +
shared-mesh remap), weld equality, mocap body, contact excludes, 36 bodies.
Split from test_fields_build_ab.mojo because its comptime XML parse dominates
compile time.

Same contract: every record tensor bit-exact except `sites` (legacy left
them zero — latent gap; the new build fills them).

Run: pixi run mojo run -I . tests/physics3d/test_fields_build_ab_sawyer.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import ModelFields
from mojo_rl.envs.metaworld.sawyer_reach_xml import SawyerReachModel


def _cmp[
    DTYPE: DType
](name: String, a: TensorImpl[DTYPE], b: TensorImpl[DTYPE]) raises -> Int:
    if len(a.data) != len(b.data):
        print("    MISMATCH", name, ": len", len(a.data), "vs", len(b.data))
        return 1
    var bad = 0
    for i in range(len(a.data)):
        if a.data[i] != b.data[i]:
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


def main() raises:
    comptime M = SawyerReachModel
    comptime NMV = 16 * 256  # MAX_GPU_MESHES * 256 (mesh model)
    var ctx = DeviceContext()
    var mf_old = ModelFields[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, NMV,
    ]()
    M.init_fields[DType.float64, NMV](ctx, mf_old)
    var mf_new = ModelFields[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, NMV,
    ]()
    M.init_fields_v2[DType.float64, NMV](ctx, mf_new)

    print("--- SawyerReach ---")
    var bad = 0
    bad += _cmp("bodies", mf_old.bodies, mf_new.bodies)
    bad += _cmp("joints", mf_old.joints, mf_new.joints)
    bad += _cmp("meta", mf_old.meta, mf_new.meta)
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

    var new_nonzero = 0
    for i in range(len(mf_new.sites.data)):
        if mf_new.sites.data[i] != 0:
            new_nonzero += 1
    print("    sites: new nonzero elems =", new_nonzero)

    if bad > 0:
        raise Error("sawyer fields build A/B mismatches: " + String(bad))
    print("test_fields_build_ab_sawyer: ALL PASS")
