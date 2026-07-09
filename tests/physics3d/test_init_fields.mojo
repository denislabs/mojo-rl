"""Stage-B gate: MODEL_DEF.init_fields (fields-native model build) parity.

Part A (Walker2D, non-mesh): init_fields must produce field tensors BIT-EXACT
to the legacy init_model_gpu -> load_from_slab path (same records, no slab kept
by the caller).

Part B (SawyerReach, mesh): init_fields must populate mesh_meta + mesh_verts
CORRECTLY — the legacy init_model_gpu under-sized mesh models (buffer built
without the NMESH_VERTS padding), so this path is the fix. Non-vacuity: a mesh
is present and its verts are non-zero.

Model-build only (no physics kernels) — light on the GPU.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_init_fields.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import ModelFields
from mojo_rl.physics3d.gpu.constants import (
    model_size_with_invweight,
    MODEL_MESH_META_SIZE,
    MAX_GPU_MESHES,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.metaworld.sawyer_reach_xml import SawyerReachModel

comptime DT = DType.float32


def _cmp(name: String, a: List[Scalar[DT]], b: List[Scalar[DT]]) raises -> Int:
    if len(a) != len(b):
        print("  ", name, ": LEN mismatch", len(a), "vs", len(b))
        return 1
    var bad = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            if bad < 3:
                print("  ", name, "[", i, "]:", a[i], "vs", b[i])
            bad += 1
    return bad


def main() raises:
    print("=== Stage-B init_fields parity ===")
    var ctx = DeviceContext()

    # ── Part A: Walker2D, init_fields vs legacy slab path ────────────────────
    comptime W_NV = Walker2dModel.NV
    comptime W_NBODY = Walker2dModel.NBODY
    comptime W_NJOINT = Walker2dModel.NJOINT
    comptime W_NGEOM = Walker2dModel.NGEOM
    comptime W_NEQ = Walker2dModel.MAX_EQUALITY
    comptime W_NTD = Walker2dModel.MAX_TENDON
    comptime W_NSITE = Walker2dModel.NSITE
    comptime W_NEXCL = Walker2dModel.nexclude
    comptime W_MS = model_size_with_invweight[W_NBODY, W_NJOINT, W_NV, W_NGEOM]()

    var mf_new = ModelFields[
        DT, W_NV, W_NBODY, W_NJOINT, W_NGEOM, W_NEQ, W_NTD, W_NSITE, W_NEXCL, 0
    ]()
    Walker2dModel.init_fields[DT, 0](ctx, mf_new)

    var model_t = TensorImpl[DT].alloc(W_MS)
    model_t.upload(ctx)
    Walker2dModel.init_model_gpu(ctx, model_t.dev.value())
    model_t.download(ctx)
    var mf_old = ModelFields[
        DT, W_NV, W_NBODY, W_NJOINT, W_NGEOM, W_NEQ, W_NTD, W_NSITE, W_NEXCL, 0
    ]()
    mf_old.load_from_slab(model_t.data)

    var bad = 0
    bad += _cmp("bodies", mf_new.bodies.data, mf_old.bodies.data)
    bad += _cmp("joints", mf_new.joints.data, mf_old.joints.data)
    bad += _cmp("geoms", mf_new.geoms.data, mf_old.geoms.data)
    bad += _cmp("meta", mf_new.meta.data, mf_old.meta.data)
    bad += _cmp("tendons", mf_new.tendons.data, mf_old.tendons.data)
    bad += _cmp("body_invw0", mf_new.body_invweight0.data, mf_old.body_invweight0.data)
    bad += _cmp("dof_invw0", mf_new.dof_invweight0.data, mf_old.dof_invweight0.data)
    if bad != 0:
        raise Error("Walker2D init_fields != legacy path (" + String(bad) + ")")
    print("  Part A PASS: Walker2D init_fields == legacy path BIT-EXACT")

    # ── Part B: SawyerReach mesh model, init_fields correctly sizes mesh ──────
    comptime S_NV = SawyerReachModel.NV
    comptime S_NBODY = SawyerReachModel.NBODY
    comptime S_NJOINT = SawyerReachModel.NJOINT
    comptime S_NGEOM = SawyerReachModel.NGEOM
    comptime S_NEQ = SawyerReachModel.MAX_EQUALITY
    comptime S_NTD = SawyerReachModel.MAX_TENDON
    comptime S_NSITE = SawyerReachModel.NSITE
    comptime S_NEXCL = SawyerReachModel.nexclude
    comptime NMESHV = MAX_GPU_MESHES * 256

    var mf_s = ModelFields[
        DT, S_NV, S_NBODY, S_NJOINT, S_NGEOM, S_NEQ, S_NTD, S_NSITE, S_NEXCL,
        NMESHV,
    ]()
    SawyerReachModel.init_fields[DT, NMESHV](ctx, mf_s)

    var num_meshes = 0
    for m in range(MAX_GPU_MESHES):
        var nverts = Int(mf_s.mesh_meta.data[m * MODEL_MESH_META_SIZE + 1])
        if nverts > 0:
            num_meshes += 1
    var nonzero_verts = 0
    for i in range(NMESHV * 3):
        if mf_s.mesh_verts.data[i] != Scalar[DT](0):
            nonzero_verts += 1
    print("  sawyer meshes:", num_meshes, " non-zero mesh_vert entries:", nonzero_verts)
    if num_meshes == 0 or nonzero_verts == 0:
        raise Error("Part B: mesh not populated by init_fields (sizing bug?)")
    print("  Part B PASS: SawyerReach mesh populated + correctly sized")

    print("test_init_fields: ALL PASS")
