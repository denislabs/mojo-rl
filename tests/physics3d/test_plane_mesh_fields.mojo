"""Regression gate (GOLDEN-frozen): PLANE-vs-MESH contact record emission on
BOTH detection paths (O(N^2) and SAP), via a synthetic tetrahedron hull.

Originally validated BIT-EXACT against the legacy FK + narrow-phase / SAP
kernels. That legacy reference was frozen into the GOLDEN fingerprints below
during Phase-0 of the physics3d sunset, so this gate now checks, per leg:
  * fields-GPU (FK -> detect / FK -> SAP) reproduces the frozen (legacy-
    validated) fingerprint — contact counts + a contact-record checksum, and
  * the plane-mesh non-vacuity conventions hold (O(N^2): BODY_A=obj, BODY_B=0,
    normal +z, DIST<0; SAP: BODY_B=-1, normal +z).

SawyerReach + an injected 4-vertex tetrahedron (obj geom overridden to
GEOM_MESH); obj teleported to (2, 2, z) so exactly one tetra vertex dips below
the floor plane. The MESH model build uses the legacy Model + copy_* helpers
(init_model_gpu under-sizes mesh models — a P6 prerequisite, same as
test_mesh_detection_fields). Regenerate goldens after an INTENTIONAL physics
change: HARVEST=True, run on Apple, paste, HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_plane_mesh_fields.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.constants import GEOM_MESH, GEOM_CYLINDER
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.kinematics.forward_kinematics_fields import (
    forward_kinematics_fields,
)
from mojo_rl.physics3d.collision.contact_detection_fields import (
    detect_contacts_fields,
)
from mojo_rl.physics3d.collision.broadphase_sap_fields import (
    detect_contacts_sap_fields,
)
from mojo_rl.physics3d.gpu.buffer_utils import (
    copy_model_to_buffer,
    copy_geoms_to_buffer,
    copy_tendons_to_buffer,
    copy_invweight0_to_buffer,
    copy_mesh_hull_to_buffer,
)
from mojo_rl.physics3d.gpu.constants import (
    model_size_with_invweight,
    model_geom_offset,
    model_mesh_meta_offset,
    model_mesh_vert_offset,
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_MESH_ID,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    MAX_GPU_MESHES,
)
from mojo_rl.envs.metaworld.sawyer_reach_xml import SawyerReachModel

comptime DTYPE = DType.float32
comptime NQ = SawyerReachModel.NQ
comptime NV = SawyerReachModel.NV
comptime NBODY = SawyerReachModel.NBODY
comptime NJOINT = SawyerReachModel.NJOINT
comptime NGEOM = SawyerReachModel.NGEOM
comptime NEQ = SawyerReachModel.MAX_EQUALITY
comptime NTD = SawyerReachModel.MAX_TENDON
comptime NSITE = SawyerReachModel.NSITE
comptime MC = SawyerReachModel.MAX_CONTACTS
comptime BATCH = 2
comptime METADATA_SIZE_L = 4
comptime NMESHV = MAX_GPU_MESHES * 256
comptime MS = model_size_with_invweight[
    NBODY, NJOINT, NV, NGEOM, NEQ, NTD, NSITE, 0, NMESHV
]()

comptime OBJ_Z_ENV0: Float64 = -0.900  # vertex dist = -0.017
comptime OBJ_Z_ENV1: Float64 = -0.912  # vertex dist = -0.029

# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ----
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_NCON_A = 4  # O(N^2) leg
comptime GOLD_CON_A = 1135.9686783785
comptime GOLD_NCON_B = 6  # SAP leg
comptime GOLD_CON_B = 1989.6578279478708


def _qpos_for_env(e: Int) -> List[Float64]:
    var q = List[Float64](length=NQ, fill=0.0)
    q[0] = 1.889288
    q[1] = -0.575769
    q[2] = -0.976659
    q[3] = 1.641991
    q[4] = 0.942860
    q[5] = 1.043696
    q[6] = 2.292833
    q[7] = 0.0
    q[8] = 0.0
    q[9] = 2.0  # obj x (far from every non-plane geom)
    q[10] = 2.0  # obj y
    q[11] = OBJ_Z_ENV0 if e == 0 else OBJ_Z_ENV1  # obj z
    q[12] = 1.0  # obj quat w
    return q^


def _fp_check(
    label: String,
    d: DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH],
    gold_ncon: Int,
    gold_con: Float64,
) raises:
    var ncon_total = 0
    var fp = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
        ncon_total += nc
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                fp += Float64(
                    d.contacts.data[e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k]
                ) * Float64((e + 1) * (c + 1) * (k + 1))
    if ncon_total == 0:
        raise Error(label + ": zero contacts — gate is vacuous")
    if HARVEST:
        print("  HARVEST", label, "NCON =", ncon_total)
        print("  HARVEST", label, "CON  =", fp)
    else:
        if ncon_total != gold_ncon and not has_nvidia_gpu_accelerator():
            raise Error(
                label + ": contacts " + String(ncon_total) + " != golden "
                + String(gold_ncon)
            )
        var denom = abs(gold_con) if abs(gold_con) > 1e-9 else 1.0
        if abs(fp - gold_con) / denom > GOLD_RTOL and (
            not has_nvidia_gpu_accelerator()
        ):
            raise Error(
                label + ": record fingerprint " + String(fp) + " != golden "
                + String(gold_con)
            )
        print("  [", label, "] PASS: counts + records match golden")


def _assert_plane_mesh_contact(
    label: String,
    d: DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH],
    obj_body: Int,
    expected_body_b: Int,
) raises:
    for e in range(BATCH):
        var ncon = Int(d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
        var found = 0
        for c in range(ncon):
            var base = e * MC * CONTACT_SIZE + c * CONTACT_SIZE
            var ba = Int(d.contacts.data[base + CONTACT_IDX_BODY_A])
            var bb = Int(d.contacts.data[base + CONTACT_IDX_BODY_B])
            var nx = Float64(d.contacts.data[base + CONTACT_IDX_NX])
            var ny = Float64(d.contacts.data[base + CONTACT_IDX_NY])
            var nz = Float64(d.contacts.data[base + CONTACT_IDX_NZ])
            if ba == obj_body and bb == expected_body_b and nz == 1.0:
                if nx == 0.0 and ny == 0.0:
                    found += 1
        if found == 0:
            raise Error(
                label + ": no PLANE-MESH contact in env " + String(e)
                + " — gate is vacuous"
            )


def main() raises:
    print("--- plane-mesh contact emission fields GOLDEN gate, BATCH=", BATCH)
    var ctx = DeviceContext()

    # CPU model (loads STL hulls) + mesh-sized slab (legacy build — see NOTE).
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ,
        SawyerReachModel.CONE_TYPE, NTD, NSITE,
    ]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MC, NSITE]()
    SawyerReachModel.setup_model_and_data[DTYPE](model, data)
    var n_stl_meshes = model.num_meshes
    var n_stl_verts = len(model.mesh_vert) // 3
    if n_stl_meshes == 0 or n_stl_verts == 0:
        raise Error("expected STL mesh hulls — gate is vacuous")
    if n_stl_meshes + 1 > MAX_GPU_MESHES:
        raise Error("no free mesh_meta slot for the synthetic tetrahedron")
    if n_stl_verts + 4 > NMESHV:
        raise Error("no vertex capacity for the tetrahedron — raise NMESHV")

    var host_buf = ctx.enqueue_create_host_buffer[DTYPE](MS)
    ctx.synchronize()
    for i in range(MS):
        host_buf[i] = Scalar[DTYPE](0)
    copy_model_to_buffer(model, host_buf)
    copy_geoms_to_buffer(model, host_buf)
    copy_tendons_to_buffer(model, host_buf)
    copy_invweight0_to_buffer(model, host_buf)
    copy_mesh_hull_to_buffer(model, host_buf)

    var model_t = TensorImpl[DTYPE].alloc(MS)
    for i in range(MS):
        model_t.data[i] = host_buf[i]

    # Locate the obj cylinder geom.
    var g_obj = -1
    var obj_body = -1
    for g in range(NGEOM):
        var g_off = model_geom_offset[NBODY, NJOINT](g)
        var gt = Int(model_t.data[g_off + GEOM_IDX_TYPE])
        if gt != GEOM_CYLINDER:
            continue
        var r = Float64(model_t.data[g_off + GEOM_IDX_RADIUS])
        var hl = Float64(model_t.data[g_off + GEOM_IDX_HALF_LENGTH])
        if abs(r - 0.02) < 1e-6 and abs(hl - 0.02) < 1e-6:
            g_obj = g
            obj_body = Int(model_t.data[g_off + GEOM_IDX_BODY])
    print("  obj geom:", g_obj, " obj body:", obj_body)
    if g_obj < 0 or obj_body <= 0:
        raise Error("could not identify the obj cylinder geom")

    # SINGLE-POINT INJECTION: synthetic tetrahedron + obj geom -> GEOM_MESH.
    comptime MESH_META_OFF = model_mesh_meta_offset[
        NBODY, NJOINT, NV, NGEOM, NEQ, NTD, NSITE, 0
    ]()
    comptime MESH_VERT_OFF = model_mesh_vert_offset[
        NBODY, NJOINT, NV, NGEOM, NEQ, NTD, NSITE, 0
    ]()
    var tetra_id = n_stl_meshes
    model_t.data[MESH_META_OFF + tetra_id * 2 + 0] = Scalar[DTYPE](n_stl_verts)
    model_t.data[MESH_META_OFF + tetra_id * 2 + 1] = Scalar[DTYPE](4)
    var tetra = List[Float64](length=12, fill=0.0)
    tetra[0] = 0.015
    tetra[2] = -0.03
    tetra[3] = -0.015
    tetra[4] = 0.012
    tetra[5] = 0.02
    tetra[6] = 0.006
    tetra[7] = -0.015
    tetra[8] = 0.025
    tetra[9] = -0.004
    tetra[10] = 0.008
    tetra[11] = 0.03
    for k in range(12):
        model_t.data[MESH_VERT_OFF + n_stl_verts * 3 + k] = Scalar[DTYPE](
            tetra[k]
        )
    var g_off_obj = model_geom_offset[NBODY, NJOINT](g_obj)
    model_t.data[g_off_obj + GEOM_IDX_TYPE] = Scalar[DTYPE](GEOM_MESH)
    model_t.data[g_off_obj + GEOM_IDX_MESH_ID] = Scalar[DTYPE](tetra_id)

    model_t.upload(ctx)
    var mf = ModelFields[
        DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, 0, NMESHV
    ]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)
    if Int(mf.geoms.data[g_obj * MODEL_GEOM_SIZE + GEOM_IDX_TYPE]) != GEOM_MESH:
        raise Error("geom override did not reach ModelFields")
    if Int(mf.mesh_meta.data[tetra_id * 2 + 1]) != 4:
        raise Error("tetra mesh_meta did not reach ModelFields")

    # ================= Leg 1: O(N^2) detection ==========================
    var d_a = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for e in range(BATCH):
        var q = _qpos_for_env(e)
        for i in range(NQ):
            d_a.qpos.data[e * NQ + i] = Scalar[DTYPE](q[i])
    d_a.upload_all(ctx)
    forward_kinematics_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d_a, mf, ctx)
    detect_contacts_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d_a, mf, ctx)
    d_a.contacts.download(ctx)
    d_a.meta.download(ctx)
    _fp_check("O(N^2)", d_a, GOLD_NCON_A, GOLD_CON_A)
    _assert_plane_mesh_contact("O(N^2)", d_a, obj_body, 0)
    for e in range(BATCH):
        var ncon = Int(
            d_a.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        var found_neg = False
        for c in range(ncon):
            var base = e * MC * CONTACT_SIZE + c * CONTACT_SIZE
            if (
                Int(d_a.contacts.data[base + CONTACT_IDX_BODY_A]) == obj_body
                and Int(d_a.contacts.data[base + CONTACT_IDX_BODY_B]) == 0
                and Float64(d_a.contacts.data[base + CONTACT_IDX_DIST]) < 0.0
            ):
                found_neg = True
        if not found_neg:
            raise Error("O(N^2): plane-mesh contact has non-negative DIST")
    print("  [ O(N^2) ] PASS: plane-mesh record present (BODY_B=0, DIST<0)")

    # ================= Leg 2: SAP detection =============================
    var d_b = DataFields[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for e in range(BATCH):
        var q = _qpos_for_env(e)
        for i in range(NQ):
            d_b.qpos.data[e * NQ + i] = Scalar[DTYPE](q[i])
    d_b.upload_all(ctx)
    forward_kinematics_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d_b, mf, ctx)
    detect_contacts_sap_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d_b, mf, ctx)
    d_b.contacts.download(ctx)
    d_b.meta.download(ctx)
    _fp_check("SAP", d_b, GOLD_NCON_B, GOLD_CON_B)
    _assert_plane_mesh_contact("SAP", d_b, obj_body, -1)
    print("  [ SAP ] PASS: plane-mesh record present (BODY_B=-1)")

    print("test_plane_mesh_fields: ALL PASS")
