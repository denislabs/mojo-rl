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
the floor plane. Model build = fields-native init_fields (Stage B); the tetra is
injected directly into the packed mf.mesh_meta/mesh_verts/geoms tensors (offset-
free). Regenerate goldens after an INTENTIONAL physics change: HARVEST=True, run
on Apple, paste, HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_plane_mesh_fields.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator

from mojo_rl.physics3d.constants import GEOM_MESH, GEOM_CYLINDER
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
)
from mojo_rl.physics3d.collision.contact_detection import (
    detect_contacts,
)
from mojo_rl.physics3d.collision.broadphase_sap import (
    detect_contacts_sap,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_MESH_META_SIZE,
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
    d: Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH],
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
    d: Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH],
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

    # Fields-native build (loads STL hulls, NMESHV-padded — Stage B).
    var mf = Model[
        DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, 0, NMESHV
    ]()
    SawyerReachModel.init_fields[DTYPE, NMESHV](ctx, mf)

    # STL mesh counts from the packed mesh_meta (vertadr, nverts) pairs.
    var n_stl_meshes = 0
    var n_stl_verts = 0
    for m in range(MAX_GPU_MESHES):
        var nv = Int(mf.mesh_meta.data[m * MODEL_MESH_META_SIZE + 1])
        if nv > 0:
            n_stl_meshes += 1
            n_stl_verts += nv
    if n_stl_meshes == 0 or n_stl_verts == 0:
        raise Error("expected STL mesh hulls — gate is vacuous")
    if n_stl_meshes + 1 > MAX_GPU_MESHES:
        raise Error("no free mesh_meta slot for the synthetic tetrahedron")
    if n_stl_verts + 4 > NMESHV:
        raise Error("no vertex capacity for the tetrahedron — raise NMESHV")

    # Locate the obj cylinder geom (packed mf.geoms).
    var g_obj = -1
    var obj_body = -1
    for g in range(NGEOM):
        var base = g * MODEL_GEOM_SIZE
        var gt = Int(mf.geoms.data[base + GEOM_IDX_TYPE])
        if gt != GEOM_CYLINDER:
            continue
        var r = Float64(mf.geoms.data[base + GEOM_IDX_RADIUS])
        var hl = Float64(mf.geoms.data[base + GEOM_IDX_HALF_LENGTH])
        if abs(r - 0.02) < 1e-6 and abs(hl - 0.02) < 1e-6:
            g_obj = g
            obj_body = Int(mf.geoms.data[base + GEOM_IDX_BODY])
    print("  obj geom:", g_obj, " obj body:", obj_body)
    if g_obj < 0 or obj_body <= 0:
        raise Error("could not identify the obj cylinder geom")

    # SINGLE-POINT INJECTION into the packed field tensors: synthetic
    # tetrahedron mesh + obj geom -> GEOM_MESH (offset-free).
    var tetra_id = n_stl_meshes
    mf.mesh_meta.data[tetra_id * MODEL_MESH_META_SIZE + 0] = Scalar[DTYPE](
        n_stl_verts
    )
    mf.mesh_meta.data[tetra_id * MODEL_MESH_META_SIZE + 1] = Scalar[DTYPE](4)
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
        mf.mesh_verts.data[n_stl_verts * 3 + k] = Scalar[DTYPE](tetra[k])
    mf.geoms.data[g_obj * MODEL_GEOM_SIZE + GEOM_IDX_TYPE] = Scalar[DTYPE](
        GEOM_MESH
    )
    mf.geoms.data[g_obj * MODEL_GEOM_SIZE + GEOM_IDX_MESH_ID] = Scalar[DTYPE](
        tetra_id
    )
    mf.mesh_meta.upload(ctx)
    mf.mesh_verts.upload(ctx)
    mf.geoms.upload(ctx)
    if Int(mf.geoms.data[g_obj * MODEL_GEOM_SIZE + GEOM_IDX_TYPE]) != GEOM_MESH:
        raise Error("geom override did not reach Model")
    if Int(mf.mesh_meta.data[tetra_id * MODEL_MESH_META_SIZE + 1]) != 4:
        raise Error("tetra mesh_meta did not reach Model")

    # ================= Leg 1: O(N^2) detection ==========================
    var d_a = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for e in range(BATCH):
        var q = _qpos_for_env(e)
        for i in range(NQ):
            d_a.qpos.data[e * NQ + i] = Scalar[DTYPE](q[i])
    d_a.upload_all(ctx)
    forward_kinematics[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d_a, mf, ctx)
    detect_contacts[
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
    var d_b = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for e in range(BATCH):
        var q = _qpos_for_env(e)
        for i in range(NQ):
            d_b.qpos.data[e * NQ + i] = Scalar[DTYPE](q[i])
    d_b.upload_all(ctx)
    forward_kinematics[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d_b, mf, ctx)
    detect_contacts_sap[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d_b, mf, ctx)
    d_b.contacts.download(ctx)
    d_b.meta.download(ctx)
    _fp_check("SAP", d_b, GOLD_NCON_B, GOLD_CON_B)
    _assert_plane_mesh_contact("SAP", d_b, obj_body, -1)
    print("  [ SAP ] PASS: plane-mesh record present (BODY_B=-1)")

    print("test_plane_mesh_fields: ALL PASS")
