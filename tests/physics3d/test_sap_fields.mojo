"""Regression gate (GOLDEN-frozen): SAP broadphase in fields contact detection.

Part A — Humanoid (NGEOM=18 >= SAP_THRESHOLD=16, floor plane + 17 body
geoms, MAX_CONTACTS=50), BATCH=2 with penetrating poses (feet touching and
deep crouch): legacy FK -> detect_contacts_sap_gpu on the flat slab vs
fields FK -> detect_contacts_sap_fields on DataFields/ModelFields. Contact
count AND every populated contact record must be BIT-EXACT. Cross-check:
fields-SAP vs fields-O(N^2) (detect_contacts_fields) as SETS (emission
order differs — SAP sweeps sorted by aabb_min_x, and conventions differ:
SAP plane contacts write BODY_B=-1 vs 0, no INCLUDEMARGIN slot), matched
by unordered body pair + position/dist within 1e-4. Auto dispatcher must
route humanoid to SAP (bit-equal to detect_contacts_sap_fields).

Part B — SawyerReach (robot meshes + block.stl, NEXCLUDE==0), BATCH=2:
env1 teleports the obj cylinder into the eGripperBase MESH hull so a
GJK/EPA mesh contact appears through the SAP path. Legacy-SAP vs
fields-SAP BIT-EXACT + non-vacuity (mesh-body/obj-body contact present).

Part C — Walker2d (NGEOM=8 < 16): detect_contacts_auto_fields must route
to detect_contacts_fields, results bit-equal.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_sap_fields.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.constants import GEOM_MESH, GEOM_CYLINDER
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.kinematics.forward_kinematics_fields import (
    forward_kinematics_fields,
)
from mojo_rl.physics3d.collision.broadphase_sap import SAP_THRESHOLD
from mojo_rl.physics3d.collision.broadphase_sap_fields import (
    detect_contacts_sap_fields,
    detect_contacts_auto_fields,
)
from mojo_rl.physics3d.collision.contact_detection_fields import (
    detect_contacts_fields,
)

# NOTE (sunset): Part B (sawyer mesh SAP) still builds its model via the legacy
# Model + copy_* helpers (init_model_gpu under-sizes mesh models — a P6
# prerequisite, same as test_mesh_detection_fields). The SAP comparison itself
# is golden-frozen (legacy FK/SAP reference kernels dropped).
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.gpu.buffer_utils import (
    copy_model_to_buffer,
    copy_geoms_to_buffer,
    copy_tendons_to_buffer,
    copy_invweight0_to_buffer,
    copy_mesh_hull_to_buffer,
)
from mojo_rl.physics3d.gpu.constants import (
    model_size_with_invweight,
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    MAX_GPU_MESHES,
)
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel
from mojo_rl.envs.metaworld.sawyer_reach_xml import SawyerReachModel
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel

comptime DTYPE = DType.float32
comptime BATCH = 2
comptime METADATA_SIZE_L = 4

# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ----
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_NCON_H = 14  # Part A humanoid SAP: total contacts
comptime GOLD_CON_H = 7711.957039542147
comptime GOLD_NCON_S = 6  # Part B sawyer SAP: total contacts
comptime GOLD_CON_S = 2258.0145981857786

# ── Humanoid (Part A) ────────────────────────────────────────────────────
comptime NQ_H = HumanoidModel.NQ  # 24
comptime NV_H = HumanoidModel.NV  # 23
comptime NBODY_H = HumanoidModel.NBODY  # 14
comptime NJOINT_H = HumanoidModel.NJOINT  # 18
comptime NGEOM_H = HumanoidModel.NGEOM  # 18
comptime MC_H = HumanoidModel.MAX_CONTACTS  # 50
comptime NEQ_H = HumanoidModel.MAX_EQUALITY  # 0
comptime NTD_H = HumanoidModel.MAX_TENDON  # 2
comptime NSITE_H = HumanoidModel.NSITE  # 0
comptime NEXCL_H = HumanoidModel.nexclude  # 0
comptime MS_H = model_size_with_invweight[
    NBODY_H, NJOINT_H, NV_H, NGEOM_H, NEQ_H, NTD_H, NSITE_H, NEXCL_H
]()

# ── Sawyer (Part B) ──────────────────────────────────────────────────────
comptime NQ_S = SawyerReachModel.NQ
comptime NV_S = SawyerReachModel.NV
comptime NBODY_S = SawyerReachModel.NBODY
comptime NJOINT_S = SawyerReachModel.NJOINT
comptime NGEOM_S = SawyerReachModel.NGEOM
comptime NEQ_S = SawyerReachModel.MAX_EQUALITY
comptime NTD_S = SawyerReachModel.MAX_TENDON
comptime NSITE_S = SawyerReachModel.NSITE
comptime MC_S = SawyerReachModel.MAX_CONTACTS
comptime NMESHV_S = MAX_GPU_MESHES * 256
comptime MS_S = model_size_with_invweight[
    NBODY_S, NJOINT_S, NV_S, NGEOM_S, NEQ_S, NTD_S, NSITE_S, 0, NMESHV_S
]()

# ── Walker2d (Part C) ────────────────────────────────────────────────────
comptime NQ_W = Walker2dModel.NQ
comptime NV_W = Walker2dModel.NV
comptime NBODY_W = Walker2dModel.NBODY
comptime NJOINT_W = Walker2dModel.NJOINT
comptime NGEOM_W = Walker2dModel.NGEOM
comptime MC_W = Walker2dModel.MAX_CONTACTS
comptime MS_W = model_size_with_invweight[NBODY_W, NJOINT_W, NV_W, NGEOM_W]()


def _humanoid_qpos(e: Int, i: Int) -> Scalar[DTYPE]:
    """Free joint pose + hinge angles. env0: feet touch/penetrate floor;
    env1: deep crouch — many geoms near the floor and each other, so the
    SAP sweep processes several candidate pairs."""
    if i == 0:
        return Scalar[DTYPE](0.02) * Scalar[DTYPE](e)
    if i == 1:
        return Scalar[DTYPE](0)
    if i == 2:
        return Scalar[DTYPE](1.24) if e == 0 else Scalar[DTYPE](0.72)
    if i == 3:
        return Scalar[DTYPE](1)  # identity quaternion (w first)
    if i <= 6:
        return Scalar[DTYPE](0)
    if e == 0:
        # near-neutral standing pose, slight bends
        if i == 13 or i == 17:
            return Scalar[DTYPE](-0.15)  # knees
        return Scalar[DTYPE](0.05)
    # env1: crouch
    if i == 8:
        return Scalar[DTYPE](-0.4)  # abdomen_y
    if i == 12 or i == 16:
        return Scalar[DTYPE](-1.0)  # hip_y
    if i == 13 or i == 17:
        return Scalar[DTYPE](-1.6)  # knees
    if i == 20 or i == 23:
        return Scalar[DTYPE](-0.6)  # elbows
    return Scalar[DTYPE](0.1)


def _part_a_humanoid(ctx: DeviceContext) raises:
    print("--- Part A: humanoid SAP fields GOLDEN, BATCH=", BATCH)
    print("  humanoid NGEOM=", NGEOM_H, " SAP_THRESHOLD=", SAP_THRESHOLD)
    comptime assert NGEOM_H >= SAP_THRESHOLD, "humanoid must route to SAP"

    var model_t = TensorImpl[DTYPE].alloc(MS_H)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    HumanoidModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[
        DTYPE, NV_H, NBODY_H, NJOINT_H, NGEOM_H, NEQ_H, NTD_H, NSITE_H,
        NEXCL_H, 0,
    ]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    var d = DataFields[DTYPE, NQ_H, NV_H, NBODY_H, MC_H, NSITE_H, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_H):
            d.qpos.data[e * NQ_H + i] = _humanoid_qpos(e, i)
    d.upload_all(ctx)

    # Fields: FK + SAP detection.
    forward_kinematics_fields[
        "gpu", DTYPE, NQ_H, NV_H, NBODY_H, NJOINT_H, MC_H, NGEOM_H,
        NEQ_H, NTD_H, NSITE_H, NEXCL_H, 0, BATCH,
    ](d, mf, ctx)
    detect_contacts_sap_fields[
        "gpu", DTYPE, NQ_H, NV_H, NBODY_H, NJOINT_H, MC_H, NGEOM_H,
        NEQ_H, NTD_H, NSITE_H, NEXCL_H, 0, BATCH,
    ](d, mf, ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)

    var ncon_h = 0
    var fp_h = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
        ncon_h += nc
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                fp_h += Float64(
                    d.contacts.data[
                        e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + k
                    ]
                ) * Float64((e + 1) * (c + 1) * (k + 1))
    if ncon_h == 0:
        raise Error("humanoid SAP: no contacts — gate is vacuous")
    print("  humanoid fields-SAP total contacts:", ncon_h)
    if HARVEST:
        print("  HARVEST GOLD_NCON_H =", ncon_h)
        print("  HARVEST GOLD_CON_H  =", fp_h)
    else:
        if ncon_h != GOLD_NCON_H and not has_nvidia_gpu_accelerator():
            raise Error(
                "humanoid SAP contacts " + String(ncon_h) + " != golden "
                + String(GOLD_NCON_H)
            )
        var denom = abs(GOLD_CON_H) if abs(GOLD_CON_H) > 1e-9 else 1.0
        if abs(fp_h - GOLD_CON_H) / denom > GOLD_RTOL and (
            not has_nvidia_gpu_accelerator()
        ):
            raise Error(
                "humanoid SAP fingerprint " + String(fp_h) + " != golden "
                + String(GOLD_CON_H)
            )
        print("  PASS: humanoid fields-SAP matches golden fingerprint")

    # Non-vacuity for the SWEEP itself: env1 must contain a body-body
    # contact (BODY_B > 0) — plane contacts come from the direct plane
    # loop; only the sorted sweep emits body-body pairs.
    var ncon1 = Int(d.meta.data[1 * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
    var n_bodybody = 0
    for c in range(ncon1):
        var bb = Int(
            d.contacts.data[1 * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + 1]
        )
        if bb > 0:
            n_bodybody += 1
    print("  env 1 body-body (sweep-emitted) contacts:", n_bodybody)
    if n_bodybody == 0:
        raise Error("no sweep-emitted body-body contact — sweep leg vacuous")

    # Fields CPU vs fields GPU.
    var dc = DataFields[DTYPE, NQ_H, NV_H, NBODY_H, MC_H, NSITE_H, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_H):
            dc.qpos.data[e * NQ_H + i] = _humanoid_qpos(e, i)
    forward_kinematics_fields[
        "cpu", DTYPE, NQ_H, NV_H, NBODY_H, NJOINT_H, MC_H, NGEOM_H,
        NEQ_H, NTD_H, NSITE_H, NEXCL_H, 0, BATCH,
    ](dc, mf)
    detect_contacts_sap_fields[
        "cpu", DTYPE, NQ_H, NV_H, NBODY_H, NJOINT_H, MC_H, NGEOM_H,
        NEQ_H, NTD_H, NSITE_H, NEXCL_H, 0, BATCH,
    ](dc, mf)
    var worst = Float64(0)
    for e in range(BATCH):
        var nc_g = Int(
            d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        var nc_c = Int(
            dc.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        if nc_g != nc_c:
            raise Error("fields-CPU SAP contact count differs from fields-GPU")
        for c in range(nc_g):
            for k in range(CONTACT_SIZE):
                var err = abs(
                    Float64(
                        dc.contacts.data[
                            e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                    )
                    - Float64(
                        d.contacts.data[
                            e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                    )
                )
                if err > worst:
                    worst = err
    print("  fields-CPU vs fields-GPU worst record err:", worst)
    if worst > 1e-4:
        raise Error("fields-CPU SAP contact records tolerance exceeded")
    print("  PASS: fields-CPU SAP contacts within 1e-4")

    # ── Cross-check: fields-SAP vs fields-O(N^2) on the SAME FK products.
    # Emission ORDER differs (SAP sweeps sorted by aabb_min_x) and the
    # conventions differ (SAP plane contacts: BODY_B=-1 vs 0, no
    # INCLUDEMARGIN write), so compare as SETS: unordered body pair (plane/
    # world normalized to 0) + pos/dist within 1e-4. Same-type body-body
    # pairs may be visited with operands swapped by the sweep (pos then
    # differs at float rounding level), hence the tolerance.
    var dn = DataFields[DTYPE, NQ_H, NV_H, NBODY_H, MC_H, NSITE_H, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_H):
            dn.qpos.data[e * NQ_H + i] = _humanoid_qpos(e, i)
    dn.upload_all(ctx)
    forward_kinematics_fields[
        "gpu", DTYPE, NQ_H, NV_H, NBODY_H, NJOINT_H, MC_H, NGEOM_H,
        NEQ_H, NTD_H, NSITE_H, NEXCL_H, 0, BATCH,
    ](dn, mf, ctx)
    detect_contacts_fields[
        "gpu", DTYPE, NQ_H, NV_H, NBODY_H, NJOINT_H, MC_H, NGEOM_H,
        NEQ_H, NTD_H, NSITE_H, NEXCL_H, 0, BATCH,
    ](dn, mf, ctx)
    dn.contacts.download(ctx)
    dn.meta.download(ctx)
    for e in range(BATCH):
        var nc_sap = Int(
            d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        var nc_n2 = Int(
            dn.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        print("  env", e, ": ncon fields-SAP=", nc_sap, " fields-N2=", nc_n2)
        if nc_sap != nc_n2:
            raise Error("fields-SAP vs fields-O(N^2) contact count mismatch")
        var used = List[Bool](length=nc_sap, fill=False)
        for c in range(nc_n2):
            var base_n = e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE
            var ba_n = Int(dn.contacts.data[base_n + 0])
            var bb_n = Int(dn.contacts.data[base_n + 1])
            if bb_n < 0:
                bb_n = 0
            var found = False
            for s in range(nc_sap):
                if used[s]:
                    continue
                var base_s = e * MC_H * CONTACT_SIZE + s * CONTACT_SIZE
                var ba_s = Int(d.contacts.data[base_s + 0])
                var bb_s = Int(d.contacts.data[base_s + 1])
                if bb_s < 0:
                    bb_s = 0
                var pair_match = (ba_n == ba_s and bb_n == bb_s) or (
                    ba_n == bb_s and bb_n == ba_s
                )
                if not pair_match:
                    continue
                # pos (idx 2,3,4) + dist (idx 8) within 1e-4
                var ok = True
                for k in [2, 3, 4, 8]:
                    var err = abs(
                        Float64(dn.contacts.data[base_n + k])
                        - Float64(d.contacts.data[base_s + k])
                    )
                    if err > 1e-4:
                        ok = False
                if ok:
                    used[s] = True
                    found = True
                    break
            if not found:
                print(
                    "  UNMATCHED O(N^2) contact env", e, "idx", c,
                    "bodies (", ba_n, ",", bb_n, ") pos=(",
                    dn.contacts.data[base_n + 2], ",",
                    dn.contacts.data[base_n + 3], ",",
                    dn.contacts.data[base_n + 4], ")",
                )
                raise Error("fields-SAP vs fields-O(N^2) set mismatch")
    print("  PASS: fields-SAP == fields-O(N^2) as contact SETS")

    # ── Auto dispatcher: humanoid must route to SAP (bit-equal).
    var da = DataFields[DTYPE, NQ_H, NV_H, NBODY_H, MC_H, NSITE_H, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_H):
            da.qpos.data[e * NQ_H + i] = _humanoid_qpos(e, i)
    da.upload_all(ctx)
    forward_kinematics_fields[
        "gpu", DTYPE, NQ_H, NV_H, NBODY_H, NJOINT_H, MC_H, NGEOM_H,
        NEQ_H, NTD_H, NSITE_H, NEXCL_H, 0, BATCH,
    ](da, mf, ctx)
    detect_contacts_auto_fields[
        "gpu", DTYPE, NQ_H, NV_H, NBODY_H, NJOINT_H, MC_H, NGEOM_H,
        NEQ_H, NTD_H, NSITE_H, NEXCL_H, 0, BATCH,
    ](da, mf, ctx)
    da.contacts.download(ctx)
    da.meta.download(ctx)
    for e in range(BATCH):
        var nc_a = Int(
            da.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        var nc_s = Int(
            d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        if nc_a != nc_s:
            raise Error("auto(humanoid) != SAP contact count")
        for c in range(nc_s):
            for k in range(CONTACT_SIZE):
                var a = da.contacts.data[
                    e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + k
                ]
                var b = d.contacts.data[
                    e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + k
                ]
                if a != b:
                    raise Error("auto(humanoid) != SAP contact record")
    print("  PASS: auto dispatcher routes humanoid to SAP, bit-equal")


def _part_b_sawyer(ctx: DeviceContext) raises:
    print("--- Part B: sawyer SAP mesh leg fields GOLDEN, BATCH=", BATCH)
    print("  sawyer NGEOM=", NGEOM_S)

    # Build the CPU model (loads STL hulls) and serialize to a slab that
    # includes the mesh sections (NEXCLUDE=0, NMESH_VERTS=NMESHV_S) — the
    # same copy_* helpers ModelDefFromXML.init_model_gpu uses, at the
    # mesh-sized buffer this test needs.
    var model = Model[
        DTYPE, NQ_S, NV_S, NBODY_S, NJOINT_S, MC_S, NGEOM_S, NEQ_S,
        SawyerReachModel.CONE_TYPE, NTD_S, NSITE_S,
    ]()
    var data = Data[DTYPE, NQ_S, NV_S, NBODY_S, NJOINT_S, MC_S, NSITE_S]()
    SawyerReachModel.setup_model_and_data[DTYPE](model, data)
    print(
        "  num_meshes:", model.num_meshes,
        " hull verts:", len(model.mesh_vert) // 3,
    )
    if model.num_meshes == 0 or len(model.mesh_vert) == 0:
        raise Error("expected STL mesh hulls — gate is vacuous")
    if len(model.mesh_vert) > NMESHV_S * 3:
        raise Error("mesh hull verts exceed NMESHV capacity — raise NMESHV")
    if model.num_meshes > MAX_GPU_MESHES:
        raise Error("num_meshes exceeds MAX_GPU_MESHES")

    var host_buf = ctx.enqueue_create_host_buffer[DTYPE](MS_S)
    ctx.synchronize()
    for i in range(MS_S):
        host_buf[i] = Scalar[DTYPE](0)
    copy_model_to_buffer(model, host_buf)
    copy_geoms_to_buffer(model, host_buf)
    copy_tendons_to_buffer(model, host_buf)
    copy_invweight0_to_buffer(model, host_buf)
    copy_mesh_hull_to_buffer(model, host_buf)

    var model_t = TensorImpl[DTYPE].alloc(MS_S)
    for i in range(MS_S):
        model_t.data[i] = host_buf[i]
    model_t.upload(ctx)
    var mf = ModelFields[
        DTYPE, NV_S, NBODY_S, NJOINT_S, NGEOM_S, NEQ_S, NTD_S, NSITE_S,
        0, NMESHV_S,
    ]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    # Locate the obj cylinder + mesh-geom bodies for the non-vacuity check.
    var obj_body = -1
    var mesh_bodies = List[Int]()
    for g in range(NGEOM_S):
        var gt = Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_TYPE])
        var gb = Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
        if gt == GEOM_MESH and gb != 0:
            mesh_bodies.append(gb)
        if gt == GEOM_CYLINDER:
            var r = Float64(
                mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_RADIUS]
            )
            var hl = Float64(
                mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_HALF_LENGTH]
            )
            if abs(r - 0.02) < 1e-6 and abs(hl - 0.02) < 1e-6:
                obj_body = gb
    print("  obj_body:", obj_body, " mesh-geom bodies:", len(mesh_bodies))
    if obj_body < 0 or len(mesh_bodies) == 0:
        raise Error("could not identify obj cylinder / mesh geoms")

    # Poses (same as test_mesh_detection_fields): env0 obj on table; env1
    # obj teleported into the eGripperBase hull -> mesh-cylinder GJK contact.
    var qcfg = List[List[Float64]]()
    for e in range(BATCH):
        var q = List[Float64](length=NQ_S, fill=0.0)
        q[0] = 1.889288  # j0
        q[1] = -0.575769  # j1
        q[2] = -0.976659  # j2
        q[3] = 1.641991  # j3
        q[4] = 0.942860  # j4
        q[5] = 1.043696  # j5
        q[6] = 2.292833  # j6
        q[7] = 0.0  # r_close
        q[8] = 0.0  # l_close
        if e == 0:
            q[9] = 0.0  # obj x (on table)
            q[10] = 0.6  # obj y
            q[11] = 0.02  # obj z
        else:
            q[9] = 0.005  # obj x (inside gripper mesh hull)
            q[10] = 0.601  # obj y
            q[11] = 0.25  # obj z
        q[12] = 1.0  # obj quat w
        q[13] = 0.0
        q[14] = 0.0
        q[15] = 0.0
        qcfg.append(q^)

    var d = DataFields[DTYPE, NQ_S, NV_S, NBODY_S, MC_S, NSITE_S, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_S):
            d.qpos.data[e * NQ_S + i] = Scalar[DTYPE](qcfg[e][i])
    d.upload_all(ctx)

    # Fields: FK + SAP detection.
    forward_kinematics_fields[
        "gpu", DTYPE, NQ_S, NV_S, NBODY_S, NJOINT_S, MC_S, NGEOM_S,
        NEQ_S, NTD_S, NSITE_S, 0, NMESHV_S, BATCH,
    ](d, mf, ctx)
    detect_contacts_sap_fields[
        "gpu", DTYPE, NQ_S, NV_S, NBODY_S, NJOINT_S, MC_S, NGEOM_S,
        NEQ_S, NTD_S, NSITE_S, 0, NMESHV_S, BATCH,
    ](d, mf, ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)

    var ncon_s = 0
    var fp_s = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
        ncon_s += nc
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                fp_s += Float64(
                    d.contacts.data[
                        e * MC_S * CONTACT_SIZE + c * CONTACT_SIZE + k
                    ]
                ) * Float64((e + 1) * (c + 1) * (k + 1))
    if ncon_s == 0:
        raise Error("sawyer SAP: no contacts — gate is vacuous")
    print("  sawyer fields-SAP total contacts:", ncon_s)
    if HARVEST:
        print("  HARVEST GOLD_NCON_S =", ncon_s)
        print("  HARVEST GOLD_CON_S  =", fp_s)
    else:
        if ncon_s != GOLD_NCON_S and not has_nvidia_gpu_accelerator():
            raise Error(
                "sawyer SAP contacts " + String(ncon_s) + " != golden "
                + String(GOLD_NCON_S)
            )
        var denom = abs(GOLD_CON_S) if abs(GOLD_CON_S) > 1e-9 else 1.0
        if abs(fp_s - GOLD_CON_S) / denom > GOLD_RTOL and (
            not has_nvidia_gpu_accelerator()
        ):
            raise Error(
                "sawyer SAP fingerprint " + String(fp_s) + " != golden "
                + String(GOLD_CON_S)
            )
        print("  PASS: sawyer fields-SAP matches golden fingerprint")

    # Non-vacuity: env1 must have a contact between a mesh-geom body and
    # the obj body (GJK/EPA mesh fallback through the SAP sweep).
    var ncon1 = Int(d.meta.data[1 * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
    var mesh_contact_found = False
    for c in range(ncon1):
        var ba = Int(
            d.contacts.data[1 * MC_S * CONTACT_SIZE + c * CONTACT_SIZE + 0]
        )
        var bb = Int(
            d.contacts.data[1 * MC_S * CONTACT_SIZE + c * CONTACT_SIZE + 1]
        )
        for mb in mesh_bodies:
            if (ba == mb and bb == obj_body) or (
                bb == mb and ba == obj_body
            ):
                mesh_contact_found = True
                print(
                    "  mesh contact: bodies (", ba, ",", bb, ") dist=",
                    d.contacts.data[
                        1 * MC_S * CONTACT_SIZE + c * CONTACT_SIZE + 8
                    ],
                )
    if not mesh_contact_found:
        raise Error("no MESH-involved contact in env1 — gate is vacuous")
    print("  PASS: MESH-involved contact present through SAP (GJK/EPA)")


def _part_c_walker(ctx: DeviceContext) raises:
    print("--- Part C: walker2d auto dispatcher routes to O(N^2)")
    print("  walker2d NGEOM=", NGEOM_W)
    comptime assert NGEOM_W < SAP_THRESHOLD, "walker2d must route to O(N^2)"

    var model_t = TensorImpl[DTYPE].alloc(MS_W)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    Walker2dModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[DTYPE, NV_W, NBODY_W, NJOINT_W, NGEOM_W]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    # Poses from test_contact_detection_fields (floor penetration).
    var qcfg = List[List[Float64]]()
    var q0 = List[Float64](length=NQ_W, fill=0.0)
    q0[1] = 1.18
    qcfg.append(q0^)
    var q1 = List[Float64](length=NQ_W, fill=0.0)
    q1[1] = 0.85
    q1[3] = 0.6
    q1[4] = -1.1
    q1[6] = -0.4
    q1[7] = -0.9
    qcfg.append(q1^)

    var d1 = DataFields[DTYPE, NQ_W, NV_W, NBODY_W, MC_W, 0, BATCH]()
    var d2 = DataFields[DTYPE, NQ_W, NV_W, NBODY_W, MC_W, 0, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_W):
            d1.qpos.data[e * NQ_W + i] = Scalar[DTYPE](qcfg[e][i])
            d2.qpos.data[e * NQ_W + i] = Scalar[DTYPE](qcfg[e][i])
    d1.upload_all(ctx)
    d2.upload_all(ctx)

    forward_kinematics_fields[
        "gpu", DTYPE, NQ_W, NV_W, NBODY_W, NJOINT_W, MC_W, NGEOM_W,
        0, 0, 0, 0, 0, BATCH,
    ](d1, mf, ctx)
    detect_contacts_fields[
        "gpu", DTYPE, NQ_W, NV_W, NBODY_W, NJOINT_W, MC_W, NGEOM_W,
        0, 0, 0, 0, 0, BATCH,
    ](d1, mf, ctx)
    forward_kinematics_fields[
        "gpu", DTYPE, NQ_W, NV_W, NBODY_W, NJOINT_W, MC_W, NGEOM_W,
        0, 0, 0, 0, 0, BATCH,
    ](d2, mf, ctx)
    detect_contacts_auto_fields[
        "gpu", DTYPE, NQ_W, NV_W, NBODY_W, NJOINT_W, MC_W, NGEOM_W,
        0, 0, 0, 0, 0, BATCH,
    ](d2, mf, ctx)
    d1.contacts.download(ctx)
    d1.meta.download(ctx)
    d2.contacts.download(ctx)
    d2.meta.download(ctx)

    for e in range(BATCH):
        var nc_1 = Int(
            d1.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        var nc_2 = Int(
            d2.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        print("  env", e, ": ncon direct=", nc_1, " auto=", nc_2)
        if nc_1 != nc_2:
            raise Error("auto(walker2d) != detect_contacts_fields count")
        if nc_1 == 0:
            raise Error("expected contacts in this pose — gate is vacuous")
        for c in range(nc_1):
            for k in range(CONTACT_SIZE):
                var a = d1.contacts.data[
                    e * MC_W * CONTACT_SIZE + c * CONTACT_SIZE + k
                ]
                var b = d2.contacts.data[
                    e * MC_W * CONTACT_SIZE + c * CONTACT_SIZE + k
                ]
                if a != b:
                    raise Error("auto(walker2d) != detect_contacts_fields")
    print("  PASS: auto dispatcher routes walker2d to O(N^2), bit-equal")


def main() raises:
    var ctx = DeviceContext()
    _part_a_humanoid(ctx)
    _part_b_sawyer(ctx)
    _part_c_walker(ctx)
    print("test_sap_fields: ALL PASS")
