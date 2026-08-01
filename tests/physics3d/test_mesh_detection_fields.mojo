"""Regression gate (GOLDEN-frozen): MESH narrow-phase in fields contact
detection, SawyerReach (robot meshes + block.stl).

Originally validated BIT-EXACT against the legacy FK + narrow-phase kernels.
That legacy reference was frozen into the GOLDEN fingerprints below during
Phase-0 of the physics3d sunset, so this gate survives deletion of the legacy
slab/kernels. It checks:
  * fields-GPU (FK -> detect) reproduces the frozen fingerprint — per-env
    contact counts + a contact-record checksum,
  * a MESH-involved contact (mesh-geom body vs obj body) is present in env1
    (GJK/EPA fallback — non-vacuous), and
  * fields-CPU == fields-GPU on records, fed the GPU FK products (isolates the
    detection port from FK differences).

env0 = canonical reset (obj on table); env1 = obj teleported INTO the
eGripperBase mesh hull.

The env1 z used to be 0.25, where the obj is in fact 15.1 mm CLEAR of the hull
— the gate's mesh contact was a phantom, manufactured by a flat GJK simplex
being read as an enclosure of the origin (see `_closest_point_on_simplex`).
Both the count and the checksum were frozen around it. z=0.28 is a real
overlap: float64 CPU, float32 CPU and float32 GPU agree there to 6 digits,
where at z=0.25 float64 said +0.0151 and float32 said -0.0553.

Contact DEPTH on the mesh path is still the crude fallback in `gjk_epa` (the
Minkowski-difference extent along the centre line), not a true EPA depth, so
the golden pins what the engine computes rather than ground truth. Separation
and the contact/no-contact verdict ARE trustworthy; the depth is not.

Model build = fields-native init_fields (Stage B; the NMESHV-padded mesh
build). Regenerate goldens after an INTENTIONAL physics change: HARVEST=True,
run on Apple, paste, False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_mesh_detection_fields.mojo
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
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
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
comptime NMESHV = MAX_GPU_MESHES * 256
comptime METADATA_SIZE_L = 4

# --- GOLDEN fingerprints (regenerated after the flat-simplex fix) ------------
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_NCON = 2  # total contacts across both envs
# ⚠ GOLD_CON moved +32.0 on 2026-08-01 with the narrow-phase CONTACT DIRECTION
# fix (see `collision/broadphase_sap.mojo`). Accounted for exactly: the
# fractional part is UNCHANGED, so only integer fields moved, and 33 - 1 = 32
# is exactly one `(body_a, body_b)` relabel on the env0 obj(33)/table(1)
# contact at weight (e+1)(c+1) = 1. Normal, position and dist are bit-identical
# and the count is unchanged. Not re-recorded blind.
comptime GOLD_CON = 495.781851073727  # order-sensitive contact-record checksum


def main() raises:
    print("--- mesh contact detection fields GOLDEN gate: sawyer BATCH=", BATCH)
    var ctx = DeviceContext()

    # Fields-native model build (loads STL hulls, NMESHV-padded — Stage B).
    var mf = Model[
        DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, 0, NMESHV
    ]()
    SawyerReachModel.init_fields[DTYPE, NMESHV](ctx, mf)

    # Locate the obj cylinder + mesh-geom bodies for the non-vacuity check.
    var obj_body = -1
    var mesh_bodies = List[Int]()
    for g in range(NGEOM):
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

    # Poses: env0 obj on table; env1 obj teleported into eGripperBase hull.
    var qcfg = List[List[Float64]]()
    for e in range(BATCH):
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
        if e == 0:
            q[9] = 0.0
            q[10] = 0.6
            q[11] = 0.02
        else:
            q[9] = 0.005
            q[10] = 0.601
            q[11] = 0.28  # inside the hull — see the module docstring
        q[12] = 1.0
        qcfg.append(q^)

    var d = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            d.qpos.data[e * NQ + i] = Scalar[DTYPE](qcfg[e][i])
    d.upload_all(ctx)

    # Fields GPU: FK + detection.
    forward_kinematics[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d, mf, ctx)
    detect_contacts[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](d, mf, ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)

    var ncon_total = 0
    var fp_con = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
        ncon_total += nc
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                fp_con += Float64(
                    d.contacts.data[e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k]
                ) * Float64((e + 1) * (c + 1) * (k + 1))
    if ncon_total == 0:
        raise Error("expected contacts in these poses — gate is vacuous")
    print("  fields-GPU total contacts:", ncon_total)

    if HARVEST:
        print("  HARVEST GOLD_NCON =", ncon_total)
        print("  HARVEST GOLD_CON  =", fp_con)
    else:
        if ncon_total != GOLD_NCON and not has_nvidia_gpu_accelerator():
            raise Error(
                "total contacts " + String(ncon_total) + " != golden "
                + String(GOLD_NCON)
            )
        var denom = abs(GOLD_CON) if abs(GOLD_CON) > 1e-9 else 1.0
        if abs(fp_con - GOLD_CON) / denom > GOLD_RTOL and (
            not has_nvidia_gpu_accelerator()
        ):
            raise Error(
                "contact-record fingerprint " + String(fp_con) + " != golden "
                + String(GOLD_CON)
            )
        print("  PASS: fields-GPU matches golden fingerprint")

    # Non-vacuity: env1 must have a mesh-geom-body vs obj-body contact.
    var ncon1 = Int(d.meta.data[1 * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
    var mesh_contact_found = False
    for c in range(ncon1):
        var ba = Int(
            d.contacts.data[1 * MC * CONTACT_SIZE + c * CONTACT_SIZE + 0]
        )
        var bb = Int(
            d.contacts.data[1 * MC * CONTACT_SIZE + c * CONTACT_SIZE + 1]
        )
        for mb in mesh_bodies:
            if (ba == mb and bb == obj_body) or (bb == mb and ba == obj_body):
                mesh_contact_found = True
    if not mesh_contact_found:
        raise Error("no MESH-involved contact in env1 — gate is vacuous")
    print("  PASS: MESH-involved contact present (GJK/EPA fallback)")

    # --- fields-CPU vs fields-GPU records (fed GPU FK products) ---
    var dc = Data[DTYPE, NQ, NV, NBODY, MC, NSITE, BATCH]()
    d.xpos.download(ctx)
    d.xquat.download(ctx)
    for i in range(BATCH * NBODY * 3):
        dc.xpos.data[i] = d.xpos.data[i]
    for i in range(BATCH * NBODY * 4):
        dc.xquat.data[i] = d.xquat.data[i]
    detect_contacts[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE,
        0, NMESHV, BATCH,
    ](dc, mf)
    var worst = Float64(0)
    for e in range(BATCH):
        var nc_g = Int(d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
        var nc_c = Int(
            dc.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        if nc_g != nc_c:
            raise Error("fields-CPU contact count differs from fields-GPU")
        for c in range(nc_g):
            for k in range(CONTACT_SIZE):
                var err = abs(
                    Float64(
                        dc.contacts.data[
                            e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                    )
                    - Float64(
                        d.contacts.data[
                            e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                    )
                )
                if err > worst:
                    worst = err
    print("  fields-CPU vs fields-GPU worst record err:", worst)
    if worst > 1e-4:
        raise Error("fields-CPU contact records tolerance exceeded")
    print("  PASS: fields-CPU contacts within 1e-4")
    print("test_mesh_detection_fields: ALL PASS")
