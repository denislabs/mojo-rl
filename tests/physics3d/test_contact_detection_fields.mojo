"""P4 gate: per-field contact detection vs the legacy narrow-phase kernel.

Walker2D (floor plane + 7 body capsules, MAX_CONTACTS=20), BATCH=2 with
penetrating poses (slight and heavy floor penetration, bent legs): legacy
FK -> detect_contacts_gpu on the flat slab vs fields FK ->
detect_contacts_fields on DataFields/ModelFields. Contact count AND the
first ncon full records (23 fields each) must be BIT-EXACT; fields-CPU must
agree with fields-GPU on count + records within 1e-4.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_contact_detection_fields.mojo
"""

from std.math import abs
from std.gpu import block_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics_gpu,
)
from mojo_rl.physics3d.kinematics.forward_kinematics_fields import (
    forward_kinematics_fields,
)
from mojo_rl.physics3d.collision.contact_detection import detect_contacts_gpu
from mojo_rl.physics3d.collision.contact_detection_fields import (
    detect_contacts_fields,
)
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    contacts_offset,
    metadata_offset,
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
comptime BATCH = 2
comptime SS = state_size[NQ, NV, NBODY, MC, 0]()
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()


def _legacy_fk_detect_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    forward_kinematics_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_
    ](env, state, model)
    detect_contacts_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_, NGEOM
    ](env, state, model)


def main() raises:
    print("--- contact detection: fields vs legacy, walker2d BATCH=", BATCH)
    var ctx = DeviceContext()

    var model_t = TensorImpl[DTYPE].alloc(MS)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    Walker2dModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    # Poses: env0 slight floor penetration (standing, rootz below rest);
    # env1 heavy penetration + bent legs.
    comptime O_QPOS = qpos_offset[NQ, NV]()
    var slab_t = TensorImpl[DTYPE].alloc(BATCH * SS)
    var d = DataFields[DTYPE, NQ, NV, NBODY, MC, 0, BATCH]()
    var qcfg = List[List[Float64]]()
    var q0 = List[Float64](length=NQ, fill=0.0)
    q0[1] = 1.18  # rootz slightly below standing 1.25 -> feet touch/penetrate
    qcfg.append(q0^)
    var q1 = List[Float64](length=NQ, fill=0.0)
    q1[1] = 0.85
    q1[3] = 0.6
    q1[4] = -1.1
    q1[6] = -0.4
    q1[7] = -0.9
    qcfg.append(q1^)
    for e in range(BATCH):
        for i in range(NQ):
            slab_t.data[e * SS + O_QPOS + i] = Scalar[DTYPE](qcfg[e][i])
            d.qpos.data[e * NQ + i] = Scalar[DTYPE](qcfg[e][i])
    slab_t.upload(ctx)
    d.upload_all(ctx)

    # Legacy: FK + detection in one launch.
    ctx.enqueue_function[_legacy_fk_detect_kernel[BATCH]](
        slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
        model_t.lt["gpu", Layout.row_major(1, MS)](),
        grid_dim=(BATCH,),
        block_dim=(1,),
    )
    slab_t.download(ctx)

    # Fields: FK + detection.
    forward_kinematics_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, BATCH,
    ](d, mf, ctx)
    detect_contacts_fields[
        "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, BATCH,
    ](d, mf, ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)

    comptime O_CON = contacts_offset[NQ, NV, NBODY]()
    comptime O_META = metadata_offset[NQ, NV, NBODY, MC]()
    comptime METADATA_SIZE_L = 4
    var bad = 0
    for e in range(BATCH):
        var ncon_legacy = Int(
            slab_t.data[e * SS + O_META + META_IDX_NUM_CONTACTS]
        )
        var ncon_fields = Int(
            d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS]
        )
        print("  env", e, ": ncon legacy=", ncon_legacy, " fields=", ncon_fields)
        if ncon_legacy != ncon_fields:
            bad += 1
            continue
        if ncon_legacy == 0:
            raise Error("expected contacts in this pose — gate is vacuous")
        for c in range(ncon_legacy):
            for k in range(CONTACT_SIZE):
                var a = d.contacts.data[
                    e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k
                ]
                var b = slab_t.data[e * SS + O_CON + c * CONTACT_SIZE + k]
                if a != b:
                    if bad < 5:
                        print(
                            "  MISMATCH env", e, "contact", c, "field", k,
                            ": fields=", a, " legacy=", b,
                        )
                    bad += 1
    if bad != 0:
        raise Error("contact detection fields-GPU vs legacy-GPU mismatch")
    print("  PASS: contact records + counts BIT-EXACT vs legacy")

    # Fields CPU vs fields GPU.
    var dc = DataFields[DTYPE, NQ, NV, NBODY, MC, 0, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            dc.qpos.data[e * NQ + i] = Scalar[DTYPE](qcfg[e][i])
    forward_kinematics_fields[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, BATCH,
    ](dc, mf)
    detect_contacts_fields[
        "cpu", DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, BATCH,
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

    print("test_contact_detection_fields: ALL PASS")
