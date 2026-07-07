"""P4 gate: full Euler step WITH CONTACTS — fields vs legacy PGS pipeline.

Walker2D dropped onto the floor (rootz=1.10, feet penetrating), BATCH=2,
3 consecutive full steps. Legacy per substep: step_kernel ->
detect_contacts_gpu -> PGSSolver.solve_gpu (2D-threaded: contacts + limits
+ equality) -> finalize. Fields: EulerIntegratorFields.step with
CONTACTS=True (detection -> serialized contact PGS -> limits -> finalize).
qpos/qvel/qacc AND the solved contact forces written back into the records
must be BIT-EXACT after every step; fields-CPU within 1e-2 after 3 steps
(contact PGS is iterative -> cross-target drift larger than smooth
dynamics).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_contact_solve_fields.mojo
"""

from std.math import abs
from std.gpu import block_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.integrator.euler_fields import EulerIntegratorFields
from mojo_rl.physics3d.solver.pgs_solver import PGSSolver
from mojo_rl.physics3d.collision.contact_detection import detect_contacts_gpu
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    ws_solver_offset,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    MODEL_META_IDX_TIMESTEP,
    META_IDX_NUM_CONTACTS,
    CONTACT_SIZE,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
comptime CONE = Walker2dModel.CONE_TYPE
comptime BATCH = 2
comptime N_STEPS = 3
comptime SS = state_size[NQ, NV, NBODY, MC, 0]()
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
# Full PGS workspace: integrator temps + M_inv + solver block.
comptime WS = ws_solver_offset[NV, NBODY]() + 81 * MC + 12 * MC * NV


def _legacy_step_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    EulerIntegrator[SOLVER=PGSSolver].step_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_, WS
    ](state, model, workspace)


def _legacy_detect_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    detect_contacts_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_, NGEOM
    ](env, state, model)


def _legacy_pgs_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    PGSSolver.solve_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, NV, B_, WS, NGEOM,
        0, CONE, 0, 0,
    ](state, model, workspace)


def _legacy_finalize_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    EulerIntegrator[SOLVER=PGSSolver].step_finalize_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, SS, MS, B_, WS
    ](state, model, workspace)


def main() raises:
    print("--- Euler full step WITH CONTACTS: fields vs legacy PGS, BATCH=", BATCH)
    var ctx = DeviceContext()

    var model_t = TensorImpl[DTYPE].alloc(MS)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    Walker2dModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    comptime O_QPOS = qpos_offset[NQ, NV]()
    comptime O_QVEL = qvel_offset[NQ, NV]()
    comptime O_QACC = qacc_offset[NQ, NV]()
    comptime O_QFRC = qfrc_offset[NQ, NV]()

    var slab_t = TensorImpl[DTYPE].alloc(BATCH * SS)
    var d = DataFields[DTYPE, NQ, NV, NBODY, MC, 0, BATCH]()
    var dc = DataFields[DTYPE, NQ, NV, NBODY, MC, 0, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10  # feet penetrate the floor
            slab_t.data[e * SS + O_QPOS + i] = qp
            d.qpos.data[e * NQ + i] = qp
            dc.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5  # falling
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            slab_t.data[e * SS + O_QVEL + i] = qv
            slab_t.data[e * SS + O_QFRC + i] = qf
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf
            dc.qvel.data[e * NV + i] = qv
            dc.qfrc.data[e * NV + i] = qf
    slab_t.upload(ctx)
    d.upload_all(ctx)
    var ws_t = TensorImpl[DTYPE].alloc(BATCH * WS)
    ws_t.upload(ctx)

    var integ = EulerIntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, CONE, BATCH,
    ]()
    integ.prepare_gpu(ctx)
    var integ_c = EulerIntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MC, NGEOM, 0, 0, 0, 0, 0, CONE, BATCH,
    ]()

    comptime O_CON = contacts_offset[NQ, NV, NBODY]()
    comptime O_META = metadata_offset[NQ, NV, NBODY, MC]()
    comptime METADATA_SIZE_L = 4

    for step in range(N_STEPS):
        ctx.enqueue_function[_legacy_step_kernel[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_detect_kernel[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_pgs_kernel[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1, MC),
        )
        ctx.enqueue_function[_legacy_finalize_kernel[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        integ.step["gpu"](d, mf, ctx)
        integ_c.step["cpu"](dc, mf)

        slab_t.download(ctx)
        d.qpos.download(ctx)
        d.qvel.download(ctx)
        d.qacc.download(ctx)
        d.contacts.download(ctx)
        d.meta.download(ctx)
        var bad = 0
        var ncon_seen = 0
        for e in range(BATCH):
            var nc = Int(d.meta.data[e * METADATA_SIZE_L + META_IDX_NUM_CONTACTS])
            var nc_l = Int(slab_t.data[e * SS + O_META + META_IDX_NUM_CONTACTS])
            if nc != nc_l:
                print("  ncon mismatch env", e, ": fields", nc, " legacy", nc_l)
                bad += 1
                continue
            ncon_seen += nc
            for i in range(NQ):
                if d.qpos.data[e * NQ + i] != slab_t.data[e * SS + O_QPOS + i]:
                    if bad < 4:
                        print(
                            "  qpos diff e", e, "i", i, ":",
                            d.qpos.data[e * NQ + i], "vs",
                            slab_t.data[e * SS + O_QPOS + i],
                        )
                    bad += 1
            for i in range(NV):
                if d.qvel.data[e * NV + i] != slab_t.data[e * SS + O_QVEL + i]:
                    if bad < 4:
                        print(
                            "  qvel diff e", e, "i", i, ":",
                            d.qvel.data[e * NV + i], "vs",
                            slab_t.data[e * SS + O_QVEL + i],
                        )
                    bad += 1
                if d.qacc.data[e * NV + i] != slab_t.data[e * SS + O_QACC + i]:
                    bad += 1
            for c in range(nc):
                for k in range(CONTACT_SIZE):
                    if (
                        d.contacts.data[
                            e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                        != slab_t.data[e * SS + O_CON + c * CONTACT_SIZE + k]
                    ):
                        if bad < 4:
                            print(
                                "  record diff e", e, "c", c, "k", k, ":",
                                d.contacts.data[
                                    e * MC * CONTACT_SIZE
                                    + c * CONTACT_SIZE + k
                                ],
                                "vs",
                                slab_t.data[
                                    e * SS + O_CON + c * CONTACT_SIZE + k
                                ],
                            )
                        bad += 1
        if bad != 0:
            raise Error("step " + String(step) + ": contact step mismatch")
        if ncon_seen == 0:
            raise Error("no contacts at step " + String(step) + " — gate vacuous")
        print(
            "  step", step, ": BIT-EXACT (qpos/qvel/qacc + contact records),"
            " total contacts:", ncon_seen,
        )

    var worst = Float64(0)
    d.qpos.download(ctx)
    for i in range(BATCH * NQ):
        var err = abs(Float64(dc.qpos.data[i]) - Float64(d.qpos.data[i]))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU final qpos worst err:", worst)
    if worst > 1e-2:
        raise Error("fields-CPU contact dynamics diverged from GPU")
    print("  PASS: fields-CPU within 1e-2 after", N_STEPS, "contact steps")
    print("test_contact_solve_fields: ALL PASS")
