"""P2 pilot gate: stateful per-field Euler integrator vs the legacy Euler
step, FULL STEP, contact-free comparison (solver/contacts/limits skipped on
BOTH sides — the legacy step_kernel + step_finalize_kernel pair without the
intervening contact/solver kernels equals unconstrained dynamics, which is
exactly what EulerIntegratorFields.step implements).

Walker2D, BATCH=3 (distinct qpos/qvel/qfrc per env), 3 CONSECUTIVE steps:
- fields-GPU vs legacy-GPU: qpos/qvel/qacc BIT-EXACT after every step.
- fields-CPU (same formula bodies) vs fields-GPU: 1e-3 after 3 steps.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_euler_fields.mojo
"""

from std.math import abs
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.integrator.euler_fields import EulerIntegratorFields
from mojo_rl.physics3d.solver.newton_solver import NewtonSolver
from mojo_rl.physics3d.constraints import detect_and_solve_limits_gpu
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    ws_m_inv_offset,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xvel_offset,
    xangvel_offset,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MAX_CONTACTS = Walker2dModel.MAX_CONTACTS
comptime BATCH = 3
comptime N_STEPS = 3
comptime SS = state_size[NQ, NV, NBODY, MAX_CONTACTS, 0]()
comptime MS = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
# step_kernel computes M_inv (NEEDS_M_INV solvers) — ws must cover it.
comptime WS = ws_m_inv_offset[NV, NBODY]() + NV * NV


def _legacy_step_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    EulerIntegrator[SOLVER=NewtonSolver].step_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SS, MS, B_, WS
    ](state, model, workspace)


def _legacy_limits_kernel[
    B_: Int
](
    dt: Scalar[DTYPE],
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    if env >= B_:
        return
    detect_and_solve_limits_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SS, MS, WS, B_, 50, NGEOM
    ](env, dt, state, model, workspace)


def _legacy_finalize_kernel[
    B_: Int
](
    state: LayoutTensor[DTYPE, Layout.row_major(B_, SS), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MS), MutAnyOrigin],
    workspace: LayoutTensor[DTYPE, Layout.row_major(B_, WS), MutAnyOrigin],
):
    EulerIntegrator[SOLVER=NewtonSolver].step_finalize_kernel[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SS, MS, B_, WS
    ](state, model, workspace)


def main() raises:
    print("--- Euler full-step A/B: fields vs legacy, walker2d BATCH=", BATCH)
    var ctx = DeviceContext()

    # Model on device + bridge to fields.
    var model_t = TensorImpl[DTYPE].alloc(MS)
    model_t.upload(ctx)
    var mbuf = model_t.dev.value()
    Walker2dModel.init_model_gpu(ctx, mbuf)
    model_t.download(ctx)
    var mf = ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM]()
    mf.load_from_slab(model_t.data)
    mf.upload_all(ctx)

    # Initial conditions: distinct per env.
    comptime O_QPOS = qpos_offset[NQ, NV]()
    comptime O_QVEL = qvel_offset[NQ, NV]()
    comptime O_QACC = qacc_offset[NQ, NV]()
    comptime O_QFRC = qfrc_offset[NQ, NV]()

    var slab_t = TensorImpl[DTYPE].alloc(BATCH * SS)
    var d = DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, 0, BATCH]()
    var dc = DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, 0, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 7 + i * 3) % 5 - 2) / 20.0
            if i == 1:
                qp = 1.25  # rootz standing height
            slab_t.data[e * SS + O_QPOS + i] = qp
            d.qpos.data[e * NQ + i] = qp
            dc.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 11 + i * 5) % 7 - 3) / 10.0
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 2.0
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
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH=BATCH,
    ]()
    integ.prepare_gpu(ctx)
    var integ_c = EulerIntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        0, 0, 0, 0, 0, BATCH=BATCH,
    ]()

    # dt for the legacy limits kernel (host-read from the flat model meta).
    from mojo_rl.physics3d.gpu.constants import (
        model_metadata_offset,
        MODEL_META_IDX_TIMESTEP,
    )
    comptime O_META_M = model_metadata_offset[NBODY, NJOINT]()
    var dt = model_t.data[O_META_M + MODEL_META_IDX_TIMESTEP]

    # DIAGNOSTIC: don't abort on the first GPU-vs-GPU mismatch — keep stepping
    # so the fields-CPU-vs-fields-GPU comparison below can run and tell us WHICH
    # GPU path is wrong (fields-GPU miscompiles vs legacy-GPU is the odd one out).
    var gpu_mismatch = False
    for step in range(N_STEPS):
        # Legacy: step_kernel + limits + finalize (contact-free but
        # limit-AWARE — the thigh=0.5 config violates walker2d's (-150,0)deg
        # range, so the limit path is genuinely exercised).
        ctx.enqueue_function[_legacy_step_kernel[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_limits_kernel[BATCH]](
            dt,
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ctx.enqueue_function[_legacy_finalize_kernel[BATCH]](
            slab_t.lt["gpu", Layout.row_major(BATCH, SS)](),
            model_t.lt["gpu", Layout.row_major(1, MS)](),
            ws_t.lt["gpu", Layout.row_major(BATCH, WS)](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        # Fields: stateful integrator step.
        integ.step["gpu", False](d, mf, ctx)  # CONTACTS=False: legacy side has no contact kernels
        integ_c.step["cpu", False](dc, mf)

        # Compare fields-GPU vs legacy-GPU bit-exact.
        slab_t.download(ctx)
        d.qpos.download(ctx)
        d.qvel.download(ctx)
        d.qacc.download(ctx)
        d.xvel.download(ctx)
        d.xangvel.download(ctx)
        comptime O_XVEL = xvel_offset[NQ, NV, NBODY]()
        comptime O_XANG = xangvel_offset[NQ, NV, NBODY]()
        var bad = 0
        var worst = Float64(0)
        var w_e = -1
        var w_i = -1
        var w_f = String("")
        var w_fv = Float64(0)
        var w_lv = Float64(0)
        for e in range(BATCH):
            for i in range(NQ):
                var fv = Float64(d.qpos.data[e * NQ + i])
                var lv = Float64(slab_t.data[e * SS + O_QPOS + i])
                if fv != lv:
                    bad += 1
                    var dd = fv - lv
                    if dd < 0:
                        dd = -dd
                    if dd > worst:
                        worst = dd
                        w_e = e
                        w_i = i
                        w_f = "qpos"
                        w_fv = fv
                        w_lv = lv
            for i in range(NV):
                var fv = Float64(d.qvel.data[e * NV + i])
                var lv = Float64(slab_t.data[e * SS + O_QVEL + i])
                if fv != lv:
                    bad += 1
                    var dd = fv - lv
                    if dd < 0:
                        dd = -dd
                    if dd > worst:
                        worst = dd
                        w_e = e
                        w_i = i
                        w_f = "qvel"
                        w_fv = fv
                        w_lv = lv
                var av = Float64(d.qacc.data[e * NV + i])
                var alv = Float64(slab_t.data[e * SS + O_QACC + i])
                if av != alv:
                    bad += 1
                    var dd = av - alv
                    if dd < 0:
                        dd = -dd
                    if dd > worst:
                        worst = dd
                        w_e = e
                        w_i = i
                        w_f = "qacc"
                        w_fv = av
                        w_lv = alv
            for i in range(NBODY * 3):
                var xv = Float64(d.xvel.data[e * NBODY * 3 + i])
                var xlv = Float64(slab_t.data[e * SS + O_XVEL + i])
                if xv != xlv:
                    bad += 1
                    var dd = xv - xlv
                    if dd < 0:
                        dd = -dd
                    if dd > worst:
                        worst = dd
                        w_e = e
                        w_i = i
                        w_f = "xvel"
                        w_fv = xv
                        w_lv = xlv
                var gv = Float64(d.xangvel.data[e * NBODY * 3 + i])
                var glv = Float64(slab_t.data[e * SS + O_XANG + i])
                if gv != glv:
                    bad += 1
                    var dd = gv - glv
                    if dd < 0:
                        dd = -dd
                    if dd > worst:
                        worst = dd
                        w_e = e
                        w_i = i
                        w_f = "xangvel"
                        w_fv = gv
                        w_lv = glv
        if bad != 0:
            print(
                "  MISMATCH @ step",
                step,
                ": bad_elems=",
                bad,
                " worst|delta|=",
                worst,
                " field=",
                w_f,
                " env=",
                w_e,
                " idx=",
                w_i,
            )
            print("    fields-GPU=", w_fv, " legacy-GPU=", w_lv)
            gpu_mismatch = True
        else:
            print(
                "  step", step,
                ": fields-GPU == legacy-GPU BIT-EXACT"
                " (qpos/qvel/qacc/xvel/xangvel)",
            )

    # fields-CPU vs fields-GPU after N_STEPS. This is the DISCRIMINATOR: a small
    # err means fields-GPU == fields-CPU (so fields-GPU is correct and the
    # legacy-GPU path is the one diverging on this device); a large err means
    # fields-GPU itself miscomputes on the GPU.
    var worst = Float64(0)
    for i in range(BATCH * NQ):
        var err = abs(Float64(dc.qpos.data[i]) - Float64(d.qpos.data[i]))
        if err > worst:
            worst = err
    for i in range(BATCH * NV):
        var err = abs(Float64(dc.qvel.data[i]) - Float64(d.qvel.data[i]))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU after", N_STEPS, "steps, worst err:", worst)
    if gpu_mismatch:
        if worst > 1e-3:
            print(
                "  => fields-GPU disagrees with fields-CPU too: the FIELDS GPU"
                " path miscomputes on this device."
            )
        else:
            print(
                "  => fields-GPU MATCHES fields-CPU: fields-GPU is correct; the"
                " LEGACY-GPU reference is the one diverging on this device."
            )
    if worst > 1e-3:
        raise Error("fields-CPU tolerance exceeded")
    if gpu_mismatch:
        raise Error("fields-GPU != legacy-GPU (see MISMATCH + discriminator above)")
    print("  PASS: fields-CPU within 1e-3 after", N_STEPS, "steps")

    print("test_euler_fields: ALL PASS")
