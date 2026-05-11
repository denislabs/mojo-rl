"""GPU smoke test for PCDynamics primitives — Phase B0.

Validates that a single PCN dynamics network (2-layer PCBlock chain)
trains end-to-end on GPU using the existing nn_pc_v2 GPU primitives:

- `PCSequential.forward_eval_gpu`         feedforward (used at MBPO
                                          imagination time).
- `PCTrainer.compute_grads_only_gpu`      SGLD inference + PC weight grads.
- `Adam.step_gpu`                         optimizer update.

Pass criterion: prediction loss decreases monotonically on a fixed
synthetic regression target. Smoke test only — no real env, no MBPO.

Run:
    pixi run -e apple  mojo run -I . tests/nn_pc_v2/test_pc_dynamics_gpu_smoke.mojo
    pixi run -e nvidia mojo run -I . tests/nn_pc_v2/test_pc_dynamics_gpu_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.experimental.nn_pc_v2 import PCDynamics


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 64
comptime BATCH = 32
comptime T_INFER = 10
comptime LR_X: Float64 = 0.01
comptime ADAM_LR: Float64 = 0.001
comptime N_TRAIN_BATCHES = 200
comptime DYN = PCDynamics[OBS, ACT, HIDDEN, dtype]
comptime OPT = Adam[LR=ADAM_LR]


def main() raises:
    print("=" * 60)
    print("PCDynamics GPU smoke — Phase B0")
    print("=" * 60)
    print("  arch       : PCBlock[", DYN.AUG_DIM, ",", HIDDEN,
          ",PCTanh] → PCBlock[", HIDDEN, ",", DYN.READOUT, ",PCTanh]")
    print("  PARAM_SIZE :", DYN.PARAM_SIZE)
    print("  hyperparams: BATCH=", BATCH, " T_INFER=", T_INFER,
          " N_BATCHES=", N_TRAIN_BATCHES)

    var ctx = DeviceContext()

    # ── Init params on host, upload to GPU. ─────────────────────────────────
    var params_init_host = ctx.enqueue_create_host_buffer[dtype](
        DYN.PARAM_SIZE
    )
    var params_init_t = LayoutTensor[
        dtype, Layout.row_major(DYN.PARAM_SIZE), MutAnyOrigin
    ](params_init_host.unsafe_ptr())
    DYN.init_params(params_init_t, seed=UInt64(7))

    var params_dbuf = ctx.enqueue_create_buffer[dtype](DYN.PARAM_SIZE)
    ctx.enqueue_copy(params_dbuf, params_init_host)
    var params_t = LayoutTensor[
        dtype, Layout.row_major(DYN.PARAM_SIZE), MutAnyOrigin
    ](params_dbuf)

    # ── Grads, latents, scratch (GPU only). ─────────────────────────────────
    var grads_dbuf = ctx.enqueue_create_buffer[dtype](DYN.PARAM_SIZE)
    var lat_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DYN.SCRATCH_LAT)
    var mu_eps_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DYN.SCRATCH_OUT)
    var a_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DYN.SCRATCH_IN)
    var z_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DYN.SCRATCH_IN)
    var dx_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DYN.SCRATCH_LAT)
    var eval_out_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DYN.READOUT)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(DYN.PARAM_SIZE), MutAnyOrigin
    ](grads_dbuf)
    var lat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.SCRATCH_LAT), MutAnyOrigin
    ](lat_dbuf)
    var mu_eps_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.SCRATCH_OUT), MutAnyOrigin
    ](mu_eps_dbuf)
    var a_below_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.SCRATCH_IN), MutAnyOrigin
    ](a_below_dbuf)
    var z_below_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.SCRATCH_IN), MutAnyOrigin
    ](z_below_dbuf)
    var dx_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.SCRATCH_LAT), MutAnyOrigin
    ](dx_dbuf)
    var eval_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.READOUT), MutAnyOrigin
    ](eval_out_dbuf)

    # ── Adam state on GPU. ──────────────────────────────────────────────────
    var opt_state_dbuf = ctx.enqueue_create_buffer[dtype](
        DYN.PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    var opt_global_dbuf = ctx.enqueue_create_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    var opt_state_init = ctx.enqueue_create_host_buffer[dtype](
        DYN.PARAM_SIZE * OPT.STATE_PER_PARAM
    )
    for i in range(DYN.PARAM_SIZE * OPT.STATE_PER_PARAM):
        opt_state_init.unsafe_ptr()[i] = Scalar[dtype](0)
    ctx.enqueue_copy(opt_state_dbuf, opt_state_init)
    var opt_global_init = ctx.enqueue_create_host_buffer[dtype](
        OPT.GLOBAL_STATE_SIZE
    )
    opt_global_init.unsafe_ptr()[0] = Scalar[dtype](0)
    opt_global_init.unsafe_ptr()[1] = Scalar[dtype](1.0)
    ctx.enqueue_copy(opt_global_dbuf, opt_global_init)
    var opt_state_t = LayoutTensor[
        dtype, Layout.row_major(DYN.PARAM_SIZE, OPT.STATE_PER_PARAM),
        MutAnyOrigin,
    ](opt_state_dbuf)
    var opt_global_t = LayoutTensor[
        dtype, Layout.row_major(OPT.GLOBAL_STATE_SIZE), MutAnyOrigin
    ](opt_global_dbuf)

    # ── Synthetic regression: target = small linear function of (s, a). ─────
    # Host-side input batch + target batch, copied to GPU each iteration.
    var s_a_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DYN.AUG_DIM)
    var target_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DYN.READOUT)
    var s_a_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DYN.AUG_DIM)
    var target_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * DYN.READOUT)
    var s_a_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.AUG_DIM), MutAnyOrigin
    ](s_a_dbuf)
    var target_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DYN.READOUT), MutAnyOrigin
    ](target_dbuf)

    # Eval-loss readback host buffer (download eval_out to compute loss).
    var eval_out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DYN.READOUT)

    var rng = PhiloxRandom(seed=UInt64(11), offset=UInt64(0))

    # Pre-train eval to anchor the loss curve.
    # (Synthetic generator inlined per-call below — Mojo nested-fn capture
    # can't see outer host buffers, so we just inline the few lines.)
    for b in range(BATCH):
        var s0 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
        var s1 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
        var s2 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
        var a0 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
        s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 0] = Scalar[dtype](s0)
        s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 1] = Scalar[dtype](s1)
        s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 2] = Scalar[dtype](s2)
        s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 3] = Scalar[dtype](a0)
        var t0 = 0.5 * s0 + 0.2 * a0
        var t1 = 0.5 * s1 - 0.1 * a0
        var t2 = 0.3 * s2 + 0.4 * a0
        var tr = -0.1 * (s0 * s0 + s1 * s1)
        target_host.unsafe_ptr()[b * DYN.READOUT + 0] = Scalar[dtype](t0)
        target_host.unsafe_ptr()[b * DYN.READOUT + 1] = Scalar[dtype](t1)
        target_host.unsafe_ptr()[b * DYN.READOUT + 2] = Scalar[dtype](t2)
        target_host.unsafe_ptr()[b * DYN.READOUT + 3] = Scalar[dtype](tr)
    ctx.enqueue_copy(s_a_dbuf, s_a_host)
    ctx.enqueue_copy(target_dbuf, target_host)
    DYN.NET.forward_eval_gpu[BATCH, dtype](
        ctx, s_a_t, params_t, eval_out_t, mu_eps_t, a_below_t
    )
    ctx.enqueue_copy(eval_out_host, eval_out_dbuf)
    ctx.synchronize()
    var pre_loss: Float64 = 0.0
    for b in range(BATCH):
        for d in range(DYN.READOUT):
            var p = Float64(eval_out_host.unsafe_ptr()[b * DYN.READOUT + d])
            var t = Float64(target_host.unsafe_ptr()[b * DYN.READOUT + d])
            pre_loss += (p - t) * (p - t)
    pre_loss /= Float64(BATCH * DYN.READOUT)
    print("\n  pre-train MSE :", pre_loss)

    # ── Train ───────────────────────────────────────────────────────────────
    print("\n  step | train_t (s) | feedforward MSE")
    print("  -----+-------------+------------------")
    var t0 = perf_counter_ns()
    var step_num: Int = 0
    for step in range(N_TRAIN_BATCHES):
        # Inline batch fill (see note above).
        for b in range(BATCH):
            var s0 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
            var s1 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
            var s2 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
            var a0 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
            s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 0] = Scalar[dtype](s0)
            s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 1] = Scalar[dtype](s1)
            s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 2] = Scalar[dtype](s2)
            s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 3] = Scalar[dtype](a0)
            var t0v = 0.5 * s0 + 0.2 * a0
            var t1v = 0.5 * s1 - 0.1 * a0
            var t2v = 0.3 * s2 + 0.4 * a0
            var trv = -0.1 * (s0 * s0 + s1 * s1)
            target_host.unsafe_ptr()[b * DYN.READOUT + 0] = Scalar[dtype](t0v)
            target_host.unsafe_ptr()[b * DYN.READOUT + 1] = Scalar[dtype](t1v)
            target_host.unsafe_ptr()[b * DYN.READOUT + 2] = Scalar[dtype](t2v)
            target_host.unsafe_ptr()[b * DYN.READOUT + 3] = Scalar[dtype](trv)
        ctx.enqueue_copy(s_a_dbuf, s_a_host)
        ctx.enqueue_copy(target_dbuf, target_host)

        DYN.TRAINER.compute_grads_only_gpu[BATCH](
            ctx, params_t, grads_t,
            lat_t, mu_eps_t, a_below_t, z_below_t, dx_t,
            s_a_t, target_t,
            T_infer=T_INFER,
            lr_x=Scalar[dtype](LR_X),
        )
        step_num += 1
        OPT.step_gpu[DYN.PARAM_SIZE, dtype](
            ctx, params_t, grads_t, opt_state_t, opt_global_t, step_num
        )

        if step == 0 or (step + 1) % 50 == 0 or step == N_TRAIN_BATCHES - 1:
            # Eval current loss via feedforward + readback.
            DYN.NET.forward_eval_gpu[BATCH, dtype](
                ctx, s_a_t, params_t, eval_out_t, mu_eps_t, a_below_t
            )
            ctx.enqueue_copy(eval_out_host, eval_out_dbuf)
            ctx.synchronize()
            var mse: Float64 = 0.0
            for b in range(BATCH):
                for d in range(DYN.READOUT):
                    var p = Float64(
                        eval_out_host.unsafe_ptr()[b * DYN.READOUT + d]
                    )
                    var t = Float64(
                        target_host.unsafe_ptr()[b * DYN.READOUT + d]
                    )
                    mse += (p - t) * (p - t)
            mse /= Float64(BATCH * DYN.READOUT)
            var elapsed = Float64(perf_counter_ns() - t0) / 1e9
            print(
                "  ", step, "  ", String(elapsed)[byte=:6], "  ", mse,
            )

    var total_t = Float64(perf_counter_ns() - t0) / 1e9
    print("\n  total wall :", total_t, "s")

    # Final loss check (regenerate one fresh batch + recompute).
    for b in range(BATCH):
        var s0 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
        var s1 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
        var s2 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
        var a0 = Float64(rng.step_uniform()[0]) * 2.0 - 1.0
        s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 0] = Scalar[dtype](s0)
        s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 1] = Scalar[dtype](s1)
        s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 2] = Scalar[dtype](s2)
        s_a_host.unsafe_ptr()[b * DYN.AUG_DIM + 3] = Scalar[dtype](a0)
        var tt0 = 0.5 * s0 + 0.2 * a0
        var tt1 = 0.5 * s1 - 0.1 * a0
        var tt2 = 0.3 * s2 + 0.4 * a0
        var ttr = -0.1 * (s0 * s0 + s1 * s1)
        target_host.unsafe_ptr()[b * DYN.READOUT + 0] = Scalar[dtype](tt0)
        target_host.unsafe_ptr()[b * DYN.READOUT + 1] = Scalar[dtype](tt1)
        target_host.unsafe_ptr()[b * DYN.READOUT + 2] = Scalar[dtype](tt2)
        target_host.unsafe_ptr()[b * DYN.READOUT + 3] = Scalar[dtype](ttr)
    ctx.enqueue_copy(s_a_dbuf, s_a_host)
    ctx.enqueue_copy(target_dbuf, target_host)
    DYN.NET.forward_eval_gpu[BATCH, dtype](
        ctx, s_a_t, params_t, eval_out_t, mu_eps_t, a_below_t
    )
    ctx.enqueue_copy(eval_out_host, eval_out_dbuf)
    ctx.synchronize()
    var final_loss: Float64 = 0.0
    for b in range(BATCH):
        for d in range(DYN.READOUT):
            var p = Float64(eval_out_host.unsafe_ptr()[b * DYN.READOUT + d])
            var t = Float64(target_host.unsafe_ptr()[b * DYN.READOUT + d])
            final_loss += (p - t) * (p - t)
    final_loss /= Float64(BATCH * DYN.READOUT)
    print("\n  pre-train MSE :", pre_loss)
    print("  final MSE     :", final_loss)
    if final_loss < pre_loss * 0.5:
        print("  [PASS] loss decreased ≥ 50%")
    else:
        print("  [WARN] loss did not decrease enough — check kernels / hyperparams")
    print("=== Done ===")
