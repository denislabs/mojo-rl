"""CPU vs GPU equivalence test for nn_pc.

Same fixed-seed Xavier init, same Philox-randn latent init, same
hyperparameters. Run one batch on each backend, then compare post-training
params + latents element-wise. Disagreement above ~1e-4 indicates a GPU bug.

Architecture mirrors the smoke test:
    PCLinear[3, 5]
    PCLinear[5, 4]
    PCLinear[2, 4, PCIdentity]

Run:
    pixi run -e apple  mojo run -I . tests/nn_pc/test_pc_cpu_vs_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn_pc/test_pc_cpu_vs_gpu.mojo
"""

from std.math import abs
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn_pc import PCLinear, PCSequential, PCIdentity, PCTrainer


comptime BATCH = 4
comptime T_INFER = 5
comptime T_LEARN = 8

comptime TRAINER = PCTrainer[
    PCLinear[3, 5],
    PCLinear[5, 4],
    PCLinear[2, 4, PCIdentity],
    dtype=dtype,
]


def main() raises:
    print("=== nn_pc CPU vs GPU equivalence ===")
    print("  PARAM_SIZE              =", TRAINER.MODEL.PARAM_SIZE)
    print("  LATENT_SIZE_PER_SAMPLE  =", TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE)
    print("  BATCH=", BATCH, "T_INFER=", T_INFER, "T_LEARN=", T_LEARN)

    var ctx = DeviceContext()

    # ── Host-side initial state (identical for CPU and GPU runs) ──
    var p_init = alloc[Scalar[dtype]](TRAINER.MODEL.PARAM_SIZE)
    memset(p_init, 0, TRAINER.MODEL.PARAM_SIZE)
    var p_init_t = LayoutTensor[
        dtype, Layout.row_major(TRAINER.MODEL.PARAM_SIZE), MutAnyOrigin
    ](p_init)
    TRAINER.MODEL.initialize_params[Xavier[], dtype](p_init_t)

    var lat_init = alloc[Scalar[dtype]](
        BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE
    )
    memset(lat_init, 0, BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE)
    var lat_init_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE),
        MutAnyOrigin,
    ](lat_init)
    TRAINER.randn_init_latents[BATCH](lat_init_t, seed=UInt64(7777), offset=UInt64(0))

    # Inputs + targets via Philox (same for both)
    var x_in = alloc[Scalar[dtype]](BATCH * TRAINER.MODEL.IN_DIM)
    var y_tgt = alloc[Scalar[dtype]](BATCH * TRAINER.MODEL.OUT_DIM)
    memset(x_in, 0, BATCH * TRAINER.MODEL.IN_DIM)
    memset(y_tgt, 0, BATCH * TRAINER.MODEL.OUT_DIM)

    var rng = PhiloxRandom(seed=UInt64(2024), offset=UInt64(0))
    for i in range(BATCH * TRAINER.MODEL.IN_DIM):
        var r = rng.step_uniform()
        x_in[i] = Scalar[dtype](Float32(r[0]) * 2.0 - 1.0)
    # one-hot targets: sample b -> class b % NUM_CLASSES
    for b in range(BATCH):
        y_tgt[b * TRAINER.MODEL.OUT_DIM + (b % TRAINER.MODEL.OUT_DIM)] = (
            Scalar[dtype](1.0)
        )

    # ── CPU run ──
    print("\n[CPU] running...")
    var p_cpu = alloc[Scalar[dtype]](TRAINER.MODEL.PARAM_SIZE)
    var lat_cpu = alloc[Scalar[dtype]](
        BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE
    )
    for i in range(TRAINER.MODEL.PARAM_SIZE):
        p_cpu[i] = p_init[i]
    for i in range(BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE):
        lat_cpu[i] = lat_init[i]

    var p_cpu_t = LayoutTensor[
        dtype, Layout.row_major(TRAINER.MODEL.PARAM_SIZE), MutAnyOrigin
    ](p_cpu)
    var lat_cpu_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE),
        MutAnyOrigin,
    ](lat_cpu)
    var x_cpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TRAINER.MODEL.IN_DIM), MutAnyOrigin
    ](x_in)
    var y_cpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TRAINER.MODEL.OUT_DIM), MutAnyOrigin
    ](y_tgt)

    var cpu_res = TRAINER.train_one_batch[BATCH](
        p_cpu_t, lat_cpu_t, x_cpu_t, y_cpu_t,
        T_infer=T_INFER, T_learn=T_LEARN,
        eta_infer=Scalar[dtype](0.05),
        eta_learn=Scalar[dtype](0.005),
    )
    print("  CPU final energy:", cpu_res.energy, "sup_loss:", cpu_res.sup_loss)

    # ── GPU run ──
    print("\n[GPU] running...")
    var p_h_gpu = ctx.enqueue_create_host_buffer[dtype](TRAINER.MODEL.PARAM_SIZE)
    var lat_h_gpu = ctx.enqueue_create_host_buffer[dtype](
        BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE
    )
    var x_h_gpu = ctx.enqueue_create_host_buffer[dtype](BATCH * TRAINER.MODEL.IN_DIM)
    var y_h_gpu = ctx.enqueue_create_host_buffer[dtype](BATCH * TRAINER.MODEL.OUT_DIM)
    for i in range(TRAINER.MODEL.PARAM_SIZE):
        p_h_gpu.unsafe_ptr()[i] = p_init[i]
    for i in range(BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE):
        lat_h_gpu.unsafe_ptr()[i] = lat_init[i]
    for i in range(BATCH * TRAINER.MODEL.IN_DIM):
        x_h_gpu.unsafe_ptr()[i] = x_in[i]
    for i in range(BATCH * TRAINER.MODEL.OUT_DIM):
        y_h_gpu.unsafe_ptr()[i] = y_tgt[i]

    var p_dbuf = ctx.enqueue_create_buffer[dtype](TRAINER.MODEL.PARAM_SIZE)
    var lat_dbuf = ctx.enqueue_create_buffer[dtype](
        BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE
    )
    var x_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * TRAINER.MODEL.IN_DIM)
    var y_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * TRAINER.MODEL.OUT_DIM)
    ctx.enqueue_copy(p_dbuf, p_h_gpu)
    ctx.enqueue_copy(lat_dbuf, lat_h_gpu)
    ctx.enqueue_copy(x_dbuf, x_h_gpu)
    ctx.enqueue_copy(y_dbuf, y_h_gpu)

    var p_gpu_t = LayoutTensor[
        dtype, Layout.row_major(TRAINER.MODEL.PARAM_SIZE), MutAnyOrigin
    ](p_dbuf)
    var lat_gpu_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE),
        MutAnyOrigin,
    ](lat_dbuf)
    var x_gpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TRAINER.MODEL.IN_DIM), MutAnyOrigin
    ](x_dbuf)
    var y_gpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TRAINER.MODEL.OUT_DIM), MutAnyOrigin
    ](y_dbuf)

    TRAINER.train_one_batch_gpu[BATCH](
        ctx, p_gpu_t, lat_gpu_t, x_gpu_t, y_gpu_t,
        T_infer=T_INFER, T_learn=T_LEARN,
        eta_infer=Scalar[dtype](0.05),
        eta_learn=Scalar[dtype](0.005),
    )

    # Copy GPU results back to host for comparison
    var p_back = ctx.enqueue_create_host_buffer[dtype](TRAINER.MODEL.PARAM_SIZE)
    var lat_back = ctx.enqueue_create_host_buffer[dtype](
        BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE
    )
    ctx.enqueue_copy(p_back, p_dbuf)
    ctx.enqueue_copy(lat_back, lat_dbuf)
    ctx.synchronize()
    print("  GPU run complete.")

    # ── Compare ──
    var TOL: Float64 = 1.0e-4
    var max_p_err: Float64 = 0.0
    var max_l_err: Float64 = 0.0
    var argmax_p = 0
    var argmax_l = 0
    for i in range(TRAINER.MODEL.PARAM_SIZE):
        var diff = abs(Float64(p_cpu[i]) - Float64(p_back.unsafe_ptr()[i]))
        if diff > max_p_err:
            max_p_err = diff
            argmax_p = i
    for i in range(BATCH * TRAINER.MODEL.LATENT_SIZE_PER_SAMPLE):
        var diff = abs(
            Float64(lat_cpu[i]) - Float64(lat_back.unsafe_ptr()[i])
        )
        if diff > max_l_err:
            max_l_err = diff
            argmax_l = i

    print("\n  max param error :", max_p_err, " at idx", argmax_p)
    print("  max latent error:", max_l_err, " at idx", argmax_l)

    var ok = True
    if max_p_err >= TOL:
        print("  [FAIL] params disagree above tol", TOL)
        print(
            "    cpu[", argmax_p, "] =", Float64(p_cpu[argmax_p]),
            "  gpu[", argmax_p, "] =", Float64(p_back.unsafe_ptr()[argmax_p]),
        )
        ok = False
    else:
        print("  [PASS] params CPU == GPU within", TOL)
    if max_l_err >= TOL:
        print("  [FAIL] latents disagree above tol", TOL)
        ok = False
    else:
        print("  [PASS] latents CPU == GPU within", TOL)

    p_init.free()
    lat_init.free()
    x_in.free()
    y_tgt.free()
    p_cpu.free()
    lat_cpu.free()

    if not ok:
        raise Error("CPU vs GPU equivalence failed")
    print("=== Done ===")
