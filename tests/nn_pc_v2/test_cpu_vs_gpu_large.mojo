"""CPU vs GPU parity test for nn_pc_v2 at realistic dimensions.

Stresses the optimized GPU matmul path (max_matmul on NVIDIA, naive fallback
elsewhere) with a 4-block PCN sized to actually exercise the GEMM tiling:

    BATCH = 64, arch: 64 → 256 → 256 → 128 → 10

At these dims the tile-blocked reduction order in `linalg.matmul` no longer
matches the CPU's natural row-major order, so we use a coarser tolerance
than `test_cpu_vs_gpu.mojo` (1e-3 instead of 1e-4).

Run:
    pixi run -e apple  mojo run -I . tests/nn_pc_v2/test_cpu_vs_gpu_large.mojo
    pixi run -e nvidia mojo run -I . tests/nn_pc_v2/test_cpu_vs_gpu_large.mojo
"""

from std.math import abs as mabs
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier
from mojo_rl.experimental.nn_pc_v2 import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)


comptime BATCH = 64
comptime T_INFER = 4
comptime LR_X: Float64 = 0.05

comptime NET = PCSequential[
    PCBlock[64, 256, PCIdentity],
    PCBlock[256, 256, PCReLU],
    PCBlock[256, 128, PCReLU],
    PCBlock[128, 10, PCReLU],
]
comptime TRAINER = PCTrainer[
    PCBlock[64, 256, PCIdentity],
    PCBlock[256, 256, PCReLU],
    PCBlock[256, 128, PCReLU],
    PCBlock[128, 10, PCReLU],
    dtype=dtype,
]

# Tile-blocked GEMM reductions diverge from CPU natural order at these sizes;
# 1e-3 absolute tolerance is comfortable while still catching real bugs.
comptime TOL: Float64 = 1.0e-3


def main() raises:
    print("=" * 60)
    print("nn_pc_v2 CPU vs GPU parity test (large dims)")
    print("=" * 60)
    print("  arch       : 64 → 256 → 256 → 128 → 10")
    print("  PARAM_SIZE :", NET.PARAM_SIZE)
    print("  LATENT_DIM :", NET.LATENT_DIM)
    print("  BATCH=", BATCH, " T_INFER=", T_INFER)
    print("  tolerance  :", TOL)

    var ctx = DeviceContext()

    # ── Allocate params on host, init with Xavier (deterministic seed) ────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    memset(params_buf, 0, NET.PARAM_SIZE)
    var params = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    NET.initialize_params[Xavier[], dtype](params)

    # ── Allocate input + target (deterministic Philox) ────────────────────────
    var x_in_buf = alloc[Scalar[dtype]](BATCH * NET.IN_DIM)
    var y_tgt_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM)
    var rng = PhiloxRandom(seed=UInt64(13), offset=UInt64(0))
    for i in range(BATCH * NET.IN_DIM):
        var r = rng.step_uniform()
        x_in_buf[i] = Scalar[dtype](Float32(r[0]) * 2.0 - 1.0)
    for i in range(BATCH * NET.OUT_DIM):
        y_tgt_buf[i] = Scalar[dtype](0)
    for b in range(BATCH):
        y_tgt_buf[b * NET.OUT_DIM + (b % NET.OUT_DIM)] = Scalar[dtype](1.0)

    var x_in = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.IN_DIM), MutAnyOrigin
    ](x_in_buf)
    var y_target = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.OUT_DIM), MutAnyOrigin
    ](y_tgt_buf)

    # =================================================================
    # CPU run
    # =================================================================
    var grads_cpu_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    var lat_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM)
    var mu_eps_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM)
    var a_below_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM)
    var z_below_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM)
    var dx_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM)
    memset(grads_cpu_buf, 0, NET.PARAM_SIZE)
    memset(lat_cpu_buf, 0, BATCH * NET.LATENT_DIM)
    memset(mu_eps_cpu_buf, 0, BATCH * NET.SCRATCH_OUT_DIM)
    memset(a_below_cpu_buf, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(z_below_cpu_buf, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(dx_cpu_buf, 0, BATCH * NET.LATENT_DIM)

    var grads_cpu = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](grads_cpu_buf)
    var lat_cpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](lat_cpu_buf)
    var mu_eps_cpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_OUT_DIM), MutAnyOrigin
    ](mu_eps_cpu_buf)
    var a_below_cpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](a_below_cpu_buf)
    var z_below_cpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](z_below_cpu_buf)
    var dx_cpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](dx_cpu_buf)

    var cpu_result = TRAINER.compute_grads_only[BATCH](
        params,
        grads_cpu,
        lat_cpu,
        mu_eps_cpu,
        a_below_cpu,
        z_below_cpu,
        dx_cpu,
        x_in,
        y_target,
        T_infer=T_INFER,
        lr_x=Scalar[dtype](LR_X),
    )

    print("\n  CPU run:")
    print("    energy_initial:", cpu_result.energy_initial)
    print("    energy_final  :", cpu_result.energy_final)
    print("    output_loss   :", cpu_result.output_loss_final)

    # =================================================================
    # GPU run — same params, same input/target
    # =================================================================
    var params_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var grads_dbuf = ctx.enqueue_create_buffer[dtype](NET.PARAM_SIZE)
    var lat_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var mu_eps_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_OUT_DIM)
    var a_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var z_below_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.SCRATCH_IN_DIM)
    var dx_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.LATENT_DIM)
    var x_in_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.IN_DIM)
    var y_target_dbuf = ctx.enqueue_create_buffer[dtype](BATCH * NET.OUT_DIM)

    var params_host = ctx.enqueue_create_host_buffer[dtype](NET.PARAM_SIZE)
    for i in range(NET.PARAM_SIZE):
        params_host.unsafe_ptr()[i] = params_buf[i]
    ctx.enqueue_copy(params_dbuf, params_host)

    var x_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NET.IN_DIM)
    for i in range(BATCH * NET.IN_DIM):
        x_host.unsafe_ptr()[i] = x_in_buf[i]
    ctx.enqueue_copy(x_in_dbuf, x_host)

    var y_host = ctx.enqueue_create_host_buffer[dtype](BATCH * NET.OUT_DIM)
    for i in range(BATCH * NET.OUT_DIM):
        y_host.unsafe_ptr()[i] = y_tgt_buf[i]
    ctx.enqueue_copy(y_target_dbuf, y_host)

    var params_t_gpu = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](params_dbuf)
    var grads_t_gpu = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](grads_dbuf)
    var lat_t_gpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](lat_dbuf)
    var mu_eps_t_gpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_OUT_DIM), MutAnyOrigin
    ](mu_eps_dbuf)
    var a_below_t_gpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](a_below_dbuf)
    var z_below_t_gpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.SCRATCH_IN_DIM), MutAnyOrigin
    ](z_below_dbuf)
    var dx_t_gpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.LATENT_DIM), MutAnyOrigin
    ](dx_dbuf)
    var x_in_t_gpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.IN_DIM), MutAnyOrigin
    ](x_in_dbuf)
    var y_target_t_gpu = LayoutTensor[
        dtype, Layout.row_major(BATCH, NET.OUT_DIM), MutAnyOrigin
    ](y_target_dbuf)

    TRAINER.compute_grads_only_gpu[BATCH](
        ctx,
        params_t_gpu,
        grads_t_gpu,
        lat_t_gpu,
        mu_eps_t_gpu,
        a_below_t_gpu,
        z_below_t_gpu,
        dx_t_gpu,
        x_in_t_gpu,
        y_target_t_gpu,
        T_infer=T_INFER,
        lr_x=Scalar[dtype](LR_X),
    )
    ctx.synchronize()

    var grads_gpu_host = ctx.enqueue_create_host_buffer[dtype](NET.PARAM_SIZE)
    ctx.enqueue_copy(grads_gpu_host, grads_dbuf)
    var lat_gpu_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * NET.LATENT_DIM
    )
    ctx.enqueue_copy(lat_gpu_host, lat_dbuf)
    ctx.synchronize()

    # =================================================================
    # Compare
    # =================================================================
    var max_grad_diff: Float64 = 0.0
    var idx_max_grad: Int = 0
    for i in range(NET.PARAM_SIZE):
        var g_cpu = Float64(grads_cpu_buf[i])
        var g_gpu = Float64(grads_gpu_host.unsafe_ptr()[i])
        var d = mabs(g_cpu - g_gpu)
        if d > max_grad_diff:
            max_grad_diff = d
            idx_max_grad = i

    var max_lat_diff: Float64 = 0.0
    var idx_max_lat: Int = 0
    for i in range(BATCH * NET.LATENT_DIM):
        var l_cpu = Float64(lat_cpu_buf[i])
        var l_gpu = Float64(lat_gpu_host.unsafe_ptr()[i])
        var d = mabs(l_cpu - l_gpu)
        if d > max_lat_diff:
            max_lat_diff = d
            idx_max_lat = i

    print("\n  Parity:")
    print("    max |grads_cpu - grads_gpu|   :", max_grad_diff, " at idx", idx_max_grad)
    print("    max |latents_cpu - latents_gpu|:", max_lat_diff, " at idx", idx_max_lat)
    print("    tolerance                     :", TOL)

    if max_grad_diff <= TOL and max_lat_diff <= TOL:
        print("\n  [PASS] CPU and GPU agree within tolerance")
    else:
        print("\n  [FAIL] CPU vs GPU disagreement exceeds tolerance")
        raise Error("parity test failed")

    params_buf.free()
    x_in_buf.free()
    y_tgt_buf.free()
    grads_cpu_buf.free()
    lat_cpu_buf.free()
    mu_eps_cpu_buf.free()
    a_below_cpu_buf.free()
    z_below_cpu_buf.free()
    dx_cpu_buf.free()
    print("=== Done ===")
