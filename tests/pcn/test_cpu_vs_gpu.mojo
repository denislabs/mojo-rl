"""CPU vs GPU parity test for pcn.

Builds a small 3-block PCN, runs `compute_grads_only` on CPU and
`compute_grads_only_gpu` on GPU starting from identical params + identical
input/target, then compares the resulting grads and latents element-wise.

Tolerance:
  - Apple: ≤1e-4 max abs diff (naive kernels do reductions in the same order
    as CPU, so should be near-bitwise; allow some slack for floating-point).
  - NVIDIA: same tolerance for now (we're using naive kernels, no MMA).

Run:
    pixi run -e apple  mojo run -I . tests/pcn/test_cpu_vs_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/pcn/test_cpu_vs_gpu.mojo
"""

from std.math import abs as mabs
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT as dtype
from mojo_rl.experimental.pcn.pc_initializer import PCXavier
from mojo_rl.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)


comptime BATCH = 4
comptime T_INFER = 5
comptime LR_X: Float64 = 0.05

comptime NET = PCSequential[
    PCBlock[6, 8, PCIdentity],
    PCBlock[8, 5, PCReLU],
    PCBlock[5, 3, PCReLU],
]
comptime TRAINER = PCTrainer[
    PCBlock[6, 8, PCIdentity],
    PCBlock[8, 5, PCReLU],
    PCBlock[5, 3, PCReLU],
    dtype=dtype,
]

comptime TOL: Float64 = 1.0e-4


def main() raises:
    print("=" * 60)
    print("pcn CPU vs GPU parity test")
    print("=" * 60)
    print("  arch       : 6 → 8 → 5 → 3")
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
    NET.pc_init_params[PCXavier, dtype](params)

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

    # Upload params + input/target
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

    # Download grads + latents
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
