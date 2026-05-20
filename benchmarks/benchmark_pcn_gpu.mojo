"""Wall-clock benchmark for PCN compute_grads_only (CPU vs GPU).

Measures the cost of one full PCN inference + grad pass at a
training-realistic shape:

    BATCH = 64, arch: 64 → 256 → 256 → 128 → 10, T_INFER = 8

For Apple, GPU runs through the 2×2 register-tiled kernels. For NVIDIA with
USE_MAX_KERNELS=True (default), GPU runs through `linalg.matmul` (max_matmul).

Run:
    pixi run -e apple  mojo run -I . benchmarks/benchmark_pcn_gpu.mojo
    pixi run -e nvidia mojo run -I . benchmarks/benchmark_pcn_gpu.mojo
"""

from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Xavier
from mojo_rl.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)


comptime BATCH = 256
comptime T_INFER = 8
comptime LR_X: Float64 = 0.05

comptime WARMUP_ITERS = 5
comptime BENCH_ITERS = 30

# Larger PCN, training-realistic for MNIST-class workloads.
comptime NET = PCSequential[
    PCBlock[784, 512, PCIdentity],
    PCBlock[512, 512, PCReLU],
    PCBlock[512, 256, PCReLU],
    PCBlock[256, 10, PCReLU],
]
comptime TRAINER = PCTrainer[
    PCBlock[784, 512, PCIdentity],
    PCBlock[512, 512, PCReLU],
    PCBlock[512, 256, PCReLU],
    PCBlock[256, 10, PCReLU],
    dtype=dtype,
]


def main() raises:
    print("=" * 60)
    print("pcn compute_grads_only benchmark")
    print("=" * 60)
    print("  arch       : 784 → 512 → 512 → 256 → 10")
    print("  PARAM_SIZE :", NET.PARAM_SIZE)
    print("  BATCH=", BATCH, " T_INFER=", T_INFER)
    print("  warmup=", WARMUP_ITERS, " bench=", BENCH_ITERS, "iterations")

    var ctx = DeviceContext()

    # ── Init params + input/target (Xavier, deterministic Philox) ─────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    memset(params_buf, 0, NET.PARAM_SIZE)
    var params = LayoutTensor[
        dtype, Layout.row_major(NET.PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    NET.initialize_params[Xavier[], dtype](params)

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
    # CPU benchmark
    # =================================================================
    var grads_cpu_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE)
    var lat_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM)
    var mu_eps_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM)
    var a_below_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM)
    var z_below_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM)
    var dx_cpu_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM)

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

    print("\n  CPU warmup ...")
    for _ in range(WARMUP_ITERS):
        memset(grads_cpu_buf, 0, NET.PARAM_SIZE)
        _ = TRAINER.compute_grads_only[BATCH](
            params, grads_cpu, lat_cpu, mu_eps_cpu, a_below_cpu,
            z_below_cpu, dx_cpu, x_in, y_target,
            T_infer=T_INFER, lr_x=Scalar[dtype](LR_X),
        )

    print("  CPU benchmarking ...")
    var cpu_total_ns: UInt = 0
    for _ in range(BENCH_ITERS):
        memset(grads_cpu_buf, 0, NET.PARAM_SIZE)
        var t0 = perf_counter_ns()
        _ = TRAINER.compute_grads_only[BATCH](
            params, grads_cpu, lat_cpu, mu_eps_cpu, a_below_cpu,
            z_below_cpu, dx_cpu, x_in, y_target,
            T_infer=T_INFER, lr_x=Scalar[dtype](LR_X),
        )
        cpu_total_ns += perf_counter_ns() - t0
    var cpu_mean_ms = Float64(cpu_total_ns) / Float64(BENCH_ITERS) / 1.0e6

    # =================================================================
    # GPU benchmark
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

    print("\n  GPU warmup ...")
    for _ in range(WARMUP_ITERS):
        TRAINER.compute_grads_only_gpu[BATCH](
            ctx, params_t_gpu, grads_t_gpu, lat_t_gpu, mu_eps_t_gpu,
            a_below_t_gpu, z_below_t_gpu, dx_t_gpu, x_in_t_gpu, y_target_t_gpu,
            T_infer=T_INFER, lr_x=Scalar[dtype](LR_X),
        )
    ctx.synchronize()

    print("  GPU benchmarking ...")
    var gpu_total_ns: UInt = 0
    for _ in range(BENCH_ITERS):
        var t0 = perf_counter_ns()
        TRAINER.compute_grads_only_gpu[BATCH](
            ctx, params_t_gpu, grads_t_gpu, lat_t_gpu, mu_eps_t_gpu,
            a_below_t_gpu, z_below_t_gpu, dx_t_gpu, x_in_t_gpu, y_target_t_gpu,
            T_infer=T_INFER, lr_x=Scalar[dtype](LR_X),
        )
        ctx.synchronize()
        gpu_total_ns += perf_counter_ns() - t0
    var gpu_mean_ms = Float64(gpu_total_ns) / Float64(BENCH_ITERS) / 1.0e6

    print("\n  Results:")
    print("    CPU mean :", cpu_mean_ms, "ms / iter")
    print("    GPU mean :", gpu_mean_ms, "ms / iter")
    print("    speedup  :", cpu_mean_ms / gpu_mean_ms, "x")

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
