"""Benchmark Mojo idiomatic CPU perf: scalar vs SIMD vs parallelize.

Question this answers:
    "Now that `linalg.matmul[target="cpu"]` gave us 250-762× on big GEMMs, where
     ELSE in the nn library can we get cheap wins from Mojo's CPU SIMD and
     parallelize primitives, without touching Accelerate?"

Tests the patterns common across `mojo_rl/nn/`:
- Elementwise unary (ReLU)                          ← cheap baseline
- Elementwise transcendental (Tanh, Mish)           ← biggest expected win
- Elementwise read-modify-write (Adam.step)         ← bandwidth-bound + FMA
- Per-row reductions (LayerNorm forward)            ← reduce + broadcast
- BATCH-parallel composite (LayerNorm parallelize)  ← test parallelize threshold

Uses explicit SIMD load/store loops instead of `vectorize[...]` helper —
emits the same code but is easier to reason about for a microbench.

Run:
    pixi run mojo run -I . benchmarks/benchmark_vectorize_cpu.mojo
"""

from std.algorithm.functional import parallelize
from std.math import tanh, exp, sqrt, log
from std.memory import alloc
from std.random import seed, random_float64
from std.sys import simd_width_of
from std.time import perf_counter_ns

from mojo_rl.nn.constants import dtype


comptime SIMD_WIDTH = simd_width_of[dtype]()


# =============================================================================
# Helpers
# =============================================================================


def fill_random(p: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = Scalar[dtype](random_float64(-1.0, 1.0))


def max_abs_diff(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
) -> Float64:
    var m: Float64 = 0.0
    for i in range(n):
        var d = Float64(a[i]) - Float64(b[i])
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


# =============================================================================
# ReLU
# =============================================================================


def relu_scalar(
    inp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    res: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    for i in range(n):
        var v = inp[i]
        res[i] = v if v > 0 else 0


def relu_simd(
    inp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    res: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    var i = 0
    var zero_v = SIMD[dtype, SIMD_WIDTH](0)
    while i + SIMD_WIDTH <= n:
        var v = inp.load[width=SIMD_WIDTH](i)
        var mask = v.gt(zero_v)
        res.store(i, mask.select(v, zero_v))
        i += SIMD_WIDTH
    while i < n:
        var v = inp[i]
        res[i] = v if v > 0 else 0
        i += 1


# =============================================================================
# Tanh
# =============================================================================


def tanh_scalar(
    inp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    res: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    for i in range(n):
        res[i] = tanh(inp[i])


def tanh_simd(
    inp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    res: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    var i = 0
    while i + SIMD_WIDTH <= n:
        var v = inp.load[width=SIMD_WIDTH](i)
        res.store(i, tanh(v))
        i += SIMD_WIDTH
    while i < n:
        res[i] = tanh(inp[i])
        i += 1


# =============================================================================
# Mish: x * tanh(log(1 + exp(x)))
# =============================================================================


def mish_scalar(
    inp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    res: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    for i in range(n):
        var x = inp[i]
        res[i] = x * tanh(log(Scalar[dtype](1) + exp(x)))


def mish_simd(
    inp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    res: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    var i = 0
    var one_v = SIMD[dtype, SIMD_WIDTH](1)
    while i + SIMD_WIDTH <= n:
        var x = inp.load[width=SIMD_WIDTH](i)
        res.store(i, x * tanh(log(one_v + exp(x))))
        i += SIMD_WIDTH
    var one = Scalar[dtype](1)
    while i < n:
        var x = inp[i]
        res[i] = x * tanh(log(one + exp(x)))
        i += 1


# =============================================================================
# Adam step
# =============================================================================


def adam_scalar(
    params: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grads: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    state: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    lr: Scalar[dtype],
    bc1: Scalar[dtype],
    bc2: Scalar[dtype],
    omb1: Scalar[dtype],
    omb2: Scalar[dtype],
    b1: Scalar[dtype],
    b2: Scalar[dtype],
    eps: Scalar[dtype],
):
    for i in range(n):
        var g = grads[i]
        var m = state[2 * i]
        var v = state[2 * i + 1]
        var m_new = b1 * m + omb1 * g
        var v_new = b2 * v + omb2 * g * g
        state[2 * i] = m_new
        state[2 * i + 1] = v_new
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        params[i] -= lr * m_hat / (sqrt(v_hat) + eps)


def adam_simd(
    params: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grads: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    state_m: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    state_v: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    lr: Scalar[dtype],
    bc1: Scalar[dtype],
    bc2: Scalar[dtype],
    omb1: Scalar[dtype],
    omb2: Scalar[dtype],
    b1: Scalar[dtype],
    b2: Scalar[dtype],
    eps: Scalar[dtype],
):
    var i = 0
    while i + SIMD_WIDTH <= n:
        var g = grads.load[width=SIMD_WIDTH](i)
        var m = state_m.load[width=SIMD_WIDTH](i)
        var v = state_v.load[width=SIMD_WIDTH](i)
        var m_new = b1 * m + omb1 * g
        var v_new = b2 * v + omb2 * g * g
        state_m.store(i, m_new)
        state_v.store(i, v_new)
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        var p = params.load[width=SIMD_WIDTH](i)
        params.store(i, p - lr * m_hat / (sqrt(v_hat) + eps))
        i += SIMD_WIDTH
    while i < n:
        var g = grads[i]
        var m = state_m[i]
        var v = state_v[i]
        var m_new = b1 * m + omb1 * g
        var v_new = b2 * v + omb2 * g * g
        state_m[i] = m_new
        state_v[i] = v_new
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        params[i] -= lr * m_hat / (sqrt(v_hat) + eps)
        i += 1


# =============================================================================
# LayerNorm forward
# =============================================================================


def ln_scalar(
    inp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    res: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bta: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    batch: Int,
    dim: Int,
    eps: Scalar[dtype],
):
    for b in range(batch):
        var off = b * dim
        var s: Scalar[dtype] = 0
        for j in range(dim):
            s += inp[off + j]
        var mean = s / Scalar[dtype](dim)
        var vs: Scalar[dtype] = 0
        for j in range(dim):
            var d = inp[off + j] - mean
            vs += d * d
        var inv_std = Scalar[dtype](1) / sqrt(vs / Scalar[dtype](dim) + eps)
        for j in range(dim):
            res[off + j] = (inp[off + j] - mean) * inv_std * gamma[j] + bta[j]


def ln_simd_row(
    inp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    res: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bta: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    off: Int,
    dim: Int,
    eps: Scalar[dtype],
):
    """Compute one LayerNorm row using SIMD reductions + SIMD affine."""
    var s_acc = SIMD[dtype, SIMD_WIDTH](0)
    var s_tail: Scalar[dtype] = 0
    var j = 0
    while j + SIMD_WIDTH <= dim:
        s_acc += inp.load[width=SIMD_WIDTH](off + j)
        j += SIMD_WIDTH
    while j < dim:
        s_tail += inp[off + j]
        j += 1
    var mean = (s_acc.reduce_add() + s_tail) / Scalar[dtype](dim)

    var v_acc = SIMD[dtype, SIMD_WIDTH](0)
    var v_tail: Scalar[dtype] = 0
    var mean_v = SIMD[dtype, SIMD_WIDTH](mean)
    j = 0
    while j + SIMD_WIDTH <= dim:
        var d = inp.load[width=SIMD_WIDTH](off + j) - mean_v
        v_acc += d * d
        j += SIMD_WIDTH
    while j < dim:
        var d = inp[off + j] - mean
        v_tail += d * d
        j += 1
    var vs = v_acc.reduce_add() + v_tail
    var inv_std = Scalar[dtype](1) / sqrt(vs / Scalar[dtype](dim) + eps)
    var inv_std_v = SIMD[dtype, SIMD_WIDTH](inv_std)

    j = 0
    while j + SIMD_WIDTH <= dim:
        var x = inp.load[width=SIMD_WIDTH](off + j)
        var g = gamma.load[width=SIMD_WIDTH](j)
        var t = bta.load[width=SIMD_WIDTH](j)
        res.store(off + j, (x - mean_v) * inv_std_v * g + t)
        j += SIMD_WIDTH
    while j < dim:
        res[off + j] = (inp[off + j] - mean) * inv_std * gamma[j] + bta[j]
        j += 1


def ln_simd(
    inp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    res: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bta: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    batch: Int,
    dim: Int,
    eps: Scalar[dtype],
):
    for b in range(batch):
        ln_simd_row(inp, res, gamma, bta, b * dim, dim, eps)


def ln_par(
    inp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    res: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bta: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    batch: Int,
    dim: Int,
    eps: Scalar[dtype],
):
    @parameter
    def row(b: Int):
        try:
            ln_simd_row(inp, res, gamma, bta, b * dim, dim, eps)
        except e:
            pass

    parallelize[row](batch)


# =============================================================================
# Driver helpers
# =============================================================================


def run_unary(name: String, n: Int, iters: Int) raises:
    var inp = alloc[Scalar[dtype]](n)
    var rs = alloc[Scalar[dtype]](n)
    var rv = alloc[Scalar[dtype]](n)
    seed(42)
    fill_random(inp, n)

    var t_s: Float64 = 0
    var t_v: Float64 = 0
    var err: Float64 = 0

    if name == "ReLU":
        relu_scalar(inp, rs, n)
        relu_simd(inp, rv, n)
        var t0 = perf_counter_ns()
        for _ in range(iters):
            relu_scalar(inp, rs, n)
        var t1 = perf_counter_ns()
        t_s = Float64(t1 - t0) / Float64(iters)
        t0 = perf_counter_ns()
        for _ in range(iters):
            relu_simd(inp, rv, n)
        t1 = perf_counter_ns()
        t_v = Float64(t1 - t0) / Float64(iters)
        err = max_abs_diff(rs, rv, n)
    elif name == "Tanh":
        tanh_scalar(inp, rs, n)
        tanh_simd(inp, rv, n)
        var t0 = perf_counter_ns()
        for _ in range(iters):
            tanh_scalar(inp, rs, n)
        var t1 = perf_counter_ns()
        t_s = Float64(t1 - t0) / Float64(iters)
        t0 = perf_counter_ns()
        for _ in range(iters):
            tanh_simd(inp, rv, n)
        t1 = perf_counter_ns()
        t_v = Float64(t1 - t0) / Float64(iters)
        err = max_abs_diff(rs, rv, n)
    else:
        mish_scalar(inp, rs, n)
        mish_simd(inp, rv, n)
        var t0 = perf_counter_ns()
        for _ in range(iters):
            mish_scalar(inp, rs, n)
        var t1 = perf_counter_ns()
        t_s = Float64(t1 - t0) / Float64(iters)
        t0 = perf_counter_ns()
        for _ in range(iters):
            mish_simd(inp, rv, n)
        t1 = perf_counter_ns()
        t_v = Float64(t1 - t0) / Float64(iters)
        err = max_abs_diff(rs, rv, n)

    print(
        name,
        " n=",
        n,
        " | scalar=",
        Int(t_s / 1000.0),
        "us | simd=",
        Int(t_v / 1000.0),
        "us | speedup=",
        Float64(Int((t_s / t_v) * 100)) / 100.0,
        "x | max_diff=",
        err,
    )


def main() raises:
    print("=" * 90)
    print("Mojo CPU SIMD / parallelize bench")
    print("  dtype =", dtype)
    print("  SIMD_WIDTH =", SIMD_WIDTH)
    print("=" * 90)

    print("\n--- ReLU ---")
    run_unary("ReLU", 256 * 256, 500)
    run_unary("ReLU", 1024 * 1024, 200)
    run_unary("ReLU", 8 * 1024 * 1024, 30)

    print("\n--- Tanh ---")
    run_unary("Tanh", 256 * 256, 500)
    run_unary("Tanh", 1024 * 1024, 200)
    run_unary("Tanh", 8 * 1024 * 1024, 20)

    print("\n--- Mish ---")
    run_unary("Mish", 256 * 256, 200)
    run_unary("Mish", 1024 * 1024, 50)
    run_unary("Mish", 8 * 1024 * 1024, 10)

    print("\n--- Adam.step ---")
    var adam_sizes = List[Int]()
    adam_sizes.append(32_768)
    adam_sizes.append(262_144)
    adam_sizes.append(1_048_576)
    adam_sizes.append(8_388_608)
    for n in adam_sizes:
        var params_s = alloc[Scalar[dtype]](n)
        var grads_s = alloc[Scalar[dtype]](n)
        var state_s = alloc[Scalar[dtype]](2 * n)
        var params_v = alloc[Scalar[dtype]](n)
        var grads_v = alloc[Scalar[dtype]](n)
        var sm_v = alloc[Scalar[dtype]](n)
        var sv_v = alloc[Scalar[dtype]](n)
        seed(42)
        fill_random(params_s, n)
        fill_random(grads_s, n)
        for i in range(n):
            params_v[i] = params_s[i]
            grads_v[i] = grads_s[i]
            state_s[2 * i] = 0
            state_s[2 * i + 1] = 0
            sm_v[i] = 0
            sv_v[i] = 0

        var lr = Scalar[dtype](0.001)
        var bc1 = Scalar[dtype](0.1)
        var bc2 = Scalar[dtype](0.001)
        var omb1 = Scalar[dtype](0.1)
        var omb2 = Scalar[dtype](0.001)
        var b1 = Scalar[dtype](0.9)
        var b2 = Scalar[dtype](0.999)
        var eps = Scalar[dtype](1e-8)

        adam_scalar(
            params_s,
            grads_s,
            state_s,
            n,
            lr,
            bc1,
            bc2,
            omb1,
            omb2,
            b1,
            b2,
            eps,
        )
        adam_simd(
            params_v,
            grads_v,
            sm_v,
            sv_v,
            n,
            lr,
            bc1,
            bc2,
            omb1,
            omb2,
            b1,
            b2,
            eps,
        )
        var iters = max(3, 10_000_000 // n)
        var t0 = perf_counter_ns()
        for _ in range(iters):
            adam_scalar(
                params_s,
                grads_s,
                state_s,
                n,
                lr,
                bc1,
                bc2,
                omb1,
                omb2,
                b1,
                b2,
                eps,
            )
        var t1 = perf_counter_ns()
        var t_s = Float64(t1 - t0) / Float64(iters)
        t0 = perf_counter_ns()
        for _ in range(iters):
            adam_simd(
                params_v,
                grads_v,
                sm_v,
                sv_v,
                n,
                lr,
                bc1,
                bc2,
                omb1,
                omb2,
                b1,
                b2,
                eps,
            )
        t1 = perf_counter_ns()
        var t_v = Float64(t1 - t0) / Float64(iters)
        print(
            "Adam n=",
            n,
            "| scalar=",
            Int(t_s / 1000.0),
            "us | simd=",
            Int(t_v / 1000.0),
            "us | speedup=",
            Float64(Int((t_s / t_v) * 100)) / 100.0,
            "x",
        )

    print("\n--- LayerNorm forward ---")
    var ln_batches = List[Int]()
    ln_batches.append(32)
    ln_batches.append(128)
    ln_batches.append(512)
    var ln_dims = List[Int]()
    ln_dims.append(128)
    ln_dims.append(512)
    ln_dims.append(2048)
    for batch in ln_batches:
        for dim in ln_dims:
            var inp = alloc[Scalar[dtype]](batch * dim)
            var rs = alloc[Scalar[dtype]](batch * dim)
            var rv = alloc[Scalar[dtype]](batch * dim)
            var rp = alloc[Scalar[dtype]](batch * dim)
            var gamma = alloc[Scalar[dtype]](dim)
            var bta = alloc[Scalar[dtype]](dim)
            seed(42)
            fill_random(inp, batch * dim)
            for j in range(dim):
                gamma[j] = 1.0
                bta[j] = 0.0
            var eps = Scalar[dtype](1e-5)
            ln_scalar(inp, rs, gamma, bta, batch, dim, eps)
            ln_simd(inp, rv, gamma, bta, batch, dim, eps)
            ln_par(inp, rp, gamma, bta, batch, dim, eps)
            var iters = max(5, 2_000_000 // (batch * dim))
            var t0 = perf_counter_ns()
            for _ in range(iters):
                ln_scalar(inp, rs, gamma, bta, batch, dim, eps)
            var t1 = perf_counter_ns()
            var t_s = Float64(t1 - t0) / Float64(iters)
            t0 = perf_counter_ns()
            for _ in range(iters):
                ln_simd(inp, rv, gamma, bta, batch, dim, eps)
            t1 = perf_counter_ns()
            var t_v = Float64(t1 - t0) / Float64(iters)
            t0 = perf_counter_ns()
            for _ in range(iters):
                ln_par(inp, rp, gamma, bta, batch, dim, eps)
            t1 = perf_counter_ns()
            var t_p = Float64(t1 - t0) / Float64(iters)
            var err_v = max_abs_diff(rs, rv, batch * dim)
            var err_p = max_abs_diff(rs, rp, batch * dim)
            print(
                "LN b=",
                batch,
                "d=",
                dim,
                "| scalar=",
                Int(t_s / 1000.0),
                "us | simd=",
                Int(t_v / 1000.0),
                "us (",
                Float64(Int((t_s / t_v) * 100)) / 100.0,
                "x) | par=",
                Int(t_p / 1000.0),
                "us (",
                Float64(Int((t_s / t_p) * 100)) / 100.0,
                "x) | err_v=",
                err_v,
                " err_p=",
                err_p,
            )

    print("\n" + "=" * 90)
    print("Done.")
    print("=" * 90)
