"""`Linear.K_PAD` is a FLOOR of 128, but it is also the *N* of the backward.
Should it be a MODULUS of 128 instead?

WHERE THIS COMES FROM
---------------------
`linear.mojo:322` pads the contraction dim with

    PAD_TO = 32,  K_MIN = 128,  K_PAD = max(round_up(IN_, 32), 128)

and the comment above it is explicit that "K is a FLOOR, N is a MODULUS. The
two axes do not obey the same test." That is right — for the FORWARD, where
`K_PAD` is the contraction length. `multi_gemm_cond` is

    m > 1  and  n % 128 == 0  and  k % 32 == 0  and  k >= 128

But `K_PAD` is used TWICE, on two different axes:

    forward      y_pad[B, N_PAD]  = x_pad[B, K_PAD] @ w_pad[K_PAD, N_PAD]
                 K = K_PAD   -> a FLOOR test, satisfied by K_MIN
    grad_input   gi_pad[B, K_PAD] = go_pad[B, N_PAD] @ w_padᵀ
                 N = K_PAD   -> a MODULUS test, NOT satisfied by K_MIN
    grad_weight  dW_pad[K_PAD, N_PAD] = cT_pad @ go_pad
                 M = K_PAD   -> no test; M only affects the tile grid

So the two shapes that comment calls "free" are free in the forward and land
on cuBLAS in the backward:

    IN_ = 160 -> K_PAD = 160   160 % 128 = 32   grad_input -> VENDOR
    IN_ = 192 -> K_PAD = 192   192 % 128 = 64   grad_input -> VENDOR
    IN_ = 518 -> K_PAD = 544   544 % 128 = 32   grad_input -> VENDOR

518 is TD-MPC2's `ZA = LATENT + ACT`, named in that same comment as the
motivating shape. The vendor path allocates and memsets 32 MB per call
(`matmul/vendor/blas.mojo:780`), so this is the identical defect the K floor
was added to fix, one GEMM over.

WHAT THIS MEASURES
------------------
Raising `K_PAD` to a multiple of 128 is not free: it widens K in the forward
and M in the dW. So the question is not "does grad_input get faster" (it
obviously does, it comes off cuBLAS) but whether the WHOLE LAYER does. All
three GEMMs are timed at both paddings and summed.

    pixi run -e nvidia mojo run -I . benchmarks/bench_linear_kpad_modulus.mojo

⚠ Per-call sums are an UPPER BOUND on what a step will show — back-to-back
launches with one sync measure throughput under saturation, and a step that is
already ~95% GPU-busy does not compose 1:1. Quote the ratio, not the ms.
"""

from std.math import ceildiv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.splitk_gemm import multistage_shape_ok


comptime WARMUP = 5
comptime REPS = 50


def _round_up(v: Int, m: Int) -> Int:
    return ((v + m - 1) // m) * m


def kpad_floor(IN_: Int) -> Int:
    """What `Linear` does today: multiple of 32, floor of 128."""
    var r = _round_up(IN_, 32)
    return r if r > 128 else 128


def kpad_modulus(IN_: Int) -> Int:
    """The proposal: multiple of 128 (which subsumes both existing terms)."""
    return _round_up(IN_, 128) if IN_ > 0 else 128


def route(m: Int, n: Int, k: Int) -> String:
    return "multistage" if multistage_shape_ok(m, n, k) else "VENDOR    "


def time_gemm[
    transpose_b: Bool
](M: Int, N: Int, K: Int, ctx: DeviceContext) raises -> Float64:
    """Microseconds per call for `[M,K] @ [K,N]` (or `@ [N,K]ᵀ`)."""
    var ab = ctx.enqueue_create_buffer[DT](M * K)
    var bb = ctx.enqueue_create_buffer[DT](K * N)
    var cb = ctx.enqueue_create_buffer[DT](M * N)
    ab.enqueue_fill(Float32(0.01))
    bb.enqueue_fill(Float32(0.02))
    cb.enqueue_fill(Float32(0.0))
    ctx.synchronize()

    var cv = TileTensor(cb, row_major(M, N))
    var av = TileTensor(ab, row_major(M, K))
    # transpose_b reads B as [N, K]; the buffer is the same K*N elements.
    var bv_n = TileTensor(bb, row_major(K, N))
    var bv_t = TileTensor(bb, row_major(N, K))

    comptime if transpose_b:
        for _ in range(WARMUP):
            max_matmul[transpose_b=True, target="gpu"](cv, av, bv_t, ctx)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(REPS):
            max_matmul[transpose_b=True, target="gpu"](cv, av, bv_t, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        # Keep every operand alive past the last launch: Mojo destroys at LAST
        # USE, and a freed operand under an async launch is a use-after-free.
        _ = ab^
        _ = bb^
        _ = cb^
        return Float64(t1 - t0) / 1e3 / Float64(REPS)
    else:
        for _ in range(WARMUP):
            max_matmul[target="gpu"](cv, av, bv_n, ctx)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(REPS):
            max_matmul[target="gpu"](cv, av, bv_n, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        _ = ab^
        _ = bb^
        _ = cb^
        return Float64(t1 - t0) / 1e3 / Float64(REPS)


def one(label: String, B: Int, IN_: Int, OUT_: Int, ctx: DeviceContext) raises:
    var N_PAD = _round_up(OUT_, 128)
    var kf = kpad_floor(IN_)
    var km = kpad_modulus(IN_)

    print("──", label, " B=", B, " IN=", IN_, " OUT=", OUT_,
          "   K_PAD ", kf, " -> ", km, "   N_PAD=", N_PAD, sep="")
    if kf == km:
        print("   K_PAD already a multiple of 128 — control, expect no change")

    # forward:     [B, K_PAD] @ [K_PAD, N_PAD]          K = K_PAD  (floor)
    # grad_input:  [B, N_PAD] @ [K_PAD, N_PAD]^T        N = K_PAD  (MODULUS)
    # grad_weight: [K_PAD, B] @ [B, N_PAD]              M = K_PAD  (no test)
    var f_a = time_gemm[False](B, N_PAD, kf, ctx)
    var f_b = time_gemm[False](B, N_PAD, km, ctx)
    var g_a = time_gemm[True](B, kf, N_PAD, ctx)
    var g_b = time_gemm[True](B, km, N_PAD, ctx)
    var w_a = time_gemm[False](kf, N_PAD, B, ctx)
    var w_b = time_gemm[False](km, N_PAD, B, ctx)

    print("   forward     ", route(B, N_PAD, kf), f_a, " us  ->  ",
          route(B, N_PAD, km), f_b, " us", sep="")
    print("   grad_input  ", route(B, kf, N_PAD), g_a, " us  ->  ",
          route(B, km, N_PAD), g_b, " us", sep="")
    print("   grad_weight ", route(kf, N_PAD, B), w_a, " us  ->  ",
          route(km, N_PAD, B), w_b, " us", sep="")
    var ta = f_a + g_a + w_a
    var tb = f_b + g_b + w_b
    print("   LAYER TOTAL  ", ta, " us  ->  ", tb, " us   ratio ",
          (ta / tb) if tb > 0.0 else 0.0, "x", sep="")
    print()


def main() raises:
    with DeviceContext() as ctx:
        print("Linear's three GPU GEMMs at K_PAD = floor-128 vs modulus-128.")
        print("multi_gemm_cond: m>1 and n%128==0 and k%32==0 and k>=128")
        print()

        print("=== K_PAD is NOT a multiple of 128 today — grad_input on cuBLAS")
        # TD-MPC2 ZA = LATENT + ACT, the shape linear.mojo's comment names.
        one("tdmpc2 ZA   ", 256, 518, 512, ctx)
        one("tdmpc2 ZA   ", 960, 518, 512, ctx)
        one("concat 160  ", 256, 160, 256, ctx)
        one("concat 192  ", 960, 192, 256, ctx)

        print("=== controls: K_PAD already a multiple of 128, must not move")
        one("sac obs|act ", 256, 6, 256, ctx)     # K_PAD = 128 (the K_MIN case)
        one("act dim     ", 2592, 256, 256, ctx)
        one("act ff      ", 2592, 1024, 256, ctx)

        print("⚠ Per-call sums are a CEILING on a step, not an estimate.")
