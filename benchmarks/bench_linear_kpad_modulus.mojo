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

⚠ EVERY DIMENSION HERE IS COMPTIME, AND THAT IS LOAD-BEARING.
The first version of this file built its `TileTensor`s with a RUNTIME
`row_major(M, N)`. MAX then sees dynamic shapes — its own dispatch log says
`Static shapes available: N= True  K= True` when they are static — takes an
unspecialized path, and the loop becomes host-bound. Every GEMM measured
~350 us across an 81x span in work (0.05 implied TFLOPS on a 5090), and the
null control (K_PAD 128 vs 128, the SAME configuration twice) reported a 3.3%
difference. `Linear` passes static shapes, so a harness that does not is
measuring a different dispatch than the one under test.

`_assert_work_bound` below fails the run if that happens again: across an 80x
work span the times must span at least 3x, or the harness is timing dispatch
rather than arithmetic and every ratio it prints is noise.

⚠ Per-call sums are an UPPER BOUND on what a step will show — back-to-back
launches with one sync measure throughput under saturation, and a step that is
already ~95% GPU-busy does not compose 1:1. Quote the ratio, not the ms.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.splitk_gemm import multistage_shape_ok


comptime WARMUP = 20
comptime REPS = 200


def _round_up(v: Int, m: Int) -> Int:
    return ((v + m - 1) // m) * m


def _kpad_floor(IN_: Int) -> Int:
    """What `Linear` does today: multiple of 32, floor of 128."""
    var r = _round_up(IN_, 32)
    return r if r > 128 else 128


def _kpad_modulus(IN_: Int) -> Int:
    """The proposal: multiple of 128 (which subsumes both existing terms)."""
    return _round_up(IN_, 128)


def route(m: Int, n: Int, k: Int) -> String:
    return "multistage" if multistage_shape_ok(m, n, k) else "VENDOR"


def time_gemm[
    M: Int, N: Int, K: Int, transpose_b: Bool
](ctx: DeviceContext) raises -> Float64:
    """Microseconds per call for `[M,K] @ [K,N]` (or `@ [N,K]ᵀ`).

    All three dims are COMPTIME so `max_matmul` sees static shapes and takes
    the same dispatch `Linear` gets. See the header."""
    var ab = ctx.enqueue_create_buffer[DT](M * K)
    var bb = ctx.enqueue_create_buffer[DT](K * N)
    var cb = ctx.enqueue_create_buffer[DT](M * N)
    ab.enqueue_fill(Float32(0.01))
    bb.enqueue_fill(Float32(0.02))
    cb.enqueue_fill(Float32(0.0))
    ctx.synchronize()

    var cv = TileTensor(cb, row_major[M, N]())
    var av = TileTensor(ab, row_major[M, K]())

    var t0: Int
    var t1: Int
    comptime if transpose_b:
        var bv = TileTensor(bb, row_major[N, K]())
        for _ in range(WARMUP):
            max_matmul[transpose_b=True, target="gpu"](cv, av, bv, ctx)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(REPS):
            max_matmul[transpose_b=True, target="gpu"](cv, av, bv, ctx)
        ctx.synchronize()
        t1 = perf_counter_ns()
    else:
        var bv = TileTensor(bb, row_major[K, N]())
        for _ in range(WARMUP):
            max_matmul[target="gpu"](cv, av, bv, ctx)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(REPS):
            max_matmul[target="gpu"](cv, av, bv, ctx)
        ctx.synchronize()
        t1 = perf_counter_ns()

    # Mojo destroys at LAST USE; a freed operand under an async launch is a
    # use-after-free, and the sync above is the last mention of all three.
    _ = ab^
    _ = bb^
    _ = cb^
    return Float64(t1 - t0) / 1e3 / Float64(REPS)


def one[
    B: Int, IN_: Int, OUT_: Int
](label: String, ctx: DeviceContext, mut probe: List[Float64]) raises:
    comptime N_PAD = _round_up(OUT_, 128)
    comptime KF = _kpad_floor(IN_)
    comptime KM = _kpad_modulus(IN_)

    print(
        "── ", label, "  B=", B, " IN=", IN_, " OUT=", OUT_,
        "   K_PAD ", KF, " -> ", KM, "   N_PAD=", N_PAD, sep="",
    )
    comptime if KF == KM:
        print("   (control: K_PAD unchanged — both arms are the SAME shape,")
        print("    so any ratio away from 1.000 is this harness's noise floor)")

    # forward      [B, K_PAD] @ [K_PAD, N_PAD]      K = K_PAD  (floor test)
    # grad_input   [B, N_PAD] @ [K_PAD, N_PAD]^T    N = K_PAD  (MODULUS test)
    # grad_weight  [K_PAD, B] @ [B, N_PAD]          M = K_PAD  (no test)
    var f_a = time_gemm[B, N_PAD, KF, False](ctx)
    var f_b = time_gemm[B, N_PAD, KM, False](ctx)
    var g_a = time_gemm[B, KF, N_PAD, True](ctx)
    var g_b = time_gemm[B, KM, N_PAD, True](ctx)
    var w_a = time_gemm[KF, N_PAD, B, False](ctx)
    var w_b = time_gemm[KM, N_PAD, B, False](ctx)

    probe.append(f_a)

    print("   forward      ", route(B, N_PAD, KF), " ", f_a, " us   ->   ",
          route(B, N_PAD, KM), " ", f_b, " us", sep="")
    print("   grad_input   ", route(B, KF, N_PAD), " ", g_a, " us   ->   ",
          route(B, KM, N_PAD), " ", g_b, " us", sep="")
    print("   grad_weight  ", route(KF, N_PAD, B), " ", w_a, " us   ->   ",
          route(KM, N_PAD, B), " ", w_b, " us", sep="")
    var ta = f_a + g_a + w_a
    var tb = f_b + g_b + w_b
    print("   LAYER TOTAL   ", ta, " us   ->   ", tb, " us    speedup ",
          (ta / tb) if tb > 0.0 else 0.0, "x", sep="")
    print()


def _assert_work_bound(probe: List[Float64]) raises:
    """Fail the run if the harness is timing dispatch instead of arithmetic.

    The forward shapes below span ~80x in FLOPs. If their measured times do
    not span at least 3x, the loop is host-bound and every ratio printed above
    is noise — which is exactly what the first version of this file did, at
    350 us for everything. A benchmark that cannot fail is not a measurement.
    """
    var lo = probe[0]
    var hi = probe[0]
    for i in range(len(probe)):
        if probe[i] < lo:
            lo = probe[i]
        if probe[i] > hi:
            hi = probe[i]
    var span = (hi / lo) if lo > 0.0 else 0.0
    print("harness check: forward times span ", span,
          "x across ~80x of work (need >= 3x)", sep="")
    if span < 3.0:
        raise Error(
            "HOST-BOUND HARNESS: the forward GEMMs measured nearly the same"
            " time across an 80x work span, so this run timed dispatch, not"
            " the GEMM. Every ratio above is noise. Check that every dim is"
            " comptime (static shapes) before reading anything."
        )


def main() raises:
    with DeviceContext() as ctx:
        print("Linear's three GPU GEMMs at K_PAD = floor-128 vs modulus-128.")
        print("multi_gemm_cond: m>1 and n%128==0 and k%32==0 and k>=128")
        print()
        var probe = List[Float64]()

        print("=== K_PAD is NOT a multiple of 128 today — grad_input on cuBLAS")
        # TD-MPC2 ZA = LATENT + ACT, the shape linear.mojo's comment names.
        one[256, 518, 512]("tdmpc2 ZA  ", ctx, probe)
        one[960, 518, 512]("tdmpc2 ZA  ", ctx, probe)
        one[256, 160, 256]("concat 160 ", ctx, probe)
        one[960, 192, 256]("concat 192 ", ctx, probe)

        print("=== controls: K_PAD already a multiple of 128, must not move")
        one[256, 6, 256]("sac obs|act", ctx, probe)     # K_PAD = 128 (K_MIN)
        one[2592, 256, 256]("act dim    ", ctx, probe)
        one[2592, 1024, 256]("act ff     ", ctx, probe)

        _assert_work_bound(probe)
        print()
        print("⚠ Per-call sums are a CEILING on a step, not an estimate.")
