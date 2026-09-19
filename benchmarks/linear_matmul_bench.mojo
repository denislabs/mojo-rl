# +--------------------------------------------------------------------------+ #
# | Linear's three GEMMs at SAC's shapes — MAX's `mm` vs the tiled kernels
# +--------------------------------------------------------------------------+ #
"""What `Linear`/`LinearAct` pay per matmul on THIS device at the shapes a
SAC update sends, and whether `mm_tiled` beats MAX there.

    pixi run -e apple  mojo run -I . benchmarks/linear_matmul_bench.mojo
    pixi run -e nvidia mojo run -I . benchmarks/linear_matmul_bench.mojo

⚠⚠ WHY. The SO-101 tower family trains on the laptop's Metal at 17.9
env-steps/s (19 Sep): about 38 ms per SAC update of three 256-wide MLPs at
batch 256. `docs/CROSS_ATTENTION_OPTIMIZATION.md` §2.3 found MAX's matmul at
88 GFLOP/s on an M1 Pro for attention's shapes and a plain tiled kernel at
740, so the question is whether Linear's shapes sit on the same floor. This
prints the answer per shape; it does NOT change any dispatch.

THE SHAPES. `Linear` pads K and N to 128 on the GPU (`PAD_TO`/`N_PAD_TO`),
so at HIDDEN 256, BATCH 256, OBS 49 (+ACT 6 for the critic) the products are:

    forward   y  = x  @ W      [B, K_PAD] @ [K_PAD, N_PAD]
    backward  dW = xT @ go     [K_PAD, B] @ [B, N_PAD]
    backward  gi = go @ WT     [B, N_PAD] @ [K_PAD, N_PAD]^T   (transpose_b)

for (K_PAD, N_PAD) in {(128, 256) input layer, (256, 256) hidden,
(256, 128) the critic's 1-wide head}. Every one is 8-17 M MACs: a hundred
microseconds of work at the machine's rate, so the launch floor and MAX's
dispatch matter as much as the tile.

VARIANTS. M0 is what ships (`mm`, runtime views on Apple, static on NVIDIA).
M1 is `bmm_tiled` with BH = 1, the kernel attention's products moved to. For
the transposed product M1 is given a PRE-TRANSPOSED operand (a `[N_PAD,
K_PAD]` copy of the weight), which is what a Linear would cache next to
`w_pad`; the transpose itself is not timed, because it is paid once per
optimizer step, not per matmul.

CORRECTNESS. Float64 reference on a sample of outputs, in std units, and a
bit count against M0. Destinations are pre-poisoned so a kernel that writes
nothing fails instead of inheriting M0's answer from reused memory.
"""

from std.math import sqrt
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.mm import mm
from mojo_rl.nn.core.mm_tiled import bmm_tiled, bmm_tiled_uses_big_tile
from mojo_rl.nn.core.tensor import Tensor


comptime WARMUP = 3
comptime REPS = 10
comptime N_ENQ = 20
"""Launches per sync. ⚠ A synced single launch on Metal costs ~250 us
whatever the kernel (the first run of this bench: every shape, both kernels,
250-350 us). A training step enqueues hundreds of kernels between syncs, so
the number that transfers is the per-launch cost INSIDE a stream: `N_ENQ`
back-to-back, one sync, divided."""
comptime GATE_STD_UNITS = 1.0e-4
comptime POISON = 12345.0
comptime N_SAMPLE = 2048

comptime B = 256
"""SAC's batch (`sac_family_driver.BATCH`)."""


struct Lcg(Movable):
    var s: UInt64

    def __init__(out self, seed: Int):
        self.s = UInt64(seed) * 6364136223846793005 + 1442695040888963407

    def unit(mut self) -> Float64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float64((self.s >> 11) & 0x1FFFFFFFFFFFFF) / Float64(1 << 53)

    def below(mut self, n: Int) -> Int:
        return Int(self.unit() * Float64(n)) % n


def _fmt(x: Float64, digits: Int) -> String:
    var scale = 1.0
    for _ in range(digits):
        scale *= 10.0
    var r = Float64(Int(x * scale + (0.5 if x >= 0.0 else -0.5))) / scale
    return String(r)


def _pad(s: String, w: Int) -> String:
    var out = s
    while out.byte_length() < w:
        out += " "
    return out


def run_gemm[
    M: Int, N: Int, K: Int, transpose_b: Bool
](name: String, mut ctx: DeviceContext) raises -> Bool:
    """`o[M, N] = a[M, K] @ b` with `b` stored `[K, N]` (or `[N, K]` when
    `transpose_b`, as M0 reads it; M1 reads a `[K, N]` copy)."""
    comptime AN = M * K
    comptime BN = K * N
    comptime ON = M * N
    comptime FLOP = 2.0 * Float64(M) * Float64(N) * Float64(K)

    var rng = Lcg(M * 131 + N * 17 + K + (7 if transpose_b else 0))
    var a = Tensor.alloc(AN)
    var b_kn = Tensor.alloc(BN)   # [K, N]
    var b_nk = Tensor.alloc(BN)   # [N, K], the same matrix transposed
    for n in range(AN):
        a.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
    for k in range(K):
        for j in range(N):
            var v = Scalar[DT](rng.unit() * 2.0 - 1.0)
            b_kn.data[k * N + j] = v
            b_nk.data[j * K + k] = v
    a.upload(ctx)
    b_kn.upload(ctx)
    b_nk.upload(ctx)

    var s_idx = List[Int](length=N_SAMPLE, fill=0)
    var s_ref = List[Float64](length=N_SAMPLE, fill=0.0)
    for t in range(N_SAMPLE):
        var i = rng.below(M)
        var j = rng.below(N)
        var acc = 0.0
        for k in range(K):
            acc += Float64(a.data[i * K + k]) * Float64(b_kn.data[k * N + j])
        s_idx[t] = i * N + j
        s_ref[t] = acc

    var names: List[String] = [
        String("M0  mm (MAX matmul)")
        + (String(", transpose_b") if transpose_b else String("")),
        String("M1  bmm_tiled BH=1, ")
        + (
            String("64x64 4x4/thread")
            if bmm_tiled_uses_big_tile[1, M, N, K]()
            else String("16x16 1/thread")
        )
        + (String(", WT cached") if transpose_b else String("")),
    ]
    print("")
    print(
        "── " + name + ":  [" + String(M) + " x " + String(K) + "] @ ["
        + String(K) + " x " + String(N) + "]  (" + _fmt(FLOP / 1.0e6, 1)
        + " MFLOP) ──"
    )
    print(
        "   " + _pad(String("variant"), 40) + _pad(String("us/launch"), 10)
        + _pad(String("host us"), 10) + _pad(String("vs M0"), 8) + _pad(String("GFLOP/s"), 10)
        + _pad(String("sampled (std u.)"), 26) + "bits != M0"
    )

    var ok = True
    var best_m0 = 0.0
    var m0_out = List[Scalar[DT]]()
    for variant in range(2):
        var o = Tensor()
        o.ensure_gpu(ctx, ON)
        o.dev.value().enqueue_fill(Scalar[DT](POISON))
        var best = 1.0e30
        var best_enq = 1.0e30
        for rep in range(WARMUP + REPS):
            ctx.synchronize()
            var t0 = perf_counter_ns()
            for _ in range(N_ENQ):
                if variant == 0:
                    comptime if transpose_b:
                        mm[transpose_b=True, A0=M, A1=K, B0=N, B1=K, O0=M, O1=N](
                            o.dev.value(), a.dev.value(), b_nk.dev.value(), ctx
                        )
                    else:
                        mm[A0=M, A1=K, B0=K, B1=N, O0=M, O1=N](
                            o.dev.value(), a.dev.value(), b_kn.dev.value(), ctx
                        )
                else:
                    bmm_tiled[BH=1, M=M, N=N, K=K](
                        o.dev.value(), a.dev.value(), b_kn.dev.value(), ctx
                    )
            # ⚠ HOST RETURN vs COMPLETION. If `enq` ~= the synced time, the
            # call WAITS inside (a Metal `waitUntilCompleted`), and the
            # "kernel time" is really a host stall — see `nn/core/fill.mojo`.
            var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / Float64(N_ENQ)
            ctx.synchronize()
            var dt = Float64(perf_counter_ns() - t0) / 1e3 / Float64(N_ENQ)
            if rep >= WARMUP and dt < best:
                best = dt
            if rep >= WARMUP and t_enq < best_enq:
                best_enq = t_enq
        o.download(ctx)

        var ss = 0.0
        for t in range(N_SAMPLE):
            ss += s_ref[t] * s_ref[t]
        var rms = sqrt(ss / Float64(N_SAMPLE))
        if rms < 1.0e-30:
            rms = 1.0
        var worst = 0.0
        for t in range(N_SAMPLE):
            worst = max(worst, abs(Float64(o.data[s_idx[t]]) - s_ref[t]))
        var sample_err = worst / rms

        var bits = 0
        if variant == 0:
            best_m0 = best
            for n in range(ON):
                m0_out.append(o.data[n])
        else:
            for n in range(ON):
                if o.data[n] != m0_out[n]:
                    bits += 1
        var flag = String("")
        if sample_err > GATE_STD_UNITS:
            flag = "   ⚠⚠ DISAGREES"
            ok = False
        print(
            "   " + _pad(names[variant], 40) + _pad(_fmt(best, 1), 10)
            + _pad(_fmt(best_enq, 1), 10)
            + _pad(_fmt(best_m0 / best, 2) + "x", 8)
            + _pad(_fmt(FLOP / (best / 1.0e6) / 1.0e9, 1), 10)
            + _pad(String(sample_err), 26)
            + String(bits) + "/" + String(ON) + flag
        )
    return ok


def run_layer[K_PAD: Int, N_PAD: Int](
    name: String, mut ctx: DeviceContext
) raises -> Bool:
    var ok = True
    if not run_gemm[B, N_PAD, K_PAD, False](name + " fwd  y = x @ W", ctx):
        ok = False
    if not run_gemm[K_PAD, N_PAD, B, False](name + " bwd  dW = xT @ go", ctx):
        ok = False
    if not run_gemm[B, K_PAD, N_PAD, True](name + " bwd  gi = go @ WT", ctx):
        ok = False
    return ok


def main() raises:
    comptime assert has_accelerator(), "this benchmark times GPU kernels"
    var ctx = DeviceContext()
    print("=" * 104)
    print("Linear's GEMMs at SAC's shapes (batch " + String(B) + ") — " + String(ctx.name()))
    print("=" * 104)
    var ok = True
    # OBS 49 / OBS+ACT 55 -> K_PAD 128; HIDDEN 256 -> N_PAD 256.
    if not run_layer[128, 256](String("input  49->256"), ctx):
        ok = False
    if not run_layer[256, 256](String("hidden 256->256"), ctx):
        ok = False
    # The critic head: OUT 1 -> N_PAD 128.
    if not run_layer[256, 128](String("head   256->1  "), ctx):
        ok = False
    print("")
    if not ok:
        raise Error("a variant disagrees with the float64 reference")
    print("  every variant within 1e-4 std units of float64 on the sample")
