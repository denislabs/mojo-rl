# +--------------------------------------------------------------------------+ #
# | SmolVLA's denoising step, one stage at a time — WHERE are the 28 ms?
# +--------------------------------------------------------------------------+ #
"""One denoising step of the expert at the deploy's real shape, split by op.

    pixi run -e jetson smolvla-denoise-bench-jetson          # the board
    pixi run -e apple mojo run -I . benchmarks/smolvla_denoise_stages_bench.mojo

⚠⚠ WHY THIS EXISTS. After the attention pass the SmolVLA query on the Orin
is 656 ms and the denoise phase is its largest slice: 282 ms, ten Euler
steps of 28 ms (docs/CROSS_ATTENTION_OPTIMIZATION.md §4.6). One step is
~5.3 GMAC — 4.4 ms at the board's fp32 peak — so 28 ms is 6x over the
arithmetic floor, and the expert's attention was already taken 6.1x. What
is left is either the expert's LINEARS missing MAX's GEMM dispatch at M = 50
rows (the way the attention matmuls did, §2.3) or the LAUNCH floor of ~370
kernels per step on a Tegra. The fix is different for each — a GEMM path
for the first, a CUDA graph for the second — and this file says which.

It is the forward twin of the fine-tune's `backward by op` table: the same
`_prof_tick` at the same stage boundaries, in `SmolVLADenoise.step`, behind
its own flag (`profile_step`) so the fine-tune's numbers do not move.

⚠ READING THE RESULT.

  * `un-profiled` is the truth: one step, timed whole, median of REPS.
    The stage table DRAINS the device ~210 times per step, so its sum is
    LARGER than the un-profiled step; the difference is the drains, and
    the per-stage numbers are read as SHARES, not as absolute savings.
  * `us/tick` of the trivial stages (norms, rope, residual adds: 36-48 K
    floats each, memory-bound, ~0 arithmetic) IS the launch floor on this
    device. Multiply it by the ~370 launches per step: that is what a CUDA
    graph would remove.
  * `GFLOPS` of q/k/v, o, up+gate and down against the peak column is the
    GEMM question. At M = 50 rows a well-dispatched fp32 GEMM on the Orin
    should sit in the hundreds of GFLOPS; tens means it is the naive path.
  * The peak column is the ORIN's fp32 peak. Elsewhere read GFLOPS only.

⚠ The weights are `Deterministic`, not the checkpoint: timing does not
depend on the values, and 3.2 GB of weights would turn a benchmark into a
download. The cache is filled with a pseudo-random prefix so every
`_require_filled` guard passes; the masks are the policy's own.
"""

from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.smolvla.text import (
    SMOLLM_DIM, SMOLLM_HEADS, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM,
    SMOLLM_KV_W, SMOLLM_LAYERS,
)
from mojo_rl.deep_agents.smolvla.expert import (
    SmolVLAExpert, EXPERT_W, EXPERT_FF,
)
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from mojo_rl.deep_agents.smolvla.fused import SmolVLADenoise
from mojo_rl.deep_agents.smolvla.attn_mask import att_2d_mask, smolvla_ar


# ── the deploy's real shape (N_CAM 2, IMG_TOK 64, N_LANG 6, CHUNK 50) ──────
comptime B = 1
comptime N_IMG = 2 * 64
comptime N_LANG = 6
comptime P = N_IMG + N_LANG + 1      # 135 prefix tokens
comptime S = 50                      # the action chunk
comptime L = SMOLLM_LAYERS           # 16 expert layers, self/cross alternating
comptime W = SMOLLM_DIM              # 960, q/o meet the VLM here
comptime EW = EXPERT_W               # 720
comptime EFF = EXPERT_FF             # 2048
comptime KVW = SMOLLM_KV_W           # 320
comptime HEADS = SMOLLM_HEADS
comptime HD = SMOLLM_HEAD_DIM
comptime N_SELF = (L + 1) // 2
comptime N_CROSS = L - N_SELF

comptime Expert = SmolVLAExpert[L, EW, EFF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, S, SMOLLM_KV_HEADS, HD, B]
comptime Den = SmolVLADenoise[P, S, B]

comptime WARMUP = 3
comptime REPS = 20
comptime STEPS_DEPLOY = 10
"""The deploy's Euler step count (`smolvla_so101_deploy_real.mojo`, STEPS)."""

comptime ORIN_FP32_PEAK_GFLOPS = 2.0 * 1024.0 * 1.173
"""Same figure as `smolvla_block_attention_bench.mojo`: 1024 CUDA cores at
1.173 GHz, 2 FLOPs per core-cycle. Only meaningful on the Orin."""


# ── per-stage arithmetic, MACs per STEP, from the shapes above ────────────
# self layer: q [EW->W], k,v [EW->KVW] over S rows;  cross: q over S, k,v
# [KVW->KVW] over P rows.
comptime MAC_QKV = (
    N_SELF * S * EW * (W + 2 * KVW)
    + N_CROSS * (S * EW * W + 2 * P * KVW * KVW)
)
comptime MAC_O = L * S * W * EW
comptime MAC_UPGATE = L * 2 * S * EW * EFF
comptime MAC_DOWN = L * S * EFF * EW
# attention: q.k and p.v, HEADS x S x KL x HD each; KL = P+S self, P cross
comptime MAC_ATTN = (
    N_SELF * HEADS * S * (P + S) * HD * 2
    + N_CROSS * HEADS * S * P * HD * 2
)
comptime MAC_TOTAL = MAC_QKV + MAC_O + MAC_UPGATE + MAC_DOWN + MAC_ATTN

# ── ticks per step per stage, from the tick placement in `step` ───────────
# glue: input copy + 2 residual adds per layer.  norm: 2 per layer + final.
# qkv: 1 (self) / 2 (cross, q then k/v).  rope: 1.  rep: 1 (self: scratch +
# 2 repeats) / 2 (cross: cache read, then 2 repeats).  attn, o, up+gate,
# glu, down: 1 each.
comptime TICKS_GLUE = 1 + 2 * L
comptime TICKS_NORM = 2 * L + 1
comptime TICKS_QKV = N_SELF + 2 * N_CROSS
comptime TICKS_ROPE = L
comptime TICKS_REP = N_SELF + 2 * N_CROSS
comptime TICKS_ONE = L


def _fmt(x: Float64, digits: Int) -> String:
    var p = 1.0
    for _ in range(digits):
        p *= 10.0
    var v = Float64(Int(x * p + (0.5 if x >= 0.0 else -0.5))) / p
    return String(v)


def _pad(s: String, w: Int) -> String:
    var out = s
    while out.byte_length() < w:
        out += " "
    return out^


def _rpad(s: String, w: Int) -> String:
    var out = String("")
    while out.byte_length() + s.byte_length() < w:
        out += " "
    return out + s


def _median(mut xs: List[Float64]) -> Float64:
    for a in range(len(xs)):
        for c in range(a + 1, len(xs)):
            if xs[c] < xs[a]:
                var t = xs[a]
                xs[a] = xs[c]
                xs[c] = t
    return xs[len(xs) // 2]


def _min(ref xs: List[Float64]) -> Float64:
    var m = xs[0]
    for i in range(1, len(xs)):
        if xs[i] < m:
            m = xs[i]
    return m


def _row(
    name: String, ns: Int, ticks: Int, macs: Int, step_ms: Float64
):
    var ms = Float64(ns) / Float64(REPS) / 1e6
    var line = _pad(name, 14) + _rpad(_fmt(ms, 3), 9)
    line += _rpad(_fmt(100.0 * ms / step_ms, 1) + "%", 8)
    line += _rpad(String(ticks), 7)
    line += _rpad(_fmt(1000.0 * ms / Float64(ticks), 1), 10)
    if macs > 0:
        var gflop = 2.0 * Float64(macs) / 1e9
        var gflops = gflop / (ms / 1e3)
        line += _rpad(_fmt(gflop, 2), 9)
        line += _rpad(_fmt(gflops, 1), 10)
        line += _rpad(_fmt(100.0 * gflops / ORIN_FP32_PEAK_GFLOPS, 1) + "%", 9)
    print("  " + line)


def main() raises:
    comptime assert has_accelerator(), "this benchmark times a GPU"
    var ctx = DeviceContext()
    print("=" * 96)
    print(
        "SmolVLA denoising step by stage — " + String(ctx.name())
        + ", " + String(L) + " layers, P " + String(P) + " S " + String(S)
    )
    print("=" * 96)
    print(
        "  " + _fmt(2.0 * Float64(MAC_TOTAL) / 1e9, 2)
        + " GFLOP per step; " + _fmt(
            2.0 * Float64(MAC_TOTAL) / 1e9 / ORIN_FP32_PEAK_GFLOPS * 1e3, 2
        ) + " ms at the Orin's fp32 peak"
    )

    # ── fixture: the policy's masks, a filled cache, seeded weights ─────
    var ar = smolvla_ar(N_IMG, N_LANG, 1, S)
    var mask_self = att_2d_mask(ar, P, P + S, 0, P + S)   # [S, P+S]
    var mask_cross = att_2d_mask(ar, P, P + S, 0, P)      # [S, P]

    var expert = Expert.make["gpu", Deterministic](Optional(ctx))
    var cache = Cache.make["gpu"](Optional(ctx))
    var den = Den.make["gpu"](mask_self, mask_cross, Optional(ctx))

    var kp = Tensor.alloc(Cache.LAYER_N)
    var vp = Tensor.alloc(Cache.LAYER_N)
    for i in range(Cache.LAYER_N):
        kp.data[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.05
        vp.data[i] = Scalar[DT](((i * 31) % 19) - 9) * 0.05
    kp.upload(ctx)
    vp.upload(ctx)
    for layer in range(L):
        cache.write_prefix["gpu"](layer, kp, vp, Optional(ctx))

    var xs = Tensor.alloc(B * S * EW)
    for i in range(B * S * EW):
        xs.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.03
    xs.upload(ctx)
    var out = Tensor.alloc(B * S * EW)
    out.ensure_gpu(ctx, B * S * EW)

    # ── 1. the step, whole ──────────────────────────────────────────────
    var times = List[Float64]()
    for rep in range(WARMUP + REPS):
        var t0 = perf_counter_ns()
        den.step["gpu"](expert, cache, xs, out, Optional(ctx))
        ctx.synchronize()
        var dt = Float64(perf_counter_ns() - t0) / 1e6
        if rep >= WARMUP:
            times.append(dt)
    var step_ms = _median(times)
    var step_min = _min(times)
    out.download(ctx)
    var finite = True
    for i in range(B * S * EW):
        var v = Float64(out.data[i])
        if v != v or v > 1e30 or v < -1e30:
            finite = False
    print("")
    print(
        "  un-profiled step   " + _fmt(step_ms, 2) + " ms median, "
        + _fmt(step_min, 2) + " ms best of " + String(REPS)
        + ("" if finite else "   ⚠⚠ OUTPUT NOT FINITE")
    )
    print(
        "  -> " + String(STEPS_DEPLOY) + " steps = "
        + _fmt(step_ms * Float64(STEPS_DEPLOY), 1) + " ms of the query; "
        + _fmt(
            2.0 * Float64(MAC_TOTAL) / 1e9 / (step_ms / 1e3), 1
        ) + " GFLOPS achieved, "
        + _fmt(
            100.0 * 2.0 * Float64(MAC_TOTAL) / 1e9 / (step_ms / 1e3)
            / ORIN_FP32_PEAK_GFLOPS, 1
        ) + "% of the Orin's peak"
    )

    # ── 2. the same step, drained at every stage boundary ───────────────
    den.profile_step = True
    den.reset_profile()
    for _ in range(WARMUP):
        den.step["gpu"](expert, cache, xs, out, Optional(ctx))
    ctx.synchronize()
    den.reset_profile()
    var tp0 = perf_counter_ns()
    for _ in range(REPS):
        den.step["gpu"](expert, cache, xs, out, Optional(ctx))
    ctx.synchronize()
    var prof_ms = Float64(perf_counter_ns() - tp0) / Float64(REPS) / 1e6
    var sum_ns = 0
    for i in range(Den.PR_N):
        sum_ns += den.prof_step[i]

    print("")
    print(
        "  "
        + _pad("stage", 14) + _rpad("ms/step", 9) + _rpad("share", 8)
        + _rpad("ticks", 7) + _rpad("us/tick", 10) + _rpad("GFLOP", 9)
        + _rpad("GFLOPS", 10) + _rpad("Orin pk", 9)
    )
    print("  " + "-" * 76)
    _row("q/k/v", den.prof_step[Den.PR_QKV], TICKS_QKV, MAC_QKV, prof_ms)
    _row("up+gate", den.prof_step[Den.PR_MLP_UPGATE], TICKS_ONE, MAC_UPGATE, prof_ms)
    _row("mlp.down", den.prof_step[Den.PR_MLP_DOWN], TICKS_ONE, MAC_DOWN, prof_ms)
    _row("o", den.prof_step[Den.PR_O], TICKS_ONE, MAC_O, prof_ms)
    _row("attention", den.prof_step[Den.PR_ATTN], TICKS_ONE, MAC_ATTN, prof_ms)
    _row("swiglu", den.prof_step[Den.PR_GLU], TICKS_ONE, 0, prof_ms)
    _row("kv-repeat", den.prof_step[Den.PR_REP], TICKS_REP, 0, prof_ms)
    _row("norms", den.prof_step[Den.PR_NORM], TICKS_NORM, 0, prof_ms)
    _row("rope", den.prof_step[Den.PR_ROPE], TICKS_ROPE, 0, prof_ms)
    _row("glue", den.prof_step[Den.PR_GLUE], TICKS_GLUE, 0, prof_ms)
    print("  " + "-" * 76)
    var ticks = (
        TICKS_QKV + 5 * TICKS_ONE + TICKS_REP + TICKS_NORM + TICKS_ROPE
        + TICKS_GLUE
    )
    var sum_ms = Float64(sum_ns) / Float64(REPS) / 1e6
    print(
        "  " + _pad("sum", 14) + _rpad(_fmt(sum_ms, 3), 9)
        + _rpad("", 8) + _rpad(String(ticks), 7)
        + "    profiled step " + _fmt(prof_ms, 2)
        + " ms = un-profiled " + _fmt(step_ms, 2) + " + drains "
        + _fmt(prof_ms - step_ms, 2) + " ms ("
        + _fmt(1000.0 * (prof_ms - step_ms) / Float64(ticks), 1)
        + " us per drain)"
    )

    # ── 3. the two hypotheses, priced ───────────────────────────────────
    var lin_ns = (
        den.prof_step[Den.PR_QKV] + den.prof_step[Den.PR_MLP_UPGATE]
        + den.prof_step[Den.PR_MLP_DOWN] + den.prof_step[Den.PR_O]
    )
    var lin_ms = Float64(lin_ns) / Float64(REPS) / 1e6
    var lin_macs = MAC_QKV + MAC_UPGATE + MAC_DOWN + MAC_O
    var lin_gflops = 2.0 * Float64(lin_macs) / 1e9 / (lin_ms / 1e3)
    var small_ns = (
        den.prof_step[Den.PR_NORM] + den.prof_step[Den.PR_ROPE]
        + den.prof_step[Den.PR_GLUE]
    )
    var small_ticks = TICKS_NORM + TICKS_ROPE + TICKS_GLUE
    var floor_raw = Float64(small_ns) / Float64(REPS) / 1e3 / Float64(
        small_ticks
    )
    # ⚠ Every tick carries ONE drain, so the raw per-tick number is the
    # kernel plus the drain. Net the drain out (the profiled-minus-whole
    # difference spread over the ticks) before calling it a launch floor —
    # on Metal the drain alone is ~250-500 us and would be the whole number.
    var drain_us = 1000.0 * (prof_ms - step_ms) / Float64(ticks)
    var floor_us = floor_raw - drain_us
    if floor_us < 0.0:
        floor_us = 0.0
    print("")
    print(
        "  linears (q/k/v, o, up+gate, down): " + _fmt(lin_ms, 2)
        + " ms = " + _fmt(100.0 * lin_ms / sum_ms, 0) + "% of the sum, "
        + _fmt(lin_gflops, 0) + " GFLOPS  ("
        + _fmt(100.0 * lin_gflops / ORIN_FP32_PEAK_GFLOPS, 1)
        + "% of the Orin's fp32 peak at M = " + String(S) + " rows)"
    )
    print(
        "  small stages (norms, rope, glue): " + _fmt(floor_raw, 1)
        + " us per tick, " + _fmt(floor_us, 1) + " net of the "
        + _fmt(drain_us, 1) + " us drain — the launch floor; x ~370"
        + " launches/step = " + _fmt(370.0 * floor_us / 1e3, 1)
        + " ms a CUDA graph would take back"
    )
    print("")
    print("  READ: linears at a few % of peak  = the GEMM path at M=50 (dispatch, MMT_MIN_WORK, mm vs max_matmul).")
    print("        floor x 370 close to the step = launches; capture the ten steps in one graph.")
    print("        both                          = do the GEMM first; the graph's saving shrinks with each kernel removed.")

