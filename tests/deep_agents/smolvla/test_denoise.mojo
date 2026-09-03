"""One denoising step, and ten of them, over a prefilled cache.

The step itself running proves little — what matters is that the two layer
kinds take different paths and that the cache survives:

  1. **Prefill then denoise runs**, output finite and varying.
  2. **The cache is BIT-IDENTICAL after ten steps.** This is the property the
     immutable-prefix design exists for: the reference appends the suffix K/V
     into the cache and must `crop(prefix_len)` it back, and forgetting that
     grows the cache while silently changing what each of the ten Euler steps
     attends to. Here the self layers extend a scratch copy, so there is nothing
     to undo — and this is the assertion that says so.
  3. **Ten steps on the same input give the same answer.** A step that mutated
     any shared state would drift, which is exactly how the crop bug manifests:
     not a crash, a slightly different action chunk each time.

Run:
  pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_denoise.mojo
"""

from std.math import abs
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.smolvla.text import (
    SmolVLMTextLayers, SMOLLM_DIM, SMOLLM_FF, SMOLLM_KV_W, SMOLLM_LAYERS,
    SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, SMOLLM_HEADS,
)
from mojo_rl.deep_agents.smolvla.expert import (
    SmolVLAExpert, EXPERT_W, EXPERT_FF,
)
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from mojo_rl.deep_agents.smolvla.fused import SmolVLAPrefill, SmolVLADenoise
from mojo_rl.deep_agents.smolvla.attn_mask import (
    att_2d_mask, att_2d_mask_square, smolvla_ar,
)

comptime P = 24
comptime S = 6
comptime B = 1
comptime L = SMOLLM_LAYERS
comptime W = SMOLLM_DIM
comptime KVW = SMOLLM_KV_W
comptime XN_P = P * W
comptime XN_S = S * EXPERT_W

comptime Tower = SmolVLMTextLayers[L, W, SMOLLM_FF, KVW]
comptime Expert = SmolVLAExpert[L, EXPERT_W, EXPERT_FF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, S, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, B]
comptime Pre = SmolVLAPrefill[P, S, B]
comptime Den = SmolVLADenoise[P, S, B]


def main() raises:
    print("=" * 70)
    print("SmolVLA denoise — one step, then ten, over a prefilled cache")
    print("=" * 70)
    print("  prefix", P, " suffix", S, " layers", L,
          " (8 self + 8 cross)")

    var d = DeviceContext()
    # prefix ar: 8 image + 15 language + 1 state; suffix: one block per action
    var ar_pre = smolvla_ar(8, 15, 1, 0)
    assert_equal(len(ar_pre), P, "prefix ar")
    var ar_full = smolvla_ar(8, 15, 1, S)
    assert_equal(len(ar_full), P + S, "full ar")

    var mask_pre = att_2d_mask_square(ar_pre)
    var mask_self = att_2d_mask(ar_full, P, P + S, 0, P + S)   # [S, P+S]
    var mask_cross = att_2d_mask(ar_full, P, P + S, 0, P)      # [S, P]

    var tower = Tower.make["gpu", Deterministic](Optional(d))
    var expert = Expert.make["gpu", Deterministic](Optional(d))
    var cache = Cache.make["gpu"](Optional(d))
    var pre = Pre.make["gpu"](mask_pre, Optional(d))
    var den = Den.make["gpu"](mask_self, mask_cross, Optional(d))

    # ── prefill ──────────────────────────────────────────────────────────
    var xp = Tensor.alloc(B * XN_P)
    for i in range(B * XN_P):
        xp.data[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.05
    xp.upload(d)
    var pre_out = Tensor.alloc(B * XN_P)
    pre.run["gpu"](tower, cache, xp, pre_out, Optional(d))
    assert_equal(cache.n_filled(), L, "prefill must fill every layer")

    # snapshot the cache
    cache.k.download(d)
    cache.v.download(d)
    var k0 = List[Scalar[DT]](unsafe_uninit_length=Cache.TOTAL)
    var v0 = List[Scalar[DT]](unsafe_uninit_length=Cache.TOTAL)
    for i in range(Cache.TOTAL):
        k0[i] = cache.k.data[i]
        v0[i] = cache.v.data[i]

    # ── one step ─────────────────────────────────────────────────────────
    var xs = Tensor.alloc(B * XN_S)
    for i in range(B * XN_S):
        xs.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.03
    xs.upload(d)
    var out1 = Tensor.alloc(B * XN_S)
    den.step["gpu"](expert, cache, xs, out1, Optional(d))
    out1.download(d)

    var nan = 0
    var lo = out1.data[0]
    var hi = out1.data[0]
    for i in range(B * XN_S):
        var y = out1.data[i]
        if y != y:
            nan += 1
        if y < lo:
            lo = y
        if y > hi:
            hi = y
    print("  [1] step: compared", B * XN_S, " nan", nan, " min", lo,
          " max", hi)
    assert_true(nan == 0, "denoise produced NaN")
    assert_true(hi - lo > 1e-6, "output is constant — the expert computed"
                                " nothing")

    # ── ten steps: cache untouched, answer reproducible ──────────────────
    var outN = Tensor.alloc(B * XN_S)
    for _ in range(10):
        xs.upload(d)
        den.step["gpu"](expert, cache, xs, outN, Optional(d))
    outN.download(d)
    cache.k.download(d)
    cache.v.download(d)

    var drift = 0
    for i in range(Cache.TOTAL):
        if cache.k.data[i] != k0[i]:
            drift += 1
        if cache.v.data[i] != v0[i]:
            drift += 1
    print("  [2] cache after 10 steps: compared", 2 * Cache.TOTAL,
          " drifted", drift)
    assert_true(drift == 0, "the cache changed across denoising steps — the"
                            " failure `crop` exists to undo")

    var diff = 0
    var worst = Scalar[DT](0)
    for i in range(B * XN_S):
        var dd = abs(outN.data[i] - out1.data[i])
        if dd > worst:
            worst = dd
        if dd != Scalar[DT](0):
            diff += 1
    print("  [3] step 11 vs step 1 on the same input: compared", B * XN_S,
          " differing", diff, " worst", worst)
    assert_true(diff == 0, "repeated steps on identical input drift — some"
                           " state is being mutated between steps")

    print()
    print("PASSED")
