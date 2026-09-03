"""The prefill pass runs, and leaves the cache holding what it claims to.

"It ran and the output is finite" is nearly worthless here: the interesting
failures all produce finite output. So beyond running the 16 layers this checks
the things that would otherwise be silent:

  1. **All 16 layers are marked filled.** A loop that wrote only layer 0, or
     skipped the write entirely, still produces a perfectly good prefix output —
     the damage appears ten Euler steps later, in the denoiser.
  2. **The 16 cached slabs are DISTINCT.** Writing every layer to the same
     offset passes (1) and gives a model where every layer cross-attends to
     layer 15.
  3. **The cache holds POST-RoPE, PRE-broadcast K.** Recomputing layer 0's key
     independently — `RoPE(k_proj(input_layernorm(x)))` — must reproduce the
     cached slab exactly. Caching pre-RoPE keys, or caching the 15-head
     broadcast instead of the 5-head projection, are both shape-plausible and
     both wrong.
  4. The output varies (a dead stack emits a constant) and is finite.

Run:
  pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_prefill.mojo
"""

from std.math import abs
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.smolvla.text import (
    SmolVLMTextLayers, SMOLLM_DIM, SMOLLM_FF, SMOLLM_KV_W, SMOLLM_LAYERS,
    SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM,
)
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from mojo_rl.deep_agents.smolvla.fused import SmolVLAPrefill
from mojo_rl.deep_agents.smolvla.attn_mask import att_2d_mask_square, smolvla_ar

comptime P = 32          # prefix tokens (8 image + 23 language + 1 state)
comptime SUFFIX = 8
comptime B = 1
comptime W = SMOLLM_DIM
comptime KVW = SMOLLM_KV_W
comptime L = SMOLLM_LAYERS
comptime XN = P * W
comptime KVN = P * KVW

comptime Tower = SmolVLMTextLayers[L, W, SMOLLM_FF, KVW]
comptime Cache = SmolVLAKVCache[L, P, SUFFIX, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, B]
comptime Pre = SmolVLAPrefill[P, SUFFIX, B]


def main() raises:
    print("=" * 70)
    print("SmolVLA prefill — 16 layers, filling the KV cache")
    print("=" * 70)
    print("  prefix", P, "tokens x", W, " cache", Cache.TOTAL, "elems")

    var d = DeviceContext()
    var ar = smolvla_ar(8, 23, 1, 0)
    assert_equal(len(ar), P, "ar should cover the prefix exactly")
    var mask = att_2d_mask_square(ar)

    var tower = Tower.make["gpu", Deterministic](Optional(d))
    var cache = Cache.make["gpu"](Optional(d))
    var pre = Pre.make["gpu"](mask, Optional(d))

    var x = Tensor.alloc(B * XN)
    for i in range(B * XN):
        x.data[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.05
    x.upload(d)
    var x0 = Tensor.alloc(B * XN)
    for i in range(B * XN):
        x0.data[i] = x.data[i]
    x0.upload(d)

    var out = Tensor.alloc(B * XN)
    pre.run["gpu"](tower, cache, x, out, Optional(d))
    out.download(d)

    # ── 1. every layer filled ────────────────────────────────────────────
    print("  [1] layers filled:", cache.n_filled(), "/", L)
    assert_equal(cache.n_filled(), L, "prefill must write every layer")

    # ── 2. the slabs are distinct ────────────────────────────────────────
    cache.k.download(d)
    var same_pairs = 0
    for a in range(L):
        for b in range(a + 1, L):
            var oa = cache.offset_of(a)
            var ob = cache.offset_of(b)
            var identical = True
            for t in range(0, Cache.LAYER_N, 97):   # strided probe
                if cache.k.data[oa + t] != cache.k.data[ob + t]:
                    identical = False
                    break
            if identical:
                same_pairs += 1
    print("  [2] identical layer pairs:", same_pairs, "of",
          L * (L - 1) // 2, "compared")
    assert_true(same_pairs == 0, "two cached layers are identical — the write"
                                 " offset does not depend on the layer")

    # ── 3. the cache holds post-RoPE, pre-broadcast K for layer 0 ────────
    var h = Tensor.alloc(B * XN)
    var k = Tensor.alloc(B * KVN)
    var kr = Tensor.alloc(B * KVN)
    tower.layers[0].input_layernorm.forward["gpu", B * P](
        TensorRefs[1](x0), h, Optional(d)
    )
    tower.layers[0].k.forward["gpu", B * P](TensorRefs[1](h), k, Optional(d))
    pre.rope_k.forward["gpu", B](TensorRefs[1](k), kr, Optional(d))
    kr.download(d)
    var cmp = 0
    var bad = 0
    for i in range(B * KVN):
        cmp += 1
        if cache.k.data[i] != kr.data[i]:
            bad += 1
    print("  [3] cached K == RoPE(k_proj(ln(x))) for layer 0: compared", cmp,
          " wrong", bad)
    assert_equal(cmp, B * KVN, "must compare layer 0's whole slab")
    assert_true(bad == 0, "the cache does not hold post-RoPE, pre-broadcast K")

    # ── 4. the output is alive ───────────────────────────────────────────
    var nan = 0
    var lo = out.data[0]
    var hi = out.data[0]
    for i in range(B * XN):
        var y = out.data[i]
        if y != y:
            nan += 1
        if y < lo:
            lo = y
        if y > hi:
            hi = y
    print("  [4] output: compared", B * XN, " nan", nan, " min", lo, " max", hi)
    assert_true(nan == 0, "prefill produced NaN")
    assert_true(hi - lo > 1e-6, "output is constant — the stack computed"
                                " nothing")

    print()
    print("PASSED")
