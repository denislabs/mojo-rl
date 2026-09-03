"""End to end: prefill the cache, then ten Euler steps to an action chunk.

The whole V1 inference path in one run — prefix through the VLM, ten denoising
steps through the expert against the cached prefix, each step projecting back to
action space and taking one negative Euler step.

Checks chosen because "it produced numbers" is not evidence:

  1. **The output moved off the noise.** A sampler whose velocity never reaches
     `x_t` returns the input unchanged, which is finite, correctly shaped and
     useless.
  2. **Ten steps ≠ one step.** Ten identical returns would mean the loop is not
     accumulating.
  3. **Rerunning gives the same answer.** Any state mutated between calls — the
     cache above all — shows up here as drift.
  4. Finite, and varying across the chunk.

Run:
  pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_sample_actions.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.deep_agents.smolvla.text import (
    SmolVLMTextLayers, SMOLLM_DIM, SMOLLM_FF, SMOLLM_KV_W, SMOLLM_LAYERS,
    SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM,
)
from mojo_rl.deep_agents.smolvla.expert import (
    SmolVLAExpert, EXPERT_W, EXPERT_FF,
)
from mojo_rl.deep_agents.smolvla.heads import SMOLVLA_EXPERT_W
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from mojo_rl.deep_agents.smolvla.fused import SmolVLAPrefill, SmolVLADenoise
from mojo_rl.deep_agents.smolvla.policy import SmolVLAActionSampler
from mojo_rl.deep_agents.smolvla.attn_mask import (
    att_2d_mask, att_2d_mask_square, smolvla_ar,
)

comptime P = 24
comptime CHUNK = 6
comptime ADIM = 32
comptime B = 1
comptime L = SMOLLM_LAYERS
comptime W = SMOLLM_DIM
comptime KVW = SMOLLM_KV_W
comptime EW = SMOLVLA_EXPERT_W
comptime STEPS = 10

comptime Tower = SmolVLMTextLayers[L, W, SMOLLM_FF, KVW]
comptime Expert = SmolVLAExpert[L, EW, EXPERT_FF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, CHUNK, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, B]
comptime Pre = SmolVLAPrefill[P, CHUNK, B]
comptime Den = SmolVLADenoise[P, CHUNK, B]
comptime Sam = SmolVLAActionSampler[CHUNK, ADIM, EW, STEPS, B]


def main() raises:
    print("=" * 70)
    print("SmolVLA — sample_actions end to end")
    print("=" * 70)
    print("  prefix", P, " chunk", CHUNK, " action dim", ADIM, " steps", STEPS)

    var d = DeviceContext()
    var ar_pre = smolvla_ar(8, 15, 1, 0)
    var ar_full = smolvla_ar(8, 15, 1, CHUNK)
    var mask_pre = att_2d_mask_square(ar_pre)
    var mask_self = att_2d_mask(ar_full, P, P + CHUNK, 0, P + CHUNK)
    var mask_cross = att_2d_mask(ar_full, P, P + CHUNK, 0, P)

    var tower = Tower.make["gpu", Deterministic](Optional(d))
    var expert = Expert.make["gpu", Deterministic](Optional(d))
    var cache = Cache.make["gpu"](Optional(d))
    var pre = Pre.make["gpu"](mask_pre, Optional(d))
    var den = Den.make["gpu"](mask_self, mask_cross, Optional(d))
    var sam = Sam.make["gpu"](Optional(d))

    var action_in = Linear[ADIM, EW].make["gpu", Deterministic](Optional(d))
    var tmlp_in = Linear[2 * EW, EW].make["gpu", Deterministic](Optional(d))
    var tmlp_out = Linear[EW, EW].make["gpu", Deterministic](Optional(d))
    var action_out = Linear[EW, ADIM].make["gpu", Deterministic](Optional(d))

    # ── prefill ──────────────────────────────────────────────────────────
    var xp = Tensor.alloc(B * P * W)
    for i in range(B * P * W):
        xp.data[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.05
    xp.upload(d)
    var pout = Tensor.alloc(B * P * W)
    pre.run["gpu"](tower, cache, xp, pout, Optional(d))
    assert_equal(cache.n_filled(), L, "prefill must fill every layer")
    print("  prefill: cache filled", cache.n_filled(), "/", L)

    # ── sample ───────────────────────────────────────────────────────────
    comptime XN = B * CHUNK * ADIM
    var noise = Tensor.alloc(XN)
    for i in range(XN):
        noise.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.1
    var noise_host = List[Scalar[DT]](unsafe_uninit_length=XN)
    for i in range(XN):
        noise_host[i] = noise.data[i]
    noise.upload(d)

    var chunk = Tensor.alloc(XN)
    sam.sample["gpu", P](
        expert, cache, den, action_in, tmlp_in, tmlp_out, action_out,
        noise, chunk, Optional(d),
    )
    chunk.download(d)

    var nan = 0
    var lo = chunk.data[0]
    var hi = chunk.data[0]
    var moved = 0
    var max_move = Scalar[DT](0)
    for i in range(XN):
        var y = chunk.data[i]
        if y != y:
            nan += 1
        if y < lo:
            lo = y
        if y > hi:
            hi = y
        var mv = abs(y - noise_host[i])
        if mv > max_move:
            max_move = mv
        if mv > Scalar[DT](1e-6):
            moved += 1
    print("  [1] chunk: compared", XN, " nan", nan, " min", lo, " max", hi)
    print("      moved off the noise:", moved, "/", XN, " max move", max_move)
    assert_true(nan == 0, "sampling produced NaN")
    assert_true(hi - lo > 1e-6, "the chunk is constant")
    assert_true(moved == XN, "the sampler returned the noise — the velocity"
                             " never reached x_t")

    # ── 2. ten steps differ from one ─────────────────────────────────────
    comptime Sam1 = SmolVLAActionSampler[CHUNK, ADIM, EW, 1, B]
    var sam1 = Sam1.make["gpu"](Optional(d))
    var noise1 = Tensor.alloc(XN)
    for i in range(XN):
        noise1.data[i] = noise_host[i]
    noise1.upload(d)
    var chunk1 = Tensor.alloc(XN)
    sam1.sample["gpu", P](
        expert, cache, den, action_in, tmlp_in, tmlp_out, action_out,
        noise1, chunk1, Optional(d),
    )
    chunk1.download(d)
    var differ = 0
    for i in range(XN):
        if abs(chunk1.data[i] - chunk.data[i]) > Scalar[DT](1e-6):
            differ += 1
    print("  [2] 10 steps vs 1 step: differing", differ, "/", XN)
    assert_true(differ > 0, "ten steps equal one step — the loop is not"
                            " accumulating")

    # ── 3. rerunning reproduces ──────────────────────────────────────────
    var noise2 = Tensor.alloc(XN)
    for i in range(XN):
        noise2.data[i] = noise_host[i]
    noise2.upload(d)
    var chunk2 = Tensor.alloc(XN)
    sam.sample["gpu", P](
        expert, cache, den, action_in, tmlp_in, tmlp_out, action_out,
        noise2, chunk2, Optional(d),
    )
    chunk2.download(d)
    var drift = 0
    for i in range(XN):
        if chunk2.data[i] != chunk.data[i]:
            drift += 1
    print("  [3] rerun: differing", drift, "/", XN)
    assert_true(drift == 0, "a second call gives a different answer — some"
                            " state is mutated between calls")

    print()
    print("PASSED — prefill + 10 Euler steps ->", XN, "action values")
