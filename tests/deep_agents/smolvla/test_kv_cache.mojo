"""The static prefix KV cache: per-layer isolation, immutability, and the guard.

Four properties, each chosen because its failure is silent:

  1. **Per-layer isolation.** Writing every layer to offset 0 passes any
     single-layer test and produces a model where all sixteen layers attend to
     layer 15's keys — finite, plausible, wrong. So each layer is written a
     DISTINCT pattern and all sixteen are read back.
  2. **Scratch is exactly `[prefix; suffix]`.** Checked element by element, both
     halves, with the compared count printed.
  3. **The prefix survives ten steps.** This is the property the design exists
     for: the reference appends the suffix into the cache and must `crop` it
     back, and forgetting that grows the cache while changing what each step
     sees. Here the prefix slab is immutable, so building scratch ten times with
     ten different suffixes must leave the prefix half bit-identical every time.
  4. **Reading an unfilled layer raises.** Attention over a zero cache is finite
     with plausible magnitudes and no NaN, so this must fail loudly. The test
     asserts the raise actually happened — a `try` whose `except` never runs
     proves nothing.

Run:
  pixi run mojo run -I . tests/deep_agents/smolvla/test_kv_cache.mojo
  pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_kv_cache.mojo
"""

from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache

comptime LAYERS = 16
comptime PREFIX = 24
comptime SUFFIX = 8
comptime N_KV = 5
comptime HD = 64
comptime B = 1
comptime C = SmolVLAKVCache[LAYERS, PREFIX, SUFFIX, N_KV, HD, B]
comptime LN = C.LAYER_N
comptime SN = C.SUFFIX_N

# ── the B > 1 fixture (leg [6]) ──────────────────────────────────────────
# ⚠ Tiny and separate ON PURPOSE. Every other leg here runs at B == 1, which
# is exactly the value at which the layout bug leg [6] exists to catch is
# invisible. A gate that only scales the main fixture would have inherited the
# blind spot.
comptime B2 = 2
comptime P2 = 2
comptime S2 = 1
comptime NKV2 = 1
comptime HD2 = 2
comptime C2 = SmolVLAKVCache[1, P2, S2, NKV2, HD2, B2]
comptime KVW2 = NKV2 * HD2


def _pattern(layer: Int, i: Int) -> Scalar[DT]:
    """Distinct per (layer, element) — so a layer-offset bug cannot alias."""
    return Scalar[DT](layer * 100000 + i) * 0.001


def main() raises:
    print("=" * 70)
    print("SmolVLA static KV cache")
    print("=" * 70)
    print("  ", LAYERS, "layers x", PREFIX, "prefix x", N_KV * HD,
          "kv width  (slab", C.TOTAL, "elems)")

    var c = C.make["cpu"]()

    # ── 4. the guard fires BEFORE anything is written ────────────────────
    var raised = False
    var empty_k = Tensor.alloc(SN)
    var empty_v = Tensor.alloc(SN)
    try:
        c.build_scratch["cpu"](0, empty_k, empty_v, None)
    except:
        raised = True
    print("  [4] unfilled read raised:", raised)
    assert_true(raised, "reading an unwritten layer did NOT raise — a zero"
                        " cache would have been used silently")

    # ── 1. write a distinct pattern per layer, read all sixteen back ─────
    var ksrc = Tensor.alloc(LN)
    var vsrc = Tensor.alloc(LN)
    for l in range(LAYERS):
        for i in range(LN):
            ksrc.data[i] = _pattern(l, i)
            vsrc.data[i] = _pattern(l, i) + Scalar[DT](0.5)
        c.write_prefix["cpu"](l, ksrc, vsrc, None)
    assert_equal(c.n_filled(), LAYERS, "every layer should be marked filled")

    var cmp = 0
    var bad = 0
    for l in range(LAYERS):
        var off = c.offset_of(l)
        for i in range(LN):
            cmp += 1
            if c.k.data[off + i] != _pattern(l, i):
                bad += 1
            if c.v.data[off + i] != _pattern(l, i) + Scalar[DT](0.5):
                bad += 1
    print("  [1] per-layer isolation: compared", cmp * 2, " wrong", bad)
    assert_equal(cmp, LAYERS * LN, "must compare every layer's whole slab")
    assert_true(bad == 0, "layers alias — a layer-offset bug")

    # ── 2. scratch is exactly [prefix; suffix] ───────────────────────────
    var ksuf = Tensor.alloc(SN)
    var vsuf = Tensor.alloc(SN)
    for i in range(SN):
        ksuf.data[i] = Scalar[DT](-1000 - i)
        vsuf.data[i] = Scalar[DT](-2000 - i)
    c.build_scratch["cpu"](7, ksuf, vsuf, None)
    var scmp = 0
    var sbad = 0
    for i in range(LN):
        scmp += 1
        if c.sk.data[i] != _pattern(7, i):
            sbad += 1
    for i in range(SN):
        scmp += 1
        if c.sk.data[LN + i] != Scalar[DT](-1000 - i):
            sbad += 1
        if c.sv.data[LN + i] != Scalar[DT](-2000 - i):
            sbad += 1
    print("  [2] scratch = [prefix; suffix]: compared", scmp, " wrong", sbad)
    assert_equal(scmp, LN + SN, "must compare both halves in full")
    assert_true(sbad == 0, "scratch is not the concatenation")

    # ── 3. ten steps leave the prefix bit-identical ──────────────────────
    # The reference appends into the cache and crops it back; this design makes
    # that impossible, and here is the demonstration.
    var drift = 0
    for step in range(10):
        for i in range(SN):
            ksuf.data[i] = Scalar[DT](step * 31 + i)
            vsuf.data[i] = Scalar[DT](step * 17 + i)
        c.build_scratch["cpu"](7, ksuf, vsuf, None)
        for i in range(LN):
            if c.sk.data[i] != _pattern(7, i):
                drift += 1
    var off7 = c.offset_of(7)
    for i in range(LN):
        if c.k.data[off7 + i] != _pattern(7, i):
            drift += 1
    print("  [3] prefix after 10 steps: compared", 11 * LN, " drifted", drift)
    assert_true(drift == 0, "the prefix changed across denoising steps — the"
                            " very failure `crop` exists to undo")

    # ── GPU parity ───────────────────────────────────────────────────────
    var d = DeviceContext()
    var g = C.make["gpu"](Optional(d))
    var gk = Tensor.alloc(LN)
    var gv = Tensor.alloc(LN)
    for l in range(LAYERS):
        for i in range(LN):
            gk.data[i] = _pattern(l, i)
            gv.data[i] = _pattern(l, i) + Scalar[DT](0.5)
        gk.upload(d)
        gv.upload(d)
        g.write_prefix["gpu"](l, gk, gv, Optional(d))
    var gks = Tensor.alloc(SN)
    var gvs = Tensor.alloc(SN)
    for i in range(SN):
        gks.data[i] = Scalar[DT](-1000 - i)
        gvs.data[i] = Scalar[DT](-2000 - i)
    gks.upload(d)
    gvs.upload(d)
    g.build_scratch["gpu"](7, gks, gvs, Optional(d))
    g.sk.download(d)
    g.sv.download(d)
    var gbad = 0
    for i in range(LN):
        if g.sk.data[i] != c.sk.data[i]:
            gbad += 1
    for i in range(SN):
        if g.sk.data[LN + i] != Scalar[DT](-1000 - i):
            gbad += 1
        if g.sv.data[LN + i] != Scalar[DT](-2000 - i):
            gbad += 1
    print("  [5] GPU vs CPU scratch: compared", LN + 2 * SN, " wrong", gbad)
    assert_true(gbad == 0, "the GPU sub-buffer copies disagree with the CPU"
                           " path")

    # ── [6] the scratch is BATCH-MAJOR, which only B > 1 can see ─────────
    # `build_scratch` splices a prefix and a suffix into one slab that
    # `RepeatKVHeads` and then `BlockCrossAttention` read as [B, FULL*KVW]:
    # batch b's prefix followed by batch b's OWN suffix. Splicing the whole
    # prefix slab and then the whole suffix slab gives the identical bytes at
    # B == 1 and a different tensor at B == 2, where batch 1's prefix lands
    # inside batch 0's key window. Nothing crashes — attention over the wrong
    # keys is finite, plausible, and wrong.
    #
    # ⚠ This is on V2's critical path in a way it was not on V1's: inference
    # runs one observation, training runs batches.
    var c2 = C2.make["cpu"]()
    var kp2 = Tensor.alloc(B2 * P2 * KVW2)
    var vp2 = Tensor.alloc(B2 * P2 * KVW2)
    for b in range(B2):
        for t in range(P2):
            for e in range(KVW2):
                # 100b + 10t + e: batch, token and channel all separable, so a
                # transposition cannot alias into a right-looking value.
                kp2.data[b * P2 * KVW2 + t * KVW2 + e] = Scalar[DT](
                    100 * b + 10 * t + e
                )
                vp2.data[b * P2 * KVW2 + t * KVW2 + e] = Scalar[DT](
                    -(100 * b + 10 * t + e)
                )
    c2.write_prefix["cpu"](0, kp2, vp2)
    var ks2 = Tensor.alloc(B2 * S2 * KVW2)
    var vs2 = Tensor.alloc(B2 * S2 * KVW2)
    for b in range(B2):
        for e in range(KVW2):
            ks2.data[b * S2 * KVW2 + e] = Scalar[DT](900 + 100 * b + e)
            vs2.data[b * S2 * KVW2 + e] = Scalar[DT](-(900 + 100 * b + e))
    c2.build_scratch["cpu"](0, ks2, vs2)

    var b2cmp = 0
    var b2bad = 0
    for b in range(B2):
        for t in range(P2 + S2):
            for e in range(KVW2):
                var want: Scalar[DT]
                if t < P2:
                    want = Scalar[DT](100 * b + 10 * t + e)
                else:
                    want = Scalar[DT](900 + 100 * b + e)
                var got = c2.sk.data[b * (P2 + S2) * KVW2 + t * KVW2 + e]
                b2cmp += 1
                if got != want:
                    b2bad += 1
                if c2.sv.data[b * (P2 + S2) * KVW2 + t * KVW2 + e] != -want:
                    b2bad += 1
                b2cmp += 1
    print("  [6] B=2 scratch is batch-major: compared", b2cmp, " wrong",
          b2bad)
    assert_equal(
        b2cmp, 2 * B2 * (P2 + S2) * KVW2, "every scratch slot must be probed"
    )
    assert_true(
        b2bad == 0,
        "build_scratch is not batch-major: at B > 1 one batch's prefix sits"
        " in another's key window",
    )

    print()
    print("PASSED")
