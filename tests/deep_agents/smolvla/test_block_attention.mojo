"""BlockCrossAttention: vs the tested CrossAttention, and vs a masked reference.

Two checks, and the first is the important one:

  1. **With an ALL-ALLOW mask it must equal `CrossAttention`.** That leaf is
     already gated against torch (`test_cross_attention_vs_torch.mojo`), so
     agreeing with it pins the unmasked math — scaling, head split, softmax,
     the context sum — to something independently validated. Writing a fourth
     attention leaf is only safe because this check exists.
  2. **With a real block mask, against a reference computed in the test** from
     the definition, so the masking itself is not checked against its own
     implementation.

Plus GPU/CPU parity, and a fully-masked row, which must give a zero context
vector rather than a NaN from dividing by an empty denominator.

Run:
  pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_block_attention.mojo
"""

from std.math import abs, exp, sqrt
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.cross_attention import CrossAttention
from mojo_rl.deep_agents.smolvla.block_attention import (
    BlockCrossAttention, BA_MASK_NEG,
)

comptime DIM = 64
comptime HEADS = 4
comptime QL = 5
comptime KL = 9
comptime HD = DIM // HEADS
comptime B = 2
comptime QN = QL * DIM
comptime KN = KL * DIM
comptime BA = BlockCrossAttention[DIM, HEADS, QL, KL]
comptime XA = CrossAttention[DIM, HEADS, QL, KL, False]


def _ref(
    ref q: List[Scalar[DT]], ref k: List[Scalar[DT]], ref v: List[Scalar[DT]],
    ref mask: List[Scalar[DT]],
) -> List[Scalar[DT]]:
    """Masked attention straight from the definition."""
    var out = List[Scalar[DT]](unsafe_uninit_length=B * QN)
    var scale = Scalar[DT](1.0) / sqrt(Scalar[DT](HD))
    for b in range(B):
        for h in range(HEADS):
            for i in range(QL):
                var qb = b * QN + i * DIM + h * HD
                var sc = List[Scalar[DT]](unsafe_uninit_length=KL)
                var mx = BA_MASK_NEG
                for j in range(KL):
                    var s = Scalar[DT](0)
                    var kb = b * KN + j * DIM + h * HD
                    for d in range(HD):
                        s += q[qb + d] * k[kb + d]
                    sc[j] = s * scale + mask[i * KL + j]
                    if mask[i * KL + j] > BA_MASK_NEG and sc[j] > mx:
                        mx = sc[j]
                var den = Scalar[DT](0)
                for d in range(HD):
                    out[qb + d] = Scalar[DT](0)
                for j in range(KL):
                    if mask[i * KL + j] <= BA_MASK_NEG:
                        continue
                    var w = exp(sc[j] - mx)
                    den += w
                    var kb = b * KN + j * DIM + h * HD
                    for d in range(HD):
                        out[qb + d] += w * v[kb + d]
                var inv = Scalar[DT](0)
                if den > Scalar[DT](1e-30):
                    inv = Scalar[DT](1.0) / den
                for d in range(HD):
                    out[qb + d] *= inv
    return out^


def main() raises:
    print("=" * 70)
    print("BlockCrossAttention")
    print("=" * 70)

    var qh = List[Scalar[DT]]()
    var kh = List[Scalar[DT]]()
    var vh = List[Scalar[DT]]()
    for i in range(B * QN):
        qh.append(Scalar[DT](((i * 37) % 19) - 9) * 0.1)
    for i in range(B * KN):
        kh.append(Scalar[DT](((i * 53) % 23) - 11) * 0.07)
        vh.append(Scalar[DT](((i * 29) % 17) - 8) * 0.05)

    var pack = TensorPack[3]()
    pack[0].ensure(B * QN)
    pack[1].ensure(B * KN)
    pack[2].ensure(B * KN)
    for i in range(B * QN):
        pack[0].data[i] = qh[i]
    for i in range(B * KN):
        pack[1].data[i] = kh[i]
        pack[2].data[i] = vh[i]

    # ── 1. all-allow == CrossAttention ───────────────────────────────────
    var allow = List[Scalar[DT]]()
    for _ in range(QL * KL):
        allow.append(Scalar[DT](0.0))
    var ba = BA.make["cpu"](allow)
    var mine = Tensor.alloc(B * QN)
    ba.forward["cpu", B](pack[0], pack[1], pack[2], mine, None)

    var xa = XA.make["cpu", Deterministic]()
    var theirs = Tensor.alloc(B * QN)
    xa.forward["cpu", B](TensorRefs[3](pack[0], pack[1], pack[2]), theirs, None)

    var cmp = 0
    var worst = Scalar[DT](0)
    for i in range(B * QN):
        cmp += 1
        var d = abs(mine.data[i] - theirs.data[i])
        if d > worst:
            worst = d
    print("  [1] all-allow vs CrossAttention: compared", cmp, " worst",
          worst)
    assert_true(cmp == B * QN, "must compare every output element")
    assert_true(worst < Scalar[DT](1e-5), "disagrees with the torch-gated"
                                          " CrossAttention on the unmasked case")

    # ── 2. a real block mask, vs the definition ──────────────────────────
    # rows 0..1 see all keys; rows 2..4 see keys 0..3 plus causally among 4..8;
    # and one row is fully masked, to exercise the floored denominator.
    var mask = List[Scalar[DT]]()
    for i in range(QL):
        for j in range(KL):
            var allow_ij: Bool
            if i < 2:
                allow_ij = True
            elif i == 4:
                allow_ij = False              # fully masked row
            else:
                allow_ij = (j < 4) or (j <= 4 + (i - 2))
            mask.append(Scalar[DT](0.0) if allow_ij else BA_MASK_NEG)
    var ba2 = BA.make["cpu"](mask)
    var m2 = Tensor.alloc(B * QN)
    ba2.forward["cpu", B](pack[0], pack[1], pack[2], m2, None)
    var want = _ref(qh, kh, vh, mask)
    var bad = 0
    var w2 = Scalar[DT](0)
    for i in range(B * QN):
        var d = abs(m2.data[i] - want[i])
        if d > w2:
            w2 = d
        if d > Scalar[DT](1e-5):
            bad += 1
    print("  [2] masked vs the definition: compared", B * QN, " wrong", bad,
          " worst", w2)
    assert_true(bad == 0, "masked attention disagrees with the definition")

    # the fully-masked row must be a zero context vector, not NaN
    var nan = 0
    var nonzero = 0
    for b in range(B):
        for h in range(HEADS):
            var qb = b * QN + 4 * DIM + h * HD
            for d in range(HD):
                var y = m2.data[qb + d]
                if y != y:
                    nan += 1
                elif y != Scalar[DT](0):
                    nonzero += 1
    print("  [3] fully-masked row: nan", nan, " nonzero", nonzero,
          " (of", B * HEADS * HD, ")")
    assert_true(nan == 0, "a fully-masked row produced NaN")
    assert_true(nonzero == 0, "a fully-masked row should give a zero context")

    # ── 4. GPU parity ────────────────────────────────────────────────────
    var c = DeviceContext()
    var gpack = TensorPack[3]()
    gpack[0].ensure(B * QN)
    gpack[1].ensure(B * KN)
    gpack[2].ensure(B * KN)
    for i in range(B * QN):
        gpack[0].data[i] = qh[i]
    for i in range(B * KN):
        gpack[1].data[i] = kh[i]
        gpack[2].data[i] = vh[i]
    gpack[0].upload(c)
    gpack[1].upload(c)
    gpack[2].upload(c)
    var gba = BA.make["gpu"](mask, Optional(c))
    var g = Tensor.alloc(B * QN)
    gba.forward["gpu", B](gpack[0], gpack[1], gpack[2], g, Optional(c))
    g.download(c)
    var gbad = 0
    var gw = Scalar[DT](0)
    for i in range(B * QN):
        var d = abs(g.data[i] - m2.data[i])
        if d > gw:
            gw = d
        if d > Scalar[DT](1e-5):
            gbad += 1
    print("  [4] GPU vs CPU: compared", B * QN, " wrong", gbad, " worst", gw)
    assert_true(gbad == 0, "the GPU kernel disagrees with the CPU path")

    print()
    print("PASSED")
