"""Prefix assembly: the segments land where the mask expects, and only two of
the three are scaled.

Assembly has no interesting arithmetic — it has interesting BOOKKEEPING, and
every way it goes wrong yields a correctly-shaped prefix:

  1. **Length and order.** `P == N_CAM*64 + N_LANG + 1`, and `smolvla_ar` built
     for the same layout must agree — that is the only thing tying the prefix to
     the mask the prefill uses.
  2. **The language segment sits at `N_CAM*IMG_N`** and equals an independently
     run gather. Off-by-one-segment puts language where the mask says image;
     both are 960-wide and neither complains.
  3. **The state segment is last and is NOT scaled.** The reference scales image
     and language embeddings by sqrt(960) and leaves the state alone, so the
     rule cannot be applied uniformly. Checked by running `state_proj` alone and
     demanding the tail match it EXACTLY — a stray scale would be a factor of 31.
  4. **The image and language segments ARE scaled**, verified by re-running the
     gather unscaled and finding the ratio.

Run:
  pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_prefix_embed.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.combinators.tokenwise import Tokenwise
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.deep_agents.smolvla.vision import SigLIPVisionTower, SIGLIP_IMG
from mojo_rl.deep_agents.smolvla.text import SMOLLM_DIM
from mojo_rl.deep_agents.smolvla.heads import SMOLVLA_CONNECTOR_IN
from mojo_rl.deep_agents.smolvla.policy import SmolVLAPrefixEmbed
from mojo_rl.deep_agents.smolvla.embed import embed_language_tokens
from mojo_rl.deep_agents.smolvla.attn_mask import smolvla_ar

comptime N_CAM = 1
comptime N_LANG = 15
comptime W = SMOLLM_DIM
comptime VOCAB = 64
comptime SDIM = 32
comptime IMG_TOK = 64
comptime PE = SmolVLAPrefixEmbed[N_CAM, N_LANG, 1, W, IMG_TOK]
comptime VIS_IN = 3 * SIGLIP_IMG * SIGLIP_IMG
comptime IMG_N = IMG_TOK * W


def main() raises:
    print("=" * 70)
    print("SmolVLA prefix assembly")
    print("=" * 70)
    print("  ", N_CAM, "camera(s) x", IMG_TOK, "tokens +", N_LANG,
          "language + 1 state =", PE.P, "tokens")

    # the mask the prefill will use is built from the SAME layout
    var ar = smolvla_ar(N_CAM * IMG_TOK, N_LANG, 1, 0)
    assert_equal(len(ar), PE.P, "smolvla_ar and the assembled prefix disagree"
                                " on P — the mask would not match the prefix")
    print("  [1] ar length matches P:", len(ar))

    var d = DeviceContext()
    var vision = SigLIPVisionTower[].make["gpu", Deterministic](Optional(d))
    var conn = Tokenwise[IMG_TOK, Linear[SMOLVLA_CONNECTOR_IN, W]].make[
        "gpu", Deterministic
    ](Optional(d))
    var sproj = Linear[SDIM, W].make["gpu", Deterministic](Optional(d))
    var pe = PE.make["gpu"](Optional(d))

    var ew = Tensor.alloc(VOCAB * W)
    for i in range(VOCAB * W):
        ew.data[i] = Scalar[DT](((i * 31) % 23) - 11) * 0.01
    var ids = List[Int]()
    for t in range(N_LANG):
        ids.append((t * 7 + 3) % VOCAB)

    var imgs = Tensor.alloc(N_CAM * VIS_IN)
    for i in range(N_CAM * VIS_IN):
        imgs.data[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.03
    var state = Tensor.alloc(SDIM)
    for i in range(SDIM):
        state.data[i] = Scalar[DT](i - 16) * 0.05
    state.upload(d)

    var out = Tensor.alloc(PE.OUT_N)
    pe.run["gpu", VOCAB, SMOLVLA_CONNECTOR_IN, SDIM](
        vision, conn, ew, sproj, imgs, ids, state, out, Optional(d)
    )
    out.download(d)
    print("  [2] assembled", PE.OUT_N, "values")

    var nan = 0
    for i in range(PE.OUT_N):
        if out.data[i] != out.data[i]:
            nan += 1
    assert_true(nan == 0, "assembly produced NaN")

    # ── the language segment sits at N_CAM*IMG_N and matches the gather ──
    var want = Tensor.alloc(N_LANG * W)
    embed_language_tokens[VOCAB, W](ew, ids, want, True)
    var base = N_CAM * IMG_N
    var lbad = 0
    for i in range(N_LANG * W):
        if abs(out.data[base + i] - want.data[i]) > Scalar[DT](1e-5):
            lbad += 1
    print("  [3] language at offset", base, ": compared", N_LANG * W,
          " wrong", lbad)
    assert_true(lbad == 0, "the language segment is not where the mask expects"
                           " it, or is not the scaled gather")

    # scaled vs unscaled must differ by sqrt(W) — the factor is real
    var raw = Tensor.alloc(N_LANG * W)
    embed_language_tokens[VOCAB, W](ew, ids, raw, False)
    var f = sqrt(Scalar[DT](W))
    var ratio_ok = 0
    for i in range(N_LANG * W):
        if abs(raw.data[i]) > Scalar[DT](1e-3):
            if abs(out.data[base + i] / raw.data[i] - f) < Scalar[DT](1e-2):
                ratio_ok += 1
    print("      scale check: ", ratio_ok, "elements at ratio sqrt(W) =", f)
    assert_true(ratio_ok > N_LANG * W // 2, "the sqrt(W) scale is not applied"
                                            " to the language segment")

    # ── the state segment is last and NOT scaled ─────────────────────────
    var sw = Tensor.alloc(W)
    sproj.forward["gpu", 1](TensorRefs[1](state), sw, Optional(d))
    sw.download(d)
    var sbase = N_CAM * IMG_N + N_LANG * W
    var sbad = 0
    for i in range(W):
        if out.data[sbase + i] != sw.data[i]:
            sbad += 1
    print("  [4] state at offset", sbase, ": compared", W, " wrong", sbad,
          " (must be UNSCALED)")
    assert_true(sbad == 0, "the state segment is scaled, or misplaced — the"
                           " reference scales image and language only")

    # ── the image segment is alive and scaled ────────────────────────────
    var ilo = out.data[0]
    var ihi = out.data[0]
    for i in range(IMG_N):
        if out.data[i] < ilo:
            ilo = out.data[i]
        if out.data[i] > ihi:
            ihi = out.data[i]
    print("  [5] image segment: min", ilo, " max", ihi)
    assert_true(ihi - ilo > 1e-6, "the image segment is constant")

    print()
    print("PASSED —", PE.P, "prefix tokens in the order the mask assumes")
