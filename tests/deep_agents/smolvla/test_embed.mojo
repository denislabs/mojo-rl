"""Prefix/suffix assembly pieces, against numpy-computed references.

The time embedding's expectations come from a numpy transcription of openpi's
`create_sinusoidal_pos_embedding`, run separately — a different language and a
different `linspace`, so agreeing pins the convention rather than restating it.
Two things it pins:

  * `concat([sin, cos])` — the first D/2 entries are SINES, the last D/2 cosines.
    Not interleaved, and deliberately not RoPE's channel-pair convention two
    files over. Interleaving would be the same shape and a different embedding.
  * `linspace(0, 1, D/2)` includes BOTH endpoints, so `fraction[i] = i/(D/2-1)`.
    Using `i/(D/2)` shifts every period slightly — a few percent, entirely
    plausible-looking, and wrong.

The language gather is checked for the sqrt(DIM) scale that `embed_prefix`
applies to image and language embeddings alike. Dropping it leaves every prefix
token ~31x too small, with no NaN and no crash.

Run:
  pixi run mojo run -I . tests/deep_agents/smolvla/test_embed.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.smolvla.embed import (
    sinusoidal_time_embedding, embed_language_tokens,
)


def _check(
    ref got: List[Scalar[DT]], ref want: List[Float64], label: String
) raises -> Int:
    var bad = 0
    var worst = Float64(0)
    for i in range(len(want)):
        var d = abs(Float64(got[i]) - want[i])
        if d > worst:
            worst = d
        if d > 1e-6:
            bad += 1
    print("  ", label, ": compared", len(want), " wrong", bad, " worst", worst)
    assert_true(bad == 0, label + ": disagrees with the numpy reference")
    return len(want)


def main() raises:
    print("=" * 70)
    print("SmolVLA prefix/suffix assembly")
    print("=" * 70)
    var checked = 0

    # ── the time embedding, vs numpy ─────────────────────────────────────
    var e1 = sinusoidal_time_embedding[8](0.001)
    var w1: List[Float64] = [
        1.0, 0.156434465, 0.015707317, 0.001570796,
        0.0, 0.987688341, 0.999876632, 0.999998766,
    ]
    checked += _check(e1, w1, String("[1] t=0.001 D=8 "))

    var e2 = sinusoidal_time_embedding[8](0.5)
    var w2: List[Float64] = [
        0.0, 0.0, 1.0, 0.707106781,
        1.0, -1.0, 0.0, 0.707106781,
    ]
    checked += _check(e2, w2, String("[2] t=0.5   D=8 "))

    var e3 = sinusoidal_time_embedding[8](0.999)
    var w3: List[Float64] = [
        -1.0, -0.156434465, 0.015707317, 0.999998766,
        0.0, 0.987688341, -0.999876632, 0.001570796,
    ]
    checked += _check(e3, w3, String("[3] t=0.999 D=8 "))

    # the real width, spot-checked across both halves
    var e4 = sinusoidal_time_embedding[720](0.37)
    var idx: List[Int] = [0, 1, 359, 360, 361, 719]
    var w4: List[Float64] = [
        0.0, -0.996747946, 0.549022818, -1.0, -0.080582459, 0.835807361,
    ]
    var bad = 0
    var worst = Float64(0)
    for k in range(len(idx)):
        var d = abs(Float64(e4[idx[k]]) - w4[k])
        if d > worst:
            worst = d
        if d > 1e-6:
            bad += 1
    print("   [4] t=0.37 D=720 : compared", len(idx), " wrong", bad,
          " worst", worst)
    assert_true(bad == 0, "the 720-wide embedding disagrees with numpy")
    assert_true(len(e4) == 720, "wrong length")
    checked += len(idx)
    # index 359 is the LAST sine and 360 the FIRST cosine: if the halves were
    # interleaved these two would not straddle the boundary as they do.
    print("       boundary: e[359]=", e4[359], " e[360]=", e4[360],
          " (last sine | first cosine)")

    # ── the language gather and its sqrt(DIM) scale ──────────────────────
    comptime VOCAB = 7
    comptime DIM = 4
    var w = Tensor.alloc(VOCAB * DIM)
    for t in range(VOCAB):
        for d in range(DIM):
            w.data[t * DIM + d] = Scalar[DT](t * 10 + d)
    var ids: List[Int] = [3, 0, 6]
    var out = Tensor.alloc(len(ids) * DIM)
    embed_language_tokens[VOCAB, DIM](w, ids, out, True)
    var f = sqrt(Scalar[DT](DIM))
    var gbad = 0
    for t in range(len(ids)):
        for d in range(DIM):
            var want = Scalar[DT](ids[t] * 10 + d) * f
            if abs(out.data[t * DIM + d] - want) > Scalar[DT](1e-4):
                gbad += 1
    print("   [5] language gather x sqrt(DIM): compared", len(ids) * DIM,
          " wrong", gbad, " (scale", f, ")")
    assert_true(gbad == 0, "the gather or the sqrt(DIM) scale is wrong")

    # unscaled must differ — otherwise the flag does nothing and the check above
    # would pass with the scale silently dropped
    var raw = Tensor.alloc(len(ids) * DIM)
    embed_language_tokens[VOCAB, DIM](w, ids, raw, False)
    var differs = 0
    for i in range(len(ids) * DIM):
        if raw.data[i] != out.data[i]:
            differs += 1
    assert_true(differs > 0, "scaled and unscaled gathers are identical — the"
                             " sqrt(DIM) factor is not being applied")
    print("       unscaled differs in", differs, "of", len(ids) * DIM,
          "-> the scale is real")

    # out-of-vocab must raise rather than read past the table
    var raised = False
    var bad_ids: List[Int] = [VOCAB]
    var tmp = Tensor.alloc(DIM)
    try:
        embed_language_tokens[VOCAB, DIM](w, bad_ids, tmp, True)
    except:
        raised = True
    assert_true(raised, "an out-of-vocab id did not raise")
    print("   [6] out-of-vocab id raised:", raised)

    print()
    print("PASSED —", checked, "values against numpy")
