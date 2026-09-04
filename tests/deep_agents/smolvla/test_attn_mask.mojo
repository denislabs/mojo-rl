"""The prefix-LM block mask, against the reference's own worked examples.

`make_att_2d_masks` documents three cases in its docstring. They are an
independent specification — written by the authors, not derived from our
transcription — so the gate checks those first, computing each expectation from
its stated MEANING rather than from the cumsum rule under test:

    [1 1 1 1 1 1]           pure causal            -> j <= i
    [0 0 0 1 1 1]           prefix-LM              -> first 3 mutual, rest causal
    [1 0 1 0 1 0 0 1 0 0]   4 blocks               -> block(j) <= block(i)

Then SmolVLA's own layout, checked against the claims its source states in
comments: image and language must NOT see state or actions; the action chunk
must be causal within itself.

Then the three windows must agree with the square mask they slice, since the
prefill and denoising masks are separately derived and drifting apart is exactly
how one of them ends up right and the other quietly wrong.

Finally — and this is the leg V2 rests on — the prefix must be CLOSED to the
suffix. `lerobot`'s training forward is a different mode from its inference
forward: one joint `[P+S, P+S]` pass with no cache, versus a prefill and a
denoising step. They compute the same thing only if no prefix token attends to
an action token, and that is a property of this mask and nothing else. If it
holds, V2 needs no new forward driver at all; if it ever stopped holding, the
inference path would still run, still produce plausible chunks, and quietly
stop being the model we are training.

⚠ Under most mutations of the mask rule legs [1]-[3] fire first, so leg [6] is
rarely the one that speaks. MEASURED, over the two `ar` variants `smolvla_ar`'s
own docstring calls out, on a 8-image/1-state/5-action layout:

    shipped, actions [1]*c        45 quadrant entries, 0 open
    actions [1] + [0]*(c-1)       45 quadrant entries, 0 open
    actions [0]*c                 45 quadrant entries, 5 open

Only the third opens it, and it opens exactly the STATE row — image and
language sit at cumsum 0 and stay closed regardless, while state at 1 becomes
mutual with an action span that opens no block. So the property this leg
guards is narrower than it looks: it is the state token, and the pi-0-style
single bidirectional chunk does NOT threaten it.

Run:
  pixi run mojo run -I . tests/deep_agents/smolvla/test_attn_mask.mojo
"""

from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.masked_attention import MASK_NEG
from mojo_rl.deep_agents.smolvla.attn_mask import (
    att_2d_mask, att_2d_mask_square, smolvla_ar, cumsum_blocks,
)


def _allow(ref m: List[Scalar[DT]], cols: Int, i: Int, j: Int) -> Bool:
    return m[i * cols + j] == Scalar[DT](0.0)


def _case(ref ar: List[Int], ref expect: List[Bool], label: String) raises:
    var n = len(ar)
    var m = att_2d_mask_square(ar)
    assert_equal(len(m), n * n, label + ": wrong mask size")
    var cmp = 0
    var bad = 0
    for i in range(n):
        for j in range(n):
            cmp += 1
            if _allow(m, n, i, j) != expect[i * n + j]:
                bad += 1
    print("  ", label, ": compared", cmp, " wrong", bad)
    assert_true(bad == 0, label + ": mask disagrees with the documented meaning")


def main() raises:
    print("=" * 70)
    print("SmolVLA prefix-LM block mask")
    print("=" * 70)

    # ── 1. [1 1 1 1 1 1] is pure causal ──────────────────────────────────
    var ar1 = List[Int]()
    for _ in range(6):
        ar1.append(1)
    var e1 = List[Bool]()
    for i in range(6):
        for j in range(6):
            e1.append(j <= i)          # the MEANING: causal
    _case(ar1, e1, String("[1] pure causal        "))

    # ── 2. [0 0 0 1 1 1] is prefix-LM ────────────────────────────────────
    var ar2: List[Int] = [0, 0, 0, 1, 1, 1]
    var e2 = List[Bool]()
    for i in range(6):
        for j in range(6):
            # first three mutual; a later token sees the prefix and back to itself
            if i < 3:
                e2.append(j < 3)
            else:
                e2.append(j <= i)
    _case(ar2, e2, String("[2] prefix-LM          "))

    # ── 3. [1 0 1 0 1 0 0 1 0 0] is four blocks ──────────────────────────
    var ar3: List[Int] = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    # blocks by construction: {0,1} {2,3} {4,5,6} {7,8,9}
    var blk: List[Int] = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3]
    var e3 = List[Bool]()
    for i in range(10):
        for j in range(10):
            e3.append(blk[j] <= blk[i])   # same block, or an earlier one
    _case(ar3, e3, String("[3] four blocks        "))

    # ── 4. SmolVLA's layout, against the claims its source makes ─────────
    comptime NIMG = 8
    comptime NLANG = 4
    comptime NSTATE = 1
    comptime CHUNK = 5
    var ar = smolvla_ar(NIMG, NLANG, NSTATE, CHUNK)
    var n = len(ar)
    assert_equal(n, NIMG + NLANG + NSTATE + CHUNK, "ar length")
    var m = att_2d_mask_square(ar)
    comptime P = NIMG + NLANG          # the bidirectional visual+text block
    comptime S0 = P + NSTATE           # first action token

    var checked = 0
    # the prefix is ONE bidirectional block: every pair mutually visible
    for i in range(P):
        for j in range(P):
            checked += 1
            assert_true(_allow(m, n, i, j), "prefix should be bidirectional")
    # "image and language inputs do not attend to state or actions"
    for i in range(P):
        for j in range(P, n):
            checked += 1
            assert_true(not _allow(m, n, i, j),
                        "prefix must NOT see state or actions")
    # state sees the prefix and itself, not the actions
    for j in range(P + NSTATE):
        checked += 1
        assert_true(_allow(m, n, P, j), "state should see prefix + itself")
    for j in range(S0, n):
        checked += 1
        assert_true(not _allow(m, n, P, j), "state must NOT see actions")
    # the action chunk is CAUSAL within itself
    for a in range(CHUNK):
        var i = S0 + a
        for b in range(CHUNK):
            var j = S0 + b
            checked += 1
            assert_true(_allow(m, n, i, j) == (b <= a),
                        "the action chunk should be causal within itself")
        for j in range(P + NSTATE):
            checked += 1
            assert_true(_allow(m, n, i, j), "actions should see prefix+state")
    print("   [4] SmolVLA layout      : compared", checked, " wrong 0")

    # ── 5. the windows agree with the square mask they slice ─────────────
    comptime PFX = P + NSTATE
    var self_m = att_2d_mask(ar, PFX, n, 0, n)        # denoise self  [S, P+S]
    var cross_m = att_2d_mask(ar, PFX, n, 0, PFX)     # denoise cross [S, P]
    var wcmp = 0
    var wbad = 0
    for a in range(n - PFX):
        for j in range(n):
            wcmp += 1
            if (self_m[a * n + j] == Scalar[DT](0.0)) != _allow(m, n, PFX + a, j):
                wbad += 1
        for j in range(PFX):
            wcmp += 1
            if (cross_m[a * PFX + j] == Scalar[DT](0.0)) != _allow(m, n, PFX + a, j):
                wbad += 1
    print("   [5] windows vs square   : compared", wcmp, " wrong", wbad)
    assert_equal(len(self_m), (n - PFX) * n, "self window size")
    assert_equal(len(cross_m), (n - PFX) * PFX, "cross window size")
    assert_true(wbad == 0, "a windowed mask disagrees with the square one")

    # ── 6. the prefix cannot see the suffix — what lets V2 reuse V1 ──────
    # ⚠ This quadrant is why the TRAINING forward is not a new driver.
    #
    # `modeling_smolvla.forward` (the loss path) runs mode 1: both streams
    # concatenated, `use_cache=False`, one `[P+S, P+S]` attention per layer.
    # Inference runs modes 2+3: prefill into a cache, then a denoising step
    # reading it. Those are the same computation ONLY IF no prefix query
    # attends to a suffix key — otherwise the prefix's output would depend on
    # the action tokens, the cache would be a function of the noise, and the
    # two decompositions would silently diverge.
    #
    # Leg [5] compares the SUFFIX rows. This one is about the PREFIX rows,
    # which no window in this file covers: it is the quadrant the inference
    # path never materialises precisely because it is assumed empty.
    var quad = 0
    var leak = 0
    for i in range(PFX):
        for j in range(PFX, n):
            quad += 1
            if _allow(m, n, i, j):
                leak += 1
    print("   [6] prefix -> suffix    : compared", quad, " open", leak,
          " (must be 0)")
    assert_equal(quad, PFX * (n - PFX), "the whole quadrant must be probed")
    assert_true(
        leak == 0,
        "a prefix token attends to an action token: the training forward no"
        " longer decomposes into prefill + one denoising step",
    )

    print()
    print("PASSED")
