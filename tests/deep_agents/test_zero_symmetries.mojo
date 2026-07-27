"""Board-symmetry augmenter unit tests (CPU).

Validates the invariants every augmenter must satisfy:
  * `sym_idx == 0` is the identity (exact copy).
  * each symmetry is a permutation (bijection) of the cells.
  * obs and policy are permuted *consistently* — driving both with a
    position-encoding (value == cell index) yields identical permutations, so
    the supervised (obs, policy) pairing stays valid.
  * involutions (h-flip, v-flip, 180°, transpose, anti-transpose) square to the
    identity; the two 90° rotations are mutual inverses.

Run:
    pixi run mojo run -I . tests/deep_agents/test_zero_symmetries.mojo
"""

from std.testing import assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.zero.symmetries import (
    IdentityAugmenter, D4SquareAugmenter, HFlipColumnAugmenter,
)


def _buf(n: Int) -> List[Scalar[DT]]:
    return List[Scalar[DT]](length=n, fill=Scalar[DT](0))


def main() raises:
    comptime SIDE = 3
    comptime PLANES = 3
    comptime OBS = PLANES * SIDE * SIDE   # 27
    comptime ACT = SIDE * SIDE            # 9
    comptime D4 = D4SquareAugmenter[SIDE, PLANES]

    # Position-encoded inputs: obs plane p, cell j → value (p*100 + j); policy
    # cell j → value j. Then aug must move values consistently.
    var obs = _buf(OBS)
    var pol = _buf(ACT)
    var aobs = _buf(OBS)
    var apol = _buf(ACT)
    for p in range(PLANES):
        for j in range(ACT):
            obs[p * ACT + j] = Scalar[DT](p * 100 + j)
    for j in range(ACT):
        pol[j] = Scalar[DT](j)

    # Identity (sym 0): exact copy.
    D4.augment_obs[OBS](obs, 0, 0, aobs)
    D4.augment_policy[ACT](pol, 0, 0, apol)
    for i in range(OBS):
        assert_equal(aobs[i], obs[i], "D4 identity obs not a copy")
    for j in range(ACT):
        assert_equal(apol[j], pol[j], "D4 identity policy not a copy")

    # All 8 D4 syms: permutation + obs/policy consistency.
    for s in range(8):
        D4.augment_obs[OBS](obs, 0, s, aobs)
        D4.augment_policy[ACT](pol, 0, s, apol)
        # Plane 0 of obs is position-encoded == policy encoding → identical perm.
        for j in range(ACT):
            assert_equal(
                aobs[j], apol[j],
                "D4 sym " + String(s) + ": obs/policy permutation mismatch",
            )
        # Each plane is a bijection of its cells (sum of indices preserved).
        for p in range(PLANES):
            var seen = InlineArray[Bool, ACT](fill=False)
            for j in range(ACT):
                var v = Int(Float64(aobs[p * ACT + j])) - p * 100
                assert_true(0 <= v and v < ACT, "D4 perm out of range")
                assert_true(not seen[v], "D4 sym not a bijection (dup)")
                seen[v] = True

    # Involutions square to identity: 1 h-flip, 2 v-flip, 3 rot180, 6 transpose,
    # 7 anti-transpose.
    var tmp = _buf(ACT)
    for s_i in [1, 2, 3, 6, 7]:
        var s = s_i
        D4.augment_policy[ACT](pol, 0, s, apol)
        D4.augment_policy[ACT](apol, 0, s, tmp)
        for j in range(ACT):
            assert_equal(tmp[j], pol[j], "D4 sym " + String(s) + " not involutive")

    # 90° (sym 4) and 270° (sym 5) are mutual inverses.
    D4.augment_policy[ACT](pol, 0, 4, apol)
    D4.augment_policy[ACT](apol, 0, 5, tmp)
    for j in range(ACT):
        assert_equal(tmp[j], pol[j], "D4 rot90∘rot270 != identity")

    print("D4SquareAugmenter: OK")

    # ── HFlipColumnAugmenter (Connect4 geometry) ──
    comptime ROWS = 6
    comptime COLS = 7
    comptime C_OBS = 3 * ROWS * COLS
    comptime HF = HFlipColumnAugmenter[ROWS, COLS, 3]

    var cobs = _buf(C_OBS)
    var caobs = _buf(C_OBS)
    var ctmp = _buf(C_OBS)
    for i in range(C_OBS):
        cobs[i] = Scalar[DT](i)
    # h-flip is an involution on the obs.
    HF.augment_obs[C_OBS](cobs, 0, 1, caobs)
    HF.augment_obs[C_OBS](caobs, 0, 1, ctmp)
    for i in range(C_OBS):
        assert_equal(ctmp[i], cobs[i], "HFlip obs not involutive")

    var cpol = _buf(COLS)
    var capol = _buf(COLS)
    for c in range(COLS):
        cpol[c] = Scalar[DT](c)
    HF.augment_policy[COLS](cpol, 0, 1, capol)
    for c in range(COLS):
        assert_equal(
            capol[c], cpol[COLS - 1 - c], "HFlip policy != column reversal"
        )
    print("HFlipColumnAugmenter: OK")

    # Identity augmenter sanity.
    var ip = _buf(ACT)
    IdentityAugmenter.augment_policy[ACT](pol, 0, 0, ip)
    for j in range(ACT):
        assert_equal(ip[j], pol[j], "IdentityAugmenter not a copy")
    print("zero symmetries: OK")
