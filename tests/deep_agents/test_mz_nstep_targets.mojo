"""MuZero n-step value targets — hand-computed trajectories, pure CPU.

Three cases, each checked against a value computed by hand:
  (A) single-player, no terminal: plain discounted n-step + bootstrap.
  (B) single-player, terminal inside the window: bootstrap dropped.
  (C) two-player zero-sum, terminal reward: the P0 sign flip. With the flip
      omitted (the legacy bug) case (C) would give the OPPOSITE sign — this is
      the regression guard.

Run:
    pixi run mojo run -I . tests/deep_agents/test_mz_nstep_targets.mojo
"""

from std.memory import alloc
from std.testing import assert_almost_equal

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.zero.nstep_targets import compute_nstep_value_targets


def _alloc(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def main() raises:
    # ── (A) single-player, K=1, N=2, γ=0.5, no terminal ──
    # to_play all 0 → no sign flips.
    #   k=0: 2 + 0.5·4 + 0.5²·root[2](=5)  = 2 + 2 + 1.25 = 5.25
    #   k=1: 4 + 0.5·6 + 0.5²·root[3](=1)  = 4 + 3 + 0.25 = 7.25
    var rew = _alloc(3)
    rew[0] = 2.0; rew[1] = 4.0; rew[2] = 6.0
    var dne = _alloc(3)
    dne[0] = 0.0; dne[1] = 0.0; dne[2] = 0.0
    var rv = _alloc(4)
    rv[0] = 0.0; rv[1] = 0.0; rv[2] = 5.0; rv[3] = 1.0
    var tp = _alloc(4)
    for i in range(4):
        tp[i] = 0.0
    var vt = _alloc(2)
    compute_nstep_value_targets[1, 2](rew, dne, rv, tp, Scalar[DT](0.5), vt)
    assert_almost_equal(vt[0], Scalar[DT](5.25), atol=1e-5, rtol=1e-5)
    assert_almost_equal(vt[1], Scalar[DT](7.25), atol=1e-5, rtol=1e-5)
    print("(A) single-player + bootstrap: OK")

    # ── (B) single-player, terminal at step 1 inside the window ──
    #   K=1, N=2, γ=1.0, dones=[0,1,0]. For k=0 the sum hits the terminal at
    #   step 1 and stops → no bootstrap:  2 + 1·4 = 6.  (root[2] ignored.)
    dne[0] = 0.0; dne[1] = 1.0; dne[2] = 0.0
    rv[2] = 99.0  # must NOT be used
    compute_nstep_value_targets[1, 2](rew, dne, rv, tp, Scalar[DT](1.0), vt)
    assert_almost_equal(vt[0], Scalar[DT](6.0), atol=1e-5, rtol=1e-5)
    print("(B) terminal cuts the bootstrap: OK")

    rew.free(); dne.free(); rv.free(); tp.free(); vt.free()

    # ── (C) two-player zero-sum, K=1, N=4, γ=1.0 ──
    # to_play=[0,1,0,1,0,1], rewards=[0,0,0,1,0], dones=[0,0,0,1,0].
    # A win delivered as reward +1 at step 3 (the move made by player
    # to_play[3]=1). N=4 so step 3 falls INSIDE k=0's n-step sum (i=3), not at
    # the bootstrap horizon.
    #   k=0 (perspective P0): reward at step3 flips (to_play3=1≠0) → -1 → target -1
    #       (P0's view of "P1 won" = loss).
    #   k=1 (perspective P1): reward at step3 keeps sign (to_play3=1==1) → +1
    #       (P1's view of its own win).
    # WITHOUT the sign flip, k=0 would be +1 — the legacy P0 bug.
    var rew2 = _alloc(5)
    rew2[0] = 0.0; rew2[1] = 0.0; rew2[2] = 0.0; rew2[3] = 1.0; rew2[4] = 0.0
    var dne2 = _alloc(5)
    dne2[0] = 0.0; dne2[1] = 0.0; dne2[2] = 0.0; dne2[3] = 1.0; dne2[4] = 0.0
    var rv2 = _alloc(6)
    for i in range(6):
        rv2[i] = 0.0
    var tp2 = _alloc(6)
    tp2[0] = 0.0; tp2[1] = 1.0; tp2[2] = 0.0
    tp2[3] = 1.0; tp2[4] = 0.0; tp2[5] = 1.0
    var vt2 = _alloc(2)
    compute_nstep_value_targets[1, 4](rew2, dne2, rv2, tp2, Scalar[DT](1.0), vt2)
    assert_almost_equal(
        vt2[0], Scalar[DT](-1.0), atol=1e-5, rtol=1e-5,
        msg="P0 perspective of a P1 win must be -1 (sign flip)",
    )
    assert_almost_equal(
        vt2[1], Scalar[DT](1.0), atol=1e-5, rtol=1e-5,
        msg="P1 perspective of its own win must be +1",
    )
    print("(C) two-player sign flip (P0-bug guard): OK")

    rew2.free(); dne2.free(); rv2.free(); tp2.free(); vt2.free()
    print("MuZero n-step value targets + two-player sign flips: OK")
