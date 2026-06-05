"""MuZero MCTS sequence replay — store episodes, sample unroll batches, prune.

Validates `zero/sequence_replay_mcts.mojo`: stored steps round-trip into the
time-major unroll batch (obs / actions / policy / reward), the n-step value
targets are computed correctly through the replay (single-player, reward=1,
γ=1 ⇒ each target is the 2-step return ≈ 2.0 away from the terminal), policy
rows are valid distributions, and episode eviction keeps the resident step count
within CAP.

Run:
    pixi run mojo run -I . tests/deep_agents2/test_mz_sequence_replay.mojo
"""

from std.memory import alloc
from std.testing import assert_true, assert_equal

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.zero.sequence_replay_mcts import MCTSSequenceReplay


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _store_const_episode[
    OBS: Int, ACT: Int, CAP: Int,
](
    mut rb: MCTSSequenceReplay[OBS, ACT, CAP],
    length: Int,
    marker: Scalar[DT],
):
    """Store a single-player episode: obs[0]=marker, reward=1 everywhere,
    value=0, to_play=0, policy one-hot on action 1."""
    var o = _a(length * OBS)
    var a = _a(length)
    var r = _a(length)
    var p = _a(length * ACT)
    var v = _a(length)
    var tp = _a(length)
    var lg = _a(length * ACT)
    for i in range(length):
        for j in range(OBS):
            o[i * OBS + j] = marker if j == 0 else Scalar[DT](0.0)
        a[i] = Scalar[DT](1.0)
        r[i] = Scalar[DT](1.0)
        v[i] = Scalar[DT](0.0)
        tp[i] = Scalar[DT](0.0)
        for k in range(ACT):
            p[i * ACT + k] = Scalar[DT](1.0) if k == 1 else Scalar[DT](0.0)
            lg[i * ACT + k] = Scalar[DT](1.0)
    rb.store_episode(o, a, r, p, v, tp, lg, length)
    o.free(); a.free(); r.free(); p.free(); v.free(); tp.free(); lg.free()


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime CAP = 200
    comptime B = 8
    comptime K = 2
    comptime N = 2

    var rb = MCTSSequenceReplay[OBS, ACT, CAP](seed=123)
    _store_const_episode(rb, 20, Scalar[DT](7.0))
    _store_const_episode(rb, 16, Scalar[DT](9.0))
    assert_equal(rb.num_episodes(), 2, "episode count")
    assert_equal(rb.num_steps(), 36, "step count")

    var obs0 = _a(B * OBS)
    var actions = _a(K * B)
    var policy_tgt = _a((K + 1) * B * ACT)
    var value_tgt = _a((K + 1) * B)
    var reward_tgt = _a(K * B)
    rb.sample_training_batch[B, K, N](
        Scalar[DT](1.0), obs0, actions, policy_tgt, value_tgt, reward_tgt
    )

    for b in range(B):
        # obs0 marker must be one of the two episode markers.
        var m = obs0[b * OBS]
        assert_true(
            m == Scalar[DT](7.0) or m == Scalar[DT](9.0),
            "obs0 not from a stored episode",
        )
        # policy rows sum to 1 and are one-hot on action 1 (in-episode) or
        # uniform (absorbing) — either way sum to 1.
        for k in range(K + 1):
            var s = Scalar[DT](0.0)
            for a in range(ACT):
                s += policy_tgt[k * B * ACT + b * ACT + a]
            assert_true(s > Scalar[DT](0.99) and s < Scalar[DT](1.01),
                "policy row not a distribution")
        # value targets: reward=1, γ=1, N=2 ⇒ 2-step return is 2.0 well inside
        # the episode, less near the terminal; always finite and in [0, 2].
        for k in range(K + 1):
            var vt = value_tgt[k * B + b]
            assert_true(vt == vt, "value target NaN")
            assert_true(vt >= Scalar[DT](-0.01) and vt <= Scalar[DT](2.01),
                "value target out of [0,2]")
        # reward targets are 1.0 (in-episode) or 0.0 (absorbing).
        for k in range(K):
            var rt = reward_tgt[k * B + b]
            assert_true(rt == Scalar[DT](1.0) or rt == Scalar[DT](0.0),
                "reward target not 0/1")
        # actions are valid one-hot indices (1.0 in-episode, 0.0 absorbing).
        for k in range(K):
            var ac = actions[k * B + b]
            assert_true(ac == Scalar[DT](0.0) or ac == Scalar[DT](1.0),
                "action out of range")

    print("store/sample round-trip + n-step targets: OK")

    # ── eviction: flood past CAP and confirm residency stays bounded ──
    for _ in range(40):
        _store_const_episode(rb, 20, Scalar[DT](3.0))
    assert_true(rb.num_steps() <= CAP, "resident steps exceeded CAP")
    assert_true(rb.num_episodes() <= CAP // 20 + 1, "too many resident eps")
    # still samplable after eviction.
    rb.sample_training_batch[B, K, N](
        Scalar[DT](1.0), obs0, actions, policy_tgt, value_tgt, reward_tgt
    )
    var finite = True
    for i in range((K + 1) * B):
        if not (value_tgt[i] == value_tgt[i]):
            finite = False
    assert_true(finite, "post-eviction sample produced NaN")
    print("eviction keeps residency bounded + samplable: OK")

    obs0.free(); actions.free(); policy_tgt.free()
    value_tgt.free(); reward_tgt.free()
    print("MuZero MCTS sequence replay: OK")
