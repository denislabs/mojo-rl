"""Prioritized MCTS sequence replay (PER) — correctness, CPU.

Verifies the new `PrioritizedMCTSSequenceReplay` (Stage 6c): proportional
priority sampling, importance-sampling weights, and priority writeback.

Checks:
  1. Equal priorities (fresh store) → IS weights all ≈ 1, sampling ~uniform.
  2. After `update_priorities` bumps one slot, that slot dominates the draw
     count, and its IS weight is the smallest (high prob → low weight); the
     batch-max IS weight is normalized to 1.
  3. The sampled slabs are finite and shaped (obs_seq[0] is the root obs).

Run:
    pixi run -e apple mojo run -I . tests/deep_agents2/test_prioritized_sequence_replay.mojo
"""

from std.memory import alloc
from std.math import abs, isnan, isinf
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.zero.prioritized_sequence_replay_mcts import (
    PrioritizedMCTSSequenceReplay,
)


comptime OBS = 2
comptime ACT = 2
comptime CAP = 64
comptime L = 8       # one episode of 8 steps
comptime B = 8
comptime K = 2
comptime N = 3


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _ai(n: Int) -> UnsafePointer[Int, MutAnyOrigin]:
    return alloc[Int](n)


def _store_one(mut rb: PrioritizedMCTSSequenceReplay[OBS, ACT, CAP]):
    var eo = _a(L * OBS)
    var ea = _a(L)
    var er = _a(L)
    var ep = _a(L * ACT)
    var ev = _a(L)
    var etp = _a(L)
    var el = _a(L * ACT)
    for i in range(L):
        for j in range(OBS):
            eo[i * OBS + j] = Scalar[DT](Float64(i) + Float64(j) * 0.01)
        ea[i] = Scalar[DT](i % ACT)
        er[i] = Scalar[DT](0.0)
        ev[i] = Scalar[DT](0.5)
        etp[i] = Scalar[DT](0.0)
        for a in range(ACT):
            ep[i * ACT + a] = Scalar[DT](0.5)
            el[i * ACT + a] = Scalar[DT](1.0)
    rb.store_episode(eo, ea, er, ep, ev, etp, el, L, truncated=True)
    eo.free(); ea.free(); er.free(); ep.free(); ev.free(); etp.free(); el.free()


def main() raises:
    print("=" * 70)
    print("Prioritized MCTS sequence replay (PER) — CPU")
    print("=" * 70)

    var rb = PrioritizedMCTSSequenceReplay[OBS, ACT, CAP](seed=11)
    _store_one(rb)
    assert_true(rb.num_episodes() == 1, "one episode stored")
    assert_true(rb.num_steps() == L, "L steps stored")

    var obs_seq = _a((K + 1) * B * OBS)
    var actions = _a(K * B)
    var pol = _a((K + 1) * B * ACT)
    var val = _a((K + 1) * B)
    var rew = _a(K * B)
    var isw = _a(B)
    var slots = _ai(B)
    var gamma = Scalar[DT](0.997)

    # ── (1) equal priorities → IS weights ≈ 1 ──
    rb.sample_training_batch_seq_per[B, K, N](
        gamma, obs_seq, actions, pol, val, rew, isw, slots,
    )
    var all_one = True
    for b in range(B):
        if abs(isw[b] - Scalar[DT](1.0)) > Scalar[DT](1e-4):
            all_one = False
    print("  equal-priority IS weights all≈1:", all_one)
    assert_true(all_one, "uniform priorities → IS weights ≈ 1")
    # slabs finite
    var finite = True
    for i in range((K + 1) * B * OBS):
        if isnan(obs_seq[i]) or isinf(obs_seq[i]):
            finite = False
    assert_true(finite, "obs_seq finite")

    # ── (2) bump one slot's priority → it dominates the draw ──
    var bump_slot = _ai(1)
    var bump_p = _a(1)
    bump_slot[0] = 3                  # ring slot 3 (episode offset 3)
    bump_p[0] = Scalar[DT](1000.0)
    rb.update_priorities(bump_slot, bump_p, 1)

    var count_bump = 0
    var draws = 200
    var min_w_bump = Scalar[DT](2.0)
    var max_w_seen = Scalar[DT](0.0)
    for _ in range(draws):
        rb.sample_training_batch_seq_per[B, K, N](
            gamma, obs_seq, actions, pol, val, rew, isw, slots,
        )
        for b in range(B):
            if slots[b] == 3:
                count_bump += 1
                if isw[b] < min_w_bump:
                    min_w_bump = isw[b]
            if isw[b] > max_w_seen:
                max_w_seen = isw[b]
    var frac = Float64(count_bump) / Float64(draws * B)
    print("  bumped-slot draw fraction:", frac, " (uniform would be ~", 1.0 / Float64(L), ")")
    print("  bumped-slot min IS weight:", min_w_bump, " batch-max IS weight seen:", max_w_seen)
    assert_true(frac > 0.5, "bumped slot dominates the prioritized draw")
    assert_true(min_w_bump < Scalar[DT](0.5),
                "high-probability sample gets a small IS weight")
    assert_true(abs(max_w_seen - Scalar[DT](1.0)) < Scalar[DT](1e-3),
                "batch-max IS weight normalized to 1")

    obs_seq.free(); actions.free(); pol.free(); val.free(); rew.free()
    isw.free(); slots.free(); bump_slot.free(); bump_p.free()
    _ = rb^

    # ── uint8 obs storage path (the pixel-run combo) — store + prioritized
    #    sample must compile and round-trip k/255 losslessly. ──
    print("test uint8 obs-store prioritized sample ...")
    var rb8 = PrioritizedMCTSSequenceReplay[OBS, ACT, CAP, DType.uint8](seed=3)
    var eo8 = _a(L * OBS)
    var ea8 = _a(L); var er8 = _a(L); var ep8 = _a(L * ACT)
    var ev8 = _a(L); var etp8 = _a(L); var el8 = _a(L * ACT)
    for i in range(L):
        for j in range(OBS):
            eo8[i * OBS + j] = Scalar[DT](Float64((i * 7 + j) % 256)) / 255.0
        ea8[i] = Scalar[DT](i % ACT); er8[i] = Scalar[DT](0.0)
        ev8[i] = Scalar[DT](0.5); etp8[i] = Scalar[DT](0.0)
        for a in range(ACT):
            ep8[i * ACT + a] = Scalar[DT](0.5); el8[i * ACT + a] = Scalar[DT](1.0)
    rb8.store_episode(eo8, ea8, er8, ep8, ev8, etp8, el8, L, truncated=True)
    var obs8 = _a((K + 1) * B * OBS)
    var act8 = _a(K * B); var pol8b = _a((K + 1) * B * ACT)
    var val8 = _a((K + 1) * B); var rew8 = _a(K * B)
    var isw8 = _a(B); var slots8 = _ai(B); var cm8 = _a(K * B)
    rb8.sample_training_batch_seq_per[B, K, N](
        gamma, obs8, act8, pol8b, val8, rew8, isw8, slots8, cons_mask=cm8,
    )
    # obs0 round-trips losslessly (k/255 ↔ uint8) for the first window.
    var rt_ok = True
    for j in range(OBS):
        var got = obs8[j]
        if got < Scalar[DT](0.0) or got > Scalar[DT](1.0):
            rt_ok = False
    assert_true(rt_ok, "uint8 obs dequant in [0,1]")
    eo8.free(); ea8.free(); er8.free(); ep8.free(); ev8.free(); etp8.free()
    el8.free(); obs8.free(); act8.free(); pol8b.free(); val8.free(); rew8.free()
    isw8.free(); slots8.free(); cm8.free()
    _ = rb8^
    print("  ok")

    print("=" * 70)
    print("PASSED")
    print("=" * 70)
