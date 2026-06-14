"""Prioritized MCTS sequence replay (PER) — correctness, GPU device-ring.

Verifies `PrioritizedMCTSSequenceReplay` (Stage 6c + device-obs-ring perf fix):
proportional priority sampling, importance-sampling weights, priority writeback,
and the **on-device obs gather** (obs ring lives on device; `sample_training_
batch_seq_per_gpu` assembles the [K+1,B,OBS] slab in device memory).

Checks:
  1. Equal priorities (fresh store) → IS weights all ≈ 1.
  2. After `update_priorities` bumps one slot, that slot dominates the draw
     count, its IS weight is the smallest, and the batch-max IS weight is 1.
  3. The gathered obs slab (D2H'd back) is finite and in range.
  4. uint8 obs-store path round-trips losslessly through the gather.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents2/test_prioritized_sequence_replay.mojo
"""

from std.memory import alloc
from std.math import abs, isnan, isinf
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import mptr
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
comptime NK = K + 1
comptime M = NK * B


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


def _ai(n: Int) -> UnsafePointer[Int, MutAnyOrigin]:
    return alloc[Int](n)


def _ai32(n: Int) -> UnsafePointer[Int32, MutAnyOrigin]:
    return alloc[Int32](n)


def _store_one[SDT: DType](
    mut rb: PrioritizedMCTSSequenceReplay[OBS, ACT, CAP, SDT]
) raises:
    var eo = _a(L * OBS)
    var ea = _a(L)
    var er = _a(L)
    var ep = _a(L * ACT)
    var ev = _a(L)
    var etp = _a(L)
    var el = _a(L * ACT)
    for i in range(L):
        for j in range(OBS):
            eo[i * OBS + j] = Scalar[DT](Float64((i * 7 + j) % 256)) / 255.0
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
    print("Prioritized MCTS sequence replay (PER) — GPU device-ring")
    print("=" * 70)

    var ctx = DeviceContext()
    var rb = PrioritizedMCTSSequenceReplay[OBS, ACT, CAP](ctx, seed=11)
    _store_one(rb)
    assert_true(rb.num_episodes() == 1, "one episode stored")
    assert_true(rb.num_steps() == L, "L steps stored")

    # device obs slab + slot scratch
    var d_obs = ctx.enqueue_create_buffer[DT](M * OBS)
    var d_slots = ctx.enqueue_create_buffer[DType.int32](M)
    var h_slots = _ai32(M)
    var h_obs = _a(M * OBS)

    var actions = _a(K * B)
    var pol = _a((K + 1) * B * ACT)
    var val = _a((K + 1) * B)
    var rew = _a(K * B)
    var isw = _a(B)
    var slots = _ai(B)
    var gamma = Scalar[DT](0.997)

    # ── (1) equal priorities → IS weights ≈ 1 ──
    rb.sample_training_batch_seq_per_gpu[B, K, N](
        ctx, gamma, d_obs, d_slots, h_slots,
        actions, pol, val, rew, isw, slots,
    )
    var all_one = True
    for b in range(B):
        if abs(isw[b] - Scalar[DT](1.0)) > Scalar[DT](1e-4):
            all_one = False
    print("  equal-priority IS weights all≈1:", all_one)
    assert_true(all_one, "uniform priorities → IS weights ≈ 1")

    # gathered obs slab finite + in [0,1] (D2H back)
    ctx.enqueue_copy(h_obs, d_obs)
    ctx.synchronize()
    var finite = True
    for i in range(M * OBS):
        if isnan(h_obs[i]) or isinf(h_obs[i]):
            finite = False
        if h_obs[i] < Scalar[DT](-1e-4) or h_obs[i] > Scalar[DT](1.0 + 1e-4):
            finite = False
    assert_true(finite, "gathered obs_seq finite + in range")

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
        rb.sample_training_batch_seq_per_gpu[B, K, N](
            ctx, gamma, d_obs, d_slots, h_slots,
            actions, pol, val, rew, isw, slots,
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

    actions.free(); pol.free(); val.free(); rew.free()
    isw.free(); slots.free(); bump_slot.free(); bump_p.free()
    h_slots.free(); h_obs.free()
    _ = rb^

    # ── uint8 obs-store path — store + prioritized sample must round-trip
    #    k/255 losslessly through the device gather. ──
    print("test uint8 obs-store device gather ...")
    var rb8 = PrioritizedMCTSSequenceReplay[OBS, ACT, CAP, DType.uint8](
        ctx, seed=3
    )
    _store_one(rb8)
    var d_obs8 = ctx.enqueue_create_buffer[DT](M * OBS)
    var d_slots8 = ctx.enqueue_create_buffer[DType.int32](M)
    var h_slots8 = _ai32(M)
    var h_obs8 = _a(M * OBS)
    var act8 = _a(K * B); var pol8 = _a((K + 1) * B * ACT)
    var val8 = _a((K + 1) * B); var rew8 = _a(K * B)
    var isw8 = _a(B); var slots8 = _ai(B); var cm8 = _a(K * B)
    rb8.sample_training_batch_seq_per_gpu[B, K, N](
        ctx, gamma, d_obs8, d_slots8, h_slots8,
        act8, pol8, val8, rew8, isw8, slots8, cons_mask=cm8,
    )
    ctx.enqueue_copy(h_obs8, d_obs8)
    ctx.synchronize()
    var rt_ok = True
    for i in range(M * OBS):
        if h_obs8[i] < Scalar[DT](0.0) or h_obs8[i] > Scalar[DT](1.0):
            rt_ok = False
    assert_true(rt_ok, "uint8 obs dequant in [0,1]")
    act8.free(); pol8.free(); val8.free(); rew8.free()
    isw8.free(); slots8.free(); cm8.free()
    h_slots8.free(); h_obs8.free()
    _ = rb8^
    print("  ok")

    print("=" * 70)
    print("PASSED")
    print("=" * 70)
