"""GPUMCTSSequenceReplay — device obs ring correctness (Phase 2).

Validates the device-obs MuZero replay in isolation:
  * the **strided-by-N_ENVS** slot formula + device obs store/gather + uint8
    quant round-trip, via `read_obs`: each stored obs row is encoded as
    ``(abs_pos % 256) / 255`` (all lanes equal), so reading episode-relative
    ``offset`` of env 0's episode (abs start 0, stride N_ENVS) must return
    exactly ``((offset · N_ENVS) % 256) / 255`` — bit-lossless for k/255.
  * episode bookkeeping: closing one env's episode at a time grows
    `num_episodes` / `num_steps` correctly.
  * `sample_training_batch_dev`: gathered obs rows are coherent (constant per
    row, in [0,1]) and the metadata slabs are finite.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_gpu_mcts_replay.mojo
"""

from std.memory import alloc
from std.testing import assert_equal, assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import mptr
from mojo_rl.deep_agents.zero.gpu_sequence_replay_mcts import (
    GPUMCTSSequenceReplay,
)


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(alloc[Scalar[DT]](n))


def _ai(n: Int) -> UnsafePointer[Int, MutAnyOrigin]:
    return alloc[Int](n)


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime CAP = 64
    comptime N_ENVS = 2
    comptime T = 5    # iterations recorded before closing

    var ctx = DeviceContext()
    var rb = GPUMCTSSequenceReplay[OBS, ACT, CAP, N_ENVS, DType.uint8](
        ctx, seed=1
    )

    var d_src = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var h_src = _a(N_ENVS * OBS)
    var h_act = _a(N_ENVS)
    var h_pol = _a(N_ENVS * ACT)
    var h_val = _a(N_ENVS)
    var h_rew = _a(N_ENVS)
    var h_done = _a(N_ENVS)
    var h_term = _a(N_ENVS)

    # ── record T iterations; obs row of env e at iter it encodes abs pos ──
    for it in range(T):
        for e in range(N_ENVS):
            var abs_pos = it * N_ENVS + e
            var v = Scalar[DT](Float64(abs_pos % 256) / 255.0)
            for j in range(OBS):
                h_src[e * OBS + j] = v
            h_act[e] = Scalar[DT](e % ACT)
            h_val[e] = Scalar[DT](0.5)
            for a in range(ACT):
                h_pol[e * ACT + a] = Scalar[DT](1.0) / Scalar[DT](ACT)
            h_rew[e] = Scalar[DT](0.1)
            h_done[e] = Scalar[DT](0.0)
            h_term[e] = Scalar[DT](0.0)
        ctx.enqueue_copy(d_src, h_src)
        rb.record_obs_meta(d_src, h_act, h_pol, h_val)
        # sync before the next iteration overwrites the reused host obs buffer
        # (the real driver reads env.obs_ptr() device→device, so no such hazard).
        ctx.synchronize()
        rb.record_outcome(h_rew, h_done, h_term, max_ep_steps=10_000)

    assert_equal(rb.num_episodes(), 0, "no episodes closed yet")

    # ── close env 0's episode only (terminated) ──
    h_done[0] = Scalar[DT](1.0); h_term[0] = Scalar[DT](1.0)
    h_done[1] = Scalar[DT](0.0); h_term[1] = Scalar[DT](0.0)
    # record one more step so the close lands on a real last step.
    for e in range(N_ENVS):
        var abs_pos = T * N_ENVS + e
        var v = Scalar[DT](Float64(abs_pos % 256) / 255.0)
        for j in range(OBS):
            h_src[e * OBS + j] = v
        h_act[e] = Scalar[DT](0)
        h_val[e] = Scalar[DT](0.5)
        for a in range(ACT):
            h_pol[e * ACT + a] = Scalar[DT](1.0) / Scalar[DT](ACT)
    ctx.enqueue_copy(d_src, h_src)
    rb.record_obs_meta(d_src, h_act, h_pol, h_val)
    ctx.synchronize()
    rb.record_outcome(h_rew, h_done, h_term, max_ep_steps=10_000)

    assert_equal(rb.num_episodes(), 1, "env0 episode closed")
    assert_equal(rb.num_steps(), T + 1, "env0 episode length")

    # ── strided round-trip: env0 episode starts at abs 0, stride N_ENVS ──
    var out = _a(OBS)
    var max_err = 0.0
    for off in range(T + 1):
        rb.read_obs(0, off, out)
        var abs_pos = off * N_ENVS   # ep_start = 0, stride N_ENVS
        # match the dequant exactly: uint8 k → Scalar[DT](k)/255 (float32).
        var expect = Scalar[DT](abs_pos % 256) / Scalar[DT](255.0)
        for j in range(OBS):
            var e = Float64(out[j] - expect)
            if e < 0.0:
                e = -e
            if e > max_err:
                max_err = e
            assert_true(out[j] == expect, "strided obs round-trip not lossless")
    print("strided obs round-trip max |err| =", max_err, "(expect 0.0)")

    # ── close env 1 too, then sample a batch ──
    h_done[1] = Scalar[DT](1.0); h_term[1] = Scalar[DT](1.0)
    h_done[0] = Scalar[DT](0.0); h_term[0] = Scalar[DT](0.0)
    for e in range(N_ENVS):
        for j in range(OBS):
            h_src[e * OBS + j] = Scalar[DT](0.25)
        h_act[e] = Scalar[DT](0); h_val[e] = Scalar[DT](0.5)
        for a in range(ACT):
            h_pol[e * ACT + a] = Scalar[DT](0.5)
    ctx.enqueue_copy(d_src, h_src)
    rb.record_obs_meta(d_src, h_act, h_pol, h_val)
    rb.record_outcome(h_rew, h_done, h_term, max_ep_steps=10_000)
    ctx.synchronize()
    assert_true(rb.num_episodes() >= 2, "env1 episode closed")

    comptime B = 8
    comptime K = 2
    comptime NS = 3
    var d_obs0 = ctx.enqueue_create_buffer[DT](B * OBS)
    var t_act = _a(K * B)
    var t_pol = _a((K + 1) * B * ACT)
    var t_val = _a((K + 1) * B)
    var t_rew = _a(K * B)
    rb.sample_training_batch_dev[B, K, NS](
        Scalar[DT](0.99), d_obs0, t_act, t_pol, t_val, t_rew
    )
    var h_obs0 = _a(B * OBS)
    ctx.enqueue_copy(h_obs0, d_obs0)
    ctx.synchronize()

    # gathered obs rows coherent (constant per row, in [0,1]); slabs finite.
    var ok = True
    for b in range(B):
        var v0 = Float64(h_obs0[b * OBS])
        if v0 < -1e-6 or v0 > 1.0 + 1e-6:
            ok = False
        for j in range(OBS):
            if Float64(h_obs0[b * OBS + j]) != v0:
                ok = False
    assert_true(ok, "gathered obs0 rows not coherent / in range")
    var fin = True
    for i in range((K + 1) * B):
        if not (Float64(t_val[i]) == Float64(t_val[i])):
            fin = False
    assert_true(fin, "value targets non-finite")
    print("sample_training_batch_dev: coherent obs0 + finite targets OK")

    # ── PER path (per=True): prioritized sampler + IS weights + writeback ──
    var rbp = GPUMCTSSequenceReplay[OBS, ACT, CAP, N_ENVS, DType.uint8](
        ctx, seed=7, per=True, alpha=Scalar[DT](1.0), beta=Scalar[DT](1.0)
    )
    # record a handful of steps, closing both envs every few iters so the
    # sum-tree fills with closed steps.
    for it in range(6):
        for e in range(N_ENVS):
            var abs_pos = it * N_ENVS + e
            var v = Scalar[DT](Float64(abs_pos % 256) / 255.0)
            for j in range(OBS):
                h_src[e * OBS + j] = v
            h_act[e] = Scalar[DT](e % ACT); h_val[e] = Scalar[DT](0.3)
            for a in range(ACT):
                h_pol[e * ACT + a] = Scalar[DT](1.0) / Scalar[DT](ACT)
            h_rew[e] = Scalar[DT](0.1)
            var close = it % 3 == 2
            h_done[e] = Scalar[DT](1.0) if close else Scalar[DT](0.0)
            h_term[e] = Scalar[DT](1.0) if close else Scalar[DT](0.0)
        ctx.enqueue_copy(d_src, h_src)
        rbp.record_obs_meta(d_src, h_act, h_pol, h_val)
        ctx.synchronize()
        rbp.record_outcome(h_rew, h_done, h_term, max_ep_steps=10_000)
    assert_true(rbp.num_episodes() > 0, "PER: no episodes closed")

    var d_obs0p = ctx.enqueue_create_buffer[DT](B * OBS)
    var p_act = _a(K * B)
    var p_pol = _a((K + 1) * B * ACT)
    var p_val = _a((K + 1) * B)
    var p_rew = _a(K * B)
    var p_isw = _a(B)
    var p_slots = _ai(B)
    rbp.sample_training_batch_per_dev[B, K, NS](
        Scalar[DT](0.99), d_obs0p, p_act, p_pol, p_val, p_rew, p_isw, p_slots
    )
    var hp_obs0 = _a(B * OBS)
    ctx.enqueue_copy(hp_obs0, d_obs0p)
    ctx.synchronize()
    # IS weights normalized to (0, 1]; targets finite; obs coherent in range.
    var okp = True
    for b in range(B):
        var w = Float64(p_isw[b])
        if w <= 0.0 or w > 1.0 + 1e-6:
            okp = False
        var v0 = Float64(hp_obs0[b * OBS])
        if v0 < -1e-6 or v0 > 1.0 + 1e-6:
            okp = False
    assert_true(okp, "PER: IS weights / obs out of range")
    var finp = True
    for i in range((K + 1) * B):
        if not (Float64(p_val[i]) == Float64(p_val[i])):
            finp = False
    assert_true(finp, "PER: value targets non-finite")
    # priority writeback (use the value targets as a stand-in error signal).
    rbp.update_priorities(p_slots, p_val, B)
    print("sample_training_batch_per_dev: IS weights + finite targets OK")

    print("GPUMCTSSequenceReplay correctness: OK")
