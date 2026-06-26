"""SequenceReplayBuffer trait + CPU/GPU sibling (test_gpu_sequence_replay).

Verifies:
  1. CPU SequenceReplay conforms (make["cpu"], record, count, can_sample,
     sample_batch) and gathers contiguous windows.
  2. GPU GPUSequenceReplay conforms (make["gpu"], record, count,
     sample_batch_dev native + sample_batch host bridge) and gathers the
     same contiguous windows.
  3. record_batch (device multi-env store) lands N_ENVS lockstep slots.

Correctness is RNG-independent: slot i is tagged so every field == i. Any
valid window then has frame k tag == (frame k-1 tag + 1) mod CAP — a check
that holds for both backends regardless of which start each draws.

Run:  pixi run -e apple mojo run -I . tests/nn/dreamerv3/test_gpu_sequence_replay.mojo
"""

from std.memory import alloc
from std.testing import assert_true, assert_equal
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.data.sequence_replay import SequenceReplay
from mojo_rl.deep_agents.data.gpu_sequence_replay import GPUSequenceReplay

comptime OBS = 3
comptime ACT = 2
comptime CAP = 32
comptime B = 8
comptime T = 4


def _check_window_contiguous[
    BB: Int, TT: Int
](
    obs_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    act_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    rew_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    dne_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    name: String,
) raises:
    """Each window's frame tags must increase by 1 (mod CAP). obs has TT+1
    frames, act/rew/dne have TT frames; obs[k] tag == act/rew/dne[k] tag."""
    for b in range(BB):
        # tag of obs frame 0 (all OBS lanes equal)
        var t0 = obs_out[b * (TT + 1) * OBS + 0]
        for k in range(TT + 1):
            var tag = obs_out[b * (TT + 1) * OBS + k * OBS]
            # all OBS lanes equal the frame tag
            for i in range(OBS):
                assert_equal(
                    obs_out[b * (TT + 1) * OBS + k * OBS + i], tag,
                    name + ": obs lane mismatch",
                )
            var expected = Scalar[DT](
                (Int(t0) + k) % CAP
            )
            assert_equal(tag, expected, name + ": obs frame not contiguous")
        for k in range(TT):
            var atag = act_out[b * TT * ACT + k * ACT]
            for j in range(ACT):
                assert_equal(
                    act_out[b * TT * ACT + k * ACT + j], atag,
                    name + ": act lane mismatch",
                )
            var expected = Scalar[DT]((Int(t0) + k) % CAP)
            assert_equal(atag, expected, name + ": act tag mismatch")
            assert_equal(
                rew_out[b * TT + k], expected, name + ": rew tag mismatch"
            )
            assert_equal(
                dne_out[b * TT + k], expected, name + ": dne tag mismatch"
            )


def test_cpu() raises:
    print("--- CPU SequenceReplay ---")
    var buf = SequenceReplay[OBS, ACT, CAP].make["cpu"]()
    assert_equal(buf.count(), 0, "fresh count")
    assert_true(not buf.can_sample[T](), "cannot sample empty")

    var o = alloc[Scalar[DT]](OBS)
    var a = alloc[Scalar[DT]](ACT)
    # Record CAP transitions; slot i tagged with value i in every field.
    for i in range(CAP):
        for d in range(OBS):
            o[d] = Scalar[DT](i)
        for j in range(ACT):
            a[j] = Scalar[DT](i)
        buf.record(o, a, Scalar[DT](i), Scalar[DT](i))
    assert_equal(buf.count(), CAP, "saturated count")
    assert_true(buf.can_sample[T](), "can sample when full")

    var obs_out = alloc[Scalar[DT]](B * (T + 1) * OBS)
    var act_out = alloc[Scalar[DT]](B * T * ACT)
    var rew_out = alloc[Scalar[DT]](B * T)
    var dne_out = alloc[Scalar[DT]](B * T)
    buf.sample_batch[B, T](obs_out, act_out, rew_out, dne_out)
    _check_window_contiguous[B, T](obs_out, act_out, rew_out, dne_out, "cpu")
    print("  OK  count=", buf.count(), " windows contiguous")


def test_gpu(ctx: DeviceContext) raises:
    print("--- GPU GPUSequenceReplay ---")
    var buf = GPUSequenceReplay[OBS, ACT, CAP].make["gpu"](ctx=ctx)
    assert_equal(buf.count(), 0, "fresh count")
    assert_true(not buf.can_sample[T](), "cannot sample empty")

    var o = alloc[Scalar[DT]](OBS)
    var a = alloc[Scalar[DT]](ACT)
    for i in range(CAP):
        for d in range(OBS):
            o[d] = Scalar[DT](i)
        for j in range(ACT):
            a[j] = Scalar[DT](i)
        buf.record(o, a, Scalar[DT](i), Scalar[DT](i))
    ctx.synchronize()
    assert_equal(buf.count(), CAP, "saturated count")
    assert_true(buf.can_sample[T](), "can sample when full")

    # Host-bridge sample.
    var obs_out = alloc[Scalar[DT]](B * (T + 1) * OBS)
    var act_out = alloc[Scalar[DT]](B * T * ACT)
    var rew_out = alloc[Scalar[DT]](B * T)
    var dne_out = alloc[Scalar[DT]](B * T)
    buf.sample_batch[B, T](obs_out, act_out, rew_out, dne_out)
    _check_window_contiguous[B, T](
        obs_out, act_out, rew_out, dne_out, "gpu-bridge"
    )

    # Native device sample → copy out → same check.
    var d_obs = ctx.enqueue_create_buffer[DT](B * (T + 1) * OBS)
    var d_act = ctx.enqueue_create_buffer[DT](B * T * ACT)
    var d_rew = ctx.enqueue_create_buffer[DT](B * T)
    var d_dne = ctx.enqueue_create_buffer[DT](B * T)
    buf.sample_batch_dev[B, T](ctx, d_obs, d_act, d_rew, d_dne)
    ctx.enqueue_copy(obs_out, d_obs)
    ctx.enqueue_copy(act_out, d_act)
    ctx.enqueue_copy(rew_out, d_rew)
    ctx.enqueue_copy(dne_out, d_dne)
    ctx.synchronize()
    _check_window_contiguous[B, T](
        obs_out, act_out, rew_out, dne_out, "gpu-dev"
    )
    print("  OK  count=", buf.count(), " host-bridge + device windows contiguous")


def test_gpu_record_batch(ctx: DeviceContext) raises:
    print("--- GPU record_batch (multi-env store) ---")
    comptime N_ENVS = 4
    var buf = GPUSequenceReplay[OBS, ACT, CAP].make["gpu"](ctx=ctx)
    # Two lockstep stores of N_ENVS each → 8 slots, contiguous tags 0..7.
    var s_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var s_act = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var s_rew = ctx.enqueue_create_buffer[DT](N_ENVS)
    var s_dne = ctx.enqueue_create_buffer[DT](N_ENVS)
    var ho = alloc[Scalar[DT]](N_ENVS * OBS)
    var ha = alloc[Scalar[DT]](N_ENVS * ACT)
    var hr = alloc[Scalar[DT]](N_ENVS)
    var hd = alloc[Scalar[DT]](N_ENVS)
    var rounds = (T + 2) // N_ENVS + 2  # enough to exceed T+1
    for rnd in range(rounds):
        for e in range(N_ENVS):
            var tag = rnd * N_ENVS + e
            for d in range(OBS):
                ho[e * OBS + d] = Scalar[DT](tag)
            for j in range(ACT):
                ha[e * ACT + j] = Scalar[DT](tag)
            hr[e] = Scalar[DT](tag)
            hd[e] = Scalar[DT](tag)
        ctx.enqueue_copy(s_obs, ho)
        ctx.enqueue_copy(s_act, ha)
        ctx.enqueue_copy(s_rew, hr)
        ctx.enqueue_copy(s_dne, hd)
        buf.record_batch[N_ENVS](ctx, s_obs, s_act, s_rew, s_dne)
    ctx.synchronize()
    assert_equal(buf.count(), rounds * N_ENVS, "record_batch count")
    assert_true(buf.can_sample[T](), "can sample after record_batch")

    var obs_out = alloc[Scalar[DT]](B * (T + 1) * OBS)
    var act_out = alloc[Scalar[DT]](B * T * ACT)
    var rew_out = alloc[Scalar[DT]](B * T)
    var dne_out = alloc[Scalar[DT]](B * T)
    buf.sample_batch[B, T](obs_out, act_out, rew_out, dne_out)
    _check_window_contiguous[B, T](
        obs_out, act_out, rew_out, dne_out, "gpu-batch"
    )
    print("  OK  count=", buf.count(), " windows contiguous after record_batch")


def test_gpu_fst(ctx: DeviceContext) raises:
    """`fst` (is_first) ring + `record_terminal` + `sample_batch_fst_dev`.
    RNG-independent: slot i tagged i in obs, `done` every 5th frame → the
    record/pending_first logic gives `fst[i] = 1 iff i%5==0`; the sampled
    `fst_out[b,k]` must equal that rule keyed on the obs frame tag."""
    print("--- GPU fst / record_terminal / sample_batch_fst_dev ---")
    var buf = GPUSequenceReplay[OBS, ACT, CAP].make["gpu"](ctx=ctx)
    var o = alloc[Scalar[DT]](OBS)
    var a = alloc[Scalar[DT]](ACT)
    for i in range(CAP):
        for d in range(OBS):
            o[d] = Scalar[DT](i)
        for j in range(ACT):
            a[j] = Scalar[DT](i)
        var dn = Scalar[DT](1.0) if i % 5 == 4 else Scalar[DT](0.0)
        buf.record(o, a, Scalar[DT](i), dn)
    ctx.synchronize()

    # (a) download the fst ring → expect 1 iff i%5==0.
    var hf = alloc[Scalar[DT]](CAP)
    ctx.enqueue_copy(hf, buf.fst)
    ctx.synchronize()
    for i in range(CAP):
        var exp = Scalar[DT](1.0) if i % 5 == 0 else Scalar[DT](0.0)
        assert_equal(hf[i], exp, "fst ring slot " + String(i))

    # (b) sample_batch_fst_dev → fst_out aligns with obs frame tags.
    var d_obs = ctx.enqueue_create_buffer[DT](B * (T + 1) * OBS)
    var d_act = ctx.enqueue_create_buffer[DT](B * T * ACT)
    var d_rew = ctx.enqueue_create_buffer[DT](B * T)
    var d_dne = ctx.enqueue_create_buffer[DT](B * T)
    var d_fst = ctx.enqueue_create_buffer[DT](B * (T + 1))
    buf.sample_batch_fst_dev[B, T](ctx, d_obs, d_act, d_rew, d_dne, d_fst)
    var obs_out = alloc[Scalar[DT]](B * (T + 1) * OBS)
    var fst_out = alloc[Scalar[DT]](B * (T + 1))
    ctx.enqueue_copy(obs_out, d_obs)
    ctx.enqueue_copy(fst_out, d_fst)
    ctx.synchronize()
    for b in range(B):
        for k in range(T + 1):
            var tag = Int(obs_out[b * (T + 1) * OBS + k * OBS])
            var exp = Scalar[DT](1.0) if tag % 5 == 0 else Scalar[DT](0.0)
            assert_equal(
                fst_out[b * (T + 1) + k], exp, "sampled fst vs obs tag"
            )

    # (c) record_terminal leaves pending_first armed → terminal frame fst=0,
    #     the FOLLOWING reset frame fst=1. Fresh buf: record(done=1) →
    #     record_terminal → record(done=0) ⇒ fst[0,1,2] = [1, 0, 1].
    var buf2 = GPUSequenceReplay[OBS, ACT, CAP].make["gpu"](ctx=ctx)
    buf2.record(o, a, Scalar[DT](0.0), Scalar[DT](1.0))  # slot0: fst=1 (init)
    buf2.record_terminal(o)                               # slot1: fst=0
    buf2.record(o, a, Scalar[DT](0.0), Scalar[DT](0.0))  # slot2: fst=1 (armed)
    ctx.synchronize()
    var hf2 = alloc[Scalar[DT]](CAP)
    ctx.enqueue_copy(hf2, buf2.fst)
    ctx.synchronize()
    assert_equal(hf2[0], Scalar[DT](1.0), "terminal seq fst[0]")
    assert_equal(hf2[1], Scalar[DT](0.0), "terminal frame fst[1]")
    assert_equal(hf2[2], Scalar[DT](1.0), "post-terminal reset fst[2]")
    print("  OK  fst ring + sampled fst + record_terminal")


def main() raises:
    print("==============================================================")
    print("SequenceReplayBuffer trait — CPU + GPU sibling")
    print("==============================================================")
    test_cpu()
    var ctx = DeviceContext()
    test_gpu(ctx)
    test_gpu_record_batch(ctx)
    test_gpu_fst(ctx)
    print("==============================================================")
    print("ALL PASSED")
    print("==============================================================")
