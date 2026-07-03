"""SequenceReplay online-queue unit test (reference replay `online: True`).

Checks the CPU backend's online mixing semantics (the GPU backend shares the
identical host-side queue logic; only the starts-overwrite kernel differs):
  1. every T-th appended frame enqueues the freshest T-window,
  2. `sample_batch_fst` serves queued windows into the FIRST batch rows
     (exactly once — the queue drains),
  3. with online disabled (default) the queue stays empty (byte-identical
     legacy behavior),
  4. window content: row 0's obs are exactly the last T+1 recorded frames.

Run: pixi run mojo run -I . tests/deep_agents/test_sequence_replay_online.mojo
"""

from std.random import seed
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.data.sequence_replay import SequenceReplay

comptime OBS = 2
comptime ACT = 1
comptime CAP = 64
comptime T = 4
comptime B = 2


def main() raises:
    seed(7)
    print("SequenceReplay online-queue test")

    # ── 3. default off: no queue activity ─────────────────────────────
    var off = SequenceReplay[OBS, ACT, CAP].new()
    var s_buf = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    var a_buf = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    for i in range(20):
        s_buf[0] = Scalar[DT](i)
        s_buf[1] = Scalar[DT](i)
        off.record(
            s_buf.unsafe_ptr(), a_buf.unsafe_ptr(),
            Scalar[DT](0.0), Scalar[DT](0.0),
        )
    assert_equal(len(off.online_q), 0, "online off => empty queue")

    # ── 1. enqueue cadence: every T-th append once size >= T+1 ────────
    var rep = SequenceReplay[OBS, ACT, CAP].new()
    rep.set_online(T)
    for i in range(13):  # appends 1..13
        s_buf[0] = Scalar[DT](i)
        s_buf[1] = Scalar[DT](100 + i)
        rep.record(
            s_buf.unsafe_ptr(), a_buf.unsafe_ptr(),
            Scalar[DT](i), Scalar[DT](0.0),
        )
    # first enqueue at append 5 (tick>=4 and size>=5), then 9, 13 → 3 queued
    assert_equal(len(rep.online_q), 3, "3 fresh windows queued after 13 appends")

    # ── 2 + 4. sample serves queued windows first, exactly once ───────
    var obs_out = List[Scalar[DT]](length=B * (T + 1) * OBS, fill=Scalar[DT](0))
    var act_out = List[Scalar[DT]](length=B * T * ACT, fill=Scalar[DT](0))
    var rew_out = List[Scalar[DT]](length=B * T, fill=Scalar[DT](0))
    var dne_out = List[Scalar[DT]](length=B * T, fill=Scalar[DT](0))
    var fst_out = List[Scalar[DT]](length=B * (T + 1), fill=Scalar[DT](0))
    rep.sample_batch_fst[B, T](
        obs_out.unsafe_ptr(), act_out.unsafe_ptr(), rew_out.unsafe_ptr(),
        dne_out.unsafe_ptr(), fst_out.unsafe_ptr(),
    )
    # Queue order is FIFO: row 0 = window enqueued at append 5 (frames 0..4),
    # row 1 = window enqueued at append 9 (frames 4..8).
    for k in range(T + 1):
        assert_equal(
            obs_out[k * OBS], Scalar[DT](k),
            "row0 frame " + String(k) + " = recorded frame " + String(k),
        )
        assert_equal(
            obs_out[(T + 1) * OBS + k * OBS], Scalar[DT](4 + k),
            "row1 frame " + String(k) + " = recorded frame " + String(4 + k),
        )
    assert_equal(len(rep.online_q), 1, "two of three queued windows consumed")

    # Second sample drains the last queued window into row 0; the queue is
    # then empty and further samples are pure uniform.
    rep.sample_batch_fst[B, T](
        obs_out.unsafe_ptr(), act_out.unsafe_ptr(), rew_out.unsafe_ptr(),
        dne_out.unsafe_ptr(), fst_out.unsafe_ptr(),
    )
    for k in range(T + 1):
        assert_equal(
            obs_out[k * OBS], Scalar[DT](8 + k),
            "row0 frame " + String(k) + " = recorded frame " + String(8 + k),
        )
    assert_equal(len(rep.online_q), 0, "queue fully drained")

    print("SEQUENCE REPLAY ONLINE OK")
