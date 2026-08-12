"""TD-MPC2 MULTI-TASK checkpoint round-trip on GPU — the task-embedding table.

`tests/deep_agents/test_tdmpc2_checkpoint.mojo` already covers save/load, but
it cannot cover this on two counts: it runs on CPU (where the H2D upload is a
no-op) and it uses the SINGLE-task agent, which has no embedding table at all.
The multi-task table is the one piece of agent state that is NOT a `Param` and
so rides outside `for_each_param` — `save_body` / `load_body` hand-roll it.

The invariant under test is the last line of `TaskEmbedding.load_body`:

    self.upload_from_host()     # ← delete this and everything still "works"

`load_body` parses into the HOST slabs, but every consumer of the table —
`gather` in `_mt_gather_emb` / `_mt_gather_row`, hence the encoder, the policy
and the MPPI callback — reads the DEVICE slab. Drop that upload and the
restored rows never reach the GPU: the agent silently keeps its RANDOM init, a
resumed multi-task run relearns the embedding from scratch, and a checkpoint
eval scores a model it never trained. Nothing raises, no number is NaN, and
every net around it loads perfectly. The only symptom is a per-task result
quietly worse than the run that produced it.

Verified to discriminate: with that one line commented out, both gates below
fail (row 0 reads back 0.1456 instead of 1.0; task-0 action drifts by 0.20).

Two gates, deliberately at different levels:

  1. The device slab itself holds the checkpointed rows after `load_state`.
     Read back by zeroing the host copy and pulling D2H, so the assert cannot
     be satisfied by the host slab it is checking against.
  2. End to end: a freshly built agent that loads the checkpoint reproduces the
     saved agent's greedy action FOR EACH TASK. This is the one that would have
     caught the bug through its actual consequence.

Run: `pixi run -e apple mojo run -I . tests/deep_agents/test_tdmpc2_multitask_checkpoint_gpu.mojo`
"""

from std.math import abs
from std.random import seed
from std.testing import assert_true, assert_almost_equal, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.agent_mt import TDMPC2MultiTaskAgent

comptime MAX_OBS = 3
comptime ENC = 32
comptime MAX_ACT = 1
comptime LATENT = 32
comptime MLP = 32
comptime BINS = 21
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 8
comptime H = 3
comptime CAP = 512
comptime NUM_TASKS = 3
comptime TASK_EMB = 8
comptime PATH = "tdmpc2_mt_ckpt_test.ckpt"

comptime Ag = TDMPC2MultiTaskAgent[
    "gpu", MAX_OBS, ENC, MAX_ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H,
    CAP, NUM_TASKS, TASK_EMB,
]


def _probe() -> List[Scalar[DT]]:
    var obs = List[Scalar[DT]](length=MAX_OBS, fill=Scalar[DT](0))
    obs[0] = Scalar[DT](0.3)
    obs[1] = Scalar[DT](-0.5)
    obs[2] = Scalar[DT](1.2)
    return obs^


def _stamp_rows(mut ag: Ag, ctx: DeviceContext) raises:
    """Give each task a distinct, recognisable row and push it to the device.

    Hand-set rather than trained: the point is a value we can assert on, and
    a row that is FAR from any random init so a dropped restore cannot be
    mistaken for float noise.
    """
    for t in range(NUM_TASKS):
        for e in range(TASK_EMB):
            ag.task_emb.param.data[t * TASK_EMB + e] = Scalar[DT](
                1.0 + Float64(t)
            )
    ag.task_emb.param.upload(ctx)


def _greedy(mut ag: Ag, task: Int, obs: List[Scalar[DT]]) raises -> Scalar[DT]:
    var act = List[Scalar[DT]](length=MAX_ACT, fill=Scalar[DT](0))
    ag.set_task(task)
    ag.select_greedy_action(obs, act)
    return act[0]


def test_load_state_uploads_task_embedding_to_device() raises:
    seed(1)
    var ctx = DeviceContext()
    var a = Ag.make(action_scale=Scalar[DT](2.0), ctx=ctx)
    _stamp_rows(a, ctx)
    a.save_state(PATH)

    seed(2)
    var b = Ag.make(action_scale=Scalar[DT](2.0), ctx=ctx)
    b.load_state(PATH)

    # ⚠ Poison the HOST slab before reading back, so what we assert on can only
    # have come from the device. Without this the assert is satisfied by the
    # host copy `load_body` just wrote and the upload is never exercised.
    for i in range(NUM_TASKS * TASK_EMB):
        b.task_emb.param.data[i] = Scalar[DT](-999.0)
    b.task_emb.sync_to_host()

    for t in range(NUM_TASKS):
        var want = Scalar[DT](1.0 + Float64(t))
        for e in range(TASK_EMB):
            assert_almost_equal(
                b.task_emb.param.data[t * TASK_EMB + e], want, atol=1e-5,
                msg=(
                    "task-embedding row was restored into the host slab but"
                    " never uploaded — every device-side consumer (encoder,"
                    " policy, MPPI callback) still sees the random init"
                ),
            )
        print("  row", t, "→", b.task_emb.param.data[t * TASK_EMB])


def test_loaded_agent_reproduces_per_task_actions() raises:
    """The consequence gate: same weights + same task ⇒ same action."""
    seed(1)
    var ctx = DeviceContext()
    var obs = _probe()

    var a = Ag.make(action_scale=Scalar[DT](2.0), ctx=ctx)
    _stamp_rows(a, ctx)
    var a0 = _greedy(a, 0, obs)
    var a1 = _greedy(a, 1, obs)
    var a2 = _greedy(a, 2, obs)
    a.save_state(PATH)

    # A different seed → a different init, so an agent that failed to load
    # anything would disagree. Asserted below rather than assumed.
    seed(2)
    var b = Ag.make(action_scale=Scalar[DT](2.0), ctx=ctx)
    var pre = _greedy(b, 0, obs)
    assert_true(
        abs(Float64(pre - a0)) > 1e-4,
        "fresh agent should differ before load — otherwise this test is"
        " vacuous and would pass with load_state removed entirely",
    )

    b.load_state(PATH)
    var b0 = _greedy(b, 0, obs)
    var b1 = _greedy(b, 1, obs)
    var b2 = _greedy(b, 2, obs)

    print("  task 0  A=", a0, " B=", b0)
    print("  task 1  A=", a1, " B=", b1)
    print("  task 2  A=", a2, " B=", b2)
    assert_almost_equal(b0, a0, atol=1e-5, msg="task 0 action must match")
    assert_almost_equal(b1, a1, atol=1e-5, msg="task 1 action must match")
    assert_almost_equal(b2, a2, atol=1e-5, msg="task 2 action must match")

    # The tasks must also still differ from EACH OTHER after the round-trip.
    # A restore that collapsed the table to one row would satisfy every assert
    # above if the saved rows happened to be close; this rules that out.
    var spread = abs(Float64(b0 - b1)) + abs(Float64(b1 - b2))
    print("  |b0-b1| + |b1-b2| =", spread)
    assert_true(
        spread > 1e-4,
        "all three tasks act identically after load — the table collapsed",
    )


def main() raises:
    print("=" * 70)
    print("TD-MPC2 multi-task checkpoint round-trip (GPU)")
    print("=" * 70)
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("=" * 70)
    print("MULTI-TASK CHECKPOINT ROUND-TRIP PASSED")
    print("=" * 70)
