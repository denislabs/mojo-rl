"""TD-MPC2 MULTI-TASK `select_action_mpc` smoke (GPU / Apple Metal).

Gates the task-conditioned MPPI path: `TDMPC2RolloutCallbackGPUMT` +
`MPPIGPUBatched` on the multi-task agent. The multi-task planner did not exist
until now — `agent_mt.mojo` shipped MPC-off with the planner "deferred".

Two gates, and the second is the one that matters:

  1. A plan produces a finite action inside [-scale, scale].
  2. Two DIFFERENT tasks produce DIFFERENT actions from the SAME observation.

(2) is what proves the task embedding actually reaches the planner. It is easy
to write a rollout callback that concatenates the embedding into the wrong
slice, or forgets to broadcast it at all: the plan still runs, the action is
still finite and in range, and every task silently plans identically. Gate (1)
alone cannot see that.

The embedding rows are randomly initialised (`1/sqrt(TASK_EMB)` scale), so two
task ids genuinely differ at init — no training needed for this gate. To make
the signal unmistakable the two rows are additionally pushed apart by hand.

Run: `pixi run -e apple mojo run -I . tests/deep_agents/test_tdmpc2_multitask_mpc_gpu.mojo`
"""

from std.math import isfinite, abs
from std.random import seed
from std.testing import assert_true, TestSuite
from mojo_rl.core.logger import NoOpLogger
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.agent_mt import TDMPC2MultiTaskAgent
from mojo_rl.deep_agents.training.batched_env import BatchedGpuEnv
from mojo_rl.envs.pendulum import PendulumV2

comptime MAX_OBS = 3
comptime ENC = 32
comptime MAX_ACT = 1   # PendulumV2's real action dim — see the batched gate
comptime LATENT = 32
comptime MLP = 32
comptime BINS = 21
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 8
comptime H = 3
comptime CAP = 2048
comptime NUM_TASKS = 2
comptime TASK_EMB = 8
# small planning budget — this is a wiring gate, not a quality one
comptime NS = 16
comptime NPT = 4
comptime NE = 8
comptime NI = 2

comptime Ag = TDMPC2MultiTaskAgent[
    "gpu", MAX_OBS, ENC, MAX_ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H,
    CAP, NUM_TASKS, TASK_EMB, 0.0, NS, NPT, NE, NI,
]


def _plan_for(mut ag: Ag, task: Int, mut obs: List[Scalar[DT]]) raises -> List[
    Scalar[DT]
]:
    var act = List[Scalar[DT]](length=MAX_ACT, fill=Scalar[DT](0))
    ag.set_task(task)
    ag.mpc_start_episode()
    ag.select_action_mpc(obs, act, explore=False)
    return act^


def test_mt_mpc_action_is_finite_and_bounded() raises:
    seed(0)
    var ctx = DeviceContext()
    var ag = Ag.make(
        lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=0,
        ctx=ctx,
    )
    var obs = List[Scalar[DT]](length=MAX_OBS, fill=Scalar[DT](0))
    obs[0] = Scalar[DT](0.3)
    obs[1] = Scalar[DT](-0.5)
    obs[2] = Scalar[DT](1.2)

    var a = _plan_for(ag, 0, obs)
    for j in range(MAX_ACT):
        assert_true(isfinite(Float64(a[j])), "MPC action must be finite")
        assert_true(
            a[j] >= -2.0001 and a[j] <= 2.0001,
            "MPC action must lie in [-action_scale, action_scale]",
        )
    print("  task 0 MPC action =", a[0])


def test_different_tasks_plan_differently() raises:
    """The gate that catches a mis-spliced or unbroadcast task embedding."""
    seed(0)
    var ctx = DeviceContext()
    var ag = Ag.make(
        lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=0,
        ctx=ctx,
    )
    # Push the two rows apart so the difference cannot be mistaken for noise.
    for e in range(TASK_EMB):
        ag.task_emb.param.data[e] = Scalar[DT](1.0)              # task 0
        ag.task_emb.param.data[TASK_EMB + e] = Scalar[DT](-1.0)  # task 1
    # Push the host edit to the device — `gather` reads the device slab.
    ag.task_emb.param.upload(ctx)

    var obs = List[Scalar[DT]](length=MAX_OBS, fill=Scalar[DT](0))
    obs[0] = Scalar[DT](0.3)
    obs[1] = Scalar[DT](-0.5)
    obs[2] = Scalar[DT](1.2)

    var a0 = _plan_for(ag, 0, obs)
    var a1 = _plan_for(ag, 1, obs)

    var diff: Scalar[DT] = 0.0
    for j in range(MAX_ACT):
        diff += abs(a0[j] - a1[j])
    print("  task 0 →", a0[0])
    print("  task 1 →", a1[0])
    print("  |a0 - a1| =", diff)
    assert_true(
        diff > Scalar[DT](1e-4),
        "the two tasks planned IDENTICALLY from the same observation — the"
        " task embedding is not reaching the planner (wrong concat slice, or"
        " never broadcast over the planning candidates)",
    )


def test_train_batched_mt_with_mpc_runs() raises:
    """Instantiate the BATCHED multi-task MPC path end to end.

    Separate from the single-obs `select_action_mpc` gate above because it is
    a DIFFERENT instantiation: an N_ENVS-wide planner and an
    N_ENVS x (NUM_SAMPLES + NUM_PI_TRAJS) callback, hoisted out of the loop.
    An uninstantiated generic is uncompiled code, so without this test the
    batched planner could be broken and every other test would still pass.
    """
    seed(4)
    var ctx = DeviceContext()
    # ⚠ MAX_ACT must be PendulumV2's REAL action dim (1). Declaring 2 here
    # wrapped a 1-action env as 2-action: every dim assert passed (they all
    # compare the wrapper's declared value) and the env kernel strided the
    # action slab wrong, surfacing as a NaN world-model loss 30 minutes later.
    # `BatchedGpuEnv.__init__` now rejects the mismatch at compile time.
    comptime NE_ENVS = 2
    comptime GEnv = BatchedGpuEnv[PendulumV2[DT], NE_ENVS, MAX_OBS, MAX_ACT]
    var env = GEnv(ctx)
    var ag = Ag.make(
        lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=8,
        ctx=ctx,
    )
    var best = ag.train_batched_mt[GEnv, NE_ENVS, NoOpLogger, True](
        env,
        1,                      # task 1 — exercises a non-zero embedding row
        64,
        rng_seed=UInt64(7),
        updates_per_step=1,
        print_every=0,
        verbose=False,
    )
    assert_true(isfinite(Float64(best)) or best < Scalar[DT](-1.0e29),
                "best-eval sentinel must be finite or the initial sentinel")
    assert_true(ag.replay.count() == 64, "every transition recorded")
    assert_true(
        isfinite(Float64(ag.last_wm_loss())), "WM loss must be finite"
    )
    print("  batched MT + MPC ran; wm =", ag.last_wm_loss(),
          " replay =", ag.replay.count())


def main() raises:
    print("=" * 70)
    print("TD-MPC2 multi-task select_action_mpc smoke (GPU)")
    print("=" * 70)
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("=" * 70)
    print("MULTI-TASK MPC SMOKE PASSED")
    print("=" * 70)
