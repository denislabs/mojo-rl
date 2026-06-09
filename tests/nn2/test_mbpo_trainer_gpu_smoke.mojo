"""MBPOTrainer GPU smoke (Phase 4.3d/e).

Runs the GPU MBPO trainer end-to-end against the CPU Pendulum env (cpu env
+ gpu train) for a short horizon, exercising the full device path:
  * GPU dynamics-ensemble training (bootstrap sample → concat → NLL step),
  * GPU synthetic rollout (real-buffer start draw → batched actor+rsample →
    elite dynamics forward → posterior box-muller sampling → device batch
    store into the GPU synth replay),
  * GPU dual-sample mixed real+synth minibatch,
  * the SAC sub-update reusing the already-GPU SAC blocks.

Asserts: training updates ran, the on-device SAC actor/critic/alpha +
host-accumulated dynamics NLL drain to finite values, mean_return finite.
NVIDIA numeric convergence is a separate HW-gated step.

Run (Apple): pixi run -e apple mojo run -I . \
    tests/nn2/test_mbpo_trainer_gpu_smoke.mojo
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.mbpo.trainer import MBPOTrainer
from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime HIDDEN = 64
comptime BATCH = 128
comptime REPLAY_CAP = 20_000
comptime SYNTH_CAP = 40_000
comptime N_ENS = 3
comptime N_ELITES = 2
comptime REAL_PCT = 10
comptime WARMUP = 256
comptime TOTAL = 1_500

comptime ActorNet = StochasticActor[
    OBS, ACT,
    Linear[OBS, HIDDEN], ReLU[HIDDEN], Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, HIDDEN], ReLU[HIDDEN], Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN], Linear[HIDDEN, 1],
]
# Dynamics net: IN = OBS+ACT, OUT = 2*(1+OBS) = mean+logvar of [reward, Δobs].
comptime DynNet = Sequential[
    Linear[OBS + ACT, HIDDEN], ReLU[HIDDEN], Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN], Linear[HIDDEN, 2 * (1 + OBS)],
]
comptime Trainer = MBPOTrainer[
    "gpu", ActorNet, CriticNet, DynNet,
    OBS, ACT, BATCH, REPLAY_CAP, SYNTH_CAP, N_ENS, N_ELITES, REAL_PCT,
]


def _finite(v: Float64, tag: String) raises:
    assert_true(not isnan(v), tag + ": NaN")
    assert_true(not isinf(v), tag + ": Inf")


def test_mbpo_gpu_smoke() raises:
    print("--- MBPO GPU smoke ---")
    seed(42)
    var ctx = DeviceContext()
    var trainer = Trainer.make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](3e-4),
        alpha_lr=Scalar[DT](3e-4),
        model_lr=Scalar[DT](1e-3),
        action_scale=Scalar[DT](2.0),
        learning_starts=WARMUP,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
        model_train_freq=100,
        dyn_epochs_per_round=2,
        rollout_length=1,
        num_rollouts_per_step=200,
        sac_updates_per_step=5,
        dyn_batch_size=128,
    )

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var env = PendulumEnv[DT]()
    _ = env.reset()
    var obs_self = env.get_obs_list()

    var step: Int = 0
    while step < TOTAL:
        for d in range(OBS):
            obs[d] = obs_self[d]
        trainer.select_action(obs, action, step)
        var step_res = env.step_continuous(action[0])
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(OBS):
            next_obs[d] = nxt[d]
        trainer.record(
            obs, action, reward, next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        if done:
            trainer.end_episode()
            _ = env.reset()
            obs_self = env.get_obs_list()
        else:
            obs_self = nxt.copy()
        step += 1
        _ = trainer.train_step(step)

    var m = trainer.flush_metrics()
    var al = m.actor_loss.to_f64()
    var cl = m.critic_loss.to_f64()
    var alpha = m.alpha.to_f64()
    var dl = m.dyn_loss.to_f64()
    var nup = m.n_updates.to_f64()
    var mq = m.mean_q.to_f64()
    var mr = m.mean_reward.to_f64()
    var mret = Float64(trainer.mean_return())
    print("  actor_loss  =", al)
    print("  critic_loss =", cl)
    print("  alpha       =", alpha)
    print("  dyn_loss    =", dl)
    print("  n_updates   =", nup)
    print("  mean_q      =", mq)
    print("  mean_reward =", mr)
    print("  mean_ret(10)=", mret)

    _finite(al, "actor_loss")
    _finite(cl, "critic_loss")
    _finite(alpha, "alpha")
    _finite(dl, "dyn_loss")
    _finite(mq, "mean_q")
    _finite(mr, "mean_reward")
    _finite(mret, "mean_return")
    assert_true(nup > 0.0, "no SAC sub-updates ran")
    assert_true(cl >= 0.0, "twin-critic MSE loss should be >= 0")
    assert_true(alpha > 0.0, "alpha should be positive (= exp(log_alpha))")
    # Device-side diag reductions (the fix): before wiring, both read a
    # hard 0.0 on GPU. mean_reward over a Pendulum minibatch is strictly
    # negative (the reward is a cost), and mean_q is essentially never
    # exactly 0 once the critic has trained — so a non-zero pair proves the
    # device accumulators are folding the mb_q / mb_r buffers in.
    assert_true(mr < 0.0, "mean_reward should be < 0 on Pendulum (device diag)")
    assert_true(mq != 0.0, "mean_q should be non-zero on GPU (device diag)")
    print("PASS")


def main() raises:
    print("=" * 70)
    print("MBPO GPU smoke (Phase 4.3)")
    print("=" * 70)
    test_mbpo_gpu_smoke()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
