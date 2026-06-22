"""REDQ-OFE AGENT convergence on Pendulum-v1 — storage REDQOFETrainer gate.

Builds the REDQ-OFE agent via the `REDQOFE6["cpu", 3, 1, BATCH, CAP]()` cheap
preset (SAC-shape REDQ knobs N=2/N_MIN=2/UTD=1/POLICY_DELAY=1 + 6-block OFE
DenseNet branches; storage trainer + drivers), trains single-env on the real
Pendulum env through `run_offpolicy_train` (via `agent.train_single`), then
greedy-evals. Asserts the trained policy clearly beats random and improves over
its own untrained baseline. Prints + checks the DIAG metrics (incl. aux_loss).

REDQ-OFE specifics: a DenseNet feature extractor computes φ(s) / φ(s,a); the
ensemble actor/critic operate on those features; an auxiliary next-state
prediction loss trains the feature nets.

Run: pixi run mojo run -I . tests/deep_agents/test_storage_redq_ofe_cpu_pendulum.mojo
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.redq_ofe.config import REDQOFE6
from mojo_rl.envs.pendulum.pendulum_v1 import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime H = 128
comptime BATCH = 128
comptime CAP = 100_000
comptime PER_UNIT = 16   # cheap OFE width for the smoke (φ adds 6·16 = 96 feats)
comptime ASCALE = Scalar[DT](2.0)   # Pendulum torque in [-2, 2]


def main() raises:
    seed(42)
    print("=" * 60)
    print("REDQ-OFE AGENT Pendulum-v1 convergence (storage trainer, CPU)")
    print("=" * 60)

    var env = PendulumEnv[DT]()

    # REDQOFE6: N=2/N_MIN=2/UTD=1/POLICY_DELAY=1 + 6-block OFE branches.
    # action_scale must match the env torque range (±2); critic_lr=1e-3 default.
    var agent = REDQOFE6["cpu", OBS, ACT, BATCH, CAP, H, PER_UNIT](
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        ofe_lr=Scalar[DT](3e-4),
        alpha_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=ASCALE,
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-Float64(ACT)),
        learning_starts=1_000,
    )

    var rand_eval = agent.eval(env, num_episodes=5, max_steps_per_episode=200)
    print("eval @0 (untrained baseline):", rand_eval)

    _ = agent.train_single(
        env, total_timesteps=18_000, print_every=3_000, diag_every=3_000
    )

    var metrics = agent.flush_metrics()
    print("-" * 60)
    print("DIAG  actor_loss   :", metrics.actor_loss)
    print("DIAG  critic_loss  :", metrics.critic_loss)
    print("DIAG  alpha        :", metrics.alpha)
    print("DIAG  log_prob_mean:", metrics.log_prob_mean)
    print("DIAG  aux_loss     :", metrics.aux_loss)
    print("DIAG  n_updates    :", metrics.n_updates)
    print("DIAG  n_actor_upd  :", metrics.n_actor_updates)
    print("DIAG  mean_reward  : (see per-print rows above)")
    print("-" * 60)

    var final_eval = agent.eval(
        env, num_episodes=10, max_steps_per_episode=200
    )
    print("FINAL greedy eval(10):", final_eval)

    assert_true(
        final_eval > Scalar[DT](-700.0),
        "REDQ-OFE agent learns Pendulum (eval return > -700)",
    )
    assert_true(
        final_eval > rand_eval + Scalar[DT](300.0),
        "REDQ-OFE agent clearly beats its untrained baseline (+300)",
    )
    assert_true(
        metrics.critic_loss > Scalar[DT](0.0),
        "critic_loss diagnostic populated (> 0)",
    )
    assert_true(
        metrics.aux_loss > Scalar[DT](0.0),
        "aux_loss diagnostic populated (> 0)",
    )
    assert_true(
        metrics.n_updates > 0,
        "n_updates advanced",
    )
    print("REDQ-OFE AGENT PENDULUM CONVERGENCE OK")
