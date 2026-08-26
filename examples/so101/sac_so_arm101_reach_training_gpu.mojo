"""SAC training on SO-ARM101 reach (GPU, multi-env).

The sim half of `ROADMAP_2026_08.md` §5.4's vertical: train here, evaluate on
CPU with `sac_so_arm101_reach_eval_cpu.mojo`, then run the same checkpoint on
the physical follower through `mojo_rl/robot/so101/`.

Shaped exactly like `examples/half_cheetah/sac_half_cheetah_training_gpu.mojo`
— same `SACAgent["gpu", ...]` facade, same batched off-policy driver, same
one-file `nn-ckpt v2` output — so the two are diffable and a change to the
driver shows up in both.

⚠⚠ **THE ACTION SPACE IS JOINT ANGLES IN RADIANS, NOT [-1, 1].** Both SO-ARM
models drive `<position>` servos whose `ctrlrange` IS the joint range, so an
action is a commanded angle and `action_scale` has to span that range rather
than the usual unit box. `ACTION_SCALE = 2.0` covers every joint's limit
except `wrist_roll`'s +2.84 rad, and the env clamps to `ctrlrange` regardless.

⚠ The gripper's range is ASYMMETRIC (-0.17 .. +1.75 rad) while `tanh · scale`
is symmetric, so roughly half the gripper's action range is unreachable and
the rest is clamped. That is harmless for *reach* — the task is scored on the
moving jaw's position and never on the jaw opening — and it is NOT harmless
for a future grasp task, which will want a per-joint action scale.

Task (`SoArmReachConfig`): a mocap target drawn per episode from an azimuth
cone × elevation band × radial shell (0.15–0.30 m); reward is dm_control's
shaped `tolerance` on jaw-to-target distance with a 2 cm radius; no early
termination.

  * 21D observation — qpos(6) + qvel(6) + ee(3) + target(3) + ee_to_target(3)
  * 6D continuous action — joint angle targets, radians

Run:
    pixi run -e apple  mojo run -I . examples/so101/sac_so_arm101_reach_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/so101/sac_so_arm101_reach_training_gpu.mojo
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model
from mojo_rl.envs.robots.so_arm101 import SoArm101ReachConfig


comptime N_ENVS = 32

comptime BatchedEnvT = Phyics3dBatchedEnv[
    SoArm101Model, SoArm101ReachConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime OBS_DIM = BatchedEnvT.OBS_DIM  # 21
comptime ACT_DIM = 6
comptime HIDDEN = 256

comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000

comptime NUM_STEPS = 400_000
comptime WARMUP_STEPS = 10_000
comptime PRINT_EVERY = 25_000
comptime DIAG_EVERY = 1000
comptime CHECKPOINT_EVERY = 25_000
comptime CHECKPOINT_PATH = "sac_so_arm101_reach.ckpt"

# See the module docstring: radians, not a unit box.
comptime ACTION_SCALE = Scalar[DT](2.0)

comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    LinearReLU[OBS_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
comptime CriticNet = Sequential[
    LinearReLU[OBS_DIM + ACT_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC — SO-ARM101 reach, GPU (multi-env)")
    print("=" * 70)
    print("  OBS_DIM          =", OBS_DIM)
    print("  ACT_DIM          =", ACT_DIM, "(joint angles in RADIANS)")
    print("  HIDDEN           =", HIDDEN)
    print("  N_ENVS           =", N_ENVS)
    print("  NUM_STEPS        =", NUM_STEPS)
    print("  action_scale     =", ACTION_SCALE)
    print("=" * 70)

    with DeviceContext() as ctx:
        var env_vars = load_dotenv()
        var logger = RemoteLogger(
            server_url=env_vars.get("RL_MONITOR_URL", ""),
            run_name="SAC SO-ARM101 reach (GPU)",
            buffer_size=64,
            api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
        )
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", "SoArm101Reach")
        logger.set_config("target", "gpu")
        logger.set_config("n_envs", String(N_ENVS))
        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        var agent = SACAgent[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet,
            CriticNet,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            action_scale=ACTION_SCALE,
            init_alpha=0.2,
            target_entropy=-Scalar[DT](ACT_DIM),
            learning_starts=WARMUP_STEPS,
            window_size=100,
            # ⚠⚠ NOT THE DEFAULT. SAC seeds its return window with
            # `initial_episode_fill = -1250.0`, a HalfCheetah-flavoured
            # pessimistic value. This task's reward is a shaped `tolerance` in
            # [0, 1], so a return is in [0, 500] and CANNOT be negative — with
            # the default, every reading before the window fills is a blend of
            # real returns and sentinels, and it reads as a broken reward.
            #
            # Measured: a 24k-step smoke run printed `mean_ret = -793.2661`
            # at 32 episodes. Solving (32R + 68*(-1250))/100 for R gives
            # R = 177.3, reported as "still exploring". The template sets this
            # to 0.0 for the same reason; dropping the line is what produced
            # the false reading.
            #
            # ⚠ AND 177 IS NOT YET "REACHING TARGETS", which an earlier version
            # of this comment claimed. The untrained baseline is 46 (see the
            # verdict bands below), so 177 is ~4x the floor after 24k steps —
            # real movement, not a solved task.
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)

        print("Starting GPU training...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        _ = agent.train[
            BatchedEnvT,
            N_ENVS=N_ENVS,
            USE_TRAIN_CUDA_GRAPH=True,
            USE_ENV_CUDA_GRAPH=False,
            L=RemoteLogger,
        ](
            env,
            NUM_STEPS,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=logger_ptr,
            diag_every=DIAG_EVERY,
            episode_sync_every=32,
            checkpoint_every=CHECKPOINT_EVERY,
            checkpoint_path=CHECKPOINT_PATH,
        )
        var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
        logger.close()
        _ = logger

        print("-" * 70)
        print("Training complete")
        print("  total env_steps           =", NUM_STEPS)
        print("  elapsed                   =", elapsed_s, "s")
        print("  mean ep return (last 100) =", agent.mean_return())
        print("  episodes completed        =", agent.ep_count())
        print("=" * 70)

        # ⚠ Reward is a shaped `tolerance` in [0, 1] per step over
        # `MAX_STEPS = 500` control steps, so the ceiling is 500 and a policy
        # that reaches the target quickly and HOLDS it is what scores.
        #
        # ⚠⚠ THE FLOOR IS NOT ZERO, AND THAT IS THE WHOLE POINT OF THESE
        # BANDS. An UNTRAINED actor measured **45.9** over 11 episodes on this
        # env (greedy, 2026-08-26; per-episode 3.5 .. 123.8) — the 0.25 m
        # reward margin covers most of a 0.15-0.30 m target shell, so flailing
        # near the middle of the workspace pays. Anything under ~90 has not
        # been shown to have learned anything. Same bands as
        # `sac_so_arm101_reach_eval_cpu.mojo`; keep them in step.
        var m = Float64(agent.mean_return())
        if m > 400.0:
            print("EXCELLENT — reaches and holds (mean > 400 / 500).")
        elif m > 200.0:
            print("STRONG — reaches most targets (mean > 200).")
        elif m > 90.0:
            print("PROGRESS — measurably above the untrained baseline (~46).")
        else:
            print("NO BETTER THAN AN UNTRAINED NET (baseline mean ~46).")
        print("=" * 70)
