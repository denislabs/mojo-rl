"""Evaluate + render a trained SO-ARM101 reach checkpoint (CPU, single env).

The middle rung of `ROADMAP_2026_08.md` §5.4's vertical: train GPU-batched with
`sac_so_arm101_reach_training_gpu.mojo`, watch it here, then run the SAME
checkpoint on the physical follower with `deploy_reach_real.mojo`.

Counterpart of `examples/half_cheetah/sac_half_cheetah_nn_eval_cpu.mojo`, and
built through the same `SAC[...]` preset so the checkpoint's parameter layout
matches the trainer's exactly.

⚠⚠ `action_scale` MUST match training. The greedy action is
`a = tanh(mu) * action_scale`, and for these arms `a` is a JOINT ANGLE IN
RADIANS, not a normalised torque — so a mismatched scale does not merely
weaken the policy, it commands a different pose. `2.0` is the trainer's value.

⚠ THE OBSERVATION MUST MATCH TOO, and that is a sharper constraint than it
looks: the trainer fills it with `custom_extract_obs_gpu` and this eval with
`custom_extract_obs_cpu`. Those are two implementations of one contract —
qpos(6) + qvel(6) + ee(3) + target(3) + ee_to_target(3) — and a permutation in
either is a policy that works on one device and is nonsense on the other, with
no error anywhere.

⚠ RESET FOLDS THE ARM TO HOME and draws a fresh target each episode; the
target is a mocap body, so it is visible in the render as the thing the jaw
should be touching.

Run:
    pixi run mojo run -I . examples/so101/sac_so_arm101_reach_eval_cpu.mojo
"""

from std.random import seed

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.sac import SAC
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model
from mojo_rl.envs.robots.so_arm101 import SoArm101ReachConfig


comptime EnvT = Phyics3dEnv[
    SoArm101Model, SoArm101ReachConfig, DT, TERMINATE_ON_UNHEALTHY=False
]
comptime OBS_DIM = EnvT.OBS_DIM  # 21
comptime ACT_DIM = EnvT.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime BATCH = 256
# Not in the checkpoint and unused by greedy eval — keep it small so the CPU
# allocation stays trivial.
comptime REPLAY_CAPACITY = 100_000

comptime CHECKPOINT_PATH = "sac_so_arm101_reach.ckpt"
comptime ACTION_SCALE = Scalar[DT](2.0)  # radians; MUST match the trainer

comptime NUM_EPISODES = 10
comptime MAX_STEPS = 500  # SoArmReachConfig.MAX_STEPS
comptime FRAME_DELAY_MS = 16  # ~60 FPS playback


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC — SO-ARM101 reach, CPU eval + 3D rendering")
    print("=" * 70)
    print("  OBS_DIM         =", OBS_DIM)
    print("  ACT_DIM         =", ACT_DIM, "(joint angles in RADIANS)")
    print("  action_scale    =", ACTION_SCALE)
    print("  Checkpoint      =", CHECKPOINT_PATH)
    print("  Episodes        =", NUM_EPISODES)
    print("=" * 70)

    var agent = SAC["cpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN](
        action_scale=ACTION_SCALE,
    )

    print("Loading checkpoint...")
    try:
        agent.load(CHECKPOINT_PATH)
        print("Checkpoint loaded.")
    except e:
        print("ERROR loading checkpoint:", e)
        print("Train first:")
        print(
            "  pixi run -e nvidia mojo run -I ."
            " examples/so101/sac_so_arm101_reach_training_gpu.mojo"
        )
        return
    print()

    print("-" * 70)
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    var avg_reward = Float64(
        agent.eval_render[EnvT](
            env,
            NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS,
            frame_delay_ms=FRAME_DELAY_MS,
            verbose=True,
        )
    )

    print("-" * 70)
    print("EVAL SUMMARY — SO-ARM101 reach")
    print("-" * 70)
    print("  Episodes        =", NUM_EPISODES)
    print("  Average reward  =", avg_reward)
    # Reward is a shaped `tolerance` in [0, 1] per step over MAX_STEPS control
    # steps, so the ceiling is 500 and what scores is reaching EARLY and
    # HOLDING. A 24k-step smoke run on Apple Silicon reached ~177.
    if avg_reward > 400.0:
        print("  Result: EXCELLENT — reaches and holds (mean > 400 / 500).")
    elif avg_reward > 200.0:
        print("  Result: STRONG — reaches most targets (mean > 200).")
    elif avg_reward > 50.0:
        print("  Result: PROGRESS — approaching the target (mean > 50).")
    else:
        print("  Result: EARLY — the jaw rarely enters the margin.")
    print("=" * 70)
