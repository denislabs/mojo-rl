"""HalfCheetah Policy Diagnostic — physics behavior with learned policy actions.

Loads a trained PPO checkpoint and runs CPU evaluation while logging
detailed physics diagnostics (penetration, impulses, rootz, velocities)
to identify ground penetration and bouncing issues.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_cheetah_policy_diag.mojo
"""

from math import sqrt, pi
from builtin.math import abs
from random import seed

from deep_agents.ppo import DeepPPOContinuousAgent
from deep_rl.constants import dtype
from envs.half_cheetah import HalfCheetah, HalfCheetahParams
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahParams,
    BODY_TORSO,
    BODY_BTHIGH,
    BODY_BSHIN,
    BODY_BFOOT,
    BODY_FTHIGH,
    BODY_FSHIN,
    BODY_FFOOT,
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    JOINT_BTHIGH,
    JOINT_BSHIN,
    JOINT_BFOOT,
    JOINT_FTHIGH,
    JOINT_FSHIN,
    JOINT_FFOOT,
)


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime C = HalfCheetahParams[DType.float32]
comptime OBS_DIM = C.OBS_DIM  # 17
comptime ACTION_DIM = C.ACTION_DIM  # 6
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048

# Evaluation settings
comptime NUM_EPISODES = 3
comptime MAX_STEPS = 1000
comptime DETERMINISTIC = True  # Use mean policy (no sampling noise)
comptime CHECKPOINT_PATH = "ppo_half_cheetah.ckpt"


fn main() raises:
    seed(42)
    print("=" * 80)
    print("HalfCheetah Policy Diagnostic")
    print("  Checkpoint:", CHECKPOINT_PATH)
    print("  Deterministic:", DETERMINISTIC)
    print("  Episodes:", NUM_EPISODES)
    print("  Max steps:", MAX_STEPS)
    print("=" * 80)

    # =========================================================================
    # Create agent and load checkpoint
    # =========================================================================

    var agent = DeepPPOContinuousAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        rollout_len=ROLLOUT_LEN,
        n_envs=N_ENVS,
        gpu_minibatch_size=GPU_MINIBATCH_SIZE,
        clip_value=True,
    ](
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        actor_lr=0.0003,
        critic_lr=0.0003,
        entropy_coef=0.0,
        value_loss_coef=0.5,
        num_epochs=10,
        target_kl=0.0,
        max_grad_norm=0.5,
        anneal_lr=False,
    )

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint(CHECKPOINT_PATH)
        print("Checkpoint loaded successfully!")
    except:
        print("ERROR: Could not load checkpoint!")
        print("Train first with:")
        print(
            "  pixi run -e apple mojo run"
            " tests/test_ppo_half_cheetah_continuous_gpu.mojo"
        )
        return

    print()

    # =========================================================================
    # Run episodes
    # =========================================================================

    var env = HalfCheetah[DType.float64, False]()

    for episode in range(NUM_EPISODES):
        print()
        print("=" * 80)
        print("Episode", episode + 1, "/", NUM_EPISODES)
        print("=" * 80)

        _ = env.reset()

        # Get initial observation
        var obs_list = env.get_obs_list()
        var obs = InlineArray[Scalar[dtype], OBS_DIM](uninitialized=True)
        for i in range(OBS_DIM):
            obs[i] = Scalar[dtype](obs_list[i])

        var max_pen = Float64(0.0)
        var max_rootz = Float64(env.data.qpos[JOINT_ROOTZ])
        var min_rootz = Float64(env.data.qpos[JOINT_ROOTZ])
        var max_vz = Float64(0.0)
        var min_vz = Float64(0.0)
        var max_imp_n = Float64(0.0)
        var episode_reward = Float64(0.0)
        var worst_pen_step = 0
        var worst_pen_actions = InlineArray[Float64, 6](0, 0, 0, 0, 0, 0)

        print()
        print(
            "  step | rootz    | vz       | rooty    | bfoot_z "
            " | ffoot_z  | contacts | max_pen  | max_imp_n"
            " | reward   | actions"
        )
        print("  " + "-" * 140)

        for step in range(MAX_STEPS):
            # Select action from policy
            var action_result = agent.select_action(
                obs, training=not DETERMINISTIC
            )
            var raw_actions = action_result[0].copy()

            # Clip actions to [-1, 1] (matching eval code)
            var actions = InlineArray[Float64, 6](uninitialized=True)
            for j in range(ACTION_DIM):
                var a = Float64(raw_actions[j])
                if a > 1.0:
                    a = 1.0
                elif a < -1.0:
                    a = -1.0
                actions[j] = a

            # Build action list for env
            var action_list = List[Scalar[dtype]]()
            for j in range(ACTION_DIM):
                action_list.append(Scalar[dtype](actions[j]))

            # Step the environment
            var result = env.step_continuous_vec[dtype](action_list)
            var reward = Float64(result[1])
            var done = result[2]
            episode_reward += reward

            # Update observation for next step
            var next_obs_list = result[0].copy()
            for i in range(OBS_DIM):
                obs[i] = next_obs_list[i]

            # Read physics state
            var rootz = Float64(env.data.qpos[JOINT_ROOTZ])
            var vz = Float64(env.data.qvel[JOINT_ROOTZ])
            var rooty = Float64(env.data.qpos[JOINT_ROOTY])
            var nc = Int(env.data.num_contacts)
            var bfoot_z = Float64(env.data.xpos[BODY_BFOOT * 3 + 2])
            var ffoot_z = Float64(env.data.xpos[BODY_FFOOT * 3 + 2])

            # Find max penetration and impulse
            var step_max_pen = Float64(0.0)
            var step_max_imp = Float64(0.0)
            for c in range(nc):
                var pen = -Float64(env.data.contacts[c].dist)
                var imp = Float64(env.data.contacts[c].force_n)
                if pen > step_max_pen:
                    step_max_pen = pen
                if imp > step_max_imp:
                    step_max_imp = imp

            # Track stats
            if step_max_pen > max_pen:
                max_pen = step_max_pen
                worst_pen_step = step + 1
                for j in range(6):
                    worst_pen_actions[j] = actions[j]
            if rootz > max_rootz:
                max_rootz = rootz
            if rootz < min_rootz:
                min_rootz = rootz
            if vz > max_vz:
                max_vz = vz
            if vz < min_vz:
                min_vz = vz
            if step_max_imp > max_imp_n:
                max_imp_n = step_max_imp

            # Print every 50 steps, plus first 5, and whenever penetration > 5mm
            var should_print = (
                step < 5
                or (step + 1) % 50 == 0
                or step == MAX_STEPS - 1
                or step_max_pen > 0.005
            )
            if should_print:
                print(
                    "  ",
                    step + 1,
                    " | ",
                    rootz,
                    " | ",
                    vz,
                    " | ",
                    rooty,
                    " | ",
                    bfoot_z,
                    " | ",
                    ffoot_z,
                    " | ",
                    nc,
                    " | ",
                    step_max_pen,
                    " | ",
                    step_max_imp,
                    " | ",
                    reward,
                    " | [",
                    actions[0],
                    actions[1],
                    actions[2],
                    actions[3],
                    actions[4],
                    actions[5],
                    "]",
                )

            if done:
                print("  >>> Episode terminated at step", step + 1)
                break

        print()
        print("  EPISODE", episode + 1, "SUMMARY:")
        print("    Total reward:", episode_reward)
        print("    Max penetration:", max_pen, "m (", max_pen * 1000, "mm )")
        print("    Worst penetration at step:", worst_pen_step)
        print(
            "    Actions at worst penetration: [",
            worst_pen_actions[0],
            worst_pen_actions[1],
            worst_pen_actions[2],
            worst_pen_actions[3],
            worst_pen_actions[4],
            worst_pen_actions[5],
            "]",
        )
        print("    RootZ range: [", min_rootz, ",", max_rootz, "]")
        print("    Vz range: [", min_vz, ",", max_vz, "]")
        print("    Max force_n:", max_imp_n)

    print()
    print("=" * 80)
    print("Diagnostic complete")
    print("=" * 80)
