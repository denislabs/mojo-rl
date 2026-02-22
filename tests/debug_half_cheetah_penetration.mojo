"""Debug script: step-by-step HalfCheetah eval with contact/penetration monitoring.

Runs the eval loop manually to inspect physics state at each step.
Prints per-step contact info and flags deep penetration events.

Run with:
    cd mojo-rl && pixi run mojo run tests/debug_half_cheetah_penetration.mojo
"""

from random import seed

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.half_cheetah import HalfCheetah, HalfCheetahParams
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
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

comptime C = HalfCheetahParams[DType.float32]
comptime OBS_DIM = C.OBS_DIM
comptime ACTION_DIM = C.ACTION_DIM
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048

comptime MAX_STEPS = 150
comptime NUM_EPISODES = 3
comptime BFOOT_BODY: Int = 4
comptime FFOOT_BODY: Int = 7
comptime DEEP_PENETRATION_THRESHOLD: Float64 = -0.03

# Concrete type alias — must supply explicit params for comptime type alias
comptime HCEnv = HalfCheetah[DType.float64, True]


fn print_deep_penetration_detail(env: HCEnv, step: Int):
    """Print full physics state on deep penetration."""
    print("  >>> Deep penetration at step", step, "<<<")
    print(
        "  qpos: rootx=", env.get_qpos(JOINT_ROOTX),
        "rootz=", env.get_qpos(JOINT_ROOTZ),
        "rooty=", env.get_qpos(JOINT_ROOTY),
    )
    print(
        "  qpos: bthigh=", env.get_qpos(JOINT_BTHIGH),
        "bshin=", env.get_qpos(JOINT_BSHIN),
        "bfoot=", env.get_qpos(JOINT_BFOOT),
    )
    print(
        "  qpos: fthigh=", env.get_qpos(JOINT_FTHIGH),
        "fshin=", env.get_qpos(JOINT_FSHIN),
        "ffoot=", env.get_qpos(JOINT_FFOOT),
    )
    print(
        "  qvel: vz=", env.get_qvel(JOINT_ROOTZ),
        "vy=", env.get_qvel(JOINT_ROOTY),
    )
    print(
        "  Body z: bfoot=", env.get_xpos(BFOOT_BODY * 3 + 2),
        " ffoot=", env.get_xpos(FFOOT_BODY * 3 + 2),
    )
    var ncon = env.data.num_contacts
    print("  Contacts (", ncon, "):")
    for c in range(ncon):
        var con = env.data.contacts[c]
        print(
            "    [", c, "]",
            " body_a=", con.body_a,
            " body_b=", con.body_b,
            " dist=", Float64(con.dist),
            " force_n=", Float64(con.force_n),
            " pos_z=", Float64(con.pos_z),
        )


fn run_episode(
    agent: DeepPPOContinuousAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        rollout_len=ROLLOUT_LEN,
        n_envs=N_ENVS,
        gpu_minibatch_size=GPU_MINIBATCH_SIZE,
        clip_value=True,
    ],
    episode: Int,
) raises:
    print("=" * 70)
    print("EPISODE", episode + 1)
    print("=" * 70)

    var env = HCEnv()
    var obs_list = env.reset_obs_list()
    var obs = InlineArray[Scalar[DType.float32], OBS_DIM](uninitialized=True)
    for i in range(OBS_DIM):
        obs[i] = Scalar[DType.float32](obs_list[i])

    var total_reward: Float64 = 0.0
    var worst_dist_ever: Float64 = 0.0
    var step_worst: Int = 0
    var first_fall_step: Int = -1

    print("Step | rootz   | ncon | worst_dist | sum_force_n | flags")
    print("-" * 67)

    for step in range(MAX_STEPS):
        # Select action (deterministic)
        var action_result = agent.select_action(obs, training=False)
        var actions = action_result[0].copy()

        var action_list = List[Scalar[DType.float32]]()
        for j in range(ACTION_DIM):
            var a = Float64(actions[j])
            if a > 1.0:
                a = 1.0
            elif a < -1.0:
                a = -1.0
            action_list.append(Scalar[DType.float32](a))

        var result = env.step_continuous_vec[DType.float32](action_list)
        var next_obs_list = result[0].copy()
        var reward = Float64(result[1])
        var done = result[2]

        total_reward += reward

        # Collect physics info using accessor methods
        var rootz = Float64(env.get_qpos(JOINT_ROOTZ))
        var ncon = env.data.num_contacts
        var worst_dist: Float64 = 0.0
        var sum_force_n: Float64 = 0.0
        for c in range(ncon):
            var con = env.data.contacts[c]
            var d = Float64(con.dist)
            var fn_val = Float64(con.force_n)
            if d < worst_dist:
                worst_dist = d
            sum_force_n += fn_val

        if worst_dist < worst_dist_ever:
            worst_dist_ever = worst_dist
            step_worst = step

        if rootz < 0.0 and first_fall_step < 0:
            first_fall_step = step

        # Flags
        var flags = String("")
        if worst_dist < DEEP_PENETRATION_THRESHOLD:
            flags = flags + " DEEP_PEN"
        if rootz < 0.15:
            flags = flags + " LOW_Z"
        if ncon == 0 and rootz < 0.3:
            flags = flags + " NO_CONTACT_NEAR_GROUND"

        print(
            String(step).ascii_rjust(4),
            "|",
            String(rootz)[:7].ascii_ljust(7),
            "|",
            String(ncon).ascii_rjust(4),
            "|",
            String(worst_dist)[:10].ascii_ljust(10),
            "|",
            String(sum_force_n)[:11].ascii_ljust(11),
            "|",
            flags,
        )

        if worst_dist < DEEP_PENETRATION_THRESHOLD:
            print_deep_penetration_detail(env, step)

        for i in range(OBS_DIM):
            obs[i] = next_obs_list[i]

        if done:
            print("  [Episode ended at step", step, "]")
            break

    print()
    print("Episode", episode + 1, "summary:")
    print("  Total reward:", total_reward)
    print("  Worst contact dist:", worst_dist_ever, "at step", step_worst)
    if first_fall_step >= 0:
        print("  First fall (rootz<0) at step:", first_fall_step)
    else:
        print("  No fall detected in first", MAX_STEPS, "steps")
    print()


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DEBUG: Step-by-step HalfCheetah penetration analysis")
    print("=" * 70)
    print()

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
        agent.load_checkpoint("ppo_half_cheetah.ckpt")
        print("Checkpoint loaded.")
    except:
        print("ERROR: Could not load checkpoint. Train first.")
        return
    print()

    for episode in range(NUM_EPISODES):
        run_episode(agent, episode)

    print("=" * 70)
    print("Debug complete.")
    print("WHAT TO LOOK FOR:")
    print("  - Does 'worst_dist' gradually worsen or suddenly jump?")
    print("  - Does ncon drop to 0 near the ground? (missed contact)")
    print("  - What is rootz at moment of deep penetration?")
    print("    If rootz >> 0: feet penetrating while body is high = physics bug")
    print("    If rootz ~0 : robot is falling/crashed = policy mismatch")
    print("=" * 70)
