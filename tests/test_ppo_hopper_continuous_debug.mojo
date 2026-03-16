"""Debug script to compare CPU vs GPU action/reward computation."""

from std.random import seed

from mojo_rl.deep_agents.ppo import DeepPPOContinuousAgent
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.nn import dtype as agent_dtype


# Constants (must match training)
comptime OBS_DIM = HopperConfig.OBS_DIM  # 11
comptime ACTION_DIM = HopperConfig.ACTION_DIM  # 3
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DEBUG: Comparing CPU action/reward computation")
    print("=" * 70)
    print()

    # Create agent
    var agent = DeepPPOContinuousAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        rollout_len=ROLLOUT_LEN,
        n_envs=N_ENVS,
        gpu_minibatch_size=GPU_MINIBATCH_SIZE,
        clip_value=True,
        actor_lr=0.0003,
        critic_lr=0.0003,
    ](
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.0,
        value_loss_coef=0.5,
        num_epochs=10,
        target_kl=0.0,
        max_grad_norm=0.5,
    )

    # Load checkpoint
    print("Loading checkpoint...")
    agent.load_checkpoint("ppo_hopper.ckpt")
    print("Checkpoint loaded!")
    print()

    # Show log_std values (important for understanding action distribution)
    var log_std_offset = len(agent.actor.params) - ACTION_DIM
    print(
        "log_std params:",
        agent.actor.params[log_std_offset],
        agent.actor.params[log_std_offset + 1],
        agent.actor.params[log_std_offset + 2],
    )
    print(
        "  std values:",
        "exp(-4.72)≈0.009" if agent.actor.params[log_std_offset]
        < -4.0 else "normal",
    )
    print()

    # Create CPU environment (float64)
    var env = Hopper()

    # Reset and get initial observation
    var obs_list = env.reset_obs_list()
    print("Initial observation (first reset):")
    for i in range(len(obs_list)):
        print("  obs[" + String(i) + "]:", obs_list[i])
    print()

    # Convert to agent's dtype
    var obs = InlineArray[Scalar[agent_dtype], OBS_DIM](uninitialized=True)
    for i in range(OBS_DIM):
        obs[i] = Scalar[agent_dtype](obs_list[i])

    # Get action from policy (deterministic)
    var action_result = agent.select_action(obs, training=False)
    var actions = action_result[0]

    print("Network output (deterministic - using mean):")
    for j in range(ACTION_DIM):
        print("  action[" + String(j) + "]:", actions[j])
    print()

    # Take one step
    var action_list = List[Scalar[agent_dtype]]()
    for j in range(ACTION_DIM):
        var action_val = Float64(actions[j])
        # Clip (should be identity if network outputs in [-1,1])
        if action_val > 1.0:
            action_val = 1.0
            print(
                "  WARNING: action[" + String(j) + "] clipped from",
                actions[j],
                "to 1.0",
            )
        elif action_val < -1.0:
            action_val = -1.0
            print(
                "  WARNING: action[" + String(j) + "] clipped from",
                actions[j],
                "to -1.0",
            )
        action_list.append(Scalar[agent_dtype](action_val))

    print("Computing torques (action * 200):")
    for j in range(ACTION_DIM):
        var torque = Float64(action_list[j]) * 200.0
        print("  torque[" + String(j) + "]:", torque, "N·m")
    print()

    print("Computing control cost (0.05 * sum(torque^2)):")
    var ctrl_cost: Float64 = 0.0
    for j in range(ACTION_DIM):
        var torque = Float64(action_list[j]) * 200.0
        ctrl_cost += 0.05 * torque * torque
    print("  ctrl_cost:", ctrl_cost)
    print()

    # Step environment
    var result = env.step_continuous_vec[agent_dtype](action_list)
    var reward = result[1]
    var done = result[2]

    print("Step result:")
    print("  reward:", reward)
    print("  done:", done)
    print()

    # Decompose reward
    # reward = forward_vel + healthy_reward - ctrl_cost
    # If we know ctrl_cost and healthy_reward=1.0, we can infer forward_vel
    var healthy_reward: Float64 = 1.0 if not done else 0.0
    var inferred_forward_vel = Float64(reward) + ctrl_cost - healthy_reward
    print("Reward decomposition (assuming healthy):")
    print("  forward_vel (inferred):", inferred_forward_vel)
    print("  healthy_reward:", healthy_reward)
    print("  ctrl_cost:", ctrl_cost)
    print(
        "  reward = forward_vel + healthy - ctrl_cost =",
        inferred_forward_vel + healthy_reward - ctrl_cost,
    )
    print()

    # Run a few more steps to see pattern
    print("Running 10 steps:")
    var total_reward: Float64 = Float64(reward)
    for step in range(1, 10):
        # Get next observation
        var next_obs_list = result[0]
        for i in range(OBS_DIM):
            obs[i] = next_obs_list[i]

        # Get action
        action_result = agent.select_action(obs, training=False)
        actions = action_result[0]

        # Build action list
        action_list = List[Scalar[agent_dtype]]()
        for j in range(ACTION_DIM):
            var av = Float64(actions[j])
            if av > 1.0:
                av = 1.0
            elif av < -1.0:
                av = -1.0
            action_list.append(Scalar[agent_dtype](av))

        # Step
        result = env.step_continuous_vec[agent_dtype](action_list)
        reward = result[1]
        done = result[2]
        total_reward += Float64(reward)

        print(
            "  Step",
            step + 1,
            "| reward:",
            String(reward)[:8],
            "| actions:",
            String(actions[0])[:6],
            String(actions[1])[:6],
            String(actions[2])[:6],
        )

        if done:
            print("  Episode terminated at step", step + 1)
            break

    print()
    print("Total reward (10 steps):", total_reward)
    print("Average reward per step:", total_reward / 10.0)
    print()
    print(">>> Debug complete <<<")
