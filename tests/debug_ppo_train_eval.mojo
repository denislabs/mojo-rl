"""Debug script to trace PPO training vs evaluation behavior.

This script adds detailed diagnostics to understand the train-eval gap.

Run with:
    pixi run -e apple mojo run tests/debug_ppo_train_eval.mojo
"""

from random import seed
from time import perf_counter_ns
from math import sqrt

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.pendulum import PendulumV2, PConstants


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = PConstants.OBS_DIM  # 3
comptime NUM_ACTIONS = PConstants.ACTION_DIM  # 1
comptime HIDDEN_DIM = 64
comptime ROLLOUT_LEN = 200
comptime N_ENVS = 64  # Smaller for debugging
comptime GPU_MINIBATCH_SIZE = 64

comptime dtype = DType.float32


fn compute_weight_stats(params: List[Scalar[dtype]]) -> Tuple[Float64, Float64, Float64]:
    """Compute L1 norm, L2 norm, and max absolute value of weights."""
    var l1: Float64 = 0.0
    var l2: Float64 = 0.0
    var max_abs: Float64 = 0.0
    for i in range(len(params)):
        var val = Float64(abs(params[i]))
        l1 += val
        l2 += val * val
        if val > max_abs:
            max_abs = val
    return (l1, sqrt(l2), max_abs)


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DEBUG: PPO Train-Eval Gap Investigation")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # Create agent
        var agent = DeepPPOContinuousAgent[
            obs_dim=OBS_DIM,
            action_dim=NUM_ACTIONS,
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
            critic_lr=0.001,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=4,
        )

        # =====================================================================
        # Step 1: Check initial weights
        # =====================================================================
        print("=" * 70)
        print("STEP 1: Initial Weight Analysis")
        print("=" * 70)

        var init_stats = compute_weight_stats(agent.actor.params)
        print("Initial actor weights:")
        print("  L1 norm:", init_stats[0])
        print("  L2 norm:", init_stats[1])
        print("  Max abs:", init_stats[2])
        print("  First 5:", agent.actor.params[0], agent.actor.params[1],
              agent.actor.params[2], agent.actor.params[3], agent.actor.params[4])

        # Log_std is the last parameter
        var log_std_idx = len(agent.actor.params) - NUM_ACTIONS
        print("  log_std:", agent.actor.params[log_std_idx])
        print()

        # =====================================================================
        # Step 2: Evaluate BEFORE training (sanity check)
        # =====================================================================
        print("=" * 70)
        print("STEP 2: Evaluate BEFORE Training")
        print("=" * 70)

        var env = PendulumV2[dtype]()
        var pre_train_reward = agent.evaluate(
            env,
            num_episodes=5,
            max_steps=200,
            verbose=True,
            stochastic=True,
        )
        print()
        print("Pre-training average reward:", pre_train_reward)
        print("(Expected: ~-1200 to -1600 for untrained policy)")
        print()

        # =====================================================================
        # Step 3: Train for a short period
        # =====================================================================
        print("=" * 70)
        print("STEP 3: Training (500 episodes)")
        print("=" * 70)

        var metrics = agent.train_gpu[PendulumV2[dtype]](
            ctx,
            num_episodes=500,
            verbose=True,
            print_every=1,
        )

        var training_avg = metrics.mean_reward_last_n(100)
        print()
        print("Training average (last 100 episodes):", training_avg)
        print()

        # =====================================================================
        # Step 4: Check weights after training
        # =====================================================================
        print("=" * 70)
        print("STEP 4: Post-Training Weight Analysis")
        print("=" * 70)

        var post_stats = compute_weight_stats(agent.actor.params)
        print("Post-training actor weights:")
        print("  L1 norm:", post_stats[0])
        print("  L2 norm:", post_stats[1])
        print("  Max abs:", post_stats[2])
        print("  First 5:", agent.actor.params[0], agent.actor.params[1],
              agent.actor.params[2], agent.actor.params[3], agent.actor.params[4])
        print("  log_std:", agent.actor.params[log_std_idx])
        print()

        var weight_change_l1 = post_stats[0] - init_stats[0]
        var weight_change_l2 = post_stats[1] - init_stats[1]
        print("Weight changes:")
        print("  L1 delta:", weight_change_l1)
        print("  L2 delta:", weight_change_l2)
        print()

        # =====================================================================
        # Step 5: Evaluate AFTER training (CPU)
        # =====================================================================
        print("=" * 70)
        print("STEP 5: Evaluate AFTER Training (CPU)")
        print("=" * 70)

        var post_train_cpu_reward = agent.evaluate(
            env,
            num_episodes=10,
            max_steps=200,
            verbose=True,
            stochastic=True,
        )
        print()
        print("Post-training CPU evaluation average:", post_train_cpu_reward)
        print()

        # =====================================================================
        # Step 6: Evaluate AFTER training (GPU)
        # =====================================================================
        print("=" * 70)
        print("STEP 6: Evaluate AFTER Training (GPU)")
        print("=" * 70)

        var post_train_gpu_reward = agent.evaluate_gpu[PendulumV2[dtype]](
            ctx,
            num_episodes=100,
            max_steps=200,
            verbose=False,
            stochastic=True,
        )
        print("Post-training GPU evaluation average:", post_train_gpu_reward)
        print()

        # =====================================================================
        # Step 7: Compare forward pass outputs (CPU vs GPU)
        # =====================================================================
        print("=" * 70)
        print("STEP 7: Forward Pass Comparison (CPU vs GPU)")
        print("=" * 70)

        # Create a fixed observation for comparison
        var test_obs = InlineArray[Scalar[dtype], OBS_DIM](fill=Scalar[dtype](0.5))

        # CPU forward pass
        var cpu_output = InlineArray[Scalar[dtype], NUM_ACTIONS * 2](uninitialized=True)
        agent.actor.forward[1](test_obs, cpu_output)

        print("Test observation:", test_obs[0], test_obs[1], test_obs[2])
        print("CPU forward output (mean, log_std):", cpu_output[0], cpu_output[1])
        print()

        # GPU forward pass
        comptime ACTOR_OUT = NUM_ACTIONS * 2
        var obs_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM)
        var out_buf = ctx.enqueue_create_buffer[dtype](ACTOR_OUT)
        var params_buf = ctx.enqueue_create_buffer[dtype](len(agent.actor.params))
        comptime WORKSPACE = 4 * HIDDEN_DIM
        var workspace_buf = ctx.enqueue_create_buffer[dtype](WORKSPACE)

        # Copy obs to GPU
        var obs_host = ctx.enqueue_create_host_buffer[dtype](OBS_DIM)
        for i in range(OBS_DIM):
            obs_host[i] = test_obs[i]
        ctx.enqueue_copy(obs_buf, obs_host)

        # Copy params to GPU
        ctx.enqueue_copy(params_buf, agent.actor.params.unsafe_ptr())
        ctx.synchronize()

        # Forward pass on GPU
        agent.actor.model.forward_gpu_no_cache_ws[1](
            ctx,
            out_buf,
            obs_buf,
            params_buf,
            workspace_buf,
        )
        ctx.synchronize()

        # Copy output back
        var out_host = ctx.enqueue_create_host_buffer[dtype](ACTOR_OUT)
        ctx.enqueue_copy(out_host, out_buf)
        ctx.synchronize()

        print("GPU forward output (mean, log_std):", out_host[0], out_host[1])
        print()

        var mean_diff = Float64(abs(cpu_output[0] - out_host[0]))
        var log_std_diff = Float64(abs(cpu_output[1] - out_host[1]))
        print("Differences:")
        print("  Mean diff:", mean_diff)
        print("  Log_std diff:", log_std_diff)
        print()

        # =====================================================================
        # Summary
        # =====================================================================
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        print("Training reported average:      ", training_avg)
        print("Pre-training eval (CPU):        ", pre_train_reward)
        print("Post-training eval (CPU):       ", post_train_cpu_reward)
        print("Post-training eval (GPU):       ", post_train_gpu_reward)
        print()

        var train_eval_gap = training_avg - post_train_cpu_reward
        print("Train-Eval Gap (training - CPU eval):", train_eval_gap)
        print()

        if train_eval_gap > 500.0:
            print("LARGE GAP DETECTED!")
            print("This confirms a systematic issue with training vs evaluation.")
        elif train_eval_gap > 100.0:
            print("MODERATE GAP - some discrepancy exists")
        else:
            print("SMALL GAP - training and evaluation are consistent")

        print()
        print("=" * 70)

    print(">>> Debug script completed <<<")
