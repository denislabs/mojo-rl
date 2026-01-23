"""Debug script to verify checkpoint loading.

This script creates an agent, loads a checkpoint, and verifies
the parameters were loaded correctly.

Run with:
    pixi run -e apple mojo run tests/debug_checkpoint_load.mojo
"""

from random import seed

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.pendulum import PendulumV2, PConstants


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = PConstants.OBS_DIM  # 3
comptime NUM_ACTIONS = PConstants.ACTION_DIM  # 1
comptime HIDDEN_DIM = 64
comptime ROLLOUT_LEN = 200
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 256

comptime dtype = DType.float32


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DEBUG: Checkpoint Loading Verification")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 1: Create agent and show INITIAL params
    # =========================================================================
    print("STEP 1: Creating new agent (initial params)")
    print("-" * 60)

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

    var num_actor_params = len(agent.actor.params)
    var log_std_idx = num_actor_params - NUM_ACTIONS

    print("  Total actor params:", num_actor_params)
    print("  log_std index:", log_std_idx)
    print()
    print("  Initial params (first 5):")
    print("    [0]:", agent.actor.params[0])
    print("    [1]:", agent.actor.params[1])
    print("    [2]:", agent.actor.params[2])
    print("    [3]:", agent.actor.params[3])
    print("    [4]:", agent.actor.params[4])
    print()
    print("  Initial params (last 5):")
    for i in range(5):
        var idx = num_actor_params - 5 + i
        print("    [" + String(idx) + "]:", agent.actor.params[idx])
    print()
    print("  Initial log_std:", agent.actor.params[log_std_idx])
    print("  (Expected: -0.5 from init_params_small)")
    print()

    # Test forward pass BEFORE loading
    print("  Forward pass with test obs [0.5, 0.5, 0.5]:")
    var test_obs = InlineArray[Scalar[dtype], OBS_DIM](fill=Scalar[dtype](0.5))
    var output_before = InlineArray[Scalar[dtype], NUM_ACTIONS * 2](uninitialized=True)
    agent.actor.forward[1](test_obs, output_before)
    print("    Mean:", output_before[0])
    print("    log_std:", output_before[1])
    print()

    # =========================================================================
    # Step 2: Load checkpoint
    # =========================================================================
    print("STEP 2: Loading checkpoint")
    print("-" * 60)

    var checkpoint_path = "ppo_pendulum_gpu.ckpt"
    print("  Path:", checkpoint_path)

    try:
        agent.load_checkpoint(checkpoint_path)
        print("  Load successful!")
    except e:
        print("  ERROR:", String(e))
        return

    print()

    # =========================================================================
    # Step 3: Show LOADED params
    # =========================================================================
    print("STEP 3: Verifying loaded params")
    print("-" * 60)

    print("  Loaded params (first 5):")
    print("    [0]:", agent.actor.params[0])
    print("    [1]:", agent.actor.params[1])
    print("    [2]:", agent.actor.params[2])
    print("    [3]:", agent.actor.params[3])
    print("    [4]:", agent.actor.params[4])
    print()
    print("  Loaded params (last 5):")
    for i in range(5):
        var idx = num_actor_params - 5 + i
        print("    [" + String(idx) + "]:", agent.actor.params[idx])
    print()
    print("  Loaded log_std:", agent.actor.params[log_std_idx])
    print("  (Should NOT be -0.5 if training changed it)")
    print()

    # Test forward pass AFTER loading
    print("  Forward pass with test obs [0.5, 0.5, 0.5]:")
    var output_after = InlineArray[Scalar[dtype], NUM_ACTIONS * 2](uninitialized=True)
    agent.actor.forward[1](test_obs, output_after)
    print("    Mean:", output_after[0])
    print("    log_std:", output_after[1])
    print()

    # Compare
    print("  Forward pass difference:")
    print("    Mean diff:", Float64(output_after[0] - output_before[0]))
    print("    log_std diff:", Float64(output_after[1] - output_before[1]))
    print()

    # =========================================================================
    # Step 4: Quick evaluation
    # =========================================================================
    print("STEP 4: Quick CPU evaluation (5 episodes)")
    print("-" * 60)

    var env = PendulumV2[dtype]()
    var avg_reward = agent.evaluate(
        env,
        num_episodes=5,
        max_steps=200,
        verbose=True,
        stochastic=True,
    )
    print()
    print("  Average reward:", avg_reward)
    print()

    # =========================================================================
    # Analysis
    # =========================================================================
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    var output_diff = Float64(abs(output_after[0] - output_before[0]))
    var log_std_diff = Float64(abs(output_after[1] - output_before[1]))

    if output_diff < 0.001 and log_std_diff < 0.001:
        print("WARNING: Forward pass output is IDENTICAL before and after loading!")
        print("This means either:")
        print("  1. The checkpoint wasn't loaded at all")
        print("  2. The checkpoint contains initial (untrained) weights")
        print("  3. There's a bug in the loading code")
    else:
        print("Forward pass outputs differ after loading - checkpoint loaded successfully")

    if Float64(agent.actor.params[log_std_idx]) == -0.5:
        print()
        print("WARNING: log_std is still -0.5 (initial value)!")
        print("The checkpoint may not contain trained log_std values.")
    else:
        print()
        print("log_std was modified from initial value - training affected it")

    if avg_reward > -300.0:
        print()
        print("Evaluation shows GOOD performance - agent learned!")
    elif avg_reward > -800.0:
        print()
        print("Evaluation shows MODERATE performance")
    else:
        print()
        print("Evaluation shows POOR performance (random policy level)")
        print("This indicates train-eval gap issue")

    print()
    print("=" * 70)
    print(">>> Debug script completed <<<")
