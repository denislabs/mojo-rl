"""GPU Multi-Armed Bandit - Simplest possible GPU RL.

This is a proof-of-concept for a full GPU RL pipeline:
- Environment runs on GPU (bandit reward sampling)
- Algorithm runs on GPU (gradient bandit updates)
- No CPU<->GPU transfers during training (except for logging)

The K-Armed Bandit problem:
- K arms (actions), each with hidden reward probability
- Goal: maximize total reward by learning which arms are best
- No state, no transitions - simplest possible RL

Gradient Bandit Algorithm:
- Maintain preferences H[a] for each action
- Policy: π[a] = softmax(H) = exp(H[a]) / Σexp(H[k])
- Update: H[a] += α * (R - baseline) * (1 - π[a])  for chosen action
          H[a] -= α * (R - baseline) * π[a]        for other actions

Run with:
    pixi run -e apple mojo run examples/gpu_bandit.mojo
"""

from time import perf_counter_ns
from math import exp, log
from random import seed, random_float64

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor


# =============================================================================
# GPU Random Number Generator (xorshift32)
# =============================================================================


fn xorshift32(state: Scalar[DType.uint32]) -> Scalar[DType.uint32]:
    """Simple xorshift PRNG - each thread maintains its own state."""
    var x = state
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


fn random_uniform(state: Scalar[DType.uint32]) -> Scalar[DType.float32]:
    """Convert random uint32 to uniform [0, 1) float."""
    return Scalar[DType.float32](state) / Scalar[DType.float32](
        Scalar[DType.uint32].MAX
    )


# =============================================================================
# GPU Bandit Kernel - Everything in One!
# =============================================================================


fn bandit_multistep_kernel[
    dtype: DType,
    NUM_BANDITS: Int,  # Number of parallel bandit instances
    NUM_ARMS: Int,  # K in K-armed bandit
    STEPS_PER_KERNEL: Int,  # Number of steps to run per kernel launch
    TPB: Int,  # Threads per block
](
    # Bandit state (mutable)
    preferences: LayoutTensor[
        dtype, Layout.row_major(NUM_BANDITS * NUM_ARMS), MutAnyOrigin
    ],
    baseline: LayoutTensor[dtype, Layout.row_major(NUM_BANDITS), MutAnyOrigin],
    total_reward: LayoutTensor[
        dtype, Layout.row_major(NUM_BANDITS), MutAnyOrigin
    ],
    rng_state: LayoutTensor[
        DType.uint32, Layout.row_major(NUM_BANDITS), MutAnyOrigin
    ],
    step_count: LayoutTensor[
        DType.int32, Layout.row_major(NUM_BANDITS), MutAnyOrigin
    ],
    # Arm probabilities (immutable) - the "true" reward probabilities
    arm_probs: LayoutTensor[dtype, Layout.row_major(NUM_ARMS), ImmutAnyOrigin],
    # Hyperparameters
    alpha: Scalar[dtype],  # Learning rate
):
    """Single kernel that runs MULTIPLE bandit steps per thread.

    Each thread handles one bandit instance and runs STEPS_PER_KERNEL steps:
    1. Compute softmax policy from preferences
    2. Sample action from policy
    3. Sample reward from arm's probability
    4. Update baseline (running average)
    5. Update preferences (gradient bandit)
    6. Accumulate total reward
    7. Repeat for STEPS_PER_KERNEL iterations
    """
    var bandit_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    if bandit_idx >= NUM_BANDITS:
        return

    # Get this bandit's state into registers
    var rng = rebind[Scalar[DType.uint32]](rng_state[bandit_idx])
    var current_baseline = rebind[Scalar[dtype]](baseline[bandit_idx])
    var current_total = rebind[Scalar[dtype]](total_reward[bandit_idx])
    var current_step = Int(step_count[bandit_idx])

    # Load preferences into registers
    var H = InlineArray[Scalar[dtype], NUM_ARMS](fill=Scalar[dtype](0))
    for a in range(NUM_ARMS):
        H[a] = rebind[Scalar[dtype]](preferences[bandit_idx * NUM_ARMS + a])

    # Run multiple steps without any memory sync
    for step in range(STEPS_PER_KERNEL):
        # =================================================================
        # Step 1: Compute softmax policy π[a] = exp(H[a]) / Σexp(H[k])
        # =================================================================

        # Find max for numerical stability
        var max_h = H[0]
        for a in range(1, NUM_ARMS):
            if H[a] > max_h:
                max_h = H[a]

        # Compute exp(H - max) and sum
        var exp_h = InlineArray[Scalar[dtype], NUM_ARMS](fill=Scalar[dtype](0))
        var sum_exp: Scalar[dtype] = 0
        for a in range(NUM_ARMS):
            exp_h[a] = exp(H[a] - max_h)
            sum_exp += exp_h[a]

        # Normalize to get probabilities
        var pi = InlineArray[Scalar[dtype], NUM_ARMS](fill=Scalar[dtype](0))
        for a in range(NUM_ARMS):
            pi[a] = exp_h[a] / sum_exp

        # =================================================================
        # Step 2: Sample action from policy (inverse CDF sampling)
        # =================================================================

        rng = xorshift32(rng)
        var u = random_uniform(rng)

        var action = 0
        var cumsum = pi[0]
        for a in range(1, NUM_ARMS):
            if Scalar[dtype](u) > cumsum:
                action = a
                cumsum += pi[a]

        # =================================================================
        # Step 3: Sample reward from arm's Bernoulli distribution
        # =================================================================

        rng = xorshift32(rng)
        var reward_sample = random_uniform(rng)
        var arm_prob = rebind[Scalar[dtype]](arm_probs[action])
        var reward = Scalar[dtype](1.0) if Scalar[dtype](
            reward_sample
        ) < arm_prob else Scalar[dtype](0.0)

        # =================================================================
        # Step 4: Update baseline (incremental mean)
        # =================================================================

        current_step += 1
        var n = Scalar[dtype](current_step)
        current_baseline = current_baseline + (reward - current_baseline) / n

        # =================================================================
        # Step 5: Update preferences (gradient bandit update)
        # =================================================================

        var advantage = reward - current_baseline

        for a in range(NUM_ARMS):
            if a == action:
                H[a] = H[a] + alpha * advantage * (Scalar[dtype](1.0) - pi[a])
            else:
                H[a] = H[a] - alpha * advantage * pi[a]

        # =================================================================
        # Step 6: Accumulate total reward
        # =================================================================

        current_total += reward

    # Write back state to global memory (only once at end!)
    for a in range(NUM_ARMS):
        preferences[bandit_idx * NUM_ARMS + a] = H[a]
    baseline[bandit_idx] = current_baseline
    total_reward[bandit_idx] = current_total
    step_count[bandit_idx] = Int32(current_step)
    rng_state[bandit_idx] = rng


# =============================================================================
# CPU Baseline for Comparison
# =============================================================================


fn cpu_bandit_step[
    NUM_ARMS: Int
](
    mut preferences: List[Float32],
    mut baseline: Float32,
    mut total_reward: Float32,
    arm_probs: List[Float32],
    alpha: Float32,
    step_count: Int,
) -> Float32:
    """CPU version of one bandit step for comparison."""
    # Softmax
    var max_h: Float32 = preferences[0]
    for a in range(1, NUM_ARMS):
        if preferences[a] > max_h:
            max_h = preferences[a]

    var sum_exp: Float32 = 0
    var exp_h = List[Float32](capacity=NUM_ARMS)
    for a in range(NUM_ARMS):
        var e = exp(preferences[a] - max_h)
        exp_h.append(e)
        sum_exp += e

    var pi = List[Float32](capacity=NUM_ARMS)
    for a in range(NUM_ARMS):
        pi.append(exp_h[a] / sum_exp)

    # Sample action
    var u = Float32(random_float64())
    var action = 0
    var cumsum = pi[0]
    for a in range(1, NUM_ARMS):
        if u > cumsum:
            action = a
            cumsum += pi[a]

    # Sample reward
    var reward: Float32 = Float32(1.0) if Float32(random_float64()) < arm_probs[
        action
    ] else Float32(0.0)

    # Update baseline
    var n = Float32(step_count + 1)
    baseline = baseline + (reward - baseline) / n

    # Update preferences
    var advantage = reward - baseline
    for a in range(NUM_ARMS):
        if a == action:
            preferences[a] += alpha * advantage * (1.0 - pi[a])
        else:
            preferences[a] -= alpha * advantage * pi[a]

    total_reward += reward
    return reward


# =============================================================================
# Benchmark Functions
# =============================================================================


fn benchmark_gpu[
    NUM_BANDITS: Int, NUM_ARMS: Int, STEPS_PER_KERNEL: Int = 100
](
    ctx: DeviceContext,
    num_steps: Int,
    arm_probs_cpu: List[Float32],
) raises -> Tuple[Float64, Float32]:
    """Benchmark GPU bandit and return (time_ns, avg_reward).

    Uses multi-step kernel: each kernel launch runs STEPS_PER_KERNEL steps.
    This amortizes kernel launch overhead across many steps.
    """
    comptime dtype = DType.float32
    comptime TPB = 256

    # Allocate GPU buffers
    var preferences = ctx.enqueue_create_buffer[dtype](NUM_BANDITS * NUM_ARMS)
    var baseline = ctx.enqueue_create_buffer[dtype](NUM_BANDITS)
    var total_reward = ctx.enqueue_create_buffer[dtype](NUM_BANDITS)
    var rng_state = ctx.enqueue_create_buffer[DType.uint32](NUM_BANDITS)
    var step_count = ctx.enqueue_create_buffer[DType.int32](NUM_BANDITS)
    var arm_probs = ctx.enqueue_create_buffer[dtype](NUM_ARMS)

    # Initialize
    preferences.enqueue_fill(0)  # Start with equal preferences
    baseline.enqueue_fill(0)
    total_reward.enqueue_fill(0)
    step_count.enqueue_fill(0)

    # Initialize RNG with different seeds per bandit
    with rng_state.map_to_host() as host:
        for i in range(NUM_BANDITS):
            host[i] = UInt32(i + 12345)  # Different seed per bandit

    # Copy arm probabilities
    with arm_probs.map_to_host() as host:
        for i in range(NUM_ARMS):
            host[i] = Scalar[dtype](arm_probs_cpu[i])

    ctx.synchronize()

    # Create tensors
    var pref_t = LayoutTensor[
        dtype, Layout.row_major(NUM_BANDITS * NUM_ARMS), MutAnyOrigin
    ](preferences)
    var base_t = LayoutTensor[
        dtype, Layout.row_major(NUM_BANDITS), MutAnyOrigin
    ](baseline)
    var reward_t = LayoutTensor[
        dtype, Layout.row_major(NUM_BANDITS), MutAnyOrigin
    ](total_reward)
    var rng_t = LayoutTensor[
        DType.uint32, Layout.row_major(NUM_BANDITS), MutAnyOrigin
    ](rng_state)
    var step_t = LayoutTensor[
        DType.int32, Layout.row_major(NUM_BANDITS), MutAnyOrigin
    ](step_count)
    var probs_t = LayoutTensor[
        dtype, Layout.row_major(NUM_ARMS), ImmutAnyOrigin
    ](arm_probs)

    var alpha = Scalar[dtype](0.1)
    comptime num_blocks = (NUM_BANDITS + TPB - 1) // TPB
    comptime kernel = bandit_multistep_kernel[
        dtype, NUM_BANDITS, NUM_ARMS, STEPS_PER_KERNEL, TPB
    ]

    # Calculate number of kernel launches needed
    var num_launches = (num_steps + STEPS_PER_KERNEL - 1) // STEPS_PER_KERNEL

    # Warm up (10 launches = 10 * STEPS_PER_KERNEL steps)
    for _ in range(10):
        ctx.enqueue_function[kernel, kernel](
            pref_t,
            base_t,
            reward_t,
            rng_t,
            step_t,
            probs_t,
            alpha,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    # Reset for actual benchmark
    preferences.enqueue_fill(0)
    baseline.enqueue_fill(0)
    total_reward.enqueue_fill(0)
    step_count.enqueue_fill(0)
    ctx.synchronize()

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(num_launches):
        ctx.enqueue_function[kernel, kernel](
            pref_t,
            base_t,
            reward_t,
            rng_t,
            step_t,
            probs_t,
            alpha,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    # Read back total reward (actual steps = num_launches * STEPS_PER_KERNEL)
    var actual_steps = num_launches * STEPS_PER_KERNEL
    var avg_reward: Float32 = 0
    with total_reward.map_to_host() as host:
        for i in range(NUM_BANDITS):
            avg_reward += Float32(host[i])
    avg_reward /= Float32(NUM_BANDITS * actual_steps)

    return (Float64(end - start), avg_reward)


fn benchmark_cpu[
    NUM_ARMS: Int
](
    num_bandits: Int,
    num_steps: Int,
    arm_probs_cpu: List[Float32],
) -> Tuple[
    Float64, Float32
]:
    """Benchmark CPU bandit."""
    # Initialize bandits
    var all_preferences = List[List[Float32]](capacity=num_bandits)
    var all_baselines = List[Float32](capacity=num_bandits)
    var all_rewards = List[Float32](capacity=num_bandits)

    for _ in range(num_bandits):
        var prefs = List[Float32](capacity=NUM_ARMS)
        for _ in range(NUM_ARMS):
            prefs.append(Float32(0))
        all_preferences.append(prefs^)
        all_baselines.append(Float32(0))
        all_rewards.append(Float32(0))

    # Warm up
    for step in range(100):
        for b in range(num_bandits):
            _ = cpu_bandit_step[NUM_ARMS](
                all_preferences[b],
                all_baselines[b],
                all_rewards[b],
                arm_probs_cpu,
                Float32(0.1),
                step,
            )

    # Reset
    for b in range(num_bandits):
        for a in range(NUM_ARMS):
            all_preferences[b][a] = 0
        all_baselines[b] = 0
        all_rewards[b] = 0

    # Benchmark
    var start = perf_counter_ns()
    for step in range(num_steps):
        for b in range(num_bandits):
            _ = cpu_bandit_step[NUM_ARMS](
                all_preferences[b],
                all_baselines[b],
                all_rewards[b],
                arm_probs_cpu,
                Float32(0.1),
                step,
            )
    var end = perf_counter_ns()

    # Compute average reward
    var avg_reward: Float32 = 0
    for b in range(num_bandits):
        avg_reward += all_rewards[b]
    avg_reward /= Float32(num_bandits * num_steps)

    return (Float64(end - start), avg_reward)


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("GPU Multi-Armed Bandit - Full GPU RL Pipeline POC")
    print("=" * 70)
    print()

    # Problem setup
    comptime NUM_ARMS = 10

    # True arm probabilities (arm 7 is best with p=0.9)
    var arm_probs = List[Float32](capacity=NUM_ARMS)
    arm_probs.append(Float32(0.1))  # Arm 0
    arm_probs.append(Float32(0.2))  # Arm 1
    arm_probs.append(Float32(0.3))  # Arm 2
    arm_probs.append(Float32(0.4))  # Arm 3
    arm_probs.append(Float32(0.5))  # Arm 4
    arm_probs.append(Float32(0.6))  # Arm 5
    arm_probs.append(Float32(0.7))  # Arm 6
    arm_probs.append(Float32(0.9))  # Arm 7 - BEST
    arm_probs.append(Float32(0.3))  # Arm 8
    arm_probs.append(Float32(0.2))  # Arm 9

    print("10-Armed Bandit Problem:")
    print(
        "  Arm probabilities: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.3,"
        " 0.2]"
    )
    print("  Best arm: 7 (p=0.9)")
    print("  Algorithm: Gradient Bandit with α=0.1")
    print()

    var num_steps = 1000

    # =========================================================================
    # Test 1: Small scale - verify correctness
    # =========================================================================
    print("-" * 70)
    print("Test 1: Correctness Check (100 bandits, 1000 steps)")
    print("-" * 70)

    with DeviceContext() as ctx:
        var result = benchmark_gpu[100, NUM_ARMS](ctx, num_steps, arm_probs)
        var gpu_time = result[0]
        var gpu_reward = result[1]
        print("  GPU avg reward per step: " + String(gpu_reward)[:6])
        print("  (Optimal would be 0.9, random would be ~0.42)")

        if gpu_reward > 0.7:
            print("  -> Learning is working! (reward > 0.7)")
        else:
            print("  -> Hmm, reward seems low...")

    print()

    # =========================================================================
    # Test 2: Scaling - GPU vs CPU with Multi-Step Kernels
    # =========================================================================
    print("-" * 70)
    print("Test 2: GPU vs CPU Performance (Multi-Step Kernel)")
    print("-" * 70)
    print(
        "GPU uses STEPS_PER_KERNEL=100 (only 10 kernel launches for 1000 steps)"
    )
    print()

    with DeviceContext() as ctx:
        # Small scale
        print("  NUM_BANDITS=1000, NUM_STEPS=1000:")
        var cpu_result_1k = benchmark_cpu[NUM_ARMS](1000, 1000, arm_probs)
        var cpu_time_1k = cpu_result_1k[0]
        var gpu_result_1k = benchmark_gpu[1000, NUM_ARMS, 100](
            ctx, 1000, arm_probs
        )
        var gpu_time_1k = gpu_result_1k[0]
        print("    CPU: " + String(cpu_time_1k / 1e6)[:8] + " ms")
        print(
            "    GPU: "
            + String(gpu_time_1k / 1e6)[:8]
            + " ms  (10 kernel launches)"
        )
        print("    Speedup: " + String(cpu_time_1k / gpu_time_1k)[:5] + "x")
        print()

        # Medium scale
        print("  NUM_BANDITS=10000, NUM_STEPS=1000:")
        var cpu_result_10k = benchmark_cpu[NUM_ARMS](10000, 1000, arm_probs)
        var cpu_time_10k = cpu_result_10k[0]
        var gpu_result_10k = benchmark_gpu[10000, NUM_ARMS, 100](
            ctx, 1000, arm_probs
        )
        var gpu_time_10k = gpu_result_10k[0]
        print("    CPU: " + String(cpu_time_10k / 1e6)[:8] + " ms")
        print(
            "    GPU: "
            + String(gpu_time_10k / 1e6)[:8]
            + " ms  (10 kernel launches)"
        )
        print("    Speedup: " + String(cpu_time_10k / gpu_time_10k)[:5] + "x")
        print()

        # Large scale
        print("  NUM_BANDITS=100000, NUM_STEPS=1000:")
        var gpu_result_100k = benchmark_gpu[100000, NUM_ARMS, 100](
            ctx, 1000, arm_probs
        )
        var gpu_time_100k = gpu_result_100k[0]
        var gpu_reward_100k = gpu_result_100k[1]
        print(
            "    GPU: "
            + String(gpu_time_100k / 1e6)[:8]
            + " ms  (10 kernel launches)"
        )
        print("    GPU avg reward: " + String(gpu_reward_100k)[:6])
        print(
            "    Throughput: "
            + String(Int(100000.0 * 1000.0 / (gpu_time_100k / 1e9)))
            + " bandit-steps/sec"
        )
        print()

        # Very large scale with more steps per kernel
        print("  NUM_BANDITS=100000, NUM_STEPS=10000 (STEPS_PER_KERNEL=1000):")
        var gpu_result_big = benchmark_gpu[100000, NUM_ARMS, 1000](
            ctx, 10000, arm_probs
        )
        var gpu_time_big = gpu_result_big[0]
        var gpu_reward_big = gpu_result_big[1]
        print(
            "    GPU: "
            + String(gpu_time_big / 1e6)[:8]
            + " ms  (10 kernel launches)"
        )
        print("    GPU avg reward: " + String(gpu_reward_big)[:6])
        print(
            "    Throughput: "
            + String(Int(100000.0 * 10000.0 / (gpu_time_big / 1e9)))
            + " bandit-steps/sec"
        )
        print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("Summary: GPU Multi-Armed Bandit")
    print("=" * 70)
    print()
    print("This POC demonstrates a FULL GPU RL pipeline:")
    print("  1. Environment (reward sampling) runs on GPU")
    print("  2. Algorithm (gradient bandit) runs on GPU")
    print("  3. RNG runs on GPU (xorshift32 per thread)")
    print("  4. NO CPU<->GPU transfers during training!")
    print()
    print("Key insights:")
    print("  - GPU wins when we have many parallel instances")
    print("  - Each thread handles one complete bandit instance")
    print("  - All state (preferences, baseline, RNG) lives on GPU")
    print("  - Only transfer at end for logging/evaluation")
    print()
    print("Next steps for DQN:")
    print("  - Replace bandit with vectorized CartPole env")
    print("  - Replace preferences with neural network weights")
    print("  - Add replay buffer on GPU")
    print("  - Keep same pattern: one kernel does everything")
    print("=" * 70)
