"""Benchmark: CPU vs GPU vectorized environment stepping.

This benchmark helps identify where GPU becomes worthwhile for RL environments.

Tests:
1. CPU loop-based CartPole stepping (baseline)
2. GPU kernel-based CartPole stepping

We measure:
- Time per step for different batch sizes (64, 128, 256, 512, 1024)
- Steps per second throughput
- Crossover point where GPU wins

Run with:
    pixi run -e apple mojo run examples/benchmark_vec_env.mojo
"""

from time import perf_counter_ns
from math import cos, sin
from random import random_float64, seed

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor


# =============================================================================
# CPU Implementation: Loop-based vectorized CartPole
# =============================================================================


struct CPUVecCartPole[num_envs: Int]:
    """CPU vectorized CartPole using simple loops (not SIMD-limited)."""

    # Physics constants
    var gravity: Float64
    var masscart: Float64
    var masspole: Float64
    var total_mass: Float64
    var length: Float64
    var polemass_length: Float64
    var force_mag: Float64
    var tau: Float64

    # Thresholds
    var theta_threshold: Float64
    var x_threshold: Float64

    # State arrays
    var x: InlineArray[Float64, Self.num_envs]
    var x_dot: InlineArray[Float64, Self.num_envs]
    var theta: InlineArray[Float64, Self.num_envs]
    var theta_dot: InlineArray[Float64, Self.num_envs]
    var steps: InlineArray[Int, Self.num_envs]
    var max_steps: Int

    fn __init__(out self, max_steps: Int = 500):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = 1.1
        self.length = 0.5
        self.polemass_length = 0.05
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold = 0.2095  # 12 degrees
        self.x_threshold = 2.4
        self.max_steps = max_steps

        self.x = InlineArray[Float64, Self.num_envs](fill=0)
        self.x_dot = InlineArray[Float64, Self.num_envs](fill=0)
        self.theta = InlineArray[Float64, Self.num_envs](fill=0)
        self.theta_dot = InlineArray[Float64, Self.num_envs](fill=0)
        self.steps = InlineArray[Int, Self.num_envs](fill=0)

    fn reset(mut self):
        """Reset all environments."""
        for i in range(Self.num_envs):
            self.x[i] = (random_float64() - 0.5) * 0.1
            self.x_dot[i] = (random_float64() - 0.5) * 0.1
            self.theta[i] = (random_float64() - 0.5) * 0.1
            self.theta_dot[i] = (random_float64() - 0.5) * 0.1
            self.steps[i] = 0

    fn step(
        mut self,
        actions: InlineArray[Int, Self.num_envs],
        mut rewards: InlineArray[Float64, Self.num_envs],
        mut dones: InlineArray[Bool, Self.num_envs],
    ):
        """Step all environments."""
        for i in range(Self.num_envs):
            var force = self.force_mag if actions[i] == 1 else -self.force_mag

            var costheta = cos(self.theta[i])
            var sintheta = sin(self.theta[i])

            var temp = (
                force + self.polemass_length * self.theta_dot[i] * self.theta_dot[i] * sintheta
            ) / self.total_mass

            var thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
            )

            var xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            # Euler integration
            self.x[i] += self.tau * self.x_dot[i]
            self.x_dot[i] += self.tau * xacc
            self.theta[i] += self.tau * self.theta_dot[i]
            self.theta_dot[i] += self.tau * thetaacc
            self.steps[i] += 1

            # Check termination
            var x_out = self.x[i] < -self.x_threshold or self.x[i] > self.x_threshold
            var theta_out = self.theta[i] < -self.theta_threshold or self.theta[i] > self.theta_threshold
            var truncated = self.steps[i] >= self.max_steps

            dones[i] = x_out or theta_out or truncated
            rewards[i] = 0.0 if (x_out or theta_out) else 1.0

            # Auto-reset if done
            if dones[i]:
                self.x[i] = (random_float64() - 0.5) * 0.1
                self.x_dot[i] = (random_float64() - 0.5) * 0.1
                self.theta[i] = (random_float64() - 0.5) * 0.1
                self.theta_dot[i] = (random_float64() - 0.5) * 0.1
                self.steps[i] = 0


# =============================================================================
# GPU Implementation: Kernel-based vectorized CartPole
# =============================================================================


fn cartpole_step_kernel[
    dtype: DType,
    num_envs: Int,
    TPB: Int,
](
    # State (in/out)
    x: LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin],
    x_dot: LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin],
    theta: LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin],
    theta_dot: LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin],
    steps: LayoutTensor[DType.int32, Layout.row_major(num_envs), MutAnyOrigin],
    # Actions (in)
    actions: LayoutTensor[DType.int32, Layout.row_major(num_envs), ImmutAnyOrigin],
    # Outputs
    rewards: LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin],
    dones: LayoutTensor[DType.int32, Layout.row_major(num_envs), MutAnyOrigin],
    # Random seeds for reset (in)
    random_vals: LayoutTensor[dtype, Layout.row_major(num_envs * 4), ImmutAnyOrigin],
):
    """GPU kernel for stepping CartPole environments."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if i >= num_envs:
        return

    # Physics constants
    comptime gravity: Scalar[dtype] = 9.8
    comptime total_mass: Scalar[dtype] = 1.1
    comptime length: Scalar[dtype] = 0.5
    comptime polemass_length: Scalar[dtype] = 0.05
    comptime force_mag: Scalar[dtype] = 10.0
    comptime tau: Scalar[dtype] = 0.02
    comptime theta_threshold: Scalar[dtype] = 0.2095
    comptime x_threshold: Scalar[dtype] = 2.4
    comptime masspole: Scalar[dtype] = 0.1
    comptime max_steps: Int32 = 500

    # Load current state
    var x_i = rebind[Scalar[dtype]](x[i])
    var x_dot_i = rebind[Scalar[dtype]](x_dot[i])
    var theta_i = rebind[Scalar[dtype]](theta[i])
    var theta_dot_i = rebind[Scalar[dtype]](theta_dot[i])
    var steps_i = rebind[Int32](steps[i])

    # Determine force
    var action_i = rebind[Int32](actions[i])
    var force: Scalar[dtype] = force_mag if action_i == 1 else -force_mag

    # Physics
    var costheta = cos(Float64(theta_i))
    var sintheta = sin(Float64(theta_i))
    var costheta_s = Scalar[dtype](costheta)
    var sintheta_s = Scalar[dtype](sintheta)

    var temp = (force + polemass_length * theta_dot_i * theta_dot_i * sintheta_s) / total_mass

    var thetaacc = (gravity * sintheta_s - costheta_s * temp) / (
        length * (Scalar[dtype](4.0 / 3.0) - masspole * costheta_s * costheta_s / total_mass)
    )

    var xacc = temp - polemass_length * thetaacc * costheta_s / total_mass

    # Euler integration
    x_i += tau * x_dot_i
    x_dot_i += tau * xacc
    theta_i += tau * theta_dot_i
    theta_dot_i += tau * thetaacc
    steps_i += 1

    # Check termination
    var x_out = x_i < -x_threshold or x_i > x_threshold
    var theta_out = theta_i < -theta_threshold or theta_i > theta_threshold
    var truncated = steps_i >= max_steps
    var done = x_out or theta_out or truncated

    # Reward
    var reward: Scalar[dtype] = Scalar[dtype](0.0) if (x_out or theta_out) else Scalar[dtype](1.0)

    # Auto-reset if done
    if done:
        x_i = (rebind[Scalar[dtype]](random_vals[i * 4 + 0]) - Scalar[dtype](0.5)) * Scalar[dtype](0.1)
        x_dot_i = (rebind[Scalar[dtype]](random_vals[i * 4 + 1]) - Scalar[dtype](0.5)) * Scalar[dtype](0.1)
        theta_i = (rebind[Scalar[dtype]](random_vals[i * 4 + 2]) - Scalar[dtype](0.5)) * Scalar[dtype](0.1)
        theta_dot_i = (rebind[Scalar[dtype]](random_vals[i * 4 + 3]) - Scalar[dtype](0.5)) * Scalar[dtype](0.1)
        steps_i = 0

    # Store results
    x[i] = x_i
    x_dot[i] = x_dot_i
    theta[i] = theta_i
    theta_dot[i] = theta_dot_i
    steps[i] = steps_i
    rewards[i] = reward
    dones[i] = 1 if done else 0


# =============================================================================
# Benchmark Functions
# =============================================================================


fn benchmark_cpu[num_envs: Int](num_steps: Int) -> Float64:
    """Benchmark CPU vectorized CartPole."""
    var env = CPUVecCartPole[num_envs]()
    env.reset()

    var actions = InlineArray[Int, num_envs](fill=0)
    var rewards = InlineArray[Float64, num_envs](fill=0)
    var dones = InlineArray[Bool, num_envs](fill=False)

    # Warm up
    for _ in range(100):
        for j in range(num_envs):
            actions[j] = 1 if random_float64() > 0.5 else 0
        env.step(actions, rewards, dones)

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(num_steps):
        for j in range(num_envs):
            actions[j] = 1 if random_float64() > 0.5 else 0
        env.step(actions, rewards, dones)
    var end = perf_counter_ns()

    var total_env_steps = num_steps * num_envs
    var elapsed_sec = Float64(end - start) / 1e9
    return Float64(total_env_steps) / elapsed_sec  # Steps per second


fn benchmark_gpu[num_envs: Int](ctx: DeviceContext, num_steps: Int) raises -> Float64:
    """Benchmark GPU vectorized CartPole."""
    comptime dtype = DType.float32
    comptime TPB = 64

    # Allocate GPU buffers
    var x_gpu = ctx.enqueue_create_buffer[dtype](num_envs)
    var x_dot_gpu = ctx.enqueue_create_buffer[dtype](num_envs)
    var theta_gpu = ctx.enqueue_create_buffer[dtype](num_envs)
    var theta_dot_gpu = ctx.enqueue_create_buffer[dtype](num_envs)
    var steps_gpu = ctx.enqueue_create_buffer[DType.int32](num_envs)
    var actions_gpu = ctx.enqueue_create_buffer[DType.int32](num_envs)
    var rewards_gpu = ctx.enqueue_create_buffer[dtype](num_envs)
    var dones_gpu = ctx.enqueue_create_buffer[DType.int32](num_envs)
    var random_gpu = ctx.enqueue_create_buffer[dtype](num_envs * 4)

    # Initialize state
    with x_gpu.map_to_host() as host:
        for i in range(num_envs):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    with x_dot_gpu.map_to_host() as host:
        for i in range(num_envs):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    with theta_gpu.map_to_host() as host:
        for i in range(num_envs):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    with theta_dot_gpu.map_to_host() as host:
        for i in range(num_envs):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    steps_gpu.enqueue_fill(0)

    ctx.synchronize()

    # Create layout tensors
    var x_t = LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin](x_gpu)
    var x_dot_t = LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin](x_dot_gpu)
    var theta_t = LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin](theta_gpu)
    var theta_dot_t = LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin](theta_dot_gpu)
    var steps_t = LayoutTensor[DType.int32, Layout.row_major(num_envs), MutAnyOrigin](steps_gpu)
    var actions_t = LayoutTensor[DType.int32, Layout.row_major(num_envs), ImmutAnyOrigin](actions_gpu)
    var rewards_t = LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin](rewards_gpu)
    var dones_t = LayoutTensor[DType.int32, Layout.row_major(num_envs), MutAnyOrigin](dones_gpu)
    var random_t = LayoutTensor[dtype, Layout.row_major(num_envs * 4), ImmutAnyOrigin](random_gpu)

    comptime num_blocks = (num_envs + TPB - 1) // TPB
    comptime kernel = cartpole_step_kernel[dtype, num_envs, TPB]

    # Warm up
    for _ in range(100):
        # Generate random actions on CPU, copy to GPU
        with actions_gpu.map_to_host() as host:
            for i in range(num_envs):
                host[i] = Int32(1) if random_float64() > 0.5 else Int32(0)
        with random_gpu.map_to_host() as host:
            for i in range(num_envs * 4):
                host[i] = Scalar[dtype](random_float64())

        ctx.enqueue_function_checked[kernel, kernel](
            x_t, x_dot_t, theta_t, theta_dot_t, steps_t,
            actions_t, rewards_t, dones_t, random_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(num_steps):
        # Generate random actions and copy to GPU
        with actions_gpu.map_to_host() as host:
            for i in range(num_envs):
                host[i] = Int32(1) if random_float64() > 0.5 else Int32(0)
        with random_gpu.map_to_host() as host:
            for i in range(num_envs * 4):
                host[i] = Scalar[dtype](random_float64())

        ctx.enqueue_function_checked[kernel, kernel](
            x_t, x_dot_t, theta_t, theta_dot_t, steps_t,
            actions_t, rewards_t, dones_t, random_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    var total_env_steps = num_steps * num_envs
    var elapsed_sec = Float64(end - start) / 1e9
    return Float64(total_env_steps) / elapsed_sec


fn benchmark_gpu_no_transfer[num_envs: Int](ctx: DeviceContext, num_steps: Int) raises -> Float64:
    """Benchmark GPU vectorized CartPole WITHOUT per-step CPU-GPU transfers.

    This represents the best-case scenario where actions come from GPU (e.g., from
    a neural network forward pass) and random numbers are pre-generated.
    """
    comptime dtype = DType.float32
    comptime TPB = 64

    # Allocate GPU buffers
    var x_gpu = ctx.enqueue_create_buffer[dtype](num_envs)
    var x_dot_gpu = ctx.enqueue_create_buffer[dtype](num_envs)
    var theta_gpu = ctx.enqueue_create_buffer[dtype](num_envs)
    var theta_dot_gpu = ctx.enqueue_create_buffer[dtype](num_envs)
    var steps_gpu = ctx.enqueue_create_buffer[DType.int32](num_envs)
    var actions_gpu = ctx.enqueue_create_buffer[DType.int32](num_envs)
    var rewards_gpu = ctx.enqueue_create_buffer[dtype](num_envs)
    var dones_gpu = ctx.enqueue_create_buffer[DType.int32](num_envs)
    var random_gpu = ctx.enqueue_create_buffer[dtype](num_envs * 4)

    # Initialize all buffers once
    with x_gpu.map_to_host() as host:
        for i in range(num_envs):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    with x_dot_gpu.map_to_host() as host:
        for i in range(num_envs):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    with theta_gpu.map_to_host() as host:
        for i in range(num_envs):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    with theta_dot_gpu.map_to_host() as host:
        for i in range(num_envs):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    steps_gpu.enqueue_fill(0)

    # Pre-fill actions and random (simulating GPU-generated actions)
    with actions_gpu.map_to_host() as host:
        for i in range(num_envs):
            host[i] = Int32(1) if random_float64() > 0.5 else Int32(0)
    with random_gpu.map_to_host() as host:
        for i in range(num_envs * 4):
            host[i] = Scalar[dtype](random_float64())

    ctx.synchronize()

    # Create layout tensors
    var x_t = LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin](x_gpu)
    var x_dot_t = LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin](x_dot_gpu)
    var theta_t = LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin](theta_gpu)
    var theta_dot_t = LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin](theta_dot_gpu)
    var steps_t = LayoutTensor[DType.int32, Layout.row_major(num_envs), MutAnyOrigin](steps_gpu)
    var actions_t = LayoutTensor[DType.int32, Layout.row_major(num_envs), ImmutAnyOrigin](actions_gpu)
    var rewards_t = LayoutTensor[dtype, Layout.row_major(num_envs), MutAnyOrigin](rewards_gpu)
    var dones_t = LayoutTensor[DType.int32, Layout.row_major(num_envs), MutAnyOrigin](dones_gpu)
    var random_t = LayoutTensor[dtype, Layout.row_major(num_envs * 4), ImmutAnyOrigin](random_gpu)

    comptime num_blocks = (num_envs + TPB - 1) // TPB
    comptime kernel = cartpole_step_kernel[dtype, num_envs, TPB]

    # Warm up
    for _ in range(100):
        ctx.enqueue_function_checked[kernel, kernel](
            x_t, x_dot_t, theta_t, theta_dot_t, steps_t,
            actions_t, rewards_t, dones_t, random_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    # Benchmark - NO CPU-GPU transfers in the loop
    var start = perf_counter_ns()
    for _ in range(num_steps):
        ctx.enqueue_function_checked[kernel, kernel](
            x_t, x_dot_t, theta_t, theta_dot_t, steps_t,
            actions_t, rewards_t, dones_t, random_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    var total_env_steps = num_steps * num_envs
    var elapsed_sec = Float64(end - start) / 1e9
    return Float64(total_env_steps) / elapsed_sec


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("Vectorized Environment Benchmark: CPU vs GPU")
    print("=" * 70)
    print()

    var num_steps = 1000

    print("Configuration:")
    print("  Steps per benchmark: " + String(num_steps))
    print()

    # CPU benchmarks
    print("-" * 70)
    print("CPU Benchmarks (loop-based)")
    print("-" * 70)

    var cpu_64 = benchmark_cpu[64](num_steps)
    print("  64 envs:  " + String(Int(cpu_64)) + " steps/sec")

    var cpu_128 = benchmark_cpu[128](num_steps)
    print("  128 envs: " + String(Int(cpu_128)) + " steps/sec")

    var cpu_256 = benchmark_cpu[256](num_steps)
    print("  256 envs: " + String(Int(cpu_256)) + " steps/sec")

    var cpu_512 = benchmark_cpu[512](num_steps)
    print("  512 envs: " + String(Int(cpu_512)) + " steps/sec")

    var cpu_1024 = benchmark_cpu[1024](num_steps)
    print("  1024 envs: " + String(Int(cpu_1024)) + " steps/sec")

    print()

    # GPU benchmarks
    with DeviceContext() as ctx:
        print("-" * 70)
        print("GPU Benchmarks (WITH per-step CPU->GPU action transfer)")
        print("-" * 70)

        var gpu_64 = benchmark_gpu[64](ctx, num_steps)
        print("  64 envs:  " + String(Int(gpu_64)) + " steps/sec")

        var gpu_128 = benchmark_gpu[128](ctx, num_steps)
        print("  128 envs: " + String(Int(gpu_128)) + " steps/sec")

        var gpu_256 = benchmark_gpu[256](ctx, num_steps)
        print("  256 envs: " + String(Int(gpu_256)) + " steps/sec")

        var gpu_512 = benchmark_gpu[512](ctx, num_steps)
        print("  512 envs: " + String(Int(gpu_512)) + " steps/sec")

        var gpu_1024 = benchmark_gpu[1024](ctx, num_steps)
        print("  1024 envs: " + String(Int(gpu_1024)) + " steps/sec")

        print()
        print("-" * 70)
        print("GPU Benchmarks (NO per-step transfer - best case)")
        print("-" * 70)

        var gpu_nt_64 = benchmark_gpu_no_transfer[64](ctx, num_steps)
        print("  64 envs:  " + String(Int(gpu_nt_64)) + " steps/sec")

        var gpu_nt_128 = benchmark_gpu_no_transfer[128](ctx, num_steps)
        print("  128 envs: " + String(Int(gpu_nt_128)) + " steps/sec")

        var gpu_nt_256 = benchmark_gpu_no_transfer[256](ctx, num_steps)
        print("  256 envs: " + String(Int(gpu_nt_256)) + " steps/sec")

        var gpu_nt_512 = benchmark_gpu_no_transfer[512](ctx, num_steps)
        print("  512 envs: " + String(Int(gpu_nt_512)) + " steps/sec")

        var gpu_nt_1024 = benchmark_gpu_no_transfer[1024](ctx, num_steps)
        print("  1024 envs: " + String(Int(gpu_nt_1024)) + " steps/sec")

        print()
        print("-" * 70)
        print("Summary: GPU speedup vs CPU")
        print("-" * 70)
        print("  64 envs:  GPU=" + String(gpu_nt_64 / cpu_64)[:4] + "x (no transfer), " + String(gpu_64 / cpu_64)[:4] + "x (with transfer)")
        print("  128 envs: GPU=" + String(gpu_nt_128 / cpu_128)[:4] + "x (no transfer), " + String(gpu_128 / cpu_128)[:4] + "x (with transfer)")
        print("  256 envs: GPU=" + String(gpu_nt_256 / cpu_256)[:4] + "x (no transfer), " + String(gpu_256 / cpu_256)[:4] + "x (with transfer)")
        print("  512 envs: GPU=" + String(gpu_nt_512 / cpu_512)[:4] + "x (no transfer), " + String(gpu_512 / cpu_512)[:4] + "x (with transfer)")
        print("  1024 envs: GPU=" + String(gpu_nt_1024 / cpu_1024)[:4] + "x (no transfer), " + String(gpu_1024 / cpu_1024)[:4] + "x (with transfer)")

        print()
        print("=" * 70)
        print("Key insight: GPU wins when:")
        print("  1. Actions come from GPU (neural network forward pass)")
        print("  2. Random numbers generated on GPU (or pre-batched)")
        print("  3. Enough parallel envs to amortize kernel launch overhead")
        print("=" * 70)
