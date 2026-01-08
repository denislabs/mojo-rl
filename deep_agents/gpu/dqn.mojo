"""GPU-Accelerated Deep Q-Network (DQN) Agent for discrete action spaces.

This is the GPU-accelerated version of DQN, using Metal/CUDA for neural network
computations while keeping the replay buffer on CPU (for random sampling).

Key GPU operations:
- Forward pass: Tiled matmul with fused bias + ReLU activation

Architecture: obs_dim -> hidden (relu) -> hidden (relu) -> num_actions (linear)

Reference: Mnih et al. "Playing Atari with Deep Reinforcement Learning" (2013)
           Mnih et al. "Human-level control through deep RL" (2015, Nature)

Example usage:
    from deep_agents.gpu import GPUDeepDQNAgent
    from envs import LunarLanderEnv

    with DeviceContext() as ctx:
        var env = LunarLanderEnv()
        var agent = GPUDeepDQNAgent[obs_dim=8, num_actions=4, hidden_dim=128]()
        var metrics = agent.train(ctx, env, num_episodes=500)
"""

from random import random_float64
from math import sqrt

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from deep_rl.cpu import ReplayBuffer
from core import TrainingMetrics, BoxDiscreteActionEnv

# Import GPU kernels from deep_rl.gpu (avoid duplication)
from deep_rl.gpu import (
    linear_forward_kernel,
    linear_forward_relu_kernel,
    linear_backward_dW_kernel,
    linear_backward_db_kernel,
    linear_backward_dx_kernel,
    adam_update_kernel,
    soft_update_kernel,
    relu_grad_mul_kernel,
)


# =============================================================================
# GPU Q-Network for DQN (CPU weights, GPU forward pass)
# =============================================================================


struct GPUQNetwork[
    obs_dim: Int,
    num_actions: Int,
    hidden1_dim: Int = 256,
    hidden2_dim: Int = 256,
    batch_size: Int = 64,
    dtype: DType = DType.float32,
]:
    """GPU-accelerated Q-Network.

    Architecture: obs_dim -> hidden1 (relu) -> hidden2 (relu) -> num_actions (linear).

    Weights are stored on CPU. GPU buffers are created on-demand for forward pass.
    """

    # Sizes
    comptime W1_SIZE = Self.obs_dim * Self.hidden1_dim
    comptime B1_SIZE = Self.hidden1_dim
    comptime W2_SIZE = Self.hidden1_dim * Self.hidden2_dim
    comptime B2_SIZE = Self.hidden2_dim
    comptime W3_SIZE = Self.hidden2_dim * Self.num_actions
    comptime B3_SIZE = Self.num_actions
    comptime H1_SIZE = Self.batch_size * Self.hidden1_dim
    comptime H2_SIZE = Self.batch_size * Self.hidden2_dim
    comptime OUT_SIZE = Self.batch_size * Self.num_actions
    comptime OBS_SIZE = Self.batch_size * Self.obs_dim

    # Tile size for GPU matmul
    comptime TILE: Int = 16
    comptime TPB: Int = 64

    # CPU storage for weights
    var W1: List[Scalar[Self.dtype]]
    var b1: List[Scalar[Self.dtype]]
    var W2: List[Scalar[Self.dtype]]
    var b2: List[Scalar[Self.dtype]]
    var W3: List[Scalar[Self.dtype]]
    var b3: List[Scalar[Self.dtype]]

    # Adam optimizer state (CPU)
    var m_W1: List[Scalar[Self.dtype]]
    var v_W1: List[Scalar[Self.dtype]]
    var m_b1: List[Scalar[Self.dtype]]
    var v_b1: List[Scalar[Self.dtype]]
    var m_W2: List[Scalar[Self.dtype]]
    var v_W2: List[Scalar[Self.dtype]]
    var m_b2: List[Scalar[Self.dtype]]
    var v_b2: List[Scalar[Self.dtype]]
    var m_W3: List[Scalar[Self.dtype]]
    var v_W3: List[Scalar[Self.dtype]]
    var m_b3: List[Scalar[Self.dtype]]
    var v_b3: List[Scalar[Self.dtype]]

    var adam_t: Int

    fn __init__(out self):
        """Initialize Q-Network with Xavier initialization."""
        self.adam_t = 0

        # Initialize weights with Xavier
        var std1 = sqrt(2.0 / Float64(Self.obs_dim + Self.hidden1_dim))
        var std2 = sqrt(2.0 / Float64(Self.hidden1_dim + Self.hidden2_dim))
        var std3 = sqrt(2.0 / Float64(Self.hidden2_dim + Self.num_actions))

        self.W1 = List[Scalar[Self.dtype]](capacity=Self.W1_SIZE)
        for i in range(Self.W1_SIZE):
            self.W1.append(
                Scalar[Self.dtype]((random_float64() * 2 - 1) * std1)
            )

        self.W2 = List[Scalar[Self.dtype]](capacity=Self.W2_SIZE)
        for i in range(Self.W2_SIZE):
            self.W2.append(
                Scalar[Self.dtype]((random_float64() * 2 - 1) * std2)
            )

        self.W3 = List[Scalar[Self.dtype]](capacity=Self.W3_SIZE)
        for i in range(Self.W3_SIZE):
            self.W3.append(
                Scalar[Self.dtype]((random_float64() * 2 - 1) * std3)
            )

        # Initialize biases to zero
        self.b1 = List[Scalar[Self.dtype]](capacity=Self.B1_SIZE)
        for i in range(Self.B1_SIZE):
            self.b1.append(Scalar[Self.dtype](0))

        self.b2 = List[Scalar[Self.dtype]](capacity=Self.B2_SIZE)
        for i in range(Self.B2_SIZE):
            self.b2.append(Scalar[Self.dtype](0))

        self.b3 = List[Scalar[Self.dtype]](capacity=Self.B3_SIZE)
        for i in range(Self.B3_SIZE):
            self.b3.append(Scalar[Self.dtype](0))

        # Initialize Adam state to zero
        self.m_W1 = List[Scalar[Self.dtype]](capacity=Self.W1_SIZE)
        self.v_W1 = List[Scalar[Self.dtype]](capacity=Self.W1_SIZE)
        for i in range(Self.W1_SIZE):
            self.m_W1.append(Scalar[Self.dtype](0))
            self.v_W1.append(Scalar[Self.dtype](0))

        self.m_b1 = List[Scalar[Self.dtype]](capacity=Self.B1_SIZE)
        self.v_b1 = List[Scalar[Self.dtype]](capacity=Self.B1_SIZE)
        for i in range(Self.B1_SIZE):
            self.m_b1.append(Scalar[Self.dtype](0))
            self.v_b1.append(Scalar[Self.dtype](0))

        self.m_W2 = List[Scalar[Self.dtype]](capacity=Self.W2_SIZE)
        self.v_W2 = List[Scalar[Self.dtype]](capacity=Self.W2_SIZE)
        for i in range(Self.W2_SIZE):
            self.m_W2.append(Scalar[Self.dtype](0))
            self.v_W2.append(Scalar[Self.dtype](0))

        self.m_b2 = List[Scalar[Self.dtype]](capacity=Self.B2_SIZE)
        self.v_b2 = List[Scalar[Self.dtype]](capacity=Self.B2_SIZE)
        for i in range(Self.B2_SIZE):
            self.m_b2.append(Scalar[Self.dtype](0))
            self.v_b2.append(Scalar[Self.dtype](0))

        self.m_W3 = List[Scalar[Self.dtype]](capacity=Self.W3_SIZE)
        self.v_W3 = List[Scalar[Self.dtype]](capacity=Self.W3_SIZE)
        for i in range(Self.W3_SIZE):
            self.m_W3.append(Scalar[Self.dtype](0))
            self.v_W3.append(Scalar[Self.dtype](0))

        self.m_b3 = List[Scalar[Self.dtype]](capacity=Self.B3_SIZE)
        self.v_b3 = List[Scalar[Self.dtype]](capacity=Self.B3_SIZE)
        for i in range(Self.B3_SIZE):
            self.m_b3.append(Scalar[Self.dtype](0))
            self.v_b3.append(Scalar[Self.dtype](0))

    fn forward_gpu(
        self,
        ctx: DeviceContext,
        obs: List[Scalar[Self.dtype]],
    ) raises -> List[Scalar[Self.dtype]]:
        """Forward pass on GPU: obs -> Q-values for all actions.

        Creates GPU buffers on-demand and copies weights from CPU.
        """
        var result = self.forward_gpu_with_cache(ctx, obs)
        return result[0].copy()

    fn forward_gpu_with_cache(
        self,
        ctx: DeviceContext,
        obs: List[Scalar[Self.dtype]],
    ) raises -> Tuple[
        List[Scalar[Self.dtype]],
        List[Scalar[Self.dtype]],
        List[Scalar[Self.dtype]],
    ]:
        """Forward pass on GPU returning Q-values and cached activations (h1, h2).

        Creates GPU buffers on-demand and copies weights from CPU.
        Returns: (q_values, h1, h2)
        """
        comptime BATCH = Self.batch_size
        comptime OBS = Self.obs_dim
        comptime H1 = Self.hidden1_dim
        comptime H2 = Self.hidden2_dim
        comptime OUT = Self.num_actions

        # Create GPU buffers
        var obs_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * OBS)
        var W1_gpu = ctx.enqueue_create_buffer[Self.dtype](OBS * H1)
        var b1_gpu = ctx.enqueue_create_buffer[Self.dtype](H1)
        var h1_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H1)
        var W2_gpu = ctx.enqueue_create_buffer[Self.dtype](H1 * H2)
        var b2_gpu = ctx.enqueue_create_buffer[Self.dtype](H2)
        var h2_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H2)
        var W3_gpu = ctx.enqueue_create_buffer[Self.dtype](H2 * OUT)
        var b3_gpu = ctx.enqueue_create_buffer[Self.dtype](OUT)
        var out_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * OUT)

        # Copy data to GPU
        with obs_gpu.map_to_host() as host:
            for i in range(len(obs)):
                host[i] = obs[i]
        with W1_gpu.map_to_host() as host:
            for i in range(Self.W1_SIZE):
                host[i] = self.W1[i]
        with b1_gpu.map_to_host() as host:
            for i in range(Self.B1_SIZE):
                host[i] = self.b1[i]
        with W2_gpu.map_to_host() as host:
            for i in range(Self.W2_SIZE):
                host[i] = self.W2[i]
        with b2_gpu.map_to_host() as host:
            for i in range(Self.B2_SIZE):
                host[i] = self.b2[i]
        with W3_gpu.map_to_host() as host:
            for i in range(Self.W3_SIZE):
                host[i] = self.W3[i]
        with b3_gpu.map_to_host() as host:
            for i in range(Self.B3_SIZE):
                host[i] = self.b3[i]

        # Initialize output buffers
        h1_gpu.enqueue_fill(0)
        h2_gpu.enqueue_fill(0)
        out_gpu.enqueue_fill(0)

        # Create layout tensors
        var obs_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OBS), ImmutAnyOrigin
        ](obs_gpu)
        var h1_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, H1), MutAnyOrigin
        ](h1_gpu)
        var h2_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, H2), MutAnyOrigin
        ](h2_gpu)
        var out_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
        ](out_gpu)
        var W1_t = LayoutTensor[
            Self.dtype, Layout.row_major(OBS, H1), ImmutAnyOrigin
        ](W1_gpu)
        var b1_t = LayoutTensor[
            Self.dtype, Layout.row_major(H1), ImmutAnyOrigin
        ](b1_gpu)
        var W2_t = LayoutTensor[
            Self.dtype, Layout.row_major(H1, H2), ImmutAnyOrigin
        ](W2_gpu)
        var b2_t = LayoutTensor[
            Self.dtype, Layout.row_major(H2), ImmutAnyOrigin
        ](b2_gpu)
        var W3_t = LayoutTensor[
            Self.dtype, Layout.row_major(H2, OUT), ImmutAnyOrigin
        ](W3_gpu)
        var b3_t = LayoutTensor[
            Self.dtype, Layout.row_major(OUT), ImmutAnyOrigin
        ](b3_gpu)

        # Grid dimensions
        comptime blocks_y1 = (BATCH + Self.TILE - 1) // Self.TILE
        comptime blocks_x1 = (H1 + Self.TILE - 1) // Self.TILE
        comptime blocks_y2 = (BATCH + Self.TILE - 1) // Self.TILE
        comptime blocks_x2 = (H2 + Self.TILE - 1) // Self.TILE
        comptime blocks_y3 = (BATCH + Self.TILE - 1) // Self.TILE
        comptime blocks_x3 = (OUT + Self.TILE - 1) // Self.TILE

        # Layer 1: ReLU(obs @ W1 + b1)
        comptime kernel1 = linear_forward_relu_kernel[
            Self.dtype, BATCH, OBS, H1, Self.TILE
        ]
        ctx.enqueue_function_checked[kernel1, kernel1](
            h1_t,
            obs_t,
            W1_t,
            b1_t,
            grid_dim=(blocks_x1, blocks_y1),
            block_dim=(Self.TILE, Self.TILE),
        )

        # Layer 2: ReLU(h1 @ W2 + b2)
        var h1_immut = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, H1), ImmutAnyOrigin
        ](h1_gpu)
        comptime kernel2 = linear_forward_relu_kernel[
            Self.dtype, BATCH, H1, H2, Self.TILE
        ]
        ctx.enqueue_function_checked[kernel2, kernel2](
            h2_t,
            h1_immut,
            W2_t,
            b2_t,
            grid_dim=(blocks_x2, blocks_y2),
            block_dim=(Self.TILE, Self.TILE),
        )

        # Layer 3: h2 @ W3 + b3 (no activation)
        var h2_immut = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, H2), ImmutAnyOrigin
        ](h2_gpu)
        comptime kernel3 = linear_forward_kernel[
            Self.dtype, BATCH, H2, OUT, Self.TILE
        ]
        ctx.enqueue_function_checked[kernel3, kernel3](
            out_t,
            h2_immut,
            W3_t,
            b3_t,
            grid_dim=(blocks_x3, blocks_y3),
            block_dim=(Self.TILE, Self.TILE),
        )

        ctx.synchronize()

        # Copy Q-values and activations back to CPU
        var q_values = List[Scalar[Self.dtype]](capacity=BATCH * OUT)
        with out_gpu.map_to_host() as host:
            for i in range(BATCH * OUT):
                q_values.append(host[i])

        var h1_out = List[Scalar[Self.dtype]](capacity=BATCH * H1)
        with h1_gpu.map_to_host() as host:
            for i in range(BATCH * H1):
                h1_out.append(host[i])

        var h2_out = List[Scalar[Self.dtype]](capacity=BATCH * H2)
        with h2_gpu.map_to_host() as host:
            for i in range(BATCH * H2):
                h2_out.append(host[i])

        return (q_values^, h1_out^, h2_out^)

    fn forward_cpu(
        self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
    ) -> InlineArray[Scalar[Self.dtype], Self.num_actions]:
        """CPU forward pass for single observation (action selection)."""
        # Layer 1: ReLU(obs @ W1 + b1)
        var h1 = InlineArray[Scalar[Self.dtype], Self.hidden1_dim](fill=0)
        for j in range(Self.hidden1_dim):
            var sum_val: Scalar[Self.dtype] = self.b1[j]
            for k in range(Self.obs_dim):
                sum_val += obs[k] * self.W1[k * Self.hidden1_dim + j]
            h1[j] = sum_val if sum_val > 0 else 0

        # Layer 2: ReLU(h1 @ W2 + b2)
        var h2 = InlineArray[Scalar[Self.dtype], Self.hidden2_dim](fill=0)
        for j in range(Self.hidden2_dim):
            var sum_val: Scalar[Self.dtype] = self.b2[j]
            for k in range(Self.hidden1_dim):
                sum_val += h1[k] * self.W2[k * Self.hidden2_dim + j]
            h2[j] = sum_val if sum_val > 0 else 0

        # Layer 3: h2 @ W3 + b3
        var q = InlineArray[Scalar[Self.dtype], Self.num_actions](fill=0)
        for j in range(Self.num_actions):
            var sum_val: Scalar[Self.dtype] = self.b3[j]
            for k in range(Self.hidden2_dim):
                sum_val += h2[k] * self.W3[k * Self.num_actions + j]
            q[j] = sum_val

        return q^

    fn backward_gpu(
        mut self,
        ctx: DeviceContext,
        obs: List[Scalar[Self.dtype]],
        h1: List[Scalar[Self.dtype]],
        h2: List[Scalar[Self.dtype]],
        dq: List[Scalar[Self.dtype]],
    ) raises -> Tuple[
        List[Scalar[Self.dtype]],
        List[Scalar[Self.dtype]],
        List[Scalar[Self.dtype]],
        List[Scalar[Self.dtype]],
        List[Scalar[Self.dtype]],
        List[Scalar[Self.dtype]],
    ]:
        """GPU backward pass. Returns gradients for all weights."""
        comptime BATCH = Self.batch_size
        comptime OBS = Self.obs_dim
        comptime H1 = Self.hidden1_dim
        comptime H2 = Self.hidden2_dim
        comptime OUT = Self.num_actions

        # Create GPU buffers
        var obs_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * OBS)
        var h1_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H1)
        var h2_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H2)
        var dout_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * OUT)
        var W2_gpu = ctx.enqueue_create_buffer[Self.dtype](H1 * H2)
        var W3_gpu = ctx.enqueue_create_buffer[Self.dtype](H2 * OUT)

        var dW1_gpu = ctx.enqueue_create_buffer[Self.dtype](OBS * H1)
        var db1_gpu = ctx.enqueue_create_buffer[Self.dtype](H1)
        var dW2_gpu = ctx.enqueue_create_buffer[Self.dtype](H1 * H2)
        var db2_gpu = ctx.enqueue_create_buffer[Self.dtype](H2)
        var dW3_gpu = ctx.enqueue_create_buffer[Self.dtype](H2 * OUT)
        var db3_gpu = ctx.enqueue_create_buffer[Self.dtype](OUT)
        var dh1_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H1)
        var dh2_gpu = ctx.enqueue_create_buffer[Self.dtype](BATCH * H2)

        # Copy inputs to GPU
        with obs_gpu.map_to_host() as host:
            for i in range(len(obs)):
                host[i] = obs[i]
        with h1_gpu.map_to_host() as host:
            for i in range(len(h1)):
                host[i] = h1[i]
        with h2_gpu.map_to_host() as host:
            for i in range(len(h2)):
                host[i] = h2[i]
        with dout_gpu.map_to_host() as host:
            for i in range(len(dq)):
                host[i] = dq[i]
        with W2_gpu.map_to_host() as host:
            for i in range(Self.W2_SIZE):
                host[i] = self.W2[i]
        with W3_gpu.map_to_host() as host:
            for i in range(Self.W3_SIZE):
                host[i] = self.W3[i]

        # Initialize gradient buffers to 0
        dW1_gpu.enqueue_fill(0)
        db1_gpu.enqueue_fill(0)
        dW2_gpu.enqueue_fill(0)
        db2_gpu.enqueue_fill(0)
        dW3_gpu.enqueue_fill(0)
        db3_gpu.enqueue_fill(0)
        dh1_gpu.enqueue_fill(0)
        dh2_gpu.enqueue_fill(0)

        # Create layout tensors
        var obs_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OBS), ImmutAnyOrigin
        ](obs_gpu)
        var h1_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, H1), ImmutAnyOrigin
        ](h1_gpu)
        var h2_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, H2), ImmutAnyOrigin
        ](h2_gpu)
        var dout_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin
        ](dout_gpu)
        var W2_t = LayoutTensor[
            Self.dtype, Layout.row_major(H1, H2), ImmutAnyOrigin
        ](W2_gpu)
        var W3_t = LayoutTensor[
            Self.dtype, Layout.row_major(H2, OUT), ImmutAnyOrigin
        ](W3_gpu)

        var dW1_t = LayoutTensor[
            Self.dtype, Layout.row_major(OBS, H1), MutAnyOrigin
        ](dW1_gpu)
        var db1_t = LayoutTensor[
            Self.dtype, Layout.row_major(H1), MutAnyOrigin
        ](db1_gpu)
        var dW2_t = LayoutTensor[
            Self.dtype, Layout.row_major(H1, H2), MutAnyOrigin
        ](dW2_gpu)
        var db2_t = LayoutTensor[
            Self.dtype, Layout.row_major(H2), MutAnyOrigin
        ](db2_gpu)
        var dW3_t = LayoutTensor[
            Self.dtype, Layout.row_major(H2, OUT), MutAnyOrigin
        ](dW3_gpu)
        var db3_t = LayoutTensor[
            Self.dtype, Layout.row_major(OUT), MutAnyOrigin
        ](db3_gpu)
        var dh1_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, H1), MutAnyOrigin
        ](dh1_gpu)
        var dh2_t = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, H2), MutAnyOrigin
        ](dh2_gpu)

        # Grid dimensions
        comptime blocks_dW3_y = (H2 + Self.TILE - 1) // Self.TILE
        comptime blocks_dW3_x = (OUT + Self.TILE - 1) // Self.TILE
        comptime blocks_dh2_y = (BATCH + Self.TILE - 1) // Self.TILE
        comptime blocks_dh2_x = (H2 + Self.TILE - 1) // Self.TILE
        comptime blocks_dW2_y = (H1 + Self.TILE - 1) // Self.TILE
        comptime blocks_dW2_x = (H2 + Self.TILE - 1) // Self.TILE
        comptime blocks_dh1_y = (BATCH + Self.TILE - 1) // Self.TILE
        comptime blocks_dh1_x = (H1 + Self.TILE - 1) // Self.TILE
        comptime blocks_dW1_y = (OBS + Self.TILE - 1) // Self.TILE
        comptime blocks_dW1_x = (H1 + Self.TILE - 1) // Self.TILE

        # Layer 3 backward: dW3 = h2.T @ dq
        comptime kernel_dW3 = linear_backward_dW_kernel[
            Self.dtype, BATCH, H2, OUT, Self.TILE
        ]
        ctx.enqueue_function_checked[kernel_dW3, kernel_dW3](
            dW3_t,
            h2_t,
            dout_t,
            grid_dim=(blocks_dW3_x, blocks_dW3_y),
            block_dim=(Self.TILE, Self.TILE),
        )

        # db3 = sum(dq, axis=0)
        comptime kernel_db3 = linear_backward_db_kernel[
            Self.dtype, BATCH, OUT, Self.TPB
        ]
        ctx.enqueue_function_checked[kernel_db3, kernel_db3](
            db3_t,
            dout_t,
            grid_dim=(OUT,),
            block_dim=(Self.TPB,),
        )

        # dh2_pre = dq @ W3.T
        comptime kernel_dx3 = linear_backward_dx_kernel[
            Self.dtype, BATCH, H2, OUT, Self.TILE
        ]
        ctx.enqueue_function_checked[kernel_dx3, kernel_dx3](
            dh2_t,
            dout_t,
            W3_t,
            grid_dim=(blocks_dh2_x, blocks_dh2_y),
            block_dim=(Self.TILE, Self.TILE),
        )

        # Apply ReLU gradient: dh2 = dh2_pre * (h2 > 0)
        comptime blocks_relu2 = (BATCH * H2 + Self.TPB - 1) // Self.TPB
        var dh2_flat = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH * H2), MutAnyOrigin
        ](dh2_gpu)
        var dh2_in_flat = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH * H2), ImmutAnyOrigin
        ](dh2_gpu)
        var h2_flat = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH * H2), ImmutAnyOrigin
        ](h2_gpu)
        comptime kernel_relu2 = relu_grad_mul_kernel[
            Self.dtype, BATCH * H2, Self.TPB
        ]
        ctx.enqueue_function_checked[kernel_relu2, kernel_relu2](
            dh2_flat,
            dh2_in_flat,
            h2_flat,
            grid_dim=(blocks_relu2,),
            block_dim=(Self.TPB,),
        )

        # Layer 2 backward: dW2 = h1.T @ dh2
        var dh2_immut = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, H2), ImmutAnyOrigin
        ](dh2_gpu)
        comptime kernel_dW2 = linear_backward_dW_kernel[
            Self.dtype, BATCH, H1, H2, Self.TILE
        ]
        ctx.enqueue_function_checked[kernel_dW2, kernel_dW2](
            dW2_t,
            h1_t,
            dh2_immut,
            grid_dim=(blocks_dW2_x, blocks_dW2_y),
            block_dim=(Self.TILE, Self.TILE),
        )

        # db2 = sum(dh2, axis=0)
        comptime kernel_db2 = linear_backward_db_kernel[
            Self.dtype, BATCH, H2, Self.TPB
        ]
        ctx.enqueue_function_checked[kernel_db2, kernel_db2](
            db2_t,
            dh2_immut,
            grid_dim=(H2,),
            block_dim=(Self.TPB,),
        )

        # dh1_pre = dh2 @ W2.T
        comptime kernel_dx2 = linear_backward_dx_kernel[
            Self.dtype, BATCH, H1, H2, Self.TILE
        ]
        ctx.enqueue_function_checked[kernel_dx2, kernel_dx2](
            dh1_t,
            dh2_immut,
            W2_t,
            grid_dim=(blocks_dh1_x, blocks_dh1_y),
            block_dim=(Self.TILE, Self.TILE),
        )

        # Apply ReLU gradient: dh1 = dh1_pre * (h1 > 0)
        comptime blocks_relu1 = (BATCH * H1 + Self.TPB - 1) // Self.TPB
        var dh1_flat = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH * H1), MutAnyOrigin
        ](dh1_gpu)
        var dh1_in_flat = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH * H1), ImmutAnyOrigin
        ](dh1_gpu)
        var h1_flat = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH * H1), ImmutAnyOrigin
        ](h1_gpu)
        comptime kernel_relu1 = relu_grad_mul_kernel[
            Self.dtype, BATCH * H1, Self.TPB
        ]
        ctx.enqueue_function_checked[kernel_relu1, kernel_relu1](
            dh1_flat,
            dh1_in_flat,
            h1_flat,
            grid_dim=(blocks_relu1,),
            block_dim=(Self.TPB,),
        )

        # Layer 1 backward: dW1 = obs.T @ dh1
        var dh1_immut = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, H1), ImmutAnyOrigin
        ](dh1_gpu)
        comptime kernel_dW1 = linear_backward_dW_kernel[
            Self.dtype, BATCH, OBS, H1, Self.TILE
        ]
        ctx.enqueue_function_checked[kernel_dW1, kernel_dW1](
            dW1_t,
            obs_t,
            dh1_immut,
            grid_dim=(blocks_dW1_x, blocks_dW1_y),
            block_dim=(Self.TILE, Self.TILE),
        )

        # db1 = sum(dh1, axis=0)
        comptime kernel_db1 = linear_backward_db_kernel[
            Self.dtype, BATCH, H1, Self.TPB
        ]
        ctx.enqueue_function_checked[kernel_db1, kernel_db1](
            db1_t,
            dh1_immut,
            grid_dim=(H1,),
            block_dim=(Self.TPB,),
        )

        ctx.synchronize()

        # Copy gradients back to CPU
        var dW1 = List[Scalar[Self.dtype]](capacity=Self.W1_SIZE)
        var db1 = List[Scalar[Self.dtype]](capacity=Self.B1_SIZE)
        var dW2 = List[Scalar[Self.dtype]](capacity=Self.W2_SIZE)
        var db2 = List[Scalar[Self.dtype]](capacity=Self.B2_SIZE)
        var dW3 = List[Scalar[Self.dtype]](capacity=Self.W3_SIZE)
        var db3 = List[Scalar[Self.dtype]](capacity=Self.B3_SIZE)

        with dW1_gpu.map_to_host() as host:
            for i in range(Self.W1_SIZE):
                dW1.append(host[i])
        with db1_gpu.map_to_host() as host:
            for i in range(Self.B1_SIZE):
                db1.append(host[i])
        with dW2_gpu.map_to_host() as host:
            for i in range(Self.W2_SIZE):
                dW2.append(host[i])
        with db2_gpu.map_to_host() as host:
            for i in range(Self.B2_SIZE):
                db2.append(host[i])
        with dW3_gpu.map_to_host() as host:
            for i in range(Self.W3_SIZE):
                dW3.append(host[i])
        with db3_gpu.map_to_host() as host:
            for i in range(Self.B3_SIZE):
                db3.append(host[i])

        return (dW1^, db1^, dW2^, db2^, dW3^, db3^)

    fn update_adam(
        mut self,
        dW1: List[Scalar[Self.dtype]],
        db1: List[Scalar[Self.dtype]],
        dW2: List[Scalar[Self.dtype]],
        db2: List[Scalar[Self.dtype]],
        dW3: List[Scalar[Self.dtype]],
        db3: List[Scalar[Self.dtype]],
        lr: Scalar[Self.dtype] = 0.001,
        beta1: Scalar[Self.dtype] = 0.9,
        beta2: Scalar[Self.dtype] = 0.999,
        eps: Scalar[Self.dtype] = 1e-8,
    ):
        """Adam optimizer update on CPU."""
        self.adam_t += 1
        var bias_correction1 = Scalar[Self.dtype](
            1.0 - (Float64(beta1) ** self.adam_t)
        )
        var bias_correction2 = Scalar[Self.dtype](
            1.0 - (Float64(beta2) ** self.adam_t)
        )

        # Update W1
        for i in range(Self.W1_SIZE):
            self.m_W1[i] = beta1 * self.m_W1[i] + (1 - beta1) * dW1[i]
            self.v_W1[i] = beta2 * self.v_W1[i] + (1 - beta2) * dW1[i] * dW1[i]
            var m_hat = self.m_W1[i] / bias_correction1
            var v_hat = self.v_W1[i] / bias_correction2
            self.W1[i] -= lr * m_hat / Scalar[Self.dtype](sqrt(Float64(v_hat)) + Float64(eps))

        # Update b1
        for i in range(Self.B1_SIZE):
            self.m_b1[i] = beta1 * self.m_b1[i] + (1 - beta1) * db1[i]
            self.v_b1[i] = beta2 * self.v_b1[i] + (1 - beta2) * db1[i] * db1[i]
            var m_hat = self.m_b1[i] / bias_correction1
            var v_hat = self.v_b1[i] / bias_correction2
            self.b1[i] -= lr * m_hat / Scalar[Self.dtype](sqrt(Float64(v_hat)) + Float64(eps))

        # Update W2
        for i in range(Self.W2_SIZE):
            self.m_W2[i] = beta1 * self.m_W2[i] + (1 - beta1) * dW2[i]
            self.v_W2[i] = beta2 * self.v_W2[i] + (1 - beta2) * dW2[i] * dW2[i]
            var m_hat = self.m_W2[i] / bias_correction1
            var v_hat = self.v_W2[i] / bias_correction2
            self.W2[i] -= lr * m_hat / Scalar[Self.dtype](sqrt(Float64(v_hat)) + Float64(eps))

        # Update b2
        for i in range(Self.B2_SIZE):
            self.m_b2[i] = beta1 * self.m_b2[i] + (1 - beta1) * db2[i]
            self.v_b2[i] = beta2 * self.v_b2[i] + (1 - beta2) * db2[i] * db2[i]
            var m_hat = self.m_b2[i] / bias_correction1
            var v_hat = self.v_b2[i] / bias_correction2
            self.b2[i] -= lr * m_hat / Scalar[Self.dtype](sqrt(Float64(v_hat)) + Float64(eps))

        # Update W3
        for i in range(Self.W3_SIZE):
            self.m_W3[i] = beta1 * self.m_W3[i] + (1 - beta1) * dW3[i]
            self.v_W3[i] = beta2 * self.v_W3[i] + (1 - beta2) * dW3[i] * dW3[i]
            var m_hat = self.m_W3[i] / bias_correction1
            var v_hat = self.v_W3[i] / bias_correction2
            self.W3[i] -= lr * m_hat / Scalar[Self.dtype](sqrt(Float64(v_hat)) + Float64(eps))

        # Update b3
        for i in range(Self.B3_SIZE):
            self.m_b3[i] = beta1 * self.m_b3[i] + (1 - beta1) * db3[i]
            self.v_b3[i] = beta2 * self.v_b3[i] + (1 - beta2) * db3[i] * db3[i]
            var m_hat = self.m_b3[i] / bias_correction1
            var v_hat = self.v_b3[i] / bias_correction2
            self.b3[i] -= lr * m_hat / Scalar[Self.dtype](sqrt(Float64(v_hat)) + Float64(eps))

    fn soft_update_from(mut self, source: Self, tau: Scalar[Self.dtype]):
        """Soft update from source network: theta = tau * source + (1-tau) * theta."""
        for i in range(Self.W1_SIZE):
            self.W1[i] = tau * source.W1[i] + (1 - tau) * self.W1[i]
        for i in range(Self.B1_SIZE):
            self.b1[i] = tau * source.b1[i] + (1 - tau) * self.b1[i]
        for i in range(Self.W2_SIZE):
            self.W2[i] = tau * source.W2[i] + (1 - tau) * self.W2[i]
        for i in range(Self.B2_SIZE):
            self.b2[i] = tau * source.b2[i] + (1 - tau) * self.b2[i]
        for i in range(Self.W3_SIZE):
            self.W3[i] = tau * source.W3[i] + (1 - tau) * self.W3[i]
        for i in range(Self.B3_SIZE):
            self.b3[i] = tau * source.b3[i] + (1 - tau) * self.b3[i]

    fn copy_from(mut self, source: Self):
        """Hard copy weights from source network."""
        self.soft_update_from(source, Scalar[Self.dtype](1.0))

    fn num_parameters(self) -> Int:
        """Total number of parameters."""
        return (
            Self.W1_SIZE
            + Self.B1_SIZE
            + Self.W2_SIZE
            + Self.B2_SIZE
            + Self.W3_SIZE
            + Self.B3_SIZE
        )

    fn print_info(self, name: String = "GPUQNetwork"):
        """Print network architecture."""
        print(name + ":")
        print(
            "  Architecture: "
            + String(Self.obs_dim)
            + " -> "
            + String(Self.hidden1_dim)
            + " (relu)"
            + " -> "
            + String(Self.hidden2_dim)
            + " (relu)"
            + " -> "
            + String(Self.num_actions)
        )
        print("  Parameters: " + String(self.num_parameters()))


# =============================================================================
# GPU Deep DQN Agent
# =============================================================================


struct GPUDeepDQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    dtype: DType = DType.float32,
    double_dqn: Bool = True,
]:
    """GPU-Accelerated Deep DQN Agent.

    Parameters:
        obs_dim: Observation dimension (e.g., 8 for LunarLander).
        num_actions: Number of discrete actions (e.g., 4 for LunarLander).
        hidden_dim: Hidden layer size for Q-network.
        buffer_capacity: Replay buffer capacity.
        batch_size: Training batch size.
        dtype: Data type for computations.
        double_dqn: If True, use Double DQN (recommended).
    """

    var q_network: GPUQNetwork[
        Self.obs_dim,
        Self.num_actions,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.batch_size,
        Self.dtype,
    ]
    var target_network: GPUQNetwork[
        Self.obs_dim,
        Self.num_actions,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.batch_size,
        Self.dtype,
    ]

    var buffer: ReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, 1, DType.float64
    ]

    var gamma: Scalar[Self.dtype]
    var tau: Scalar[Self.dtype]
    var lr: Scalar[Self.dtype]
    var epsilon: Scalar[Self.dtype]
    var epsilon_min: Scalar[Self.dtype]
    var epsilon_decay: Scalar[Self.dtype]

    var total_steps: Int
    var total_episodes: Int

    fn __init__(
        out self,
        gamma: Scalar[Self.dtype] = 0.99,
        tau: Scalar[Self.dtype] = 0.005,
        lr: Scalar[Self.dtype] = 0.0005,
        epsilon: Scalar[Self.dtype] = 1.0,
        epsilon_min: Scalar[Self.dtype] = 0.01,
        epsilon_decay: Scalar[Self.dtype] = 0.995,
    ):
        """Initialize GPU Deep DQN agent."""
        self.q_network = GPUQNetwork[
            Self.obs_dim,
            Self.num_actions,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.batch_size,
            Self.dtype,
        ]()

        self.target_network = GPUQNetwork[
            Self.obs_dim,
            Self.num_actions,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.batch_size,
            Self.dtype,
        ]()

        # Copy Q-network weights to target network
        self.target_network.copy_from(self.q_network)

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, DType.float64
        ]()

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0
        self.total_episodes = 0

    fn select_action(
        self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        training: Bool = True,
    ) -> Int:
        """Select action using epsilon-greedy policy (CPU forward pass)."""
        if training and random_float64() < Float64(self.epsilon):
            return (
                Int(random_float64() * Float64(Self.num_actions))
                % Self.num_actions
            )

        var q_values = self.q_network.forward_cpu(obs)

        var best_action = 0
        var best_q = q_values[0]
        for a in range(1, Self.num_actions):
            if q_values[a] > best_q:
                best_q = q_values[a]
                best_action = a

        return best_action

    fn store_transition(
        mut self,
        obs: InlineArray[Scalar[DType.float64], Self.obs_dim],
        action: Int,
        reward: Scalar[DType.float64],
        next_obs: InlineArray[Scalar[DType.float64], Self.obs_dim],
        done: Bool,
    ):
        """Store a transition in the CPU replay buffer."""
        var action_arr = InlineArray[Scalar[DType.float64], 1](fill=0)
        action_arr[0] = Scalar[DType.float64](action)
        self.buffer.add(obs, action_arr, reward, next_obs, done)
        self.total_steps += 1

    fn train_step(mut self, ctx: DeviceContext) raises -> Scalar[Self.dtype]:
        """Perform one training step."""
        if not self.buffer.is_ready[Self.batch_size]():
            return 0.0

        # Sample batch from CPU buffer
        var batch_obs = InlineArray[
            Scalar[DType.float64], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_actions = InlineArray[Scalar[DType.float64], Self.batch_size](
            fill=0
        )
        var batch_rewards = InlineArray[Scalar[DType.float64], Self.batch_size](
            fill=0
        )
        var batch_next_obs = InlineArray[
            Scalar[DType.float64], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_dones = InlineArray[Scalar[DType.float64], Self.batch_size](
            fill=0
        )

        var batch_actions_arr = InlineArray[
            Scalar[DType.float64], Self.batch_size
        ](fill=0)
        self.buffer.sample[Self.batch_size](
            batch_obs,
            batch_actions_arr,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )
        for i in range(Self.batch_size):
            batch_actions[i] = batch_actions_arr[i]

        # Convert to dtype
        var obs_list = List[Scalar[Self.dtype]](
            capacity=Self.batch_size * Self.obs_dim
        )
        for i in range(Self.batch_size * Self.obs_dim):
            obs_list.append(Scalar[Self.dtype](batch_obs[i]))

        var next_obs_list = List[Scalar[Self.dtype]](
            capacity=Self.batch_size * Self.obs_dim
        )
        for i in range(Self.batch_size * Self.obs_dim):
            next_obs_list.append(Scalar[Self.dtype](batch_next_obs[i]))

        # Forward pass for current Q-values (on GPU) with cached activations
        var fwd_result = self.q_network.forward_gpu_with_cache(ctx, obs_list)
        var current_q = fwd_result[0].copy()
        var h1_cache = fwd_result[1].copy()
        var h2_cache = fwd_result[2].copy()

        # Forward pass for target Q-values (on GPU)
        var target_q = self.target_network.forward_gpu(ctx, next_obs_list)

        var online_next_q = List[Scalar[Self.dtype]]()

        @parameter
        if Self.double_dqn:
            online_next_q = self.q_network.forward_gpu(ctx, next_obs_list)

        # Compute max next Q-values
        var max_next_q = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )

        @parameter
        if Self.double_dqn:
            for i in range(Self.batch_size):
                var best_action = 0
                var best_online_q = online_next_q[i * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = online_next_q[i * Self.num_actions + a]
                    if q > best_online_q:
                        best_online_q = q
                        best_action = a
                max_next_q[i] = target_q[i * Self.num_actions + best_action]
        else:
            for i in range(Self.batch_size):
                var max_q = target_q[i * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = target_q[i * Self.num_actions + a]
                    if q > max_q:
                        max_q = q
                max_next_q[i] = max_q

        # Compute targets
        var target_values = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )
        for i in range(Self.batch_size):
            target_values[i] = (
                Scalar[Self.dtype](batch_rewards[i])
                + self.gamma
                * (1.0 - Scalar[Self.dtype](batch_dones[i]))
                * max_next_q[i]
            )

        # Compute loss and gradients
        var loss: Scalar[Self.dtype] = 0.0
        var dq = List[Scalar[Self.dtype]](
            capacity=Self.batch_size * Self.num_actions
        )
        for i in range(Self.batch_size * Self.num_actions):
            dq.append(Scalar[Self.dtype](0))

        var batch_size_scalar = Scalar[Self.dtype](Self.batch_size)

        for i in range(Self.batch_size):
            var action_idx = Int(batch_actions[i])
            var q_idx = i * Self.num_actions + action_idx
            var td_error = current_q[q_idx] - target_values[i]
            loss += td_error * td_error
            dq[q_idx] = 2.0 * td_error / batch_size_scalar

        loss /= batch_size_scalar

        # Backward pass on GPU
        var grads = self.q_network.backward_gpu(ctx, obs_list, h1_cache, h2_cache, dq)

        # Update parameters on CPU
        self.q_network.update_adam(
            grads[0], grads[1], grads[2], grads[3], grads[4], grads[5], self.lr
        )

        # Soft update target network
        self.target_network.soft_update_from(self.q_network, self.tau)

        return loss

    fn decay_epsilon(mut self):
        """Decay exploration rate."""
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    fn print_info(self):
        """Print agent information."""

        @parameter
        if Self.double_dqn:
            print("GPU Deep Double DQN Agent:")
        else:
            print("GPU Deep DQN Agent:")
        print("  Obs dim: " + String(Self.obs_dim))
        print("  Num actions: " + String(Self.num_actions))
        print("  Hidden dim: " + String(Self.hidden_dim))
        print("  Buffer capacity: " + String(Self.buffer_capacity))
        print("  Batch size: " + String(Self.batch_size))
        print("  Double DQN: " + String(Self.double_dqn))
        print("  Gamma: " + String(self.gamma)[:6])
        print("  Tau: " + String(self.tau)[:6])
        print("  LR: " + String(self.lr)[:8])
        print("  Epsilon: " + String(self.epsilon)[:5])
        self.q_network.print_info("  Q-Network")

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        ctx: DeviceContext,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 1000,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        verbose: Bool = True,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        """Train the GPU Deep DQN agent on a discrete action environment."""
        var metrics = TrainingMetrics(
            algorithm_name="GPU Deep DQN",
            environment_name=environment_name,
        )

        if verbose:
            print("=" * 60)
            print("GPU Deep DQN Training on " + environment_name)
            print("=" * 60)
            self.print_info()
            print("-" * 60)

        # Warmup phase
        if verbose:
            print(
                "Warmup: collecting "
                + String(warmup_steps)
                + " random steps..."
            )

        var warmup_done = 0
        while warmup_done < warmup_steps:
            var obs_list = env.reset_obs_list()
            var done = False

            while not done and warmup_done < warmup_steps:
                var action = (
                    Int(random_float64() * Float64(Self.num_actions))
                    % Self.num_actions
                )
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                var obs = _list_to_inline[Self.obs_dim, DType.float64](obs_list)
                var next_obs = _list_to_inline[Self.obs_dim, DType.float64](
                    step_result[0]
                )

                self.store_transition(
                    obs, action, Scalar[DType.float64](reward), next_obs, done
                )
                obs_list = env.get_obs_list()
                warmup_done += 1

        if verbose:
            print("Warmup complete. Buffer size: " + String(self.buffer.len()))
            print("-" * 60)

        # Training loop
        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var episode_reward: Float64 = 0.0
            var steps = 0
            var done = False

            while not done and steps < max_steps_per_episode:
                var obs_dtype = _list_to_inline_dtype[Self.obs_dim, Self.dtype](
                    obs_list
                )
                var action = self.select_action(obs_dtype, training=True)
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                var obs = _list_to_inline[Self.obs_dim, DType.float64](obs_list)
                var next_obs = _list_to_inline[Self.obs_dim, DType.float64](
                    step_result[0]
                )
                self.store_transition(
                    obs, action, Scalar[DType.float64](reward), next_obs, done
                )

                if steps % train_every == 0:
                    _ = self.train_step(ctx)

                episode_reward += reward
                obs_list = env.get_obs_list()
                steps += 1

            metrics.log_episode(
                episode, episode_reward, steps, Float64(self.epsilon)
            )
            self.total_episodes += 1
            self.decay_epsilon()

            if verbose and (episode + 1) % print_every == 0:
                var start_idx = max(0, len(metrics.episodes) - print_every)
                var sum_reward: Float64 = 0.0
                for j in range(start_idx, len(metrics.episodes)):
                    sum_reward += metrics.episodes[j].total_reward
                var avg_reward = sum_reward / Float64(
                    len(metrics.episodes) - start_idx
                )
                print(
                    "Episode "
                    + String(episode + 1)
                    + " | Avg Reward: "
                    + String(avg_reward)[:8]
                    + " | Steps: "
                    + String(steps)
                    + " | Epsilon: "
                    + String(self.epsilon)[:5]
                )

        if verbose:
            print("-" * 60)
            print("Training complete!")
            var start_idx = max(0, len(metrics.episodes) - 100)
            var sum_reward: Float64 = 0.0
            for j in range(start_idx, len(metrics.episodes)):
                sum_reward += metrics.episodes[j].total_reward
            var final_avg = sum_reward / Float64(
                len(metrics.episodes) - start_idx
            )
            print("Final avg reward (last 100): " + String(final_avg)[:8])

        return metrics^

    fn evaluate[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
    ) -> Float64:
        """Evaluate the trained agent using greedy policy."""
        var total_reward: Float64 = 0.0

        for ep in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var episode_reward: Float64 = 0.0
            var done = False
            var steps = 0

            while not done and steps < max_steps_per_episode:
                var obs = _list_to_inline_dtype[Self.obs_dim, Self.dtype](
                    obs_list
                )
                var action = self.select_action(obs, training=False)
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                episode_reward += reward
                obs_list = step_result[0].copy()
                steps += 1

            total_reward += episode_reward

            if verbose:
                print(
                    "  Eval episode "
                    + String(ep + 1)
                    + ": "
                    + String(episode_reward)[:10]
                    + " (steps: "
                    + String(steps)
                    + ")"
                )

        return total_reward / Float64(num_episodes)


# ============================================================================
# Helper functions
# ============================================================================


fn _list_to_inline[
    size: Int, dtype: DType = DType.float64
](obs_list: List[Float64]) -> InlineArray[Scalar[dtype], size]:
    """Convert List[Float64] to InlineArray."""
    var obs = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        if i < len(obs_list):
            obs[i] = Scalar[dtype](obs_list[i])
    return obs^


fn _list_to_inline_dtype[
    size: Int, dtype: DType
](obs_list: List[Float64]) -> InlineArray[Scalar[dtype], size]:
    """Convert List[Float64] to InlineArray with specific dtype."""
    var obs = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        if i < len(obs_list):
            obs[i] = Scalar[dtype](obs_list[i])
    return obs^


fn max(a: Int, b: Int) -> Int:
    """Return maximum of two integers."""
    return a if a > b else b
