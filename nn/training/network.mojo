"""Network wrapper for RL agents.

Wraps a stateless Model with its parameters, gradients, and optimizer state.
Designed for reinforcement learning where we need:
- Forward pass for inference (action selection)
- Forward pass with caching (for training)
- Backward pass and parameter updates
- Target network operations (soft_update, copy_params)

Usage:
    from nn import seq, Linear, ReLU, Adam, Kaiming
    from nn.training import Network

    # Define Q-network: obs -> hidden -> hidden -> num_actions
    var q_model = seq(
        Linear[4, 64](), ReLU[64](),
        Linear[64, 64](), ReLU[64](),
        Linear[64, 2](),
    )

    # Create online and target networks
    var online = Network(q_model, Adam(lr=0.001), Kaiming())
    var target = Network(q_model, Adam(lr=0.001), Kaiming())

    # Initialize target with same weights as online
    target.copy_params_from(online)

    # Forward pass for action selection
    online.forward[batch_size](obs, q_values)

    # Training step
    online.forward_with_cache[batch_size](obs, q_values, cache)
    # ... compute TD targets and grad_output ...
    online.zero_grads()
    online.backward[batch_size](grad_output, grad_input, cache)
    online.update()

    # Soft update target network
    target.soft_update_from(online, tau=0.005)
"""

from ..model import Model
from ..optimizer import Optimizer
from ..initializer import Initializer, Xavier
from ..constants import dtype
from ..checkpoint import (
    write_checkpoint_header,
    write_float_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_float_section,
    save_checkpoint_file,
)

from layout import Layout, LayoutTensor
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer


struct Network[
    MODEL: Model,
    OPTIMIZER: Optimizer,
    INITIALIZER: Initializer = Xavier,
]:
    """Wraps a stateless Model with parameters and optimizer state.

    This struct manages the state needed for a neural network:
    - params: The network weights
    - grads: Gradients for backpropagation
    - optimizer_state: Optimizer-specific state (e.g., Adam moments)

    Parameters:
        MODEL: The model architecture (implements Model trait).
        OPTIMIZER: The optimizer to use (implements Optimizer trait).
        INITIALIZER: Weight initialization strategy (default: Xavier).
    """

    # Expose model dimensions for external use
    comptime IN_DIM: Int = Self.MODEL.IN_DIM
    comptime OUT_DIM: Int = Self.MODEL.OUT_DIM
    comptime PARAM_SIZE: Int = Self.MODEL.PARAM_SIZE
    comptime CACHE_SIZE: Int = Self.MODEL.CACHE_SIZE
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.MODEL.WORKSPACE_SIZE_PER_SAMPLE

    var optimizer: Self.OPTIMIZER
    var initializer: Self.INITIALIZER
    # Heap-allocated arrays to support large hidden dimensions
    var params: List[Scalar[dtype]]
    var grads: List[Scalar[dtype]]
    var optimizer_state: List[Scalar[dtype]]

    fn __init__(
        out self,
        optimizer: Self.OPTIMIZER,
        initializer: Self.INITIALIZER,
    ):
        """Initialize network with given model, optimizer, and initializer.

        Args:
            optimizer: The optimizer instance.
            initializer: The weight initializer.
        """

        self.optimizer = optimizer
        self.initializer = initializer

        # Initialize params using the initializer (returns heap-allocated List)
        self.params = self.initializer.init[
            Self.MODEL.PARAM_SIZE, Self.MODEL.IN_DIM, Self.MODEL.OUT_DIM
        ]()

        # Initialize grads to zero
        self.grads = List[Scalar[dtype]](capacity=Self.MODEL.PARAM_SIZE)
        for _ in range(Self.MODEL.PARAM_SIZE):
            self.grads.append(Scalar[dtype](0))

        # Initialize optimizer state to zero
        comptime STATE_SIZE = Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM
        self.optimizer_state = List[Scalar[dtype]](capacity=STATE_SIZE)
        for _ in range(STATE_SIZE):
            self.optimizer_state.append(Scalar[dtype](0))

    # =========================================================================
    # CPU Forward Pass
    # =========================================================================

    fn forward[
        BATCH: Int
    ](
        self,
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
    ):
        """Forward pass without caching (for inference/action selection).

        Args:
            input: Input tensor [BATCH * IN_DIM].
            output: Output tensor [BATCH * OUT_DIM] (written).

        """

        var params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.params.unsafe_ptr())

        Self.MODEL.forward[BATCH](
            input,
            output,
            params_tensor,
        )

    fn forward_with_cache[
        BATCH: Int
    ](
        self,
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass with caching (for training).

        Args:
            input: Input tensor [BATCH * IN_DIM].
            output: Output tensor [BATCH * OUT_DIM] (written).
            cache: Cache tensor [BATCH * CACHE_SIZE] for backward pass (written).

        """

        var params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.params.unsafe_ptr())

        Self.MODEL.forward[BATCH](
            input,
            output,
            params_tensor,
            cache,
        )

    # =========================================================================
    # CPU Backward Pass
    # =========================================================================

    fn zero_grads(mut self):
        """Zero all gradients before backward pass."""
        for i in range(Self.MODEL.PARAM_SIZE):
            self.grads[i] = 0

    fn backward[
        BATCH: Int
    ](
        mut self,
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Backward pass: compute gradients w.r.t. input and accumulate param grads.

        Call zero_grads() before this if you want fresh gradients.

        Args:
            grad_output: Gradient of loss w.r.t. output [BATCH * OUT_DIM].
            grad_input: Gradient of loss w.r.t. input [BATCH * IN_DIM] (written).
            cache: Cache from forward_with_cache [BATCH * CACHE_SIZE].
        """

        var params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.params.unsafe_ptr())
        var grads_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.grads.unsafe_ptr())

        Self.MODEL.backward[BATCH](
            grad_output,
            grad_input,
            params_tensor,
            cache,
            grads_tensor,
        )

    # =========================================================================
    # CPU Optimizer Step
    # =========================================================================

    fn update(mut self):
        """Update parameters using the optimizer."""
        var params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.params.unsafe_ptr())
        var grads_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.grads.unsafe_ptr())
        var state_tensor = LayoutTensor[
            dtype,
            Layout.row_major(
                Self.MODEL.PARAM_SIZE, Self.OPTIMIZER.STATE_PER_PARAM
            ),
            MutAnyOrigin,
        ](self.optimizer_state.unsafe_ptr())

        self.optimizer.step[Self.MODEL.PARAM_SIZE](
            params_tensor, grads_tensor, state_tensor
        )

    # =========================================================================
    # Target Network Operations
    # =========================================================================

    fn copy_params_from(mut self, source: Self):
        """Copy all parameters from source network (hard update).

        Used to initialize target network with online network weights.

        Args:
            source: The network to copy parameters from.
        """
        for i in range(Self.MODEL.PARAM_SIZE):
            self.params[i] = source.params[i]

    fn soft_update_from(mut self, source: Self, tau: Float64):
        """Soft update parameters: self = tau * source + (1 - tau) * self.

        Used for target network updates in DQN, DDPG, TD3, SAC.

        Args:
            source: The network to blend from (usually online network).
            tau: Interpolation factor (typically 0.001 to 0.01).
        """
        var tau_scalar = Scalar[dtype](tau)
        var one_minus_tau = Scalar[dtype](1.0 - tau)
        for i in range(Self.MODEL.PARAM_SIZE):
            self.params[i] = (
                tau_scalar * source.params[i] + one_minus_tau * self.params[i]
            )

    # =========================================================================
    # GPU Forward/Backward with Workspace (avoids internal allocation)
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        input_buf: DeviceBuffer[dtype],
        mut output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass without caching, using pre-allocated workspace.

        Use this to avoid GPU memory leaks from repeated internal allocations.
        Workspace size must be BATCH * MODEL.WORKSPACE_SIZE_PER_SAMPLE.

        Args:
            ctx: GPU device context.
            input_buf: Input buffer [BATCH * IN_DIM].
            output_buf: Output buffer [BATCH * OUT_DIM] (written).
            params_buf: Parameters buffer [PARAM_SIZE].
            workspace_buf: Pre-allocated workspace [BATCH * WORKSPACE_SIZE_PER_SAMPLE].
        """
        Self.MODEL.forward_gpu_no_cache[BATCH](
            ctx,
            output_buf,
            input_buf,
            params_buf,
            workspace_buf,
        )

    @staticmethod
    fn forward_gpu_with_cache[
        BATCH: Int
    ](
        ctx: DeviceContext,
        input_buf: DeviceBuffer[dtype],
        mut output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        mut cache_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass with caching, using pre-allocated workspace.

        Use this to avoid GPU memory leaks from repeated internal allocations.
        Workspace size must be BATCH * MODEL.WORKSPACE_SIZE_PER_SAMPLE.

        Args:
            ctx: GPU device context.
            input_buf: Input buffer [BATCH * IN_DIM].
            output_buf: Output buffer [BATCH * OUT_DIM] (written).
            params_buf: Parameters buffer [PARAM_SIZE].
            cache_buf: Cache buffer [BATCH * CACHE_SIZE] (written).
            workspace_buf: Pre-allocated workspace [BATCH * WORKSPACE_SIZE_PER_SAMPLE].
        """
        Self.MODEL.forward_gpu[BATCH](
            ctx,
            output_buf,
            input_buf,
            params_buf,
            cache_buf,
            workspace_buf,
        )

    @staticmethod
    fn backward_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        grad_output_buf: DeviceBuffer[dtype],
        mut grad_input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        mut grads_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU backward pass using pre-allocated workspace.

        Use this to avoid GPU memory leaks from repeated internal allocations.
        Workspace size must be BATCH * MODEL.WORKSPACE_SIZE_PER_SAMPLE.

        Args:
            ctx: GPU device context.
            grad_output_buf: Gradient w.r.t. output [BATCH * OUT_DIM].
            grad_input_buf: Gradient w.r.t. input [BATCH * IN_DIM] (written).
            params_buf: Parameters buffer [PARAM_SIZE].
            cache_buf: Cache from forward [BATCH * CACHE_SIZE].
            grads_buf: Parameter gradients [PARAM_SIZE] (accumulated).
            workspace_buf: Pre-allocated workspace [BATCH * WORKSPACE_SIZE_PER_SAMPLE].
        """
        Self.MODEL.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
            workspace_buf,
        )

    # =========================================================================
    # GPU Optimizer Step
    # =========================================================================

    fn update_gpu(
        mut self,
        ctx: DeviceContext,
        mut params_buf: DeviceBuffer[dtype],
        grads_buf: DeviceBuffer[dtype],
        mut state_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU optimizer step.

        Args:
            ctx: GPU device context.
            params_buf: Parameters buffer [PARAM_SIZE] (updated in-place).
            grads_buf: Gradients buffer [PARAM_SIZE].
            state_buf: Optimizer state buffer [PARAM_SIZE * STATE_PER_PARAM].
        """
        self.optimizer.step_gpu[Self.MODEL.PARAM_SIZE](
            ctx, params_buf, grads_buf, state_buf
        )

    # =========================================================================
    # GPU Buffer Management
    # =========================================================================

    fn copy_params_to_device(
        self,
        ctx: DeviceContext,
        mut params_buf: DeviceBuffer[dtype],
    ) raises:
        """Copy CPU parameters to GPU buffer.

        Args:
            ctx: GPU device context.
            params_buf: Device buffer to copy to [PARAM_SIZE].
        """
        var params_host = ctx.enqueue_create_host_buffer[dtype](
            Self.MODEL.PARAM_SIZE
        )
        for i in range(Self.MODEL.PARAM_SIZE):
            params_host[i] = self.params[i]
        ctx.enqueue_copy(params_buf, params_host)

    fn copy_params_from_device(
        mut self,
        ctx: DeviceContext,
        params_buf: DeviceBuffer[dtype],
    ) raises:
        """Copy GPU parameters back to CPU.

        Args:
            ctx: GPU device context.
            params_buf: Device buffer to copy from [PARAM_SIZE].
        """
        var params_host = ctx.enqueue_create_host_buffer[dtype](
            Self.MODEL.PARAM_SIZE
        )
        ctx.enqueue_copy(params_host, params_buf)
        ctx.synchronize()
        for i in range(Self.MODEL.PARAM_SIZE):
            self.params[i] = params_host[i]

    fn copy_state_to_device(
        self,
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[dtype],
    ) raises:
        """Copy CPU optimizer state to GPU buffer.

        Args:
            ctx: GPU device context.
            state_buf: Device buffer to copy to [PARAM_SIZE * STATE_PER_PARAM].
        """
        comptime STATE_SIZE = Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM
        var state_host = ctx.enqueue_create_host_buffer[dtype](STATE_SIZE)
        for i in range(STATE_SIZE):
            state_host[i] = self.optimizer_state[i]
        ctx.enqueue_copy(state_buf, state_host)

    fn copy_state_from_device(
        mut self,
        ctx: DeviceContext,
        state_buf: DeviceBuffer[dtype],
    ) raises:
        """Copy GPU optimizer state back to CPU.

        Args:
            ctx: GPU device context.
            state_buf: Device buffer to copy from [PARAM_SIZE * STATE_PER_PARAM].
        """
        comptime STATE_SIZE = Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM
        var state_host = ctx.enqueue_create_host_buffer[dtype](STATE_SIZE)
        ctx.enqueue_copy(state_host, state_buf)
        ctx.synchronize()
        for i in range(STATE_SIZE):
            self.optimizer_state[i] = state_host[i]

    # =========================================================================
    # Checkpoint Save/Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        """Save network parameters and optimizer state to a checkpoint file.

        Args:
            filepath: Path to save the checkpoint file.

        Example:
            network.save_checkpoint("model.ckpt")
        """
        comptime PARAM_SIZE = Self.MODEL.PARAM_SIZE
        comptime STATE_SIZE = Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM

        var content = write_checkpoint_header("network", PARAM_SIZE, STATE_SIZE)

        # Write params section (manual for List compatibility)
        content += "params:\n"
        for i in range(PARAM_SIZE):
            content += String(Float64(self.params[i])) + "\n"

        # Write optimizer_state section
        content += "optimizer_state:\n"
        for i in range(STATE_SIZE):
            content += String(Float64(self.optimizer_state[i])) + "\n"

        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        """Load network parameters and optimizer state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.

        Example:
            network.load_checkpoint("model.ckpt")
        """
        comptime PARAM_SIZE = Self.MODEL.PARAM_SIZE
        comptime STATE_SIZE = Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM

        var content = read_checkpoint_file(filepath)
        var header = parse_checkpoint_header(content)

        # Validate sizes match
        if header.param_size != PARAM_SIZE:
            print(
                "Warning: checkpoint param_size ("
                + String(header.param_size)
                + ") != network PARAM_SIZE ("
                + String(PARAM_SIZE)
                + ")"
            )

        # Load parameters
        var loaded_params = read_float_section[PARAM_SIZE](content, "params:")
        for i in range(PARAM_SIZE):
            self.params[i] = loaded_params[i]

        # Load optimizer state
        var loaded_state = read_float_section[STATE_SIZE](
            content, "optimizer_state:"
        )
        for i in range(STATE_SIZE):
            self.optimizer_state[i] = loaded_state[i]
