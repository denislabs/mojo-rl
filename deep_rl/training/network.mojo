"""Network wrapper for RL agents.

Wraps a stateless Model with its parameters, gradients, and optimizer state.
Designed for reinforcement learning where we need:
- Forward pass for inference (action selection)
- Forward pass with caching (for training)
- Backward pass and parameter updates
- Target network operations (soft_update, copy_params)

Usage:
    from deep_rl import seq, Linear, ReLU, Adam, Kaiming
    from deep_rl.training import Network

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

    var model: Self.MODEL
    var optimizer: Self.OPTIMIZER
    var initializer: Self.INITIALIZER
    var params: InlineArray[Scalar[dtype], Self.MODEL.PARAM_SIZE]
    var grads: InlineArray[Scalar[dtype], Self.MODEL.PARAM_SIZE]
    var optimizer_state: InlineArray[
        Scalar[dtype], Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM
    ]

    fn __init__(
        out self,
        model: Self.MODEL,
        optimizer: Self.OPTIMIZER,
        initializer: Self.INITIALIZER,
    ):
        """Initialize network with given model, optimizer, and initializer.

        Args:
            model: The model architecture.
            optimizer: The optimizer instance.
            initializer: The weight initializer.
        """
        self.model = model
        self.optimizer = optimizer
        self.initializer = initializer

        # Initialize params using the initializer
        self.params = self.initializer.init[
            Self.MODEL.PARAM_SIZE, Self.MODEL.IN_DIM, Self.MODEL.OUT_DIM
        ]()

        # Initialize grads to zero
        self.grads = InlineArray[Scalar[dtype], Self.MODEL.PARAM_SIZE](
            uninitialized=True
        )
        for i in range(Self.MODEL.PARAM_SIZE):
            self.grads[i] = 0

        # Initialize optimizer state to zero
        self.optimizer_state = InlineArray[
            Scalar[dtype],
            Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM,
        ](uninitialized=True)
        for i in range(Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM):
            self.optimizer_state[i] = 0

    # =========================================================================
    # CPU Forward Pass
    # =========================================================================

    fn forward[
        BATCH: Int
    ](
        self,
        input: InlineArray[Scalar[dtype], BATCH * Self.MODEL.IN_DIM],
        mut output: InlineArray[Scalar[dtype], BATCH * Self.MODEL.OUT_DIM],
    ):
        """Forward pass without caching (for inference/action selection).

        Args:
            input: Input tensor [BATCH * IN_DIM].
            output: Output tensor [BATCH * OUT_DIM] (written).
        """
        var input_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](input.unsafe_ptr())
        var output_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](output.unsafe_ptr())
        var params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.params.unsafe_ptr())

        self.model.forward[BATCH](
            input_tensor,
            output_tensor,
            params_tensor,
        )

    fn forward_with_cache[
        BATCH: Int
    ](
        self,
        input: InlineArray[Scalar[dtype], BATCH * Self.MODEL.IN_DIM],
        mut output: InlineArray[Scalar[dtype], BATCH * Self.MODEL.OUT_DIM],
        mut cache: InlineArray[Scalar[dtype], BATCH * Self.MODEL.CACHE_SIZE],
    ):
        """Forward pass with caching (for training).

        Args:
            input: Input tensor [BATCH * IN_DIM].
            output: Output tensor [BATCH * OUT_DIM] (written).
            cache: Cache tensor [BATCH * CACHE_SIZE] for backward pass (written).
        """
        var input_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](input.unsafe_ptr())
        var output_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](output.unsafe_ptr())
        var params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.params.unsafe_ptr())
        var cache_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ](cache.unsafe_ptr())

        self.model.forward[BATCH](
            input_tensor,
            output_tensor,
            params_tensor,
            cache_tensor,
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
        grad_output: InlineArray[Scalar[dtype], BATCH * Self.MODEL.OUT_DIM],
        mut grad_input: InlineArray[Scalar[dtype], BATCH * Self.MODEL.IN_DIM],
        cache: InlineArray[Scalar[dtype], BATCH * Self.MODEL.CACHE_SIZE],
    ):
        """Backward pass: compute gradients w.r.t. input and accumulate param grads.

        Call zero_grads() before this if you want fresh gradients.

        Args:
            grad_output: Gradient of loss w.r.t. output [BATCH * OUT_DIM].
            grad_input: Gradient of loss w.r.t. input [BATCH * IN_DIM] (written).
            cache: Cache from forward_with_cache [BATCH * CACHE_SIZE].
        """
        var grad_output_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](grad_output.unsafe_ptr())
        var grad_input_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](grad_input.unsafe_ptr())
        var params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.params.unsafe_ptr())
        var cache_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ](cache.unsafe_ptr())
        var grads_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.grads.unsafe_ptr())

        self.model.backward[BATCH](
            grad_output_tensor,
            grad_input_tensor,
            params_tensor,
            cache_tensor,
            grads_tensor,
        )

    fn backward_input[
        BATCH: Int
    ](
        mut self,
        grad_output: InlineArray[Scalar[dtype], BATCH * Self.MODEL.OUT_DIM],
        cache: InlineArray[Scalar[dtype], BATCH * Self.MODEL.CACHE_SIZE],
    ) -> InlineArray[Scalar[dtype], BATCH * Self.MODEL.IN_DIM]:
        """Backward pass that returns input gradients (for critic→actor chain).

        Use this when you need to chain gradients from one network's output
        to another network's input, such as in SAC/DDPG/TD3 actor updates
        where we need dQ/da from the critic to update the actor.

        Call zero_grads() before this if you want fresh gradients.

        Args:
            grad_output: Gradient of loss w.r.t. output [BATCH * OUT_DIM].
            cache: Cache from forward_with_cache [BATCH * CACHE_SIZE].

        Returns:
            Gradient of loss w.r.t. input [BATCH * IN_DIM].
        """
        var grad_input = InlineArray[Scalar[dtype], BATCH * Self.MODEL.IN_DIM](
            uninitialized=True
        )

        var grad_output_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](grad_output.unsafe_ptr())
        var grad_input_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](grad_input.unsafe_ptr())
        var params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.params.unsafe_ptr())
        var cache_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ](cache.unsafe_ptr())
        var grads_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.grads.unsafe_ptr())

        self.model.backward[BATCH](
            grad_output_tensor,
            grad_input_tensor,
            params_tensor,
            cache_tensor,
            grads_tensor,
        )

        return grad_input

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
    # GPU Forward Pass
    # =========================================================================

    fn forward_gpu[
        BATCH: Int
    ](
        self,
        ctx: DeviceContext,
        input_buf: DeviceBuffer[dtype],
        mut output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass without caching (for inference).

        Args:
            ctx: GPU device context.
            input_buf: Input buffer [BATCH * IN_DIM].
            output_buf: Output buffer [BATCH * OUT_DIM] (written).
            params_buf: Parameters buffer [PARAM_SIZE].
        """
        Self.MODEL.forward_gpu_no_cache[BATCH](
            ctx,
            output_buf,
            input_buf,
            params_buf,
        )

    fn forward_gpu_with_cache[
        BATCH: Int
    ](
        self,
        ctx: DeviceContext,
        input_buf: DeviceBuffer[dtype],
        mut output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        mut cache_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU forward pass with caching (for training).

        Args:
            ctx: GPU device context.
            input_buf: Input buffer [BATCH * IN_DIM].
            output_buf: Output buffer [BATCH * OUT_DIM] (written).
            params_buf: Parameters buffer [PARAM_SIZE].
            cache_buf: Cache buffer [BATCH * CACHE_SIZE] (written).
        """
        Self.MODEL.forward_gpu[BATCH](
            ctx,
            output_buf,
            input_buf,
            params_buf,
            cache_buf,
        )

    # =========================================================================
    # GPU Backward Pass
    # =========================================================================

    fn backward_gpu[
        BATCH: Int
    ](
        self,
        ctx: DeviceContext,
        grad_output_buf: DeviceBuffer[dtype],
        mut grad_input_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        mut grads_buf: DeviceBuffer[dtype],
    ) raises:
        """GPU backward pass.

        Args:
            ctx: GPU device context.
            grad_output_buf: Gradient w.r.t. output [BATCH * OUT_DIM].
            grad_input_buf: Gradient w.r.t. input [BATCH * IN_DIM] (written).
            params_buf: Parameters buffer [PARAM_SIZE].
            cache_buf: Cache from forward [BATCH * CACHE_SIZE].
            grads_buf: Parameter gradients [PARAM_SIZE] (accumulated).
        """
        Self.MODEL.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
        )

    fn backward_input_gpu[
        BATCH: Int
    ](
        self,
        ctx: DeviceContext,
        grad_output_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        cache_buf: DeviceBuffer[dtype],
        mut grads_buf: DeviceBuffer[dtype],
    ) raises -> DeviceBuffer[dtype]:
        """GPU backward pass that returns input gradients (for critic→actor chain).

        Use this when you need to chain gradients from one network's output
        to another network's input, such as in SAC/DDPG/TD3 actor updates
        where we need dQ/da from the critic to update the actor.

        Args:
            ctx: GPU device context.
            grad_output_buf: Gradient w.r.t. output [BATCH * OUT_DIM].
            params_buf: Parameters buffer [PARAM_SIZE].
            cache_buf: Cache from forward [BATCH * CACHE_SIZE].
            grads_buf: Parameter gradients [PARAM_SIZE] (accumulated).

        Returns:
            DeviceBuffer containing gradient w.r.t. input [BATCH * IN_DIM].
        """
        # Allocate buffer for input gradients
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * Self.MODEL.IN_DIM
        )

        Self.MODEL.backward_gpu[BATCH](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
        )

        return grad_input_buf

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
