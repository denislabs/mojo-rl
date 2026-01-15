from ..model import Model
from ..optimizer import Optimizer
from ..loss import LossFunction
from ..constants import dtype
from ..initializer import Initializer, Xavier

from layout import Layout, LayoutTensor
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer


struct TrainResult:
    """Result of training containing loss history."""

    var final_loss: Float64
    var epochs_trained: Int

    fn __init__(out self, final_loss: Float64, epochs_trained: Int):
        self.final_loss = final_loss
        self.epochs_trained = epochs_trained


struct Trainer[
    MODEL: Model,
    OPTIMIZER: Optimizer,
    LOSS_FUNCTION: LossFunction,
    INITIALIZER: Initializer = Xavier,
]:
    """Training configuration for neural networks.

    The Trainer manages model parameters and gradients externally.
    Parameters are initialized using the specified Initializer.

    Usage:
        # Default Xavier initialization
        var trainer = Trainer[MODEL, OPTIMIZER, LOSS_FUNCTION](
            model, optimizer, loss_function
        )

        # Kaiming initialization for ReLU networks
        var trainer = Trainer[MODEL, OPTIMIZER, LOSS_FUNCTION, Kaiming](
            model, optimizer, loss_function, Kaiming()
        )

    Parameters:
        MODEL: The model to train (stateless).
        OPTIMIZER: The optimizer to use.
        LOSS_FUNCTION: The loss function to use.
        INITIALIZER: The initializer for parameters (default: Xavier).
    """

    var epochs: Int
    var print_every: Int
    var model: Self.MODEL
    var optimizer: Self.OPTIMIZER
    var loss_function: Self.LOSS_FUNCTION
    var initializer: Self.INITIALIZER
    # Use heap-allocated List instead of stack-allocated InlineArray
    # to avoid stack overflow with large models
    var params: List[Scalar[dtype]]
    var grads: List[Scalar[dtype]]
    var optimizer_state: List[Scalar[dtype]]

    fn __init__(
        out self,
        model: Self.MODEL,
        optimizer: Self.OPTIMIZER,
        loss_function: Self.LOSS_FUNCTION,
        initializer: Self.INITIALIZER,
        epochs: Int = 100,
        print_every: Int = 10,
    ):
        """Initialize trainer with the specified initializer.

        Args:
            model: The model to train.
            optimizer: The optimizer to use.
            loss_function: The loss function to use.
            initializer: The weight initializer.
            epochs: Number of training epochs.
            print_every: Print loss every N epochs (0 to disable).
        """
        self.epochs = epochs
        self.print_every = print_every
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.initializer = initializer

        # Initialize params using the initializer (heap-allocated)
        var init_params = self.initializer.init[
            Self.MODEL.PARAM_SIZE, Self.MODEL.IN_DIM, Self.MODEL.OUT_DIM
        ]()
        self.params = List[Scalar[dtype]](capacity=Self.MODEL.PARAM_SIZE)
        for i in range(Self.MODEL.PARAM_SIZE):
            self.params.append(init_params[i])

        # Initialize grads to zero (heap-allocated)
        self.grads = List[Scalar[dtype]](capacity=Self.MODEL.PARAM_SIZE)
        for i in range(Self.MODEL.PARAM_SIZE):
            self.grads.append(0)

        # Initialize optimizer state to zero (heap-allocated)
        comptime STATE_SIZE = Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM
        self.optimizer_state = List[Scalar[dtype]](capacity=STATE_SIZE)
        for i in range(STATE_SIZE):
            self.optimizer_state.append(0)

    fn train[
        BATCH: Int
    ](
        mut self,
        input: InlineArray[Scalar[dtype], BATCH * Self.MODEL.IN_DIM],
        target: InlineArray[Scalar[dtype], BATCH * Self.MODEL.OUT_DIM],
    ) -> TrainResult:
        """Train the model for the configured number of epochs.

        Returns:
            TrainResult with final loss and epochs trained.
        """
        # Allocate storage and create LayoutTensor views
        var output_storage = InlineArray[
            Scalar[dtype], BATCH * Self.MODEL.OUT_DIM
        ](uninitialized=True)
        var grad_output_storage = InlineArray[
            Scalar[dtype], BATCH * Self.MODEL.OUT_DIM
        ](uninitialized=True)
        var grad_input_storage = InlineArray[
            Scalar[dtype], BATCH * Self.MODEL.IN_DIM
        ](uninitialized=True)
        var cache_storage = InlineArray[
            Scalar[dtype], BATCH * Self.MODEL.CACHE_SIZE
        ](uninitialized=True)

        # Create LayoutTensor views using unsafe_ptr()
        var input_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](input.unsafe_ptr())
        var output_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](output_storage.unsafe_ptr())
        var params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.params.unsafe_ptr())
        var grads_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.grads.unsafe_ptr())
        var cache_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.CACHE_SIZE), MutAnyOrigin
        ](cache_storage.unsafe_ptr())
        var grad_output_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](grad_output_storage.unsafe_ptr())
        var grad_input_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](grad_input_storage.unsafe_ptr())
        var optimizer_state_tensor = LayoutTensor[
            dtype,
            Layout.row_major(
                Self.MODEL.PARAM_SIZE, Self.OPTIMIZER.STATE_PER_PARAM
            ),
            MutAnyOrigin,
        ](self.optimizer_state.unsafe_ptr())

        var final_loss: Float64 = 0.0

        for epoch in range(self.epochs):
            # Forward pass (with cache and params)
            self.model.forward[BATCH](
                input_tensor,
                output_tensor,
                params_tensor,
                cache_tensor,
            )

            # Compute loss and gradient (still uses InlineArray for now)
            var loss = self.loss_function.forward[BATCH * Self.MODEL.OUT_DIM](
                output_storage, target
            )
            self.loss_function.backward[BATCH * Self.MODEL.OUT_DIM](
                output_storage, target, grad_output_storage
            )

            # Zero gradients before backward pass
            for i in range(Self.MODEL.PARAM_SIZE):
                self.grads[i] = 0

            # Backward pass (with cache, params, and grads)
            self.model.backward[BATCH](
                grad_output_tensor,
                grad_input_tensor,
                params_tensor,
                cache_tensor,
                grads_tensor,
            )

            # Update parameters using optimizer
            self.optimizer.step[Self.MODEL.PARAM_SIZE](
                params_tensor, grads_tensor, optimizer_state_tensor
            )

            final_loss = loss

            if self.print_every > 0 and epoch % self.print_every == 0:
                print("Epoch " + String(epoch) + " - Loss: " + String(loss))

        return TrainResult(final_loss, self.epochs)

    fn evaluate[
        BATCH: Int
    ](
        self,
        input: InlineArray[Scalar[dtype], BATCH * Self.MODEL.IN_DIM],
        target: InlineArray[Scalar[dtype], BATCH * Self.MODEL.OUT_DIM],
    ) -> Float64:
        """Evaluate the model on the given input and target (no cache allocation).
        """
        var output_storage = InlineArray[
            Scalar[dtype], BATCH * Self.MODEL.OUT_DIM
        ](uninitialized=True)

        # Create LayoutTensor views using unsafe_ptr()
        var input_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.IN_DIM), MutAnyOrigin
        ](input.unsafe_ptr())
        var output_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.MODEL.OUT_DIM), MutAnyOrigin
        ](output_storage.unsafe_ptr())
        var params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.MODEL.PARAM_SIZE), MutAnyOrigin
        ](self.params.unsafe_ptr())

        # Use forward - no cache needed for evaluation
        self.model.forward[BATCH](
            input_tensor,
            output_tensor,
            params_tensor,
        )

        return self.loss_function.forward[BATCH * Self.MODEL.OUT_DIM](
            output_storage, target
        )

    fn train_gpu[
        BATCH: Int
    ](
        mut self,
        ctx: DeviceContext,
        input: InlineArray[Scalar[dtype], BATCH * Self.MODEL.IN_DIM],
        target: InlineArray[Scalar[dtype], BATCH * Self.MODEL.OUT_DIM],
    ) raises -> TrainResult:
        """Train the model on GPU for the configured number of epochs.

        Args:
            ctx: GPU device context.
            input: Input data [BATCH * IN_DIM].
            target: Target data [BATCH * OUT_DIM].

        Returns:
            TrainResult with final loss and epochs trained.
        """
        # Dimension constants
        comptime IN_SIZE = BATCH * Self.MODEL.IN_DIM
        comptime OUT_SIZE = BATCH * Self.MODEL.OUT_DIM
        comptime PARAM_SIZE = Self.MODEL.PARAM_SIZE
        comptime CACHE_SIZE = BATCH * Self.MODEL.CACHE_SIZE
        comptime STATE_SIZE = PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM
        comptime WORKSPACE_SIZE = BATCH * Self.MODEL.WORKSPACE_SIZE_PER_SAMPLE

        # Create host buffers for input/target and copy data
        var input_host = ctx.enqueue_create_host_buffer[dtype](IN_SIZE)
        var target_host = ctx.enqueue_create_host_buffer[dtype](OUT_SIZE)
        for i in range(IN_SIZE):
            input_host[i] = input[i]
        for i in range(OUT_SIZE):
            target_host[i] = target[i]

        # Create host buffer for params and copy current params
        var params_host = ctx.enqueue_create_host_buffer[dtype](PARAM_SIZE)
        for i in range(PARAM_SIZE):
            params_host[i] = self.params[i]

        # Create host buffer for optimizer state and copy
        var state_host = ctx.enqueue_create_host_buffer[dtype](STATE_SIZE)
        for i in range(STATE_SIZE):
            state_host[i] = self.optimizer_state[i]

        # Create device buffers
        var input_buf = ctx.enqueue_create_buffer[dtype](IN_SIZE)
        var target_buf = ctx.enqueue_create_buffer[dtype](OUT_SIZE)
        var output_buf = ctx.enqueue_create_buffer[dtype](OUT_SIZE)
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](CACHE_SIZE)
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](OUT_SIZE)
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](IN_SIZE)
        var state_buf = ctx.enqueue_create_buffer[dtype](STATE_SIZE)
        var loss_buf = ctx.enqueue_create_buffer[dtype](1)
        # Pre-allocate workspace buffer (avoids allocation on every forward/backward)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](
            WORKSPACE_SIZE if WORKSPACE_SIZE > 0 else 1
        )

        # Copy input, target, params, and state to device
        ctx.enqueue_copy(input_buf, input_host)
        ctx.enqueue_copy(target_buf, target_host)
        ctx.enqueue_copy(params_buf, params_host)
        ctx.enqueue_copy(state_buf, state_host)

        # Host buffer for reading loss back
        var loss_host = ctx.enqueue_create_host_buffer[dtype](1)

        var final_loss: Float64 = 0.0

        for epoch in range(self.epochs):
            # Zero gradients on GPU
            ctx.enqueue_memset(grads_buf, 0)

            # Forward pass (using workspace to avoid internal allocation)
            Self.MODEL.forward_gpu_ws[BATCH](
                ctx,
                output_buf,
                input_buf,
                params_buf,
                cache_buf,
                workspace_buf,
            )

            # Compute loss gradient (backward of loss function)
            Self.LOSS_FUNCTION.backward_gpu[BATCH, Self.MODEL.OUT_DIM](
                ctx, grad_output_buf, output_buf, target_buf
            )

            # Backward pass through model (using workspace to avoid internal allocation)
            Self.MODEL.backward_gpu_ws[BATCH](
                ctx,
                grad_input_buf,
                grad_output_buf,
                params_buf,
                cache_buf,
                grads_buf,
                workspace_buf,
            )

            # Optimizer step
            self.optimizer.step_gpu[PARAM_SIZE](
                ctx, params_buf, grads_buf, state_buf
            )

            # Optionally compute and print loss
            if self.print_every > 0 and epoch % self.print_every == 0:
                # Compute loss value
                Self.LOSS_FUNCTION.forward_gpu[BATCH, Self.MODEL.OUT_DIM](
                    ctx, loss_buf, output_buf, target_buf
                )
                # Copy loss back to host (only sync when printing)
                ctx.enqueue_copy(loss_host, loss_buf)
                ctx.synchronize()
                final_loss = Float64(loss_host[0])
                print(
                    "Epoch " + String(epoch) + " - Loss: " + String(final_loss)
                )
            # Note: No sync here - GPU ops queue up and execute in order.
            # We only sync when reading results back (loss printing) or at the end.

        # Compute final loss if not already computed
        if self.print_every == 0 or (self.epochs - 1) % self.print_every != 0:
            Self.LOSS_FUNCTION.forward_gpu[BATCH, Self.MODEL.OUT_DIM](
                ctx, loss_buf, output_buf, target_buf
            )
            ctx.enqueue_copy(loss_host, loss_buf)
            ctx.synchronize()
            final_loss = Float64(loss_host[0])

        # Copy updated params and state back to host
        ctx.enqueue_copy(params_host, params_buf)
        ctx.enqueue_copy(state_host, state_buf)
        ctx.synchronize()

        # Update trainer's params and optimizer state
        for i in range(PARAM_SIZE):
            self.params[i] = params_host[i]
        for i in range(STATE_SIZE):
            self.optimizer_state[i] = state_host[i]

        return TrainResult(final_loss, self.epochs)
