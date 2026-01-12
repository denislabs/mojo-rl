from ..model import Model
from ..optimizer import Optimizer
from ..loss import LossFunction
from ..constants import dtype
from ..initializer import Initializer, Xavier

from layout import Layout, LayoutTensor


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
    var params: InlineArray[Scalar[dtype], Self.MODEL.PARAM_SIZE]
    var grads: InlineArray[Scalar[dtype], Self.MODEL.PARAM_SIZE]
    var optimizer_state: InlineArray[
        Scalar[dtype], Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM
    ]

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

        # Initialize optimizer state to zero (moments for Adam, unused for SGD)
        self.optimizer_state = InlineArray[
            Scalar[dtype], Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM
        ](uninitialized=True)
        for i in range(Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM):
            self.optimizer_state[i] = 0

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
            Layout.row_major(Self.MODEL.PARAM_SIZE, Self.OPTIMIZER.STATE_PER_PARAM),
            MutAnyOrigin,
        ](self.optimizer_state.unsafe_ptr())

        var final_loss: Float64 = 0.0

        for epoch in range(self.epochs):
            # Forward pass (with cache and params)
            self.model.forward[BATCH](
                input_tensor, output_tensor, params_tensor, cache_tensor
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
        self.model.forward[BATCH](input_tensor, output_tensor, params_tensor)

        for i in range(BATCH):
            for j in range(Self.MODEL.OUT_DIM):
                print(
                    "Output["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]: "
                    + String(
                        Float64(output_storage[i * Self.MODEL.OUT_DIM + j])
                    )
                )
                print(
                    "Target["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]: "
                    + String(Float64(target[i * Self.MODEL.OUT_DIM + j]))
                )

        return self.loss_function.forward[BATCH * Self.MODEL.OUT_DIM](
            output_storage, target
        )
