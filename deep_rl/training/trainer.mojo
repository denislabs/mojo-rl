from ..model import Model
from ..optimizer import Optimizer
from ..loss import LossFunction
from ..constants import dtype


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
]:
    """Training configuration for neural networks.

    Usage:
        var trainer = Trainer[BATCH, IN, OUT, PARAM_SIZE](
            epochs=100,
            print_every=20
        )
        var result = trainer.train(model, optimizer, input, target)

    Parameters:
        MODEL: The model to train.
        OPTIMIZER: The optimizer to use.
        LOSS_FUNCTION: The loss function to use.
    """

    var epochs: Int
    var print_every: Int
    var model: Self.MODEL
    var optimizer: Self.OPTIMIZER
    var loss_function: Self.LOSS_FUNCTION

    fn __init__(
        out self,
        model: Self.MODEL,
        optimizer: Self.OPTIMIZER,
        loss_function: Self.LOSS_FUNCTION,
        epochs: Int = 100,
        print_every: Int = 10,
    ):
        """Initialize trainer with configuration."""
        self.epochs = epochs
        self.print_every = print_every
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

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
        var output = InlineArray[Scalar[dtype], BATCH * Self.MODEL.OUT_DIM](
            uninitialized=True
        )
        var grad_output = InlineArray[
            Scalar[dtype], BATCH * Self.MODEL.OUT_DIM
        ](uninitialized=True)
        var grad_input = InlineArray[Scalar[dtype], BATCH * Self.MODEL.IN_DIM](
            uninitialized=True
        )
        # Allocate cache buffer for forward/backward passes
        var cache = InlineArray[Scalar[dtype], BATCH * Self.MODEL.CACHE_SIZE](
            uninitialized=True
        )

        var final_loss: Float64 = 0.0

        for epoch in range(self.epochs):
            # Forward pass (with cache)
            self.model.forward[BATCH](input, output, cache)

            # Compute loss and gradient
            var loss = self.loss_function.forward[BATCH * Self.MODEL.OUT_DIM](
                output, target
            )
            self.loss_function.backward[BATCH * Self.MODEL.OUT_DIM](
                output, target, grad_output
            )

            # Backward pass (with cache)
            self.model.zero_grad()
            self.model.backward[BATCH](grad_output, grad_input, cache)

            # Update parameters (rebind to unify types)
            var params = self.model.get_params()
            var grads = self.model.get_grads()
            var params_cast = rebind[
                InlineArray[Scalar[dtype], Self.OPTIMIZER.PARAM_SIZE]
            ](params)
            var grads_cast = rebind[
                InlineArray[Scalar[dtype], Self.OPTIMIZER.PARAM_SIZE]
            ](grads)
            self.optimizer.step(params_cast, grads_cast)
            # Use params_cast (the updated copy) when setting back to model
            self.model.set_params(
                rebind[InlineArray[Scalar[dtype], Self.MODEL.PARAM_SIZE]](
                    params_cast
                )
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
        var output = InlineArray[Scalar[dtype], BATCH * Self.MODEL.OUT_DIM](
            uninitialized=True
        )
        # Use forward - no cache needed for evaluation
        self.model.forward[BATCH](input, output)

        for i in range(BATCH):
            for j in range(Self.MODEL.OUT_DIM):
                print(
                    "Output["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]: "
                    + String(Float64(output[i * Self.MODEL.OUT_DIM + j]))
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
            output, target
        )
