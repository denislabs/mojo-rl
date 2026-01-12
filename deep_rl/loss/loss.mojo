from ..constants import dtype


trait LossFunction(Movable & ImplicitlyCopyable):
    """Base trait for loss functions.

    Loss functions have:
    - forward() for computing loss
    - backward() for computing gradients
    """

    fn forward[
        SIZE: Int
    ](
        self,
        output: InlineArray[Scalar[dtype], SIZE],
        target: InlineArray[Scalar[dtype], SIZE],
    ) -> Float64:
        """Forward pass for loss function."""
        ...

    fn backward[
        SIZE: Int
    ](
        self,
        output: InlineArray[Scalar[dtype], SIZE],
        target: InlineArray[Scalar[dtype], SIZE],
        mut grad: InlineArray[Scalar[dtype], SIZE],
    ):
        """Backward pass for loss function."""
        ...
