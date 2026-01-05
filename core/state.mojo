trait State(Copyable, Movable, ImplicitlyCopyable):
    """Base trait for environment states.

    States must be copyable for use in generic training loops.
    """

    fn __eq__(self, other: Self) -> Bool:
        ...
