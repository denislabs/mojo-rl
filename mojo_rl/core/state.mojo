trait State(Copyable, ImplicitlyCopyable, Movable, ImplicitlyDestructible):
    """Base trait for environment states.

    States must be copyable for use in generic training loops.
    """

    def __eq__(self, other: Self) -> Bool:
        ...
