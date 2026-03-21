"""Shared types for board game environments."""

from mojo_rl.core import State, Action

comptime board_dtype = DType.float32


@fieldwise_init
struct BoardGameState(Copyable, ImplicitlyCopyable, Movable, State):
    """Generic state wrapper for board games (needed for Env trait)."""

    var index: Int

    fn __init__(out self, *, copy: Self):
        self.index = copy.index

    fn __init__(out self, *, deinit take: Self):
        self.index = take.index

    fn __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct BoardGameAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Generic discrete action wrapper for board games."""

    var value: Int

    fn __init__(out self, *, copy: Self):
        self.value = copy.value

    fn __init__(out self, *, deinit take: Self):
        self.value = take.value
