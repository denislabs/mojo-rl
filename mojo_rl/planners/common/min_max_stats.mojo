"""MinMaxStats — running Q-value range tracker for PUCT normalization.

Promoted from ``mojo_rl/deep_agents/muzero/utils.mojo`` so MCTS (now in
``planners/tree_search``) can use it without a downward import into a
specific agent. The original copy stays where it is — MuZero training
targets / scalar transforms still live there, and a shim is cheap.

Tracks the min and max of every Q-value seen during a single MCTS
search. Normalizing Q to [0, 1] before the PUCT prior term keeps the
exploration constant ``c`` scale-invariant across value ranges
(reward-shaped envs, sparse-reward games, ...).
"""


struct MinMaxStats(ImplicitlyCopyable, Movable):
    """Track min/max Q-values inside an MCTS tree.

    Initialized with extreme sentinels so the first ``update`` call
    always replaces both. ``normalize`` returns the raw value
    unchanged if the observed range is zero (no values, or all equal).
    """

    var minimum: Float64
    var maximum: Float64

    def __init__(out self):
        self.minimum = Float64(1e18)
        self.maximum = Float64(-1e18)

    def __init__(out self, *, copy: Self):
        self.minimum = copy.minimum
        self.maximum = copy.maximum

    def __init__(out self, *, deinit move: Self):
        self.minimum = move.minimum
        self.maximum = move.maximum

    def update(mut self, value: Float64):
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

    def normalize(self, value: Float64) -> Float64:
        var delta = self.maximum - self.minimum
        if delta > 0.0:
            return (value - self.minimum) / delta
        return value
