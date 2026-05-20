"""Value encoding strategies — how scalar values/rewards are represented.
Shared by tree search (MCTS/MuZero/EZv2) and trajectory optimizers that bootstrap
on a learned value head (MPPI/TD-MPC2).

Promoted from `mojo_rl/deep_agents/muzero/strategies.mojo` in Phase 0 of the
planners package refactor (see `docs/PLANNERS_PACKAGE.md`).
"""


# ═══════════════════════════════════════════════════════════════════════════
# ValueEncoding
# ═══════════════════════════════════════════════════════════════════════════


trait ValueEncoding:
    """How scalar values/rewards are encoded for the value head."""

    comptime IS_DISTRIBUTIONAL: Bool
    comptime USE_SCALAR_TRANSFORM: Bool


struct CategoricalEncoding(ValueEncoding):
    """Distributional: two-hot over NUM_BINS support bins."""

    comptime IS_DISTRIBUTIONAL: Bool = True
    comptime USE_SCALAR_TRANSFORM: Bool = True


struct ScalarEncoding(ValueEncoding):
    """Direct scalar prediction. For bounded-reward envs."""

    comptime IS_DISTRIBUTIONAL: Bool = False
    comptime USE_SCALAR_TRANSFORM: Bool = False


struct SymlogEncoding(ValueEncoding):
    """Scalar with symlog transform (DreamerV3-style)."""

    comptime IS_DISTRIBUTIONAL: Bool = False
    comptime USE_SCALAR_TRANSFORM: Bool = True
