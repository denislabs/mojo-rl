"""MuZero strategy traits — re-export shim.

The actual definitions live in `mojo_rl/planners/`:
  - `planners/tree_search/strategies.mojo` — SearchMode, HiddenScaling,
    ExplorationNoise, PUCTFormula, BackupMode, PlayerMode.
  - `planners/common/value_encoding.mojo` — ValueEncoding,
    CategoricalEncoding, ScalarEncoding, SymlogEncoding.

This file remains as a source-compatibility shim so existing imports keep
working through the strangler migration:

    from mojo_rl.deep_agents.muzero.strategies import (
        SearchMode, LearnedDynamics, ...
    )

See `docs/PLANNERS_PACKAGE.md` for the full plan.
"""

from mojo_rl.planners.tree_search.strategies import (
    SearchMode, LearnedDynamics, TrueGameRules,
    HiddenScaling, MinMaxScale, NoScale,
    ExplorationNoise, DirichletNoise, EpsilonNoise, NoNoise,
    PUCTFormula, MuZeroPUCT, AlphaGoPUCT, UCB1Formula,
    BackupMode, NStepBootstrap, MonteCarloReturn, LambdaReturn,
    PlayerMode, SinglePlayer, SelfPlay,
)
from mojo_rl.planners.common.value_encoding import (
    ValueEncoding,
    CategoricalEncoding,
    ScalarEncoding,
    SymlogEncoding,
)
