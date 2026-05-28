"""MuZero policy-mode comptime selector.

Picks which GPU MCTS orchestrator the MuZero agent dispatches to:
  * ``MuZeroPUCTPolicy``      → ``GenericGPUMCTS`` (vanilla MuZero/AlphaZero
    PUCT selection with Dirichlet root noise, current production path).
  * ``GumbelMuZeroPolicy[K]`` → ``GumbelGPUMCTS`` (Full Gumbel MuZero from
    Danihelka et al. 2022 — Gumbel-Top-k root sampling, Sequential Halving
    over ``log2(K)`` phases, deterministic interior σ(Q) − N/(1+ΣN)
    selection, improved-policy training target). Provably better policy
    improvement at low simulation budgets (the MuZero/EZv2 regime).

The PolicyMode lives on ``MuZeroConfig`` so the choice is a compile-time
dispatch in the agent — no runtime cost. ``gumbel_scale`` is a runtime
parameter (passed to ``GumbelGPUMCTS`` ctor) so the same compiled config
can switch noise on/off between training and evaluation.
"""


trait MuZeroPolicyMode:
    """Compile-time selector between MuZero MCTS variants.

    The MuZero agent's ``train_gpu`` / ``train_selfplay_gpu`` branches
    on ``IS_GUMBEL`` to instantiate either ``GenericGPUMCTS`` (PUCT) or
    ``GumbelGPUMCTS`` (Gumbel-MuZero). All other fields are only
    consulted by the chosen orchestrator.
    """

    comptime IS_GUMBEL: Bool
    comptime MAX_K: Int
    """Maximum Gumbel-Top-k root candidates (ignored when
    ``IS_GUMBEL=False``). Must be a power of two ≤ ``ACT``."""


struct MuZeroPUCTPolicy(MuZeroPolicyMode):
    """Vanilla MuZero PUCT search (the production / mctx ``muzero_policy``
    equivalent). Routes to ``GenericGPUMCTS``."""

    comptime IS_GUMBEL: Bool = False
    comptime MAX_K: Int = 1


struct GumbelMuZeroPolicy[max_k: Int = 8](MuZeroPolicyMode):
    """Full Gumbel-MuZero (mctx ``gumbel_muzero_policy`` equivalent).
    Routes to ``GumbelGPUMCTS``.

    Parameters:
        max_k: Sequential-Halving root candidate budget. Defaults to 8.
            Must be a power of two and ≤ ``Config.action_dim``. The
            orchestrator clips at runtime if needed."""

    comptime IS_GUMBEL: Bool = True
    comptime MAX_K: Int = Self.max_k
