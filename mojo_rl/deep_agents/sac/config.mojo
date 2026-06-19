"""SAC named preset — config descriptor + factory (Design F).

Additive sugar over the primitive
`SACAgent[train_target, SAMPLE, ACTOR, CRITIC]`. The primitive stays the
source of truth for arbitrary actor/critic/replay combinations; this
module names the canonical Soft Actor-Critic setup and bundles its tuned
defaults + a default fused-`LinearReLU` actor/critic.

Same shape as `c51/config.mojo` and `redq/config.mojo`:

  1. `SACConfigT` — a trait bundling the FULL compile-time identity of the
     algorithm: the deployment `TARGET`, the replay `SAMPLE` block, the
     actor + critic nets, plus tuned scalar defaults (`DEF_*`).

  2. `SACConfig` — a zero-field conformer struct parametrized by `target`.
     ONE config covers both CPU and GPU because the replay block is
     target-generic (`ReplaySampleStep[AnyReplay[target, …]]`).

  3. `agent_from_config` + the capitalized preset `SAC`. The preset is a
     SINGLE function taking `target` as a parameter and reads like a
     constructor at the call site:

         var agent = SAC["cpu", OBS, ACT, BATCH, CAP]()
         var agent = SAC["gpu", OBS, ACT, BATCH, CAP](ctx=ctx)

The default nets use FUSED `LinearReLU` (matmul + bias + ReLU in one
kernel) for the hidden layers — same parameter layout as the unfused
`Linear` + `ReLU` pair, so checkpoints written from a `SAC[...]` agent on
one target load on any other `SAC[...]` agent of matching dims. The fused
form halves the per-hidden-layer kernel-launch count on the eager GPU
path.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.linear_relu import LinearReLU
from mojo_rl.nn.storage.combinators.sequential import Sequential

from ..primitives.stochastic_actor import StochasticActor
from ..training.blocks import SampleBlock, ReplaySampleStep
from ..data.any_replay import AnyReplay

from .agent import SACAgent


# ──────────────────────────────────────────────────────────────────────
# Net presets — target-agnostic, parametrized comptime aliases.
# Hidden layers are FUSED LinearReLU; the critic's scalar head stays a
# plain Linear (no output activation).
# ──────────────────────────────────────────────────────────────────────


comptime SACActorNet[OBS: Int, ACT: Int, HIDDEN: Int] = StochasticActor[
    OBS, ACT,
    LinearReLU[OBS, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
"""2-layer fused-MLP trunk + (μ, log σ) heads — SAC's stochastic actor."""


comptime SACCriticNet[OBS: Int, ACT: Int, HIDDEN: Int] = Sequential[
    LinearReLU[OBS + ACT, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]
"""2-layer fused-MLP critic. The twin critics in the SAC trainer are each
one of these."""


# ──────────────────────────────────────────────────────────────────────
# Config trait — full compile-time identity + tuned scalar defaults.
# ──────────────────────────────────────────────────────────────────────


trait SACConfigT(Copyable, Movable, ImplicitlyDeletable):
    """Compile-time descriptor of a SAC-family algorithm. Conformers are
    zero-field comptime tags — never instantiated at runtime; only their
    comptime members are read."""

    comptime TARGET: StaticString
    comptime SAMPLE: SampleBlock
    comptime ACTOR: Module
    comptime CRITIC: Module

    # Tuned scalar defaults (read into __init__ kwarg defaults).
    comptime DEF_ACTOR_LR: Scalar[DT]
    comptime DEF_CRITIC_LR: Scalar[DT]
    comptime DEF_ALPHA_LR: Scalar[DT]
    comptime DEF_GAMMA: Scalar[DT]
    comptime DEF_TAU: Scalar[DT]
    comptime DEF_ACTION_SCALE: Scalar[DT]
    comptime DEF_INIT_ALPHA: Scalar[DT]
    comptime DEF_TARGET_ENTROPY: Scalar[DT]
    comptime DEF_LEARNING_STARTS: Int
    comptime DEF_MAX_GRAD_NORM: Scalar[DT]


# ──────────────────────────────────────────────────────────────────────
# Conformer — one struct, parametrized by `target`.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct SACConfig[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
](SACConfigT):
    """Soft Actor-Critic (Haarnoja et al. 2018) — twin critics, max-
    entropy stochastic policy, auto-tuned temperature α, uniform replay
    (1-step). Default target entropy is the `-ACT` heuristic. One config
    covers cpu + gpu via the target-generic replay block."""

    comptime TARGET = Self.target
    comptime SAMPLE = ReplaySampleStep[
        AnyReplay[Self.target, Self.OBS, Self.ACT, Self.CAP], Self.BATCH,
    ]
    comptime ACTOR = SACActorNet[Self.OBS, Self.ACT, Self.HIDDEN]
    comptime CRITIC = SACCriticNet[Self.OBS, Self.ACT, Self.HIDDEN]

    comptime DEF_ACTOR_LR = Scalar[DT](3e-4)
    comptime DEF_CRITIC_LR = Scalar[DT](3e-4)
    comptime DEF_ALPHA_LR = Scalar[DT](3e-4)
    comptime DEF_GAMMA = Scalar[DT](0.99)
    comptime DEF_TAU = Scalar[DT](0.005)
    comptime DEF_ACTION_SCALE = Scalar[DT](1.0)
    comptime DEF_INIT_ALPHA = Scalar[DT](0.2)
    comptime DEF_TARGET_ENTROPY = Scalar[DT](-Float64(Self.ACT))
    comptime DEF_LEARNING_STARTS = 1_000
    comptime DEF_MAX_GRAD_NORM = Scalar[DT](0.0)


# ──────────────────────────────────────────────────────────────────────
# Generic factory — any SACConfigT → primitive agent, defaults applied.
# ──────────────────────────────────────────────────────────────────────


def agent_from_config[
    CONFIG: SACConfigT,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = CONFIG.DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = CONFIG.DEF_CRITIC_LR,
    alpha_lr: Scalar[DT] = CONFIG.DEF_ALPHA_LR,
    gamma: Scalar[DT] = CONFIG.DEF_GAMMA,
    tau: Scalar[DT] = CONFIG.DEF_TAU,
    action_scale: Scalar[DT] = CONFIG.DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = CONFIG.DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = CONFIG.DEF_TARGET_ENTROPY,
    learning_starts: Int = CONFIG.DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    max_grad_norm: Scalar[DT] = CONFIG.DEF_MAX_GRAD_NORM,
    per_alpha: Scalar[DT] = Scalar[DT](0.6),
    per_beta: Scalar[DT] = Scalar[DT](0.4),
    per_epsilon: Scalar[DT] = Scalar[DT](1e-6),
    use_bf16: Bool = False,
    use_ere: Bool = False,
    ere_eta: Scalar[DT] = Scalar[DT](0.996),
    ere_c_min: Int = 1,
    ere_k_max: Int = 1000,
) raises -> SACAgent[
    CONFIG.TARGET,
    CONFIG.SAMPLE,
    CONFIG.ACTOR,
    CONFIG.CRITIC,
]:
    """Build the primitive `SACAgent` from any `SACConfigT`. Every scalar
    defaults to the config's tuned value but stays overridable. The
    deployment target and replay block are read off the config, so this
    one function serves cpu and gpu."""
    return SACAgent[
        CONFIG.TARGET,
        CONFIG.SAMPLE,
        CONFIG.ACTOR,
        CONFIG.CRITIC,
    ](
        ctx=ctx,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        gamma=gamma,
        tau=tau,
        action_scale=action_scale,
        init_alpha=init_alpha,
        target_entropy=target_entropy,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
        per_alpha=per_alpha,
        per_beta=per_beta,
        per_epsilon=per_epsilon,
        use_bf16=use_bf16,
        use_ere=use_ere,
        ere_eta=ere_eta,
        ere_c_min=ere_c_min,
        ere_k_max=ere_k_max,
    )


# ──────────────────────────────────────────────────────────────────────
# Capitalized preset — single function, `target` as a parameter.
# Reads like a constructor. Full tuning surface, defaults from the config.
# ──────────────────────────────────────────────────────────────────────


def SAC[
    target: StaticString,
    OBS: Int, ACT: Int, BATCH: Int, CAP: Int,
    HIDDEN: Int = 256,
](
    ctx: Optional[DeviceContext] = None,
    actor_lr: Scalar[DT] = SACConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ACTOR_LR,
    critic_lr: Scalar[DT] = SACConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_CRITIC_LR,
    alpha_lr: Scalar[DT] = SACConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ALPHA_LR,
    gamma: Scalar[DT] = SACConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_GAMMA,
    tau: Scalar[DT] = SACConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TAU,
    action_scale: Scalar[DT] = SACConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_ACTION_SCALE,
    init_alpha: Scalar[DT] = SACConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_INIT_ALPHA,
    target_entropy: Scalar[DT] = SACConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_TARGET_ENTROPY,
    learning_starts: Int = SACConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_LEARNING_STARTS,
    window_size: Int = 10,
    initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    max_grad_norm: Scalar[DT] = SACConfig[target, OBS, ACT, BATCH, CAP, HIDDEN].DEF_MAX_GRAD_NORM,
    per_alpha: Scalar[DT] = Scalar[DT](0.6),
    per_beta: Scalar[DT] = Scalar[DT](0.4),
    per_epsilon: Scalar[DT] = Scalar[DT](1e-6),
    use_bf16: Bool = False,
    use_ere: Bool = False,
    ere_eta: Scalar[DT] = Scalar[DT](0.996),
    ere_c_min: Int = 1,
    ere_k_max: Int = 1000,
) raises -> SACAgent[
    target,
    ReplaySampleStep[AnyReplay[target, OBS, ACT, CAP], BATCH],
    SACActorNet[OBS, ACT, HIDDEN],
    SACCriticNet[OBS, ACT, HIDDEN],
]:
    """Soft Actor-Critic with the canonical fused-`LinearReLU` actor +
    twin critics, uniform replay, auto-tuned α. `target` selects cpu/gpu;
    all scalars default to the tuned config value but stay overridable."""
    return agent_from_config[
        SACConfig[target, OBS, ACT, BATCH, CAP, HIDDEN]
    ](
        ctx=ctx,
        actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr,
        gamma=gamma, tau=tau,
        action_scale=action_scale,
        init_alpha=init_alpha, target_entropy=target_entropy,
        learning_starts=learning_starts,
        window_size=window_size,
        initial_episode_fill=initial_episode_fill,
        max_grad_norm=max_grad_norm,
        per_alpha=per_alpha, per_beta=per_beta, per_epsilon=per_epsilon,
        use_bf16=use_bf16, use_ere=use_ere,
        ere_eta=ere_eta, ere_c_min=ere_c_min, ere_k_max=ere_k_max,
    )
