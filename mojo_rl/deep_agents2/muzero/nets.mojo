"""MuZero networks (nn2) — the learned model h / g / f.

Three torsos, one per MuZero function, ported from the legacy
``deep_agents/muzero/configs.mojo`` MLP config onto nn2 ``Sequential`` /
``Parallel`` aliases. Their external contracts match the planner's GPU model
traits (``model_traits_gpu.mojo``) exactly, so the MCTS adapters wrap them
verbatim:

  * ``MZRepNet[OBS, LATENT, H]`` — h: ``obs → z``. Ends in ``MinMaxNorm`` so the
    latent is min-max scaled to [0,1] (``models.py:138``). The GPU orchestrator
    *also* applies `mcts_gpu_scale_hidden_kernel`, but min-max is **idempotent**
    on already-[0,1] data, so that is a harmless no-op — and crucially the norm
    is *inside* the autodiff graph, so training gets the scaling gradient (the
    legacy post-hoc kernel left the rep net with no signal about raw-output
    magnitude → activations exploded; see `configs.mojo:178-187`). OUT_DIM=LATENT.

  * ``MZDynNet[LATENT, ACT, BINS, H]`` — g: ``[z ⊕ onehot(a)] → [z' | reward_logits]``.
    A shared trunk then ``Parallel[latent_head + MinMaxNorm, reward_head]`` so
    only the latent half is normalized; the ``BINS`` reward bins stay raw
    categorical logits (``models.py:147-170``). IN_DIM=LATENT+ACT,
    OUT_DIM=LATENT+BINS.

  * ``MZPredNet[LATENT, ACT, BINS, H]`` — f: ``z → [policy_logits | value_logits]``.
    Shared trunk → ``Parallel[policy_head, value_head]``. Value is **categorical**
    (``BINS`` bins), not a scalar tanh (that is AlphaZero). IN_DIM=LATENT,
    OUT_DIM=ACT+BINS.

``Parallel[A, B]`` column-concatenates ``[A(x) | B(x)]``, producing exactly the
packings the planner kernels slice. All three are plain ``Module`` aliases, so
``make[target,INIT]`` / ``forward[target,B]`` / ``vjp[target]`` / Adam all apply.
Reward & value share the same ``BINS`` and the same ``[v_min,v_max]`` support as
`zero/twohot_targets.mojo` and the planner constructor — keep them in sync.
"""

from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.combinators.parallel import Parallel
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.linear_mish import LinearMish
from mojo_rl.nn2.primitives.min_max_norm import MinMaxNorm


# ──────────────────────────────────────────────────────────────────────
# h — representation: obs → z  (min-max scaled latent)
# ──────────────────────────────────────────────────────────────────────


comptime MZRepNet[OBS: Int, LATENT: Int, H: Int] = Sequential[
    LinearMish[OBS, H],
    LinearMish[H, H],
    Linear[H, LATENT],
    MinMaxNorm[LATENT],
]


# ──────────────────────────────────────────────────────────────────────
# g — dynamics: [z ⊕ onehot(a)] → [z' | reward_logits]
# ──────────────────────────────────────────────────────────────────────


comptime MZDynNet[
    LATENT: Int, ACT: Int, BINS: Int, H: Int,
] = Sequential[
    LinearMish[LATENT + ACT, H],
    LinearMish[H, H],
    Parallel[
        Sequential[Linear[H, LATENT], MinMaxNorm[LATENT]],
        Linear[H, BINS],
    ],
]


# ──────────────────────────────────────────────────────────────────────
# f — prediction: z → [policy_logits | value_logits]  (value categorical)
# ──────────────────────────────────────────────────────────────────────


comptime MZPredNet[
    LATENT: Int, ACT: Int, BINS: Int, H: Int,
] = Sequential[
    LinearMish[LATENT, H],
    Parallel[
        Linear[H, ACT],
        Linear[H, BINS],
    ],
]
