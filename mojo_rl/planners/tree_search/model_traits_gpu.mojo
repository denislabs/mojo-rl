"""GPU tree-search model contract — Representation / Dynamics / Prediction.

The GPU counterpart of ``model_traits.mojo``. Same three-trait split, but
the surface speaks ``LayoutTensor`` views with a method-level ``B: Int``
comptime parameter — exactly the shape ``Network[…].forward_gpu[B]``
expects. Adapters wrap a ``GPUNetworkState`` and call into ``Network``;
they don't need to know about the MCTS kernels.

These traits are scaffolding for the generic GPU MCTS orchestrator (the
``GenericGPUMCTS`` struct, landing later in Phase 3b together with the
agent rewiring). They are not consumed yet — the production GPU search
loop currently lives inline inside ``muzero.mojo:select_action_gpu`` and
calls the kernels in ``planners/tree_search/mcts_gpu.mojo`` directly,
interleaving its own ``Network.forward_gpu`` calls. Shipping the trait
surface now gives that rewiring a fixed target to bind to.

**Why this is just three forward methods** (no softmax / decode /
scaling): all of those live in the MCTS GPU kernels already
(``gpu_mcts_init_root_kernel`` softmaxes priors, the
``…_expand_backup_…`` family decodes categorical reward / value with the
inverse scalar transform, hidden MinMax scaling runs inline). The agent
adapter's only job is to feed network params + state and produce the raw
network output — same way the inline orchestrator does today.

The future ``GenericGPUMCTS`` struct (sketched in
``docs/PLANNERS_PACKAGE.md`` Phase 3b) will own the ``GPUMCTSState`` and
the search loop, taking ``REP: RepresentationGPU``, ``DYN: DynamicsGPU``,
``PRED: PredictionGPU`` as comptime params plus the existing strategy
traits (``PUCTFormula`` / ``ExplorationNoise`` / ``PlayerMode``).
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype


trait RepresentationGPU(ImplicitlyDestructible):
    """Batched GPU encoder: obs → hidden state.

    Called once per ``search`` at the root, with ``B = N_ENVS``. The
    adapter wraps the agent's ``Network[RepModel, Opt].forward_gpu[B]``
    and is responsible for any in-network state normalization
    (``MinMaxNorm`` layer at the tail of MuZero's representation net,
    etc.). MCTS reads ``hidden_out`` verbatim and stores it in the
    root slot of the GPU node pool.
    """

    comptime OBS_DIM: Int
    comptime LATENT_DIM: Int

    def encode_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        obs: LayoutTensor[
            dtype, Layout.row_major(B, Self.OBS_DIM), MutAnyOrigin
        ],
        mut hidden_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
    ) raises:
        """Encode ``obs`` of shape ``(B, OBS_DIM)`` into ``hidden_out``
        of shape ``(B, LATENT_DIM)``. Pure forward; no host sync. The
        planner is allowed to launch this and the prediction call
        immediately after, then kick off ``init_root_kernel`` — no
        intervening sync.
        """
        ...


trait DynamicsGPU(ImplicitlyDestructible):
    """Batched GPU dynamics forward: (hidden, one_hot_action) → (next_hidden, reward_logits).

    ``DYN_OUT_DIM = LATENT_DIM + REWARD_HEAD_DIM`` where the reward
    head is either 1 (scalar reward) or ``NUM_BINS`` (categorical).
    The agent's adapter knows the right output dimension at construction
    time; the MCTS expand kernels comptime-branch on ``DYN_OUT − LATENT``
    to decode.

    ``DYN_IN_DIM = LATENT_DIM + ACTION_DIM`` — the
    ``…_build_dyn_input_kernel`` family builds this layout before the
    forward call.
    """

    comptime LATENT_DIM: Int
    comptime ACTION_DIM: Int
    comptime DYN_IN_DIM: Int
    """LATENT_DIM + ACTION_DIM. Stated as a separate constant so the
    planner can size scratch buffers without referencing two adapters."""
    comptime DYN_OUT_DIM: Int
    """LATENT_DIM + (1 or NUM_BINS) — caller does the decode."""

    def step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        dyn_in: LayoutTensor[
            dtype, Layout.row_major(B, Self.DYN_IN_DIM), MutAnyOrigin
        ],
        mut dyn_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.DYN_OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        """Forward the dynamics net for the full batch. Input is the
        ``hidden + one-hot-action`` concatenation produced upstream by
        the planner's selection kernel; output is the raw network
        output (categorical reward bins + next hidden state), decoded
        downstream by the expand kernel.
        """
        ...


trait PredictionGPU(ImplicitlyDestructible):
    """Batched GPU prediction forward: hidden → (policy_logits, value_logits).

    ``PRED_OUT_DIM = ACTION_DIM + VALUE_HEAD_DIM`` where the value head
    is 1 (scalar, tanh-squashed for AlphaZero) or ``NUM_BINS``
    (categorical, MuZero). The MCTS expand / init_root kernels softmax
    the policy slice and decode the value slice — no decoding inside
    the trait.
    """

    comptime LATENT_DIM: Int
    comptime ACTION_DIM: Int
    comptime PRED_OUT_DIM: Int

    def predict_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        hidden: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut pred_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.PRED_OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        """Forward the prediction net for the full batch. Returns raw
        logits — softmax / categorical decode happen in the MCTS
        kernels (``gpu_mcts_init_root_kernel``,
        ``gpu_mcts_batched_expand_backup_muzero_kernel``).
        """
        ...
