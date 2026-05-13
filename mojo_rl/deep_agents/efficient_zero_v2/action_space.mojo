"""ActionSpace trait — separates discrete and continuous EZ-V2 dispatch.

The K-step BPTT body in `train_step_gpu` is action-agnostic *except* for
the policy-head loss + grad. This trait + its concrete impls
(`DiscreteActionSpace`, `ContinuousActionSpace`) carry that one hook plus
acting-side helpers (root-candidate sampling, action picking) that the
agent uses outside training.

See `docs/EZV2_MODULAR_ARCHITECTURE.md` for the full design and
`docs/EZV2_CONTINUOUS_PHASE3.md` for the continuous-side rationale.

Two impls now exist:
  - `DiscreteActionSpace[ACT, K]` — softmax CE on visit-distribution target.
  - `ContinuousActionSpace[ACT_DIM, K, MAX_ACTION, MIN_STD, ENT_W]` —
    squashed-Gaussian NLL + entropy bonus on `a*` target (paper Eq. 8/9).

Sampling-side hooks (`sample_root_candidates_gpu`, etc.) still live on
the agent struct since the MCTS state types differ between action spaces.
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.deep_agents.efficient_zero_v2.kernels import (
    ezv2_policy_loss_grad_kernel,
    ezv2_policy_loss_grad_continuous_kernel,
)


comptime TPB: Int = 256


# ═══════════════════════════════════════════════════════════════════════════
# Trait
# ═══════════════════════════════════════════════════════════════════════════


trait ActionSpace:
    """Action-space dispatch for EZ-V2 — discrete vs continuous.

    The trait carries the dimensional knobs the BPTT core consults
    (`POLICY_TARGET_DIM` for the gather kernel) plus exactly one kernel
    hook (`policy_loss_grad_gpu`). Acting-side helpers
    (`sample_root_candidates_gpu`, `sample_for_acting`) are absent here
    by design — they belong on the agent struct since the MCTS state
    types differ between discrete and continuous (different tree shapes).
    """

    comptime ACT_DIM: Int
    """Width of one action slot in the replay buffer + dyn-input concat.
    Discrete: one-hot width = num_actions. Continuous: real action_dim.
    Either way `[BATCH, K, ACT_DIM]` floats — the dyn-input concat kernel
    doesn't care which."""

    comptime POLICY_OUT_DIM: Int
    """Number of *policy* outputs from the prediction net.
    Discrete: ACT_DIM (logits). Continuous: 2 * ACT_DIM (μ ‖ raw_σ before
    softplus). Pred net's full output width is `POLICY_OUT_DIM + BINS`."""

    comptime POLICY_TARGET_DIM: Int
    """Width of the per-step policy-target buffer the agent stores.
    Discrete: ACT_DIM (visit distribution). Continuous: ACT_DIM
    (a*_S — the chosen action — for the simple-best-action loss)."""

    comptime IS_CONTINUOUS: Bool
    """For agent-side dispatch only. The BPTT core never reads this."""

    comptime K_ROOT: Int
    """Gumbel-Top-k K (discrete) or sampled-K (continuous, paper default 16)."""

    comptime MAX_ACTION: Float64
    """Action magnitude bound (continuous only — discrete supplies 0.0 as
    a placeholder so trait resolution stays uniform). Read by the full-π
    loss kernel for the tanh squash. Paper default depends on env."""

    comptime MIN_STD: Float64
    """Lower bound on σ (continuous only — discrete supplies 0.0). Bounds
    `1/σ` from above so pre-training the gradient stays well-conditioned.
    Paper App. G default 0.1."""

    comptime STD_MAGNIFICATION: Float64
    """σ multiplier for the second half of root candidates (continuous
    only — discrete supplies 0.0). Paper App. A default 3.0."""

    comptime N_POLICY_AT_ROOT: Int
    """Number of root candidates drawn from the policy `N(μ, σ)`. The
    remaining `K_ROOT - N_POLICY_AT_ROOT` candidates come from
    `Uniform(-MAX_ACTION, MAX_ACTION)` (reference DMC mode, `cy_mcts.py`
    `policy_action_num=4, random_action_num=12`). When equal to
    `K_ROOT`, the legacy magnified-policy mode runs instead (half from
    `N(μ, σ)`, half from `N(μ, STD_MAGNIFICATION · σ)`). Discrete
    supplies `K` (= K_ROOT) — no uniform-random branch."""

    @staticmethod
    def policy_loss_grad_gpu[
        BATCH: Int,
        PRED_OUT: Int,
        POL_TGT_DIM: Int,
        dtype: DType,
    ](
        ctx: DeviceContext,
        pred_out_step: LayoutTensor[
            dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
        ],
        policy_target_step: LayoutTensor[
            dtype, Layout.row_major(BATCH * POL_TGT_DIM), MutAnyOrigin
        ],
        grad_pred_out_step: LayoutTensor[
            dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
        ],
        per_sample_loss: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        loss_scale: Scalar[dtype],
        ent_scale: Scalar[dtype],
        seed: UInt64,
    ) raises:
        """Forward + backward through the policy section of pred_out_step.

        Writes the grad into `grad_pred_out_step[b, 0:POLICY_OUT_DIM]`.
        The trailing BINS slots are owned by the value-loss kernel and
        must not be touched here. Writes per-sample loss into
        `per_sample_loss[b]`.

        `POL_TGT_DIM` should equal `Self.POLICY_TARGET_DIM` at the call
        site — passed as a method parameter rather than read from `Self`
        to keep the trait method's tensor types syntactically uniform
        across impls. The BPTT core supplies `Config.ActSpace.POLICY_TARGET_DIM`
        for this argument.

        `ent_scale` is the entropy-bonus weight (paper Eq. 9). Discrete
        impls may ignore it (entropy not currently part of discrete
        loss); continuous impls fold it into the per-sample loss via an
        MC entropy estimator seeded by `seed` (each train-step caller
        should pass a distinct seed; discrete impls ignore it).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Discrete impl — wraps the existing CE kernel
# ═══════════════════════════════════════════════════════════════════════════


struct DiscreteActionSpace[ACT: Int, K: Int = 8](ActionSpace):
    """Discrete-action implementation: softmax CE on a visit-distribution
    target. Wraps the existing `ezv2_policy_loss_grad_kernel`.

    Parameters:
        ACT: Number of discrete actions. Becomes ACT_DIM, POLICY_OUT_DIM,
            POLICY_TARGET_DIM.
        K: Gumbel-Top-k K — number of root candidates the MCTS keeps.
            Paper default 8 for discrete-low-dim envs.
    """

    comptime ACT_DIM: Int = Self.ACT
    comptime POLICY_OUT_DIM: Int = Self.ACT
    comptime POLICY_TARGET_DIM: Int = Self.ACT
    comptime IS_CONTINUOUS: Bool = False
    comptime K_ROOT: Int = Self.K
    comptime MAX_ACTION: Float64 = 0.0
    comptime MIN_STD: Float64 = 0.0
    comptime STD_MAGNIFICATION: Float64 = 0.0
    # Discrete doesn't use continuous root sampling; set to K_ROOT so any
    # downstream `comptime if N_POLICY_AT_ROOT == K_ROOT` evaluates True.
    comptime N_POLICY_AT_ROOT: Int = Self.K

    @staticmethod
    def policy_loss_grad_gpu[
        BATCH: Int,
        PRED_OUT: Int,
        POL_TGT_DIM: Int,
        dtype: DType,
    ](
        ctx: DeviceContext,
        pred_out_step: LayoutTensor[
            dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
        ],
        policy_target_step: LayoutTensor[
            dtype, Layout.row_major(BATCH * POL_TGT_DIM), MutAnyOrigin
        ],
        grad_pred_out_step: LayoutTensor[
            dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
        ],
        per_sample_loss: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        loss_scale: Scalar[dtype],
        ent_scale: Scalar[dtype],
        seed: UInt64,
    ) raises:
        # Discrete: ent_scale + seed ignored (the existing CE pipeline
        # doesn't compute an entropy term here; entropy regularization
        # for discrete EZ-V2 is handled at a higher loss-aggregation
        # level if at all). The kernel's ACT parameter is bound to
        # POL_TGT_DIM so the LayoutTensor layout types match the
        # wrapper's signature.
        comptime kernel = ezv2_policy_loss_grad_kernel[
            BATCH, POL_TGT_DIM, PRED_OUT, dtype
        ]
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[kernel](
            pred_out_step,
            policy_target_step,
            grad_pred_out_step,
            per_sample_loss,
            loss_scale,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Continuous impl — squashed-Gaussian NLL + entropy bonus
# ═══════════════════════════════════════════════════════════════════════════


struct ContinuousActionSpace[
    ACT_DIM_: Int,
    K: Int = 16,
    MAX_ACTION_: Float64 = 1.0,
    MIN_STD_: Float64 = 0.1,
    STD_MAGNIFICATION_: Float64 = 3.0,
    # Default `K` (all policy) preserves the legacy magnified-mode root
    # sampling. Set < K to opt into reference DMC sampling (policy +
    # uniform random) — see `SampledGumbelMCTS.N_POLICY_AT_ROOT`.
    N_POLICY_AT_ROOT_: Int = K,
](ActionSpace):
    """Continuous-action implementation: squashed-Gaussian NLL + entropy
    bonus on the search-selected target action `a*` (paper Eq. 8/9).

    Wraps `ezv2_policy_loss_grad_continuous_kernel`. The pred net's policy
    section emits `(μ_raw, σ_raw)` per dim, which the kernel forwards
    through `μ = MAX·tanh(μ_raw/MAX)` and `σ = softplus(σ_raw) + MIN_STD`
    before evaluating `−log π(a*) − ent_scale · H[π]`.

    Parameters:
        ACT_DIM_: Real action vector dimension. Becomes `ACT_DIM`,
            `POLICY_TARGET_DIM`, and `POLICY_OUT_DIM = 2 * ACT_DIM_`
            (μ ‖ σ_raw).
        K: Number of root candidates the sampled-Gumbel MCTS keeps
            (paper App. A default 16 for proprio).
        MAX_ACTION_: Action vector |a*_d| upper bound. The squash uses
            `MAX·tanh(·/MAX)`; the kernel atanh-clamps `a*/MAX` to ±0.999
            for numerical stability.
        MIN_STD_: Floor on σ; bounds `1/σ` from above so pre-training
            (σ_raw ≈ 0) the gradient stays well-conditioned. Paper App. G
            default 0.1.
        STD_MAGNIFICATION_: Used by the sampled-MCTS root-candidate
            sampler (paper App. A: half the K candidates are drawn from
            `N(μ, STD_MAGNIFICATION·σ)` for exploration). The training
            kernel here ignores it; lives on the trait so the agent can
            read it via `Config.ActSpace.STD_MAGNIFICATION` at sample time.
    """

    comptime ACT_DIM: Int = Self.ACT_DIM_
    comptime POLICY_OUT_DIM: Int = 2 * Self.ACT_DIM_
    comptime POLICY_TARGET_DIM: Int = Self.ACT_DIM_
    comptime IS_CONTINUOUS: Bool = True
    comptime K_ROOT: Int = Self.K
    comptime MAX_ACTION: Float64 = Self.MAX_ACTION_
    comptime MIN_STD: Float64 = Self.MIN_STD_
    comptime STD_MAGNIFICATION: Float64 = Self.STD_MAGNIFICATION_
    comptime N_POLICY_AT_ROOT: Int = Self.N_POLICY_AT_ROOT_

    @staticmethod
    def policy_loss_grad_gpu[
        BATCH: Int,
        PRED_OUT: Int,
        POL_TGT_DIM: Int,
        dtype: DType,
    ](
        ctx: DeviceContext,
        pred_out_step: LayoutTensor[
            dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
        ],
        policy_target_step: LayoutTensor[
            dtype, Layout.row_major(BATCH * POL_TGT_DIM), MutAnyOrigin
        ],
        grad_pred_out_step: LayoutTensor[
            dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
        ],
        per_sample_loss: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
        loss_scale: Scalar[dtype],
        ent_scale: Scalar[dtype],
        seed: UInt64,
    ) raises:
        # `POL_TGT_DIM` arrives equal to `Self.ACT_DIM_` from the caller
        # (the BPTT core supplies `Config.ActSpace.POLICY_TARGET_DIM`),
        # so the kernel's ACT_DIM template parameter is bound through it
        # to keep the LayoutTensor layout types aligned with the
        # wrapper's signature — same pattern as DiscreteActionSpace.
        comptime kernel = ezv2_policy_loss_grad_continuous_kernel[
            BATCH, POL_TGT_DIM, PRED_OUT, dtype
        ]
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[kernel](
            pred_out_step,
            policy_target_step,
            grad_pred_out_step,
            per_sample_loss,
            loss_scale,
            ent_scale,
            Scalar[dtype](Self.MAX_ACTION),
            Scalar[dtype](Self.MIN_STD),
            seed,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
