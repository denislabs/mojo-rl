"""ActionSpace trait — separates discrete and continuous EZ-V2 dispatch.

The K-step BPTT body in `train_step_gpu` is action-agnostic *except* for
the policy-head loss + grad. This trait + its concrete impls
(`DiscreteActionSpace`, `ContinuousActionSpace`) carry that one hook plus
acting-side helpers (root-candidate sampling, action picking) that the
agent uses outside training.

See `docs/EZV2_MODULAR_ARCHITECTURE.md` for the full design.

This file is currently the **dispatch spike** — only the discrete
`policy_loss_grad_gpu` hook is wired. Continuous-side hooks
(`ContinuousActionSpace`, `sample_root_candidates_gpu`) land in Phase 3
once this dispatch pattern is validated.
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.deep_agents.efficient_zero_v2.kernels import (
    ezv2_policy_loss_grad_kernel,
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

    @staticmethod
    def policy_loss_grad_gpu[
        BATCH: Int,
        PRED_OUT: Int,
        POL_TGT_DIM: Int,
        dtype: DType where dtype.is_floating_point(),
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
        loss); continuous impls fold it into the per-sample loss.
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

    @staticmethod
    def policy_loss_grad_gpu[
        BATCH: Int,
        PRED_OUT: Int,
        POL_TGT_DIM: Int,
        dtype: DType where dtype.is_floating_point(),
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
    ) raises:
        # Discrete: ent_scale ignored (the existing CE pipeline doesn't
        # compute an entropy term here; entropy regularization for
        # discrete EZ-V2 is handled at a higher loss-aggregation level
        # if at all). The kernel's ACT parameter is bound to POL_TGT_DIM
        # so the LayoutTensor layout types match the wrapper's signature.
        comptime kernel = ezv2_policy_loss_grad_kernel[
            BATCH, POL_TGT_DIM, PRED_OUT, dtype
        ]
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[kernel, kernel](
            pred_out_step,
            policy_target_step,
            grad_pred_out_step,
            per_sample_loss,
            loss_scale,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
