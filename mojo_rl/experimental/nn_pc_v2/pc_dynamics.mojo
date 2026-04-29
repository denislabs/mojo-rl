"""PCDynamics — single PCN-trained world-model dynamics for MBPO.

Architecture: 2-layer PCBlock chain `(s, a) → HIDDEN → (s_next, r)`. Trained
with the Phase-1-baseline procedure (SGLD inference for z + per-block PC
weight rule on W). Inference (used by MBPO during imagination) is plain
feedforward — Bogacz canonical PC eval property.

Why this shape:
- Phase 1 ablations (`docs/PCN_MBRL_DESIGN.md`) showed the per-step local
  energy weight rule is the source of PCN's MBRL win. Multi-layer with an
  internal latent z is required to express that bias (a 1-layer PCN
  degenerates to standard MSE).
- No encoder. Phase 2 ruled out PCN-encoder-as-frozen-representation;
  here we test PCN-as-dynamics directly. The internal latent z is settled
  by SGLD at train time, not by an amortized encoder.
- Output predicts (s_next, r) jointly (last component is reward). Caller
  handles any obs normalization — the dynamics treats obs and reward
  uniformly.

Static-method API (no per-instance state) — matches PCBlock / PCEncoder.
Caller owns all params / scratch buffers.
"""

from layout import Layout, LayoutTensor
from std.math import sqrt
from std.memory import memset
from std.random.philox import Random as PhiloxRandom

from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam

from .pc_block import PCBlock
from .pc_sequential import PCSequential
from .pc_trainer import PCTrainer
from .predictive_model import PCTanh
from .pc_utils import clip_grad_norm


struct PCDynamics[
    OBS_DIM: Int,
    ACTION_DIM: Int,
    HIDDEN_DIM: Int = 200,
    dtype: DType = DType.float32,
]:
    """Static namespace for a PCN-trained `(s, a) → (s_next, r)` dynamics.

    Compile-time constants:
      AUG_DIM    = OBS_DIM + ACTION_DIM        (PCBlock 0 input)
      READOUT    = OBS_DIM + 1                 (PCBlock 1 output: s_next, r)
      PARAM_SIZE = NET.PARAM_SIZE              (sum of both PCBlocks)
      LATENT_DIM = HIDDEN_DIM                  (z lives in HIDDEN_DIM)

    All scratch buffer sizes are exposed as compile-time constants so
    the caller can `alloc()` them once.
    """

    comptime AUG_DIM: Int = Self.OBS_DIM + Self.ACTION_DIM
    comptime READOUT: Int = Self.OBS_DIM + 1

    comptime BLOCK0 = PCBlock[Self.AUG_DIM, Self.HIDDEN_DIM, PCTanh]
    comptime BLOCK1 = PCBlock[Self.HIDDEN_DIM, Self.READOUT, PCTanh]
    comptime NET = PCSequential[Self.BLOCK0, Self.BLOCK1]
    comptime TRAINER = PCTrainer[Self.BLOCK0, Self.BLOCK1, dtype=Self.dtype]

    comptime PARAM_SIZE: Int = Self.NET.PARAM_SIZE
    comptime LATENT_DIM: Int = Self.HIDDEN_DIM

    # Scratch dimensions per BATCH (per-block sizes are exposed by NET).
    comptime SCRATCH_LAT: Int = Self.NET.LATENT_DIM
    comptime SCRATCH_OUT: Int = Self.NET.SCRATCH_OUT_DIM
    comptime SCRATCH_IN: Int = Self.NET.SCRATCH_IN_DIM

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    def init_params(
        mut params: LayoutTensor[
            Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        seed: UInt64,
    ):
        """Xavier-init both PCBlocks, with an optional seed for ensemble use.

        Note: the underlying `NET.initialize_params` uses a fixed Mojo
        global RNG, which makes ensemble seed control awkward. We re-roll
        our own Xavier init that respects `seed`, mirroring PCEncoder's
        approach. Bias terms are zeroed.
        """
        var rng = PhiloxRandom(seed=seed, offset=UInt64(0))

        # Block 0: W shape (AUG_DIM, HIDDEN_DIM), then bias (HIDDEN_DIM).
        var bound0 = sqrt(
            Float64(6.0) / Float64(Self.AUG_DIM + Self.HIDDEN_DIM)
        )
        var w0_size = Self.AUG_DIM * Self.HIDDEN_DIM
        for i in range(w0_size):
            var u = Float64(rng.step_uniform()[0])
            params.ptr[i] = Scalar[Self.dtype]((u * 2.0 - 1.0) * bound0)
        for j in range(Self.HIDDEN_DIM):
            params.ptr[w0_size + j] = Scalar[Self.dtype](0.0)

        # Block 1: offset by Block 0's PARAM_SIZE.
        comptime offset_b1 = Self.NET._param_offset[1]()
        var bound1 = sqrt(
            Float64(6.0) / Float64(Self.HIDDEN_DIM + Self.READOUT)
        )
        var w1_size = Self.HIDDEN_DIM * Self.READOUT
        for i in range(w1_size):
            var u = Float64(rng.step_uniform()[0])
            params.ptr[offset_b1 + i] = Scalar[Self.dtype](
                (u * 2.0 - 1.0) * bound1
            )
        for j in range(Self.READOUT):
            params.ptr[offset_b1 + w1_size + j] = Scalar[Self.dtype](0.0)

    # =========================================================================
    # Per-block param views (for direct PCBlock.predict usage at inference)
    # =========================================================================

    @staticmethod
    def params_b0_view(
        params_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
    ) -> LayoutTensor[
        Self.dtype, Layout.row_major(Self.BLOCK0.PARAM_SIZE), MutAnyOrigin
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.BLOCK0.PARAM_SIZE),
            MutAnyOrigin,
        ](params_buf)

    @staticmethod
    def params_b1_view(
        params_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
    ) -> LayoutTensor[
        Self.dtype, Layout.row_major(Self.BLOCK1.PARAM_SIZE), MutAnyOrigin
    ]:
        comptime offset_b1 = Self.NET._param_offset[1]()
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.BLOCK1.PARAM_SIZE),
            MutAnyOrigin,
        ](params_buf + offset_b1)

    # =========================================================================
    # Inference: feedforward `(s, a) → (s_next, r)`. Used by MBPO during
    # imagination rollouts. No SGLD, matching Bogacz canonical eval.
    # =========================================================================

    @staticmethod
    def predict_batch[
        BATCH: Int
    ](
        s_a: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.AUG_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        # Scratch (caller-allocated):
        mut a_aug: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.AUG_DIM), MutAnyOrigin
        ],
        mut z_hidden: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.HIDDEN_DIM), MutAnyOrigin
        ],
        mut a_z: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.HIDDEN_DIM), MutAnyOrigin
        ],
        # Output: (s_next, r) jointly in `out`. Caller splits.
        mut out: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.READOUT), MutAnyOrigin
        ],
    ):
        """Forward pass: `(s, a) → z = block_0(...) → out = block_1(z)`.

        `out[:, 0:OBS_DIM]` = next-obs prediction.
        `out[:, OBS_DIM]`   = reward prediction.
        """
        var pb0 = Self.params_b0_view(params.ptr)
        var pb1 = Self.params_b1_view(params.ptr)
        Self.BLOCK0.predict[BATCH, Self.dtype](s_a, pb0, z_hidden, a_aug)
        Self.BLOCK1.predict[BATCH, Self.dtype](z_hidden, pb1, out, a_z)

    # =========================================================================
    # Holdout loss for elite selection (no SGLD, just feedforward MSE).
    # =========================================================================

    @staticmethod
    def eval_loss_batch[
        BATCH: Int
    ](
        s_a: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.AUG_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.READOUT), MutAnyOrigin
        ],
        params: LayoutTensor[
            Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut a_aug: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.AUG_DIM), MutAnyOrigin
        ],
        mut z_hidden: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.HIDDEN_DIM), MutAnyOrigin
        ],
        mut a_z: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.HIDDEN_DIM), MutAnyOrigin
        ],
        mut out: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.READOUT), MutAnyOrigin
        ],
    ) -> Float64:
        """Mean squared error of feedforward prediction vs target.

        Used by the ensemble's elite selection — pick the K members with
        the lowest holdout MSE on a held-out validation slice.
        """
        Self.predict_batch[BATCH](s_a, params, a_aug, z_hidden, a_z, out)
        var sum_sq: Float64 = 0.0
        for b in range(BATCH):
            for d in range(Self.READOUT):
                var o = Float64(out.ptr[b * Self.READOUT + d])
                var t = Float64(target.ptr[b * Self.READOUT + d])
                var diff = o - t
                sum_sq += diff * diff
        return sum_sq / Float64(BATCH * Self.READOUT)

    # =========================================================================
    # Training step: SGLD inference for z + PC weight rule + Adam update.
    # =========================================================================

    @staticmethod
    def compute_grads_batch[
        BATCH: Int
    ](
        s_a: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.AUG_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.READOUT), MutAnyOrigin
        ],
        # Param + grads (caller-owned):
        params: LayoutTensor[
            Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        # SGLD inference scratch (caller-owned, can be shared across ensemble
        # members in sequence — only one member at a time uses these):
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.SCRATCH_LAT),
            MutAnyOrigin,
        ],
        mut mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.SCRATCH_OUT),
            MutAnyOrigin,
        ],
        mut a_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.SCRATCH_IN),
            MutAnyOrigin,
        ],
        mut z_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.SCRATCH_IN),
            MutAnyOrigin,
        ],
        mut dx_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.SCRATCH_LAT),
            MutAnyOrigin,
        ],
        T_infer: Int,
        lr_x: Scalar[Self.dtype],
    ) -> Float64:
        """SGLD-settle z + compute PC weight gradients. Does NOT touch params.

        Caller is expected to follow this with a `clip_grad_norm` (optional)
        and an `Adam.step` to actually apply the update — keeps the dynamics
        optimizer-agnostic and matches the PCTrainer.compute_grads_only
        contract used elsewhere in nn_pc_v2.

        Returns the final-step output loss (MSE on `target`).
        """
        var result = Self.TRAINER.compute_grads_only[BATCH](
            params, grads,
            latents, mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
            s_a, target,
            T_infer=T_infer,
            lr_x=lr_x,
        )
        return result.output_loss_final
