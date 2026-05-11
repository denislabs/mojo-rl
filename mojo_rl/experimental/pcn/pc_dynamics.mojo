"""PCDynamics — single PCN-trained world-model dynamics for MBPO.

Architecture: 5-layer PCBlock chain `(s, a) → H → H → H → H → (s_next, r)`,
matching vanilla MBPO's 4-LinearSwish + 1-Linear readout depth (same hidden
width). Trained with the Phase-1-baseline procedure (SGLD inference for
internal latents + per-block PC weight rule on W). Inference (used by MBPO
during imagination) is plain feedforward — Bogacz canonical PC eval property.

Why 5 layers (Phase B "depth match" experiment, 2026-04-30):
- The 2-layer baseline (Path B) plateaued at holdout MSE ~5 on HalfCheetah
  vs vanilla MBPO's ~0.3 (17× worse). Hypothesis: capacity gap, not training
  rule. Vanilla uses 4 hidden Swish layers; PCN had only 1 hidden tanh layer.
- This variant matches vanilla's depth at 200 hidden units, so any remaining
  MSE gap isolates the local PC weight rule itself as the bottleneck.

Why this shape (still applies):
- Phase 1 ablations showed the per-step local energy weight rule is the
  source of PCN's MBRL win. Multi-layer with internal latents is required
  to express that bias (a 1-layer PCN degenerates to standard MSE).
- No encoder. Phase 2 ruled out PCN-encoder-as-frozen-representation;
  here we test PCN-as-dynamics directly. Internal latents are settled by
  SGLD at train time, not by an amortized encoder.
- Output predicts (s_next, r) jointly (last component is reward). Caller
  handles any obs normalization — the dynamics treats obs and reward
  uniformly.

Static-method API (no per-instance state) — matches PCBlock / PCEncoder.
Caller owns all params / scratch buffers (`predict_batch` delegates to
`PCSequential.forward_eval`, which allocates its own intermediate scratch
internally — fine for the eval-only path).
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
from .predictive_model import PCSwish, PCIdentity
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

    # 5-block chain mirroring vanilla MBPO depth: 4 hidden + 1 readout.
    # Hidden blocks use PCSwish (= SiLU) — same activation vanilla MBPO uses
    # for its 4-LinearSwish dynamics ensemble. Readout PCIdentity for
    # unbounded targets (HalfCheetah rewards ∈ [-15, +20], per-step
    # delta_obs can hit ±2 on velocity dims). Caller pre-normalizes the
    # dynamics input via the GPU instance's `fit_scaler_gpu`.
    comptime BLOCK0 = PCBlock[Self.AUG_DIM, Self.HIDDEN_DIM, PCSwish]
    comptime BLOCK1 = PCBlock[Self.HIDDEN_DIM, Self.HIDDEN_DIM, PCSwish]
    comptime BLOCK2 = PCBlock[Self.HIDDEN_DIM, Self.HIDDEN_DIM, PCSwish]
    comptime BLOCK3 = PCBlock[Self.HIDDEN_DIM, Self.HIDDEN_DIM, PCSwish]
    comptime BLOCK4 = PCBlock[Self.HIDDEN_DIM, Self.READOUT, PCIdentity]
    comptime NET = PCSequential[
        Self.BLOCK0, Self.BLOCK1, Self.BLOCK2, Self.BLOCK3, Self.BLOCK4
    ]
    comptime TRAINER = PCTrainer[
        Self.BLOCK0, Self.BLOCK1, Self.BLOCK2, Self.BLOCK3, Self.BLOCK4,
        dtype=Self.dtype,
    ]

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
        """Xavier-init all 5 PCBlocks, with an optional seed for ensemble use.

        The underlying `NET.initialize_params` uses a fixed Mojo global
        RNG, which makes ensemble seed control awkward. We re-roll our
        own Xavier init that respects `seed`, mirroring PCEncoder. Bias
        terms are zeroed.

        Block fan-in/out (per Xavier formula `sqrt(6 / (in + out))`):
          B0: AUG → HIDDEN
          B1: HIDDEN → HIDDEN
          B2: HIDDEN → HIDDEN
          B3: HIDDEN → HIDDEN
          B4: HIDDEN → READOUT  (PCIdentity)
        """
        var rng = PhiloxRandom(seed=seed, offset=UInt64(0))

        @parameter
        def _init_block[
            IN_DIM: Int, OUT_DIM: Int
        ](offset: Int):
            var bound = sqrt(Float64(6.0) / Float64(IN_DIM + OUT_DIM))
            var w_size = IN_DIM * OUT_DIM
            for i in range(w_size):
                var u = Float64(rng.step_uniform()[0])
                params.ptr[offset + i] = Scalar[Self.dtype](
                    (u * 2.0 - 1.0) * bound
                )
            for j in range(OUT_DIM):
                params.ptr[offset + w_size + j] = Scalar[Self.dtype](0.0)

        _init_block[Self.AUG_DIM, Self.HIDDEN_DIM](
            Self.NET._param_offset[0]()
        )
        _init_block[Self.HIDDEN_DIM, Self.HIDDEN_DIM](
            Self.NET._param_offset[1]()
        )
        _init_block[Self.HIDDEN_DIM, Self.HIDDEN_DIM](
            Self.NET._param_offset[2]()
        )
        _init_block[Self.HIDDEN_DIM, Self.HIDDEN_DIM](
            Self.NET._param_offset[3]()
        )
        _init_block[Self.HIDDEN_DIM, Self.READOUT](
            Self.NET._param_offset[4]()
        )

    # =========================================================================
    # Inference: feedforward `(s, a) → (s_next, r)`. Used by MBPO during
    # imagination rollouts. No SGLD, matching Bogacz canonical eval.
    # Delegates to `NET.forward_eval` so depth changes propagate
    # automatically. `forward_eval` allocates its own intermediate scratch
    # internally — fine for the eval-only call sites here.
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
        # Output: (s_next, r) jointly in `out`. Caller splits.
        mut out: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.READOUT), MutAnyOrigin
        ],
    ):
        """Forward pass through the full N-block chain.

        `out[:, 0:OBS_DIM]` = next-obs prediction.
        `out[:, OBS_DIM]`   = reward prediction.
        """
        Self.NET.forward_eval[BATCH, Self.dtype](s_a, params, out)

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
        mut out: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.READOUT), MutAnyOrigin
        ],
    ) -> Float64:
        """Mean squared error of feedforward prediction vs target.

        Used by the ensemble's elite selection — pick the K members with
        the lowest holdout MSE on a held-out validation slice.
        """
        Self.predict_batch[BATCH](s_a, params, out)
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
        contract used elsewhere in pcn.

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
