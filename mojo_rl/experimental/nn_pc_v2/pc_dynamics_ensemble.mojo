"""PCDynamicsEnsemble — N independent PCN dynamics for MBPO.

Wraps `NUM_ENSEMBLE` separate `PCDynamics` instances, each initialized
with a different RNG seed (so the networks differ). Provides:

- Per-member training (caller iterates over members, calling `train_member`).
- Per-member feedforward prediction (`predict_member`).
- Holdout-loss evaluation + elite selection (top `NUM_ELITES` by loss).
- Imagination-rollout helper that samples a random elite per row of a
  batch (matches MBPO's "pick a random ensemble member each step").

Variance source: ensemble disagreement (different params across members),
not per-network logvar — this is a small departure from textbook MBPO,
chosen because PCN is naturally deterministic and adding a logvar head
would require a custom training procedure outside Phase-1's validated recipe.

All state (params, grads, optimizer state, scratch) is caller-allocated;
the struct just bundles compile-time constants and methods. Matches
PCDynamics / PCBlock / PCEncoder style.
"""

from layout import Layout, LayoutTensor
from std.memory import alloc, memset
from std.random.philox import Random as PhiloxRandom

from mojo_rl.nn.optimizer import Optimizer

from .pc_dynamics import PCDynamics
from .pc_utils import clip_grad_norm


struct PCDynamicsEnsemble[
    OBS_DIM: Int,
    ACTION_DIM: Int,
    HIDDEN_DIM: Int,
    NUM_ENSEMBLE: Int,
    NUM_ELITES: Int,
    dtype: DType = DType.float32,
]:
    """Static namespace for an N-network PCN dynamics ensemble.

    `predict_member` / `train_member` operate on a member-index `m`
    in `[0, NUM_ENSEMBLE)`. Per-member buffer slices are computed via
    `member_param_slice` etc.
    """

    comptime DYN = PCDynamics[
        Self.OBS_DIM, Self.ACTION_DIM, Self.HIDDEN_DIM, Self.dtype
    ]

    comptime PER_MEMBER_PARAM_SIZE: Int = Self.DYN.PARAM_SIZE
    comptime TOTAL_PARAM_SIZE: Int = Self.NUM_ENSEMBLE * Self.DYN.PARAM_SIZE

    # =========================================================================
    # Initialization — N members, each with a different RNG seed.
    # =========================================================================

    @staticmethod
    def init_all(
        params_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        base_seed: UInt64,
    ):
        """Initialize all NUM_ENSEMBLE members. Member m uses seed = base_seed + m."""
        for m in range(Self.NUM_ENSEMBLE):
            var member_ptr = params_buf + m * Self.PER_MEMBER_PARAM_SIZE
            var member_view = LayoutTensor[
                Self.dtype,
                Layout.row_major(Self.PER_MEMBER_PARAM_SIZE),
                MutAnyOrigin,
            ](member_ptr)
            Self.DYN.init_params(member_view, base_seed + UInt64(m))

    # =========================================================================
    # Per-member buffer slicing helpers.
    # =========================================================================

    @staticmethod
    fn member_params(
        params_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        m: Int,
    ) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.PER_MEMBER_PARAM_SIZE),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.PER_MEMBER_PARAM_SIZE),
            MutAnyOrigin,
        ](params_buf + m * Self.PER_MEMBER_PARAM_SIZE)

    # =========================================================================
    # Per-member training — caller drives the loop over members.
    # =========================================================================

    @staticmethod
    def train_member[
        BATCH: Int, OPT: Optimizer
    ](
        m: Int,
        s_a: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.DYN.AUG_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.DYN.READOUT), MutAnyOrigin
        ],
        # Whole-ensemble buffers (we slice into them):
        params_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        grads_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        opt_state_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        opt_global_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        # Shared SGLD scratch (reused across members in the loop):
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.SCRATCH_LAT),
            MutAnyOrigin,
        ],
        mut mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.SCRATCH_OUT),
            MutAnyOrigin,
        ],
        mut a_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.SCRATCH_IN),
            MutAnyOrigin,
        ],
        mut z_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.SCRATCH_IN),
            MutAnyOrigin,
        ],
        mut dx_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.SCRATCH_LAT),
            MutAnyOrigin,
        ],
        # Hyperparams + Adam step counter (1 per member, caller-stored):
        mut step_num: Int,
        T_infer: Int,
        lr_x: Scalar[Self.dtype],
        lr_scale: Float64 = 1.0,
        grad_clip_norm: Float64 = 1.0,
    ) raises -> Float64:
        """Train member `m` on one batch. Returns final-step output loss."""
        var member_params_ptr = params_buf + m * Self.PER_MEMBER_PARAM_SIZE
        var member_grads_ptr = grads_buf + m * Self.PER_MEMBER_PARAM_SIZE
        var member_opt_state_ptr = (
            opt_state_buf
            + m * Self.PER_MEMBER_PARAM_SIZE * OPT.STATE_PER_PARAM
        )
        var member_opt_global_ptr = (
            opt_global_buf + m * OPT.GLOBAL_STATE_SIZE
        )

        var p_view = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.PER_MEMBER_PARAM_SIZE),
            MutAnyOrigin,
        ](member_params_ptr)
        var g_view = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.PER_MEMBER_PARAM_SIZE),
            MutAnyOrigin,
        ](member_grads_ptr)
        var s_view = LayoutTensor[
            Self.dtype,
            Layout.row_major(
                Self.PER_MEMBER_PARAM_SIZE, OPT.STATE_PER_PARAM
            ),
            MutAnyOrigin,
        ](member_opt_state_ptr)
        var gl_view = LayoutTensor[
            Self.dtype,
            Layout.row_major(OPT.GLOBAL_STATE_SIZE),
            MutAnyOrigin,
        ](member_opt_global_ptr)

        # Zero grads, then SGLD-settle z + compute PC weight grads.
        memset(member_grads_ptr, 0, Self.PER_MEMBER_PARAM_SIZE)
        var loss = Self.DYN.compute_grads_batch[BATCH](
            s_a, target,
            p_view, g_view,
            latents, mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
            T_infer=T_infer,
            lr_x=lr_x,
        )

        # Optional grad clip.
        if grad_clip_norm > 0.0:
            clip_grad_norm[Self.PER_MEMBER_PARAM_SIZE, Self.dtype](
                g_view, grad_clip_norm
            )

        # Adam step (caller's per-member step counter).
        step_num += 1
        OPT.step[Self.PER_MEMBER_PARAM_SIZE, Self.dtype](
            p_view, g_view, s_view, gl_view,
            step_num, lr_scale=lr_scale,
        )
        return loss

    # =========================================================================
    # Per-member feedforward prediction.
    # =========================================================================

    @staticmethod
    def predict_member[
        BATCH: Int
    ](
        m: Int,
        s_a: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.DYN.AUG_DIM), MutAnyOrigin
        ],
        params_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        # Scratch (reused across calls):
        mut a_aug: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.DYN.AUG_DIM), MutAnyOrigin
        ],
        mut z_hidden: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.HIDDEN_DIM),
            MutAnyOrigin,
        ],
        mut a_z: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.HIDDEN_DIM),
            MutAnyOrigin,
        ],
        mut out: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.DYN.READOUT), MutAnyOrigin
        ],
    ):
        var member_view = Self.member_params(params_buf, m)
        Self.DYN.predict_batch[BATCH](
            s_a, member_view, a_aug, z_hidden, a_z, out
        )

    # =========================================================================
    # Holdout-loss evaluation per member (for elite selection).
    # =========================================================================

    @staticmethod
    def eval_member_loss[
        BATCH: Int
    ](
        m: Int,
        s_a: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.DYN.AUG_DIM), MutAnyOrigin
        ],
        target: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.DYN.READOUT), MutAnyOrigin
        ],
        params_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        mut a_aug: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.DYN.AUG_DIM), MutAnyOrigin
        ],
        mut z_hidden: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.HIDDEN_DIM),
            MutAnyOrigin,
        ],
        mut a_z: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.HIDDEN_DIM),
            MutAnyOrigin,
        ],
        mut out: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.DYN.READOUT), MutAnyOrigin
        ],
    ) -> Float64:
        var member_view = Self.member_params(params_buf, m)
        return Self.DYN.eval_loss_batch[BATCH](
            s_a, target, member_view, a_aug, z_hidden, a_z, out
        )

    # =========================================================================
    # Elite selection: pick the K members with the lowest losses.
    # =========================================================================

    @staticmethod
    fn select_elites(
        losses: List[Float64],
        mut elite_indices: List[Int],
    ):
        """Sort `losses` ascending; populate `elite_indices` with the top K
        member indices (lowest losses).

        `losses` must have length NUM_ENSEMBLE; `elite_indices` is
        rewritten with NUM_ELITES indices in ascending-loss order.
        """
        # Initial index list [0..NUM_ENSEMBLE).
        var idx = List[Int](capacity=Self.NUM_ENSEMBLE)
        for i in range(Self.NUM_ENSEMBLE):
            idx.append(i)
        # Selection sort (NUM_ENSEMBLE is small, ≤ 7 typically).
        for k in range(Self.NUM_ELITES):
            var best_pos = k
            var best_loss = losses[idx[k]]
            for p in range(k + 1, Self.NUM_ENSEMBLE):
                if losses[idx[p]] < best_loss:
                    best_loss = losses[idx[p]]
                    best_pos = p
            if best_pos != k:
                var tmp = idx[k]
                idx[k] = idx[best_pos]
                idx[best_pos] = tmp
        elite_indices.clear()
        for k in range(Self.NUM_ELITES):
            elite_indices.append(idx[k])
