"""PCDynamicsEnsembleGPU — GPU per-member training/predict for PCN-MBPO.

Mirrors `PCDynamicsEnsemble` (CPU) but routes everything through nn_pc_v2's
GPU primitives:

- `PCSequential.forward_eval_gpu`         feedforward (used at MBPO
                                          imagination time and for elite
                                          selection's holdout MSE).
- `PCTrainer.compute_grads_only_gpu`      SGLD inference + PC weight grads.
- `Adam.step_gpu`                         optimizer update.

Per-member state lives in one big device buffer per kind (params, grads,
opt_state, opt_global), sliced by offset for each member.

Holdout-loss eval on GPU: do feedforward on device, then download the
output and target tensors to host once and compute MSE on CPU. The
download is the only host sync per ensemble member per retrain — small
compared to SGLD inference cost.

CPU and GPU ensembles share the same compile-time `PCDynamics[...]` so
buffer sizes line up exactly across the two paths. Swap one for the other
in the agent without touching buffer allocations.
"""

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext

from mojo_rl.nn.optimizer import Optimizer

from .pc_dynamics import PCDynamics
from .pc_utils import clip_grad_norm


struct PCDynamicsEnsembleGPU[
    OBS_DIM: Int,
    ACTION_DIM: Int,
    HIDDEN_DIM: Int,
    NUM_ENSEMBLE: Int,
    NUM_ELITES: Int,
    dtype: DType = DType.float32,
]:
    """GPU twin of `PCDynamicsEnsemble`. Per-member state in flat buffers."""

    comptime DYN = PCDynamics[
        Self.OBS_DIM, Self.ACTION_DIM, Self.HIDDEN_DIM, Self.dtype
    ]

    comptime PER_MEMBER_PARAM_SIZE: Int = Self.DYN.PARAM_SIZE
    comptime TOTAL_PARAM_SIZE: Int = Self.NUM_ENSEMBLE * Self.DYN.PARAM_SIZE

    # =========================================================================
    # Initialization — runs on host, caller is expected to upload to device.
    # =========================================================================

    @staticmethod
    def init_all_host(
        host_params_buf: UnsafePointer[
            Scalar[Self.dtype], origin=MutAnyOrigin
        ],
        base_seed: UInt64,
    ):
        """Init all NUM_ENSEMBLE members into a host buffer. Caller copies
        to device with `ctx.enqueue_copy(device_buf, host_buf)` afterwards.

        Member m uses seed = base_seed + m (independent Xavier draws).
        """
        for m in range(Self.NUM_ENSEMBLE):
            var member_ptr = host_params_buf + m * Self.PER_MEMBER_PARAM_SIZE
            var member_view = LayoutTensor[
                Self.dtype,
                Layout.row_major(Self.PER_MEMBER_PARAM_SIZE),
                MutAnyOrigin,
            ](member_ptr)
            Self.DYN.init_params(member_view, base_seed + UInt64(m))

    # =========================================================================
    # Per-member buffer slicing helpers (work on either host or device ptrs).
    # =========================================================================

    @staticmethod
    fn member_params_view(
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
    # Per-member training on GPU.
    #   compute_grads_only_gpu (SGLD + PC weight grads) → optional grad clip
    #   on device → Adam.step_gpu.
    # =========================================================================

    @staticmethod
    def train_member_gpu[
        BATCH: Int, OPT: Optimizer
    ](
        ctx: DeviceContext,
        m: Int,
        s_a: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.AUG_DIM),
            MutAnyOrigin,
        ],
        target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.READOUT),
            MutAnyOrigin,
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
        # Per-member Adam step counter (caller-stored — Adam reads this for
        # bias correction; not used by GPU step counter on device, but kept
        # for parity with the Adam.step_gpu signature).
        mut step_num: Int,
        T_infer: Int,
        lr_x: Scalar[Self.dtype],
    ) raises:
        """Train member `m` on one device-resident batch.

        No host sync inside the inference + grad + optimizer chain. Caller
        decides when to `ctx.synchronize()` (typically after some number of
        member updates, or once per training round).
        """
        var p_view = Self.member_params_view(params_buf, m)
        var g_view = Self.member_params_view(grads_buf, m)
        var s_view = LayoutTensor[
            Self.dtype,
            Layout.row_major(
                Self.PER_MEMBER_PARAM_SIZE, OPT.STATE_PER_PARAM
            ),
            MutAnyOrigin,
        ](
            opt_state_buf
            + m * Self.PER_MEMBER_PARAM_SIZE * OPT.STATE_PER_PARAM
        )
        var gl_view = LayoutTensor[
            Self.dtype,
            Layout.row_major(OPT.GLOBAL_STATE_SIZE),
            MutAnyOrigin,
        ](opt_global_buf + m * OPT.GLOBAL_STATE_SIZE)

        # SGLD-settle z + compute PC weight grads on GPU.
        Self.DYN.TRAINER.compute_grads_only_gpu[BATCH](
            ctx, p_view, g_view,
            latents, mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
            s_a, target,
            T_infer=T_infer,
            lr_x=lr_x,
        )

        step_num += 1
        OPT.step_gpu[Self.PER_MEMBER_PARAM_SIZE, Self.dtype](
            ctx, p_view, g_view, s_view, gl_view, step_num
        )

    # =========================================================================
    # Per-member feedforward prediction on GPU. Used for both imagination
    # and the holdout-loss feedforward in elite selection.
    # =========================================================================

    @staticmethod
    def predict_member_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        m: Int,
        s_a: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.AUG_DIM),
            MutAnyOrigin,
        ],
        params_buf: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin],
        # Shared scratch from the agent (mu_buf, a_buf shape set by NET).
        mut mu_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.SCRATCH_OUT),
            MutAnyOrigin,
        ],
        mut a_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.SCRATCH_IN),
            MutAnyOrigin,
        ],
        mut output: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.DYN.READOUT),
            MutAnyOrigin,
        ],
    ) raises:
        var p_view = Self.member_params_view(params_buf, m)
        Self.DYN.NET.forward_eval_gpu[BATCH, Self.dtype](
            ctx, s_a, p_view, output, mu_buf, a_buf
        )

    # =========================================================================
    # Elite selection on host. Caller provides a list of NUM_ENSEMBLE host
    # losses (already downloaded from device); we sort and pick top K.
    # =========================================================================

    @staticmethod
    fn select_elites(
        losses: List[Float64],
        mut elite_indices: List[Int],
    ):
        """Identical to CPU `PCDynamicsEnsemble.select_elites` — pure host
        sort, no GPU work."""
        var idx = List[Int](capacity=Self.NUM_ENSEMBLE)
        for i in range(Self.NUM_ENSEMBLE):
            idx.append(i)
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
