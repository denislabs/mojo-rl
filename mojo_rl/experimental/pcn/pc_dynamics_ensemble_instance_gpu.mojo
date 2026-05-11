"""PCDynamicsEnsembleInstanceGPU — owning GPU wrapper for PCN dynamics ensemble.

Mirrors `PCDynamicsEnsembleInstanceCPU` but lives on `DeviceBuffer`s and
routes everything through pcn's GPU primitives. Designed to be a
drop-in replacement for `GPUDynamicsEnsemble` (vanilla MBPO) inside a
forked PCN-MBPO agent — same `train_on_buffer`-style entry point and
`predict`-style entry point, leaner because:

- No `max_logvar` / `min_logvar` buffers (PCN is deterministic; ensemble
  disagreement provides variance, no per-network logvar to learn).
- No NLL backward (`t_grad_in`/`t_grad_out`) — PCN uses local PC weight
  gradients computed inside `compute_grads_only_gpu`.
- No input scaler (caller is expected to normalize obs/action before
  staging into the input buffer; matches today's CPU PCN-MBPO test).

Buffer ownership: this struct owns all GPU + host scratch via
`DeviceBuffer` / `HostBuffer`. The agent embeds an instance and passes
mut references into its loop.
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.checkpoint import (
    write_float_section_ptr,
    read_float_section_list,
)
from mojo_rl.deep_agents.core.replay import GPUReplayBuffer
from mojo_rl.deep_agents.core.kernels import (
    compute_scaler_mean_kernel,
    compute_scaler_std_kernel,
)

from .pc_dynamics import PCDynamics
from .pc_dynamics_ensemble_gpu import PCDynamicsEnsembleGPU


struct PCDynamicsEnsembleInstanceGPU[
    OBS_DIM: Int,
    ACTION_DIM: Int,
    HIDDEN_DIM: Int = 200,
    NUM_ENSEMBLE: Int = 7,
    NUM_ELITES: Int = 5,
    DYN_BATCH: Int = 256,
    ROLLOUT_BATCH: Int = 400,
    T_INFER: Int = 10,
    LR_X_FLOAT: Float64 = 0.01,
    DYN_LR: Float64 = 0.001,
    # Workspace sizing for the actor's `forward_gpu` during rollouts. The
    # PCN-MBPO agent fork passes its actor's `WORKSPACE_SIZE_PER_SAMPLE`
    # times ROLLOUT_BATCH at instantiation. Kept comptime so allocation is
    # static. Default 1 means the actor doesn't need scratch.
    R_WS_SIZE: Int = 1,
    dtype: DType where dtype.is_floating_point() = DType.float32,
](Movable):
    """Owning GPU wrapper around `PCDynamicsEnsembleGPU`.

    Field naming mirrors `GPUDynamicsEnsemble` so the PCN-MBPO agent fork
    can read this struct's members with minimal renames. Methods are
    leaner — no logvar / NLL machinery.
    """

    comptime ENS = PCDynamicsEnsembleGPU[
        Self.OBS_DIM,
        Self.ACTION_DIM,
        Self.HIDDEN_DIM,
        Self.NUM_ENSEMBLE,
        Self.NUM_ELITES,
        Self.dtype,
    ]
    comptime DYN = Self.ENS.DYN
    comptime OPT = Adam[LR=Self.DYN_LR]

    # =========================================================================
    # MBPO-compatible comptime aliases. The agent fork reads these as
    # `gpu_dynamics.rollout_batch`, `gpu_dynamics.DYN_IN`, etc.
    #
    # PCN dynamics outputs are deterministic (delta_obs + reward only — no
    # logvar), so DYN_OUT == DYN_PRED == READOUT. Vanilla MBPO's DYN_OUT is
    # 2*DYN_PRED because it interleaves logvar.
    # =========================================================================

    comptime obs_dim: Int = Self.OBS_DIM
    comptime action_dim: Int = Self.ACTION_DIM
    comptime num_ensemble: Int = Self.NUM_ENSEMBLE
    comptime num_elites: Int = Self.NUM_ELITES
    comptime rollout_batch: Int = Self.ROLLOUT_BATCH
    comptime DYN_IN: Int = Self.DYN.AUG_DIM
    comptime DYN_PRED: Int = Self.DYN.READOUT
    comptime DYN_OUT: Int = Self.DYN.READOUT
    # Sampling-staging shared size — covers both train minibatch sampling
    # and rollout-start sampling.
    comptime SAMPLE_BATCH: Int = (
        Self.DYN_BATCH if Self.DYN_BATCH
        > Self.ROLLOUT_BATCH else Self.ROLLOUT_BATCH
    )

    # =========================================================================
    # Owned device buffers — one big block per kind, member m at offset
    # `m * PER_MEMBER_PARAM_SIZE` (or appropriate stride for opt state).
    # =========================================================================

    var params_dbuf: DeviceBuffer[Self.dtype]
    var grads_dbuf: DeviceBuffer[Self.dtype]
    var opt_state_dbuf: DeviceBuffer[Self.dtype]
    var opt_global_dbuf: DeviceBuffer[Self.dtype]

    # Per-member host-side Adam step counters.
    var step_nums: List[Int]

    # Elite indices — top NUM_ELITES members by holdout MSE. Host-resident.
    var elite_indices: List[Int]

    # SGLD scratch shared across members during a single train pass
    # (DYN_BATCH-sized).
    var lat_dbuf: DeviceBuffer[Self.dtype]
    var mu_eps_dbuf: DeviceBuffer[Self.dtype]
    var a_below_dbuf: DeviceBuffer[Self.dtype]
    var z_below_dbuf: DeviceBuffer[Self.dtype]
    var dx_dbuf: DeviceBuffer[Self.dtype]

    # Input + target staging (DYN_BATCH-sized). Caller fills these on host
    # then ctx.enqueue_copy onto device before calling train_one_pass.
    var s_a_dbuf: DeviceBuffer[Self.dtype]
    var target_dbuf: DeviceBuffer[Self.dtype]
    var s_a_host: HostBuffer[Self.dtype]
    var target_host: HostBuffer[Self.dtype]

    # Holdout-eval scratch (DYN_BATCH-sized).
    var e_a_aug_dbuf: DeviceBuffer[Self.dtype]
    var e_z_dbuf: DeviceBuffer[Self.dtype]
    var e_a_z_dbuf: DeviceBuffer[Self.dtype]
    var e_out_dbuf: DeviceBuffer[Self.dtype]
    var e_out_host: HostBuffer[Self.dtype]
    var e_target_host: HostBuffer[Self.dtype]

    # =========================================================================
    # Rollout-time MBPO-compatible buffers. Field names match
    # `GPUDynamicsEnsemble` so the agent fork's `do_model_rollouts_gpu`
    # accesses (`r_obs`, `r_actions`, `r_dyn_input`, `r_dyn_output_all`,
    # `r_alive`, `r_ws`, `r_next_obs`, `r_dones`, `r_rewards`, etc.) work
    # without renaming the agent body.
    # =========================================================================

    # Rollout state (per ROLLOUT_BATCH).
    var r_obs: DeviceBuffer[Self.dtype]  # [RB * obs_dim]
    var r_next_obs: DeviceBuffer[Self.dtype]  # [RB * obs_dim]
    var r_actions: DeviceBuffer[Self.dtype]  # [RB * action_dim]
    var r_rewards: DeviceBuffer[Self.dtype]  # [RB]
    var r_dones: DeviceBuffer[Self.dtype]  # [RB]
    var r_alive: DeviceBuffer[Self.dtype]  # [RB] alive mask multi-step

    # Rollout dynamics input/output. `r_dyn_input` is the staged [obs|action]
    # tensor (same role as MBPO's). `r_dyn_output` holds one elite member's
    # forward output (READOUT-wide, no logvar). `r_dyn_output_all` stacks
    # all elite forwards: [NUM_ELITES, RB, READOUT] for per-sample elite
    # selection in the rollout sample kernel.
    var r_dyn_input: DeviceBuffer[Self.dtype]  # [RB * AUG_DIM]
    var r_dyn_output: DeviceBuffer[Self.dtype]  # [RB * READOUT]
    var r_dyn_output_all: DeviceBuffer[
        Self.dtype
    ]  # [NUM_ELITES * RB * READOUT]

    # PCN-internal activation scratch shared with `predict_rollout_member`.
    # These are PC-graph activations (not MBPO's NLL pipeline), kept under
    # the prior names since they have no MBPO equivalent.
    # Rollout-time PCN forward scratch. `predict_member_gpu` wraps
    # `forward_eval_gpu`, which needs `mu_buf` ([RB*SCRATCH_OUT]) and
    # `a_buf` ([RB*SCRATCH_IN]) sized for the whole PCBlock chain (sum
    # of per-block OUT_DIMs / IN_DIMs across both blocks). The names are
    # legacy from a single-block draft; capacity is now full SCRATCH_*.
    var r_a_aug_dbuf: DeviceBuffer[Self.dtype]  # mu_buf  [RB * SCRATCH_OUT]
    var r_z_dbuf: DeviceBuffer[Self.dtype]  # a_buf   [RB * SCRATCH_IN]
    var r_a_z_dbuf: DeviceBuffer[Self.dtype]  # [RB * HIDDEN_DIM] (reserved)

    # Per-sample elite selection (matches MBPO).
    var r_elite_idx_per_sample: DeviceBuffer[DType.int32]  # [RB]
    var r_elite_rng: DeviceBuffer[DType.uint32]  # [1]
    # Map elite-slot -> ensemble-member index. Re-uploaded from
    # `elite_indices` after every train round (see `sync_elite_member_buf`).
    var elite_member_buf: DeviceBuffer[DType.int32]  # [NUM_ELITES]
    var elite_member_host: HostBuffer[DType.int32]  # [NUM_ELITES]

    # Actor workspace (caller sizes via R_WS_SIZE). Used by both actor and
    # any rollout-time forward that needs scratch.
    var r_ws: DeviceBuffer[Self.dtype]  # [R_WS_SIZE]

    # Replay-buffer sample staging — shared by training and rollout sample
    # paths. Sized for max(DYN_BATCH, ROLLOUT_BATCH) so both fit.
    var s_obs: DeviceBuffer[Self.dtype]  # [SB * obs_dim]
    var s_act: DeviceBuffer[Self.dtype]  # [SB * action_dim]
    var s_rew: DeviceBuffer[Self.dtype]  # [SB]
    var s_nobs: DeviceBuffer[Self.dtype]  # [SB * obs_dim]
    var s_done: DeviceBuffer[Self.dtype]  # [SB]
    var s_idx: DeviceBuffer[DType.int32]  # [SB]

    # Input scaler — initialized to identity so callers that don't fit a
    # scaler get pass-through normalization. Matches MBPO's behavior on
    # the first dynamics train (scaler refit there). Sized DYN_IN.
    var input_mean: DeviceBuffer[Self.dtype]  # [DYN_IN]
    var input_std: DeviceBuffer[Self.dtype]  # [DYN_IN]
    # Reward target scaler — fitted on the buffer's reward column. PCN's
    # unweighted MSE loss treats every output dim equally, so a single
    # high-variance reward dim (HalfCheetah: σ ≈ 1, vs Δobs σ ≈ 0.05)
    # gets averaged-out by 17 obs dims. Without this, PCN regresses to
    # mean-reward-of-warmup (≈ −0.05) and never learns the policy-induced
    # forward-velocity reward signal — SAC then bootstraps Q-values from
    # ~zero-reward synth and the policy never improves. Both size [1].
    var reward_mean: DeviceBuffer[Self.dtype]
    var reward_std: DeviceBuffer[Self.dtype]

    # =========================================================================
    # Construction / destruction.
    # =========================================================================

    def __init__(
        out self, ctx: DeviceContext, base_seed: UInt64 = UInt64(7)
    ) raises:
        """Allocate all GPU buffers and init members on host → upload."""
        # Param buffers.
        self.params_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.ENS.TOTAL_PARAM_SIZE
        )
        self.grads_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.ENS.TOTAL_PARAM_SIZE
        )
        self.opt_state_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM
        )
        self.opt_global_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE
        )

        # Init params on host, copy to device. Adam state starts at 0 (the
        # Optimizer.step_gpu preamble bumps the on-device step counter).
        var params_init = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.ENS.TOTAL_PARAM_SIZE
        )
        Self.ENS.init_all_host(params_init.unsafe_ptr(), base_seed=base_seed)
        ctx.enqueue_copy(self.params_dbuf, params_init)

        var opt_state_init = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM
        )
        for i in range(Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM):
            opt_state_init.unsafe_ptr()[i] = Scalar[Self.dtype](0)
        ctx.enqueue_copy(self.opt_state_dbuf, opt_state_init)

        var opt_global_init = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE
        )
        for m in range(Self.NUM_ENSEMBLE):
            opt_global_init.unsafe_ptr()[
                m * Self.OPT.GLOBAL_STATE_SIZE
            ] = Scalar[Self.dtype](
                0
            )  # step counter (bit-pattern UInt32)
            opt_global_init.unsafe_ptr()[
                m * Self.OPT.GLOBAL_STATE_SIZE + 1
            ] = Scalar[Self.dtype](
                1.0
            )  # lr_scale
        ctx.enqueue_copy(self.opt_global_dbuf, opt_global_init)

        # Step counters and elite indices on host.
        self.step_nums = List[Int](capacity=Self.NUM_ENSEMBLE)
        for _ in range(Self.NUM_ENSEMBLE):
            self.step_nums.append(0)
        self.elite_indices = List[Int](capacity=Self.NUM_ELITES)
        for i in range(Self.NUM_ELITES):
            self.elite_indices.append(i)

        # SGLD scratch.
        self.lat_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.SCRATCH_LAT
        )
        self.mu_eps_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.SCRATCH_OUT
        )
        self.a_below_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.SCRATCH_IN
        )
        self.z_below_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.SCRATCH_IN
        )
        self.dx_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.SCRATCH_LAT
        )

        # Train-batch input/target staging.
        self.s_a_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.AUG_DIM
        )
        self.target_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.READOUT
        )
        self.s_a_host = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.AUG_DIM
        )
        self.target_host = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.READOUT
        )

        # Holdout eval scratch (DYN_BATCH).
        self.e_a_aug_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.AUG_DIM
        )
        self.e_z_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.HIDDEN_DIM
        )
        self.e_a_z_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.HIDDEN_DIM
        )
        self.e_out_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.READOUT
        )
        self.e_out_host = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.READOUT
        )
        self.e_target_host = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.DYN_BATCH * Self.DYN.READOUT
        )

        # Rollout state (MBPO-compatible field names).
        self.r_obs = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.OBS_DIM
        )
        self.r_next_obs = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.OBS_DIM
        )
        self.r_actions = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.ACTION_DIM
        )
        self.r_rewards = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH
        )
        self.r_dones = ctx.enqueue_create_buffer[Self.dtype](Self.ROLLOUT_BATCH)
        self.r_alive = ctx.enqueue_create_buffer[Self.dtype](Self.ROLLOUT_BATCH)

        # Rollout dynamics input/output (renamed from r_s_a_dbuf / r_out_dbuf).
        self.r_dyn_input = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.DYN.AUG_DIM
        )
        self.r_dyn_output = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.DYN.READOUT
        )
        self.r_dyn_output_all = ctx.enqueue_create_buffer[Self.dtype](
            Self.NUM_ELITES * Self.ROLLOUT_BATCH * Self.DYN.READOUT
        )

        # PCN forward scratch sized for `forward_eval_gpu` (whole-chain
        # SCRATCH_OUT / SCRATCH_IN, not just one block's AUG/HIDDEN).
        self.r_a_aug_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.DYN.SCRATCH_OUT
        )
        self.r_z_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.DYN.SCRATCH_IN
        )
        self.r_a_z_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.DYN.HIDDEN_DIM
        )

        # Per-sample elite selection (default mapping = first NUM_ELITES
        # members; refreshed each train round via `sync_elite_member_buf`).
        self.r_elite_idx_per_sample = ctx.enqueue_create_buffer[DType.int32](
            Self.ROLLOUT_BATCH
        )
        self.r_elite_rng = ctx.enqueue_create_buffer[DType.uint32](1)
        self.r_elite_rng.enqueue_fill(UInt32(0xC0FFEE))
        self.elite_member_buf = ctx.enqueue_create_buffer[DType.int32](
            Self.NUM_ELITES
        )
        self.elite_member_host = ctx.enqueue_create_host_buffer[DType.int32](
            Self.NUM_ELITES
        )
        for i in range(Self.NUM_ELITES):
            self.elite_member_host[i] = Int32(i)
        ctx.enqueue_copy(self.elite_member_buf, self.elite_member_host)

        # Actor workspace.
        self.r_ws = ctx.enqueue_create_buffer[Self.dtype](Self.R_WS_SIZE)

        # Replay-buffer sample staging — shared between training-batch and
        # rollout-start sampling.
        self.s_obs = ctx.enqueue_create_buffer[Self.dtype](
            Self.SAMPLE_BATCH * Self.OBS_DIM
        )
        self.s_act = ctx.enqueue_create_buffer[Self.dtype](
            Self.SAMPLE_BATCH * Self.ACTION_DIM
        )
        self.s_rew = ctx.enqueue_create_buffer[Self.dtype](Self.SAMPLE_BATCH)
        self.s_nobs = ctx.enqueue_create_buffer[Self.dtype](
            Self.SAMPLE_BATCH * Self.OBS_DIM
        )
        self.s_done = ctx.enqueue_create_buffer[Self.dtype](Self.SAMPLE_BATCH)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](Self.SAMPLE_BATCH)

        # Input scaler — start at identity. Caller may overwrite via
        # `enqueue_fill` or by copying from a fitted scaler.
        self.input_mean = ctx.enqueue_create_buffer[Self.dtype](
            Self.DYN.AUG_DIM
        )
        self.input_std = ctx.enqueue_create_buffer[Self.dtype](Self.DYN.AUG_DIM)
        self.input_mean.enqueue_fill(Scalar[Self.dtype](0.0))
        self.input_std.enqueue_fill(Scalar[Self.dtype](1.0))

        # Reward scaler — start at identity (no-op normalization). Caller
        # refits via `fit_reward_scaler_gpu` each `train_dynamics_gpu` call.
        self.reward_mean = ctx.enqueue_create_buffer[Self.dtype](1)
        self.reward_std = ctx.enqueue_create_buffer[Self.dtype](1)
        self.reward_mean.enqueue_fill(Scalar[Self.dtype](0.0))
        self.reward_std.enqueue_fill(Scalar[Self.dtype](1.0))

        ctx.synchronize()

    def __init__(out self, *, deinit take: Self):
        self.params_dbuf = take.params_dbuf^
        self.grads_dbuf = take.grads_dbuf^
        self.opt_state_dbuf = take.opt_state_dbuf^
        self.opt_global_dbuf = take.opt_global_dbuf^
        self.step_nums = take.step_nums^
        self.elite_indices = take.elite_indices^
        self.lat_dbuf = take.lat_dbuf^
        self.mu_eps_dbuf = take.mu_eps_dbuf^
        self.a_below_dbuf = take.a_below_dbuf^
        self.z_below_dbuf = take.z_below_dbuf^
        self.dx_dbuf = take.dx_dbuf^
        self.s_a_dbuf = take.s_a_dbuf^
        self.target_dbuf = take.target_dbuf^
        self.s_a_host = take.s_a_host^
        self.target_host = take.target_host^
        self.e_a_aug_dbuf = take.e_a_aug_dbuf^
        self.e_z_dbuf = take.e_z_dbuf^
        self.e_a_z_dbuf = take.e_a_z_dbuf^
        self.e_out_dbuf = take.e_out_dbuf^
        self.e_out_host = take.e_out_host^
        self.e_target_host = take.e_target_host^
        # Rollout state (MBPO-compatible).
        self.r_obs = take.r_obs^
        self.r_next_obs = take.r_next_obs^
        self.r_actions = take.r_actions^
        self.r_rewards = take.r_rewards^
        self.r_dones = take.r_dones^
        self.r_alive = take.r_alive^
        self.r_dyn_input = take.r_dyn_input^
        self.r_dyn_output = take.r_dyn_output^
        self.r_dyn_output_all = take.r_dyn_output_all^
        self.r_a_aug_dbuf = take.r_a_aug_dbuf^
        self.r_z_dbuf = take.r_z_dbuf^
        self.r_a_z_dbuf = take.r_a_z_dbuf^
        self.r_elite_idx_per_sample = take.r_elite_idx_per_sample^
        self.r_elite_rng = take.r_elite_rng^
        self.elite_member_buf = take.elite_member_buf^
        self.elite_member_host = take.elite_member_host^
        self.r_ws = take.r_ws^
        # Replay sample staging.
        self.s_obs = take.s_obs^
        self.s_act = take.s_act^
        self.s_rew = take.s_rew^
        self.s_nobs = take.s_nobs^
        self.s_done = take.s_done^
        self.s_idx = take.s_idx^
        # Input scaler.
        self.input_mean = take.input_mean^
        self.input_std = take.input_std^
        # Reward scaler.
        self.reward_mean = take.reward_mean^
        self.reward_std = take.reward_std^

    # =========================================================================
    # Common LayoutTensor views (build once per call from device buffers).
    # Helpers to avoid copy-paste in the train / eval / rollout entry points.
    # =========================================================================

    def _lat_view(
        self,
    ) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_LAT),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_LAT),
            MutAnyOrigin,
        ](self.lat_dbuf.unsafe_ptr())

    def _mu_eps_view(
        self,
    ) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_OUT),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_OUT),
            MutAnyOrigin,
        ](self.mu_eps_dbuf.unsafe_ptr())

    def _a_below_view(
        self,
    ) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_IN),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_IN),
            MutAnyOrigin,
        ](self.a_below_dbuf.unsafe_ptr())

    def _z_below_view(
        self,
    ) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_IN),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_IN),
            MutAnyOrigin,
        ](self.z_below_dbuf.unsafe_ptr())

    def _dx_view(
        self,
    ) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_LAT),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_LAT),
            MutAnyOrigin,
        ](self.dx_dbuf.unsafe_ptr())

    def _s_a_view(
        self,
    ) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.AUG_DIM),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.AUG_DIM),
            MutAnyOrigin,
        ](self.s_a_dbuf.unsafe_ptr())

    def _target_view(
        self,
    ) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.READOUT),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.READOUT),
            MutAnyOrigin,
        ](self.target_dbuf.unsafe_ptr())

    # =========================================================================
    # Train one minibatch — assumes caller has already filled `s_a_host` /
    # `target_host` and copied to device. Trains member `m` for one Adam step.
    # =========================================================================

    def train_one_minibatch(
        mut self,
        ctx: DeviceContext,
        m: Int,
    ) raises:
        """Train member m on the current contents of `s_a_dbuf` / `target_dbuf`.

        Increments the per-member step counter; calls
        `PCDynamicsEnsembleGPU.train_member_gpu` which does SGLD inference,
        PC weight grads, and Adam.step on device.
        """
        var s_a = self._s_a_view()
        var target = self._target_view()
        var lat = self._lat_view()
        var mu_eps = self._mu_eps_view()
        var a_below = self._a_below_view()
        var z_below = self._z_below_view()
        var dx = self._dx_view()
        Self.ENS.train_member_gpu[Self.DYN_BATCH, Self.OPT](
            ctx,
            m,
            s_a,
            target,
            self.params_dbuf.unsafe_ptr(),
            self.grads_dbuf.unsafe_ptr(),
            self.opt_state_dbuf.unsafe_ptr(),
            self.opt_global_dbuf.unsafe_ptr(),
            lat,
            mu_eps,
            a_below,
            z_below,
            dx,
            self.step_nums[m],
            T_infer=Self.T_INFER,
            lr_x=Scalar[Self.dtype](Self.LR_X_FLOAT),
        )

    # =========================================================================
    # Holdout eval — feedforward member m on (s_a_dbuf, target_dbuf), then
    # download outputs + targets to host and compute MSE. Returns Float64
    # mean MSE over BATCH × READOUT.
    # =========================================================================

    def eval_member_holdout_loss(
        mut self,
        ctx: DeviceContext,
        m: Int,
    ) raises -> Float64:
        var s_a = self._s_a_view()
        var e_a_aug_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.AUG_DIM),
            MutAnyOrigin,
        ](self.e_a_aug_dbuf.unsafe_ptr())
        var e_z_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.HIDDEN_DIM),
            MutAnyOrigin,
        ](self.e_z_dbuf.unsafe_ptr())
        var e_a_z_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.HIDDEN_DIM),
            MutAnyOrigin,
        ](self.e_a_z_dbuf.unsafe_ptr())
        var e_out_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.READOUT),
            MutAnyOrigin,
        ](self.e_out_dbuf.unsafe_ptr())
        # Reuse mu_eps/a_below as the predict scratch (their shapes are
        # already DYN_BATCH × HIDDEN-or-bigger, which is fine for predict).
        var pred_mu = self._mu_eps_view()
        var pred_a = self._a_below_view()
        Self.ENS.predict_member_gpu[Self.DYN_BATCH](
            ctx,
            m,
            s_a,
            self.params_dbuf.unsafe_ptr(),
            pred_mu,
            pred_a,
            e_out_t,
        )
        # Download outputs + targets, MSE on host.
        ctx.enqueue_copy(self.e_out_host, self.e_out_dbuf)
        ctx.enqueue_copy(self.e_target_host, self.target_dbuf)
        ctx.synchronize()
        var sum_sq: Float64 = 0.0
        for b in range(Self.DYN_BATCH):
            for d in range(Self.DYN.READOUT):
                var p = Float64(
                    self.e_out_host.unsafe_ptr()[b * Self.DYN.READOUT + d]
                )
                var t = Float64(
                    self.e_target_host.unsafe_ptr()[b * Self.DYN.READOUT + d]
                )
                var diff = p - t
                sum_sq += diff * diff
        return sum_sq / Float64(Self.DYN_BATCH * Self.DYN.READOUT)

    def eval_member_holdout_mse_breakdown(
        mut self,
        ctx: DeviceContext,
        m: Int,
    ) raises -> Tuple[Float64, Float64, Float64]:
        """Same as `eval_member_holdout_loss` but splits MSE into (total,
        obs-only, reward-only) so the agent can log them separately.

        Returns `(mse_total, mse_obs, mse_reward)` where:
          mse_obs    = mean over (BATCH × OBS_DIM) of (out[d<OBS_DIM] - tgt[d])²
          mse_reward = mean over BATCH       of (out[OBS_DIM] - tgt[OBS_DIM])²
          mse_total  = mean over (BATCH × READOUT) of (out[d] - tgt[d])²

        Aggregate MSE on its own can hide a per-dim disaster: 17 obs dims
        averaged with 1 reward dim weight reward at 1/18, so a reward MSE
        of 5 with obs MSE of 0.2 produces aggregate ≈ 0.47 — looks fine
        but rewards are useless. This split surfaces that.
        """
        var s_a = self._s_a_view()
        var e_out_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.READOUT),
            MutAnyOrigin,
        ](self.e_out_dbuf.unsafe_ptr())
        var pred_mu = self._mu_eps_view()
        var pred_a = self._a_below_view()
        Self.ENS.predict_member_gpu[Self.DYN_BATCH](
            ctx, m, s_a, self.params_dbuf.unsafe_ptr(),
            pred_mu, pred_a, e_out_t,
        )
        ctx.enqueue_copy(self.e_out_host, self.e_out_dbuf)
        ctx.enqueue_copy(self.e_target_host, self.target_dbuf)
        ctx.synchronize()
        var sum_sq_obs: Float64 = 0.0
        var sum_sq_rew: Float64 = 0.0
        for b in range(Self.DYN_BATCH):
            var row = b * Self.DYN.READOUT
            for d in range(Self.OBS_DIM):
                var p = Float64(self.e_out_host.unsafe_ptr()[row + d])
                var t = Float64(self.e_target_host.unsafe_ptr()[row + d])
                var diff = p - t
                sum_sq_obs += diff * diff
            # Reward dim is at index OBS_DIM (= READOUT - 1).
            var p_r = Float64(
                self.e_out_host.unsafe_ptr()[row + Self.OBS_DIM]
            )
            var t_r = Float64(
                self.e_target_host.unsafe_ptr()[row + Self.OBS_DIM]
            )
            var diff_r = p_r - t_r
            sum_sq_rew += diff_r * diff_r
        var mse_obs = sum_sq_obs / Float64(
            Self.DYN_BATCH * Self.OBS_DIM
        )
        var mse_rew = sum_sq_rew / Float64(Self.DYN_BATCH)
        var mse_total = (sum_sq_obs + sum_sq_rew) / Float64(
            Self.DYN_BATCH * Self.DYN.READOUT
        )
        return (mse_total, mse_obs, mse_rew)

    # =========================================================================
    # Refresh elite_indices using the current training batch's holdout MSE.
    # Caller is expected to have filled s_a_dbuf/target_dbuf with a holdout
    # slice before calling.
    # =========================================================================

    def refresh_elites(mut self, ctx: DeviceContext) raises:
        var losses = List[Float64](capacity=Self.NUM_ENSEMBLE)
        for m in range(Self.NUM_ENSEMBLE):
            losses.append(self.eval_member_holdout_loss(ctx, m))
        Self.ENS.select_elites(losses, self.elite_indices)

    # =========================================================================
    # Rollout helpers — for `do_model_rollouts_gpu` in the agent fork.
    # Caller fills `r_s_a_dbuf` (the [obs|action] inputs for ROLLOUT_BATCH
    # samples), then calls `predict_rollout_member` to feedforward through
    # one ensemble member (chosen by the caller per slot or per call).
    # =========================================================================

    def predict_rollout_member(
        mut self,
        ctx: DeviceContext,
        m: Int,
    ) raises:
        """Feedforward member m over ROLLOUT_BATCH inputs in `r_dyn_input`.

        Result is left in `r_dyn_output` (caller un-normalizes + stores).
        """
        var r_s_a_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.AUG_DIM),
            MutAnyOrigin,
        ](self.r_dyn_input.unsafe_ptr())
        var r_mu_buf_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.SCRATCH_OUT),
            MutAnyOrigin,
        ](self.r_a_aug_dbuf.unsafe_ptr())
        var r_a_buf_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.SCRATCH_IN),
            MutAnyOrigin,
        ](self.r_z_dbuf.unsafe_ptr())
        var r_out_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.READOUT),
            MutAnyOrigin,
        ](self.r_dyn_output.unsafe_ptr())
        Self.ENS.predict_member_gpu[Self.ROLLOUT_BATCH](
            ctx,
            m,
            r_s_a_t,
            self.params_dbuf.unsafe_ptr(),
            r_mu_buf_t,
            r_a_buf_t,
            r_out_t,
        )

    # =========================================================================
    # Predict member m into a specific slot of r_dyn_output_all. Used by the
    # agent's per-elite forward loop in `do_model_rollouts_gpu` so each
    # elite member writes its forward into a different slot for per-sample
    # elite selection.
    # =========================================================================

    def predict_rollout_member_into_slot(
        mut self,
        ctx: DeviceContext,
        m: Int,
        slot: Int,
    ) raises:
        """Feedforward member m into `r_dyn_output_all[slot, :, :]`.

        Layout: [NUM_ELITES, ROLLOUT_BATCH, READOUT]; member-output is
        written at offset `slot * ROLLOUT_BATCH * READOUT`.
        """
        var r_s_a_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.AUG_DIM),
            MutAnyOrigin,
        ](self.r_dyn_input.unsafe_ptr())
        var r_mu_buf_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.SCRATCH_OUT),
            MutAnyOrigin,
        ](self.r_a_aug_dbuf.unsafe_ptr())
        var r_a_buf_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.SCRATCH_IN),
            MutAnyOrigin,
        ](self.r_z_dbuf.unsafe_ptr())
        var slot_ptr = (
            self.r_dyn_output_all.unsafe_ptr()
            + slot * Self.ROLLOUT_BATCH * Self.DYN.READOUT
        )
        var slot_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.READOUT),
            MutAnyOrigin,
        ](slot_ptr)
        Self.ENS.predict_member_gpu[Self.ROLLOUT_BATCH](
            ctx,
            m,
            r_s_a_t,
            self.params_dbuf.unsafe_ptr(),
            r_mu_buf_t,
            r_a_buf_t,
            slot_t,
        )

    # =========================================================================
    # Re-upload elite-slot -> ensemble-member mapping after a train round.
    # Mirrors MBPO's pattern (elite_member_buf is consumed by the rollout
    # sample kernel for per-sample elite selection).
    # =========================================================================

    def sync_elite_member_buf(mut self, ctx: DeviceContext) raises:
        """Refresh `elite_member_buf` from the host `elite_indices` list."""
        for i in range(Self.NUM_ELITES):
            self.elite_member_host[i] = Int32(self.elite_indices[i])
        ctx.enqueue_copy(self.elite_member_buf, self.elite_member_host)

    # =========================================================================
    # Input scaler — fits per-dim mean/std over (obs, action) on the agent's
    # real GPU buffer. Matches vanilla MBPO's `TensorStandardScaler.fit`
    # (bnn.py:335). Called once per dynamics-train round so every per-member
    # train minibatch + every rollout step within the round uses the same
    # scaler. Without it BLOCK0's `tanh(x_below)` saturates on raw HalfCheetah
    # obs (positions ±10, velocities ±10) and the dynamics learns garbage.
    # =========================================================================

    def fit_scaler_gpu[
        BUF_CAP: Int,
    ](
        mut self,
        ctx: DeviceContext,
        buffer: GPUReplayBuffer[BUF_CAP, Self.OBS_DIM, Self.ACTION_DIM],
    ) raises:
        """Compute per-dim mean and std of [obs || act] over the populated
        buffer. Stores into `input_mean` (sized DYN_IN) and `input_std`."""
        var n = buffer.size
        if n < 1:
            return

        var mean_obs_t = LayoutTensor[
            Self.dtype, Layout.row_major(Self.OBS_DIM), MutAnyOrigin
        ](self.input_mean.unsafe_ptr())
        var mean_act_t = LayoutTensor[
            Self.dtype, Layout.row_major(Self.ACTION_DIM), MutAnyOrigin
        ](self.input_mean.unsafe_ptr() + Self.OBS_DIM)
        var std_obs_t = LayoutTensor[
            Self.dtype, Layout.row_major(Self.OBS_DIM), MutAnyOrigin
        ](self.input_std.unsafe_ptr())
        var std_act_t = LayoutTensor[
            Self.dtype, Layout.row_major(Self.ACTION_DIM), MutAnyOrigin
        ](self.input_std.unsafe_ptr() + Self.OBS_DIM)
        # GPUReplayBuffer uses the global `dtype` from `mojo_rl.nn.constants`
        # for its underlying DeviceBuffer; rebind so the LayoutTensor view
        # matches `Self.dtype` (which is also the global by default but the
        # type checker treats them as distinct aliases).
        var obs_data_t = LayoutTensor[
            Self.dtype, Layout.row_major(BUF_CAP, Self.OBS_DIM), MutAnyOrigin
        ](
            rebind[UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]](
                buffer.states_buf.unsafe_ptr()
            )
        )
        var act_data_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(BUF_CAP, Self.ACTION_DIM),
            MutAnyOrigin,
        ](
            rebind[UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]](
                buffer.actions_buf.unsafe_ptr()
            )
        )

        # Pass 1: per-dim means on populated rows [0, n).
        comptime obs_mean_k = compute_scaler_mean_kernel[
            Self.dtype, BUF_CAP, Self.OBS_DIM
        ]
        comptime act_mean_k = compute_scaler_mean_kernel[
            Self.dtype, BUF_CAP, Self.ACTION_DIM
        ]
        ctx.enqueue_function[obs_mean_k](
            mean_obs_t,
            obs_data_t,
            n,
            grid_dim=(Self.OBS_DIM,),
            block_dim=(1,),
        )
        ctx.enqueue_function[act_mean_k](
            mean_act_t,
            act_data_t,
            n,
            grid_dim=(Self.ACTION_DIM,),
            block_dim=(1,),
        )

        # Pass 2: per-dim stds (need means already on device).
        var min_std = Scalar[Self.dtype](1e-12)
        comptime obs_std_k = compute_scaler_std_kernel[
            Self.dtype, BUF_CAP, Self.OBS_DIM
        ]
        comptime act_std_k = compute_scaler_std_kernel[
            Self.dtype, BUF_CAP, Self.ACTION_DIM
        ]
        ctx.enqueue_function[obs_std_k](
            std_obs_t,
            obs_data_t,
            mean_obs_t,
            n,
            min_std,
            grid_dim=(Self.OBS_DIM,),
            block_dim=(1,),
        )
        ctx.enqueue_function[act_std_k](
            std_act_t,
            act_data_t,
            mean_act_t,
            n,
            min_std,
            grid_dim=(Self.ACTION_DIM,),
            block_dim=(1,),
        )

    def fit_reward_scaler_gpu[
        BUF_CAP: Int,
    ](
        mut self,
        ctx: DeviceContext,
        buffer: GPUReplayBuffer[BUF_CAP, Self.OBS_DIM, Self.ACTION_DIM],
    ) raises:
        """Compute mean and std of the buffer's reward column. Stores into
        `reward_mean` and `reward_std` (each size [1]).

        Used to normalize the reward target during PCN training and
        un-normalize during rollouts. Without this, PCN's unweighted-MSE
        loss treats the (~σ=1) reward dim equally with 17 (~σ=0.05) Δobs
        dims; reward effectively gets 1/18 weight in the gradient and
        regresses to mean — synth rollouts then have reward ≈ 0 regardless
        of policy, and SAC bootstraps Q without a real reward signal.

        Reuses the existing `compute_scaler_*_kernel` with D=1 by viewing
        rewards as a [BUF_CAP, 1] tensor.
        """
        var n = buffer.size
        if n < 1:
            return

        var rew_mean_t = LayoutTensor[
            Self.dtype, Layout.row_major(1), MutAnyOrigin
        ](self.reward_mean.unsafe_ptr())
        var rew_std_t = LayoutTensor[
            Self.dtype, Layout.row_major(1), MutAnyOrigin
        ](self.reward_std.unsafe_ptr())
        # Buffer's reward column is a flat 1-D buffer; view as [BUF_CAP, 1].
        var rew_data_t = LayoutTensor[
            Self.dtype, Layout.row_major(BUF_CAP, 1), MutAnyOrigin
        ](
            rebind[UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]](
                buffer.rewards_buf.unsafe_ptr()
            )
        )

        comptime rew_mean_k = compute_scaler_mean_kernel[
            Self.dtype, BUF_CAP, 1
        ]
        comptime rew_std_k = compute_scaler_std_kernel[
            Self.dtype, BUF_CAP, 1
        ]
        ctx.enqueue_function[rew_mean_k](
            rew_mean_t, rew_data_t, n,
            grid_dim=(1,), block_dim=(1,),
        )
        var min_std = Scalar[Self.dtype](1e-12)
        ctx.enqueue_function[rew_std_k](
            rew_std_t, rew_data_t, rew_mean_t, n, min_std,
            grid_dim=(1,), block_dim=(1,),
        )

    # =========================================================================
    # Health metrics — small reductions read back from device for logging.
    # =========================================================================

    def download_input_std(mut self, ctx: DeviceContext) raises -> Float64:
        """Return mean of `input_std` across DYN_IN dims (host-side scalar).

        Useful for tracking input-scale drift over training. A healthy
        scaler should produce input_std around 1 after the first fit (we
        normalize by it); the *raw* per-dim stds before normalization
        live in the buffer's own data, not here.
        """
        var div = Self.OBS_DIM + Self.ACTION_DIM
        var host = ctx.enqueue_create_host_buffer[Self.dtype](div)
        ctx.enqueue_copy(host, self.input_std)
        ctx.synchronize()
        var s: Float64 = 0.0
        for i in range(div):
            s += Float64(host.unsafe_ptr()[i])
        return s / Float64(div)

    # =========================================================================
    # Checkpoint surface — mirrors `PCDynamicsEnsembleInstanceCPU`. Downloads
    # device-resident params + Adam state to host buffers, then emits the
    # same section format. On read, parses sections and uploads back.
    # =========================================================================

    def write_sections(
        mut self, ctx: DeviceContext, prefix: String
    ) raises -> String:
        """Serialize ensemble (params + Adam state + step counters) as text
        sections. Format identical to `PCDynamicsEnsembleInstanceCPU`."""
        var params_h = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.ENS.TOTAL_PARAM_SIZE
        )
        var opt_state_h = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM
        )
        var opt_global_h = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE
        )
        ctx.enqueue_copy(params_h, self.params_dbuf)
        ctx.enqueue_copy(opt_state_h, self.opt_state_dbuf)
        ctx.enqueue_copy(opt_global_h, self.opt_global_dbuf)
        ctx.synchronize()

        var content = write_float_section_ptr(
            prefix + "params:",
            params_h.unsafe_ptr(),
            Self.ENS.TOTAL_PARAM_SIZE,
        )
        content += write_float_section_ptr(
            prefix + "opt_state:",
            opt_state_h.unsafe_ptr(),
            Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM,
        )
        content += write_float_section_ptr(
            prefix + "opt_global:",
            opt_global_h.unsafe_ptr(),
            Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE,
        )
        var steps = prefix + "step_nums:\n"
        for m in range(Self.NUM_ENSEMBLE):
            steps += String(self.step_nums[m]) + "\n"
        content += steps
        return content

    def read_sections(
        mut self, ctx: DeviceContext, content: String, prefix: String
    ) raises:
        """Restore ensemble from sections written by `write_sections`."""
        var loaded_params = read_float_section_list[Self.dtype](
            content, prefix + "params:", Self.ENS.TOTAL_PARAM_SIZE
        )
        var loaded_opt = read_float_section_list[Self.dtype](
            content,
            prefix + "opt_state:",
            Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM,
        )
        var loaded_global = read_float_section_list[Self.dtype](
            content,
            prefix + "opt_global:",
            Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE,
        )
        var loaded_steps = read_float_section_list[Self.dtype](
            content, prefix + "step_nums:", Self.NUM_ENSEMBLE
        )

        var params_h = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.ENS.TOTAL_PARAM_SIZE
        )
        var opt_state_h = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM
        )
        var opt_global_h = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE
        )
        for i in range(Self.ENS.TOTAL_PARAM_SIZE):
            params_h.unsafe_ptr()[i] = loaded_params[i]
        for i in range(Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM):
            opt_state_h.unsafe_ptr()[i] = loaded_opt[i]
        for i in range(Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE):
            opt_global_h.unsafe_ptr()[i] = loaded_global[i]
        ctx.enqueue_copy(self.params_dbuf, params_h)
        ctx.enqueue_copy(self.opt_state_dbuf, opt_state_h)
        ctx.enqueue_copy(self.opt_global_dbuf, opt_global_h)
        ctx.synchronize()

        for m in range(Self.NUM_ENSEMBLE):
            self.step_nums[m] = Int(Float64(loaded_steps[m]))
