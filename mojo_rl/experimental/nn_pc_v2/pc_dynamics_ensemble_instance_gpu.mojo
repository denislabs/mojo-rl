"""PCDynamicsEnsembleInstanceGPU — owning GPU wrapper for PCN dynamics ensemble.

Mirrors `PCDynamicsEnsembleInstanceCPU` but lives on `DeviceBuffer`s and
routes everything through nn_pc_v2's GPU primitives. Designed to be a
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
    dtype: DType = DType.float32,
](Movable):
    """Owning GPU wrapper around `PCDynamicsEnsembleGPU`.

    Field naming mirrors `GPUDynamicsEnsemble` so the PCN-MBPO agent fork
    can read this struct's members with minimal renames. Methods are
    leaner — no logvar / NLL machinery.
    """

    comptime ENS = PCDynamicsEnsembleGPU[
        Self.OBS_DIM, Self.ACTION_DIM, Self.HIDDEN_DIM,
        Self.NUM_ENSEMBLE, Self.NUM_ELITES, Self.dtype,
    ]
    comptime DYN = Self.ENS.DYN
    comptime OPT = Adam[LR=Self.DYN_LR]

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

    # Rollout scratch (ROLLOUT_BATCH-sized — used by agent's
    # do_model_rollouts_gpu in the forked agent).
    var r_s_a_dbuf: DeviceBuffer[Self.dtype]
    var r_a_aug_dbuf: DeviceBuffer[Self.dtype]
    var r_z_dbuf: DeviceBuffer[Self.dtype]
    var r_a_z_dbuf: DeviceBuffer[Self.dtype]
    var r_out_dbuf: DeviceBuffer[Self.dtype]

    # =========================================================================
    # Construction / destruction.
    # =========================================================================

    def __init__(out self, ctx: DeviceContext, base_seed: UInt64 = UInt64(7)) raises:
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
        for i in range(
            Self.ENS.TOTAL_PARAM_SIZE * Self.OPT.STATE_PER_PARAM
        ):
            opt_state_init.unsafe_ptr()[i] = Scalar[Self.dtype](0)
        ctx.enqueue_copy(self.opt_state_dbuf, opt_state_init)

        var opt_global_init = ctx.enqueue_create_host_buffer[Self.dtype](
            Self.NUM_ENSEMBLE * Self.OPT.GLOBAL_STATE_SIZE
        )
        for m in range(Self.NUM_ENSEMBLE):
            opt_global_init.unsafe_ptr()[m * Self.OPT.GLOBAL_STATE_SIZE] = (
                Scalar[Self.dtype](0)  # step counter (bit-pattern UInt32)
            )
            opt_global_init.unsafe_ptr()[
                m * Self.OPT.GLOBAL_STATE_SIZE + 1
            ] = Scalar[Self.dtype](1.0)  # lr_scale
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

        # Rollout scratch (ROLLOUT_BATCH).
        self.r_s_a_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.DYN.AUG_DIM
        )
        self.r_a_aug_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.DYN.AUG_DIM
        )
        self.r_z_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.DYN.HIDDEN_DIM
        )
        self.r_a_z_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.DYN.HIDDEN_DIM
        )
        self.r_out_dbuf = ctx.enqueue_create_buffer[Self.dtype](
            Self.ROLLOUT_BATCH * Self.DYN.READOUT
        )

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
        self.r_s_a_dbuf = take.r_s_a_dbuf^
        self.r_a_aug_dbuf = take.r_a_aug_dbuf^
        self.r_z_dbuf = take.r_z_dbuf^
        self.r_a_z_dbuf = take.r_a_z_dbuf^
        self.r_out_dbuf = take.r_out_dbuf^

    # =========================================================================
    # Common LayoutTensor views (build once per call from device buffers).
    # Helpers to avoid copy-paste in the train / eval / rollout entry points.
    # =========================================================================

    fn _lat_view(self) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_LAT),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_LAT),
            MutAnyOrigin,
        ](self.lat_dbuf.unsafe_ptr())

    fn _mu_eps_view(self) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_OUT),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_OUT),
            MutAnyOrigin,
        ](self.mu_eps_dbuf.unsafe_ptr())

    fn _a_below_view(self) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_IN),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_IN),
            MutAnyOrigin,
        ](self.a_below_dbuf.unsafe_ptr())

    fn _z_below_view(self) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_IN),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_IN),
            MutAnyOrigin,
        ](self.z_below_dbuf.unsafe_ptr())

    fn _dx_view(self) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_LAT),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.SCRATCH_LAT),
            MutAnyOrigin,
        ](self.dx_dbuf.unsafe_ptr())

    fn _s_a_view(self) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.DYN_BATCH, Self.DYN.AUG_DIM),
        MutAnyOrigin,
    ]:
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.DYN_BATCH, Self.DYN.AUG_DIM),
            MutAnyOrigin,
        ](self.s_a_dbuf.unsafe_ptr())

    fn _target_view(self) -> LayoutTensor[
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
            ctx, m, s_a, target,
            self.params_dbuf.unsafe_ptr(),
            self.grads_dbuf.unsafe_ptr(),
            self.opt_state_dbuf.unsafe_ptr(),
            self.opt_global_dbuf.unsafe_ptr(),
            lat, mu_eps, a_below, z_below, dx,
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
            ctx, m, s_a, self.params_dbuf.unsafe_ptr(),
            pred_mu, pred_a, e_out_t,
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
        """Feedforward member m over ROLLOUT_BATCH inputs in `r_s_a_dbuf`.

        Result is left in `r_out_dbuf` (caller un-normalizes + stores).
        """
        var r_s_a_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.AUG_DIM),
            MutAnyOrigin,
        ](self.r_s_a_dbuf.unsafe_ptr())
        var r_a_aug_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.AUG_DIM),
            MutAnyOrigin,
        ](self.r_a_aug_dbuf.unsafe_ptr())
        var r_z_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.HIDDEN_DIM),
            MutAnyOrigin,
        ](self.r_z_dbuf.unsafe_ptr())
        var r_a_z_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.HIDDEN_DIM),
            MutAnyOrigin,
        ](self.r_a_z_dbuf.unsafe_ptr())
        var r_out_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.ROLLOUT_BATCH, Self.DYN.READOUT),
            MutAnyOrigin,
        ](self.r_out_dbuf.unsafe_ptr())
        Self.ENS.predict_member_gpu[Self.ROLLOUT_BATCH](
            ctx, m, r_s_a_t, self.params_dbuf.unsafe_ptr(),
            r_a_aug_t, r_z_t, r_out_t,
        )
