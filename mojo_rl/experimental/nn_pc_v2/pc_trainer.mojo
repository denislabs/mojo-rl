"""PCTrainer — Bogacz-canonical training step (CPU, Phase 1).

One training step:
  1. Forward sweep: x_l ← μ_l (init latents from current params)
  2. T_infer iterations of `_pc_inference_step` (local-rule x updates)
  3. One weight gradient pass: dE/dW_i, dE/db_i per block
  4. Vanilla SGD weight step: params -= lr_w · grads

For Phase 1 we do plain SGD on weights to minimize surface area. Adam is a
one-line drop-in once the smoke test passes.

The trainer is a thin static struct: all buffers are caller-owned so they
can be allocated once and reused across many batches.

Mojo gotcha: `PCSequential` is parametric, so we cannot constrain a struct
parameter as `NET: PCSequential`. Instead the trainer mirrors PCSequential's
variadic pattern: `*BLOCKS: PCBlockTrait`, with `comptime NET = PCSequential[*Self.BLOCKS]`
recovered inside.
"""

from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import TPB
from mojo_rl.nn.initializer import Initializer

from .pc_sequential import PCSequential
from .predictive_model import PCBlockTrait


@fieldwise_init
struct PCTrainResult(Movable & ImplicitlyCopyable):
    """Diagnostics returned from `train_one_batch`."""

    var energy_initial: Float64
    var energy_final: Float64
    var output_loss_final: Float64


struct PCTrainer[*BLOCKS: PCBlockTrait, dtype: DType = DType.float32]:
    """All-static; takes buffers from the caller for zero per-batch alloc."""

    comptime NET = PCSequential[*Self.BLOCKS]

    @staticmethod
    def train_one_batch[BATCH: Int](
        mut params: LayoutTensor[
            Self.dtype, Layout.row_major(Self.NET.PARAM_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            Self.dtype, Layout.row_major(Self.NET.PARAM_SIZE), MutAnyOrigin
        ],
        mut latents: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.LATENT_DIM), MutAnyOrigin
        ],
        mut mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_OUT_DIM),
            MutAnyOrigin,
        ],
        mut a_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
        mut z_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
        mut dx_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        x_in: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.IN_DIM), MutAnyOrigin
        ],
        y_target: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.OUT_DIM), MutAnyOrigin
        ],
        T_infer: Int,
        lr_x: Scalar[Self.dtype],
        lr_w: Scalar[Self.dtype],
    ) -> PCTrainResult:
        """One Bogacz-canonical training step.

        Returns initial-energy + final-energy + final-output-loss for
        diagnostics. The smoke test asserts final < initial.
        """

        # === 1. Forward sweep: x_l ← μ_l (init latents) =====================
        Self.NET.init_latents[BATCH, Self.dtype](x_in, params, latents)

        # Initial energy (run forward + ε pass)
        Self._forward_eps[BATCH](
            x_in, y_target, params, latents, mu_eps_buf, a_below_buf
        )
        var energy_initial = Self._total_energy[BATCH](mu_eps_buf)

        # === 2. T_infer iterations of x updates ============================
        for _ in range(T_infer):
            Self._inference_step[BATCH](
                x_in,
                y_target,
                params,
                latents,
                mu_eps_buf,
                a_below_buf,
                z_below_buf,
                dx_buf,
                lr_x,
            )

        # After inference loop, mu_eps_buf holds ε (not μ).
        var energy_final = Self._total_energy[BATCH](mu_eps_buf)
        var output_loss = Self._readout_loss[BATCH](mu_eps_buf)

        # === 3. Compute weight gradients per block =========================
        @parameter
        for i in range(Self.NET.N):
            var li_g = LayoutTensor[
                Self.dtype,
                Layout.row_major(Self.NET.block_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr + Self.NET._param_offset[i]())
            var li_eps = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[i]())
            var li_a_below = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].IN_DIM),
                MutAnyOrigin,
            ](a_below_buf.ptr + BATCH * Self.NET._in_offset[i]())

            Self.NET.block_types[i].weight_grad[BATCH, Self.dtype](
                li_eps, li_a_below, li_g
            )

        # === 4. SGD weight step: params -= lr_w · grads ====================
        for i in range(Self.NET.PARAM_SIZE):
            params.ptr[i] = params.ptr[i] - lr_w * grads.ptr[i]

        return PCTrainResult(
            energy_initial=energy_initial,
            energy_final=energy_final,
            output_loss_final=output_loss,
        )

    # =========================================================================
    # Optimizer-agnostic variant: writes gradients but doesn't update params.
    # Use when wiring up Adam/AdamW/etc. Caller invokes the optimizer's
    # `step()` after this returns.
    # =========================================================================

    @staticmethod
    def compute_grads_only[BATCH: Int](
        params: LayoutTensor[
            Self.dtype, Layout.row_major(Self.NET.PARAM_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            Self.dtype, Layout.row_major(Self.NET.PARAM_SIZE), MutAnyOrigin
        ],
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        mut mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_OUT_DIM),
            MutAnyOrigin,
        ],
        mut a_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
        mut z_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
        mut dx_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        x_in: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.IN_DIM), MutAnyOrigin
        ],
        y_target: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.OUT_DIM), MutAnyOrigin
        ],
        T_infer: Int,
        lr_x: Scalar[Self.dtype],
    ) -> PCTrainResult:
        """Run forward sweep + T_infer inference iterations + grad compute.
        Writes per-block (W, b) gradients into `grads`. Does NOT touch `params`.
        """
        Self.NET.init_latents[BATCH, Self.dtype](x_in, params, latents)

        Self._forward_eps[BATCH](
            x_in, y_target, params, latents, mu_eps_buf, a_below_buf
        )
        var energy_initial = Self._total_energy[BATCH](mu_eps_buf)

        for _ in range(T_infer):
            Self._inference_step[BATCH](
                x_in,
                y_target,
                params,
                latents,
                mu_eps_buf,
                a_below_buf,
                z_below_buf,
                dx_buf,
                lr_x,
            )

        var energy_final = Self._total_energy[BATCH](mu_eps_buf)
        var output_loss = Self._readout_loss[BATCH](mu_eps_buf)

        comptime for i in range(Self.NET.N):
            var li_g = LayoutTensor[
                Self.dtype,
                Layout.row_major(Self.NET.block_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr + Self.NET._param_offset[i]())
            var li_eps = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[i]())
            var li_a_below = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].IN_DIM),
                MutAnyOrigin,
            ](a_below_buf.ptr + BATCH * Self.NET._in_offset[i]())

            Self.NET.block_types[i].weight_grad[BATCH, Self.dtype](
                li_eps, li_a_below, li_g
            )

        return PCTrainResult(
            energy_initial=energy_initial,
            energy_final=energy_final,
            output_loss_final=output_loss,
        )

    # =========================================================================
    # Internals
    # =========================================================================

    @staticmethod
    def _forward_eps[BATCH: Int](
        x_in: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.IN_DIM), MutAnyOrigin
        ],
        y_target: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            Self.dtype, Layout.row_major(Self.NET.PARAM_SIZE), MutAnyOrigin
        ],
        latents: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.LATENT_DIM), MutAnyOrigin
        ],
        mut mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_OUT_DIM),
            MutAnyOrigin,
        ],
        mut a_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
    ):
        """Phases A+B of inference: forward predict + ε compute (no x update)."""
        @parameter
        for i in range(Self.NET.N):
            var li_p = LayoutTensor[
                Self.dtype,
                Layout.row_major(Self.NET.block_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self.NET._param_offset[i]())
            var li_a = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].IN_DIM),
                MutAnyOrigin,
            ](a_below_buf.ptr + BATCH * Self.NET._in_offset[i]())
            var li_mu = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[i]())

            @parameter
            if i == 0:
                var li_x_below = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(BATCH, Self.NET.block_types[i].IN_DIM),
                    MutAnyOrigin,
                ](x_in.ptr)
                Self.NET.block_types[i].predict[BATCH, Self.dtype](
                    li_x_below, li_p, li_mu, li_a
                )
            else:
                var li_x_below = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(BATCH, Self.NET.block_types[i].IN_DIM),
                    MutAnyOrigin,
                ](latents.ptr + BATCH * Self.NET._latent_offset[i - 1]())
                Self.NET.block_types[i].predict[BATCH, Self.dtype](
                    li_x_below, li_p, li_mu, li_a
                )

        # ε = x_above − μ (in-place)
        @parameter
        for i in range(Self.NET.N):
            var li_mu_view = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[i]())
            var li_eps_view = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[i]())

            @parameter
            if i == Self.NET.N - 1:
                var li_target = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                    MutAnyOrigin,
                ](y_target.ptr)
                Self.NET.block_types[i].eps_compute[BATCH, Self.dtype](
                    li_target, li_mu_view, li_eps_view
                )
            else:
                var li_x_above = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                    MutAnyOrigin,
                ](latents.ptr + BATCH * Self.NET._latent_offset[i]())
                Self.NET.block_types[i].eps_compute[BATCH, Self.dtype](
                    li_x_above, li_mu_view, li_eps_view
                )

    @staticmethod
    def _inference_step[BATCH: Int](
        x_in: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.IN_DIM), MutAnyOrigin
        ],
        y_target: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            Self.dtype, Layout.row_major(Self.NET.PARAM_SIZE), MutAnyOrigin
        ],
        mut latents: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.LATENT_DIM), MutAnyOrigin
        ],
        mut mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_OUT_DIM),
            MutAnyOrigin,
        ],
        mut a_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
        mut z_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
        mut dx_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        lr_x: Scalar[Self.dtype],
    ):
        """One Jacobi iteration of the local-rule x update."""

        # ===== Phase A+B: forward predict + ε compute =======================
        Self._forward_eps[BATCH](
            x_in, y_target, params, latents, mu_eps_buf, a_below_buf
        )

        # ===== Phase C: dx_l = ε_l − act'(x_l) ⊙ (W_{l+1}·ε_{l+1}) ===========
        @parameter
        for l_idx in range(Self.NET.N_LATENTS):
            comptime upper = l_idx + 1

            var li_p_upper = LayoutTensor[
                Self.dtype,
                Layout.row_major(Self.NET.block_types[upper].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self.NET._param_offset[upper]())
            var li_eps_upper = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[upper].OUT_DIM),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[upper]())
            var li_z = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[upper].IN_DIM),
                MutAnyOrigin,
            ](z_below_buf.ptr + BATCH * Self.NET._in_offset[upper]())

            # 1. z = pull_back(ε_upper, W_upper)
            Self.NET.block_types[upper].pull_back[BATCH, Self.dtype](
                li_eps_upper, li_p_upper, li_z
            )

            # 2. z ← act'(x_l) ⊙ z   (in-place gating using upper block's ACT)
            var li_x_l = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[upper].IN_DIM),
                MutAnyOrigin,
            ](latents.ptr + BATCH * Self.NET._latent_offset[l_idx]())
            Self.NET.block_types[upper].act_derivative_mul[BATCH, Self.dtype](
                li_x_l, li_z, li_z
            )

            # 3. dx_l = ε_l − z
            var li_eps_self = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[l_idx].OUT_DIM),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[l_idx]())
            var li_dx = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[l_idx].OUT_DIM),
                MutAnyOrigin,
            ](dx_buf.ptr + BATCH * Self.NET._latent_offset[l_idx]())

            for b in range(BATCH):
                for k in range(Self.NET.block_types[l_idx].OUT_DIM):
                    li_dx[b, k] = (
                        rebind[Scalar[Self.dtype]](li_eps_self[b, k])
                        - rebind[Scalar[Self.dtype]](li_z[b, k])
                    )

        # ===== Phase D: latents -= lr_x · dx ================================
        for b in range(BATCH):
            for k in range(Self.NET.LATENT_DIM):
                latents[b, k] = (
                    rebind[Scalar[Self.dtype]](latents[b, k])
                    - lr_x * rebind[Scalar[Self.dtype]](dx_buf[b, k])
                )

    @staticmethod
    def _total_energy[BATCH: Int](
        mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_OUT_DIM),
            MutAnyOrigin,
        ],
    ) -> Float64:
        """E = 0.5 · Σ_i Σ_b Σ_k ε_i[b, k]² (caller must have just run _forward_eps)."""
        var total: Float64 = 0
        for b in range(BATCH):
            for k in range(Self.NET.SCRATCH_OUT_DIM):
                var v = Float64(rebind[Scalar[Self.dtype]](mu_eps_buf[b, k]))
                total += v * v
        return 0.5 * total

    @staticmethod
    def _readout_loss[BATCH: Int](
        mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_OUT_DIM),
            MutAnyOrigin,
        ],
    ) -> Float64:
        """0.5 · Σ_b Σ_k ε_{N-1}[b, k]² — supervised output loss."""
        comptime offset = Self.NET._out_offset[Self.NET.N - 1]()
        comptime out_dim = Self.NET.block_types[Self.NET.N - 1].OUT_DIM
        var total: Float64 = 0
        for b in range(BATCH):
            for k in range(out_dim):
                var idx = b * Self.NET.SCRATCH_OUT_DIM + offset + k
                var v = Float64(mu_eps_buf.ptr[idx])
                total += v * v
        return 0.5 * total

    # =========================================================================
    # GPU paths
    # =========================================================================

    @staticmethod
    fn _dx_subtract_kernel[
        BATCH: Int, DIM: Int, KDT: DType,
    ](
        eps_self: LayoutTensor[
            KDT, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        z: LayoutTensor[KDT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
        dx: LayoutTensor[KDT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var k = idx % DIM
        dx[b, k] = (
            rebind[Scalar[KDT]](eps_self[b, k])
            - rebind[Scalar[KDT]](z[b, k])
        )

    @staticmethod
    fn _latents_apply_kernel[
        BATCH: Int, LDIM: Int, KDT: DType,
    ](
        latents: LayoutTensor[
            KDT, Layout.row_major(BATCH, LDIM), MutAnyOrigin
        ],
        dx_buf: LayoutTensor[
            KDT, Layout.row_major(BATCH, LDIM), MutAnyOrigin
        ],
        lr_x: Scalar[KDT],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * LDIM:
            return
        var b = idx // LDIM
        var k = idx % LDIM
        latents[b, k] = (
            rebind[Scalar[KDT]](latents[b, k])
            - lr_x * rebind[Scalar[KDT]](dx_buf[b, k])
        )

    # ── _forward_eps_gpu: predict + ε across all blocks ──────────────────────

    @staticmethod
    def _forward_eps_gpu[BATCH: Int](
        ctx: DeviceContext,
        x_in: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.IN_DIM), MutAnyOrigin
        ],
        y_target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.OUT_DIM),
            MutAnyOrigin,
        ],
        params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.NET.PARAM_SIZE),
            MutAnyOrigin,
        ],
        latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        mut mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_OUT_DIM),
            MutAnyOrigin,
        ],
        mut a_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
    ) raises:
        # Phase A: predict_gpu for all blocks
        comptime for i in range(Self.NET.N):
            var li_p = LayoutTensor[
                Self.dtype,
                Layout.row_major(Self.NET.block_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self.NET._param_offset[i]())
            var li_a = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].IN_DIM),
                MutAnyOrigin,
            ](a_below_buf.ptr + BATCH * Self.NET._in_offset[i]())
            var li_mu = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[i]())

            comptime if i == 0:
                var li_x_below = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(BATCH, Self.NET.block_types[i].IN_DIM),
                    MutAnyOrigin,
                ](x_in.ptr)
                Self.NET.block_types[i].predict_gpu[BATCH, Self.dtype](
                    ctx, li_x_below, li_p, li_mu, li_a
                )
            else:
                var li_x_below = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(BATCH, Self.NET.block_types[i].IN_DIM),
                    MutAnyOrigin,
                ](latents.ptr + BATCH * Self.NET._latent_offset[i - 1]())
                Self.NET.block_types[i].predict_gpu[BATCH, Self.dtype](
                    ctx, li_x_below, li_p, li_mu, li_a
                )

        # Phase B: ε = x_above − μ (in-place)
        comptime for i in range(Self.NET.N):
            var li_mu_view = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[i]())
            var li_eps_view = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[i]())

            comptime if i == Self.NET.N - 1:
                var li_target = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                    MutAnyOrigin,
                ](y_target.ptr)
                Self.NET.block_types[i].eps_compute_gpu[BATCH, Self.dtype](
                    ctx, li_target, li_mu_view, li_eps_view
                )
            else:
                var li_x_above = LayoutTensor[
                    Self.dtype,
                    Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                    MutAnyOrigin,
                ](latents.ptr + BATCH * Self.NET._latent_offset[i]())
                Self.NET.block_types[i].eps_compute_gpu[BATCH, Self.dtype](
                    ctx, li_x_above, li_mu_view, li_eps_view
                )

    # ── _inference_step_gpu: one Jacobi iteration ────────────────────────────

    @staticmethod
    def _inference_step_gpu[BATCH: Int](
        ctx: DeviceContext,
        x_in: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.IN_DIM), MutAnyOrigin
        ],
        y_target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.OUT_DIM),
            MutAnyOrigin,
        ],
        params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.NET.PARAM_SIZE),
            MutAnyOrigin,
        ],
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        mut mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_OUT_DIM),
            MutAnyOrigin,
        ],
        mut a_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
        mut z_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
        mut dx_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        lr_x: Scalar[Self.dtype],
    ) raises:
        Self._forward_eps_gpu[BATCH](
            ctx, x_in, y_target, params, latents, mu_eps_buf, a_below_buf
        )

        # Phase C: dx_l = ε_l − act'(x_l) ⊙ (W_{l+1} · ε_{l+1})
        comptime for l_idx in range(Self.NET.N_LATENTS):
            comptime upper = l_idx + 1

            var li_p_upper = LayoutTensor[
                Self.dtype,
                Layout.row_major(Self.NET.block_types[upper].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self.NET._param_offset[upper]())
            var li_eps_upper = LayoutTensor[
                Self.dtype,
                Layout.row_major(
                    BATCH, Self.NET.block_types[upper].OUT_DIM
                ),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[upper]())
            var li_z = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[upper].IN_DIM),
                MutAnyOrigin,
            ](z_below_buf.ptr + BATCH * Self.NET._in_offset[upper]())

            Self.NET.block_types[upper].pull_back_gpu[BATCH, Self.dtype](
                ctx, li_eps_upper, li_p_upper, li_z
            )

            var li_x_l = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[upper].IN_DIM),
                MutAnyOrigin,
            ](latents.ptr + BATCH * Self.NET._latent_offset[l_idx]())
            Self.NET.block_types[upper].act_derivative_mul_gpu[
                BATCH, Self.dtype
            ](ctx, li_x_l, li_z, li_z)

            var li_eps_self = LayoutTensor[
                Self.dtype,
                Layout.row_major(
                    BATCH, Self.NET.block_types[l_idx].OUT_DIM
                ),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[l_idx]())
            var li_dx = LayoutTensor[
                Self.dtype,
                Layout.row_major(
                    BATCH, Self.NET.block_types[l_idx].OUT_DIM
                ),
                MutAnyOrigin,
            ](dx_buf.ptr + BATCH * Self.NET._latent_offset[l_idx]())

            comptime sub_k = Self._dx_subtract_kernel[
                BATCH, Self.NET.block_types[l_idx].OUT_DIM, Self.dtype
            ]
            var sub_threads = BATCH * Self.NET.block_types[l_idx].OUT_DIM
            var sub_blocks = (sub_threads + TPB - 1) // TPB
            ctx.enqueue_function[sub_k, sub_k](
                li_eps_self, li_z, li_dx,
                grid_dim=(sub_blocks,), block_dim=(TPB,),
            )

        # Phase D: latents -= lr_x · dx_buf  (one fused kernel over LATENT_DIM)
        comptime apply_k = Self._latents_apply_kernel[
            BATCH, Self.NET.LATENT_DIM, Self.dtype
        ]
        var apply_threads = BATCH * Self.NET.LATENT_DIM
        var apply_blocks = (apply_threads + TPB - 1) // TPB
        ctx.enqueue_function[apply_k, apply_k](
            latents, dx_buf, lr_x,
            grid_dim=(apply_blocks,), block_dim=(TPB,),
        )

    # ── compute_grads_only_gpu: full T_infer + grad pass ─────────────────────

    @staticmethod
    def compute_grads_only_gpu[BATCH: Int](
        ctx: DeviceContext,
        params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.NET.PARAM_SIZE),
            MutAnyOrigin,
        ],
        mut grads: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.NET.PARAM_SIZE),
            MutAnyOrigin,
        ],
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        mut mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_OUT_DIM),
            MutAnyOrigin,
        ],
        mut a_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
        mut z_below_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_IN_DIM),
            MutAnyOrigin,
        ],
        mut dx_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        x_in: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.IN_DIM), MutAnyOrigin
        ],
        y_target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.OUT_DIM),
            MutAnyOrigin,
        ],
        T_infer: Int,
        lr_x: Scalar[Self.dtype],
    ) raises:
        """GPU equivalent of compute_grads_only. No diagnostic energies returned
        (would require host syncs). Caller owns all buffers; this method does
        not touch params or invoke an optimizer.
        """
        # 1. Forward sweep on GPU
        Self.NET.init_latents_gpu[BATCH, Self.dtype](
            ctx, x_in, params, latents, a_below_buf
        )

        # 2. T_infer iterations
        for _ in range(T_infer):
            Self._inference_step_gpu[BATCH](
                ctx,
                x_in,
                y_target,
                params,
                latents,
                mu_eps_buf,
                a_below_buf,
                z_below_buf,
                dx_buf,
                lr_x,
            )

        # 3. Compute weight grads per block
        comptime for i in range(Self.NET.N):
            var li_g = LayoutTensor[
                Self.dtype,
                Layout.row_major(Self.NET.block_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr + Self.NET._param_offset[i]())
            var li_eps = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].OUT_DIM),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * Self.NET._out_offset[i]())
            var li_a_below = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[i].IN_DIM),
                MutAnyOrigin,
            ](a_below_buf.ptr + BATCH * Self.NET._in_offset[i]())

            Self.NET.block_types[i].weight_grad_gpu[BATCH, Self.dtype](
                ctx, li_eps, li_a_below, li_g
            )
