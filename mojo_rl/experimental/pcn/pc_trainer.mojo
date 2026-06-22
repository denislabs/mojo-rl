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
from std.math import sqrt, log, cos, sin, tanh, pi
from std.random.philox import Random as PhiloxRandom

from .pc_constants import TPB

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

    @staticmethod
    def compute_grads_from_latents[BATCH: Int](
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
        """Like compute_grads_only but skips init_latents — caller has already
        populated `latents` (e.g., via an amortized encoder).

        Used for amortized PC (Tschantz 2023 hybrid): encoder produces an
        initial guess for z_t; this method runs T_infer refinement steps from
        there and computes W gradients at the refined state. With T_infer
        small (~3-5), the encoder dominates and the refinement is just local
        correction.
        """
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

    @staticmethod
    def compute_grads_from_latents_bounded_readout[BATCH: Int](
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
        """Like compute_grads_from_latents but applies a tanh squash to the
        readout block's output: μ_readout = tanh(W·act(z) + b).

        This bounds the model's predictions to [-1, 1] and prevents the
        decoder W column from drifting unbounded — useful when the obs space
        is bounded (e.g., normalized angles, normalized velocities) and the
        diagnostic shows unbounded prediction outputs.

        Implementation: after each forward sweep, the readout slot of
        `mu_eps_buf` is replaced with the Jacobian-adjusted error
        `(y - tanh(μ_lin)) · (1 - tanh(μ_lin)²)`. Subsequent `pull_back` and
        `weight_grad` calls automatically produce correct gradients with
        respect to the bounded output (because the chain-rule factor has
        been folded into ε in-place).

        Caller pre-populates `latents` (e.g., via an amortized encoder) —
        this method does NOT call init_latents. Use compute_grads_only or
        run init_latents externally if you want the standard forward-sweep
        initialization.
        """
        for _ in range(T_infer):
            Self._inference_step_bounded_readout[BATCH](
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

        # Energy / loss at the post-inference state. mu_eps_buf has the
        # Jacobian-adjusted ε for the readout slot, so reported numbers are
        # not the bare bounded-ε but the gradient-ready version. Useful as
        # a relative training signal.
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

            # For the readout (i = N-1), li_eps already contains the
            # Jacobian-adjusted ε from `_modify_readout_eps_for_bounded`,
            # so the standard `weight_grad` call produces the correct
            # gradient `dW = a_below.T @ ((y - μ_bnd) ⊙ (1 - μ_bnd²))`.
            Self.NET.block_types[i].weight_grad[BATCH, Self.dtype](
                li_eps, li_a_below, li_g
            )

        return PCTrainResult(
            energy_initial=0.0,
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
        comptime for l_idx in range(Self.NET.N_LATENTS):
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
    def _modify_readout_eps_for_bounded[BATCH: Int](
        mut mu_eps_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.SCRATCH_OUT_DIM),
            MutAnyOrigin,
        ],
        y_target: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.OUT_DIM), MutAnyOrigin
        ],
    ):
        """Replace `mu_eps_buf[readout_slot]` in place with the Jacobian-
        adjusted ε for a tanh-squashed readout: μ_bnd = tanh(μ_lin),
        ε_modified = (y - μ_bnd) · (1 - μ_bnd²).

        Standard `_forward_eps` writes the linear residual `ε_lin = y - μ_lin`.
        We recover μ_lin = y - ε_lin, squash it through tanh, recompute the
        bounded residual, and multiply by the local Jacobian factor — the
        chain-rule term that turns standard `pull_back` and `weight_grad`
        into correct backprop through the tanh emission. After this call
        the readout slot holds ε_modified = ε_bnd ⊙ (1 - μ_bnd²).
        """
        comptime offset_R = Self.NET._out_offset[Self.NET.N - 1]()
        comptime out_dim_R = Self.NET.block_types[Self.NET.N - 1].OUT_DIM
        var li_eps_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, out_dim_R), MutAnyOrigin
        ](mu_eps_buf.ptr + BATCH * offset_R)
        for b in range(BATCH):
            for j in range(out_dim_R):
                var eps_lin = Float64(
                    rebind[Scalar[Self.dtype]](li_eps_R[b, j])
                )
                var y_val = Float64(
                    rebind[Scalar[Self.dtype]](y_target[b, j])
                )
                var mu_lin = y_val - eps_lin
                var mu_bnd = tanh(mu_lin)
                var eps_bnd = y_val - mu_bnd
                var jac = 1.0 - mu_bnd * mu_bnd
                li_eps_R[b, j] = Scalar[Self.dtype](eps_bnd * jac)

    @staticmethod
    def _inference_step_bounded_readout[BATCH: Int](
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
        """One Jacobi inference step with a tanh-squashed readout.

        Same Jacobi update as `_inference_step` but the readout's ε is
        replaced with the Jacobian-adjusted residual after the forward
        sweep. This makes `pull_back` (Phase C) propagate the correct
        chain-rule factor for a tanh emission.
        """
        # ===== Phase A+B: forward predict + ε compute (standard) =============
        Self._forward_eps[BATCH](
            x_in, y_target, params, latents, mu_eps_buf, a_below_buf
        )

        # ===== Bounded readout adjustment ===================================
        Self._modify_readout_eps_for_bounded[BATCH](mu_eps_buf, y_target)

        # ===== Phase C: dx_l = ε_l − act'(x_l) ⊙ (W_{l+1}·ε_{l+1}) ===========
        # Identical to `_inference_step` Phase C — the readout's ε in
        # `mu_eps_buf` already carries the Jacobian factor, so `pull_back`
        # automatically backprops through the tanh emission correctly.
        comptime for l_idx in range(Self.NET.N_LATENTS):
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

            Self.NET.block_types[upper].pull_back[BATCH, Self.dtype](
                li_eps_upper, li_p_upper, li_z
            )

            var li_x_l = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[upper].IN_DIM),
                MutAnyOrigin,
            ](latents.ptr + BATCH * Self.NET._latent_offset[l_idx]())
            Self.NET.block_types[upper].act_derivative_mul[BATCH, Self.dtype](
                li_x_l, li_z, li_z
            )

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
        """0.5 · Σ_b Σ_k ε_{N-1}[b, k]² — supervised output loss.

        Block-major access: readout's slot starts at `BATCH * out_offset[N-1]`,
        with stride `out_dim` per sample (matches the per-block views).
        """
        comptime offset = Self.NET._out_offset[Self.NET.N - 1]()
        comptime out_dim = Self.NET.block_types[Self.NET.N - 1].OUT_DIM
        var total: Float64 = 0
        for b in range(BATCH):
            for k in range(out_dim):
                var idx = BATCH * offset + b * out_dim + k
                var v = Float64(mu_eps_buf.ptr[idx])
                total += v * v
        return 0.5 * total

    # =========================================================================
    # GPU paths
    # =========================================================================

    @staticmethod
    def _dx_subtract_kernel[
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
    def _latents_apply_kernel[
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
            ctx.enqueue_function[sub_k](
                li_eps_self, li_z, li_dx,
                grid_dim=(sub_blocks,), block_dim=(TPB,),
            )

        # Phase D: latents -= lr_x · dx_buf  (one fused kernel over LATENT_DIM)
        comptime apply_k = Self._latents_apply_kernel[
            BATCH, Self.NET.LATENT_DIM, Self.dtype
        ]
        var apply_threads = BATCH * Self.NET.LATENT_DIM
        var apply_blocks = (apply_threads + TPB - 1) // TPB
        ctx.enqueue_function[apply_k](
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

    # =========================================================================
    # MCPC (Monte Carlo Predictive Coding) — Bogacz notebook 2
    #
    # Adds Langevin noise to the latent SGD step (SGLD), turning MAP inference
    # into MCMC sampling. With a learned prior block as the first PCBlock and
    # no inputs (constant pseudo-input), this trains a generative model whose
    # latents sample from the data distribution.
    #
    # CPU-only for the first cut. GPU port can come later.
    # =========================================================================

    @staticmethod
    def _box_muller_fill[BATCH: Int](
        mut buf: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.LATENT_DIM), MutAnyOrigin
        ],
        seed: UInt64,
        offset_base: UInt64,
    ):
        """Fill `buf` with i.i.d. N(0, 1) samples via Box-Muller from PhiloxRandom.

        Each pair of uniforms produces two independent normals (z0 = r·cos, z1 = r·sin).
        `offset_base` should be unique per call across the SGLD trajectory.
        """
        var size = BATCH * Self.NET.LATENT_DIM
        var i = 0
        var pair_idx: UInt64 = 0
        while i < size:
            var rng1 = PhiloxRandom(seed=seed, offset=offset_base + pair_idx * 2)
            var rng2 = PhiloxRandom(seed=seed, offset=offset_base + pair_idx * 2 + 1)
            var u1 = rng1.step_uniform()[0]
            var u2 = rng2.step_uniform()[0]
            pair_idx += 1
            if u1 < 1e-10:
                u1 = 1e-10
            var r = sqrt(-2.0 * log(u1))
            var z0 = r * cos(2.0 * pi * u2)
            buf.ptr[i] = Scalar[Self.dtype](z0)
            i += 1
            if i < size:
                var z1 = r * sin(2.0 * pi * u2)
                buf.ptr[i] = Scalar[Self.dtype](z1)
                i += 1

    @staticmethod
    def _inference_step_mcpc[BATCH: Int](
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
        mut noise_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        lr_x: Scalar[Self.dtype],
        noise_coeff: Scalar[Self.dtype],
        seed: UInt64,
        offset_base: UInt64,
        clamp_output: Bool,
    ):
        """SGLD inference step: x_l ← x_l − lr_x · dx_l + noise_coeff · N(0,1).

        - clamp_output=True (training): readout block's ε = data − μ_readout.
        - clamp_output=False (generation): readout block's ε is forced to 0,
          so the supervised loss exerts no force on the top latent.
        - noise_coeff = sqrt(2 · noise_var · lr_x) for canonical SGLD.

        Reuses _forward_eps for Phases A+B, then optionally zeros the readout's
        ε slot, computes Phase C dx, and applies Phase D with added noise.
        """
        Self._forward_eps[BATCH](
            x_in, y_target, params, latents, mu_eps_buf, a_below_buf
        )

        # If generating (no data clamp): zero out the readout block's ε slot.
        # Buffer is BLOCK-major (each block's per-sample slots are contiguous,
        # NOT the row_major(BATCH, SCRATCH_OUT_DIM) the outer view declares),
        # so we use the same per-block view offsets as the rest of the trainer.
        if not clamp_output:
            comptime offset_eps_R = Self.NET._out_offset[Self.NET.N - 1]()
            comptime out_dim_R = Self.NET.block_types[Self.NET.N - 1].OUT_DIM
            var li_eps_R = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, out_dim_R),
                MutAnyOrigin,
            ](mu_eps_buf.ptr + BATCH * offset_eps_R)
            for b in range(BATCH):
                for k in range(out_dim_R):
                    li_eps_R[b, k] = Scalar[Self.dtype](0)

        # Phase C: dx_l = ε_l − act'(x_l) ⊙ (W_{l+1}·ε_{l+1})
        comptime for l_idx in range(Self.NET.N_LATENTS):
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

            Self.NET.block_types[upper].pull_back[BATCH, Self.dtype](
                li_eps_upper, li_p_upper, li_z
            )

            var li_x_l = LayoutTensor[
                Self.dtype,
                Layout.row_major(BATCH, Self.NET.block_types[upper].IN_DIM),
                MutAnyOrigin,
            ](latents.ptr + BATCH * Self.NET._latent_offset[l_idx]())
            Self.NET.block_types[upper].act_derivative_mul[BATCH, Self.dtype](
                li_x_l, li_z, li_z
            )

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

        # Phase D (SGLD): latents -= lr_x · dx + noise_coeff · N(0,1)
        Self._box_muller_fill[BATCH](noise_buf, seed, offset_base)
        for b in range(BATCH):
            for k in range(Self.NET.LATENT_DIM):
                latents[b, k] = (
                    rebind[Scalar[Self.dtype]](latents[b, k])
                    - lr_x * rebind[Scalar[Self.dtype]](dx_buf[b, k])
                    + noise_coeff * rebind[Scalar[Self.dtype]](noise_buf[b, k])
                )

    @staticmethod
    def compute_grads_only_mcpc[BATCH: Int](
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
        mut noise_buf: LayoutTensor[
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
        T_mixing: Int,
        T_sampling: Int,
        lr_x: Scalar[Self.dtype],
        noise_var: Scalar[Self.dtype],
        seed: UInt64,
        offset_base: UInt64,
    ) -> PCTrainResult:
        """MCPC training step: SGLD-aware inference + grad accumulation.

        T_mixing iterations let the chain settle (no grad accumulation needed —
        we just take the final-iteration ε's for the W gradient, which matches
        the reference implementation's `T_sampling = 1`). T_sampling > 1 would
        average grads over multiple post-burn-in iterations; we keep
        T_sampling=1 default to match the notebook.

        SGLD noise coefficient: sqrt(2 · noise_var · lr_x).

        Caller manages `offset_base` to ensure no two calls share a Philox
        substream (recommended bump: BATCH * LATENT_DIM * (T_mixing + T_sampling) per call).
        """
        Self.NET.init_latents[BATCH, Self.dtype](x_in, params, latents)

        var noise_coeff = Scalar[Self.dtype](
            sqrt(2.0 * Float64(noise_var) * Float64(lr_x))
        )
        var n_per_step = UInt64(BATCH * Self.NET.LATENT_DIM)

        # Mixing phase: SGLD steps with output clamped to data
        for t in range(T_mixing):
            Self._inference_step_mcpc[BATCH](
                x_in, y_target, params, latents,
                mu_eps_buf, a_below_buf, z_below_buf, dx_buf, noise_buf,
                lr_x, noise_coeff,
                seed, offset_base + UInt64(t) * n_per_step * 2,
                True,
            )

        # Sampling phase (we just need ε at the post-mixing point for grad).
        # For T_sampling=1, this is one more inference step; for >1, we'd
        # average grads but skip that for the first cut.
        for t in range(T_sampling):
            Self._inference_step_mcpc[BATCH](
                x_in, y_target, params, latents,
                mu_eps_buf, a_below_buf, z_below_buf, dx_buf, noise_buf,
                lr_x, noise_coeff,
                seed,
                offset_base + UInt64(T_mixing + t) * n_per_step * 2,
                True,
            )

        var energy_final = Self._total_energy[BATCH](mu_eps_buf)
        var output_loss = Self._readout_loss[BATCH](mu_eps_buf)

        # Compute weight grads at the post-sampling state
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
            energy_initial=0.0,
            energy_final=energy_final,
            output_loss_final=output_loss,
        )

    @staticmethod
    def generate_samples[BATCH: Int](
        params: LayoutTensor[
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
        mut noise_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        x_in: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.IN_DIM), MutAnyOrigin
        ],
        y_target_dummy: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.OUT_DIM), MutAnyOrigin
        ],
        mut sample_out: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.OUT_DIM), MutAnyOrigin
        ],
        T: Int,
        lr_x: Scalar[Self.dtype],
        noise_var: Scalar[Self.dtype],
        seed: UInt64,
        offset_base: UInt64,
    ):
        """Sample BATCH points from the learned generative model.

        Runs T iterations of SGLD inference WITHOUT the data clamp (clamp_output=False),
        so latents settle to draws from the prior. Then forwards the final latent
        through the readout block to produce one sample per batch row.

        `y_target_dummy` is read but its values don't affect generation since
        the readout block's ε is forced to 0 inside the inference step.
        """
        Self.NET.init_latents[BATCH, Self.dtype](x_in, params, latents)

        var noise_coeff = Scalar[Self.dtype](
            sqrt(2.0 * Float64(noise_var) * Float64(lr_x))
        )
        var n_per_step = UInt64(BATCH * Self.NET.LATENT_DIM)

        for t in range(T):
            Self._inference_step_mcpc[BATCH](
                x_in, y_target_dummy, params, latents,
                mu_eps_buf, a_below_buf, z_below_buf, dx_buf, noise_buf,
                lr_x, noise_coeff,
                seed, offset_base + UInt64(t) * n_per_step * 2,
                False,
            )

        # Final forward through the readout block to produce sample_out
        comptime read_idx = Self.NET.N - 1
        var li_p_read = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.NET.block_types[read_idx].PARAM_SIZE),
            MutAnyOrigin,
        ](params.ptr + Self.NET._param_offset[read_idx]())
        var li_a_read = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.block_types[read_idx].IN_DIM),
            MutAnyOrigin,
        ](a_below_buf.ptr + BATCH * Self.NET._in_offset[read_idx]())
        # x_below for the readout = last interior latent
        var li_x_below = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.block_types[read_idx].IN_DIM),
            MutAnyOrigin,
        ](latents.ptr + BATCH * Self.NET._latent_offset[Self.NET.N_LATENTS - 1]())
        Self.NET.block_types[read_idx].predict[BATCH, Self.dtype](
            li_x_below, li_p_read, sample_out, li_a_read
        )

    # =========================================================================
    # GPU MCPC paths
    #
    # Mirror the CPU MCPC path: SGLD-aware inference with optional output
    # clamp, plus a generation helper. Float32 internally for Metal safety,
    # so output is statistically equivalent to the CPU path but NOT bitwise
    # identical (the CPU path uses Float64 inside Box-Muller).
    # =========================================================================

    @staticmethod
    def _box_muller_fill_kernel[
        BATCH: Int, LDIM: Int, KDT: DType,
    ](
        noise_buf: LayoutTensor[
            KDT, Layout.row_major(BATCH, LDIM), MutAnyOrigin
        ],
        seed: UInt64,
        offset_base: UInt64,
    ):
        """Each thread fills one element of noise_buf with N(0,1).

        Matches CPU's Philox offset usage: pair `pair_idx` uses two RNG
        instances at offsets `offset_base + 2k` and `offset_base + 2k + 1`,
        producing two correlated normals (cos, sin). Float32 internally.
        """
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * LDIM:
            return
        var pair_idx = UInt64(idx // 2)
        var which_in_pair = idx % 2

        var rng1 = PhiloxRandom(seed=seed, offset=offset_base + pair_idx * 2)
        var rng2 = PhiloxRandom(seed=seed, offset=offset_base + pair_idx * 2 + 1)
        var u1 = Float32(rng1.step_uniform()[0])
        var u2 = Float32(rng2.step_uniform()[0])
        if u1 < Float32(1e-7):
            u1 = Float32(1e-7)
        var r = sqrt(Float32(-2.0) * log(u1))
        var two_pi_u2 = Float32(6.283185307179586) * u2
        var z: Float32 = (
            r * cos(two_pi_u2) if which_in_pair == 0 else r * sin(two_pi_u2)
        )

        var b = idx // LDIM
        var k = idx % LDIM
        noise_buf[b, k] = Scalar[KDT](z)

    @staticmethod
    def _box_muller_fill_gpu[BATCH: Int](
        ctx: DeviceContext,
        mut noise_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        seed: UInt64,
        offset_base: UInt64,
    ) raises:
        comptime k = Self._box_muller_fill_kernel[
            BATCH, Self.NET.LATENT_DIM, Self.dtype
        ]
        var threads = BATCH * Self.NET.LATENT_DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](
            noise_buf, seed, offset_base,
            grid_dim=(blocks,), block_dim=(TPB,),
        )

    @staticmethod
    def _zero_eps_kernel[
        BATCH: Int, DIM: Int, KDT: DType,
    ](
        eps: LayoutTensor[
            KDT, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var k = idx % DIM
        eps[b, k] = Scalar[KDT](0)

    @staticmethod
    def _sgld_apply_kernel[
        BATCH: Int, LDIM: Int, KDT: DType,
    ](
        latents: LayoutTensor[
            KDT, Layout.row_major(BATCH, LDIM), MutAnyOrigin
        ],
        dx_buf: LayoutTensor[
            KDT, Layout.row_major(BATCH, LDIM), MutAnyOrigin
        ],
        noise_buf: LayoutTensor[
            KDT, Layout.row_major(BATCH, LDIM), MutAnyOrigin
        ],
        lr_x: Scalar[KDT],
        noise_coeff: Scalar[KDT],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * LDIM:
            return
        var b = idx // LDIM
        var k = idx % LDIM
        latents[b, k] = (
            rebind[Scalar[KDT]](latents[b, k])
            - lr_x * rebind[Scalar[KDT]](dx_buf[b, k])
            + noise_coeff * rebind[Scalar[KDT]](noise_buf[b, k])
        )

    @staticmethod
    def _inference_step_mcpc_gpu[BATCH: Int](
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
        mut noise_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        lr_x: Scalar[Self.dtype],
        noise_coeff: Scalar[Self.dtype],
        seed: UInt64,
        offset_base: UInt64,
        clamp_output: Bool,
    ) raises:
        """SGLD inference step on GPU: x_l ← x_l − lr_x · dx_l + noise_coeff · N(0,1).

        Mirrors `_inference_step_mcpc` (CPU). `clamp_output=False` zeros out
        the readout block's ε slot so the supervised loss exerts no force on
        the top latent (used during generation).
        """
        Self._forward_eps_gpu[BATCH](
            ctx, x_in, y_target, params, latents, mu_eps_buf, a_below_buf
        )

        if not clamp_output:
            comptime offset_eps_R = Self.NET._out_offset[Self.NET.N - 1]()
            comptime out_dim_R = Self.NET.block_types[Self.NET.N - 1].OUT_DIM
            var li_eps_R = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, out_dim_R), MutAnyOrigin
            ](mu_eps_buf.ptr + BATCH * offset_eps_R)
            comptime zero_k = Self._zero_eps_kernel[
                BATCH, out_dim_R, Self.dtype
            ]
            var z_threads = BATCH * out_dim_R
            var z_blocks = (z_threads + TPB - 1) // TPB
            ctx.enqueue_function[zero_k](
                li_eps_R,
                grid_dim=(z_blocks,), block_dim=(TPB,),
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
            ctx.enqueue_function[sub_k](
                li_eps_self, li_z, li_dx,
                grid_dim=(sub_blocks,), block_dim=(TPB,),
            )

        # Generate noise on device
        Self._box_muller_fill_gpu[BATCH](ctx, noise_buf, seed, offset_base)

        # Phase D (SGLD): latents -= lr_x · dx + noise_coeff · noise (one fused kernel)
        comptime apply_k = Self._sgld_apply_kernel[
            BATCH, Self.NET.LATENT_DIM, Self.dtype
        ]
        var apply_threads = BATCH * Self.NET.LATENT_DIM
        var apply_blocks = (apply_threads + TPB - 1) // TPB
        ctx.enqueue_function[apply_k](
            latents, dx_buf, noise_buf, lr_x, noise_coeff,
            grid_dim=(apply_blocks,), block_dim=(TPB,),
        )

    @staticmethod
    def compute_grads_only_mcpc_gpu[BATCH: Int](
        ctx: DeviceContext,
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
        mut noise_buf: LayoutTensor[
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
        T_mixing: Int,
        T_sampling: Int,
        lr_x: Scalar[Self.dtype],
        noise_var: Scalar[Self.dtype],
        seed: UInt64,
        offset_base: UInt64,
    ) raises:
        """GPU equivalent of compute_grads_only_mcpc.

        No diagnostic energies returned (would require host syncs). Caller
        bumps `offset_base` between calls; recommended bump:
        BATCH * LATENT_DIM * (T_mixing + T_sampling) * 2.
        """
        Self.NET.init_latents_gpu[BATCH, Self.dtype](
            ctx, x_in, params, latents, a_below_buf
        )

        var noise_coeff = Scalar[Self.dtype](
            sqrt(2.0 * Float64(noise_var) * Float64(lr_x))
        )
        var n_per_step = UInt64(BATCH * Self.NET.LATENT_DIM)

        for t in range(T_mixing):
            Self._inference_step_mcpc_gpu[BATCH](
                ctx, x_in, y_target, params, latents,
                mu_eps_buf, a_below_buf, z_below_buf, dx_buf, noise_buf,
                lr_x, noise_coeff,
                seed, offset_base + UInt64(t) * n_per_step * 2,
                True,
            )

        for t in range(T_sampling):
            Self._inference_step_mcpc_gpu[BATCH](
                ctx, x_in, y_target, params, latents,
                mu_eps_buf, a_below_buf, z_below_buf, dx_buf, noise_buf,
                lr_x, noise_coeff,
                seed, offset_base + UInt64(T_mixing + t) * n_per_step * 2,
                True,
            )

        # Compute weight grads at post-sampling state
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

    @staticmethod
    def generate_samples_gpu[BATCH: Int](
        ctx: DeviceContext,
        params: LayoutTensor[
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
        mut noise_buf: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.LATENT_DIM),
            MutAnyOrigin,
        ],
        x_in: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.IN_DIM), MutAnyOrigin
        ],
        y_target_dummy: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.OUT_DIM), MutAnyOrigin
        ],
        mut sample_out: LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, Self.NET.OUT_DIM), MutAnyOrigin
        ],
        T: Int,
        lr_x: Scalar[Self.dtype],
        noise_var: Scalar[Self.dtype],
        seed: UInt64,
        offset_base: UInt64,
    ) raises:
        """GPU equivalent of generate_samples — imagined-rollout sampling."""
        Self.NET.init_latents_gpu[BATCH, Self.dtype](
            ctx, x_in, params, latents, a_below_buf
        )

        var noise_coeff = Scalar[Self.dtype](
            sqrt(2.0 * Float64(noise_var) * Float64(lr_x))
        )
        var n_per_step = UInt64(BATCH * Self.NET.LATENT_DIM)

        for t in range(T):
            Self._inference_step_mcpc_gpu[BATCH](
                ctx, x_in, y_target_dummy, params, latents,
                mu_eps_buf, a_below_buf, z_below_buf, dx_buf, noise_buf,
                lr_x, noise_coeff,
                seed, offset_base + UInt64(t) * n_per_step * 2,
                False,
            )

        # Final feedforward through readout to produce sample_out
        comptime read_idx = Self.NET.N - 1
        var li_p_read = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.NET.block_types[read_idx].PARAM_SIZE),
            MutAnyOrigin,
        ](params.ptr + Self.NET._param_offset[read_idx]())
        var li_a_read = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.block_types[read_idx].IN_DIM),
            MutAnyOrigin,
        ](a_below_buf.ptr + BATCH * Self.NET._in_offset[read_idx]())
        var li_x_below = LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.NET.block_types[read_idx].IN_DIM),
            MutAnyOrigin,
        ](latents.ptr + BATCH * Self.NET._latent_offset[Self.NET.N_LATENTS - 1]())
        Self.NET.block_types[read_idx].predict_gpu[BATCH, Self.dtype](
            ctx, li_x_below, li_p_read, sample_out, li_a_read
        )
