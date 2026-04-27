"""PCTrainer — CPU PCN training loop.

Algorithm (mirrors the PyTorch reference shared by the user):

For each mini-batch:
  1. Initialize latents x^(1)..x^(L) ∼ N(0, 1)  (per-sample, per-batch).
  2. INFERENCE phase — `T_infer` steps with weights frozen:
       For each step:
         compute snapshot: a_l, x_hat_l, eps_l = x^(l) - x_hat_l, h_l = eps_l*f'(a_l)
                           for all non-readout layers l = 0..L-1
         compute readout:  a_R = x_hat_R = x^(L) @ W_R^T
                           eps_sup = x_hat_R - y_target
                           eps_L_pulled = eps_sup @ W_R       (used for x^(L) update)
         update latents:   for l = 1..L:
                             err_ext_l = eps_l  if l<L  else eps_L_pulled
                             grad_X_l  = err_ext_l - (h_{l-1} @ W_{l-1})
                             x^(l)    -= eta_infer * grad_X_l

  3. LEARNING phase — `T_learn` steps with latents frozen:
       For each step:
         recompute snapshot + readout (weights have changed)
         non-readout weight update:  W_l +=  +eta_learn/B * (h_l^T @ x_above_l)
                                     for l = 0..L-1
         readout    weight update:   W_R +=  -eta_learn/B * (eps_sup^T @ x_above_R)
                                     (sign flips because eps_sup uses opposite-sign convention
                                      from non-readout eps — see PyTorch ref)

Indexing notes:
  - x_above for layer i (0-indexed in PCSequential) = latent[i] for i < L,
    and = latent[L-1] for i = L (readout shares its x_above with the last hidden layer).
  - "target" of layer i's prediction = x_input for i=0, latent[i-1] for 0<i<L.
  - Storage: latents are stored layer-major, each [BATCH, OUT_DIM_l] row-major.
    Same for the per-step scratch buffers (a, x_hat, eps, h) over IN_DIM_l.
"""

from std.math import sqrt, log, cos, sin, pi
from std.random.philox import Random as PhiloxRandom
from std.memory import alloc, memset, UnsafePointer
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype as default_dtype, TPB
from mojo_rl.nn.initializer import Initializer, Xavier

from ..predictive_model import PCLayer
from ..model.pc_sequential import PCSequential


@fieldwise_init
struct PCTrainResult(ImplicitlyCopyable, Movable):
    """Result of a single train_one_batch call."""

    var energy: Float64  # ½ Σ ‖eps_l‖² + ½ ‖eps_sup‖² (averaged over batch)
    var sup_loss: Float64  # ½ ‖eps_sup‖² (averaged over batch)


struct PCTrainer[*LAYERS: PCLayer, dtype: DType = default_dtype]:
    """All-static PCN trainer (CPU only initially).

    Parameterized by the same variadic LAYERS list as PCSequential; internally
    constructs `MODEL = PCSequential[*Self.LAYERS]` for all sizing/dispatch.

    Mirrors the project's `nn.Trainer` pattern: no stored state, caller owns
    `params` and per-batch `latents` buffers and passes them in.
    """

    # Comptime alias to the model definition — all sizing helpers + per-layer
    # dispatch live on PCSequential.
    comptime MODEL = PCSequential[*Self.LAYERS]

    # =========================================================================
    # State allocation helpers
    # =========================================================================

    @staticmethod
    def alloc_params() -> List[Scalar[Self.dtype]]:
        """Allocate a zero-filled params buffer of size MODEL.PARAM_SIZE."""
        var p = List[Scalar[Self.dtype]](capacity=Self.MODEL.PARAM_SIZE)
        for _ in range(Self.MODEL.PARAM_SIZE):
            p.append(Scalar[Self.dtype](0))
        return p^

    @staticmethod
    def init_params[INIT: Initializer = Xavier[]]() -> List[Scalar[Self.dtype]]:
        """Allocate + initialize params via INIT (defaults to Xavier)."""
        var p = Self.alloc_params()
        var p_t = LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ](p.unsafe_ptr())
        Self.MODEL.initialize_params[INIT, Self.dtype](p_t)
        return p^

    @staticmethod
    def alloc_latents[BATCH: Int]() -> List[Scalar[Self.dtype]]:
        """Allocate a zero-filled per-batch latents buffer."""
        var size = BATCH * Self.MODEL.LATENT_SIZE_PER_SAMPLE
        var L = List[Scalar[Self.dtype]](capacity=size)
        for _ in range(size):
            L.append(Scalar[Self.dtype](0))
        return L^

    @staticmethod
    def randn_init_latents[BATCH: Int](
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        seed: UInt64,
        offset: UInt64 = 0,
    ):
        """Fill `latents` ∼ N(0, 1) via Box-Muller on Philox.

        Matches the PyTorch reference:  init_latents = torch.randn(...).
        """
        var size = BATCH * Self.MODEL.LATENT_SIZE_PER_SAMPLE
        var ptr = latents.ptr
        var i = 0
        var pair_idx: UInt64 = 0
        while i < size:
            var rng1 = PhiloxRandom(seed=seed, offset=offset + pair_idx * 2)
            var rng2 = PhiloxRandom(seed=seed, offset=offset + pair_idx * 2 + 1)
            var u1 = rng1.step_uniform()[0]
            var u2 = rng2.step_uniform()[0]
            pair_idx += 1
            if u1 < 1e-10:
                u1 = 1e-10
            var r = sqrt(-2.0 * log(u1))
            var z0 = r * cos(2.0 * pi * u2)
            ptr[i] = Scalar[Self.dtype](z0)
            i += 1
            if i < size:
                var z1 = r * sin(2.0 * pi * u2)
                ptr[i] = Scalar[Self.dtype](z1)
                i += 1

    # =========================================================================
    # One full batch (T_infer inference steps + T_learn learning steps)
    # =========================================================================

    @staticmethod
    def train_one_batch[BATCH: Int](
        mut params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        x_input: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ],
        y_target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ],
        T_infer: Int,
        T_learn: Int,
        eta_infer: Scalar[Self.dtype],
        eta_learn: Scalar[Self.dtype],
    ) -> PCTrainResult:
        """Run one full PCN batch (inference + learning); return final energy."""

        # ── Allocate scratch buffers ──────────────────────────────────────
        # Four layer-scratch buffers (a, x_hat, eps, h) each sized
        # BATCH * sum(IN_DIM_i over all layers).  Layer-major layout:
        #   [layer 0's BATCH*IN_DIM_0 floats][layer 1's...][...]
        comptime SCRATCH = BATCH * Self.MODEL.LAYER_SCRATCH_PER_SAMPLE
        var a_buf = alloc[Scalar[Self.dtype]](SCRATCH)
        var xh_buf = alloc[Scalar[Self.dtype]](SCRATCH)
        var eps_buf = alloc[Scalar[Self.dtype]](SCRATCH)
        var h_buf = alloc[Scalar[Self.dtype]](SCRATCH)
        memset(a_buf, 0, SCRATCH)
        memset(xh_buf, 0, SCRATCH)
        memset(eps_buf, 0, SCRATCH)
        memset(h_buf, 0, SCRATCH)

        # eps_L pulled back from supervised error -> dim TOP_LATENT_DIM
        comptime EPSL_SIZE = BATCH * Self.MODEL.TOP_LATENT_DIM
        var epsL_buf = alloc[Scalar[Self.dtype]](EPSL_SIZE)
        memset(epsL_buf, 0, EPSL_SIZE)

        # Per-latent pull-back buffer (allocated max-sized, since each latent
        # has a different dim — we reuse via offset within latent_pb_buf).
        # Simpler: allocate once at max latent dim.
        var max_lat_dim = 0
        comptime for i in range(Self.MODEL.N_LATENTS):
            var d = Self.MODEL.layer_types[i].OUT_DIM
            if d > max_lat_dim:
                max_lat_dim = d
        var pb_buf = alloc[Scalar[Self.dtype]](BATCH * max_lat_dim)

        # ── Inference phase ───────────────────────────────────────────────
        for _ in range(T_infer):
            Self._snapshot[BATCH](
                latents, x_input, params,
                a_buf, xh_buf, eps_buf, h_buf,
            )
            Self._readout_snapshot[BATCH](
                latents, y_target, params,
                a_buf, xh_buf, eps_buf, h_buf, epsL_buf,
            )
            Self._update_latents[BATCH](
                latents, params, eps_buf, h_buf, epsL_buf, pb_buf, eta_infer,
            )

        # ── Learning phase ────────────────────────────────────────────────
        for _ in range(T_learn):
            Self._snapshot[BATCH](
                latents, x_input, params,
                a_buf, xh_buf, eps_buf, h_buf,
            )
            Self._readout_snapshot[BATCH](
                latents, y_target, params,
                a_buf, xh_buf, eps_buf, h_buf, epsL_buf,
            )
            Self._update_weights[BATCH](
                params, latents, x_input, h_buf, eta_learn,
            )

        # ── Final energy report (for the user; recompute snapshot once more) ──
        Self._snapshot[BATCH](
            latents, x_input, params,
            a_buf, xh_buf, eps_buf, h_buf,
        )
        Self._readout_snapshot[BATCH](
            latents, y_target, params,
            a_buf, xh_buf, eps_buf, h_buf, epsL_buf,
        )
        var energy = Self._compute_energy[BATCH](eps_buf)
        var sup_loss = Self._compute_sup_loss[BATCH](
            xh_buf, y_target,
        )

        a_buf.free()
        xh_buf.free()
        eps_buf.free()
        h_buf.free()
        epsL_buf.free()
        pb_buf.free()

        return PCTrainResult(energy, sup_loss)

    # =========================================================================
    # Inner-step helpers
    # =========================================================================

    @staticmethod
    def _snapshot[BATCH: Int](
        latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        x_input: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ],
        params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        a_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        xh_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        eps_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        h_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ):
        """Compute a_l, x_hat_l, eps_l, h_l for all NON-readout layers l = 0..L-1.

        eps_l = (x_input if l=0 else latent[l-1]) - x_hat_l   (PyTorch sign convention)
        """
        comptime for i in range(Self.MODEL.N_LATENTS):  # i.e., l = 0..L-1
            comptime IN_I = Self.MODEL.layer_types[i].IN_DIM
            comptime OUT_I = Self.MODEL.layer_types[i].OUT_DIM
            comptime PSZ_I = Self.MODEL.layer_types[i].PARAM_SIZE
            comptime SCR_OFF_I = Self.MODEL._layer_scratch_offset[i]()
            comptime LAT_OFF_I = Self.MODEL._latent_offset[i]()
            comptime PAR_OFF_I = Self.MODEL._param_offset[i]()

            var x_above = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, OUT_I), MutAnyOrigin
            ](latents.ptr + BATCH * LAT_OFF_I)
            var li_p = LayoutTensor[
                Self.dtype, Layout.row_major(PSZ_I), MutAnyOrigin
            ](params.ptr + PAR_OFF_I)
            var a_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_I), MutAnyOrigin
            ](a_buf + BATCH * SCR_OFF_I)
            var xh_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_I), MutAnyOrigin
            ](xh_buf + BATCH * SCR_OFF_I)
            var eps_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_I), MutAnyOrigin
            ](eps_buf + BATCH * SCR_OFF_I)
            var h_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_I), MutAnyOrigin
            ](h_buf + BATCH * SCR_OFF_I)

            # predict: a_view, xh_view = layer.predict(x_above, params)
            Self.MODEL.layer_types[i].predict[BATCH, Self.dtype](
                x_above, li_p, xh_view, a_view
            )

            # eps = target - x_hat
            comptime if i == 0:
                for b in range(BATCH):
                    for j in range(IN_I):
                        eps_view[b, j] = x_input[b, j] - xh_view[b, j]
            else:
                comptime PREV_OUT = Self.MODEL.layer_types[i - 1].OUT_DIM
                comptime PREV_LAT_OFF = Self.MODEL._latent_offset[i - 1]()
                var prev_lat = LayoutTensor[
                    Self.dtype, Layout.row_major(BATCH, PREV_OUT), MutAnyOrigin
                ](latents.ptr + BATCH * PREV_LAT_OFF)
                # prev_lat dim must equal IN_I (composition constraint)
                for b in range(BATCH):
                    for j in range(IN_I):
                        eps_view[b, j] = prev_lat[b, j] - xh_view[b, j]

            # h = eps * f'(a)
            Self.MODEL.layer_types[i].gain_modulated_error[BATCH, Self.dtype](
                eps_view, a_view, h_view
            )

    @staticmethod
    def _readout_snapshot[BATCH: Int](
        latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        y_target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ],
        params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        a_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        xh_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        eps_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        h_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        epsL_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ):
        """Readout layer (index R = N_LINEARS-1).

        a_R = x_hat_R = top_latent @ W_R^T  (identity activation)
        eps_sup = x_hat_R - y_target           ← OPPOSITE sign vs non-readout
        h_R = eps_sup                            (identity derivative = 1)
        eps_L_pulled = h_R @ W_R               ← used in latent update of x^(L)
        """
        comptime R = Self.MODEL.N_LINEARS - 1
        comptime IN_R = Self.MODEL.layer_types[R].IN_DIM        # = NUM_CLASSES
        comptime OUT_R = Self.MODEL.layer_types[R].OUT_DIM      # = TOP_LATENT_DIM
        comptime PSZ_R = Self.MODEL.layer_types[R].PARAM_SIZE
        comptime SCR_OFF_R = Self.MODEL._layer_scratch_offset[R]()
        comptime PAR_OFF_R = Self.MODEL._param_offset[R]()
        comptime TOP_LAT_OFF = Self.MODEL._latent_offset[R - 1]()  # x^(L) = latent[L-1]

        var top_latent = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OUT_R), MutAnyOrigin
        ](latents.ptr + BATCH * TOP_LAT_OFF)
        var W_R_p = LayoutTensor[
            Self.dtype, Layout.row_major(PSZ_R), MutAnyOrigin
        ](params.ptr + PAR_OFF_R)
        var a_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](a_buf + BATCH * SCR_OFF_R)
        var xh_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](xh_buf + BATCH * SCR_OFF_R)
        var eps_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](eps_buf + BATCH * SCR_OFF_R)
        var h_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](h_buf + BATCH * SCR_OFF_R)
        var epsL_view = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OUT_R), MutAnyOrigin
        ](epsL_buf)

        # predict (a = x_hat with identity activation)
        Self.MODEL.layer_types[R].predict[BATCH, Self.dtype](
            top_latent, W_R_p, xh_R, a_R
        )

        # eps_sup = x_hat_R - y_target  (OPPOSITE sign vs non-readout!)
        for b in range(BATCH):
            for j in range(IN_R):
                eps_R[b, j] = xh_R[b, j] - y_target[b, j]

        # h_R = eps_sup * f'(a) = eps_sup (identity)
        Self.MODEL.layer_types[R].gain_modulated_error[BATCH, Self.dtype](
            eps_R, a_R, h_R
        )

        # eps_L_pulled = h_R @ W_R
        Self.MODEL.layer_types[R].pull_back[BATCH, Self.dtype](
            h_R, W_R_p, epsL_view
        )

    @staticmethod
    def _update_latents[BATCH: Int](
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        eps_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        h_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        epsL_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        pb_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        eta_infer: Scalar[Self.dtype],
    ):
        """For l = 1..L:
              err_ext_l = eps_l         if l < L  (= eps_buf slice for layer l)
                        = epsL_pulled   if l = L
              grad_X_l  = err_ext_l - (h_{l-1} @ W_{l-1})
              latent[l-1] -= eta_infer * grad_X_l
        """
        comptime for li in range(Self.MODEL.N_LATENTS):
            # li is the latent index 0..L-1, corresponds to paper's l-1 = li.
            # Paper's l = li + 1.
            comptime l_paper = li + 1
            comptime LAT_DIM = Self.MODEL.layer_types[li].OUT_DIM  # dim of x^(l_paper)

            var lat_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, LAT_DIM), MutAnyOrigin
            ](latents.ptr + BATCH * Self.MODEL._latent_offset[li]())

            # ── err_ext_l ───────────────────────────────────────────────
            var err_ext = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, LAT_DIM), MutAnyOrigin
            ](pb_buf)  # reuse pb_buf as a temporary holder (we copy into it)

            comptime if l_paper < Self.MODEL.N_LATENTS:
                # err_ext = eps_buf slice for layer l_paper (which has IN_DIM = LAT_DIM)
                comptime SCR_OFF = Self.MODEL._layer_scratch_offset[l_paper]()
                var eps_l = LayoutTensor[
                    Self.dtype, Layout.row_major(BATCH, LAT_DIM), MutAnyOrigin
                ](eps_buf + BATCH * SCR_OFF)
                for b in range(BATCH):
                    for j in range(LAT_DIM):
                        err_ext[b, j] = eps_l[b, j]
            else:
                # l_paper == N_LATENTS == L → err_ext = epsL_pulled (top latent)
                var epsL_view = LayoutTensor[
                    Self.dtype, Layout.row_major(BATCH, LAT_DIM), MutAnyOrigin
                ](epsL_buf)
                for b in range(BATCH):
                    for j in range(LAT_DIM):
                        err_ext[b, j] = epsL_view[b, j]

            # ── pulled-back gain-modulated error from layer l_paper - 1 = li ──
            comptime IN_PREV = Self.MODEL.layer_types[li].IN_DIM
            comptime SCR_OFF_PREV = Self.MODEL._layer_scratch_offset[li]()
            comptime PAR_OFF_PREV = Self.MODEL._param_offset[li]()
            comptime PSZ_PREV = Self.MODEL.layer_types[li].PARAM_SIZE

            var h_prev = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_PREV), MutAnyOrigin
            ](h_buf + BATCH * SCR_OFF_PREV)
            var W_prev_p = LayoutTensor[
                Self.dtype, Layout.row_major(PSZ_PREV), MutAnyOrigin
            ](params.ptr + PAR_OFF_PREV)

            # We need a separate temp for the pull-back result (don't trash err_ext).
            # Reuse the END of pb_buf: but it's already used by err_ext. Allocate inline.
            var pb_storage = List[Scalar[Self.dtype]](capacity=BATCH * LAT_DIM)
            for _ in range(BATCH * LAT_DIM):
                pb_storage.append(Scalar[Self.dtype](0))
            var pb_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, LAT_DIM), MutAnyOrigin
            ](pb_storage.unsafe_ptr())

            Self.MODEL.layer_types[li].pull_back[BATCH, Self.dtype](
                h_prev, W_prev_p, pb_view
            )

            # ── grad_X_l = err_ext - pb;  x^(l) -= eta_infer * grad_X_l ──
            for b in range(BATCH):
                for j in range(LAT_DIM):
                    var g = err_ext[b, j] - pb_view[b, j]
                    lat_view[b, j] -= eta_infer * g

            # Keep `pb_storage` alive until end of iteration (Mojo will drop it here).
            _ = pb_storage^

    @staticmethod
    def _update_weights[BATCH: Int](
        mut params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        x_input: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ],
        h_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        eta_learn: Scalar[Self.dtype],
    ):
        """Weight updates for ALL layers (non-readout + readout).

        Non-readout (i = 0..L-1):  W_i +=  +eta/B * (h_i^T @ x_above_i)
        Readout    (i = R = L)  :  W_R +=  -eta/B * (h_R^T @ x_above_R)
                                   (h_R = eps_sup; sign flip handled via scale.)

        x_above_i = latent[i] for all i, EXCEPT readout uses latent[i-1] (= x^(L)).
        """
        var inv_B = Scalar[Self.dtype](1) / Scalar[Self.dtype](BATCH)
        var scale_pos = eta_learn * inv_B
        var scale_neg = -eta_learn * inv_B

        # Non-readout layers
        comptime for i in range(Self.MODEL.N_LATENTS):
            comptime OUT_I = Self.MODEL.layer_types[i].OUT_DIM
            comptime SCR_OFF_I = Self.MODEL._layer_scratch_offset[i]()
            comptime LAT_OFF_I = Self.MODEL._latent_offset[i]()
            comptime PAR_OFF_I = Self.MODEL._param_offset[i]()
            comptime PSZ_I = Self.MODEL.layer_types[i].PARAM_SIZE
            comptime IN_I = Self.MODEL.layer_types[i].IN_DIM

            var x_above = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, OUT_I), MutAnyOrigin
            ](latents.ptr + BATCH * LAT_OFF_I)
            var h_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_I), MutAnyOrigin
            ](h_buf + BATCH * SCR_OFF_I)
            var W_p = LayoutTensor[
                Self.dtype, Layout.row_major(PSZ_I), MutAnyOrigin
            ](params.ptr + PAR_OFF_I)

            Self.MODEL.layer_types[i].weight_grad_step[BATCH, Self.dtype](
                h_view, x_above, W_p, scale_pos
            )

        # Readout (i = R = N_LINEARS - 1)
        comptime R = Self.MODEL.N_LINEARS - 1
        comptime IN_R = Self.MODEL.layer_types[R].IN_DIM
        comptime OUT_R = Self.MODEL.layer_types[R].OUT_DIM
        comptime SCR_OFF_R = Self.MODEL._layer_scratch_offset[R]()
        comptime PAR_OFF_R = Self.MODEL._param_offset[R]()
        comptime PSZ_R = Self.MODEL.layer_types[R].PARAM_SIZE
        comptime TOP_LAT_OFF = Self.MODEL._latent_offset[R - 1]()

        var x_above_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OUT_R), MutAnyOrigin
        ](latents.ptr + BATCH * TOP_LAT_OFF)
        var h_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](h_buf + BATCH * SCR_OFF_R)
        var W_R_p = LayoutTensor[
            Self.dtype, Layout.row_major(PSZ_R), MutAnyOrigin
        ](params.ptr + PAR_OFF_R)

        Self.MODEL.layer_types[R].weight_grad_step[BATCH, Self.dtype](
            h_R, x_above_R, W_R_p, scale_neg
        )

    # =========================================================================
    # Diagnostics
    # =========================================================================

    @staticmethod
    def _compute_energy[BATCH: Int](
        eps_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ) -> Float64:
        """½ Σ ‖eps_l‖² averaged over batch (non-readout layers only)."""
        var total: Float64 = 0.0
        comptime for i in range(Self.MODEL.N_LATENTS):
            comptime IN_I = Self.MODEL.layer_types[i].IN_DIM
            comptime SCR_OFF = Self.MODEL._layer_scratch_offset[i]()
            for b in range(BATCH):
                for j in range(IN_I):
                    var v = Float64(eps_buf[BATCH * SCR_OFF + b * IN_I + j])
                    total += v * v
        return 0.5 * total / Float64(BATCH)

    @staticmethod
    def _compute_sup_loss[BATCH: Int](
        xh_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        y_target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ],
    ) -> Float64:
        """½ ‖y_hat - y_target‖² averaged over batch."""
        comptime R = Self.MODEL.N_LINEARS - 1
        comptime IN_R = Self.MODEL.layer_types[R].IN_DIM
        comptime SCR_OFF_R = Self.MODEL._layer_scratch_offset[R]()
        var total: Float64 = 0.0
        comptime OUT_DIM = Self.MODEL.OUT_DIM
        for b in range(BATCH):
            for j in range(IN_R):
                var yh = Float64(xh_buf[BATCH * SCR_OFF_R + b * IN_R + j])
                var yt = Float64(y_target.ptr[b * OUT_DIM + j])
                var d = yh - yt
                total += d * d
        return 0.5 * total / Float64(BATCH)

    # =========================================================================
    # ──────────────────────  GPU  PATH  ────────────────────────────────────
    # =========================================================================
    # All buffers (params, latents, x_input, y_target) are LayoutTensors backed
    # by DeviceBuffer device memory. Scratch is allocated per-batch via
    # ctx.enqueue_create_buffer. The structure mirrors the CPU path; per-step
    # operations dispatch to PCLinear.predict_gpu / pull_back_gpu / etc.

    # ── Elementwise helper kernels ────────────────────────────────────────

    @staticmethod
    def _sub_kernel[
        BATCH: Int, DIM: Int, dt: DType,
    ](
        target: LayoutTensor[
            dt, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        x_hat: LayoutTensor[
            dt, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        eps: LayoutTensor[
            dt, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        """eps = target - x_hat  (used for non-readout layers)."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        eps[b, i] = target[b, i] - x_hat[b, i]

    @staticmethod
    def _sub_swap_kernel[
        BATCH: Int, DIM: Int, dt: DType,
    ](
        x_hat: LayoutTensor[
            dt, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        y_target: LayoutTensor[
            dt, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        eps_sup: LayoutTensor[
            dt, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
    ):
        """eps_sup = x_hat - y_target  (readout's opposite-sign convention)."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        eps_sup[b, i] = x_hat[b, i] - y_target[b, i]

    @staticmethod
    def _latent_update_kernel[
        BATCH: Int, DIM: Int, dt: DType,
    ](
        err_ext: LayoutTensor[
            dt, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        pull_back: LayoutTensor[
            dt, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        latent: LayoutTensor[
            dt, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ],
        eta_infer: Scalar[dt],
    ):
        """latent -= eta_infer * (err_ext - pull_back)."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var j = idx % DIM
        var grad = err_ext[b, j] - pull_back[b, j]
        latent[b, j] -= eta_infer * grad

    # ── Inner GPU helpers (mirror the CPU _snapshot / _readout / _update_*) ──

    @staticmethod
    def _snapshot_gpu[BATCH: Int](
        ctx: DeviceContext,
        latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        x_input: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ],
        params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        a_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        xh_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        eps_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        h_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ) raises:
        comptime for i in range(Self.MODEL.N_LATENTS):
            comptime IN_I = Self.MODEL.layer_types[i].IN_DIM
            comptime OUT_I = Self.MODEL.layer_types[i].OUT_DIM
            comptime PSZ_I = Self.MODEL.layer_types[i].PARAM_SIZE
            comptime SCR_OFF_I = Self.MODEL._layer_scratch_offset[i]()
            comptime LAT_OFF_I = Self.MODEL._latent_offset[i]()
            comptime PAR_OFF_I = Self.MODEL._param_offset[i]()

            var x_above = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, OUT_I), MutAnyOrigin
            ](latents.ptr + BATCH * LAT_OFF_I)
            var li_p = LayoutTensor[
                Self.dtype, Layout.row_major(PSZ_I), MutAnyOrigin
            ](params.ptr + PAR_OFF_I)
            var a_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_I), MutAnyOrigin
            ](a_buf + BATCH * SCR_OFF_I)
            var xh_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_I), MutAnyOrigin
            ](xh_buf + BATCH * SCR_OFF_I)
            var eps_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_I), MutAnyOrigin
            ](eps_buf + BATCH * SCR_OFF_I)
            var h_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_I), MutAnyOrigin
            ](h_buf + BATCH * SCR_OFF_I)

            Self.MODEL.layer_types[i].predict_gpu[BATCH, Self.dtype](
                ctx, x_above, li_p, xh_view, a_view
            )

            # eps = target - x_hat
            comptime if i == 0:
                # target = x_input
                comptime k = Self._sub_kernel[BATCH, IN_I, Self.dtype]
                var threads = BATCH * IN_I
                var blocks = (threads + TPB - 1) // TPB
                ctx.enqueue_function[k, k](
                    x_input, xh_view, eps_view,
                    grid_dim=(blocks,),
                    block_dim=(TPB,),
                )
            else:
                comptime PREV_LAT_OFF = Self.MODEL._latent_offset[i - 1]()
                var prev_lat = LayoutTensor[
                    Self.dtype, Layout.row_major(BATCH, IN_I), MutAnyOrigin
                ](latents.ptr + BATCH * PREV_LAT_OFF)
                comptime k = Self._sub_kernel[BATCH, IN_I, Self.dtype]
                var threads = BATCH * IN_I
                var blocks = (threads + TPB - 1) // TPB
                ctx.enqueue_function[k, k](
                    prev_lat, xh_view, eps_view,
                    grid_dim=(blocks,),
                    block_dim=(TPB,),
                )

            Self.MODEL.layer_types[i].gain_modulated_error_gpu[
                BATCH, Self.dtype
            ](ctx, eps_view, a_view, h_view)

    @staticmethod
    def _readout_snapshot_gpu[BATCH: Int](
        ctx: DeviceContext,
        latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        y_target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ],
        params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        a_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        xh_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        eps_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        h_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        epsL_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ) raises:
        comptime R = Self.MODEL.N_LINEARS - 1
        comptime IN_R = Self.MODEL.layer_types[R].IN_DIM
        comptime OUT_R = Self.MODEL.layer_types[R].OUT_DIM
        comptime PSZ_R = Self.MODEL.layer_types[R].PARAM_SIZE
        comptime SCR_OFF_R = Self.MODEL._layer_scratch_offset[R]()
        comptime PAR_OFF_R = Self.MODEL._param_offset[R]()
        comptime TOP_LAT_OFF = Self.MODEL._latent_offset[R - 1]()

        var top_latent = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OUT_R), MutAnyOrigin
        ](latents.ptr + BATCH * TOP_LAT_OFF)
        var W_R_p = LayoutTensor[
            Self.dtype, Layout.row_major(PSZ_R), MutAnyOrigin
        ](params.ptr + PAR_OFF_R)
        var a_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](a_buf + BATCH * SCR_OFF_R)
        var xh_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](xh_buf + BATCH * SCR_OFF_R)
        var eps_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](eps_buf + BATCH * SCR_OFF_R)
        var h_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](h_buf + BATCH * SCR_OFF_R)
        var epsL_view = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OUT_R), MutAnyOrigin
        ](epsL_buf)

        Self.MODEL.layer_types[R].predict_gpu[BATCH, Self.dtype](
            ctx, top_latent, W_R_p, xh_R, a_R
        )

        # eps_sup = x_hat - y_target  (opposite sign vs non-readout)
        comptime k_sub = Self._sub_swap_kernel[BATCH, IN_R, Self.dtype]
        var threads_sub = BATCH * IN_R
        var blocks_sub = (threads_sub + TPB - 1) // TPB
        ctx.enqueue_function[k_sub, k_sub](
            xh_R, y_target, eps_R,
            grid_dim=(blocks_sub,),
            block_dim=(TPB,),
        )

        Self.MODEL.layer_types[R].gain_modulated_error_gpu[
            BATCH, Self.dtype
        ](ctx, eps_R, a_R, h_R)

        Self.MODEL.layer_types[R].pull_back_gpu[BATCH, Self.dtype](
            ctx, h_R, W_R_p, epsL_view
        )

    @staticmethod
    def _update_latents_gpu[BATCH: Int](
        ctx: DeviceContext,
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        eps_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        h_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        epsL_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        pb_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        eta_infer: Scalar[Self.dtype],
    ) raises:
        comptime for li in range(Self.MODEL.N_LATENTS):
            comptime l_paper = li + 1
            comptime LAT_DIM = Self.MODEL.layer_types[li].OUT_DIM
            comptime IN_PREV = Self.MODEL.layer_types[li].IN_DIM
            comptime SCR_OFF_PREV = Self.MODEL._layer_scratch_offset[li]()
            comptime PAR_OFF_PREV = Self.MODEL._param_offset[li]()
            comptime PSZ_PREV = Self.MODEL.layer_types[li].PARAM_SIZE

            var lat_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, LAT_DIM), MutAnyOrigin
            ](latents.ptr + BATCH * Self.MODEL._latent_offset[li]())

            var h_prev = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_PREV), MutAnyOrigin
            ](h_buf + BATCH * SCR_OFF_PREV)
            var W_prev_p = LayoutTensor[
                Self.dtype, Layout.row_major(PSZ_PREV), MutAnyOrigin
            ](params.ptr + PAR_OFF_PREV)

            # pb = h_prev @ W_prev (writes to shared pb_buf scratch)
            var pb_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, LAT_DIM), MutAnyOrigin
            ](pb_buf)
            Self.MODEL.layer_types[li].pull_back_gpu[BATCH, Self.dtype](
                ctx, h_prev, W_prev_p, pb_view
            )

            # err_ext: either eps_buf slice (l_paper < L) or epsL_buf (l_paper == L)
            comptime if l_paper < Self.MODEL.N_LATENTS:
                comptime SCR_OFF = Self.MODEL._layer_scratch_offset[l_paper]()
                var err_ext = LayoutTensor[
                    Self.dtype, Layout.row_major(BATCH, LAT_DIM), MutAnyOrigin
                ](eps_buf + BATCH * SCR_OFF)
                comptime ku = Self._latent_update_kernel[
                    BATCH, LAT_DIM, Self.dtype
                ]
                var threads = BATCH * LAT_DIM
                var blocks = (threads + TPB - 1) // TPB
                ctx.enqueue_function[ku, ku](
                    err_ext, pb_view, lat_view, eta_infer,
                    grid_dim=(blocks,),
                    block_dim=(TPB,),
                )
            else:
                var err_ext = LayoutTensor[
                    Self.dtype, Layout.row_major(BATCH, LAT_DIM), MutAnyOrigin
                ](epsL_buf)
                comptime ku = Self._latent_update_kernel[
                    BATCH, LAT_DIM, Self.dtype
                ]
                var threads = BATCH * LAT_DIM
                var blocks = (threads + TPB - 1) // TPB
                ctx.enqueue_function[ku, ku](
                    err_ext, pb_view, lat_view, eta_infer,
                    grid_dim=(blocks,),
                    block_dim=(TPB,),
                )

    @staticmethod
    def _update_weights_gpu[BATCH: Int](
        ctx: DeviceContext,
        mut params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        x_input: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ],
        h_buf: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        eta_learn: Scalar[Self.dtype],
    ) raises:
        var inv_B = Scalar[Self.dtype](1) / Scalar[Self.dtype](BATCH)
        var scale_pos = eta_learn * inv_B
        var scale_neg = -eta_learn * inv_B

        comptime for i in range(Self.MODEL.N_LATENTS):
            comptime OUT_I = Self.MODEL.layer_types[i].OUT_DIM
            comptime SCR_OFF_I = Self.MODEL._layer_scratch_offset[i]()
            comptime LAT_OFF_I = Self.MODEL._latent_offset[i]()
            comptime PAR_OFF_I = Self.MODEL._param_offset[i]()
            comptime PSZ_I = Self.MODEL.layer_types[i].PARAM_SIZE
            comptime IN_I = Self.MODEL.layer_types[i].IN_DIM

            var x_above = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, OUT_I), MutAnyOrigin
            ](latents.ptr + BATCH * LAT_OFF_I)
            var h_view = LayoutTensor[
                Self.dtype, Layout.row_major(BATCH, IN_I), MutAnyOrigin
            ](h_buf + BATCH * SCR_OFF_I)
            var W_p = LayoutTensor[
                Self.dtype, Layout.row_major(PSZ_I), MutAnyOrigin
            ](params.ptr + PAR_OFF_I)

            Self.MODEL.layer_types[i].weight_grad_step_gpu[
                BATCH, Self.dtype
            ](ctx, h_view, x_above, W_p, scale_pos)

        comptime R = Self.MODEL.N_LINEARS - 1
        comptime IN_R = Self.MODEL.layer_types[R].IN_DIM
        comptime OUT_R = Self.MODEL.layer_types[R].OUT_DIM
        comptime SCR_OFF_R = Self.MODEL._layer_scratch_offset[R]()
        comptime PAR_OFF_R = Self.MODEL._param_offset[R]()
        comptime PSZ_R = Self.MODEL.layer_types[R].PARAM_SIZE
        comptime TOP_LAT_OFF = Self.MODEL._latent_offset[R - 1]()

        var x_above_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OUT_R), MutAnyOrigin
        ](latents.ptr + BATCH * TOP_LAT_OFF)
        var h_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](h_buf + BATCH * SCR_OFF_R)
        var W_R_p = LayoutTensor[
            Self.dtype, Layout.row_major(PSZ_R), MutAnyOrigin
        ](params.ptr + PAR_OFF_R)

        Self.MODEL.layer_types[R].weight_grad_step_gpu[
            BATCH, Self.dtype
        ](ctx, h_R, x_above_R, W_R_p, scale_neg)

    # ── Public GPU entry point ────────────────────────────────────────────

    @staticmethod
    def train_one_batch_gpu[BATCH: Int](
        ctx: DeviceContext,
        mut params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        x_input: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ],
        y_target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ],
        T_infer: Int,
        T_learn: Int,
        eta_infer: Scalar[Self.dtype],
        eta_learn: Scalar[Self.dtype],
    ) raises:
        """GPU path. Mirrors `train_one_batch` exactly; energy/sup_loss are
        not computed (caller can sync + read params/latents back if needed)."""

        comptime SCRATCH = BATCH * Self.MODEL.LAYER_SCRATCH_PER_SAMPLE
        comptime EPSL_SIZE = BATCH * Self.MODEL.TOP_LATENT_DIM

        var max_lat_dim = 0
        comptime for i in range(Self.MODEL.N_LATENTS):
            var d = Self.MODEL.layer_types[i].OUT_DIM
            if d > max_lat_dim:
                max_lat_dim = d
        var pb_size = BATCH * max_lat_dim

        var a_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var xh_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var eps_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var h_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var epsL_dbuf = ctx.enqueue_create_buffer[Self.dtype](EPSL_SIZE)
        var pb_dbuf = ctx.enqueue_create_buffer[Self.dtype](pb_size)

        # Inference phase
        for _ in range(T_infer):
            Self._snapshot_gpu[BATCH](
                ctx, latents, x_input, params,
                a_dbuf.unsafe_ptr(), xh_dbuf.unsafe_ptr(),
                eps_dbuf.unsafe_ptr(), h_dbuf.unsafe_ptr(),
            )
            Self._readout_snapshot_gpu[BATCH](
                ctx, latents, y_target, params,
                a_dbuf.unsafe_ptr(), xh_dbuf.unsafe_ptr(),
                eps_dbuf.unsafe_ptr(), h_dbuf.unsafe_ptr(), epsL_dbuf.unsafe_ptr(),
            )
            Self._update_latents_gpu[BATCH](
                ctx, latents, params,
                eps_dbuf.unsafe_ptr(), h_dbuf.unsafe_ptr(), epsL_dbuf.unsafe_ptr(),
                pb_dbuf.unsafe_ptr(), eta_infer,
            )

        # Learning phase
        for _ in range(T_learn):
            Self._snapshot_gpu[BATCH](
                ctx, latents, x_input, params,
                a_dbuf.unsafe_ptr(), xh_dbuf.unsafe_ptr(),
                eps_dbuf.unsafe_ptr(), h_dbuf.unsafe_ptr(),
            )
            Self._readout_snapshot_gpu[BATCH](
                ctx, latents, y_target, params,
                a_dbuf.unsafe_ptr(), xh_dbuf.unsafe_ptr(),
                eps_dbuf.unsafe_ptr(), h_dbuf.unsafe_ptr(), epsL_dbuf.unsafe_ptr(),
            )
            Self._update_weights_gpu[BATCH](
                ctx, params, latents, x_input, h_buf=h_dbuf.unsafe_ptr(),
                eta_learn=eta_learn,
            )

    # =========================================================================
    # Inference (test-time): "free" PCN inference, no supervised signal.
    # =========================================================================

    @staticmethod
    def _zero_kernel[
        SIZE: Int, dt: DType,
    ](
        buf: LayoutTensor[
            dt, Layout.row_major(SIZE), MutAnyOrigin
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= SIZE:
            return
        buf[idx] = Scalar[dt](0)

    @staticmethod
    def inference_gpu[BATCH: Int](
        ctx: DeviceContext,
        params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        x_input: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ],
        mut y_hat: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ],
        T_infer: Int,
        eta_infer: Scalar[Self.dtype],
    ) raises:
        """Free PCN inference: no supervised signal, only prediction errors.

        Caller pre-initializes `latents` (e.g., via randn_init_latents). After
        T_infer settling steps, the readout output is written to `y_hat`.
        """
        comptime SCRATCH = BATCH * Self.MODEL.LAYER_SCRATCH_PER_SAMPLE
        comptime EPSL_SIZE = BATCH * Self.MODEL.TOP_LATENT_DIM

        var max_lat_dim = 0
        comptime for i in range(Self.MODEL.N_LATENTS):
            var d = Self.MODEL.layer_types[i].OUT_DIM
            if d > max_lat_dim:
                max_lat_dim = d

        var a_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var xh_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var eps_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var h_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var epsL_dbuf = ctx.enqueue_create_buffer[Self.dtype](EPSL_SIZE)
        var pb_dbuf = ctx.enqueue_create_buffer[Self.dtype](BATCH * max_lat_dim)

        # Zero epsL once — this remains zero throughout (no supervised pull-back).
        var epsL_view = LayoutTensor[
            Self.dtype, Layout.row_major(EPSL_SIZE), MutAnyOrigin
        ](epsL_dbuf)
        comptime kz = Self._zero_kernel[EPSL_SIZE, Self.dtype]
        var blocks_z = (EPSL_SIZE + TPB - 1) // TPB
        ctx.enqueue_function[kz, kz](
            epsL_view, grid_dim=(blocks_z,), block_dim=(TPB,)
        )

        for _ in range(T_infer):
            Self._snapshot_gpu[BATCH](
                ctx, latents, x_input, params,
                a_dbuf.unsafe_ptr(), xh_dbuf.unsafe_ptr(),
                eps_dbuf.unsafe_ptr(), h_dbuf.unsafe_ptr(),
            )
            Self._update_latents_gpu[BATCH](
                ctx, latents, params,
                eps_dbuf.unsafe_ptr(), h_dbuf.unsafe_ptr(),
                epsL_dbuf.unsafe_ptr(), pb_dbuf.unsafe_ptr(), eta_infer,
            )

        # Final readout: y_hat = readout.predict(top_latent)
        comptime R = Self.MODEL.N_LINEARS - 1
        comptime IN_R = Self.MODEL.layer_types[R].IN_DIM
        comptime OUT_R = Self.MODEL.layer_types[R].OUT_DIM
        comptime PSZ_R = Self.MODEL.layer_types[R].PARAM_SIZE
        comptime SCR_OFF_R = Self.MODEL._layer_scratch_offset[R]()
        comptime PAR_OFF_R = Self.MODEL._param_offset[R]()
        comptime TOP_LAT_OFF = Self.MODEL._latent_offset[R - 1]()

        var top_latent = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OUT_R), MutAnyOrigin
        ](latents.ptr + BATCH * TOP_LAT_OFF)
        var W_R_p = LayoutTensor[
            Self.dtype, Layout.row_major(PSZ_R), MutAnyOrigin
        ](params.ptr + PAR_OFF_R)
        var a_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](a_dbuf.unsafe_ptr() + BATCH * SCR_OFF_R)

        # Identity activation, so x_hat == a; we use y_hat as both buffers.
        Self.MODEL.layer_types[R].predict_gpu[BATCH, Self.dtype](
            ctx, top_latent, W_R_p, y_hat, a_R
        )

    @staticmethod
    def supervised_inference_gpu[BATCH: Int](
        ctx: DeviceContext,
        params: LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.MODEL.PARAM_SIZE),
            MutAnyOrigin,
        ],
        mut latents: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.LATENT_SIZE_PER_SAMPLE),
            MutAnyOrigin,
        ],
        x_input: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.IN_DIM),
            MutAnyOrigin,
        ],
        y_target: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ],
        mut y_hat: LayoutTensor[
            Self.dtype,
            Layout.row_major(BATCH, Self.MODEL.OUT_DIM),
            MutAnyOrigin,
        ],
        T_infer: Int,
        eta_infer: Scalar[Self.dtype],
    ) raises:
        """Supervised PCN inference (matches the paper's `test_pcn` protocol).

        Identical to `inference_gpu` but the supervised pull-back
        `eps_L = (y_hat - y_target) @ W_R` is recomputed each step instead of
        being held at zero. The label drives the top latent dynamics during
        settling — this is the protocol the arxiv 2506.06332 paper uses to
        report 99.92% on CIFAR-10. NOT a generalization metric: y_target is
        from the test set.
        """
        comptime SCRATCH = BATCH * Self.MODEL.LAYER_SCRATCH_PER_SAMPLE
        comptime EPSL_SIZE = BATCH * Self.MODEL.TOP_LATENT_DIM

        var max_lat_dim = 0
        comptime for i in range(Self.MODEL.N_LATENTS):
            var d = Self.MODEL.layer_types[i].OUT_DIM
            if d > max_lat_dim:
                max_lat_dim = d

        var a_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var xh_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var eps_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var h_dbuf = ctx.enqueue_create_buffer[Self.dtype](SCRATCH)
        var epsL_dbuf = ctx.enqueue_create_buffer[Self.dtype](EPSL_SIZE)
        var pb_dbuf = ctx.enqueue_create_buffer[Self.dtype](BATCH * max_lat_dim)

        for _ in range(T_infer):
            Self._snapshot_gpu[BATCH](
                ctx, latents, x_input, params,
                a_dbuf.unsafe_ptr(), xh_dbuf.unsafe_ptr(),
                eps_dbuf.unsafe_ptr(), h_dbuf.unsafe_ptr(),
            )
            Self._readout_snapshot_gpu[BATCH](
                ctx, latents, y_target, params,
                a_dbuf.unsafe_ptr(), xh_dbuf.unsafe_ptr(),
                eps_dbuf.unsafe_ptr(), h_dbuf.unsafe_ptr(),
                epsL_dbuf.unsafe_ptr(),
            )
            Self._update_latents_gpu[BATCH](
                ctx, latents, params,
                eps_dbuf.unsafe_ptr(), h_dbuf.unsafe_ptr(),
                epsL_dbuf.unsafe_ptr(), pb_dbuf.unsafe_ptr(), eta_infer,
            )

        # Final readout
        comptime R = Self.MODEL.N_LINEARS - 1
        comptime IN_R = Self.MODEL.layer_types[R].IN_DIM
        comptime OUT_R = Self.MODEL.layer_types[R].OUT_DIM
        comptime PSZ_R = Self.MODEL.layer_types[R].PARAM_SIZE
        comptime SCR_OFF_R = Self.MODEL._layer_scratch_offset[R]()
        comptime PAR_OFF_R = Self.MODEL._param_offset[R]()
        comptime TOP_LAT_OFF = Self.MODEL._latent_offset[R - 1]()

        var top_latent = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, OUT_R), MutAnyOrigin
        ](latents.ptr + BATCH * TOP_LAT_OFF)
        var W_R_p = LayoutTensor[
            Self.dtype, Layout.row_major(PSZ_R), MutAnyOrigin
        ](params.ptr + PAR_OFF_R)
        var a_R = LayoutTensor[
            Self.dtype, Layout.row_major(BATCH, IN_R), MutAnyOrigin
        ](a_dbuf.unsafe_ptr() + BATCH * SCR_OFF_R)

        Self.MODEL.layer_types[R].predict_gpu[BATCH, Self.dtype](
            ctx, top_latent, W_R_p, y_hat, a_R
        )
