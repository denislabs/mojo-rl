"""Conv2D[IC, OC, K, S, P, H, W] — 2D convolution via im2col + `max_matmul`.

Phase 5 of `nn2/PORTING_PLAN.md`. Reduces convolution to BATCH per-batch
matmuls (`out_b = weight @ im2col(input_b).T`), mirroring the legacy
`mojo_rl/nn/autodiff/primitives/conv2d.mojo:281` Apple/non-Apple path.
The matmul itself flows through `linalg.matmul`'s `max_matmul`, so on
Apple it lands on the Accelerate cblas kernel; on NVIDIA / generic CPU
it falls through to the platform-best implementation.

Layouts (per batch):
    weight:  [OC, IC·K·K]                         row-major (canonical)
    col:     [OH·OW, IC·K·K]                      row-major (im2col output)
    out:     [OC,   OH·OW]                        row-major
    out = weight @ col.T   (matmul `transpose_b=True`)

Why per-batch matmul instead of one big GEMM: keeps the convolution
free of any explicit reshape between BLAS-friendly `[OC, BATCH·OH·OW]`
and the trait-mandated `[BATCH, OC·OH·OW]` flat order. BATCH is
typically small (≤256), so the per-batch GEMM overhead is dominated by
the matmul itself. The legacy nn package has a batched-Apple variant
that re-packs the im2col across the batch to do one big sgemm; we
deliberately ship the per-batch path first and gate the Apple-batched
optimisation on a real CNN consumer that benchmarks the difference.

Backward (per batch):
    d_bias[oc] += Σ_{oh,ow} d_out[oc, oh, ow]
    d_weight   += d_out_b @ col_b     (`[OC, OH·OW] @ [OH·OW, IC·K·K]`,
                                       accumulated across batches)
    d_col_b     = d_out_b.T @ weight  (`max_matmul[transpose_a=True]`)
    d_input_b   = col2im(d_col_b)     (scatter-add into input shape)

Accumulation into `d_weight` uses Apple Accelerate's `cblas_sgemm` with
`beta=1` (single call, no temp alloc) when running on macOS fp32 — same
trick `linear.mojo` uses. On other targets we matmul into a temp slab
and add elementwise. Both paths produce identical numerics modulo
float32 rounding.

CPU only at landing. GPU follow-up is gated on a real CNN consumer.
"""

from std.math import ceildiv
from std.memory import alloc
from std.sys import CompilationTarget
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from ..constants import DT, CPU_SIMD_W
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    ParamVisitor,
    for_each_param_auto,
    zero_grad_auto,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# im2col / col2im helpers — both produce `[OH·OW, IC·K·K]` row-major
# col matrices for one batch sample. Module-level so the Conv2D body
# stays terse and the compiler doesn't have to re-instantiate them per
# struct method.
# ──────────────────────────────────────────────────────────────────────


def _im2col_one_batch[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
](
    in_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    col_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Pack the IC·H·W input slab into an [OH·OW, IC·K·K] col matrix.

    Row index = `oh·OW + ow` (one row per output spatial position).
    Col index inside each row = `(ic·K + kh)·K + kw` (matches the
    weight flat layout `[OC, IC·K·K]` directly, so the matmul lines
    up with no further transpose). Padded receptive fields contribute
    zero — we write 0 for OOB lanes."""
    comptime CK = IC * K * K
    for oh in range(OH):
        for ow in range(OW):
            var row_off = (oh * OW + ow) * CK
            for ic in range(IC):
                var in_c_base = ic * H * W
                var col_ic_base = row_off + ic * K * K
                for kh in range(K):
                    var ih = oh * S + kh - P
                    var col_kh_base = col_ic_base + kh * K
                    for kw in range(K):
                        var iw = ow * S + kw - P
                        if ih < 0 or ih >= H or iw < 0 or iw >= W:
                            col_p[col_kh_base + kw] = Scalar[DT](0.0)
                        else:
                            col_p[col_kh_base + kw] = (
                                in_p[in_c_base + ih * W + iw]
                            )


def _col2im_one_batch[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
](
    d_col_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    d_in_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Scatter-add an [OH·OW, IC·K·K] col matrix back into a [IC·H·W]
    input gradient slab. Assumes `d_in_p` was zero-filled before the
    first call (typically by the Conv2D vjp body). Padded lanes are
    skipped — they never received a meaningful col entry."""
    comptime CK = IC * K * K
    for oh in range(OH):
        for ow in range(OW):
            var row_off = (oh * OW + ow) * CK
            for ic in range(IC):
                var in_c_base = ic * H * W
                var col_ic_base = row_off + ic * K * K
                for kh in range(K):
                    var ih = oh * S + kh - P
                    if ih < 0 or ih >= H:
                        continue
                    var col_kh_base = col_ic_base + kh * K
                    for kw in range(K):
                        var iw = ow * S + kw - P
                        if iw < 0 or iw >= W:
                            continue
                        d_in_p[in_c_base + ih * W + iw] += (
                            d_col_p[col_kh_base + kw]
                        )


struct Conv2D[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
](Module):
    comptime ARITY: Int = 1
    comptime OH: Int = (Self.H + 2 * Self.P - Self.K) // Self.S + 1
    comptime OW: Int = (Self.W + 2 * Self.P - Self.K) // Self.S + 1
    comptime IN_DIM_FLAT: Int = Self.IC * Self.H * Self.W
    comptime OUT_DIM_FLAT: Int = Self.OC * Self.OH * Self.OW
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_DIM_FLAT)
    comptime OUT_DIM = Self.OUT_DIM_FLAT
    comptime W_SIZE: Int = Self.OC * Self.IC * Self.K * Self.K
    comptime B_SIZE: Int = Self.OC
    comptime COL_SIZE: Int = Self.IC * Self.K * Self.K
    comptime SPATIAL_OUT: Int = Self.OH * Self.OW

    var weight: Param["weight", True,  Self.W_SIZE]
    var bias:   Param["bias",   False, Self.B_SIZE]
    var _cached_input_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ts: TargetStorage

    def __init__(out self):
        self.weight = Param["weight", True,  Self.W_SIZE]()
        self.bias   = Param["bias",   False, Self.B_SIZE]()
        self._cached_input_ptr = UnsafePointer[
            Scalar[DT], MutAnyOrigin,
        ](unsafe_from_address=0)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "Conv2D: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.K > 0 and Self.S > 0, (
            "Conv2D: K and S must be positive"
        )
        comptime assert Self.OH > 0 and Self.OW > 0, (
            "Conv2D: invalid spatial shape — check H/W/K/S/P"
        )
        var c = Self()
        comptime if target == "cpu":
            c.weight = Param["weight", True,  Self.W_SIZE].make_cpu()
            c.bias   = Param["bias",   False, Self.B_SIZE].make_cpu()
            # fan_in = IC·K·K, fan_out = OC·K·K — the canonical Kaiming
            # convention for conv weights.
            INIT.init_weight(
                c.weight.value_unsafe_ptr_cpu(),
                Self.W_SIZE,
                Self.IC * Self.K * Self.K,
                Self.OC * Self.K * Self.K,
            )
            INIT.init_bias(c.bias.value_unsafe_ptr_cpu(), Self.B_SIZE)
            c.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Conv2D.make[target='gpu']: ctx required")
            raise Error(
                "Conv2D: GPU path not implemented yet (see"
                " PORTING_PLAN.md Phase 5)"
            )
        return c^

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["Conv2D", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                input.ptr
            )
            self._cached_input_ptr = in_p
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output_v.ptr
            )
            var w_p = self.weight.value_unsafe_ptr_cpu()
            var b_p = self.bias.value_unsafe_ptr_cpu()
            var col_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
                Scalar[DT]
            ](Self.SPATIAL_OUT * Self.COL_SIZE)
            var w_tt = TileTensor(
                self.weight.value, row_major[Self.OC, Self.COL_SIZE](),
            )
            for b in range(BATCH):
                _im2col_one_batch[
                    Self.IC, Self.K, Self.S, Self.P,
                    Self.H, Self.W, Self.OH, Self.OW,
                ](
                    in_p + b * Self.IN_DIM_FLAT,
                    col_buf,
                )
                var col_tt = TileTensor(
                    col_buf,
                    row_major[Self.SPATIAL_OUT, Self.COL_SIZE](),
                )
                var out_b_p = out_p + b * Self.OUT_DIM_FLAT
                var out_tt = TileTensor(
                    out_b_p,
                    row_major[Self.OC, Self.SPATIAL_OUT](),
                )
                # out = W @ col.T  →  [OC, SPATIAL_OUT].
                max_matmul[transpose_b=True, target="cpu"](
                    out_tt, w_tt, col_tt, None,
                )
                # Bias broadcast across SPATIAL_OUT lanes.
                for oc in range(Self.OC):
                    var row = out_b_p + oc * Self.SPATIAL_OUT
                    var bv = b_p[oc]
                    var i = 0
                    while i + CPU_SIMD_W <= Self.SPATIAL_OUT:
                        var v = row.load[width=CPU_SIMD_W](i)
                        row.store(
                            i, v + SIMD[DT, CPU_SIMD_W](bv),
                        )
                        i += CPU_SIMD_W
                    while i < Self.SPATIAL_OUT:
                        row[i] = row[i] + bv
                        i += 1
            col_buf.free()
        else:
            raise Error("Conv2D.forward[target='gpu']: not implemented")

    # ----- Backward --------------------------------------------------------

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Conv2D", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](
            grad_inputs[0]
        )

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_input_v.ptr
            )
            var x_p = self._cached_input_ptr
            var w_p = self.weight.value_unsafe_ptr_cpu()
            var dw_p = self.weight.grad_unsafe_ptr_cpu()
            var db_p = self.bias.grad_unsafe_ptr_cpu()

            # Zero d_input — col2im is scatter-add.
            for k in range(BATCH * Self.IN_DIM_FLAT):
                gi_p[k] = Scalar[DT](0.0)

            var col_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
                Scalar[DT]
            ](Self.SPATIAL_OUT * Self.COL_SIZE)
            var d_col_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
                Scalar[DT]
            ](Self.SPATIAL_OUT * Self.COL_SIZE)
            var dw_tmp: UnsafePointer[Scalar[DT], MutAnyOrigin]
            var go_b_T_buf: UnsafePointer[Scalar[DT], MutAnyOrigin]
            comptime if (
                CompilationTarget.is_macos() and DT == DType.float32
            ):
                dw_tmp = UnsafePointer[Scalar[DT], MutAnyOrigin](
                    unsafe_from_address=0
                )
                go_b_T_buf = UnsafePointer[Scalar[DT], MutAnyOrigin](
                    unsafe_from_address=0
                )
            else:
                dw_tmp = alloc[Scalar[DT]](Self.W_SIZE)
                go_b_T_buf = alloc[Scalar[DT]](
                    Self.SPATIAL_OUT * Self.OC
                )

            var w_tt = TileTensor(
                self.weight.value, row_major[Self.OC, Self.COL_SIZE](),
            )

            for b in range(BATCH):
                # ---- 1. Rebuild col_b for this batch ------------------
                _im2col_one_batch[
                    Self.IC, Self.K, Self.S, Self.P,
                    Self.H, Self.W, Self.OH, Self.OW,
                ](
                    x_p + b * Self.IN_DIM_FLAT,
                    col_buf,
                )
                var col_tt = TileTensor(
                    col_buf,
                    row_major[Self.SPATIAL_OUT, Self.COL_SIZE](),
                )

                # ---- 2. d_out_b view + d_bias accumulate --------------
                var go_b_p = go_p + b * Self.OUT_DIM_FLAT
                var go_b_tt = TileTensor(
                    go_b_p, row_major[Self.OC, Self.SPATIAL_OUT](),
                )
                comptime if mode == "all":
                    for oc in range(Self.OC):
                        var acc: Scalar[DT] = 0.0
                        var row_off = oc * Self.SPATIAL_OUT
                        for s in range(Self.SPATIAL_OUT):
                            acc += go_b_p[row_off + s]
                        db_p[oc] += acc

                # ---- 3. d_weight += d_out_b @ col_b -------------------
                #         d_out_b is [OC, SPATIAL_OUT],
                #         col_b   is [SPATIAL_OUT, COL_SIZE],
                #         result  is [OC, COL_SIZE] (= same flat as W).
                # On Apple fp32 we use one cblas_sgemm with beta=1 (no
                # temp). Elsewhere we matmul into dw_tmp and add.
                comptime if mode == "all":
                    comptime if (
                        CompilationTarget.is_macos()
                        and DT == DType.float32
                    ):
                        var cblas = get_cblas_f32_function()
                        cblas(
                            _CBLASOrder.ROW_MAJOR,
                            _CBLASTranspose.NO_TRANSPOSE,
                            _CBLASTranspose.NO_TRANSPOSE,
                            Int32(Self.OC),
                            Int32(Self.COL_SIZE),
                            Int32(Self.SPATIAL_OUT),
                            Float32(1.0),
                            rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                                go_b_p
                            ),
                            Int32(Self.SPATIAL_OUT),
                            rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                                col_buf
                            ),
                            Int32(Self.COL_SIZE),
                            Float32(1.0),
                            rebind[UnsafePointer[Float32, MutAnyOrigin]](
                                dw_p
                            ),
                            Int32(Self.COL_SIZE),
                        )
                    else:
                        var dw_tmp_tt = TileTensor(
                            dw_tmp,
                            row_major[Self.OC, Self.COL_SIZE](),
                        )
                        max_matmul[target="cpu"](
                            dw_tmp_tt, go_b_tt, col_tt, None,
                        )
                        var i = 0
                        while i + CPU_SIMD_W <= Self.W_SIZE:
                            var dwv = dw_p.load[width=CPU_SIMD_W](i)
                            var tv = dw_tmp.load[width=CPU_SIMD_W](i)
                            dw_p.store(i, dwv + tv)
                            i += CPU_SIMD_W
                        while i < Self.W_SIZE:
                            dw_p[i] = dw_p[i] + dw_tmp[i]
                            i += 1

                # ---- 4. d_col_b = d_out_b.T @ weight ------------------
                #         d_out_b.T is [SPATIAL_OUT, OC],
                #         weight     is [OC, COL_SIZE],
                #         result     is [SPATIAL_OUT, COL_SIZE].
                # `max_matmul` does NOT support `transpose_a=True`, so on
                # Apple fp32 we drop through to cblas (which does); on
                # other targets we materialise the transpose explicitly
                # into `go_b_T_buf` and call max_matmul untransposed.
                # Mirrors `linear.mojo` grad_w's Apple-vs-other split.
                comptime if (
                    CompilationTarget.is_macos() and DT == DType.float32
                ):
                    var cblas = get_cblas_f32_function()
                    cblas(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(Self.SPATIAL_OUT),
                        Int32(Self.COL_SIZE),
                        Int32(Self.OC),
                        Float32(1.0),
                        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                            go_b_p
                        ),
                        Int32(Self.SPATIAL_OUT),
                        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                            w_p
                        ),
                        Int32(Self.COL_SIZE),
                        Float32(0.0),
                        rebind[UnsafePointer[Float32, MutAnyOrigin]](
                            d_col_buf
                        ),
                        Int32(Self.COL_SIZE),
                    )
                else:
                    # Build d_out_b.T into go_b_T_buf, then untransposed
                    # matmul. The temp is SPATIAL_OUT × OC.
                    for s in range(Self.SPATIAL_OUT):
                        for oc in range(Self.OC):
                            go_b_T_buf[s * Self.OC + oc] = go_b_p[
                                oc * Self.SPATIAL_OUT + s
                            ]
                    var go_b_T_tt = TileTensor(
                        go_b_T_buf,
                        row_major[Self.SPATIAL_OUT, Self.OC](),
                    )
                    var d_col_tt = TileTensor(
                        d_col_buf,
                        row_major[Self.SPATIAL_OUT, Self.COL_SIZE](),
                    )
                    max_matmul[target="cpu"](
                        d_col_tt, go_b_T_tt, w_tt, None,
                    )

                # ---- 5. col2im → d_input_b ----------------------------
                _col2im_one_batch[
                    Self.IC, Self.K, Self.S, Self.P,
                    Self.H, Self.W, Self.OH, Self.OW,
                ](
                    d_col_buf,
                    gi_p + b * Self.IN_DIM_FLAT,
                )

            col_buf.free()
            d_col_buf.free()
            comptime if not (
                CompilationTarget.is_macos() and DT == DType.float32
            ):
                dw_tmp.free()
                go_b_T_buf.free()
        else:
            raise Error("Conv2D.vjp[target='gpu']: not implemented")

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Conv2D", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Conv2D", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
