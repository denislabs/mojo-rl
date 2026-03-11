"""Fused Conv2D + Activation op.

FusedConv2DActivation[ic, oc, k, s, p, h, w, ACT] fuses a Conv2D with an
activation into a single forward/backward pass, eliminating one full
read+write of the output tensor per layer.

Forward:  y = act(Conv2D(x))  — activation applied in-place after bias add
Backward: grad through activation first, then Conv2D backward

Cache layout: [im2col (col_size * spatial_out) | act_cache (OUT_DIM)]
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, FusedOp, OpID
from .activation import Activation
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


struct FusedConv2DActivation[
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    in_h: Int,
    in_w: Int,
    ACT: Activation,
](FusedOp):
    """Fused y = act(Conv2D(x)) in a single operation.

    Params: W (out_channels, col_size) + bias (out_channels) — same as Conv2D
    Cache: im2col buffer + activation cache (pre-act or output per ACT.cache)
    """

    comptime out_h: Int = (Self.in_h + 2 * Self.padding - Self.kernel_size) // Self.stride + 1
    comptime out_w: Int = (Self.in_w + 2 * Self.padding - Self.kernel_size) // Self.stride + 1
    comptime col_size: Int = Self.in_channels * Self.kernel_size * Self.kernel_size
    comptime spatial_out: Int = Self.out_h * Self.out_w

    comptime OP_ID: Int = Self.ACT.FUSED_CONV_OP_ID
    comptime IN_DIM: Int = Self.in_channels * Self.in_h * Self.in_w
    comptime OUT_DIM: Int = Self.out_channels * Self.out_h * Self.out_w
    comptime PARAM_SIZE: Int = Self.out_channels * Self.col_size + Self.out_channels
    # Cache: im2col (col_size * spatial_out) + activation cache (OUT_DIM)
    comptime CONV_CACHE: Int = Self.col_size * Self.spatial_out
    comptime CACHE_SIZE: Int = Self.CONV_CACHE + Self.OUT_DIM
    comptime FUSED_COUNT: Int = 2

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval
    # =========================================================================

    @staticmethod
    fn eval[
        BATCH: Int
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        for b in range(BATCH):
            # im2col into cache[b, 0..CONV_CACHE-1]
            for oh in range(Self.out_h):
                for ow in range(Self.out_w):
                    var s = oh * Self.out_w + ow
                    for c in range(Self.in_channels):
                        for kh in range(Self.kernel_size):
                            for kw in range(Self.kernel_size):
                                var ih = oh * Self.stride - Self.padding + kh
                                var iw = ow * Self.stride - Self.padding + kw
                                var c_k = c * Self.kernel_size * Self.kernel_size + kh * Self.kernel_size + kw
                                var col_idx = c_k * Self.spatial_out + s
                                if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                                    var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                                    cache[b, col_idx] = input[b, in_idx]
                                else:
                                    cache[b, col_idx] = 0

            # W @ col + bias, then activation
            for oc in range(Self.out_channels):
                for s in range(Self.spatial_out):
                    var acc: Scalar[dtype] = 0
                    for k in range(Self.col_size):
                        var w_val = rebind[Scalar[dtype]](params[oc * Self.col_size + k])
                        var c_val = rebind[Scalar[dtype]](cache[b, k * Self.spatial_out + s])
                        acc += w_val * c_val
                    # Add bias
                    acc += rebind[Scalar[dtype]](params[Self.out_channels * Self.col_size + oc])
                    # Fused activation
                    var pre_act = acc
                    var act_out = Self.ACT.forward(pre_act)
                    var out_idx = oc * Self.spatial_out + s
                    output[b, out_idx] = act_out
                    # Cache activation state for backward
                    cache[b, Self.CONV_CACHE + out_idx] = Self.ACT.cache(pre_act, act_out)

    # =========================================================================
    # CPU vjp
    # =========================================================================

    @staticmethod
    fn vjp[
        BATCH: Int
    ](
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        comptime W_SIZE = Self.out_channels * Self.col_size

        for b in range(BATCH):
            # 1. dW += masked_grad @ col.T
            for oc in range(Self.out_channels):
                for k in range(Self.col_size):
                    var acc: Scalar[dtype] = 0
                    for s in range(Self.spatial_out):
                        var out_idx = oc * Self.spatial_out + s
                        var cache_val = rebind[Scalar[dtype]](cache[b, Self.CONV_CACHE + out_idx])
                        var go_val = rebind[Scalar[dtype]](grad_output[b, out_idx])
                        var masked_go = Self.ACT.backward(cache_val, go_val)
                        var col_val = rebind[Scalar[dtype]](cache[b, k * Self.spatial_out + s])
                        acc += masked_go * col_val
                    var cur = rebind[Scalar[dtype]](grad_params[oc * Self.col_size + k])
                    grad_params[oc * Self.col_size + k] = cur + acc

            # 2. db += sum(masked_grad, over spatial)
            for oc in range(Self.out_channels):
                var acc: Scalar[dtype] = 0
                for s in range(Self.spatial_out):
                    var out_idx = oc * Self.spatial_out + s
                    var cache_val = rebind[Scalar[dtype]](cache[b, Self.CONV_CACHE + out_idx])
                    var go_val = rebind[Scalar[dtype]](grad_output[b, out_idx])
                    acc += Self.ACT.backward(cache_val, go_val)
                var cur = rebind[Scalar[dtype]](grad_params[W_SIZE + oc])
                grad_params[W_SIZE + oc] = cur + acc

            # 3. grad_input via col2im of W.T @ masked_grad
            for i in range(Self.IN_DIM):
                grad_input[b, i] = 0

            for oh in range(Self.out_h):
                for ow in range(Self.out_w):
                    var s = oh * Self.out_w + ow
                    for c in range(Self.in_channels):
                        for kh in range(Self.kernel_size):
                            for kw in range(Self.kernel_size):
                                var ih = oh * Self.stride - Self.padding + kh
                                var iw = ow * Self.stride - Self.padding + kw
                                if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                                    var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                                    var c_k = c * Self.kernel_size * Self.kernel_size + kh * Self.kernel_size + kw
                                    var dcol_val: Scalar[dtype] = 0
                                    for oc in range(Self.out_channels):
                                        var out_idx = oc * Self.spatial_out + s
                                        var cache_val = rebind[Scalar[dtype]](cache[b, Self.CONV_CACHE + out_idx])
                                        var go_val = rebind[Scalar[dtype]](grad_output[b, out_idx])
                                        var masked_go = Self.ACT.backward(cache_val, go_val)
                                        var w_val = rebind[Scalar[dtype]](params[oc * Self.col_size + c_k])
                                        dcol_val += w_val * masked_go
                                    var cur = rebind[Scalar[dtype]](grad_input[b, in_idx])
                                    grad_input[b, in_idx] = cur + dcol_val

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    fn eval_kernel_impl[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """GPU forward: one thread per (batch, out_channel, spatial_pos)."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        var total = BATCH * Self.out_channels * Self.spatial_out
        if idx >= total:
            return

        var b = idx // (Self.out_channels * Self.spatial_out)
        var rem = idx % (Self.out_channels * Self.spatial_out)
        var oc = rem // Self.spatial_out
        var s = rem % Self.spatial_out

        var oh = s // Self.out_w
        var ow = s % Self.out_w

        var acc: Scalar[dtype] = 0
        for c in range(Self.in_channels):
            for kh in range(Self.kernel_size):
                for kw in range(Self.kernel_size):
                    var ih = oh * Self.stride - Self.padding + kh
                    var iw = ow * Self.stride - Self.padding + kw
                    var c_k = c * Self.kernel_size * Self.kernel_size + kh * Self.kernel_size + kw
                    var val: Scalar[dtype] = 0
                    if ih >= 0 and ih < Self.in_h and iw >= 0 and iw < Self.in_w:
                        var in_idx = c * Self.in_h * Self.in_w + ih * Self.in_w + iw
                        val = rebind[Scalar[dtype]](input[b, in_idx])
                    acc += rebind[Scalar[dtype]](params[oc * Self.col_size + c_k]) * val
                    # Cache im2col
                    var cache_idx = c_k * Self.spatial_out + s
                    cache[b, cache_idx] = val

        # Add bias
        acc += rebind[Scalar[dtype]](params[Self.out_channels * Self.col_size + oc])
        # Fused activation
        var pre_act = acc
        var act_out = Self.ACT.forward(pre_act)
        var out_idx = oc * Self.spatial_out + s
        output[b, out_idx] = act_out
        cache[b, Self.CONV_CACHE + out_idx] = Self.ACT.cache(pre_act, act_out)

    @always_inline
    @staticmethod
    fn backward_dx_kernel_impl[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """GPU backward for grad_input with fused activation gradient."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        var total = BATCH * Self.IN_DIM
        if idx >= total:
            return

        var b = idx // Self.IN_DIM
        var in_pos = idx % Self.IN_DIM

        var c = in_pos // (Self.in_h * Self.in_w)
        var rem = in_pos % (Self.in_h * Self.in_w)
        var ih = rem // Self.in_w
        var iw = rem % Self.in_w

        var acc: Scalar[dtype] = 0
        for kh in range(Self.kernel_size):
            for kw in range(Self.kernel_size):
                var oh_num = ih + Self.padding - kh
                var ow_num = iw + Self.padding - kw
                if oh_num >= 0 and oh_num % Self.stride == 0 and ow_num >= 0 and ow_num % Self.stride == 0:
                    var oh = oh_num // Self.stride
                    var ow = ow_num // Self.stride
                    if oh < Self.out_h and ow < Self.out_w:
                        var s = oh * Self.out_w + ow
                        var c_k = c * Self.kernel_size * Self.kernel_size + kh * Self.kernel_size + kw
                        for oc in range(Self.out_channels):
                            var out_idx = oc * Self.spatial_out + s
                            var cache_val = rebind[Scalar[dtype]](cache[b, Self.CONV_CACHE + out_idx])
                            var go_val = rebind[Scalar[dtype]](grad_output[b, out_idx])
                            var masked_go = Self.ACT.backward(cache_val, go_val)
                            acc += rebind[Scalar[dtype]](params[oc * Self.col_size + c_k]) * masked_go

        grad_input[b, in_pos] = acc

    @always_inline
    @staticmethod
    fn backward_dW_kernel_impl[
        BATCH: Int
    ](
        grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """GPU backward for dW and db with fused activation gradient."""
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        var total = Self.out_channels * Self.col_size
        if idx >= total:
            return

        var oc = idx // Self.col_size
        var k = idx % Self.col_size

        var acc: Scalar[dtype] = 0
        for b in range(BATCH):
            for s in range(Self.spatial_out):
                var out_idx = oc * Self.spatial_out + s
                var cache_val = rebind[Scalar[dtype]](cache[b, Self.CONV_CACHE + out_idx])
                var go_val = rebind[Scalar[dtype]](grad_output[b, out_idx])
                var masked_go = Self.ACT.backward(cache_val, go_val)
                var col_val = rebind[Scalar[dtype]](cache[b, k * Self.spatial_out + s])
                acc += masked_go * col_val
        grad_params[oc * Self.col_size + k] = rebind[Scalar[dtype]](grad_params[oc * Self.col_size + k]) + acc

        # db: only compute once per oc (when k == 0)
        if k == 0:
            var bias_acc: Scalar[dtype] = 0
            for b in range(BATCH):
                for s in range(Self.spatial_out):
                    var out_idx = oc * Self.spatial_out + s
                    var cache_val = rebind[Scalar[dtype]](cache[b, Self.CONV_CACHE + out_idx])
                    var go_val = rebind[Scalar[dtype]](grad_output[b, out_idx])
                    bias_acc += Self.ACT.backward(cache_val, go_val)
            var bias_idx = Self.out_channels * Self.col_size + oc
            grad_params[bias_idx] = rebind[Scalar[dtype]](grad_params[bias_idx]) + bias_acc

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    fn eval_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ) raises:
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)

        var total = BATCH * Self.out_channels * Self.spatial_out
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH](output, input, params, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            params_immut,
            cache,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn vjp_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ) raises:
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)

        # Kernel 1: grad_input
        var total_dx = BATCH * Self.IN_DIM
        var grid_dx = (total_dx + TPB - 1) // TPB

        @always_inline
        fn dx_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
        ):
            Self.backward_dx_kernel_impl[BATCH](grad_input, grad_output, params, cache)

        ctx.enqueue_function[dx_wrapper, dx_wrapper](
            grad_input,
            grad_output_immut,
            params_immut,
            cache_immut,
            grid_dim=(grid_dx,),
            block_dim=(TPB,),
        )

        # Kernel 2: dW and db
        var total_dW = Self.out_channels * Self.col_size
        var grid_dW = (total_dW + TPB - 1) // TPB

        @always_inline
        fn dW_wrapper(
            grad_params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.backward_dW_kernel_impl[BATCH](grad_params, cache, grad_output)

        ctx.enqueue_function[dW_wrapper, dW_wrapper](
            grad_params,
            cache_immut,
            grad_output_immut,
            grid_dim=(grid_dW,),
            block_dim=(TPB,),
        )
