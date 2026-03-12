"""AutoFused — automatic compile-time fusion of DiffOp chains.

User writes: AutoFused[MatMul[2,4], BiasAdd[4], ReLUOp[4], MatMul[4,1], BiasAdd[1]]
Gets: A Model-conforming struct that internally executes FusedMBR[2,4] → FusedMB[4,1]

Pattern matching (greedy, left-to-right):
  M+B+Act → FusedMatMulBiasActivation (3 ops consumed, any activation in OP_ID 10-19)
  M+B     → FusedMatMulBias (2 ops consumed)
  other   → passthrough as unfused DiffOp (1 op consumed)

Supported activations: ReLU, Tanh, Sigmoid, Mish. Adding a new activation only
requires implementing the Activation trait — no changes to AutoFused needed.

Uses Variadic.slice_types + comptime assert for recursive compile-time fusion.
IMPORTANT: slice_types calls must end with trailing comma to prevent subscript
parsing of the following line.
"""

from ..constants import dtype
from ..model.model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from .op import DiffOp, OpID
from .fused import FusedMatMulBias, FusedMatMulBiasActivation
from .fused.activation import (
    ReLUActivation,
    TanhActivation,
    SigmoidActivation,
    MishActivation,
)
from layout import LayoutTensor, Layout
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.builtin.variadics import Variadic


# =============================================================================
# Helper: check if ops[2] is a supported activation for 3-op fusion
# =============================================================================

fn _is_act(op_id: Int) -> Bool:
    return op_id >= 10 and op_id <= 19


# =============================================================================
# Recursive compile-time helpers — compute fused group sizes
# =============================================================================


fn _fused_param_size[*OPS: DiffOp]() -> Int:
    """Total PARAM_SIZE across all fused groups."""
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    comptime if N == 0:
        return 0
    elif N >= 3:
        comptime assert Variadic.size(ops) >= 3
        comptime assert Variadic.size(ops) <= Variadic.size(ops)
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value
                and _is_act(ops[2].OP_ID)):
            # 3-op activation fusion
            comptime gps = ops[0].IN_DIM * ops[0].OUT_DIM + ops[0].OUT_DIM
            comptime if N == 3:
                return gps
            else:
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=3, end=Variadic.size(ops),
                ]
                return gps + _fused_param_size[*rest]()
        elif (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value):
            # 2-op linear fusion
            comptime gps = ops[0].IN_DIM * ops[0].OUT_DIM + ops[0].OUT_DIM
            comptime if N == 2:
                return gps
            else:
                comptime assert Variadic.size(ops) >= 2
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=2, end=Variadic.size(ops),
                ]
                return gps + _fused_param_size[*rest]()
        else:
            # Unfused single op passthrough
            comptime if N == 1:
                return ops[0].PARAM_SIZE
            else:
                comptime assert Variadic.size(ops) >= 1
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=1, end=Variadic.size(ops),
                ]
                return ops[0].PARAM_SIZE + _fused_param_size[*rest]()
    elif N == 2:
        comptime assert Variadic.size(ops) >= 2
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value):
            return ops[0].IN_DIM * ops[0].OUT_DIM + ops[0].OUT_DIM
        else:
            return ops[0].PARAM_SIZE + ops[1].PARAM_SIZE
    else:  # N == 1
        return ops[0].PARAM_SIZE


fn _fused_cache_size[*OPS: DiffOp]() -> Int:
    """Total CACHE_SIZE across all fused groups."""
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    comptime if N == 0:
        return 0
    elif N >= 3:
        comptime assert Variadic.size(ops) >= 3
        comptime assert Variadic.size(ops) <= Variadic.size(ops)
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value
                and _is_act(ops[2].OP_ID)):
            # 3-op: cache = in_dim + out_dim
            comptime gcs = ops[0].IN_DIM + ops[0].OUT_DIM
            comptime if N == 3:
                return gcs
            else:
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=3, end=Variadic.size(ops),
                ]
                return gcs + _fused_cache_size[*rest]()
        elif (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value):
            # 2-op: cache = in_dim (FusedMatMulBias)
            comptime gcs = ops[0].IN_DIM
            comptime if N == 2:
                return gcs
            else:
                comptime assert Variadic.size(ops) >= 2
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=2, end=Variadic.size(ops),
                ]
                return gcs + _fused_cache_size[*rest]()
        else:
            # Unfused: use op's own cache size
            comptime if N == 1:
                return ops[0].CACHE_SIZE
            else:
                comptime assert Variadic.size(ops) >= 1
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=1, end=Variadic.size(ops),
                ]
                return ops[0].CACHE_SIZE + _fused_cache_size[*rest]()
    elif N == 2:
        comptime assert Variadic.size(ops) >= 2
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value):
            return ops[0].IN_DIM
        else:
            return ops[0].CACHE_SIZE + ops[1].CACHE_SIZE
    else:  # N == 1
        return ops[0].CACHE_SIZE


fn _fused_inter_size[*OPS: DiffOp]() -> Int:
    """Total intermediate buffer size (per sample) across all group boundaries.

    Each group (except the last) produces an intermediate of size GROUP_OUT_DIM.
    """
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    comptime if N == 0:
        return 0
    elif N >= 3:
        comptime assert Variadic.size(ops) >= 3
        comptime assert Variadic.size(ops) <= Variadic.size(ops)
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value
                and _is_act(ops[2].OP_ID)):
            # 3-op: group out = ops[0].OUT_DIM
            comptime if N == 3:
                return 0  # Last group, no inter needed
            else:
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=3, end=Variadic.size(ops),
                ]
                return ops[0].OUT_DIM + _fused_inter_size[*rest]()
        elif (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value):
            comptime if N == 2:
                return 0
            else:
                comptime assert Variadic.size(ops) >= 2
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=2, end=Variadic.size(ops),
                ]
                return ops[0].OUT_DIM + _fused_inter_size[*rest]()
        else:
            # Unfused single op
            comptime if N == 1:
                return 0
            else:
                comptime assert Variadic.size(ops) >= 1
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=1, end=Variadic.size(ops),
                ]
                return ops[0].OUT_DIM + _fused_inter_size[*rest]()
    elif N == 2:
        comptime assert Variadic.size(ops) >= 2
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value):
            return 0  # Single M+B group, last group
        else:
            # Two unfused ops: first needs inter
            return ops[0].OUT_DIM
    else:  # N == 1
        return 0


# =============================================================================
# Recursive forward — CPU (with cache)
# =============================================================================


fn _auto_fused_forward[BATCH: Int, *OPS: DiffOp](
    in_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    final_out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cache_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    inter_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    param_off: Int,
    cache_off: Int,
    inter_off: Int,
):
    """Recursive forward pass. Each call handles one fused group, then recurses."""
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    comptime if N == 0:
        pass
    elif N >= 3:
        comptime assert Variadic.size(ops) >= 3
        comptime assert Variadic.size(ops) <= Variadic.size(ops)
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value
                and _is_act(ops[2].OP_ID)):
            # --- 3-op activation fusion ---
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            # Cache: FusedMB{R,T,S} = in_dim + out_dim
            comptime FCS = G_IN + G_OUT

            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](
                params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](
                cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](
                in_ptr)

            # Dispatch to FusedMatMulBiasActivation with the right Activation
            comptime if N == 3:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
                comptime if ops[2].OP_ID == OpID.RELU._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, ReLUActivation].eval[BATCH](in_v, out_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.TANH._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, TanhActivation].eval[BATCH](in_v, out_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.SIGMOID._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, SigmoidActivation].eval[BATCH](in_v, out_v, p_v, c_v)
                else:
                    FusedMatMulBiasActivation[G_IN, G_OUT, MishActivation].eval[BATCH](in_v, out_v, p_v, c_v)
            else:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](inter_ptr + BATCH * inter_off)
                comptime if ops[2].OP_ID == OpID.RELU._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, ReLUActivation].eval[BATCH](in_v, out_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.TANH._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, TanhActivation].eval[BATCH](in_v, out_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.SIGMOID._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, SigmoidActivation].eval[BATCH](in_v, out_v, p_v, c_v)
                else:
                    FusedMatMulBiasActivation[G_IN, G_OUT, MishActivation].eval[BATCH](in_v, out_v, p_v, c_v)
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=3, end=Variadic.size(ops),
                ]
                _auto_fused_forward[BATCH, *rest](
                    inter_ptr + BATCH * inter_off,
                    final_out_ptr, params_ptr, cache_ptr, inter_ptr,
                    param_off + FPS, cache_off + FCS, inter_off + G_OUT,
                )
        elif (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value):
            # --- 2-op linear fusion ---
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN

            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](
                params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](
                cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](
                in_ptr)

            comptime if N == 2:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
                FusedMatMulBias[G_IN, G_OUT].eval[BATCH](in_v, out_v, p_v, c_v)
            else:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](inter_ptr + BATCH * inter_off)
                FusedMatMulBias[G_IN, G_OUT].eval[BATCH](in_v, out_v, p_v, c_v)
                comptime assert Variadic.size(ops) >= 2
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=2, end=Variadic.size(ops),
                ]
                _auto_fused_forward[BATCH, *rest](
                    inter_ptr + BATCH * inter_off,
                    final_out_ptr, params_ptr, cache_ptr, inter_ptr,
                    param_off + FPS, cache_off + FCS, inter_off + G_OUT,
                )
        else:
            # --- Unfused single op passthrough ---
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime OPS = ops[0].PARAM_SIZE
            comptime OCS = ops[0].CACHE_SIZE

            var p_v = LayoutTensor[dtype, Layout.row_major(OPS), MutAnyOrigin](
                params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, OCS), MutAnyOrigin](
                cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](
                in_ptr)

            comptime if N == 1:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
                ops[0].eval[BATCH](in_v, out_v, p_v, c_v)
            else:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](inter_ptr + BATCH * inter_off)
                ops[0].eval[BATCH](in_v, out_v, p_v, c_v)
                comptime assert Variadic.size(ops) >= 1
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=1, end=Variadic.size(ops),
                ]
                _auto_fused_forward[BATCH, *rest](
                    inter_ptr + BATCH * inter_off,
                    final_out_ptr, params_ptr, cache_ptr, inter_ptr,
                    param_off + OPS, cache_off + OCS, inter_off + G_OUT,
                )
    elif N == 2:
        comptime assert Variadic.size(ops) >= 2
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN
            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)
            var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
            FusedMatMulBias[G_IN, G_OUT].eval[BATCH](in_v, out_v, p_v, c_v)
        else:
            # Two unfused ops: op0 → inter, op1 → output
            comptime G0_IN = ops[0].IN_DIM
            comptime G0_OUT = ops[0].OUT_DIM
            comptime G1_IN = ops[1].IN_DIM
            comptime G1_OUT = ops[1].OUT_DIM
            var p0 = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off)
            var c0 = LayoutTensor[dtype, Layout.row_major(BATCH, ops[0].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in0 = LayoutTensor[dtype, Layout.row_major(BATCH, G0_IN), MutAnyOrigin](in_ptr)
            var out0 = LayoutTensor[dtype, Layout.row_major(BATCH, G0_OUT), MutAnyOrigin](inter_ptr + BATCH * inter_off)
            ops[0].eval[BATCH](in0, out0, p0, c0)
            var p1 = LayoutTensor[dtype, Layout.row_major(ops[1].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off + ops[0].PARAM_SIZE)
            var c1 = LayoutTensor[dtype, Layout.row_major(BATCH, ops[1].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * (cache_off + ops[0].CACHE_SIZE))
            var in1 = LayoutTensor[dtype, Layout.row_major(BATCH, G1_IN), MutAnyOrigin](inter_ptr + BATCH * inter_off)
            var out1 = LayoutTensor[dtype, Layout.row_major(BATCH, G1_OUT), MutAnyOrigin](final_out_ptr)
            ops[1].eval[BATCH](in1, out1, p1, c1)
    else:  # N == 1
        comptime G_IN = ops[0].IN_DIM
        comptime G_OUT = ops[0].OUT_DIM
        var p_v = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off)
        var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, ops[0].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * cache_off)
        var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)
        var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
        ops[0].eval[BATCH](in_v, out_v, p_v, c_v)


# =============================================================================
# Recursive backward — CPU
#
# Strategy: recurse FIRST (to reach end of chain), then apply VJP on return.
# This naturally reverses execution order: last group's VJP runs first.
#
# Each level receives:
#   grad_in_ptr: where THIS group should write its grad_input
#   grad_chain_out_ptr: chain's grad_output (constant through recursion)
#   gi_ptr: gradient intermediate buffer base pointer
#   The rest: params, cache, grads base pointers + cumulative offsets
# =============================================================================


fn _auto_fused_backward[BATCH: Int, *OPS: DiffOp](
    grad_in_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_chain_out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cache_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grads_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    gi_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    param_off: Int,
    cache_off: Int,
    inter_off: Int,
):
    """Recursive backward. Recurses first (reaching last group), then VJPs
    on return — naturally reversing execution order."""
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    comptime if N == 0:
        pass
    elif N >= 3:
        comptime assert Variadic.size(ops) >= 3
        comptime assert Variadic.size(ops) <= Variadic.size(ops)
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value
                and _is_act(ops[2].OP_ID)):
            # --- 3-op activation fusion ---
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN + G_OUT

            comptime if N == 3:
                # Last group: grad_output = chain's grad_output
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](grad_chain_out_ptr)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](grads_ptr + param_off)
                comptime if ops[2].OP_ID == OpID.RELU._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, ReLUActivation].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
                elif ops[2].OP_ID == OpID.TANH._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, TanhActivation].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
                elif ops[2].OP_ID == OpID.SIGMOID._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, SigmoidActivation].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
                else:
                    FusedMatMulBiasActivation[G_IN, G_OUT, MishActivation].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
            else:
                # Not last: recurse first, then VJP
                var out_inter = gi_ptr + BATCH * inter_off
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=3, end=Variadic.size(ops),
                ]
                _auto_fused_backward[BATCH, *rest](
                    out_inter,
                    grad_chain_out_ptr, params_ptr, cache_ptr, grads_ptr, gi_ptr,
                    param_off + FPS, cache_off + FCS, inter_off + G_OUT,
                )
                # Now apply this group's VJP
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](out_inter)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](grads_ptr + param_off)
                comptime if ops[2].OP_ID == OpID.RELU._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, ReLUActivation].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
                elif ops[2].OP_ID == OpID.TANH._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, TanhActivation].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
                elif ops[2].OP_ID == OpID.SIGMOID._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, SigmoidActivation].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
                else:
                    FusedMatMulBiasActivation[G_IN, G_OUT, MishActivation].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
        elif (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value):
            # --- 2-op linear fusion ---
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN

            comptime if N == 2:
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](grad_chain_out_ptr)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](grads_ptr + param_off)
                FusedMatMulBias[G_IN, G_OUT].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
            else:
                var out_inter = gi_ptr + BATCH * inter_off
                comptime assert Variadic.size(ops) >= 2
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=2, end=Variadic.size(ops),
                ]
                _auto_fused_backward[BATCH, *rest](
                    out_inter,
                    grad_chain_out_ptr, params_ptr, cache_ptr, grads_ptr, gi_ptr,
                    param_off + FPS, cache_off + FCS, inter_off + G_OUT,
                )
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](out_inter)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](grads_ptr + param_off)
                FusedMatMulBias[G_IN, G_OUT].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
        else:
            # --- Unfused single op passthrough ---
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime OPS_ = ops[0].PARAM_SIZE
            comptime OCS = ops[0].CACHE_SIZE

            comptime if N == 1:
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](grad_chain_out_ptr)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(OPS_), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, OCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(OPS_), MutAnyOrigin](grads_ptr + param_off)
                ops[0].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
            else:
                var out_inter = gi_ptr + BATCH * inter_off
                comptime assert Variadic.size(ops) >= 1
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=1, end=Variadic.size(ops),
                ]
                _auto_fused_backward[BATCH, *rest](
                    out_inter,
                    grad_chain_out_ptr, params_ptr, cache_ptr, grads_ptr, gi_ptr,
                    param_off + OPS_, cache_off + OCS, inter_off + G_OUT,
                )
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](out_inter)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(OPS_), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, OCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(OPS_), MutAnyOrigin](grads_ptr + param_off)
                ops[0].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
    elif N == 2:
        comptime assert Variadic.size(ops) >= 2
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN
            var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](grad_chain_out_ptr)
            var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var g_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](grads_ptr + param_off)
            FusedMatMulBias[G_IN, G_OUT].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)
        else:
            # Two unfused ops — reverse order: op1 first, then op0
            comptime G0_IN = ops[0].IN_DIM
            comptime G0_OUT = ops[0].OUT_DIM
            comptime G1_IN = ops[1].IN_DIM
            comptime G1_OUT = ops[1].OUT_DIM
            # Op1 VJP (last op)
            var go1 = LayoutTensor[dtype, Layout.row_major(BATCH, G1_OUT), MutAnyOrigin](grad_chain_out_ptr)
            var gi1 = LayoutTensor[dtype, Layout.row_major(BATCH, G1_IN), MutAnyOrigin](gi_ptr + BATCH * inter_off)
            var p1 = LayoutTensor[dtype, Layout.row_major(ops[1].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off + ops[0].PARAM_SIZE)
            var c1 = LayoutTensor[dtype, Layout.row_major(BATCH, ops[1].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * (cache_off + ops[0].CACHE_SIZE))
            var g1 = LayoutTensor[dtype, Layout.row_major(ops[1].PARAM_SIZE), MutAnyOrigin](grads_ptr + param_off + ops[0].PARAM_SIZE)
            ops[1].vjp[BATCH](go1, gi1, p1, c1, g1)
            # Op0 VJP (first op)
            var go0 = LayoutTensor[dtype, Layout.row_major(BATCH, G0_OUT), MutAnyOrigin](gi_ptr + BATCH * inter_off)
            var gi0 = LayoutTensor[dtype, Layout.row_major(BATCH, G0_IN), MutAnyOrigin](grad_in_ptr)
            var p0 = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off)
            var c0 = LayoutTensor[dtype, Layout.row_major(BATCH, ops[0].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var g0 = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](grads_ptr + param_off)
            ops[0].vjp[BATCH](go0, gi0, p0, c0, g0)
    else:  # N == 1
        comptime G_IN = ops[0].IN_DIM
        comptime G_OUT = ops[0].OUT_DIM
        var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](grad_chain_out_ptr)
        var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
        var p_v = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off)
        var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, ops[0].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * cache_off)
        var g_v = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](grads_ptr + param_off)
        ops[0].vjp[BATCH](go_v, gi_v, p_v, c_v, g_v)


# =============================================================================
# Recursive GPU forward
# =============================================================================


fn _auto_fused_forward_gpu[BATCH: Int, *OPS: DiffOp](
    ctx: DeviceContext,
    in_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    final_out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cache_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    param_off: Int,
    cache_off: Int,
    inter_off: Int,
) raises:
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    comptime if N == 0:
        pass
    elif N >= 3:
        comptime assert Variadic.size(ops) >= 3
        comptime assert Variadic.size(ops) <= Variadic.size(ops)
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value
                and _is_act(ops[2].OP_ID)):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN + G_OUT
            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)

            comptime if N == 3:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
                comptime if ops[2].OP_ID == OpID.RELU._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, ReLUActivation].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.TANH._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, TanhActivation].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.SIGMOID._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, SigmoidActivation].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
                else:
                    FusedMatMulBiasActivation[G_IN, G_OUT, MishActivation].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
            else:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](ws_ptr + BATCH * inter_off)
                comptime if ops[2].OP_ID == OpID.RELU._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, ReLUActivation].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.TANH._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, TanhActivation].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.SIGMOID._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, SigmoidActivation].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
                else:
                    FusedMatMulBiasActivation[G_IN, G_OUT, MishActivation].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
                comptime rest = Variadic.slice_types[element_types=ops, start=3, end=Variadic.size(ops)]
                _auto_fused_forward_gpu[BATCH, *rest](ctx, ws_ptr + BATCH * inter_off, final_out_ptr, params_ptr, cache_ptr, ws_ptr, param_off + FPS, cache_off + FCS, inter_off + G_OUT)
        elif (ops[0].OP_ID == OpID.MATMUL._value and ops[1].OP_ID == OpID.BIAS_ADD._value):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN
            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)
            comptime if N == 2:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
                FusedMatMulBias[G_IN, G_OUT].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
            else:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](ws_ptr + BATCH * inter_off)
                FusedMatMulBias[G_IN, G_OUT].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
                comptime assert Variadic.size(ops) >= 2
                comptime rest = Variadic.slice_types[element_types=ops, start=2, end=Variadic.size(ops)]
                _auto_fused_forward_gpu[BATCH, *rest](ctx, ws_ptr + BATCH * inter_off, final_out_ptr, params_ptr, cache_ptr, ws_ptr, param_off + FPS, cache_off + FCS, inter_off + G_OUT)
        else:
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime OPS_ = ops[0].PARAM_SIZE
            comptime OCS = ops[0].CACHE_SIZE
            var p_v = LayoutTensor[dtype, Layout.row_major(OPS_), MutAnyOrigin](params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, OCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)
            comptime if N == 1:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
                ops[0].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
            else:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](ws_ptr + BATCH * inter_off)
                ops[0].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
                comptime assert Variadic.size(ops) >= 1
                comptime rest = Variadic.slice_types[element_types=ops, start=1, end=Variadic.size(ops)]
                _auto_fused_forward_gpu[BATCH, *rest](ctx, ws_ptr + BATCH * inter_off, final_out_ptr, params_ptr, cache_ptr, ws_ptr, param_off + OPS_, cache_off + OCS, inter_off + G_OUT)
    elif N == 2:
        comptime assert Variadic.size(ops) >= 2
        comptime if (ops[0].OP_ID == OpID.MATMUL._value and ops[1].OP_ID == OpID.BIAS_ADD._value):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN
            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)
            var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
            FusedMatMulBias[G_IN, G_OUT].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
        else:
            comptime G0_IN = ops[0].IN_DIM
            comptime G0_OUT = ops[0].OUT_DIM
            comptime G1_IN = ops[1].IN_DIM
            comptime G1_OUT = ops[1].OUT_DIM
            var p0 = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off)
            var c0 = LayoutTensor[dtype, Layout.row_major(BATCH, ops[0].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in0 = LayoutTensor[dtype, Layout.row_major(BATCH, G0_IN), MutAnyOrigin](in_ptr)
            var out0 = LayoutTensor[dtype, Layout.row_major(BATCH, G0_OUT), MutAnyOrigin](ws_ptr + BATCH * inter_off)
            ops[0].eval_gpu[BATCH](ctx, out0, in0, p0, c0)
            var p1 = LayoutTensor[dtype, Layout.row_major(ops[1].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off + ops[0].PARAM_SIZE)
            var c1 = LayoutTensor[dtype, Layout.row_major(BATCH, ops[1].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * (cache_off + ops[0].CACHE_SIZE))
            var in1 = LayoutTensor[dtype, Layout.row_major(BATCH, G1_IN), MutAnyOrigin](ws_ptr + BATCH * inter_off)
            var out1 = LayoutTensor[dtype, Layout.row_major(BATCH, G1_OUT), MutAnyOrigin](final_out_ptr)
            ops[1].eval_gpu[BATCH](ctx, out1, in1, p1, c1)
    else:  # N == 1
        comptime G_IN = ops[0].IN_DIM
        comptime G_OUT = ops[0].OUT_DIM
        var p_v = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off)
        var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, ops[0].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * cache_off)
        var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)
        var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
        ops[0].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)


# =============================================================================
# Recursive GPU forward — on DeviceStream
# =============================================================================


fn _auto_fused_forward_gpu_on_stream[BATCH: Int, *OPS: DiffOp](
    ctx: DeviceContext,
    stream: DeviceStream,
    in_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    final_out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cache_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    param_off: Int,
    cache_off: Int,
    inter_off: Int,
) raises:
    """Same as _auto_fused_forward_gpu but enqueues on a DeviceStream."""
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    comptime if N == 0:
        pass
    elif N >= 3:
        comptime assert Variadic.size(ops) >= 3
        comptime assert Variadic.size(ops) <= Variadic.size(ops)
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value
                and _is_act(ops[2].OP_ID)):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN + G_OUT
            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)

            comptime if N == 3:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
                comptime if ops[2].OP_ID == OpID.RELU._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, ReLUActivation].eval_gpu_on_stream[BATCH](ctx, stream, out_v, in_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.TANH._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, TanhActivation].eval_gpu_on_stream[BATCH](ctx, stream, out_v, in_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.SIGMOID._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, SigmoidActivation].eval_gpu_on_stream[BATCH](ctx, stream, out_v, in_v, p_v, c_v)
                else:
                    FusedMatMulBiasActivation[G_IN, G_OUT, MishActivation].eval_gpu_on_stream[BATCH](ctx, stream, out_v, in_v, p_v, c_v)
            else:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](ws_ptr + BATCH * inter_off)
                comptime if ops[2].OP_ID == OpID.RELU._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, ReLUActivation].eval_gpu_on_stream[BATCH](ctx, stream, out_v, in_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.TANH._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, TanhActivation].eval_gpu_on_stream[BATCH](ctx, stream, out_v, in_v, p_v, c_v)
                elif ops[2].OP_ID == OpID.SIGMOID._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, SigmoidActivation].eval_gpu_on_stream[BATCH](ctx, stream, out_v, in_v, p_v, c_v)
                else:
                    FusedMatMulBiasActivation[G_IN, G_OUT, MishActivation].eval_gpu_on_stream[BATCH](ctx, stream, out_v, in_v, p_v, c_v)
                comptime rest = Variadic.slice_types[element_types=ops, start=3, end=Variadic.size(ops)]
                _auto_fused_forward_gpu_on_stream[BATCH, *rest](ctx, stream, ws_ptr + BATCH * inter_off, final_out_ptr, params_ptr, cache_ptr, ws_ptr, param_off + FPS, cache_off + FCS, inter_off + G_OUT)
        elif (ops[0].OP_ID == OpID.MATMUL._value and ops[1].OP_ID == OpID.BIAS_ADD._value):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN
            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)
            comptime if N == 2:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
                FusedMatMulBias[G_IN, G_OUT].eval_gpu_on_stream[BATCH](ctx, stream, out_v, in_v, p_v, c_v)
            else:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](ws_ptr + BATCH * inter_off)
                FusedMatMulBias[G_IN, G_OUT].eval_gpu_on_stream[BATCH](ctx, stream, out_v, in_v, p_v, c_v)
                comptime assert Variadic.size(ops) >= 2
                comptime rest = Variadic.slice_types[element_types=ops, start=2, end=Variadic.size(ops)]
                _auto_fused_forward_gpu_on_stream[BATCH, *rest](ctx, stream, ws_ptr + BATCH * inter_off, final_out_ptr, params_ptr, cache_ptr, ws_ptr, param_off + FPS, cache_off + FCS, inter_off + G_OUT)
        else:
            # Generic fallback: use ctx.eval_gpu (default stream) for non-fusable ops
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime OPS_ = ops[0].PARAM_SIZE
            comptime OCS = ops[0].CACHE_SIZE
            var p_v = LayoutTensor[dtype, Layout.row_major(OPS_), MutAnyOrigin](params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, OCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)
            comptime if N == 1:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
                ops[0].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
            else:
                var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](ws_ptr + BATCH * inter_off)
                ops[0].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)
                comptime assert Variadic.size(ops) >= 1
                comptime rest = Variadic.slice_types[element_types=ops, start=1, end=Variadic.size(ops)]
                _auto_fused_forward_gpu_on_stream[BATCH, *rest](ctx, stream, ws_ptr + BATCH * inter_off, final_out_ptr, params_ptr, cache_ptr, ws_ptr, param_off + OPS_, cache_off + OCS, inter_off + G_OUT)
    elif N == 2:
        comptime assert Variadic.size(ops) >= 2
        comptime if (ops[0].OP_ID == OpID.MATMUL._value and ops[1].OP_ID == OpID.BIAS_ADD._value):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN
            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)
            var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
            FusedMatMulBias[G_IN, G_OUT].eval_gpu_on_stream[BATCH](ctx, stream, out_v, in_v, p_v, c_v)
        else:
            # Generic fallback: use ctx.eval_gpu (default stream) for non-fusable ops
            comptime G0_IN = ops[0].IN_DIM
            comptime G0_OUT = ops[0].OUT_DIM
            comptime G1_IN = ops[1].IN_DIM
            comptime G1_OUT = ops[1].OUT_DIM
            var p0 = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off)
            var c0 = LayoutTensor[dtype, Layout.row_major(BATCH, ops[0].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var in0 = LayoutTensor[dtype, Layout.row_major(BATCH, G0_IN), MutAnyOrigin](in_ptr)
            var out0 = LayoutTensor[dtype, Layout.row_major(BATCH, G0_OUT), MutAnyOrigin](ws_ptr + BATCH * inter_off)
            ops[0].eval_gpu[BATCH](ctx, out0, in0, p0, c0)
            var p1 = LayoutTensor[dtype, Layout.row_major(ops[1].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off + ops[0].PARAM_SIZE)
            var c1 = LayoutTensor[dtype, Layout.row_major(BATCH, ops[1].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * (cache_off + ops[0].CACHE_SIZE))
            var in1 = LayoutTensor[dtype, Layout.row_major(BATCH, G1_IN), MutAnyOrigin](ws_ptr + BATCH * inter_off)
            var out1 = LayoutTensor[dtype, Layout.row_major(BATCH, G1_OUT), MutAnyOrigin](final_out_ptr)
            ops[1].eval_gpu[BATCH](ctx, out1, in1, p1, c1)
    else:  # N == 1
        # Generic fallback: use ctx.eval_gpu (default stream) for non-fusable ops
        comptime G_IN = ops[0].IN_DIM
        comptime G_OUT = ops[0].OUT_DIM
        var p_v = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off)
        var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, ops[0].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * cache_off)
        var in_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](in_ptr)
        var out_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](final_out_ptr)
        ops[0].eval_gpu[BATCH](ctx, out_v, in_v, p_v, c_v)


# =============================================================================
# Recursive GPU backward
# =============================================================================


fn _auto_fused_backward_gpu[BATCH: Int, *OPS: DiffOp](
    ctx: DeviceContext,
    grad_in_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_chain_out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cache_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grads_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    gi_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    param_off: Int,
    cache_off: Int,
    inter_off: Int,
) raises:
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    comptime if N == 0:
        pass
    elif N >= 3:
        comptime assert Variadic.size(ops) >= 3
        comptime assert Variadic.size(ops) <= Variadic.size(ops)
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value
                and _is_act(ops[2].OP_ID)):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN + G_OUT

            comptime if N == 3:
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](grad_chain_out_ptr)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](grads_ptr + param_off)
                comptime if ops[2].OP_ID == OpID.RELU._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, ReLUActivation].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
                elif ops[2].OP_ID == OpID.TANH._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, TanhActivation].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
                elif ops[2].OP_ID == OpID.SIGMOID._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, SigmoidActivation].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
                else:
                    FusedMatMulBiasActivation[G_IN, G_OUT, MishActivation].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
            else:
                var out_inter = gi_ptr + BATCH * inter_off
                comptime rest = Variadic.slice_types[element_types=ops, start=3, end=Variadic.size(ops)]
                _auto_fused_backward_gpu[BATCH, *rest](ctx, out_inter, grad_chain_out_ptr, params_ptr, cache_ptr, grads_ptr, gi_ptr, param_off + FPS, cache_off + FCS, inter_off + G_OUT)
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](out_inter)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](grads_ptr + param_off)
                comptime if ops[2].OP_ID == OpID.RELU._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, ReLUActivation].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
                elif ops[2].OP_ID == OpID.TANH._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, TanhActivation].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
                elif ops[2].OP_ID == OpID.SIGMOID._value:
                    FusedMatMulBiasActivation[G_IN, G_OUT, SigmoidActivation].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
                else:
                    FusedMatMulBiasActivation[G_IN, G_OUT, MishActivation].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
        elif (ops[0].OP_ID == OpID.MATMUL._value and ops[1].OP_ID == OpID.BIAS_ADD._value):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN
            comptime if N == 2:
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](grad_chain_out_ptr)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](grads_ptr + param_off)
                FusedMatMulBias[G_IN, G_OUT].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
            else:
                var out_inter = gi_ptr + BATCH * inter_off
                comptime assert Variadic.size(ops) >= 2
                comptime rest = Variadic.slice_types[element_types=ops, start=2, end=Variadic.size(ops)]
                _auto_fused_backward_gpu[BATCH, *rest](ctx, out_inter, grad_chain_out_ptr, params_ptr, cache_ptr, grads_ptr, gi_ptr, param_off + FPS, cache_off + FCS, inter_off + G_OUT)
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](out_inter)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](grads_ptr + param_off)
                FusedMatMulBias[G_IN, G_OUT].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
        else:
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime OPS_ = ops[0].PARAM_SIZE
            comptime OCS = ops[0].CACHE_SIZE
            comptime if N == 1:
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](grad_chain_out_ptr)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(OPS_), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, OCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(OPS_), MutAnyOrigin](grads_ptr + param_off)
                ops[0].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
            else:
                var out_inter = gi_ptr + BATCH * inter_off
                comptime assert Variadic.size(ops) >= 1
                comptime rest = Variadic.slice_types[element_types=ops, start=1, end=Variadic.size(ops)]
                _auto_fused_backward_gpu[BATCH, *rest](ctx, out_inter, grad_chain_out_ptr, params_ptr, cache_ptr, grads_ptr, gi_ptr, param_off + OPS_, cache_off + OCS, inter_off + G_OUT)
                var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](out_inter)
                var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
                var p_v = LayoutTensor[dtype, Layout.row_major(OPS_), MutAnyOrigin](params_ptr + param_off)
                var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, OCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
                var g_v = LayoutTensor[dtype, Layout.row_major(OPS_), MutAnyOrigin](grads_ptr + param_off)
                ops[0].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
    elif N == 2:
        comptime assert Variadic.size(ops) >= 2
        comptime if (ops[0].OP_ID == OpID.MATMUL._value and ops[1].OP_ID == OpID.BIAS_ADD._value):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN
            var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](grad_chain_out_ptr)
            var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](params_ptr + param_off)
            var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var g_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](grads_ptr + param_off)
            FusedMatMulBias[G_IN, G_OUT].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)
        else:
            comptime G0_IN = ops[0].IN_DIM
            comptime G0_OUT = ops[0].OUT_DIM
            comptime G1_IN = ops[1].IN_DIM
            comptime G1_OUT = ops[1].OUT_DIM
            var go1 = LayoutTensor[dtype, Layout.row_major(BATCH, G1_OUT), MutAnyOrigin](grad_chain_out_ptr)
            var gi1 = LayoutTensor[dtype, Layout.row_major(BATCH, G1_IN), MutAnyOrigin](gi_ptr + BATCH * inter_off)
            var p1 = LayoutTensor[dtype, Layout.row_major(ops[1].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off + ops[0].PARAM_SIZE)
            var c1 = LayoutTensor[dtype, Layout.row_major(BATCH, ops[1].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * (cache_off + ops[0].CACHE_SIZE))
            var g1 = LayoutTensor[dtype, Layout.row_major(ops[1].PARAM_SIZE), MutAnyOrigin](grads_ptr + param_off + ops[0].PARAM_SIZE)
            ops[1].vjp_gpu[BATCH](ctx, go1, gi1, p1, c1, g1)
            var go0 = LayoutTensor[dtype, Layout.row_major(BATCH, G0_OUT), MutAnyOrigin](gi_ptr + BATCH * inter_off)
            var gi0 = LayoutTensor[dtype, Layout.row_major(BATCH, G0_IN), MutAnyOrigin](grad_in_ptr)
            var p0 = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off)
            var c0 = LayoutTensor[dtype, Layout.row_major(BATCH, ops[0].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * cache_off)
            var g0 = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](grads_ptr + param_off)
            ops[0].vjp_gpu[BATCH](ctx, go0, gi0, p0, c0, g0)
    else:
        comptime G_IN = ops[0].IN_DIM
        comptime G_OUT = ops[0].OUT_DIM
        var go_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin](grad_chain_out_ptr)
        var gi_v = LayoutTensor[dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin](grad_in_ptr)
        var p_v = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](params_ptr + param_off)
        var c_v = LayoutTensor[dtype, Layout.row_major(BATCH, ops[0].CACHE_SIZE), MutAnyOrigin](cache_ptr + BATCH * cache_off)
        var g_v = LayoutTensor[dtype, Layout.row_major(ops[0].PARAM_SIZE), MutAnyOrigin](grads_ptr + param_off)
        ops[0].vjp_gpu[BATCH](ctx, go_v, gi_v, p_v, c_v, g_v)


# =============================================================================
# AutoFused struct — Model conformance
# =============================================================================


@fieldwise_init
struct AutoFused[*OPS: DiffOp](Model):
    """Automatically fuses a DiffOp chain into optimized fused groups.

    Pattern matching (greedy, left-to-right):
      M+B+Act → FusedMatMulBiasActivation (ReLU, Tanh, Sigmoid, Mish)
      M+B     → FusedMatMulBias
      other   → passthrough

    Usage:
        comptime MyModel = AutoFused[
            MatMul[2,4], BiasAdd[4], ReLUOp[4],
            MatMul[4,1], BiasAdd[1],
        ]
        # Internally executes: FusedMBR[2,4] → FusedMB[4,1]
    """

    comptime op_types = Variadic.types[T=DiffOp, *Self.OPS]
    comptime N = Variadic.size(Self.op_types)

    comptime IN_DIM: Int = Self.op_types[0].IN_DIM
    comptime OUT_DIM: Int = Self.op_types[Self.N - 1].OUT_DIM

    comptime PARAM_SIZE: Int = _fused_param_size[*Self.OPS]()
    comptime CACHE_SIZE: Int = _fused_cache_size[*Self.OPS]()
    comptime INTER_SIZE_PER_SAMPLE: Int = _fused_inter_size[*Self.OPS]()
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.INTER_SIZE_PER_SAMPLE + Self.CACHE_SIZE

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    fn _param_offset_raw[idx: Int]() -> Int:
        var total = 0

        comptime for j in range(idx):
            total += Self.op_types[j].PARAM_SIZE
        return total

    @staticmethod
    fn initialize_params[INIT: Initializer](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize each DiffOp's params with its own fan dimensions."""
        comptime for i in range(Self.N):
            comptime if Self.op_types[i].PARAM_SIZE > 0:
                var op_params = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.op_types[i].PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset_raw[i]())
                INIT.init[
                    Self.op_types[i].PARAM_SIZE,
                    Self.op_types[i].IN_DIM,
                    Self.op_types[i].OUT_DIM,
                ](op_params)

    # =========================================================================
    # CPU Forward (with cache)
    # =========================================================================

    @staticmethod
    fn forward[
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
        var inter_size = BATCH * Self.WORKSPACE_SIZE_PER_SAMPLE
        var inter_storage = List[Scalar[dtype]](capacity=inter_size if inter_size > 0 else 1)
        for _ in range(inter_size if inter_size > 0 else 1):
            inter_storage.append(0)

        _auto_fused_forward[BATCH, *Self.OPS](
            input.ptr,
            output.ptr,
            params.ptr,
            cache.ptr,
            inter_storage.unsafe_ptr(),
            0, 0, 0,
        )

    # =========================================================================
    # CPU Forward (no cache — inference)
    # =========================================================================

    @staticmethod
    fn forward[
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
    ):
        var cap = BATCH * Self.CACHE_SIZE if Self.CACHE_SIZE > 0 else 1
        var dummy_cache = List[Scalar[dtype]](capacity=cap)
        for _ in range(cap):
            dummy_cache.append(0)
        var c = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.CACHE_SIZE),
            MutAnyOrigin,
        ](dummy_cache.unsafe_ptr())
        Self.forward[BATCH](input, output, params, c)

    # =========================================================================
    # CPU Backward
    # =========================================================================

    @staticmethod
    fn backward[
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
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        var gi_size = BATCH * Self.WORKSPACE_SIZE_PER_SAMPLE
        var gi_storage = List[Scalar[dtype]](capacity=gi_size if gi_size > 0 else 1)
        for _ in range(gi_size if gi_size > 0 else 1):
            gi_storage.append(0)

        _auto_fused_backward[BATCH, *Self.OPS](
            grad_input.ptr,
            grad_output.ptr,
            params.ptr,
            cache.ptr,
            grads.ptr,
            gi_storage.unsafe_ptr(),
            0, 0, 0,
        )

    # =========================================================================
    # GPU Forward (with cache)
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
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
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        _auto_fused_forward_gpu[BATCH, *Self.OPS](
            ctx,
            input.ptr,
            output.ptr,
            params.ptr,
            cache.ptr,
            workspace.unsafe_ptr(),
            0, 0, 0,
        )

    # =========================================================================
    # GPU Forward (no cache — inference)
    # =========================================================================

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int,
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
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        # Dummy cache carved from workspace (after inter region) — no allocation.
        var cache_v = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.CACHE_SIZE),
            MutAnyOrigin,
        ](workspace.unsafe_ptr() + BATCH * Self.INTER_SIZE_PER_SAMPLE)
        Self.forward_gpu[BATCH](ctx, output, input, params, cache_v, workspace)

    # =========================================================================
    # GPU Forward (no cache) — on DeviceStream
    # =========================================================================

    @staticmethod
    fn forward_gpu_no_cache_on_stream[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        var cache_v = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.CACHE_SIZE),
            MutAnyOrigin,
        ](workspace.unsafe_ptr() + BATCH * Self.INTER_SIZE_PER_SAMPLE)
        _auto_fused_forward_gpu_on_stream[BATCH, *Self.OPS](
            ctx,
            stream,
            input.ptr,
            output.ptr,
            params.ptr,
            cache_v.ptr,
            workspace.unsafe_ptr(),
            0, 0, 0,
        )

    # =========================================================================
    # GPU Backward
    # =========================================================================

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        _auto_fused_backward_gpu[BATCH, *Self.OPS](
            ctx,
            grad_input.ptr,
            grad_output.ptr,
            params.ptr,
            cache.ptr,
            grads.ptr,
            workspace.unsafe_ptr(),
            0, 0, 0,
        )
