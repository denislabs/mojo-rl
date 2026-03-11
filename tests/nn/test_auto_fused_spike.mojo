"""Spike: validate recursive comptime computation for AutoFused.

Tests three unknowns:
1. Can recursive functions with slice_types RETURN computed values?
2. Can we construct fused ops from ops[i].IN_DIM/OUT_DIM and call methods?
3. Can recursive forward execution work with buffer pointers?

Run: cd mojo-rl && pixi run mojo run tests/test_auto_fused_spike.mojo
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff import (
    DiffOp,
    OpID,
    AutoDiffChain,
    MatMul,
    BiasAdd,
    ReLUOp,
    TanhOp,
    SigmoidOp,
    FusedMatMulBias,
    FusedMatMulBiasReLU,
    FusedMatMulBiasTanh,
    FusedMatMulBiasSigmoid,
)
from layout import Layout, LayoutTensor
from std.builtin.variadics import Variadic
from std.random import seed, random_float64


# =============================================================================
# Spike 1: Recursive param size computation — can we RETURN values?
# =============================================================================


fn _spike_param_size[*OPS: DiffOp]() -> Int:
    """Recursively compute total PARAM_SIZE of fused groups."""
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    comptime if N == 0:
        return 0
    elif N >= 3:
        comptime assert Variadic.size(ops) >= 3
        comptime assert Variadic.size(ops) <= Variadic.size(ops)
        comptime if (
            ops[0].OP_ID == OpID.MATMUL._value
            and ops[1].OP_ID == OpID.BIAS_ADD._value
            and (
                ops[2].OP_ID == OpID.RELU._value
                or ops[2].OP_ID == OpID.TANH._value
                or ops[2].OP_ID == OpID.SIGMOID._value
            )
        ):
            # 3-op activation fusion: PARAM_SIZE = in*out + out
            comptime group_ps = ops[0].IN_DIM * ops[0].OUT_DIM + ops[0].OUT_DIM
            comptime if N == 3:
                return group_ps
            else:
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=3, end=Variadic.size(ops)
                ]
                return group_ps + _spike_param_size[*rest]()
        elif (
            ops[0].OP_ID == OpID.MATMUL._value
            and ops[1].OP_ID == OpID.BIAS_ADD._value
        ):
            # 2-op linear fusion
            comptime group_ps = ops[0].IN_DIM * ops[0].OUT_DIM + ops[0].OUT_DIM
            comptime if N == 2:
                return group_ps
            else:
                comptime assert Variadic.size(ops) >= 2
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=2, end=Variadic.size(ops)
                ]
                return group_ps + _spike_param_size[*rest]()
        else:
            # Unfused single op passthrough
            comptime if N == 1:
                return ops[0].PARAM_SIZE
            else:
                comptime assert Variadic.size(ops) >= 1
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=1, end=Variadic.size(ops)
                ]
                return ops[0].PARAM_SIZE + _spike_param_size[*rest]()
    elif N == 2:
        comptime assert Variadic.size(ops) >= 2
        comptime if (
            ops[0].OP_ID == OpID.MATMUL._value
            and ops[1].OP_ID == OpID.BIAS_ADD._value
        ):
            return ops[0].IN_DIM * ops[0].OUT_DIM + ops[0].OUT_DIM
        else:
            return ops[0].PARAM_SIZE + ops[1].PARAM_SIZE
    else:  # N == 1
        return ops[0].PARAM_SIZE


fn test_spike1():
    print("=== Spike 1: Recursive param size computation ===")

    # Case 1: M+B+R (3 ops) -> FusedMBR: ps = 2*4 + 4 = 12
    comptime ps1 = _spike_param_size[MatMul[2, 4], BiasAdd[4], ReLUOp[4]]()
    print("  M+B+R[2,4]: expected 12, got", ps1)

    # Case 2: M+B+R + M+B (5 ops) -> FusedMBR[2,4] + FusedMB[4,1]: 12 + (4*1+1) = 17
    comptime ps2 = _spike_param_size[
        MatMul[2, 4], BiasAdd[4], ReLUOp[4], MatMul[4, 1], BiasAdd[1]
    ]()
    print("  M+B+R[2,4]+M+B[4,1]: expected 17, got", ps2)

    # Case 3: M+B+R + M+B+T + M+B (8 ops)
    # FusedMBR[2,8]: 2*8+8=24, FusedMBT[8,4]: 8*4+4=36, FusedMB[4,1]: 4+1=5 → 65
    comptime ps3 = _spike_param_size[
        MatMul[2, 8],
        BiasAdd[8],
        ReLUOp[8],
        MatMul[8, 4],
        BiasAdd[4],
        TanhOp[4],
        MatMul[4, 1],
        BiasAdd[1],
    ]()
    print("  M+B+R[2,8]+M+B+T[8,4]+M+B[4,1]: expected 65, got", ps3)

    if ps1 == 12 and ps2 == 17 and ps3 == 65:
        print("  PASS")
    else:
        print("  FAIL")


# =============================================================================
# Spike 2: Fused op construction from ops members + method calls
# =============================================================================


fn test_spike2():
    print("\n=== Spike 2: Fused op construction + eval call ===")
    seed(42)

    comptime IN_D = 2
    comptime OUT_D = 4
    comptime BATCH = 1

    # Construct FusedMatMulBiasReLU using dims from ops array
    comptime ops = Variadic.types[
        T=DiffOp, MatMul[IN_D, OUT_D], BiasAdd[OUT_D], ReLUOp[OUT_D]
    ]
    comptime Fused = FusedMatMulBiasReLU[ops[0].IN_DIM, ops[0].OUT_DIM]

    # Allocate buffers
    var params_storage = List[Scalar[dtype]](capacity=Fused.PARAM_SIZE)
    for _ in range(Fused.PARAM_SIZE):
        params_storage.append(Scalar[dtype](random_float64(-1.0, 1.0)))
    var input_storage = List[Scalar[dtype]](capacity=BATCH * IN_D)
    for _ in range(BATCH * IN_D):
        input_storage.append(Scalar[dtype](random_float64(-1.0, 1.0)))
    var output_storage = List[Scalar[dtype]](capacity=BATCH * OUT_D)
    for _ in range(BATCH * OUT_D):
        output_storage.append(0)
    var cache_storage = List[Scalar[dtype]](capacity=BATCH * Fused.CACHE_SIZE)
    for _ in range(BATCH * Fused.CACHE_SIZE):
        cache_storage.append(0)

    var inp = LayoutTensor[dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin](
        input_storage.unsafe_ptr()
    )
    var out = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](
        output_storage.unsafe_ptr()
    )
    var par = LayoutTensor[
        dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin
    ](params_storage.unsafe_ptr())
    var cch = LayoutTensor[
        dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin
    ](cache_storage.unsafe_ptr())

    # Call eval via the constructed type
    Fused.eval[BATCH](inp, out, par, cch)

    print("  Output[0,0] =", Float64(rebind[Scalar[dtype]](out[0, 0])))
    print("  Output[0,1] =", Float64(rebind[Scalar[dtype]](out[0, 1])))
    print("  PASS (method call succeeded)")


# =============================================================================
# Spike 3: Recursive forward with buffer pointers
# =============================================================================


fn _spike_forward[
    BATCH: Int, *OPS: DiffOp
](
    in_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    final_out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    params_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cache_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    inter_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    param_off: Int,
    cache_off: Int,
    inter_off: Int,
):
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    comptime if N == 0:
        pass
    elif N >= 3:
        comptime assert Variadic.size(ops) >= 3
        comptime assert Variadic.size(ops) <= Variadic.size(ops)
        comptime if (
            ops[0].OP_ID == OpID.MATMUL._value
            and ops[1].OP_ID == OpID.BIAS_ADD._value
            and ops[2].OP_ID == OpID.RELU._value
        ):
            # FusedMBR
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN + G_OUT

            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](
                params_ptr + param_off
            )
            var c_v = LayoutTensor[
                dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin
            ](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[
                dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin
            ](in_ptr)

            comptime if N == 3:
                # Last group
                print("  [recursive] MBR last, G_IN=", G_IN, "G_OUT=", G_OUT)
                var out_v = LayoutTensor[
                    dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin
                ](final_out_ptr)
                FusedMatMulBiasReLU[G_IN, G_OUT].eval[BATCH](
                    in_v, out_v, p_v, c_v
                )
            else:
                # Not last
                print(
                    "  [recursive] MBR not-last, G_IN=",
                    G_IN,
                    "G_OUT=",
                    G_OUT,
                    "N=",
                    N,
                )
                var out_v = LayoutTensor[
                    dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin
                ](inter_ptr + BATCH * inter_off)
                FusedMatMulBiasReLU[G_IN, G_OUT].eval[BATCH](
                    in_v, out_v, p_v, c_v
                )
                # Debug: print inter values
                for _b in range(BATCH):
                    for _j in range(G_OUT):
                        print(
                            "    inter[",
                            _b,
                            ",",
                            _j,
                            "] =",
                            Float64(
                                rebind[Scalar[dtype]](
                                    (
                                        inter_ptr
                                        + BATCH * inter_off
                                        + _b * G_OUT
                                        + _j
                                    )[]
                                )
                            ),
                        )
                comptime rest = Variadic.slice_types[
                    element_types=ops,
                    start=3,
                    end=Variadic.size(ops),
                ]
                _spike_forward[BATCH, *rest](
                    inter_ptr + BATCH * inter_off,
                    final_out_ptr,
                    params_ptr,
                    cache_ptr,
                    inter_ptr,
                    param_off + FPS,
                    cache_off + FCS,
                    inter_off + G_OUT,
                )
        elif (
            ops[0].OP_ID == OpID.MATMUL._value
            and ops[1].OP_ID == OpID.BIAS_ADD._value
        ):
            # FusedMB
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN

            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](
                params_ptr + param_off
            )
            var c_v = LayoutTensor[
                dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin
            ](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[
                dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin
            ](in_ptr)

            comptime if N == 2:
                var out_v = LayoutTensor[
                    dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin
                ](final_out_ptr)
                FusedMatMulBias[G_IN, G_OUT].eval[BATCH](in_v, out_v, p_v, c_v)
            else:
                var out_v = LayoutTensor[
                    dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin
                ](inter_ptr + BATCH * inter_off)
                FusedMatMulBias[G_IN, G_OUT].eval[BATCH](in_v, out_v, p_v, c_v)
                comptime assert Variadic.size(ops) >= 2
                comptime rest = Variadic.slice_types[
                    element_types=ops,
                    start=2,
                    end=Variadic.size(ops),
                ]
                _spike_forward[BATCH, *rest](
                    inter_ptr + BATCH * inter_off,
                    final_out_ptr,
                    params_ptr,
                    cache_ptr,
                    inter_ptr,
                    param_off + FPS,
                    cache_off + FCS,
                    inter_off + G_OUT,
                )
        else:
            # Unfused passthrough
            pass
    elif N == 2:
        comptime assert Variadic.size(ops) >= 2
        comptime if (
            ops[0].OP_ID == OpID.MATMUL._value
            and ops[1].OP_ID == OpID.BIAS_ADD._value
        ):
            comptime G_IN = ops[0].IN_DIM
            comptime G_OUT = ops[0].OUT_DIM
            comptime FPS = G_IN * G_OUT + G_OUT
            comptime FCS = G_IN
            print(
                "  [recursive] MB last, G_IN=",
                G_IN,
                "G_OUT=",
                G_OUT,
                "param_off=",
                param_off,
                "cache_off=",
                cache_off,
            )
            var p_v = LayoutTensor[dtype, Layout.row_major(FPS), MutAnyOrigin](
                params_ptr + param_off
            )
            var c_v = LayoutTensor[
                dtype, Layout.row_major(BATCH, FCS), MutAnyOrigin
            ](cache_ptr + BATCH * cache_off)
            var in_v = LayoutTensor[
                dtype, Layout.row_major(BATCH, G_IN), MutAnyOrigin
            ](in_ptr)
            var out_v = LayoutTensor[
                dtype, Layout.row_major(BATCH, G_OUT), MutAnyOrigin
            ](final_out_ptr)
            FusedMatMulBias[G_IN, G_OUT].eval[BATCH](in_v, out_v, p_v, c_v)
            # Debug: print output
            for _b in range(BATCH):
                for _j in range(G_OUT):
                    print(
                        "    out[",
                        _b,
                        ",",
                        _j,
                        "] =",
                        Float64(
                            rebind[Scalar[dtype]](
                                (final_out_ptr + _b * G_OUT + _j)[]
                            )
                        ),
                    )
        else:
            pass
    else:
        pass


fn test_spike3():
    print("\n=== Spike 3: Recursive forward with buffers ===")
    seed(42)

    comptime IN_D = 2
    comptime HID = 4
    comptime OUT_D = 1
    comptime BATCH = 2

    # 5-op chain: M+B+R + M+B -> FusedMBR[2,4] + FusedMB[4,1]
    comptime PS = _spike_param_size[
        MatMul[IN_D, HID],
        BiasAdd[HID],
        ReLUOp[HID],
        MatMul[HID, OUT_D],
        BiasAdd[OUT_D],
    ]()
    # Cache: FusedMBR cache = IN_D + HID = 6, FusedMB cache = HID = 4 -> total 10
    comptime CS = IN_D + HID + HID  # 2 + 4 + 4 = 10
    # Inter: 1 inter buffer for FusedMBR output = HID = 4
    comptime INTER = HID

    var params_s = List[Scalar[dtype]](capacity=PS)
    for _ in range(PS):
        params_s.append(Scalar[dtype](random_float64(-1.0, 1.0)))
    var input_s = List[Scalar[dtype]](capacity=BATCH * IN_D)
    for _ in range(BATCH * IN_D):
        input_s.append(Scalar[dtype](random_float64(-1.0, 1.0)))
    var output_s = List[Scalar[dtype]](capacity=BATCH * OUT_D)
    for _ in range(BATCH * OUT_D):
        output_s.append(0)
    var cache_s = List[Scalar[dtype]](capacity=BATCH * CS)
    for _ in range(BATCH * CS):
        cache_s.append(0)
    var inter_s = List[Scalar[dtype]](capacity=BATCH * INTER)
    for _ in range(BATCH * INTER):
        inter_s.append(0)

    _spike_forward[
        BATCH,
        MatMul[IN_D, HID],
        BiasAdd[HID],
        ReLUOp[HID],
        MatMul[HID, OUT_D],
        BiasAdd[OUT_D],
    ](
        input_s.unsafe_ptr(),
        output_s.unsafe_ptr(),
        params_s.unsafe_ptr(),
        cache_s.unsafe_ptr(),
        inter_s.unsafe_ptr(),
        0,
        0,
        0,
    )

    # Compare with reference: AutoDiffChain[FusedMBR, FusedMB]
    comptime Ref = AutoDiffChain[
        FusedMatMulBiasReLU[IN_D, HID], FusedMatMulBias[HID, OUT_D]
    ]
    var ref_output_s = List[Scalar[dtype]](capacity=BATCH * OUT_D)
    for _ in range(BATCH * OUT_D):
        ref_output_s.append(0)
    var ref_cache_s = List[Scalar[dtype]](capacity=BATCH * Ref.CACHE_SIZE)
    for _ in range(BATCH * Ref.CACHE_SIZE):
        ref_cache_s.append(0)

    var ref_inp = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](input_s.unsafe_ptr())
    var ref_out = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](ref_output_s.unsafe_ptr())
    var ref_par = LayoutTensor[
        dtype, Layout.row_major(Ref.PARAM_SIZE), MutAnyOrigin
    ](params_s.unsafe_ptr())
    var ref_cch = LayoutTensor[
        dtype, Layout.row_major(BATCH, Ref.CACHE_SIZE), MutAnyOrigin
    ](ref_cache_s.unsafe_ptr())
    Ref.forward[BATCH](ref_inp, ref_out, ref_par, ref_cch)

    # Debug: print actual outputs
    for b in range(BATCH):
        for j in range(OUT_D):
            print(
                "  spike[",
                b,
                ",",
                j,
                "] =",
                Float64(output_s[b * OUT_D + j]),
                "ref =",
                Float64(ref_output_s[b * OUT_D + j]),
            )

    # Also do a direct manual forward for comparison
    var man_inter_s = List[Scalar[dtype]](capacity=BATCH * HID)
    for _ in range(BATCH * HID):
        man_inter_s.append(0)
    var man_cache1_s = List[Scalar[dtype]](capacity=BATCH * (IN_D + HID))
    for _ in range(BATCH * (IN_D + HID)):
        man_cache1_s.append(0)
    var man_cache2_s = List[Scalar[dtype]](capacity=BATCH * HID)
    for _ in range(BATCH * HID):
        man_cache2_s.append(0)
    var man_output_s = List[Scalar[dtype]](capacity=BATCH * OUT_D)
    for _ in range(BATCH * OUT_D):
        man_output_s.append(0)

    var m_inp = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin
    ](input_s.unsafe_ptr())
    var m_inter = LayoutTensor[
        dtype, Layout.row_major(BATCH, HID), MutAnyOrigin
    ](man_inter_s.unsafe_ptr())
    var m_p1 = LayoutTensor[
        dtype, Layout.row_major(IN_D * HID + HID), MutAnyOrigin
    ](params_s.unsafe_ptr())
    var m_c1 = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_D + HID), MutAnyOrigin
    ](man_cache1_s.unsafe_ptr())
    FusedMatMulBiasReLU[IN_D, HID].eval[BATCH](m_inp, m_inter, m_p1, m_c1)

    var m_inp2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, HID), MutAnyOrigin
    ](man_inter_s.unsafe_ptr())
    var m_out = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin
    ](man_output_s.unsafe_ptr())
    var m_p2 = LayoutTensor[
        dtype, Layout.row_major(HID * OUT_D + OUT_D), MutAnyOrigin
    ](params_s.unsafe_ptr() + IN_D * HID + HID)
    var m_c2 = LayoutTensor[dtype, Layout.row_major(BATCH, HID), MutAnyOrigin](
        man_cache2_s.unsafe_ptr()
    )
    FusedMatMulBias[HID, OUT_D].eval[BATCH](m_inp2, m_out, m_p2, m_c2)

    for b in range(BATCH):
        for j in range(OUT_D):
            print(
                "  manual[",
                b,
                ",",
                j,
                "] =",
                Float64(man_output_s[b * OUT_D + j]),
            )

    var max_d: Float64 = 0
    for b in range(BATCH):
        for j in range(OUT_D):
            var d = Float64(output_s[b * OUT_D + j]) - Float64(
                ref_output_s[b * OUT_D + j]
            )
            if d < 0:
                d = -d
            if d > max_d:
                max_d = d
    print("  Max diff spike vs ref:", max_d)

    var max_d2: Float64 = 0
    for b in range(BATCH):
        for j in range(OUT_D):
            var d = Float64(output_s[b * OUT_D + j]) - Float64(
                man_output_s[b * OUT_D + j]
            )
            if d < 0:
                d = -d
            if d > max_d2:
                max_d2 = d
    print("  Max diff spike vs manual:", max_d2)

    if max_d < 1e-6:
        print("  PASS")
    else:
        print("  FAIL")


fn main():
    print()
    test_spike1()
    test_spike2()
    test_spike3()
    print("\n=== ALL SPIKE TESTS DONE ===")
