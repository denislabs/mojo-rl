"""Proof-of-concept: compile-time OP_ID pattern matching on variadic DiffOp packs.

Tests whether we can:
1. Read OP_ID from individual ops in a variadic type pack
2. Pattern-match sequences like MatMul+BiasAdd+ReLU at compile time
3. Use pattern matching inside a struct parameterized on *OPS

Run with:
    pixi run mojo run tests/test_fusion_pass_poc.mojo
"""

from std.builtin.variadics import Variadic

from nn.constants import dtype
from nn.autodiff.op import DiffOp, OpID
from nn.autodiff import MatMul, BiasAdd, ReLUOp, TanhOp, SigmoidOp


# =============================================================================
# Step 1: Can we read OP_ID from a variadic type pack?
# =============================================================================


fn test_read_op_id() -> Int:
    print("\n" + "=" * 70)
    print("TEST: Read OP_ID from variadic type pack")
    print("=" * 70)
    var fails = 0

    comptime types = Variadic.types[
        T=DiffOp, MatMul[2, 4], BiasAdd[4], ReLUOp[4]
    ]
    comptime N = Variadic.size(types)

    comptime id0 = types[0].OP_ID
    comptime id1 = types[1].OP_ID
    comptime id2 = types[2].OP_ID

    print("  N = " + String(N))
    print("  OP_ID[0] = " + String(id0) + " (expect MATMUL=1)")
    print("  OP_ID[1] = " + String(id1) + " (expect BIAS_ADD=2)")
    print("  OP_ID[2] = " + String(id2) + " (expect RELU=10)")

    comptime if N != 3:
        print("  FAIL: N != 3")
        fails += 1
    else:
        print("  PASS: N = 3")

    comptime if id0 != OpID.MATMUL._value:
        print("  FAIL: OP_ID[0] mismatch")
        fails += 1
    else:
        print("  PASS: OP_ID[0] = MATMUL")

    comptime if id1 != OpID.BIAS_ADD._value:
        print("  FAIL: OP_ID[1] mismatch")
        fails += 1
    else:
        print("  PASS: OP_ID[1] = BIAS_ADD")

    comptime if id2 != OpID.RELU._value:
        print("  FAIL: OP_ID[2] mismatch")
        fails += 1
    else:
        print("  PASS: OP_ID[2] = RELU")

    return fails


# =============================================================================
# Step 2: Pattern matching via struct with variadic params
# =============================================================================


struct FusionAnalyzer[*OPS: DiffOp]:
    """Analyze a DiffOp chain for fusible patterns using OP_ID matching."""

    comptime op_types = Variadic.types[T=DiffOp, *Self.OPS]
    comptime N = Variadic.size(Self.op_types)

    @staticmethod
    fn _is_matmul_bias_at[idx: Int]() -> Bool:
        """Check if ops[idx:idx+2] is MatMul followed by BiasAdd."""
        comptime if idx + 1 < Self.N:
            return (
                Self.op_types[idx].OP_ID == OpID.MATMUL._value
                and Self.op_types[idx + 1].OP_ID == OpID.BIAS_ADD._value
            )
        else:
            return False

    @staticmethod
    fn _is_matmul_bias_relu_at[idx: Int]() -> Bool:
        """Check if ops[idx:idx+3] is MatMul, BiasAdd, ReLU."""
        comptime if idx + 2 < Self.N:
            return (
                Self.op_types[idx].OP_ID == OpID.MATMUL._value
                and Self.op_types[idx + 1].OP_ID == OpID.BIAS_ADD._value
                and Self.op_types[idx + 2].OP_ID == OpID.RELU._value
            )
        else:
            return False

    @staticmethod
    fn _is_matmul_bias_tanh_at[idx: Int]() -> Bool:
        """Check if ops[idx:idx+3] is MatMul, BiasAdd, Tanh."""
        comptime if idx + 2 < Self.N:
            return (
                Self.op_types[idx].OP_ID == OpID.MATMUL._value
                and Self.op_types[idx + 1].OP_ID == OpID.BIAS_ADD._value
                and Self.op_types[idx + 2].OP_ID == OpID.TANH._value
            )
        else:
            return False


fn test_pattern_matching() -> Int:
    print("\n" + "=" * 70)
    print("TEST: Compile-time pattern matching via struct")
    print("=" * 70)
    var fails = 0

    # --- MatMul -> BiasAdd -> ReLU chain ---
    comptime Analyzer1 = FusionAnalyzer[MatMul[2, 4], BiasAdd[4], ReLUOp[4]]

    comptime mb_0 = Analyzer1._is_matmul_bias_at[0]()
    comptime if mb_0:
        print("  PASS: _is_matmul_bias_at[0] = True")
    else:
        print("  FAIL: _is_matmul_bias_at[0] should be True")
        fails += 1

    comptime mb_1 = Analyzer1._is_matmul_bias_at[1]()
    comptime if not mb_1:
        print("  PASS: _is_matmul_bias_at[1] = False")
    else:
        print("  FAIL: _is_matmul_bias_at[1] should be False")
        fails += 1

    comptime mbr_0 = Analyzer1._is_matmul_bias_relu_at[0]()
    comptime if mbr_0:
        print("  PASS: _is_matmul_bias_relu_at[0] = True")
    else:
        print("  FAIL: _is_matmul_bias_relu_at[0] should be True")
        fails += 1

    # relu pattern should NOT match at boundary
    comptime mbr_1 = Analyzer1._is_matmul_bias_relu_at[1]()
    comptime if not mbr_1:
        print("  PASS: _is_matmul_bias_relu_at[1] = False (boundary)")
    else:
        print("  FAIL: should be False at boundary")
        fails += 1

    # --- MatMul -> BiasAdd -> TanhOp chain ---
    comptime Analyzer2 = FusionAnalyzer[MatMul[2, 4], BiasAdd[4], TanhOp[4]]

    comptime mbt_0 = Analyzer2._is_matmul_bias_tanh_at[0]()
    comptime if mbt_0:
        print("  PASS: _is_matmul_bias_tanh_at[0] = True (tanh chain)")
    else:
        print("  FAIL: should be True")
        fails += 1

    comptime mbr_tanh = Analyzer2._is_matmul_bias_relu_at[0]()
    comptime if not mbr_tanh:
        print("  PASS: _is_matmul_bias_relu_at[0] = False (tanh chain)")
    else:
        print("  FAIL: relu match should not fire on tanh chain")
        fails += 1

    # --- Multi-layer: MatMul->BiasAdd->ReLU->MatMul->BiasAdd ---
    comptime Analyzer3 = FusionAnalyzer[
        MatMul[2, 4], BiasAdd[4], ReLUOp[4], MatMul[4, 3], BiasAdd[3]
    ]

    comptime a3_mbr_0 = Analyzer3._is_matmul_bias_relu_at[0]()
    comptime if a3_mbr_0:
        print("  PASS: multi-layer _is_matmul_bias_relu_at[0] = True")
    else:
        print("  FAIL: should be True")
        fails += 1

    comptime a3_mb_3 = Analyzer3._is_matmul_bias_at[3]()
    comptime if a3_mb_3:
        print("  PASS: multi-layer _is_matmul_bias_at[3] = True")
    else:
        print("  FAIL: should be True")
        fails += 1

    comptime a3_mbr_3 = Analyzer3._is_matmul_bias_relu_at[3]()
    comptime if not a3_mbr_3:
        print(
            "  PASS: multi-layer _is_matmul_bias_relu_at[3] = False (only 2 ops"
            " left)"
        )
    else:
        print("  FAIL: should be False")
        fails += 1

    return fails


# =============================================================================
# Step 3: Can we build a fused AutoDiffChain from pattern matching results?
#
# Approach A: comptime type alias that selects fused vs unfused
# Approach B: variadic type list rewriting (build new pack)
# =============================================================================

fn test_comptime_type_selection() -> Int:
    """Test Approach A: comptime conditional type alias."""
    print("\n" + "=" * 70)
    print("TEST: Comptime conditional type selection")
    print("=" * 70)
    var fails = 0

    from nn.autodiff import (
        AutoDiffChain,
        FusedMatMulBias,
        FusedMatMulBiasReLU,
        FusedMatMulBiasTanh,
    )
    from layout import Layout, LayoutTensor

    # The simplest fusion: if a chain is exactly MatMul+BiasAdd+ReLU,
    # replace it with FusedMatMulBiasReLU wrapped in AutoDiffChain.
    #
    # Can we write:
    #   comptime if _is_matmul_bias_relu_at[0]():
    #       comptime FusedModel = AutoDiffChain[FusedMatMulBiasReLU[IN, OUT]]
    #   else:
    #       comptime FusedModel = AutoDiffChain[MatMul[IN, OUT], BiasAdd[OUT], ReLUOp[OUT]]
    #
    # Problem: comptime aliases are scoped to the if block. Let's see
    # if we can use a helper struct instead.

    # Direct test: both should have identical PARAM_SIZE and behavior
    comptime Unfused = AutoDiffChain[MatMul[3, 4], BiasAdd[4], ReLUOp[4]]
    comptime Fused = AutoDiffChain[FusedMatMulBiasReLU[3, 4]]

    comptime if Unfused.PARAM_SIZE == Fused.PARAM_SIZE:
        print("  PASS: Unfused.PARAM_SIZE == Fused.PARAM_SIZE = " + String(Unfused.PARAM_SIZE))
    else:
        print("  FAIL: PARAM_SIZE mismatch")
        fails += 1

    comptime if Unfused.IN_DIM == Fused.IN_DIM:
        print("  PASS: IN_DIM match = " + String(Unfused.IN_DIM))
    else:
        print("  FAIL: IN_DIM mismatch")
        fails += 1

    comptime if Unfused.OUT_DIM == Fused.OUT_DIM:
        print("  PASS: OUT_DIM match = " + String(Unfused.OUT_DIM))
    else:
        print("  FAIL: OUT_DIM mismatch")
        fails += 1

    return fails


# =============================================================================
# Step 4: Variadic type list rewriting — can we build a new type pack?
#
# This is the hard question. We need to iterate the ops, detect patterns,
# and emit a new variadic list. Mojo doesn't have "append to variadic" so
# we need creative workarounds.
#
# Strategy: use nested comptime if + manual specialization for common cases.
# E.g., a 3-op chain that matches MatMul+BiasAdd+ReLU -> single fused op.
# A 5-op chain that matches [MBR, MB] -> [FusedMBR, FusedMB].
# =============================================================================

struct AutoFuse1[*OPS: DiffOp]:
    """Auto-fuse a single-pattern chain.

    If the entire chain is exactly MatMul+BiasAdd+ReLU, fuse it.
    If the entire chain is exactly MatMul+BiasAdd+Tanh, fuse it.
    If the entire chain is exactly MatMul+BiasAdd, fuse it.
    Otherwise, return the chain as-is.

    This tests whether comptime if can select different AutoDiffChain types.
    """

    comptime op_types = Variadic.types[T=DiffOp, *Self.OPS]
    comptime N = Variadic.size(Self.op_types)

    @staticmethod
    fn _is_mbr() -> Bool:
        comptime if Self.N == 3:
            return (
                Self.op_types[0].OP_ID == OpID.MATMUL._value
                and Self.op_types[1].OP_ID == OpID.BIAS_ADD._value
                and Self.op_types[2].OP_ID == OpID.RELU._value
            )
        else:
            return False

    @staticmethod
    fn _is_mbt() -> Bool:
        comptime if Self.N == 3:
            return (
                Self.op_types[0].OP_ID == OpID.MATMUL._value
                and Self.op_types[1].OP_ID == OpID.BIAS_ADD._value
                and Self.op_types[2].OP_ID == OpID.TANH._value
            )
        else:
            return False

    @staticmethod
    fn _is_mb() -> Bool:
        comptime if Self.N == 2:
            return (
                Self.op_types[0].OP_ID == OpID.MATMUL._value
                and Self.op_types[1].OP_ID == OpID.BIAS_ADD._value
            )
        else:
            return False


fn test_auto_fuse_detection() -> Int:
    """Test that AutoFuse1 correctly detects fusible patterns."""
    print("\n" + "=" * 70)
    print("TEST: AutoFuse1 pattern detection")
    print("=" * 70)
    var fails = 0

    comptime AF_relu = AutoFuse1[MatMul[3, 4], BiasAdd[4], ReLUOp[4]]
    comptime if AF_relu._is_mbr():
        print("  PASS: [MatMul, BiasAdd, ReLU] detected as MBR")
    else:
        print("  FAIL: should detect MBR")
        fails += 1

    comptime AF_tanh = AutoFuse1[MatMul[3, 4], BiasAdd[4], TanhOp[4]]
    comptime if AF_tanh._is_mbt():
        print("  PASS: [MatMul, BiasAdd, Tanh] detected as MBT")
    else:
        print("  FAIL: should detect MBT")
        fails += 1

    comptime AF_bias = AutoFuse1[MatMul[3, 4], BiasAdd[4]]
    comptime if AF_bias._is_mb():
        print("  PASS: [MatMul, BiasAdd] detected as MB")
    else:
        print("  FAIL: should detect MB")
        fails += 1

    # Not fusible
    comptime AF_nope = AutoFuse1[ReLUOp[4], TanhOp[4]]
    comptime if not AF_nope._is_mbr() and not AF_nope._is_mbt() and not AF_nope._is_mb():
        print("  PASS: [ReLU, Tanh] not fusible")
    else:
        print("  FAIL: should not be fusible")
        fails += 1

    return fails


# =============================================================================
# Step 5: The real test — can comptime if select different concrete types?
#
# We want to write something like:
#   comptime FusedType = auto_fuse[MatMul[3,4], BiasAdd[4], ReLUOp[4]]
#   # -> resolves to AutoDiffChain[FusedMatMulBiasReLU[3,4]]
# =============================================================================

fn test_comptime_type_rewrite() -> Int:
    """Test if we can use comptime if to select between fused/unfused types
    and actually use the result for computation."""
    print("\n" + "=" * 70)
    print("TEST: Comptime type rewriting (fused computation)")
    print("=" * 70)
    var fails = 0

    from nn.autodiff import AutoDiffChain, FusedMatMulBiasReLU
    from layout import Layout, LayoutTensor
    from std.random import seed, random_float64

    comptime IN_D = 3
    comptime OUT_D = 4
    comptime BATCH = 2

    # Manually select fused type based on pattern detection
    comptime AF = AutoFuse1[MatMul[IN_D, OUT_D], BiasAdd[OUT_D], ReLUOp[OUT_D]]

    # We know it's MBR — use fused type directly
    # The real question: can we do this conditionally?
    comptime if AF._is_mbr():
        comptime FusedModel = AutoDiffChain[
            FusedMatMulBiasReLU[AF.op_types[0].IN_DIM, AF.op_types[0].OUT_DIM]
        ]
        comptime UnfusedModel = AutoDiffChain[
            MatMul[IN_D, OUT_D], BiasAdd[OUT_D], ReLUOp[OUT_D]
        ]

        # Verify they have compatible PARAM_SIZE
        comptime if FusedModel.PARAM_SIZE == UnfusedModel.PARAM_SIZE:
            print("  PASS: comptime type rewrite produced compatible PARAM_SIZE = " + String(FusedModel.PARAM_SIZE))
        else:
            print("  FAIL: PARAM_SIZE mismatch after rewrite")
            fails += 1

        # Actually run both and compare
        seed(42)

        fn rand_val() -> Scalar[dtype]:
            return Scalar[dtype](random_float64(-1.0, 1.0))

        var params = List[Scalar[dtype]](capacity=FusedModel.PARAM_SIZE)
        for _ in range(FusedModel.PARAM_SIZE):
            params.append(rand_val())

        var inp = List[Scalar[dtype]](capacity=BATCH * IN_D)
        for _ in range(BATCH * IN_D):
            inp.append(rand_val())

        # Fused forward
        var f_out = List[Scalar[dtype]](capacity=BATCH * OUT_D)
        var f_cache = List[Scalar[dtype]](capacity=BATCH * FusedModel.CACHE_SIZE)
        for _ in range(BATCH * OUT_D):
            f_out.append(0)
        for _ in range(BATCH * FusedModel.CACHE_SIZE):
            f_cache.append(0)

        var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin](inp.unsafe_ptr())
        var fo_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](f_out.unsafe_ptr())
        var fp_t = LayoutTensor[dtype, Layout.row_major(FusedModel.PARAM_SIZE), MutAnyOrigin](params.unsafe_ptr())
        var fc_t = LayoutTensor[dtype, Layout.row_major(BATCH, FusedModel.CACHE_SIZE), MutAnyOrigin](f_cache.unsafe_ptr())

        var fused = FusedModel()
        fused.forward[BATCH](inp_t, fo_t, fp_t, fc_t)

        # Unfused forward
        var u_out = List[Scalar[dtype]](capacity=BATCH * OUT_D)
        var u_cache = List[Scalar[dtype]](capacity=BATCH * UnfusedModel.CACHE_SIZE)
        for _ in range(BATCH * OUT_D):
            u_out.append(0)
        for _ in range(BATCH * UnfusedModel.CACHE_SIZE):
            u_cache.append(0)

        var uo_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](u_out.unsafe_ptr())
        var up_t = LayoutTensor[dtype, Layout.row_major(UnfusedModel.PARAM_SIZE), MutAnyOrigin](params.unsafe_ptr())
        var uc_t = LayoutTensor[dtype, Layout.row_major(BATCH, UnfusedModel.CACHE_SIZE), MutAnyOrigin](u_cache.unsafe_ptr())

        var unfused = UnfusedModel()
        unfused.forward[BATCH](inp_t, uo_t, up_t, uc_t)

        # Compare
        from std.math import abs as math_abs
        var max_d: Float64 = 0
        for i in range(BATCH * OUT_D):
            var d = math_abs(Float64(f_out[i]) - Float64(u_out[i]))
            if d > max_d:
                max_d = d

        if max_d < 1e-5:
            print("  PASS: fused output matches unfused, max_diff = " + String(max_d))
        else:
            print("  FAIL: output mismatch, max_diff = " + String(max_d))
            fails += 1
    else:
        print("  FAIL: pattern not detected (should not happen)")
        fails += 1

    return fails


# =============================================================================
# Step 6: Multi-layer partial fusion
#
# Real-world test: can we fuse a 2-layer MLP?
#   [MatMul, BiasAdd, ReLU, MatMul, BiasAdd]
#   -> [FusedMatMulBiasReLU, FusedMatMulBias]
# =============================================================================


fn test_multi_layer_fusion() -> Int:
    """Test partial fusion of a 2-layer chain using FusionAnalyzer."""
    print("\n" + "=" * 70)
    print("TEST: Multi-layer partial fusion")
    print("=" * 70)
    var fails = 0

    from nn.autodiff import (
        AutoDiffChain,
        FusedMatMulBias,
        FusedMatMulBiasReLU,
        FusedMatMulBiasTanh,
    )
    from layout import Layout, LayoutTensor
    from std.random import seed, random_float64

    comptime IN_D = 3
    comptime HIDDEN = 4
    comptime OUT_D = 2
    comptime BATCH = 2

    # The unfused chain: 3->4 (relu) -> 4->2 (linear)
    comptime Unfused = AutoDiffChain[
        MatMul[IN_D, HIDDEN], BiasAdd[HIDDEN], ReLUOp[HIDDEN],
        MatMul[HIDDEN, OUT_D], BiasAdd[OUT_D],
    ]

    # Pattern detection on the unfused chain
    comptime FA = FusionAnalyzer[
        MatMul[IN_D, HIDDEN], BiasAdd[HIDDEN], ReLUOp[HIDDEN],
        MatMul[HIDDEN, OUT_D], BiasAdd[OUT_D],
    ]

    # Verify patterns detected at correct positions
    comptime if FA._is_matmul_bias_relu_at[0]():
        print("  PASS: MBR detected at [0]")
    else:
        print("  FAIL: should detect MBR at [0]")
        fails += 1

    comptime if FA._is_matmul_bias_at[3]():
        print("  PASS: MB detected at [3]")
    else:
        print("  FAIL: should detect MB at [3]")
        fails += 1

    # Build fused version: consume [0..2] as FusedMBR, [3..4] as FusedMB
    comptime Fused = AutoDiffChain[
        FusedMatMulBiasReLU[IN_D, HIDDEN],
        FusedMatMulBias[HIDDEN, OUT_D],
    ]

    # Verify PARAM_SIZE matches
    comptime if Fused.PARAM_SIZE == Unfused.PARAM_SIZE:
        print("  PASS: PARAM_SIZE match = " + String(Fused.PARAM_SIZE))
    else:
        print("  FAIL: PARAM_SIZE mismatch fused=" + String(Fused.PARAM_SIZE) + " unfused=" + String(Unfused.PARAM_SIZE))
        fails += 1

    # Verify IN_DIM/OUT_DIM match
    comptime if Fused.IN_DIM == Unfused.IN_DIM and Fused.OUT_DIM == Unfused.OUT_DIM:
        print("  PASS: IN_DIM=" + String(Fused.IN_DIM) + " OUT_DIM=" + String(Fused.OUT_DIM))
    else:
        print("  FAIL: dim mismatch")
        fails += 1

    # Run both and compare outputs
    seed(42)

    fn rand_val() -> Scalar[dtype]:
        return Scalar[dtype](random_float64(-1.0, 1.0))

    var params = List[Scalar[dtype]](capacity=Fused.PARAM_SIZE)
    for _ in range(Fused.PARAM_SIZE):
        params.append(rand_val())

    var inp = List[Scalar[dtype]](capacity=BATCH * IN_D)
    for _ in range(BATCH * IN_D):
        inp.append(rand_val())

    # Fused forward
    var f_out = List[Scalar[dtype]](capacity=BATCH * OUT_D)
    var f_cache = List[Scalar[dtype]](capacity=BATCH * Fused.CACHE_SIZE)
    for _ in range(BATCH * OUT_D):
        f_out.append(0)
    for _ in range(BATCH * Fused.CACHE_SIZE):
        f_cache.append(0)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin](inp.unsafe_ptr())
    var fo_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](f_out.unsafe_ptr())
    var fp_t = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](params.unsafe_ptr())
    var fc_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin](f_cache.unsafe_ptr())

    var fused_model = Fused()
    fused_model.forward[BATCH](inp_t, fo_t, fp_t, fc_t)

    # Unfused forward (re-seed for same input)
    var u_out = List[Scalar[dtype]](capacity=BATCH * OUT_D)
    var u_cache = List[Scalar[dtype]](capacity=BATCH * Unfused.CACHE_SIZE)
    for _ in range(BATCH * OUT_D):
        u_out.append(0)
    for _ in range(BATCH * Unfused.CACHE_SIZE):
        u_cache.append(0)

    var uo_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](u_out.unsafe_ptr())
    var up_t = LayoutTensor[dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin](params.unsafe_ptr())
    var uc_t = LayoutTensor[dtype, Layout.row_major(BATCH, Unfused.CACHE_SIZE), MutAnyOrigin](u_cache.unsafe_ptr())

    var unfused_model = Unfused()
    unfused_model.forward[BATCH](inp_t, uo_t, up_t, uc_t)

    # Compare
    from std.math import abs as math_abs
    var max_d: Float64 = 0
    for i in range(BATCH * OUT_D):
        var d = math_abs(Float64(f_out[i]) - Float64(u_out[i]))
        if d > max_d:
            max_d = d

    if max_d < 1e-5:
        print("  PASS: multi-layer fused matches unfused, max_diff = " + String(max_d))
    else:
        print("  FAIL: output mismatch, max_diff = " + String(max_d))
        fails += 1

    # Also verify backward pass
    seed(123)
    var grad_out = List[Scalar[dtype]](capacity=BATCH * OUT_D)
    for _ in range(BATCH * OUT_D):
        grad_out.append(rand_val())

    var f_grad_in = List[Scalar[dtype]](capacity=BATCH * IN_D)
    var f_grads = List[Scalar[dtype]](capacity=Fused.PARAM_SIZE)
    for _ in range(BATCH * IN_D):
        f_grad_in.append(0)
    for _ in range(Fused.PARAM_SIZE):
        f_grads.append(0)

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_D), MutAnyOrigin](grad_out.unsafe_ptr())
    var fgi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin](f_grad_in.unsafe_ptr())
    var fg_t = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](f_grads.unsafe_ptr())

    fused_model.backward[BATCH](go_t, fgi_t, fp_t, fc_t, fg_t)

    var u_grad_in = List[Scalar[dtype]](capacity=BATCH * IN_D)
    var u_grads = List[Scalar[dtype]](capacity=Unfused.PARAM_SIZE)
    for _ in range(BATCH * IN_D):
        u_grad_in.append(0)
    for _ in range(Unfused.PARAM_SIZE):
        u_grads.append(0)

    var ugi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_D), MutAnyOrigin](u_grad_in.unsafe_ptr())
    var ug_t = LayoutTensor[dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin](u_grads.unsafe_ptr())

    unfused_model.backward[BATCH](go_t, ugi_t, up_t, uc_t, ug_t)

    # Compare gradients (input grads)
    var max_gi: Float64 = 0
    for i in range(BATCH * IN_D):
        var d = math_abs(Float64(f_grad_in[i]) - Float64(u_grad_in[i]))
        if d > max_gi:
            max_gi = d

    if max_gi < 1e-5:
        print("  PASS: backward grad_input matches, max_diff = " + String(max_gi))
    else:
        print("  FAIL: backward grad_input mismatch, max_diff = " + String(max_gi))
        fails += 1

    # Compare param grads
    var max_pg: Float64 = 0
    for i in range(Fused.PARAM_SIZE):
        var d = math_abs(Float64(f_grads[i]) - Float64(u_grads[i]))
        if d > max_pg:
            max_pg = d

    if max_pg < 1e-5:
        print("  PASS: backward param_grads matches, max_diff = " + String(max_pg))
    else:
        print("  FAIL: backward param_grads mismatch, max_diff = " + String(max_pg))
        fails += 1

    return fails


# =============================================================================
# Main
# =============================================================================


fn main():
    print("=" * 70)
    print("Fusion Pass PoC: Compile-Time OP_ID Pattern Matching")
    print("=" * 70)

    var total = 0
    total += test_read_op_id()
    total += test_pattern_matching()
    total += test_comptime_type_selection()
    total += test_auto_fuse_detection()
    total += test_comptime_type_rewrite()
    total += test_multi_layer_fusion()

    print("\n" + "=" * 70)
    if total == 0:
        print("ALL FUSION PASS POC TESTS PASSED")
    else:
        print(String(total) + " TEST(S) FAILED")
    print("=" * 70)
