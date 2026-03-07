# Autograd + IR Implementation Tasks

Tracking implementation of the compile-time autograd system described in `AUTOGRAD_IR_DESIGN.md`.

---

## Phase 1: Foundation

Core infrastructure — DiffOp trait, primitives, and AutoDiffChain. No breaking changes to existing code.

### 1.1 Scaffold `nn/autodiff/` module
- [x] Create directory structure: `nn/autodiff/{__init__.mojo, op.mojo, chain.mojo, primitives/, fused/, combinators/}`
- [x] Wire up `nn/autodiff/__init__.mojo` with public exports
- [x] Import autodiff module from `nn/__init__.mojo`

### 1.2 DiffOp trait + OpID
- [x] Define `OpID` enum struct in `nn/autodiff/op.mojo` (MATMUL, BIAS_ADD, RELU, TANH, SIGMOID, LAYER_NORM, fused ranges, combinator ranges, user-defined range)
- [x] Define `DiffOp` trait with: `OP_ID`, `IN_DIM`, `OUT_DIM`, `PARAM_SIZE`, `CACHE_SIZE`, `eval`, `vjp`, `eval_gpu`, `vjp_gpu`
- [x] Define `FusedOp` trait extending `DiffOp` with `FUSED_COUNT`

### 1.3 Core primitives — CPU
- [x] `MatMul[in_dim, out_dim]` — `nn/autodiff/primitives/matmul.mojo`
  - [x] `eval`: output = input @ W, cache input
  - [x] `vjp`: grad_input = grad_out @ W.T, dW += input.T @ grad_out
- [x] `BiasAdd[dim]` — `nn/autodiff/primitives/bias.mojo`
  - [x] `eval`: output = input + bias
  - [x] `vjp`: grad_input = grad_out, db += sum(grad_out, axis=0)
- [x] `ReLUOp[dim]` — `nn/autodiff/primitives/activations.mojo`
  - [x] `eval`: output = max(0, input), cache pre-activation
  - [x] `vjp`: grad_input = grad_out * (cache > 0)
- [x] `TanhOp[dim]`
  - [x] `eval`: output = tanh(input), cache output
  - [x] `vjp`: grad_input = grad_out * (1 - cache^2)
- [x] `SigmoidOp[dim]`
  - [x] `eval`: output = sigmoid(input), cache output
  - [x] `vjp`: grad_input = grad_out * cache * (1 - cache)

### 1.4 Core primitives — GPU
- [x] `MatMul` GPU kernel — tiled matmul with shared memory (reuse pattern from `nn/gpu/matmul.mojo`)
  - [x] `eval_kernel` with `@always_inline`
  - [x] `eval_gpu` launcher with `ctx.enqueue_function[]`
  - [x] `vjp_gpu` — backward dx kernel + backward dW kernel
- [x] `BiasAdd` GPU kernel — elementwise
- [x] `ReLUOp` GPU kernel — elementwise
- [x] `TanhOp` GPU kernel — elementwise
- [x] `SigmoidOp` GPU kernel — elementwise

### 1.5 AutoDiffChain
- [x] Implement `AutoDiffChain[*OPS: DiffOp]` in `nn/autodiff/chain.mojo`
  - [x] `Variadic.types` + `Variadic.size` for op list
  - [ ] Compile-time dimension validation with `comptime assert`
  - [x] `_param_offset`, `_cache_offset`, `_inter_offset` helpers
  - [x] `_sum_param_size`, `_sum_cache_size`, `_total_inter`
  - [x] Conform to `Model` trait
- [x] CPU `forward` — chain ops with intermediate buffers
  - [x] Single-op fast path
  - [x] Multi-op path with `comptime for` over ops
- [x] CPU `backward` — reverse iteration calling `vjp`
  - [x] Single-op fast path
  - [x] Multi-op path with `comptime for _ri in range(N)`, `i = N - 1 - _ri`
- [x] GPU `forward_gpu` — workspace-based intermediates
- [x] GPU `forward_gpu_no_cache` — inference path
- [x] GPU `backward_gpu`

### 1.6 Convenience aliases
- [x] `LinearAD[in_d, out_d]` = `AutoDiffChain[MatMul, BiasAdd]`
- [x] `LinearReLUAD[in_d, out_d]` = `AutoDiffChain[MatMul, BiasAdd, ReLUOp]`
- [x] `LinearTanhAD[in_d, out_d]` = `AutoDiffChain[MatMul, BiasAdd, TanhOp]`

### 1.7 Verification — Phase 1
- [x] Unit test: `MatMul` forward matches manual matmul computation
- [x] Unit test: `MatMul` vjp produces correct gradients
- [x] Unit test: `BiasAdd` forward/vjp correctness
- [x] Unit test: `ReLUOp` forward/vjp correctness
- [x] Unit test: `TanhOp` forward/vjp correctness
- [x] Unit test: `SigmoidOp` forward/vjp correctness
- [x] Integration test: `AutoDiffChain[MatMul[2,4], BiasAdd[4], ReLUOp[4]]` produces identical forward output as `Sequential[Linear[2,4], ReLU[4]]`
- [x] Integration test: backward pass of above produces identical gradients
- [ ] GPU test: forward/backward match CPU results
- [x] Training test: train XOR with `AutoDiffChain` MLP, converges to loss < 1e-12

---

## Phase 2: Fusion

Fused GPU kernels + compile-time pattern matching via OP_ID.

### 2.1 Fused primitives
- [x] `FusedMatMulBias[in_dim, out_dim]` — `nn/autodiff/fused/matmul_bias.mojo`
  - [x] Single GPU kernel: y = x @ W + b
  - [x] Fused VJP backward
- [x] `FusedMatMulBiasReLU[in_dim, out_dim]` — `nn/autodiff/fused/matmul_bias_relu.mojo`
  - [x] Single GPU kernel: y = relu(x @ W + b)
  - [x] Fused VJP with relu mask applied inline
- [x] `FusedMatMulBiasTanh[in_dim, out_dim]` — `nn/autodiff/fused/matmul_bias_tanh.mojo`
  - [x] Single GPU kernel: y = tanh(x @ W + b)
  - [x] Fused VJP

### 2.2 Parameterized fused activations
- [x] `Activation` trait — `nn/autodiff/fused/activation.mojo`
  - [x] `forward(pre_act) -> output`, `cache(pre_act, output) -> cache_val`, `backward(cache_val, grad_out) -> masked_grad`
  - [x] `ReLUActivation`, `TanhActivation`, `SigmoidActivation`, `MishActivation` implementations
- [x] `FusedMatMulBiasActivation[in_dim, out_dim, ACT: Activation]` — `nn/autodiff/fused/matmul_bias_act.mojo`
  - [x] Single parameterized struct (~500 lines) replacing 3 separate ~500-line files
  - [x] CPU eval/vjp using `ACT.forward()`, `ACT.cache()`, `ACT.backward()`
  - [x] GPU tiled matmul forward kernel with fused activation
  - [x] GPU dual-region backward kernel with fused activation gradient
  - [x] `rebind[Scalar[dtype]]()` needed for LayoutTensor element types passed to trait methods
- [x] Refactored `FusedMatMulBiasReLU` and `FusedMatMulBiasTanh` as thin wrapper structs
  - Concrete structs (not `comptime` aliases) to avoid comptime member folding issue
  - Each ~100 lines delegating all methods to `FusedMatMulBiasActivation`
- [x] `FusedMatMulBiasSigmoid` — first new activation added via the parameterized type
  - [x] `FUSED_MATMUL_BIAS_SIGMOID = OpID(103)` added to OpID
  - [x] Forward + backward matches unfused `AutoDiffChain[MatMul, BiasAdd, SigmoidOp]`
- [x] `FusedMatMulBiasMish` — added via `comptime` alias to `FusedMatMulBiasActivation[i, o, MishActivation]`
  - [x] `FUSED_MATMUL_BIAS_MISH = OpID(104)` added to OpID
  - [x] Forward + backward matches unfused `AutoDiffChain[MatMul, BiasAdd, MishOp]`

### 2.3 Fusion-aware aliases
- [x] `Dense[i, o]` = `AutoDiffChain[FusedMatMulBias[i, o]]`
- [x] `DenseReLU[i, o]` = `AutoDiffChain[FusedMatMulBiasReLU[i, o]]`
- [x] `DenseTanh[i, o]` = `AutoDiffChain[FusedMatMulBiasTanh[i, o]]`
- [x] `DenseSigmoid[i, o]` = `AutoDiffChain[FusedMatMulBiasSigmoid[i, o]]`

### 2.4 Fusion pass infrastructure (exploratory)
- [x] Implement `_is_matmul_bias_relu_at[idx]` using OP_ID matching
- [x] Implement `_is_matmul_bias_at[idx]` using OP_ID matching
- [x] Implement `_is_matmul_bias_tanh_at[idx]` using OP_ID matching
- [x] Implement `_is_matmul_bias_sigmoid_at[idx]` using OP_ID matching
- [x] Implement `_is_matmul_bias_activation_at[idx]` — generic check for any activation (OP_ID 10-19 range)
- [x] `_best_fusion_at[idx]` updated: returns "mbr", "mbt", "mbs", "mb", or ""
- [x] `FusedChain.one_layer_sigmoid` added
- [x] Prototype `FusedAutoDiffChain` or `auto_fuse` wrapper (depends on variadic type rewriting feasibility)
  - **Finding**: `Variadic.slice_types` + `comptime assert` enables recursive compile-time fusion on arbitrary-length op chains. A fully generic `greedy_fuse[*OPS]` IS feasible.
  - **Approach**: Use `FusionAnalyzer` struct with `_is_matmul_bias_relu_at[idx]()` etc. Pattern matchers must use `comptime if in_bounds: return (access) else: return False` to avoid compiler crashes from variadic out-of-bounds access in dead code.
  - **Verified**: Multi-layer partial fusion `[MatMul,BiasAdd,ReLU,MatMul,BiasAdd]` → `[FusedMBR,FusedMB]` matches forward+backward numerically (~3e-8 max diff).
  - **Breakthrough (slice_types + comptime assert)**: `Variadic.slice_types[element_types=ops, start=S, end=E]` slices a variadic type pack. On parametric variadics, `comptime assert Variadic.size(ops) >= E` provides evidence to the constraint checker. Combined with `greedy_fuse[*rest]()` recursive calls, this enables arbitrary-depth fusion. See `tests/test_slice_types.mojo` for 11-op → 4-fused-op recursive demo.
  - **Caveats**: (1) No transitive inequality — each distinct `end` value needs its own `comptime assert`. (2) Dynamic `end=Variadic.size(ops)` requires tautology assert: `comptime assert Variadic.size(ops) <= Variadic.size(ops)`. (3) `concat_types` returns unusable dependent type — not needed since slice_types suffices.

### 2.5 Verification — Phase 2
- [x] Unit test: `FusedMatMulBias` forward + vjp matches unfused `MatMul → BiasAdd`
- [x] Unit test: `FusedMatMulBiasReLU` forward matches unfused `MatMul → BiasAdd → ReLU`
- [x] Unit test: `FusedMatMulBiasReLU` vjp produces identical gradients to unfused chain
- [x] Unit test: `FusedMatMulBiasTanh` forward + vjp matches unfused chain
- [x] Unit test: `FusedMatMulBiasSigmoid` forward + vjp matches unfused `MatMul → BiasAdd → Sigmoid`
- [x] Unit test: Alias dimension checks (Dense, DenseReLU, DenseTanh, DenseSigmoid)
- [x] Training test: Fused MLP (DenseReLU + Dense) converges on XOR (loss < 1e-12)
- [ ] Benchmark: fused vs unfused GPU kernel launches (measure wall time)
- [ ] Benchmark: `DenseReLU` vs hand-coded `LinearReLU` (should be equivalent or faster)

---

## Phase 3: Additional Primitives

Expand the primitive catalog for broader model support.

### 3.1 Normalization
- [x] `LayerNormOp[dim]` — forward + vjp (CPU + GPU), eps=1e-5 hardcoded
- [x] `RMSNormOp[dim]` — forward + vjp (CPU + GPU), eps=1e-5 hardcoded

### 3.2 Arithmetic
- N/A `ElemAdd[dim]` — not needed: `BiasAdd` covers learned additive bias, `Residual` covers skip connections. DiffOp is unary so a true binary add doesn't fit the trait.
- [x] `ElemMul[dim]` — elementwise multiplication with learned gamma
- [x] `Scale[dim, numerator, denominator]` — multiply by compile-time constant ratio

### 3.3 Reduction
- [x] `ReduceSum[dim]` — reduce feature dim to scalar per batch element
- [x] `ReduceMean[dim]` — reduce feature dim to mean per batch element

### 3.4 Softmax & Activations
- [x] `SoftmaxOp[dim]` — forward + vjp (numerically stable with max subtraction)
- [x] `MishOp[dim]` — forward + vjp (x * tanh(softplus(x)))

### 3.5 Verification — Phase 3
- [x] Finite difference gradient check for each new primitive (all 8 ops pass, tol 1e-3)
- [x] Forward computation verification for each op
- [x] AutoDiffChain composition tests (MatMul->Softmax, MatMul->LayerNorm, MatMul->RMSNorm->Mish, MatMul->ReduceMean)
- [ ] GPU vs CPU numerical agreement tests

---

## Phase 4: Combinators

Non-sequential topologies using the trait gateway pattern.

### 4.1 Residual
- [x] Implement `Residual[Inner: Model]` — `nn/autodiff/combinators/residual.mojo`
  - [x] Conforms to `Model`
  - [x] Forward: output = Inner.forward(input) + input
  - [x] Backward: grad_input = Inner.backward(grad_output) + grad_output
  - [x] GPU forward/backward
- [ ] Convenience: `ResBlock[dim]` = `Residual[AutoDiffChain[MatMul, BiasAdd, ReLU, MatMul, BiasAdd]]`

### 4.2 Parallel (variadic)
- [x] Implement `Parallel[*BRANCHES: Model]` — `nn/autodiff/combinators/parallel.mojo`
  - [x] Conforms to `Model`, variadic N branches via `Variadic.types`
  - [x] `_sum_out_dim`, `_sum_param_size`, `_sum_cache_size`, `_sum_ws` helpers
  - [x] `_out_offset[idx]`, `_param_offset[idx]`, `_cache_offset[idx]`, `_ws_branch_offset[idx]` offset helpers
  - [x] Forward: output = concat(B0(x), B1(x), ..., B_{N-1}(x)) — flat buffer + `comptime for` interleave
  - [x] Backward: de-interleave grad, flat `N * BATCH * IN_DIM` buffer for per-branch grad_input, sum contributions
  - [x] GPU forward/backward — per-branch copy/split kernels via `comptime for`

### 4.3 Repeat
- [x] Implement `Repeat[N: Int, Inner: Model]` — `nn/autodiff/combinators/repeat.mojo`
  - [x] Conforms to `Model`
  - [x] Forward: apply Inner N times, cache each iteration
  - [x] Backward: reverse N iterations, accumulate to shared grads
  - [x] GPU forward/backward

### 4.4 Verification — Phase 4
- [x] Unit test: `Residual` forward = inner(x) + x
- [x] Unit test: `Residual` backward gradient correctness (finite diff)
- [x] Unit test: `Parallel[Dense[2,3], Dense[2,2]]` output dimensions and forward correctness
- [x] Unit test: `Parallel[Dense[2,3], Dense[2,2]]` gradient check (input + param)
- [x] Unit test: `Parallel[Dense[2,3], Dense[2,2], Dense[2,1]]` 3-branch forward + gradient check
- [x] Unit test: `Repeat[3, Dense[4,4]]` forward matches 3x manual application
- [x] Unit test: `Repeat[3, Inner]` backward accumulates grads correctly (3x accumulation)
- [x] Unit test: `Repeat[1, Dense[4,4]]` matches Dense directly
- [x] Integration test: `Sequential[DenseReLU[2,8], Residual[Dense[8,8]], Dense[8,1]]` trains XOR (loss < 0.05)
- [x] Integration test: `Residual[Sequential[DenseReLU[4,4], Dense[4,4]]]` nested gradient check
- [ ] Integration test: `Repeat[N, TransformerBlock]` compiles and trains

---

## Phase 5: Automatic Fusion

Compile-time automatic fusion of unfused op chains into fused kernels.

**User writes:** `AutoFused[MatMul[2,4], BiasAdd[4], ReLUOp[4], MatMul[4,1], BiasAdd[1]]`
**Gets:** A `Model`-conforming struct that internally executes `FusedMBR[2,4] → FusedMB[4,1]`

### 5.1 Spike: validate recursive comptime computation
- [x] Recursive param size computation — `_fused_param_size[*OPS]()` returns sum via `slice_types` recursion
- [x] Fused op construction from ops members — `FusedMatMulBiasReLU[ops[0].IN_DIM, ops[0].OUT_DIM].eval(...)` works
- [x] Recursive forward execution — threading buffer pointers + offsets through `slice_types` recursion
- [x] Spike file: `tests/test_auto_fused_spike.mojo`

### 5.2 `AutoFused[*OPS: DiffOp]` struct — core
- [x] Recursive helpers: `_fused_param_size`, `_fused_cache_size`, `_fused_inter_size`
- [x] Pattern matching: M+B+Act (3 ops, any activation OP_ID 10-19 via `_is_act()` range check), M+B (2 ops), passthrough (1 op)
  - Uses `FusedMatMulBiasActivation[in, out, ACT]` directly for all activation fusions (ReLU, Tanh, Sigmoid, Mish)
- [x] `AutoFused[*OPS: DiffOp](Model)` struct with compile-time constants
- [x] File: `nn/autodiff/auto_fused.mojo`

### 5.3 Forward pass — CPU
- [x] Recursive `_auto_fused_forward[BATCH, *OPS]()` with buffer pointer threading
- [x] Fused op `.eval[BATCH]()` calls at each recursion level
- [x] Intermediate buffer management for group boundaries

### 5.4 Backward pass — CPU
- [x] Recursive `_auto_fused_backward[BATCH, *OPS]()` — recurse first, VJP on return
- [x] Natural reverse order via recursion (last fused group's VJP runs first)
- [x] Gradient accumulation with proper offset threading

### 5.5 GPU forward/backward
- [x] `_auto_fused_forward_gpu[BATCH, *OPS]()` with `DeviceContext`
- [x] `_auto_fused_backward_gpu[BATCH, *OPS]()` with `DeviceContext`

### 5.6 Wire up exports
- [x] Add `AutoFused` to `nn/autodiff/__init__.mojo`

### 5.7 Verification — Phase 5
- [x] Compile-time dimensions: 5-op, 8-op, single-op — all correct
- [x] Forward: `AutoFused[M,B,R,M,B]` matches `AutoDiffChain[FusedMBR, FusedMB]` (diff = 0.0)
- [x] Backward: grad_input + grad_params match reference (diff = 0.0)
- [x] Forward: 8-op M+B+R + M+B+T + M+B matches reference (diff = 0.0)
- [x] Backward: 8-op grad_input + grad_params match reference (diff = 0.0)
- [x] XOR training: converges to loss < 1e-12
- [x] Single op passthrough: `AutoFused[MatMul[3,5]]` matches standalone MatMul
- [x] Sigmoid fusion: `AutoFused[M,B,S]` matches `FusedMatMulBiasSigmoid`
- [x] Mish fusion: `AutoFused[M,B,Mish]` matches `FusedMatMulBiasMish` (diff = 0.0)
- [x] Deep chain: 11-op → 4 fused groups, forward + backward match reference (diff = 0.0)
- [ ] Integration: `AutoFused` works with `Trainer`, `Residual`, `Parallel`
- [ ] GPU test: forward/backward match CPU results
- [ ] Benchmark: `AutoFused` vs hand-composed `AutoDiffChain[FusedOps]` performance

### 5.8 Update AUTOGRAD_TASKS.md
- [x] Add Phase 5: Automatic Fusion
- [x] Renumber Migration & Polish → Phase 6

---

## Phase 6: Migration & Polish

Replace hand-coded layers, final integration.

### 6.1 Replace hand-coded layers
- [ ] Verify `LinearAD` matches `Linear` numerically (forward + backward + GPU)
- [ ] Verify `LinearReLUAD` matches `LinearReLU` numerically
- [ ] Verify `LinearTanhAD` matches `LinearTanh` numerically
- [ ] Replace `Linear` usages with `LinearAD` (or `Dense`) in RL environments
- [ ] Replace `LinearReLU` usages with `DenseReLU` in RL environments
- [ ] Keep old implementations in `nn/model/` as reference (don't delete)

### 6.2 StochasticActor migration
- [ ] Evaluate: can `StochasticActor` be expressed with `AutoDiffChain` + combinators?
- [ ] If yes: refactor to use composed primitives
- [ ] If no: document what additional primitives/combinators would be needed

### 6.3 Documentation
- [ ] Add docstrings to all public types: DiffOp, AutoDiffChain, OpID, each primitive
- [ ] Update `AUTOGRAD_IR_DESIGN.md` with lessons learned during implementation
- [ ] Add usage examples to `nn/autodiff/__init__.mojo`

### 6.4 Benchmarks
- [ ] Benchmark: AutoDiffChain MLP vs hand-coded MLP (CPU throughput)
- [ ] Benchmark: AutoDiffChain MLP vs hand-coded MLP (GPU throughput)
- [ ] Benchmark: fused DenseReLU vs unfused AutoDiffChain[MatMul, BiasAdd, ReLU] (GPU kernel launches)
- [ ] Benchmark: compilation time for Sequential of 10, 50, 100 AutoDiffChain layers
- [ ] Document benchmark results

---

## Open Research / Exploration

Items that need investigation before committing to an approach.

- [x] **Variadic type rewriting**: Can Mojo build a new variadic type list from an existing one at compile time? (needed for automatic fusion pass)
  - **Answer**: YES — `Variadic.slice_types` + `comptime assert` enables recursive compile-time fusion on arbitrary-length op chains. Key technique: `comptime assert Variadic.size(ops) >= end_value` provides evidence to the constraint checker for `slice_types[element_types=ops, start=S, end=E]`. The sliced result can be unpacked with `*rest` into recursive calls. Proven with 11-op → 4-fused-op recursive greedy fusion in `tests/test_slice_types.mojo`. Note: `concat_types` is broken (returns unusable dependent type) but not needed since slice-and-recurse covers all fusion patterns.
- [ ] **@always_inline kernel fusion**: Does Mojo actually inline adjacent `@always_inline` GPU kernels into a single kernel launch via `ctx.enqueue_function[]`?
- [ ] **Compile-time scaling**: Test `comptime for` with 100+ iterations — measure compilation time impact
- [ ] **Attention primitive**: Design `ScaledDotProductAttention` as a DiffOp — what should its cache look like? Can flash-attention tiling fit the DiffOp interface?
- [ ] **Conv2D primitive**: Design `Conv2D` as a DiffOp — im2col vs direct convolution, how does the cache work for spatial inputs?
