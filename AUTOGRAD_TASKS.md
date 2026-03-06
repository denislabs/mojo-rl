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
- [ ] `LinearAD[in_d, out_d]` = `AutoDiffChain[MatMul, BiasAdd]`
- [ ] `LinearReLUAD[in_d, out_d]` = `AutoDiffChain[MatMul, BiasAdd, ReLUOp]`
- [ ] `LinearTanhAD[in_d, out_d]` = `AutoDiffChain[MatMul, BiasAdd, TanhOp]`

### 1.7 Verification — Phase 1
- [ ] Unit test: `MatMul` forward matches manual matmul computation
- [ ] Unit test: `MatMul` vjp produces correct gradients (finite difference check)
- [ ] Unit test: `BiasAdd` forward/vjp correctness
- [ ] Unit test: `ReLUOp` forward/vjp correctness
- [ ] Unit test: `TanhOp` forward/vjp correctness
- [ ] Unit test: `SigmoidOp` forward/vjp correctness
- [ ] Integration test: `AutoDiffChain[MatMul[2,64], BiasAdd[64], ReLUOp[64]]` produces identical forward output as hand-coded `LinearReLU[2, 64]`
- [ ] Integration test: backward pass of above produces identical gradients
- [ ] GPU test: forward/backward match CPU results
- [ ] Training test: train a small MLP with `AutoDiffChain`-based layers on a toy problem, verify convergence matches hand-coded equivalent

---

## Phase 2: Fusion

Fused GPU kernels + compile-time pattern matching via OP_ID.

### 2.1 Fused primitives
- [ ] `FusedMatMulBias[in_dim, out_dim]` — `nn/autodiff/fused/matmul_bias.mojo`
  - [ ] Single GPU kernel: y = x @ W + b
  - [ ] Fused VJP backward
- [ ] `FusedMatMulBiasReLU[in_dim, out_dim]` — `nn/autodiff/fused/matmul_bias_relu.mojo`
  - [ ] Single GPU kernel: y = relu(x @ W + b)
  - [ ] Fused VJP with relu mask applied inline
- [ ] `FusedMatMulBiasTanh[in_dim, out_dim]` — `nn/autodiff/fused/matmul_bias_tanh.mojo`
  - [ ] Single GPU kernel: y = tanh(x @ W + b)
  - [ ] Fused VJP

### 2.2 Fusion-aware aliases
- [ ] `Dense[i, o]` = `AutoDiffChain[FusedMatMulBias[i, o]]`
- [ ] `DenseReLU[i, o]` = `AutoDiffChain[FusedMatMulBiasReLU[i, o]]`
- [ ] `DenseTanh[i, o]` = `AutoDiffChain[FusedMatMulBiasTanh[i, o]]`

### 2.3 Fusion pass infrastructure (exploratory)
- [ ] Implement `_is_matmul_bias_relu_at[idx]` using OP_ID matching
- [ ] Implement `_is_matmul_bias_at[idx]` using OP_ID matching
- [ ] Implement `_is_matmul_bias_tanh_at[idx]` using OP_ID matching
- [ ] Prototype `FusedAutoDiffChain` or `auto_fuse` wrapper (depends on variadic type rewriting feasibility)

### 2.4 Verification — Phase 2
- [ ] Unit test: `FusedMatMulBiasReLU` forward matches unfused `MatMul → BiasAdd → ReLU`
- [ ] Unit test: `FusedMatMulBiasReLU` vjp produces identical gradients to unfused chain
- [ ] Benchmark: fused vs unfused GPU kernel launches (measure wall time)
- [ ] Benchmark: `DenseReLU` vs hand-coded `LinearReLU` (should be equivalent or faster)

---

## Phase 3: Additional Primitives

Expand the primitive catalog for broader model support.

### 3.1 Normalization
- [ ] `LayerNormOp[dim, EPSILON]` — forward + vjp (CPU + GPU)
- [ ] `RMSNormOp[dim, EPSILON]` — forward + vjp (CPU + GPU)

### 3.2 Arithmetic
- [ ] `ElemAdd[dim]` — elementwise addition of two inputs (for skip connections inside a chain)
- [ ] `ElemMul[dim]` — elementwise multiplication
- [ ] `Scale[dim, scale_value]` — multiply by constant

### 3.3 Reduction
- [ ] `ReduceSum[in_dim, out_dim, axis]`
- [ ] `ReduceMean[in_dim, out_dim, axis]`

### 3.4 Softmax
- [ ] `SoftmaxOp[dim]` — forward + vjp (numerically stable)
- [ ] `MishOp[dim]` — forward + vjp

### 3.5 Verification — Phase 3
- [ ] Finite difference gradient check for each new primitive
- [ ] GPU vs CPU numerical agreement tests

---

## Phase 4: Combinators

Non-sequential topologies using the trait gateway pattern.

### 4.1 Residual
- [ ] Implement `Residual[Inner: Model]` — `nn/autodiff/combinators/residual.mojo`
  - [ ] Conforms to `Model`
  - [ ] `comptime assert IN_DIM == OUT_DIM`
  - [ ] Forward: output = Inner.forward(input) + input
  - [ ] Backward: grad_input = Inner.backward(grad_output) + grad_output
  - [ ] GPU forward/backward
- [ ] Convenience: `ResBlock[dim]` = `Residual[AutoDiffChain[MatMul, BiasAdd, ReLU, MatMul, BiasAdd]]`

### 4.2 Parallel
- [ ] Implement `Parallel[BranchA: Model, BranchB: Model]` — `nn/autodiff/combinators/parallel.mojo`
  - [ ] Conforms to `Model`
  - [ ] `comptime assert BranchA.IN_DIM == BranchB.IN_DIM`
  - [ ] Forward: output = concat(BranchA(input), BranchB(input))
  - [ ] Backward: split grad, backward through each branch
  - [ ] GPU forward/backward

### 4.3 Repeat
- [ ] Implement `Repeat[N: Int, Inner: Model]` — `nn/autodiff/combinators/repeat.mojo`
  - [ ] Conforms to `Model`
  - [ ] `comptime assert IN_DIM == OUT_DIM`
  - [ ] Forward: apply Inner N times, cache each iteration
  - [ ] Backward: reverse N iterations, accumulate to shared grads
  - [ ] GPU forward/backward

### 4.4 Verification — Phase 4
- [ ] Unit test: `Residual` forward = inner(x) + x
- [ ] Unit test: `Residual` backward gradient correctness (finite diff)
- [ ] Unit test: `Parallel` output dimensions = BranchA.OUT + BranchB.OUT
- [ ] Unit test: `Repeat[3, Inner]` backward accumulates grads correctly (3x accumulation)
- [ ] Integration test: build ResNet-style model with `Sequential[LinearAD, ResBlock, ResBlock, LinearAD]`, train on toy problem
- [ ] Integration test: `Repeat[N, TransformerBlock]` compiles and trains

---

## Phase 5: Migration & Polish

Replace hand-coded layers, final integration.

### 5.1 Replace hand-coded layers
- [ ] Verify `LinearAD` matches `Linear` numerically (forward + backward + GPU)
- [ ] Verify `LinearReLUAD` matches `LinearReLU` numerically
- [ ] Verify `LinearTanhAD` matches `LinearTanh` numerically
- [ ] Replace `Linear` usages with `LinearAD` (or `Dense`) in RL environments
- [ ] Replace `LinearReLU` usages with `DenseReLU` in RL environments
- [ ] Keep old implementations in `nn/model/` as reference (don't delete)

### 5.2 StochasticActor migration
- [ ] Evaluate: can `StochasticActor` be expressed with `AutoDiffChain` + combinators?
- [ ] If yes: refactor to use composed primitives
- [ ] If no: document what additional primitives/combinators would be needed

### 5.3 Documentation
- [ ] Add docstrings to all public types: DiffOp, AutoDiffChain, OpID, each primitive
- [ ] Update `AUTOGRAD_IR_DESIGN.md` with lessons learned during implementation
- [ ] Add usage examples to `nn/autodiff/__init__.mojo`

### 5.4 Benchmarks
- [ ] Benchmark: AutoDiffChain MLP vs hand-coded MLP (CPU throughput)
- [ ] Benchmark: AutoDiffChain MLP vs hand-coded MLP (GPU throughput)
- [ ] Benchmark: fused DenseReLU vs unfused AutoDiffChain[MatMul, BiasAdd, ReLU] (GPU kernel launches)
- [ ] Benchmark: compilation time for Sequential of 10, 50, 100 AutoDiffChain layers
- [ ] Document benchmark results

---

## Open Research / Exploration

Items that need investigation before committing to an approach.

- [ ] **Variadic type rewriting**: Can Mojo build a new variadic type list from an existing one at compile time? (needed for automatic fusion pass)
- [ ] **@always_inline kernel fusion**: Does Mojo actually inline adjacent `@always_inline` GPU kernels into a single kernel launch via `ctx.enqueue_function[]`?
- [ ] **Compile-time scaling**: Test `comptime for` with 100+ iterations — measure compilation time impact
- [ ] **Attention primitive**: Design `ScaledDotProductAttention` as a DiffOp — what should its cache look like? Can flash-attention tiling fit the DiffOp interface?
- [ ] **Conv2D primitive**: Design `Conv2D` as a DiffOp — im2col vs direct convolution, how does the cache work for spatial inputs?
