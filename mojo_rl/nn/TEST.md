# Neural Network Test Plan

## Test Methodology

We validate the nn package at three levels:
1. **CPU Gradcheck** -- Analytical backward vs finite-difference approximation (CPU only)
2. **CPU vs GPU** -- Same model, same params, same input: compare forward outputs and backward gradients
3. **Convergence** -- Train on known problems with expected solutions (future)

**Key principle:** Test each layer independently before testing compositions.
Errors compound through Sequential/Parallel/Residual chains -- isolate each layer first.

This mirrors the physics3d testing strategy:
- Level 1 validates backward correctness against numerical derivatives (analogous to MuJoCo comparison)
- Level 2 validates GPU implementation matches CPU (analogous to CPU vs GPU physics tests)
- If Level 1 passes and Level 2 passes, GPU correctness follows by transitivity

### Running Tests

```bash
# Level 1: CPU gradcheck (no GPU needed)
pixi run mojo run -I . tests/nn/test_layer_gradcheck.mojo

# Level 2: CPU vs GPU (requires GPU)
pixi run -e apple mojo run -I . tests/nn/test_cpu_vs_gpu.mojo    # Apple Silicon
pixi run -e nvidia mojo run -I . tests/nn/test_cpu_vs_gpu.mojo   # NVIDIA
```

### Gradcheck Parameters

- **Batch size:** BS=4 (small but enough for batch-dependent layers like BN)
- **Epsilon:** 1e-4 (standard for float32)
- **Tolerance:** 1% relative error (rel_err = |ana - num| / (|ana| + |num|))
- **Input pattern:** `0.1 + (i%13)/13 * 0.8` -- values in [0.1, 0.9], avoids ReLU dead zones
- **Grad output pattern:** `0.5 + (i%7)/14 - (i%3)/6` -- varied, not all-same

---

## Test Matrix

### Level 1: CPU Numerical Gradcheck

Validates analytical `backward()` against central finite differences.
Tests both param gradients (`grad_params`) and input gradients (`grad_input`).

#### Leaf Layers (with parameters)

| Component | PS | grad_params | grad_input | Status | Notes |
|-----------|---:|:-----------:|:----------:|:------:|-------|
| Linear[8,4] | 36 | PASS | PASS | DONE | max_rel ~2.5e-5 |
| Linear[32,1] | 33 | PASS | PASS | DONE | max_rel ~2.1e-3 |
| LinearReLU[16,8] | 136 | PASS | PASS | DONE | max_rel ~4.6e-3 |
| LinearTanh[8,4] | 36 | PASS | PASS | DONE | max_rel ~3.1e-3 |
| LayerNorm[16] | 32 | PASS | PASS | DONE | max_rel ~2.7e-3 |
| Conv2DReLU[2,4,3,1,1,5,5] | 76 | PASS | PASS | DONE | max_rel ~5.0e-3 |
| Conv2DLayer[2,4,3,1,1,5,5] | 76 | PASS | PASS | DONE | max_rel ~1.1e-3 |

#### Leaf Layers (param-free, grad_input only)

| Component | CS | grad_input | Status | Notes |
|-----------|---:|:----------:|:------:|-------|
| ReLU[8] | 8 | PASS | DONE | max_rel ~6.4e-6 |
| Tanh[8] | 8 | PASS | DONE | max_rel ~1.0e-5 |
| Sigmoid[8] | 8 | PASS | DONE | max_rel ~6.9e-5 |
| Softmax[8] | 8 | PASS | DONE | max_rel ~9.0e-3 (dense Jacobian) |
| Mish[8] | 8 | PASS | DONE | max_rel ~1.9e-5 |

#### Combinators

| Component | PS | grad_params | grad_input | Status | Notes |
|-----------|---:|:-----------:|:----------:|:------:|-------|
| Sequential[LinearReLU[8,6], Linear[6,4]] | 84 | PASS | PASS | DONE | max_rel ~5.3e-4 |
| Parallel[Linear[8,4], Linear[8,1]] | 48 | PASS | ~PASS | DONE | 1 input fail (rel=0.014, tiny abs ~7e-4) |
| Parallel[Linear[8,4], Linear[8,4]] | 72 | PASS | PASS | DONE | max_rel ~2.1e-4 |
| Residual[LinearReLU[8,8]] | 72 | PASS | PASS | DONE | max_rel ~1.3e-4 |
| Repeat[2, LinearReLU[8,8]] | 72 | PASS | PASS | DONE | max_rel ~1.5e-3 |
| SkipConcat[Linear[8,4]] | 36 | PASS | PASS | DONE | max_rel ~4.0e-5 |
| DualPath[Linear[8,4], Linear[8,1]] | 45 | PASS | ~PASS | DONE | Same marginal fail as Parallel diff-size |
| SplitApply[Linear[4,3], Linear[4,2], 4] | 26 | PASS | PASS | DONE | max_rel ~4.0e-4 |
| FanOut[Linear[8,4], 2] | 72 | PASS | PASS | DONE | max_rel ~2.1e-4 |

#### Realistic Architectures

| Component | PS | grad_params | grad_input | Status | Notes |
|-----------|---:|:-----------:|:----------:|:------:|-------|
| MLP dual-head (TicTacToe-like) | 2448 | ~PASS | ~PASS | DONE | 1 marginal param fail (rel=0.019), 1 marginal input fail |
| Conv dual-head (AlphaZero-like) | 2420 | PASS | ~PASS | DONE | 4 input fails at tiny abs (~1e-4) |

#### ResBlocks

| Component | PS | grad_params | grad_input | Status | Notes |
|-----------|---:|:-----------:|:----------:|:------:|-------|
| ResBlockConv2D[4,3,1,5,5] | 296 | PASS | PASS | DONE | CPU max_rel ~1.1e-3. GPU gradcheck also PASS (5.9e-5 / 1.1e-3) |
| ResBlockConv2D[8,3,1,5,5] | 1168 | PASS | PASS | DONE | GPU gradcheck PASS (7.0e-3 / 2.2e-3) |
| ResBlockConv2DBN[4,3,1,5,5] | 328 | ~PASS | ~PASS | DONE | BN noise in finite-diff. CPU+GPU backward match (Level 2 PASS) |

#### Excluded from CPU Gradcheck

| Component | Reason |
|-----------|--------|
| BatchNorm2D | Running stats updated during forward complicate finite-diff on those param slots. Has dedicated test: `test_batch_norm_2d.mojo` |
| NoisyLinear | Noise injection during forward makes finite-diff non-deterministic |

### Level 2: CPU vs GPU Consistency

Same model + same params + same input: compare forward outputs and backward gradients.
Tolerances depend on accumulation order (GPU reductions differ from CPU sequential sums).

Results from Apple Silicon (2026-04-13). NVIDIA results pending.

| Component | forward | grad_params | grad_input | Status | Notes |
|-----------|:-------:|:-----------:|:----------:|:------:|-------|
| Linear[8,4] | PASS | PASS | PASS | DONE | All ~1e-7 (exact) |
| Linear[32,1] | PASS | PASS | PASS | DONE | All ~1e-7 (exact) |
| LinearReLU[16,8] | PASS | PASS | PASS | DONE | All ~1e-7 (exact) |
| LinearTanh[8,4] | PASS | PASS | PASS | DONE | max ~2.4e-7 |
| LayerNorm[16] | PASS | PASS | PASS | DONE | Fixed: added param grad kernel |
| Conv2DReLU[2,4,3,1,1,5,5] | PASS | PASS | PASS | DONE | max ~5.7e-6 |
| Conv2DBatchNormReLU[2,4,3,1,1,5,5] | PASS | ~PASS | PASS | DONE | 4 conv-bias noise (abs~1e-6) |
| ReLU[8] | PASS | n/a | PASS | DONE | exact (0.0) |
| Tanh[8] | PASS | n/a | PASS | DONE | max ~8.9e-8 |
| Sigmoid[8] | PASS | n/a | PASS | DONE | max ~1.5e-8 |
| Sequential[LinearReLU, Linear] | PASS | PASS | PASS | DONE | max ~1.2e-7 |
| Parallel[Linear, Linear] (diff) | PASS | PASS | PASS | DONE | exact (0.0) |
| Residual[LinearReLU] | PASS | PASS | PASS | DONE | exact (0.0) |
| Repeat[2, LinearReLU] | PASS | PASS | PASS | DONE | Fixed: temp buf + accum kernel |
| SkipConcat[Linear] | PASS | PASS | PASS | DONE | exact (0.0) |
| DualPath[Linear, Linear] | PASS | PASS | PASS | DONE | exact (0.0) |
| FanOut[Linear, 2] | PASS | PASS | PASS | DONE | exact (0.0) |
| MLP dual-head | PASS | PASS | PASS | DONE | max ~2.4e-7 |
| Conv+FC pipeline | PASS | PASS | PASS | DONE | max ~2.4e-7 |
| Conv+Parallel dual-head | PASS | PASS | PASS | DONE | max ~2.4e-7 |
| ResBlockConv2D[4,3,1,5,5] | PASS | PASS | PASS | DONE | Exact forward, grad ~4.5e-7 |
| ResBlockConv2DBN[4,3,1,5,5] | PASS | ~PASS | PASS | DONE | 5 BN noise failures (abs ~1e-5) |

### Level 3: Convergence Tests (TODO)

| Problem | Model | Expected | Status |
|---------|-------|----------|:------:|
| XOR classification | 2-layer MLP | loss < 0.01 in 500 steps | TODO |
| Quadratic regression | Linear | exact minimum | TODO |
| MNIST digit (subset) | LeNet/NatureDQN | >90% accuracy | TODO |

### Level 4: Optimizer Step Correctness

Computes expected values in Float64, compares against float32 optimizer step.
Test file: `test_optimizer_step.mojo`

| Optimizer | What | Status | Error |
|-----------|------|:------:|------:|
| SGD | Single step: p -= lr * g | DONE | ~1e-7 |
| SGD | Step with lr_scale=0.5 | DONE | ~1.4e-7 |
| Adam | Step 1: moments + bias correction | DONE | ~1.3e-8 |
| Adam | Moment state verification | DONE | ~1.5e-9 |
| Adam | Step 2: accumulated moments | DONE | ~3.2e-8 |
| AdamW | Step 1: weight decay + adam update | DONE | ~2.7e-8 |
| AdamW | Zero-grad weight decay (p shrinks) | DONE | exact |

### Level 5: Composition Invariants

Algebraic properties verified by running composed model and decomposed steps
with the same params, comparing outputs element-wise.
Test file: `test_composition_invariants.mojo`

| Property | Status | Error |
|----------|:------:|------:|
| Dimension invariants (7 comptime asserts) | DONE | compile-time |
| Residual[A]: y == x + A(x) | DONE | exact (0.0) |
| SkipConcat[A]: y == cat(x, A(x)) | DONE | exact (0.0) |
| Repeat[1, A]: y == A(x) | DONE | exact (0.0) |
| Repeat[2, A]: y == A(A(x)) (shared weights) | DONE | exact (0.0) |
| Sequential[A, B]: y == B(A(x)) | DONE | exact (0.0) |
| Parallel[A, B]: y == cat(A(x), B(x)) | DONE | exact (0.0) |
| SplitApply[L, R, s]: y == cat(L(x[:s]), R(x[s:])) | DONE | exact (0.0) |
| FanOut[A, 2]: y == cat(copy0(x), copy1(x)) | DONE | exact (0.0) |
| PARAM_SIZE sums match layer-by-layer | Covered by `test_layout_invariants.mojo` |

---

## Existing Tests

### Gradcheck / Correctness Tests

| Test File | What | Type |
|-----------|------|------|
| `test_layer_gradcheck.mojo` | CPU gradcheck for all leaf layers + combinators + realistic architectures | Level 1 |
| `test_cpu_vs_gpu.mojo` | CPU vs GPU forward/backward consistency for all layer types | Level 2 |
| `test_autodiff_gradcheck.mojo` | Numerical gradcheck for autodiff primitives (MinOp, SliceOp, etc.) | Level 1 |
| `test_autodiff_phase1.mojo` | Phase 1 autodiff: MatMul, BiasAdd, ReLU, Tanh, Sigmoid + XOR convergence | Level 1+3 |
| `test_autodiff_phase2.mojo` | Phase 2: fused ops (MatMulBias, MatMulBiasActivation) | Level 1 |
| `test_batch_norm_2d.mojo` | BatchNorm2D gradcheck: finite-diff vs analytical, Conv+BN+ReLU pipeline | Level 1 |
| `test_batch_norm_gpu.mojo` | BatchNorm2D CPU vs GPU consistency | Level 2 |
| `test_layout_invariants.mojo` | Compile-time PARAM_SIZE/CACHE_SIZE/WORKSPACE consistency across compositions | Level 5 |
| `test_compute_graph.mojo` | ComputeGraph DAG builder: chain, fan-out, dual-input, gradient correctness | Level 1 |
| `test_compute_graph_rl.mojo` | ComputeGraph for RL patterns (TD-MPC2, Dreamer, SAC actor loss) | Level 1 |
| `test_sac_autodiff.mojo` | Full SAC actor loss via autodiff composition, end-to-end gradient check | Level 1 |
| `test_rsample_op.mojo` | RSampleOp + MinOp DiffOps validation | Level 1 |
| `test_resblock_conv2d.mojo` | ResBlockConv2D fused vs decomposed equivalence | Level 5 |
| `test_resblock_conv2d_bn.mojo` | ResBlockConv2DBN correctness | Level 1 |
| `test_resnet_fused_vs_unfused.mojo` | ResNet fused vs unfused block comparison | Level 5 |
| `test_identity_composite.mojo` | Identity model + CompositeParams | Level 5 |

### Architecture-Specific Tests

| Test File | What | Type |
|-----------|------|------|
| `test_alphazero_architecture.mojo` | AlphaZero forward/backward + gradcheck (TicTacToe, ConnectFour) | Level 1+2 |
| `test_nvidia_gradcheck_isolate.mojo` | NVIDIA-specific GPU gradcheck isolation (Linear -> Parallel progression) | Level 2 |
| `test_resblock_gpu_gradcheck.mojo` | GPU numerical gradcheck for ResBlockConv2D and ResBlockConv2DBN | Level 1 (GPU) |

### Benchmarks / POCs (not correctness tests)

| Test File | What |
|-----------|------|
| `test_auto_fused.mojo` / `test_auto_fused_spike.mojo` | AutoFused fusion validation |
| `test_matmul_forward_bench.mojo` / `test_matmul_vjp_bench.mojo` | MatMul performance |
| `test_max_conv2d_bench.mojo` / `test_implicit_gemm_bench.mojo` | Conv2D kernel benchmarks |
| `test_conv2d_backward_breakdown.mojo` / `test_conv2d_breakdown.mojo` | Conv2D step-by-step analysis |
| `test_matmul_bias_backward_breakdown.mojo` | MatMul+Bias backward analysis |
| `test_bn_backward_bench.mojo` | BatchNorm backward performance |
| `test_tiletensor_conv2d_poc.mojo` / `test_tiletensor_vectorize_poc.mojo` | TileTensor POCs |
| `test_max_conv2d_poc.mojo` / `test_max_matmul_poc.mojo` | Kernel POCs |

---

## Bugs Found

### LayerNorm GPU backward: param gradients all zero (found + fixed 2026-04-13)

`test_cpu_vs_gpu.mojo` showed LayerNorm[16] GPU backward produced zero gradients for all
32 params (gamma + beta), while CPU produced correct non-zero gradients.

- **Root cause**: `backward_kernel_impl` only computed grad_input, never wrote dgamma/dbeta
- **Fix**: Added `backward_param_kernel_impl` kernel (single-thread, loops over batch)
  launched after the existing grad_input kernel in `backward_gpu`
- **File**: `mojo_rl/nn/model/layer_norm.mojo`
- **Verified**: max_rel ~2.9e-7 after fix

### Repeat combinator GPU backward: gradients under-accumulated (found + fixed 2026-04-13)

`test_cpu_vs_gpu.mojo` showed `Repeat[2, LinearReLU[8,8]]` GPU backward produced param
gradients ~4x smaller than CPU. Only the last backward iteration's grads survived.

- **Root cause**: `Inner.backward_gpu` overwrites (not accumulates) grads. With shared
  weights, all iterations wrote to the same buffer, and each overwrote the previous.
- **Fix**: When shared=True, backward into a zeroed temp buffer per iteration, then
  accumulate into main grads with an explicit `_accum_kernel` (dst[i] += src[i])
- **File**: `mojo_rl/nn/autodiff/combinators/repeat.mojo`
- **Verified**: max_rel ~4.0e-8 after fix

### Conv2DBatchNormReLU: Conv bias gradients near-zero noise (not a bug)

Conv bias params (indices 72-75) show tiny gradient differences (~1e-6) between CPU and GPU
with different signs. This is expected: BN normalization makes the Conv bias redundant,
so its gradient is essentially zero. Different reduction orders produce different noise.

---

## Known Issues

### NVIDIA max_matmul (cuBLAS) Gradient Errors

GPU gradcheck on NVIDIA shows significant mismatches for models using matmul:
- TicTacToe MLP: 41/302 params fail (max_rel=0.64)
- BN-free Parallel dual-head: 154/304 params fail (max_rel=1.0)
- Conv+Parallel dual-head: 84/321 params fail (max_rel=1.0)
- FusedResNet 1-block: 197/201 params fail (max_rel=1.0)

Apple Silicon shows much smaller discrepancies. Root cause under investigation.
See `docs/ALPHAZERO_DEBUG.md` and `test_nvidia_gradcheck_isolate.mojo`.
