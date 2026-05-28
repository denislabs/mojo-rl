# nn → nn2 porting plan

Tracker for closing the layer-coverage gap between the legacy `mojo_rl/nn/`
package and the new `mojo_rl/nn2/` package. Two corrections from the
initial gap analysis are baked in:

- **RL-specific surfaces live in `mojo_rl/deep_agents2/`, not `nn2/`.**
  Policy heads (`stochastic_actor`, `gaussian_head`, `stochastic_categorical`,
  `rsample`) and RL losses (`squashed_gaussian`) are already ported to
  `deep_agents2/primitives/` and `deep_agents2/loss/`. They are NOT nn2
  gaps. PPO-discrete is similarly unblocked via `deep_agents2/ppo/blocks/`,
  so the legacy `CategoricalLogProbOp` / `RatioOp` / `ClipSurrogateOp`
  DiffOps stay out of nn2.
- **MinMaxNorm has real consumers**: MuZero (`muzero/muzero.mojo`,
  `muzero/configs.mojo`), EZ-V2 (`efficient_zero_v2/`), and the GPU MCTS
  orchestrator (`planners/tree_search/`). Moves up from "defer" into
  Phase 2.

## Allocation rule

When deciding "nn2 or deep_agents2?" for a missing nn layer:

- **Mathy + policy-agnostic** (BN, Conv2D, Dropout, MinMaxNorm, SimNorm,
  Mish, Sigmoid, Flatten) → `nn2/primitives/`
- **Encodes an RL distribution or loss shape** (stochastic actors,
  log-probs, PPO clip, squashed-Gaussian) → `deep_agents2/primitives/`
  or `deep_agents2/loss/`

## Performance convention: `max_matmul` everywhere

Any nn2 primitive with a matmul-shaped inner kernel routes through
`from linalg.matmul import matmul as max_matmul` — never a hand-written
nested loop. Apple fp32 lands on the Accelerate cblas kernel via
`linalg.matmul.cpu.apple_accelerate.get_cblas_f32_function`; NVIDIA
and generic CPU fall through to `linalg.matmul`'s platform-tuned
kernels. `linear.mojo` is the canonical reference for the call shape.

Known `max_matmul` constraint: it rejects `transpose_a=True`. When the
gradient direction needs an A-transpose (e.g. Conv2D's `d_col = d_out_b.T
@ W`, Linear's `d_w += cache.T @ d_out`), the canonical workaround is
Apple cblas with `TRANSPOSE` on macOS fp32 and an explicit transpose
buffer + untransposed `max_matmul` elsewhere. See `linear.mojo:395-449`
and `conv2d.mojo`'s vjp for the pattern.

Gradient accumulation across batches uses cblas's `beta=1` semantics
on macOS fp32 (no temp alloc, one call) and `max_matmul` into a temp
slab + SIMD-add elsewhere — same Apple-vs-other split.

### Outstanding matmul-perf follow-ups (consumer-gated)

- **Conv2D Apple-batched single-sgemm.** Legacy nn
  (`mojo_rl/nn/autodiff/primitives/conv2d.mojo:182-210`) packs the
  im2col across the whole batch and calls `apple_sgemm_accum` once;
  nn2 currently does BATCH separate sgemm calls. Gain depends on
  BATCH and the constant-time sgemm setup overhead. Port when a CNN
  agent moves to deep_agents2 and benchmarks show it matters.
- **Conv2D GPU im2col + tiled matmul.** The current GPU path uses
  direct-convolution kernels (one thread per output / input / weight
  scalar) — correct and atomics-free, but ~K·K slower than tiled
  im2col+gemm on dense workloads. Forward only, no autodiff: cheap;
  full backward including `d_col` cblas-style transpose: not cheap.
  Consumer-gated.

### Outstanding non-matmul GPU follow-ups (consumer-gated)

- **None at landing.** Phase 2 / 4 / 5 GPU paths are now in tree
  (SimNorm / MinMaxNorm / Dropout / BatchNorm1D / BatchNorm2D /
  MaxPool / AvgPool / Conv2D direct-conv). Pool backward and Conv2D
  d_input use input-indexed kernels (one thread per input cell looping
  over overlapping output windows) — matches the nn2 / deep_agents2
  no-atomics convention. See `test_nn2_gpu_parity.mojo` for the
  CPU/GPU parity gate.

### Post-landing GPU optimizations (already applied)

- **Conv2D dW / dB block-reduce.** The first cut had one thread per
  weight scalar (or per OC for dB) doing a `BATCH·OH·OW` inner loop
  (~6k iterations per thread on NatureDQN-sized inputs). Replaced with
  one block per weight scalar / OC, `CONV_DW_TPB=128` threads reducing
  over `BATCH·OH·OW` via `block.sum` — the same primitive LayerNorm
  uses. Mirrors the legacy `nn/autodiff/primitives/conv2d.mojo:1492`
  `backward_db_kernel` pattern but cleaner (no manual smem ladder).
- **BatchNorm2D divmod hoist.** Forward / eval / backward kernels
  originally did `b = idx // SPATIAL; s = idx % SPATIAL` per element.
  Replaced with nested `for b in range(BATCH): while s < SPATIAL: ...
  s += BN2D_TPB` traversal — eliminates the per-element integer
  division (expensive on GPU at `SPATIAL = H·W = 196` for 14×14 feature
  maps).

## Status legend

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` deliberately dropped

---

## Phase 1 — Activation parity (Mish + Sigmoid) ✅

Smallest unit. Unblocks TDMPC2/MBPO composition (`Linear → LayerNorm → Mish`)
and any future Sigmoid consumer. No state-management story needed.

- [x] `primitives/ops/mish_op.mojo` — `MishOp` ElementOp, `owns_cache=False`
      (input-cached like Swish). Backward: `dy/dx = tanh_sp + x · sigmoid(x) · (1 - tanh_sp²)`
      with stable softplus `sp = max(x, 0) + log(1 + exp(-|x|))`.
- [x] `primitives/ops/sigmoid_op.mojo` — `SigmoidOp` ElementOp,
      `owns_cache=True` (output-cached like Tanh). Backward: `go · y · (1 - y)`.
- [x] `primitives/mish.mojo` + `primitives/sigmoid.mojo` — one-line aliases
      `comptime Mish[DIM] = Elementwise[DIM, MishOp]` etc.
- [x] Export both ops from `primitives/ops/__init__.mojo`.
- [x] `tests/nn2/test_mish_op.mojo` + `test_sigmoid_op.mojo` — analytic parity
      + FD gradcheck (mirror `test_swish_op.mojo`).
      - Sigmoid: fwd max |err|=0 vs closed form; bwd max |err|=1.4e-8; FD diff 4.6e-6.
      - Mish:    fwd max |err|=2.4e-7; bwd max |err|=7.6e-7; FD diff 1.1e-5.

Exit: TDMPC2-style trunk `Linear → LayerNorm → Mish` composes via
`Sequential` with no new code in nn2 or callers.

## Phase 2 — Composable norm / regularization + state-handling story ✅

Picks up the non-Conv2D blockers for TDMPC2 / DreamerV3 / MuZero / EZ-V2.

- [x] `primitives/sim_norm.mojo` — `SimNorm[DIM, GROUPS]` (TDMPC2 head:
      per-group softmax over the DIM/GROUPS subvectors).
      Forward: per-group max-subtracted softmax. Backward: standard
      softmax Jacobian per group, `grad_x = y · (grad_y - <grad_y, y>)`.
      CPU + GPU paths. GPU: one thread per (batch, group), serial inner
      loop over GROUP_SIZE (≤32 in TDMPC2). cache_y is a leaf-owned
      List + lazy DeviceBuffer. Tests: forward Σ_g sums to 1; backward
      matches analytic per-group Jacobian to 8e-10; per-group sum-zero
      invariant to 1e-7. GPU parity: fwd 3e-8, dx 1.4e-7.
- [x] `primitives/min_max_norm.mojo` — `MinMaxNorm[DIM]` (MuZero / EZ-V2
      hidden-state bound). Per-row rescaling `(x - min) / (max - min)`
      with ε clamp and degenerate-row zero-grad. Backward special-cases
      argmin/argmax lanes. CPU + GPU paths. GPU: one block per sample,
      `block.min` / `block.max` / `block.sum` for stats / argmin/argmax /
      G / Gy reductions (sentinel-index trick for argmin/argmax). Tests:
      forward [0,1] bounds + argmin→0 + argmax→1 exact; backward
      interior-lane FD 3e-5 at eps=1e-3; shift-invariance Σ grad_x = 0
      to 1e-7; degenerate row clean zero. GPU parity: fwd 0, dx 6e-6.
- [x] `primitives/flatten.mojo` — identity-copy Module with SIMD CPU
      and GPU memcpy kernel. No params, no cache. Tests: bit-identical
      round-trip at non-SIMD-aligned DIM=13.
- [x] **Train/eval design decision: per-instance runtime field.**
      Rationale documented in `primitives/dropout.mojo`. Considered
      and rejected: (a) comptime forward[MODE] would change the Module
      trait surface and force two compiled paths for every layer;
      (b) `TargetStorage.inference` was tried and removed pre-1.0
      (see `core/target_storage.mojo:46-48`) because no consumer used
      it. We keep the flag on Dropout only. Other future layers that
      need it (BatchNorm) will mirror the pattern. `set_attr["training"]`
      maps Scalar[DT] > 0.5 → True for ComputeGraph callers.
- [x] `primitives/dropout.mojo` — first nn2 leaf with non-param runtime
      state. Inverted dropout. Per-instance `training: Bool` (default
      True) + `call_counter: UInt64` (bumps on each train forward;
      gives unique Philox offsets). Cache stores the scaled mask.
      Tests: eval is identity (no counter bump); train mean preserved
      within 3σ (1.013 vs target 1.0 over 8192 lanes, p=0.3); drop
      fraction 0.291 vs target 0.3; backward equals grad_y · mask
      exactly; successive train calls differ on 127/256 lanes;
      set_attr["training"] flips state. CPU + GPU paths. GPU: flat
      1-D launch over BATCH·DIM, PhiloxRandom seeded with `Self.SEED`
      and `(call_counter, idx)` offset — host-side counter bumps each
      train forward (no on-device counter, since nn2 doesn't do CUDA
      graph capture). Cache mask in a lazy DeviceBuffer. GPU parity:
      eval is identity (0 diff); train mask is {0, 1/(1-p)} only,
      backward = grad_y·mask exactly, frac_zero ≈ 0.303 vs target 0.3.
- [-] `primitives/normed_linear.mojo` — **deliberately not ported as a
      hand-fused Module.** Composed as `Sequential[Linear, LayerNorm, Mish]`.
      Revisit only if profiling shows kernel-launch overhead matters.

Exit: TDMPC2, DreamerV3, MuZero, EZ-V2 have every non-CNN building
block they need available in nn2.

## Phase 3 — *dropped*

PPO-discrete ops (`CategoricalLogProbOp`, `RatioOp`, `ClipSurrogateOp`)
do not belong in nn2 under the allocation rule. They are absorbed by
`deep_agents2/ppo/blocks/` (already exists) and
`deep_agents2/primitives/stochastic_categorical.mojo`. Renumber later
phases mentally but keep the original numbers in this doc for traceability.

## Phase 4 — BatchNorm

Gates CNN-based agents (NatureDQN, arcade_games pixel mode, any vision
encoder). First cross-cutting running-stats story for nn2.

- [x] **State model decided: split surface.** γ/β stay as `Param`s
      (decay=False, walked by `for_each_param_auto`, gradient-tracked).
      Running mean/var are plain `List[Scalar[DT]]` side-channel fields
      — NOT walked by the param visitor (the optimizer ignores them).
      Rationale: marking them as `Param` would pull them into AdamW
      updates; building a `RunningStat` `Saveable` wrapper before any
      consumer constrains its API was scope creep. **Follow-up gated
      on first consumer**: write a `RunningStat[NAME, SIZE]` Saveable
      wrapper so checkpoints round-trip running stats. At landing,
      checkpoints save γ/β only.
- [x] `primitives/batch_norm_1d.mojo` — CPU + GPU. Per-instance
      `training: Bool` + `cache_is_training` flag. Train forward:
      batch stats + EMA-update running stats. Eval forward: running
      stats, no EMA. Backward gated on `cache_is_training` (raises
      if user calls `vjp` after an eval forward — the standard PyTorch
      footgun, caught explicitly here). Standard BN backward with
      cache_xhat + cache_inv_std. Mirrors LayerNorm's Module surface
      one-for-one. GPU: one block per feature, threads parallel-reduce
      over BATCH via `block.sum` (LayerNorm pattern, orthogonal axis).
      Train kernel updates running stats from thread 0. Tests:
      per-feature zero-mean (5e-8) + unit-var (1.2e-5); running stats
      converge to batch truth under 200 train forwards (1e-7 mean,
      exact var); eval uses running stats exactly (0); FD gradcheck
      gi 3.8e-5, dγ 1.2e-5, dβ 5.7e-6; `vjp` after eval raises.
      GPU parity: fwd 2.4e-7, dx 6.3e-7, dγ 3.8e-6, dβ 2.9e-6.
- [x] `primitives/batch_norm_2d.mojo` — landed alongside Phase 5
      (see Phase 5d below). Per-channel stats reduced over BATCH·H·W;
      otherwise identical surface + behaviour to BN1D. CPU + GPU.
      GPU: one block per channel, threads stride over the joint
      (batch, spatial) sample axis. GPU parity: fwd 1.2e-7, dx 2.9e-7,
      dγ 2.9e-6, dβ 1.9e-6.

## Phase 5 — Conv2D + Pool stack ✅

Unlocks CNN agents on nn2. Largest single chunk.

- [x] **5a — `primitives/conv2d.mojo`** — Module form, CPU + GPU.
      CPU: **im2col + `max_matmul`** reduction (matches `linear.mojo`'s
      pattern — flows through Apple Accelerate cblas on macOS fp32 and
      `linalg.matmul`'s tuned kernels elsewhere). Per-batch matmul:
      `out_b [OC, SPATIAL_OUT] = W [OC, COL_SIZE] @ col_b.T`, where
      `col_b` is the `[SPATIAL_OUT, COL_SIZE]` im2col packing of one
      batch's input slab. Comptime template `[IC, OC, K, S, P, H, W]`
      with derived `OH = (H+2P-K)//S + 1`. Module trait `IN_DIM =
      IC·H·W`, `OUT_DIM = OC·OH·OW`. Params: `weight [OC·IC·K·K]`
      decay=True, `bias [OC]` decay=False. Input-alias via
      `_cached_input_ptr` (mirror Linear/Clamp). CPU backward:
      `d_bias += column-sum(d_out_b)`; `d_weight += d_out_b @ col_b`
      via Apple cblas `beta=1` on macOS fp32 (single call, no temp)
      or `max_matmul` into a temp slab + SIMD add elsewhere;
      `d_col_b = d_out_b.T @ W` via Apple cblas `TRANSPOSE_A` on macOS
      or an explicit transpose buffer + `max_matmul` elsewhere
      (`max_matmul` rejects `transpose_a=True`, mirrors Linear); then
      col2im scatter-adds into `d_input`. GPU: direct-convolution
      kernels — one thread per output (forward), per input (d_input),
      per weight scalar (d_weight), per output channel (d_bias). No
      atomics (mirrors `c51/target_y_block.mojo:48` convention). Tests:
      1×1 identity exact; explicit 3×3 forward matches hand-computed
      reference 1e-6; FD gradcheck grad_input 1e-3, grad_weight 8e-5,
      grad_bias 1.4e-3 (tol 2e-2). GPU parity: fwd 3.8e-6, dx 2.9e-6,
      dW 1.2e-4, db 1.5e-5. Apple-batched single-sgemm and tiled
      GPU im2col+gemm deferred — gated on a real CNN consumer.
- [x] **5b — `primitives/max_pool_2d.mojo`** — no params, input-alias
      pattern (re-finds argmax in backward; K·K extra ops is trivial
      vs avoiding cache fragility). Forward picks per-window max with
      `-1e30` sentinel for padded lanes. Backward routes grad only
      to argmax lane (PyTorch-style first-occurrence tie-break). CPU
      + GPU. GPU forward: 1 thread per output. GPU backward:
      input-indexed (1 thread per input cell, looping over overlapping
      output windows that contain it). No atomics — every kernel
      writes a unique destination slot. Tests: 2×2 max-pool of 0..15
      grid picks 5/7/13/15; backward routes grad to exactly those four
      lanes, zero elsewhere. GPU parity: fwd 0, dx 0.
- [x] **5c — `primitives/avg_pool_2d.mojo`** — no params, no cache
      (gradient is uniform broadcast). `count_include_pad = True`
      convention (denominator is K·K regardless of padding overlap;
      padded cells contribute 0 in forward, receive 0 in backward).
      CPU + GPU. Same input-indexed GPU backward layout as MaxPool2D.
      Tests: 2×2 avg-pool of 0..15 → 2.5/4.5/10.5/12.5 exact; backward
      with go=1 → 0.25 per lane exact (every input lane in exactly
      one window). GPU parity: fwd 0, dx 0.
- [x] **5d — `primitives/batch_norm_2d.mojo`** — per-channel BN with
      stats reduced over BATCH·H·W. Identical surface to BN1D
      (`gamma`/`beta` Params decay=False; running_mean/var side-channel
      `List[Scalar[DT]]`; per-instance `training` + `cache_is_training`
      flags; same `vjp` requires-training-cache assertion). Tests:
      per-channel zero-mean (2e-7) + unit-var (1e-5); running stats
      converge to truth (mean 4e-7, var exact); FD gradcheck grad_input
      7e-5, dγ 3e-6, dβ 7e-6.
- [x] **5e — NatureDQN compose smoke** — scaled-down architecture
      (`[BATCH=2, 4, 16, 16] → Conv2D(4→8 k4s2p0) → ReLU → Conv2D(8→16
      k3s1p0) → ReLU → Conv2D(16→16 k3s1p0) → ReLU → Flatten →
      Linear(144→64) → ReLU → Linear(64→4)`) composes via `Sequential`
      with no extra glue. Forward + backward complete cleanly: every
      input lane (2048/2048) receives a nonzero gradient. Validates
      that the dim chain matches at every adjacent boundary and
      Sequential's variadic `Module` template handles a 10-deep mixed
      Conv/ReLU/Flatten/Linear stack.

Exit: deep_agents2 DQN / PPO CNN configs can drop their legacy `nn`
imports — every Module they need is in nn2.

## Phase 6 — Hand-fused kernels (perf-only)

Likely deferred indefinitely. nn1's bet was AutoFused; nn2's bet is
composition via `Sequential` + `ComputeGraph`. Only port if profiling
identifies a specific bottleneck.

- [ ] `Conv2D + BN + ReLU` bespoke Module.
- [ ] `Linear + BN + ReLU` bespoke Module.
- [ ] `ResBlockConv2DBN` bespoke Module.

## Phase 7 — Long tail

Low-priority gaps. Pull only when a concrete consumer needs them.

- [ ] LSTM cell (mirror `primitives/gru_cell.mojo`).
- [ ] `MinOp`, `NegateOp` (or just use `Scale[-1]`).
- [ ] Generic `GatherOp` if a non-action gather ever appears.
- [ ] Softmax as a standalone Module (today folded inside
      `loss/cross_entropy.mojo` only).

---

## Ordering

P1 → P2 → P4 sequentially (each unblocks the next consumer set).
P5 only when a CNN agent moves to deep_agents2.
P6/P7 on demand.

## Cross-references

- nn legacy layer source: `mojo_rl/nn/model/`
- nn legacy autodiff DiffOps: `mojo_rl/nn/autodiff/primitives/`
- nn2 ElementOp template: `mojo_rl/nn2/primitives/elementwise.mojo`
- nn2 ElementOp ladder: `mojo_rl/nn2/primitives/ops/`
- nn2 Module trait: `mojo_rl/nn2/core/module.mojo`
- Existing ElementOp parity tests:
  `tests/nn2/test_swish_op.mojo`, `tests/nn2/test_elementwise_*_parity.mojo`
- RL-side already-ported layers:
  `mojo_rl/deep_agents2/primitives/`, `mojo_rl/deep_agents2/loss/`
