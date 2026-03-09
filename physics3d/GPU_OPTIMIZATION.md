# GPU Optimization Plan for Physics3D Engine

## Benchmark Results (2025-03-09)

### Test Setup
- **Environment**: HalfCheetah (NQ=9, NV=9, NBODY=8, NJOINT=9, MAX_CONTACTS=20)
- **Solver**: Newton (20 solver threads per env)
- **Benchmark script**: `physics3d/tests/bench_fused_kernel.mojo`
- **Training config**: BATCH=256 envs, ROLLOUT_LEN=512, FRAME_SKIP=5

### Kernel Launch Overhead

| Metric | Apple M1 Pro | NVIDIA RTX 4090 |
|--------|-------------|-----------------|
| No-op launch latency | ~230 μs | ~2 μs |
| 3× launches per substep | ~690 μs overhead | ~6 μs overhead |
| Overhead as % of substep | **13%** | **< 1%** |

**Conclusion**: Kernel launch overhead is negligible on NVIDIA. Not worth optimizing.

### Per-Kernel Physics Timing (batch=256, with sync)

4-kernel pipeline after contact extraction:

| Kernel | M1 Pro | RTX 4090 | Speedup | % on M1 Pro | % on 4090 |
|--------|--------|----------|---------|-------------|-----------|
| Step (FK, M, bias, qacc) | 1466 μs | 469 μs | 3.1× | **45.5%** | **72.8%** |
| Contact detection | 468 μs | 33 μs | 14.2× | **14.5%** | **5.1%** |
| Solve (Newton) | 765 μs | 65 μs | 11.8× | **23.8%** | **10.1%** |
| Finalize (integration) | 520 μs | 77 μs | 6.8× | **16.1%** | **12.0%** |
| **Total substep** | **3219 μs** | **645 μs** | **5.0×** | | |

Previous 3-kernel timing (pre-optimization, batch=256):

| Kernel | M1 Pro | RTX 4090 | % on 4090 |
|--------|--------|----------|-----------|
| Step (FK, M, bias, qacc + contacts) | 3248 μs | 498 μs | **78%** |
| Solve (Newton) | 1251 μs | 64 μs | 10% |
| Finalize (integration) | 837 μs | 76 μs | 12% |
| **Total substep** | **5336 μs** | **638 μs** | |

**Contact extraction overhead on RTX 4090**: 645 vs 638 μs = +7 μs (+1.1%) — negligible.
The extra kernel launch cost (~2.5 μs) plus contact kernel execution (33 μs) is offset
by the step kernel being lighter without inline contact detection.

### Batch Scaling (RTX 4090, pipelined)

| Batch | Per substep | Per launch | Scaling |
|-------|-------------|------------|---------|
| 64 | 327 μs | 109 μs | baseline |
| 256 | 650 μs | 217 μs | 2.0× for 4× envs |
| 512 | 655 μs | 218 μs | 1.0× for 2× envs |
| 1024 | 675 μs | 225 μs | 1.03× for 2× envs |

Near-linear scaling from 256→1024 — GPU is barely loaded even at batch=1024.

### Training Pipeline Breakdown (RTX 4090, 79 rollouts)

From actual PPO HalfCheetah training run:

| Phase | Time | % |
|-------|------|---|
| Phase 1 (collect) | 180 s | 80.7% |
| Phase 2 (GAE) | 0.24 s | 0.1% |
| Phase 3 (train) | 42.7 s | 19.1% |

Collect phase breakdown:
| Sub-phase | Time |
|-----------|------|
| **Env step** | **145.0 s** |
| Reset | 31.8 s |
| NN forward | 1.8 s |
| Episode sync | 2.3 s |
| Sample+store | 0.6 s |
| Post-step | 0.8 s |

**The step kernel (78% of physics) is the bottleneck for training throughput.**

### GPU Utilization Problem

Current architecture: **1 thread per environment**.

With BATCH=256 envs, we launch 256 threads on a GPU with 16,384 CUDA cores.
That's **1.6% occupancy** for the step kernel. The GPU is massively underutilized.

The solver does better with 20 threads/env × 256 envs = 5,120 threads, but
only the parallel contact setup uses all threads — the Newton loop itself runs
on thread 0 only.

---

## Completed Optimizations

### ✅ Item 1: Phase 3 Training Timing Fix (2025-03-09)

**Problem**: Per-kernel training timings in `ppo_continuous_old.mojo` were misleading.
Only the critic optimizer timer included `ctx.synchronize()`, making it capture all
accumulated GPU work (~33s) while other sub-phase timers only measured kernel enqueue
time (~300-600μs each).

**Solution**: Added `PROFILE_PHASE3: Bool = False` comptime parameter to `train_gpu()`.
When enabled, inserts `ctx.synchronize()` before each of 10 sub-timer starts
(actor_fwd, actor_grad, actor_bwd, actor_clip, actor_optim, critic_fwd, critic_grad,
critic_bwd, critic_clip, critic_optim). Zero-cost when disabled (default).

**Files modified**:
- `deep_agents/ppo/ppo_continuous_old.mojo` — `train_gpu` function

### ✅ Item 2: GPU-Side Reset Optimization (2025-03-09)

**Problem**: `selective_reset_kernel_gpu` allocated a new `DeviceBuffer` for the model
data (~3KB) on every call. With 256 envs × ~160 resets per rollout, this caused
~40,000 unnecessary GPU memory allocations per training run, contributing to the
31.8s reset overhead (17.5% of collect phase).

**Solution**: Added `workspace_ptr` parameter to `selective_reset_kernel_gpu` trait
method. When non-null, the reset kernel reuses the pre-allocated model from the
step workspace instead of allocating a new buffer.

**Files modified**:
- `core/env_traits.mojo` — `GPUDiscreteEnv` and `GPUContinuousEnv` trait definitions
- `envs/phyics3d_env.mojo` — implementation (reuses model from workspace_ptr)
- `envs/cartpole.mojo`, `envs/car_racing/car_racing.mojo`, `envs/lunar_lander/lunar_lander.mojo`,
  `envs/bipedal_walker/bipedal_walker.mojo`, `envs/pendulum/pendulum_v2.mojo` — trait conformance
- `deep_agents/ppo/ppo_continuous_old.mojo` — training + eval reset call sites
- `deep_agents/ppo/ppo_continuous.mojo` — eval reset call site
- `deep_agents/core/gpu_onpolicy_train.mojo` — discrete + continuous reset call sites
- `deep_agents/core/gpu_offpolicy_train.mojo` — discrete + continuous reset call sites
- `deep_agents/tdmpc2/tdmpc2.mojo` — reset call site

### ❌ Item 3: Shared Memory for Model Parameters (2025-03-09) — BLOCKED

**Goal**: Load model parameters (~3.8KB) into GPU shared memory once per block,
avoiding redundant global memory reads across threads.

**Implementation attempted**: Used `LayoutTensor[..., address_space=AddressSpace.SHARED].stack_allocation()`
with cooperative loading (all threads in block load MODEL_SIZE elements in parallel,
followed by `barrier()`).

**Blocking issue**: Mojo has **no address space casting** mechanism. Sub-functions
expect `LayoutTensor[..., MutAnyOrigin]` (generic address space = pointer type 0),
but shared memory LayoutTensors have pointer type 3. All conversion approaches fail:
- `rebind` — fails: `pointer<none, 3>` does not match `pointer<none>`
- `.ptr` + constructor — fails: `LegacyUnsafePointer[..., SHARED]` cannot convert to `LegacyUnsafePointer[...]`
- No `addrspacecast` intrinsic available in Mojo

**Workaround**: Rely on GPU L1/L2 cache for model data (read-only, ~3KB, fits in L1).
Shared memory works fine for intra-kernel reductions (see `deep_agents/ppo/kernels.mojo`)
but cannot be passed to sub-functions due to this limitation.

**Status**: Reverted. Requires Mojo language support for address space casting.

### ✅ Item 4: Mass Matrix Subtree Mask Precomputation (2025-03-09)

**Problem**: `compute_mass_matrix_full_gpu` called `_is_descendant_gpu(model, k, body_i)`
in the inner loop of the NV×NV mass matrix computation. Each call walked the parent
chain from body `k` up to the root — O(depth) per call, O(NV² × depth) total.

**Solution**: Pre-compute a `subtree_mask[NBODY × NBODY]` boolean array at the start
of the function. `subtree_mask[k * NBODY + parent] = True` for all ancestors of body `k`.
Inner loop now uses `if not subtree_mask[k * NBODY + body_i]: continue` — O(1) lookup.

For HalfCheetah (NBODY=8): replaces up to 81 × 4 = 324 parent-chain walks with
64 boolean lookups.

**Files modified**:
- `physics3d/dynamics/mass_matrix.mojo` — `compute_mass_matrix_full_gpu` function

**Verification**: All 4 mass matrix CPU vs GPU tests pass (max error ~4e-7).

### ✅ Item 5: Contact Detection Kernel Extraction (2025-03-09)

**Problem**: Contact detection was inline in `step_kernel`, making the kernel larger
and preventing future parallelization of contact checks across geom pairs.

**Solution**: Extracted contact detection into a separate `contact_detection_kernel`
launched between `step_kernel` and the solver. The physics pipeline is now 4 kernels:

```
step_kernel → contact_detection_kernel → solve_gpu → step_finalize_kernel
```

In `step_kernel`, contact detection is replaced with contact count zeroing:
```mojo
state[env, meta_off_c + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](0)
```

The `contact_detection_kernel` calls `detect_contacts_auto_gpu()` after FK is
complete (FK runs in step_kernel, so xpos/xquat are ready).

**Files modified**:
- `physics3d/integrator/euler_integrator.mojo` — extracted `contact_detection_kernel`,
  modified `step_kernel` (removed contact detection, added count zeroing),
  modified `step_gpu` (added contact kernel launch)
- `physics3d/tests/bench_fused_kernel.mojo` — added contact kernel timing

**Verification**: All 6 contact CPU vs GPU tests pass. All 6 full-step-with-contact
CPU vs GPU tests pass.

**Note**: Currently 1 thread per env (same parallelism as before, just structurally
extracted). Future work: 2D kernel with per-geom-pair parallelism + atomics for
contact count.

---

## What MuJoCo Warp Does

MuJoCo Warp (NVIDIA, Warp framework) parallelizes physics simulation on GPU using:

### 1. Multi-World Batching (same as us)
Primary parallelization axis: `(nworld, ...)` for all arrays and kernels.

### 2. Kinematic Chain Decomposition
Instead of 1 thread per world for FK, launches `(nworld, nbranch)`:
- Pre-computes independent kinematic chains (branches) from leaf nodes
- Each thread processes one branch sequentially from root to leaf
- For HalfCheetah: 2 branches (back leg chain, front leg chain)
- Bodies shared between branches (e.g. torso) are computed by the first branch

### 3. Tree-Level Parallelism for CRB
Composite rigid body inertia uses bottom-up accumulation:
- Groups bodies by tree depth (`body_tree[level]`)
- Launches one kernel per level, parallelized across bodies at that level
- Uses `wp.atomic_add()` when multiple children update the same parent
- Same pattern for subtree center-of-mass computation

### 4. Per-DOF Mass Matrix
Mass matrix computed with one thread per DOF:
- Sparse variant stores only lower triangle (CSR-like)
- Dense variant fills full NV×NV matrix
- Ancestor walk for off-diagonal terms: `while dofid >= 0: dofid = dof_parentid[dofid]`

### 5. Tiled Factorization
Cholesky/LDL factorization uses tiled kernels:
- `TILE_SIZE = 16-32` for block operations
- `wp.launch_tiled()` with configurable `block_dim`
- Cache-efficient for dense matrix blocks

### 6. Per-World Solver Convergence
Constraint solver tracks convergence per world:
- `wp.capture_while(nsolving, while_body=solver_iteration)`
- CUDA graph with conditional loop (avoids redundant work for converged envs)
- Our Newton solver already exits early per-env but launches fixed grid

### 7. Configurable Block Dimensions
Every kernel launch has a tunable `block_dim` parameter per operation type:
```python
class BlockDim:
    euler_dense: int = 32
    actuator_velocity: int = 32
    crb: int = 32
    ...
```

---

## Remaining Optimization Plan (NVIDIA-focused)

### Priority 1: Multi-Thread Step Kernel (Highest Impact)

**Goal**: Break the step kernel (73% of physics time on RTX 4090) from
1-thread-per-env to N-threads-per-env, increasing GPU occupancy.

**Current step kernel pipeline** (all serial per env):
```
1. Forward kinematics     — walk kinematic tree (NBODY=8 iterations)
2. Body velocities        — walk tree again
3. (contact count zeroed) — contacts now in separate kernel
4. Compute cdof           — per-DOF spatial axes (NV=9)
5. Composite inertia      — bottom-up tree walk (NBODY=8)
6. Mass matrix (CRBA)     — NV×NV=81 elements, subtree mask lookup
7. LDL factorize + M_inv — O(NV³)=729 ops
8. Bias forces (RNE)      — two tree passes (forward + backward)
9. Passive forces         — per-joint loops
10. LDL solve             — O(NV²)=81 ops
```

**Approach**: Use 2D blocks `(envs_per_block, STEP_THREADS)` where STEP_THREADS
is tuned per robot. For HalfCheetah, start with STEP_THREADS=8 or 16.

#### Phase 1A: Per-DOF Parallelism

Parallelize steps 4, 6, 9, 10 across DOFs (NV=9 for HalfCheetah):

- **cdof computation** (step 4): Each thread computes 1-2 DOFs' spatial axes.
  Currently a loop over NJOINT. Independent per DOF.
- **Mass matrix diagonal+column** (step 6): Each thread computes one row/column
  of M using its DOF's cdof and crb. Off-diagonal terms require ancestor walk
  but each row is independent.
- **Passive forces** (step 9): Each thread handles damping/stiffness/frictionloss
  for 1-2 joints. Fully independent per joint.
- **fnet = qfrc - bias** (between steps 8-9): Elementwise, trivially parallel.

**Files to modify**:
- `physics3d/integrator/euler_integrator.mojo` — `step_kernel` method
- `physics3d/dynamics/mass_matrix.mojo` — add `compute_mass_matrix_full_gpu_mt()` variant
- `physics3d/dynamics/jacobian.mojo` — add `compute_cdof_gpu_mt()` variant

**Expected impact**: Moderate for HalfCheetah (NV=9 doesn't fill many threads),
larger for Humanoid (NV=27) and Ant (NV=14).

#### Phase 1B: Tree-Level Parallelism for FK and CRB

Parallelize the tree walks in FK (step 1) and CRB (step 5):

- **FK**: Decompose kinematic tree into branches. HalfCheetah has 2 independent
  branches (back leg, front leg). Each branch processed by 1 thread.
  Torso (root) processed by thread 0 before branching.
- **CRB**: Bottom-up accumulation by tree level. Bodies at the same level are
  independent. HalfCheetah has 4 levels: [torso], [bthigh,fthigh,head],
  [bshin,fshin], [bfoot,ffoot]. Max parallelism = 3 bodies at level 1.

**Data structures needed**:
- `body_branches[nbranch+1]`: start indices into branch body list (comptime)
- `branch_bodies[nbody]`: body IDs ordered by branch (comptime)
- `body_tree_level[nbody]`: tree depth per body (comptime)
- `body_level_start[max_depth+1]`: start indices per level (comptime)

Since the kinematic tree is known at compile time, all branch decomposition
can be computed as `comptime` values.

**Files to modify**:
- `physics3d/model/model_def.mojo` — add branch/level decomposition to ModelDef
- `physics3d/kinematics/forward_kinematics.mojo` — add `forward_kinematics_gpu_mt()`
- `physics3d/dynamics/jacobian.mojo` — add `compute_composite_inertia_gpu_mt()`

**Expected impact**: Low for HalfCheetah (only 2 branches, 8 bodies), higher
for Humanoid (4+ branches, 14 bodies) and Ant (4 branches, 14 bodies).

#### Phase 1C: Parallel Contact Detection (structural work done)

Contact detection has been extracted to a separate kernel (Item 5 above).
Next step: convert from 1-thread-per-env to per-geom-pair parallelism.

- Launch 2D kernel `(envs, geom_pairs)` instead of 1D `(envs)`
- Each thread handles one geom pair check
- Use atomics for thread-safe contact count increment
- For HalfCheetah: ~15 geom pairs × cost of capsule-plane or capsule-capsule check

**Files to modify**:
- `physics3d/integrator/euler_integrator.mojo` — `contact_detection_kernel`
- `physics3d/collision/broadphase_sap.mojo` — add `detect_contacts_auto_gpu_mt()`
- `physics3d/collision/contact_detection.mojo` — parallel narrowphase

**Expected impact**: Low on RTX 4090 (contact is only 5.1% / 33μs). More relevant on
M1 Pro (14.5% / 468μs) and for robots with more geoms (Ant has 14 geoms).

### Priority 2: Reduce Newton Solver Register Pressure

**Goal**: Enable full kernel fusion (step + solve + finalize) by reducing the
Newton solver's register usage from ~1000+ to a manageable level.

**Current problem**: The Newton solver uses massive InlineArrays:
- `H[NV×NV]`, `L_chol[NV×NV]`, `M_local[NV×NV]` = 3×81 = 243 registers
- `Jn_c[MC×NV]`, `Jt1_c[MC×NV]`, `Jt2_c[MC×NV]` = 3×180 = 540 registers
- Plus ~20 more InlineArrays for forces, jar, search direction, etc.
- Total: ~1000+ registers per thread

This makes full kernel fusion impossible (GPU compiler crashes with
"compile offload result missing kernelId").

**Approach**: Move loop-invariant data from InlineArrays to workspace memory:
- Jacobians (Jn, Jt1, Jt2) are already in workspace — remove the local caches
- Keep only iteration-local arrays (qacc, search, grad) in registers
- Accept some performance regression from extra workspace reads
- This trades register pressure for memory bandwidth

**Files to modify**:
- `physics3d/solver/newton_solver.mojo` — `solve_gpu` method (lines 1108-1600)

**Expected impact**: Enables kernel fusion which saves ~10ms per rollout on
RTX 4090 (negligible) but ~1.2s per rollout on Apple Metal (significant).
Main value is enabling future optimizations that need a single kernel.

### Priority 3: Batch Size Scaling

**Observation**: Physics time barely changes from BATCH=64 (926ms) to BATCH=1024
(1768ms) — only 1.9× for 16× more envs. This is great and means we can
increase batch sizes for better PPO statistics without proportional cost.

But the step kernel shows diminishing returns:
- BATCH=256: 498 μs/substep
- BATCH=1024: 530 μs/substep (only 6% slower for 4× more envs)

This suggests the step kernel is already memory-bandwidth limited at higher
batch sizes (each env reads ~800 floats of workspace + ~200 floats of state).

**Approach**: Optimize memory access patterns:
- Coalesce workspace reads (currently scattered offsets)
- Consider Structure-of-Arrays (SoA) layout for state buffers
- Shared memory for model parameters — blocked by Mojo address space limitation
  (see Item 3 above), revisit when Mojo adds address space casting

---

## Architecture Decision: Multi-Thread Step Kernel Design

Two approaches for adding intra-env parallelism to the step kernel:

### Option A: 2D Block with Barriers (Recommended)

Use 2D blocks `(envs_per_block, STEP_THREADS)`:
```
block_dim = (ENVS_PER_BLOCK, STEP_THREADS)
grid_dim = (ceil(BATCH / ENVS_PER_BLOCK), 1)
```

Phases that need all threads use `barrier()` for synchronization:
```
thread 0: FK root
barrier()
threads 0..1: FK branches (2 branches for HalfCheetah)
barrier()
threads 0..NV-1: cdof computation
barrier()
thread 0: CRB accumulation (tree walk, hard to parallelize for small trees)
barrier()
threads 0..NV-1: mass matrix rows
barrier()
thread 0: LDL factorize + M_inv
barrier()
threads 0..NV-1: bias forces terms
barrier()
thread 0: LDL solve + qacc
```

**Pros**: Clean synchronization, no redundant work.
**Cons**: Many barriers (latency), thread waste during serial phases.
**Best for**: Larger robots (NV > 16) where parallel phases dominate.

### Option B: Replicated Computation (Simpler)

Keep 1D blocks but add optional parallelism within specific functions:
- Mass matrix: each env still 1 thread, but vectorize inner loops with SIMD
- Contact detection: launch as separate 2D kernel (already extracted — Item 5)
- FK/CRB: keep serial (small trees not worth parallelizing)

**Pros**: No barriers, no thread waste, simpler to implement.
**Cons**: Limited speedup, doesn't address root cause.
**Best for**: Small robots like HalfCheetah (NV=9).

### Recommendation

Start with **Option B** (vectorize inner loops, optimize memory access) since
HalfCheetah has a small kinematic tree where barrier overhead would dominate
any parallelism gains. Move to **Option A** when targeting Humanoid (NV=27).

---

## Implementation Status

| # | Task | Status | Impact | Notes |
|---|------|--------|--------|-------|
| 1 | Fix Phase 3 training timings | ✅ Done | Correct metrics | `PROFILE_PHASE3` comptime flag |
| 2 | GPU-side reset optimization | ✅ Done | Save ~30s/training | `workspace_ptr` reuse |
| 3 | Shared memory for model params | ❌ Blocked | 10-20% step speedup | Mojo lacks address space casting |
| 4 | Mass matrix subtree mask | ✅ Done | O(1) vs O(depth) lookups | CPU vs GPU tests pass |
| 5 | Contact detection kernel extraction | ✅ Done | Enables parallelization | 4-kernel pipeline, tests pass |
| 6 | Per-DOF mass matrix computation | Planned | 10-20% for large robots | Phase 1A |
| 7 | FK branch decomposition | Planned | Significant for Humanoid | Phase 1B |
| 8 | Parallel contact pairs | Planned | ~14% of physics time | Phase 1C (structural work done) |
| 9 | Solver register reduction + fusion | Planned | Enables fusion (Apple) | Priority 2 |
