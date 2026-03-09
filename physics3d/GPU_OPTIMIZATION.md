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

| Kernel | M1 Pro | RTX 4090 | Speedup | % on 4090 |
|--------|--------|----------|---------|-----------|
| Step (FK, M, bias, qacc) | 3248 μs | 498 μs | 6.5× | **78%** |
| Solve (Newton) | 1251 μs | 64 μs | 19.5× | 10% |
| Finalize (integration) | 837 μs | 76 μs | 11× | 12% |
| **Total substep** | **5336 μs** | **638 μs** | **8.4×** | |

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

## Optimization Plan (NVIDIA-focused)

### Priority 1: Multi-Thread Step Kernel (Highest Impact)

**Goal**: Break the step kernel (78% of physics time) from 1-thread-per-env to
N-threads-per-env, increasing GPU occupancy.

**Current step kernel pipeline** (all serial per env):
```
1. Forward kinematics     — walk kinematic tree (NBODY=8 iterations)
2. Body velocities        — walk tree again
3. Contact detection      — loop over geom pairs
4. Compute cdof           — per-DOF spatial axes (NV=9)
5. Composite inertia      — bottom-up tree walk (NBODY=8)
6. Mass matrix (CRBA)     — NV×NV=81 elements, tree walk
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

#### Phase 1C: Parallel Contact Detection

Currently contact detection loops over all geom pairs sequentially per env.

- Each thread handles a subset of geom pair checks
- Use thread-local contact buffer + atomic write to shared contact count
- For HalfCheetah: ~15 geom pairs × cost of capsule-plane or capsule-capsule check

**Files to modify**:
- `physics3d/collision/broadphase_sap.mojo` — add `detect_contacts_auto_gpu_mt()`
- `physics3d/collision/contact_detection.mojo` — parallel narrowphase

**Expected impact**: Moderate. Contact detection is not the biggest part of the
step kernel but scales poorly with more geoms (Ant has 14 geoms).

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
- Use shared memory for model parameters (read by all envs)
- Consider Structure-of-Arrays (SoA) layout for state buffers

### Priority 4: Reset Optimization

The reset operation takes 31.8s (17.5% of collect phase). Currently:
- CPU-side reset logic per terminated env
- Re-copies initial state to GPU
- Re-zeros workspace

**Approach**: Implement GPU-side reset kernel:
- Each thread resets one env's state to initial values directly on GPU
- Avoid round-trip to CPU for reset
- Only reset envs that actually terminated (sparse reset)

**Files to modify**:
- `envs/phyics3d_env.mojo` — GPU reset path

### Priority 5: Phase 3 Training Timing Fix

**Problem**: The per-kernel training timings (Section 3 of PPO output) are
misleading because only the critic optimizer timer includes `ctx.synchronize()`.
All other sub-phase timers measure kernel enqueue time only (~300-600μs each),
while the critic optimizer timer captures all accumulated GPU work (33s).

**Fix**: Add `ctx.synchronize()` before each sub-phase timer start, OR use
GPU event-based profiling (record events around each kernel, query elapsed time
after sync). The event approach avoids pipeline stalls from excessive sync.

**Reference**: MuJoCo Warp uses `@event_scope` decorators that record GPU events
before/after each function, building a hierarchical timing tree without sync
overhead.

**Files to modify**:
- `deep_agents/ppo/ppo_continuous_old.mojo` — training loop timing

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
- Contact detection: launch as separate 2D kernel
- FK/CRB: keep serial (small trees not worth parallelizing)

**Pros**: No barriers, no thread waste, simpler to implement.
**Cons**: Limited speedup, doesn't address root cause.
**Best for**: Small robots like HalfCheetah (NV=9).

### Recommendation

Start with **Option B** (vectorize inner loops, optimize memory access) since
HalfCheetah has a small kinematic tree where barrier overhead would dominate
any parallelism gains. Move to **Option A** when targeting Humanoid (NV=27).

---

## Implementation Order

| # | Task | Expected Impact | Effort |
|---|------|----------------|--------|
| 1 | Fix Phase 3 training timings | Correct metrics | Low |
| 2 | GPU-side reset kernel | Save ~30s/training (17% of collect) | Medium |
| 3 | Shared memory for model params | 10-20% step kernel speedup | Medium |
| 4 | Vectorize mass matrix inner loops | 5-15% step kernel speedup | Medium |
| 5 | Parallel contact detection kernel | 5-10% step kernel speedup | Medium |
| 6 | Per-DOF mass matrix computation | 10-20% for large robots | High |
| 7 | FK branch decomposition | Significant for Humanoid | High |
| 8 | Solver register reduction + fusion | Enables fusion (Apple benefit) | High |

Focus on items 1-5 first. They are independent and provide compounding speedups
with moderate implementation effort.
