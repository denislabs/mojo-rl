# physics3d CPU performance — where the time goes, and what is left

Measured 2026-08-13 on Apple M1 Pro, `float32`, single env, against MuJoCo
3.10.0 (`float64`) stepping the same two XMLs.

The short version: **the remaining gap to MuJoCo is algorithmic, not Mojo-vs-C.**
Every phase where we run the same algorithm runs at MuJoCo's speed or faster.
All of the gap is collision, and most of the collision gap is one missing
acceleration structure.

---

## 1. Headline

Per **physics** step (env step ÷ `FRAME_SKIP=10`):

| model | ours | MuJoCo | ratio |
|---|---|---|---|
| SO-ARM100 | 34.6 µs | 7.8 µs | 4.4× |
| SO-ARM101 | 45.5 µs | 12.3 µs | 3.7× |

⚠ **WE ARE 4× SLOWER WHILE CARRYING HALF THE PRECISION.** These runs are
`float32`; MuJoCo is `float64` throughout. The honest gap is therefore worse
than 4×, not better.

Both models are `nq = nv = nu = 6`, `nbody = 8`, and sit at `ncon` 1 and 0
respectively — so what is being compared is almost entirely **the cost of
proving that geoms are apart**, not the cost of resolving contact.

---

## 2. Phase breakdown

Ours from `sample` on a built binary, attributed exclusively (every sample
charged to the deepest phase on its stack, so phases partition the step).
MuJoCo from its own `mjTIMER_*` counters, calibrated against wall clock.

### SO-ARM100 — 34.6 µs vs 7.8 µs

| phase | ours | MuJoCo | ratio |
|---|---|---|---|
| forward kinematics + `cdof` + `subtree_com` | 0.76 | 0.60 | 1.3× |
| mass matrix (CRBA) + LDL | 0.47 | 0.16 | 2.9× |
| constraint build + solve | 3.5 | 2.0 | 1.8× |
| **collision, total** | **29.0** | **4.2** | **7.0×** |
| — broadphase + primitive narrow phase (unsplit, see §5) | 15.2 | 1.0 | — |
| — mid-phase BVH | **0** | 1.1 | — |
| — GJK/EPA | 13.8 | 2.0 | 6.8× |
| env glue (obs, reward, action) | 0.87 | — | — |

### SO-ARM101 — 45.5 µs vs 12.3 µs

| phase | ours | MuJoCo | ratio |
|---|---|---|---|
| forward kinematics + `cdof` + `subtree_com` | 0.83 | 0.61 | 1.4× |
| mass matrix (CRBA) + LDL | 0.53 | 0.16 | 3.4× |
| constraint build + solve | 1.08 | 1.85 | **0.6×** |
| **collision, total** | **42.1** | **8.6** | **4.9×** |
| — broadphase + primitive narrow phase (unsplit) | 16.4 | 0.9 | — |
| — mid-phase BVH | **0** | 5.5 | — |
| — GJK/EPA | 25.7 | 2.2 | 11.5× |
| env glue | 0.84 | — | — |

Collision is **84% / 93%** of our step against **53% / 71%** of MuJoCo's.

---

## 3. This is not a codegen gap

The parts of the engine that implement the same algorithm as MuJoCo, on the
same data, land between **0.6× and 3.4×** — and our constraint path on
SO-ARM101 is *faster* than MuJoCo's. CRBA, RNE, forward kinematics, LDL and the
Newton solve are all dense scalar float code with no acceleration structure on
either side, which makes them the clean control for the language question.
They pass.

⚠ **A SECOND, INDEPENDENT CHECK POINTS THE SAME WAY.** Both pair loops in
`broadphase_sap.mojo` called `mix_contact_params` — a 24-field read plus
MuJoCo's priority-mixing rule — *before* the bounding-sphere reject, so nearly
every rejected pair paid for parameters it discarded. That looks like free
money. Hoisting the reject above the mix and re-measuring over 12 interleaved
rounds gave **16.46 s → 16.29 s mean** on SO-ARM100, inside the round-to-round
variance, with the modified arm *slower* in 4 of the last 5 rounds. The change
was reverted.

The compiler was already sinking those dead loads past the branch. We are not
losing time to the kind of redundancy a compiler cannot see — which is exactly
what "not a codegen gap" predicts.

---

## 4. What MuJoCo has that we do not: a mid-phase BVH

MuJoCo builds a bounding-volume hierarchy per mesh, over triangles, plus a
per-body BVH over geoms (`mjModel.mesh_bvhadr`, `body_bvhadr`, `nbvh`):

| model | mesh verts | MuJoCo BVH nodes |
|---|---|---|
| SO-ARM100 | 30 172 | 123 136 |
| SO-ARM101 | 160 796 | 645 136 |

We have none. And the trade is visible in the numbers above: on SO-ARM101
MuJoCo **spends 5.5 µs** descending that tree so that its narrow phase finishes
in **2.2 µs**. Ours skips the tree and pays **25.7 µs** in GJK. MuJoCo nets 7.7 µs
against our 25.7 — it pays to save.

⚠ **THIS IS WHY THE RATIO GETS WORSE AS GEOMETRY GROWS.** SO-ARM101 carries 5×
the mesh vertices of SO-ARM100 and our GJK cost nearly doubles while MuJoCo's
narrow phase moves 2.0 → 2.2 µs. A cost that tracks model size where the
reference's does not is the signature of a missing acceleration structure, not
of slow arithmetic.

---

## 5. ⚠ THE LARGEST SINGLE BUCKET IS NOT YET ATTRIBUTED

15.2 µs (SO-ARM100) and 16.4 µs (SO-ARM101) are charged to
`broadphase_sap::detect_contacts_sap`, which `sample` reports as a **leaf** —
everything except the out-of-line `gjk_epa_witness` call is inlined into it.
That bucket therefore contains, undifferentiated:

- world pose + AABB for all ~32 geoms,
- the SAP sweep,
- the pair filters (`find_predefined_pair`, `pair_body_filtered`, contype/
  conaffinity, both bounding-sphere rejects),
- **every primitive narrow-phase branch** (sphere, capsule, box, plane-*),
  including `_plane_mesh_contacts`.

So "our broadphase is 15× MuJoCo's" **is not a claim this profile supports** —
MuJoCo's 1.0 µs `COL_BROAD` is broadphase alone. Splitting our bucket needs
counters or ablation builds and is the prerequisite for optimising it.

A hint worth following: the bucket is **nearly identical** across two models
whose hull sizes differ by 13× (15.2 vs 16.4 µs). Whatever dominates it scales
with **geom and pair count**, not with mesh size — which points at per-pair
setup rather than at any vertex loop.

---

## 6. Levers, ranked

### 6.1 Split the 15 µs bucket (prerequisite, cheap)

Counters on: candidate pairs entering the pair loop, pairs surviving each
filter, GJK calls, `_plane_mesh_contacts` vertex iterations. Or ablation
builds that stub out one stage at a time. Nothing below should be built before
this says where the time is.

### 6.2 Mid-phase BVH (largest measured prize, largest build)

Worth 5–12× on narrow phase by the §4 comparison. Also the change most likely
to disturb contact parity, so it wants the existing MuJoCo-comparison suites
green at every step. MuJoCo's own hull graph is already parsed for the hill
climb, so some of the input structure exists.

### 6.3 Temporal coherence: a per-pair separation cache (best value/cost)

GJK already returns a **distance**, and we already carry per-pair warm-start
state. If a pair was 17 cm apart last step, and a bound on how far the two
geoms can have moved since is smaller than that, the pair **cannot** be
touching and needs no narrow phase at all.

⚠ **MAKE IT CONSERVATIVE, NOT APPROXIMATE.** Bound the closing speed from the
geoms' body velocities and their bounding radii, subtract `dt ×` that bound
from the cached distance, and only skip while the result stays positive. Built
that way it is exact — it can lose speed, never a contact — which is what keeps
the parity suites meaningful. Built as a heuristic it silently drops contacts
and no existing gate would catch it.

MuJoCo does **not** do this, so it is a place where we could go faster than the
reference rather than merely catch up.

### 6.4 Hoist static geoms

Geoms welded to the world never move, yet their world pose and AABB are
recomputed every step in stage 1/2 of `_detect_contacts_sap_env`. Small, cheap,
and it attacks the §5 bucket directly.

### 6.5 SIMD / vectorisation — see §7

### 6.6 Newton iteration count on SO-ARM100

Our constraint phase is 1.8× MuJoCo's on SO-ARM100 but **0.6×** on SO-ARM101,
which is odd for 9 rows against 6. MuJoCo converges in a single iteration on
both (`d.solver_niter == 1`). Ours is unmeasured. Worth one counter.

---

## 7. SIMD: what to expect before writing any

**Today `mojo_rl/physics3d/` contains no explicit SIMD at all** — no
`load[width=W]`, no `simd_width_of`, no `vectorize`. So this is genuinely
unexplored. Three things to know before spending time on it.

### ⚠ 7.1 Mojo does NOT autovectorize — "turning on SIMD" is not a thing

Measured previously in this repo (`benchmarks/benchmark_vectorize_layouttensor.mojo`,
ReLU at 256×256, M1 Pro): `LayoutTensor[b, i]` scalar loop **23 µs**, raw
`.ptr[i]` scalar loop **21 µs** (1.1×, i.e. no vectorisation), explicit
`load[width=W]` loop **5 µs** (**4.1×**). The nightly autovectorizer does not
engage on either indexing style. Every SIMD win has to be written by hand:

```mojo
comptime W = simd_width_of[dtype]()
var i = 0
while i + W <= N:
    var v = p.load[width=W](i)
    out.store(i, op(v))
    i += W
while i < N:          # scalar tail
    out[i] = op(p[i])
    i += 1
```

`vectorize[body, W](n)` exists and works, but the closure plumbing is finicky
in nightly and it emits the same code as the explicit loop.

### ⚠ 7.2 The hottest loop in the engine is the wrong shape for SIMD

`_support_mesh`'s hill climb (`collision/gjk.mojo`) is where narrow-phase time
goes, and it is hostile to vectorisation on three counts at once:

- **Gather, not contiguous load.** Neighbours come from `mesh_edges`, so the
  vertex reads are `mesh_verts[nb, 0..2]` at scattered `nb`.
- **Loop-carried dependency.** `best_dot` feeds the next comparison; the walk is
  serial by construction.
- **Three-wide work.** A `Vec3` dot product on a 4-lane register wastes a lane
  before any of the above.

This loop is **latency-bound, not throughput-bound**. Expect ~0 from SIMD here.
The linear-scan fallback below it *is* contiguous and vectorisable — and by
construction only runs for meshes under `_HILLCLIMB_MIN = 10` vertices.

### 7.3 Where SIMD could actually pay

**The broadphase pair loop is the one good fit.** Hundreds of independent pair
tests per step, pure arithmetic, no dependencies — the textbook case. The
blocker is layout: geoms live in an AoS `LayoutTensor[NGEOM, MODEL_GEOM_SIZE]`,
so testing 4 pairs at once means 4 strided gathers. Getting real width needs
**SoA columns** (`pos_x[]`, `pos_y[]`, `pos_z[]`, `rbound[]`), at which point
one AABB or bounding-sphere test per lane is straightforward. That is a data
layout change, not a loop rewrite, and it should follow §6.1 — there is no
point vectorising a stage before knowing what fraction of 15 µs it is.

**Dynamics is contiguous and vectorisable and not worth it.** CRBA, LDL and the
Jacobians are dense small-matrix loops. They are also **1.2 µs of a 34.6 µs
step** — Amdahl caps the whole category at ~3%.

⚠ **DO THE AMDAHL ARITHMETIC FIRST.** Collision is 84–93% of the step; the hill
climb inside it cannot be vectorised; dynamics is ~3% of it. A realistic ceiling
for SIMD on the step *as it stands today* is single-digit percent. It becomes
worth real effort only **after** §6.1 identifies a vectorisable stage inside the
15 µs bucket, or alongside the SoA change in §7.3.

### 7.4 A note on the GPU path

None of this applies to `detect_contacts_sap_fields_kernel` and the batched
training path, which already get their width from `BATCH` across envs. These
CPU numbers govern the viewer, the tests and single-env rollouts.

---

## 8. How to reproduce

Both probe scripts are in `benchmarks/`.

**MuJoCo side** — per-phase timers plus work counts, calibrated to wall clock
(MuJoCo's `mjTimerStat.duration` unit does not match the docs in the Python
bindings, so the script derives the scale from a `perf_counter` loop):

```bash
pixi run python benchmarks/physics3d_mujoco_phases.py <scene.xml> 20000 [keyframe]
```

**Our side** — build a binary, sample it, attribute the tree:

```bash
pixi run mojo build -I . <bench>.mojo -o /tmp/bench
/tmp/bench & sample $! 18 1 -f /tmp/s.txt      # from the REPO ROOT
python3 benchmarks/physics3d_sample_phases.py /tmp/s.txt
```

⚠ **`mojo run` PROFILES THE JIT.** Build a binary or the sample is warmup.

⚠ **RUN THE BINARY FROM THE REPO ROOT.** Mesh assets resolve by repo-relative
path; from anywhere else the STLs fail to load, the engine prints a warning
nobody reads, and the benchmark silently measures a model with **no mesh
collision at all**. This has bitten this project before.

⚠ **THE ATTRIBUTION SCRIPT HAS THREE TRAPS BAKED INTO IT**, each of which
corrupted an earlier version: sum only the **main thread** (idle runtime workers
each carry a full window of samples); match the **symbol head only** (Mojo
mangles the entire comptime XML into the symbol, so a substring test against the
whole line matches unrelated modules); and do **not** restrict to the `step`
subtree (inlining reports much of the physics as a sibling of `step`, not a
child).

Timings are the **MIN of interleaved rounds** against a pristine `git worktree`,
never a baseline measured earlier in a session — identical code has drifted
1.4–1.7× here.
