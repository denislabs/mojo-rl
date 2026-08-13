# physics3d CPU performance — where the time goes, and what is left

Measured 2026-08-13, revised 2026-08-14, on Apple M1 Pro, `float32`, single
env, against MuJoCo 3.10.0 (`float64`) stepping the same two XMLs.

The short version: **the gap to MuJoCo is algorithmic, not Mojo-vs-C.** Every
phase where we run the same algorithm runs at MuJoCo's speed or faster, and all
of the gap is collision.

⚠⚠ **THE 2026-08-13 REVISION GOT THE CAUSE WRONG AND §4 IS ITS RETRACTION.** It
named a missing mid-phase BVH as the largest prize, from a node count that is
99.99% mesh-face BVH the collision path never reads. What the gap actually was:
one linear scan that should have been a hill climb (§3), one silently truncated
model (§5.2), and — all that is left — the cost of a single support query
(§6.1). Two optimisations aimed at the broadphase measured exactly zero, twice,
because the broadphase is 0.91 µs (§5.1).

---

## 1. Headline

Per **physics** step (env step ÷ `FRAME_SKIP=10`), `float32`, against MuJoCo
3.10.0 (`float64`) stepping the same two XMLs.

| model | before | after | speedup |
|---|---|---|---|
| SO-ARM100 | 15.42 s | **9.47 s** | **1.63×** |
| SO-ARM101 | 20.33 s | **15.85 s** | **1.28×** |

(40 000 env steps = 400 000 physics steps; MIN of 4 interleaved rounds against
a pristine worktree. ⚠ SO-ARM100's "after" carries **two more collision meshes
than its "before"** — see §5.2 — so it is doing strictly more work.)

⚠ **WE ARE STILL SLOWER WHILE CARRYING HALF THE PRECISION.** These runs are
`float32`; MuJoCo is `float64` throughout. The honest gap is worse than the
ratio, not better.

Both models are `nq = nv = nu = 6`, `nbody = 8`, and sit at `ncon` 1 and 0 — so
what is compared is almost entirely **the cost of proving that geoms are
apart**, not the cost of resolving contact.

---

## 2. What landed, and what it was worth

| change | SO-ARM100 | SO-ARM101 |
|---|---|---|
| plane-mesh support point: full argmin → hill climb (§3) | 1.21× | 1.28× |
| `<mesh>` asset cap silently truncating the model (§5.2) | 1.63× cumulative | n/a (13 assets) |
| per-pair static filter decode hoisted out of the sweep | **reverted, 0** | **reverted, 0** |

---

## 3. `_plane_mesh_contacts` was scanning every hull vertex

MuJoCo's `mjc_PlaneConvex` calls `mjccd_support` — the hill climb — to find the
deepest vertex, then walks only that vertex's hull-edge neighbours
(`engine_collision_convex.c:1010`). Ours hill-climbed for the *neighbours* but
took a **full linear argmin over every hull vertex** for the support point
itself. The docstring said so in plain sight.

Measured with per-stage timers over 20 000 physics steps:

| stage | SO-ARM100 | SO-ARM101 |
|---|---|---|
| 1 world poses | 0.23 | 0.24 |
| 2 AABBs + pair margins | 0.11 | 0.10 |
| **3 plane loop** | **8.06** | **11.12** |
| 4 SAP sweep + narrow phase | 23.34 | 36.26 |

On SO-ARM101 that 11.12 µs was **one call per step**, scanning one ~4 000-vertex
hull — 23–25% of the whole physics step on both arms. After the change: **1.29
µs** and **0.29 µs**.

Minimising height above a plane *is* a support query: height is
`p_z + dot(v, Rᵀe_z)`, so the lowest vertex maximises `dot(v, Rᵀ(0,0,−1))`. The
hill climb is exact from any start vertex (a local maximum of a linear
functional on a convex polytope's 1-skeleton is global), so this is a pure
speed change — and it moves us *toward* the reference, not away.

⚠ ONE TIE-BREAK CHANGES. On an exact plateau (a facet lying flat on the plane)
the climb stops at the first local maximum, where the argmin took the lowest
index. `best_h` is identical either way, but the up-to-two EXTRA contacts are
drawn from that vertex's neighbours, so which extras appear can differ. The
Jaco plane-mesh contact-set gate against MuJoCo passes unchanged.

---

## 4. ⚠ RETRACTED: "MuJoCo has a mid-phase BVH worth 5–12×"

**The previous version of this document was wrong, and this was its main
recommendation.** It reported that MuJoCo builds 123 136 BVH nodes for
SO-ARM100 and 645 136 for SO-ARM101 while we build none, and ranked
implementing one as the largest available prize. Splitting that node count by
consumer kills the claim:

| model | body-geom BVH nodes | mesh-face BVH nodes |
|---|---|---|
| SO-ARM100 | **30** | 123 106 |
| SO-ARM101 | **21** | 645 115 |

`mj_collideTree` — the mid-phase — descends `body_bvhadr`, and that is the
30-and-21 column. Six of SO-ARM100's eight bodies have a **single** node, i.e.
no pruning at all. The other 99.99% are `mesh_bvhadr`, a BVH over mesh *faces*
whose only consumers in the source are `engine_ray.c`, `engine_collision_sdf.c`
and the visualiser. **Both models have zero SDF geoms and cast no rays**, so
the collision path never touches those nodes.

⚠ **AND THE PAIR COUNTS SAY THE MID-PHASE IS NOT WHERE THE GAP IS.** Replicating
`filterBodyPair` + `mj_filterSphere` on MuJoCo's own per-step state and
comparing against our stage counters:

| | static pairs, MuJoCo / ours | reaching narrow phase, MuJoCo / ours |
|---|---|---|
| SO-ARM100 | 65 + 17 / **65 + 17** | 2.0 + 5.0 / **2.02 + 4.97** |
| SO-ARM101 | 45 + 13 / **45 + 13** | 4.0 + 1.0 / **4.0 + 1.0** |

Our broadphase now selects **the same pairs MuJoCo does, pair for pair, on both
models**. There is no pruning left to win. (Before §5.2 it was 11.0 + 6.97 on
SO-ARM100 — that gap was a model bug, not a missing acceleration structure.)

---

## 5. Splitting the 15 µs bucket — and what was hiding in it

The previous §5 flagged 15.2 / 16.4 µs charged to `detect_contacts_sap`, which
`sample` reports as a leaf, and called splitting it the prerequisite for
everything else. That was right. Two instruments did it: stage counters written
into a widened `smeta`, and `perf_counter_ns` around each stage.

### 5.1 ⚠ The pair loop was NOT the bucket — the earlier inference was wrong

The counters showed ~465 candidate pairs per step on **both** arms, with the
body/weld/contype filter — pure static model data — discarding 86% and 90% of
them. The x-sweep rejects 8 of 496 possible pairs on SO-ARM100 and **none** on
SO-ARM101 (an arm is a compact object; every geom overlaps every other in x),
and the y/z AABB test rejects 4% and 0%.

That looked conclusive, and it was wrong. Hoisting the static decode to once
per geom measured **15.60 → 15.72 s over 5 interleaved rounds** — nothing —
and was reverted. The ablation says why: stubbing the geom-geom narrow phase
leaves the **entire sweep** — 487 iterations, 466 AABB tests, 65 filter and mix
evaluations, 65 bounding-sphere tests — at **0.91 µs/step**.

⚠ **THE BROADPHASE IS 0.91 µs AND WAS NEVER THE PROBLEM.** Two separate
optimisations aimed at it (this one, and the `mix_contact_params` hoist in the
previous revision) both measured zero, because there is under a microsecond
there to win. The invariance that pointed at the pair loop — the bucket being
~15 µs on both arms while `_plane_mesh_contacts` ran 7×/step on one and 1× on
the other — was a coincidence of two different costs summing alike.

### 5.2 The `<mesh>` asset table was silently truncated at 16

`full_parser.mojo` parsed `<mesh>` assets under `while mesh_count < 16`.
SO-ARM100 declares **18**, so `Moving_Jaw_Collision_2` and `_3` never entered
the asset table. A mesh geom whose name does not resolve keeps `mesh_id = -1`,
which fails silently in every direction:

- no hull is built, so **the geom has no collision geometry at all**;
- `rbound` keeps its per-type fallback — `gd.radius`, i.e. MuJoCo's default
  size **0.5** — against MuJoCo's 0.0279 and 0.0309, **16–18× too large**.

The visible symptom was performance, not a missing contact: two bounding
spheres that swallow the whole arm let **11 pairs per step into GJK where
MuJoCo narrow-phases 2**. After the fix both `rbound` values match MuJoCo to
six digits, the two hulls load at 8 and 187 vertices (MuJoCo: 8 and 187), and
the call counts match exactly (§4).

⚠ THE CAP WAS NOT `MAX_GPU_MESHES` AND MUST NOT BE CONFUSED WITH IT. That limit
is on **loaded, collidable** meshes; this was the XML's **asset table**, most of
which is usually visual-only. SO-ARM100 loads 8 collidable meshes out of 18
declared — nowhere near the real limit when this truncated it. `fields_build`
had the same silent `break` on `MAX_GPU_MESHES`; it now prints an error.

⚠ `NMESH_VERTS` HAD TO RISE 2560 → 2746, and that number had been *measured* —
with the two meshes absent. A capacity constant calibrated against a model that
is silently missing part of itself is a budget for the wrong model.

Among the repo's own baked models only SO-ARM100 exceeds 16 assets (SO-ARM101
has 13, the dm_control manipulation set 9). Menagerie trees parsed at runtime
go far higher — `trossen_wxai` 125, `flybody` 85 — so anything ported from
there was affected.

---

## 6. Where the time is now, and the levers that are left

Final stage split, 20 000 physics steps (⚠ stage timers are optimisation
barriers and add ~2%; use these for proportions, not absolutes):

| stage | SO-ARM100 | SO-ARM101 |
|---|---|---|
| world poses + AABBs | 0.52 | 0.49 |
| plane loop | 1.44 | 0.29 |
| SAP sweep proper (ablation, §5.1) | 0.91 | ~0.9 |
| **GJK/EPA** | **11.21** (2.02 calls, 5.56 µs ea) | **34.94** (4 calls, 8.74 µs ea) |
| `multi_ccd_extra_contacts` (§6.2) | ~5.2 | — (`ncon` 0, never runs) |
| contact emission + primitive branches | ~3.6 | ~8.5 |

### 6.1 GJK per-call cost is now the whole story

We run **the same number of convex calls as MuJoCo on the same pairs** (§4),
and MuJoCo's entire narrow phase — sphere rejects included — is 2.57 µs
(SO-ARM100) and 2.77 µs (SO-ARM101). Ours is ~20 µs and ~43 µs. That is
**~8–16× per call**, and it is not explained by hull size: our hulls total
2 746 and 33 280 vertices against MuJoCo's 15 689 and 50 162.

Where it goes, from a de-inlined build under `sample` (SO-ARM101):

| symbol | % of physics step |
|---|---|
| `gjk::hillclimb_support_index` | **54.1** |
| `broadphase_sap::detect_contacts_sap` | 17.9 |
| `gjk::gjk_epa_witness` | 10.8 |
| `gjk::_support` / `_support_mesh` | 8.4 |

**The support walk is 62% of the step.** The open question is whether that is
*many* queries or *long* walks per query — the counters to answer it are a step
counter threaded through `hillclimb_support_index`, which is the next thing to
build. Calibration measured so far: disabling the intra-call warm start costs
**16.45 → 21.38 s** (+30%), so walks are long enough for the seed to matter,
but the first query is only one of ~N per call, which caps a cross-step warm
cache at well under that.

### 6.2 The rest of narrow phase is `multi_ccd`, and it is §6.1 again

Separated by ablation, SO-ARM100, total `detect_contacts_sap` per step:

| build | total |
|---|---|
| as shipped | 22.90 µs |
| `MC_ENABLED = False` (native multicontact off) | 22.28 µs — **not it** |
| `multi_ccd_extra_contacts` stubbed out | **17.74 µs** |

So `multi_ccd_extra_contacts` is **~5.2 µs/step for the ONE contacting pair**,
and native multicontact is noise. That is not waste — MuJoCo runs multi-CCD too
(`mjDSBL_MULTICCD` is off by default in the 3.10.0 runtime, see
`collision/multi_ccd.mojo`) — but it works by re-running `gjk_epa` up to four
more times with perturbed directions. **It is the support walk again**, at 4×
the multiplier, which is why it does not appear in the GJK call counter (a
different call site) and why fixing §6.1 fixes this too.

⚠ THE ABSOLUTE µs MOVE BETWEEN ABLATION BUILDS — removing a large inlined block
changes register allocation in the enclosing function, and `gjk_epa_witness`'s
own measured cost fell 11.2 → 8.3 µs for the *same 2.02 calls* when multi-CCD
was stubbed. Read the TOTAL row, not the sub-rows, across builds.

### 6.3 Temporal coherence: a per-pair separation cache

Unchanged from the previous revision, and now the only structural idea left.
GJK already returns a **distance** and we already carry per-pair warm state. If
a pair was 17 cm apart last step and a bound on how far the geoms can have moved
is smaller than that, the pair cannot be touching and needs no narrow phase.

⚠ MAKE IT CONSERVATIVE, NOT APPROXIMATE. Bound closing speed from body
velocities and bounding radii, subtract `dt ×` that bound, and skip only while
the result stays positive. Built that way it can lose speed, never a contact.
MuJoCo does **not** do this, so it is a place to go faster than the reference
rather than catch up.

### 6.4 Newton iteration count on SO-ARM100

Still unmeasured. MuJoCo converges in one iteration on both models
(`d.solver_niter == 1`). Worth one counter.

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

The hill climb in `collision/gjk.mojo` is where narrow-phase time goes —
`hillclimb_support_index` alone is **54% of the SO-ARM101 physics step**
(§6.1) — and it is hostile to vectorisation on three counts at once:

- **Gather, not contiguous load.** Neighbours come from `mesh_edges`, so the
  vertex reads are `mesh_verts[nb, 0..2]` at scattered `nb`.
- **Loop-carried dependency.** `best_dot` feeds the next comparison; the walk is
  serial by construction.
- **Three-wide work.** A `Vec3` dot product on a 4-lane register wastes a lane
  before any of the above.

This loop is **latency-bound, not throughput-bound**. Expect ~0 from SIMD here.
The linear-scan fallback below it *is* contiguous and vectorisable — and by
construction only runs for meshes under `_HILLCLIMB_MIN = 10` vertices.

### ⚠ 7.3 There is no longer a candidate worth vectorising

The previous revision named the broadphase pair loop as "the one good fit" —
hundreds of independent pair tests, pure arithmetic, no dependencies — and
proposed an SoA geom layout to feed it. **§5.1 killed that: the entire sweep is
0.91 µs/step.** An SoA rewrite of the geom tables to vectorise a stage that
costs under a microsecond cannot repay itself, and the two scalar optimisations
already aimed at the same loop both measured zero.

That leaves nothing:

- **the support walk** — 62% of the step — is latency-bound and un-vectorisable
  (§7.2);
- **the broadphase** is 0.91 µs;
- **dynamics** (CRBA, LDL, Jacobians) is contiguous and vectorisable and is
  **1.2 µs of the step** — Amdahl caps the whole category at ~3%.

⚠ **DO THE AMDAHL ARITHMETIC FIRST, AND THIS TIME IT SAYS DON'T.** SIMD is not
a lever on this workload as it stands. It becomes one only if §6.1 turns the
support query into a bulk operation — e.g. evaluating a whole neighbour ring
per step rather than one vertex at a time, which *is* a gather but is at least
wide.

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

⚠ **`sample` CANNOT SEE INSIDE `gjk_epa_witness`** — the support functions, the
simplex and EPA are all inlined into it, so it reports as a leaf holding 72% of
the step. To break it open, mark `hillclimb_support_index` / `_support_mesh` /
`_support` `@no_inline` **in a throwaway worktree** and re-sample; that is where
the 54% in §6.1 comes from. It changes codegen, so use it for proportions only.

**Stage counters and stage timers** — what actually split §5's bucket, and
neither is in the repo (both are throwaway instrumentation):

1. Widen `METADATA_SIZE` (`gpu/constants.mojo`) from 8 to ~40 in a worktree.
   Everything that allocates `smeta` sizes from that constant, so slots 8+
   become free scratch reachable from `_detect_contacts_sap_env` with **no
   signature changes** — which matters, because Mojo nightly has no
   module-level mutable global to hang a counter on.
2. Increment `smeta[env, k]` at each filter stage for counts, or bracket each
   stage with `perf_counter_ns()` for times, then read `e.d.meta.data[k]` after
   the rollout.

⚠ **ACCUMULATE MICROSECONDS, NOT NANOSECONDS.** `smeta` is the model dtype;
float32's 24-bit mantissa stops resolving unit increments past ~1.7e7, and a
nanosecond total over 20 000 steps sails past that — the counter silently stops
advancing rather than overflowing.

⚠ **STAGE TIMERS ARE OPTIMISATION BARRIERS.** They prevent the compiler sinking
work across a stage boundary, so they measure a slightly different build.
Proportions are trustworthy; absolutes are not.

⚠ **PREFER ABLATION TO INFERENCE FOR THE LAST STEP.** The counters said the pair
loop dominated and that was wrong (§5.1). Stubbing the stage and re-timing is
what settled it — an ablation answers "how much does this cost" directly, where
a counter only answers "how often does this run".

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
