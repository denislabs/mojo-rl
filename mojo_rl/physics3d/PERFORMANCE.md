# physics3d CPU performance — where the time goes, and what is left

Measured 2026-08-13, revised 2026-08-14, on Apple M1 Pro, single env, against
MuJoCo 3.10.0 stepping the same XMLs. Three models: SO-ARM100, SO-ARM101
(`float32` ours vs `float64` MuJoCo) and Sawyer Reach-v3 (`float64` both).

The short version: **the gap to MuJoCo is algorithmic, not Mojo-vs-C, and it is
entirely collision.** Forward kinematics is faster than MuJoCo's on all three
models; nothing outside collision is worse than 1.7×. §10 is that table.

⚠⚠ **THE MuJoCo BASELINE ITSELF WAS WRONG UNTIL 2026-08-14 — SEE §9.** Stock
MuJoCo memsets its whole BVH-active array every step for the *visualiser*, and
every ratio this project published included it: up to **51% of the reference's
step** on SO-ARM101. Correcting it makes our numbers worse, not better.

⚠⚠ **THE 2026-08-13 REVISION GOT THE CAUSE WRONG AND §4 IS ITS RETRACTION.** It
named a missing mid-phase BVH as the largest prize, from a node count that is
99.99% mesh-face BVH the collision path never reads. What the gap actually was,
in three parts, none of them the BVH: one linear scan that should have been a
hill climb (§3), one silently truncated model (§5.2), and GJK converging to a
distance nobody reads (§6.1). Together **2.13× / 2.03×**.

⚠ **THIS DOCUMENT RECORDS ITS DEAD ENDS ON PURPOSE**, because every one of them
looked like the obvious next move and four were measured to zero or worse: the
BVH (§4), the broadphase filter hoist (§5.1, twice), a cross-step warm cache for
the hill climb (§6.3, *16% slower*), and both multicontact paths (§6.2, ~1%).
The pattern is consistent — a counter tells you how OFTEN something runs, and
an ablation tells you what it COSTS. Only the second one picks targets.

---

## 1. Headline

Per **physics** step (env step ÷ `FRAME_SKIP=10`), `float32`, against MuJoCo
3.10.0 (`float64`) stepping the same two XMLs.

| model | before | after | speedup |
|---|---|---|---|
| SO-ARM100 | 15.42 s | **7.28 s** | **2.13×** |
| SO-ARM101 | 20.33 s | **9.96 s** | **2.03×** |

(40 000 env steps = 400 000 physics steps; MIN of interleaved rounds against a
pristine worktree. ⚠ SO-ARM100's "after" carries **two more collision meshes
than its "before"** — see §5.2 — so it is doing strictly more work.)

And against MuJoCo, per physics step, ours and MuJoCo **interleaved with each
other**, MIN of 3 rounds:

| model | ours | MuJoCo 3.10.0 | ratio |
|---|---|---|---|
| SO-ARM100 | 18.09 µs | 7.53 µs | **2.40×** |
| SO-ARM101 | 24.54 µs | 5.84 µs | **4.20×** |
| Sawyer Reach-v3 | 33.40 µs | 15.49 µs | **2.16×** |

⚠⚠ **THESE ARE WORSE THAN THE 1.93× / 1.78× THIS DOCUMENT PUBLISHED ON
2026-08-13, AND THE OLD NUMBERS WERE THE WRONG ONES.** Every MuJoCo baseline
this project has ever quoted was measured with `bvactive` on, i.e. against
MuJoCo's physics **plus a per-step 700 kB memset that nothing in the dynamics
reads**. §9 has the proof. Correcting it moves SO-ARM101 from our best model
to our worst — 1.78× → 4.20× — because that model has the largest mesh BVH in
the tree and was therefore flattered the most.

⚠ **DO NOT COMPARE ANY OF THESE ACROSS SESSIONS.** Only within-session,
interleaved pairs mean anything; identical code has drifted 1.4–1.7× on this
box.

⚠ Our column is the **whole env step ÷ FRAME_SKIP**, so it includes obs/reward/
action glue that MuJoCo's `mj_step` does not do — 0.99, 0.89 and ~2.2 µs/step
respectively (§10).

⚠ **THE ARMS ARE SLOWER WHILE CARRYING HALF THE PRECISION.** Both SO-ARM runs
are `float32` against MuJoCo's `float64`, so their honest gap is worse than the
ratio. **Sawyer is not** — it runs `float64` on both sides, which is why it is
the fair one of the three.

The two arms are `nq = nv = nu = 6`, `nbody = 8/9`, at `ncon` 1 and 0 — so what
they compare is almost entirely **the cost of proving that geoms are apart**.
Sawyer is the opposite corner (`nv = 15`, `nbody = 34`, `ncon = 5`) and was
added on 2026-08-14 precisely because a single-shape workload cannot tell a
constant factor from an `O()` defect: §11 is one that only Sawyer could see.

---

## 2. What landed, and what it was worth

| change | SO-ARM100 | SO-ARM101 |
|---|---|---|
| plane-mesh support point: full argmin → hill climb (§3) | 1.21× | 1.28× |
| `<mesh>` asset cap silently truncating the model (§5.2) | 1.63× cumulative | n/a (13 assets) |
| GJK cutoff exit — stop bounding a distance nobody reads (§6.1) | 1.31× | 1.59× |
| **cumulative** | **2.13×** | **2.03×** |
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

⚠ EVERY FIGURE BELOW THE CUTOFF LANDED IS RE-MEASURED. §6.1 cut GJK by 2.6×,
which reshuffled the ranking; the pre-cutoff stage split that used to sit here
described a build that no longer exists.

`sample`, exclusive attribution, current build:

| phase | SO-ARM100 (18.20 µs) | SO-ARM101 (24.90 µs) |
|---|---|---|
| `detect_contacts_sap` (broadphase + everything inlined into it) | 51.1% | 31.3% |
| `gjk_epa_witness` (out-of-line) | 12.6% | **53.9%** |
| **Newton solver** | **23.3%** | 4.6% |
| kinematics + CRBA + LDL + cdof | 6.9% | 6.0% |
| env glue | 5.7% | 3.9% |

⚠ THE TWO ARMS NOW WANT DIFFERENT WORK. SO-ARM101 is still a narrow-phase
story; SO-ARM100 is not — its solver is now the second-largest item, because
collision shrank around it.

### 6.1 GJK converged to a distance nobody read — CLOSED, 1.31× / 1.59×

From a de-inlined build under `sample` (SO-ARM101), the support machinery was
**62% of the physics step**:

| symbol | % of physics step |
|---|---|
| `gjk::hillclimb_support_index` | **54.1** |
| `broadphase_sap::detect_contacts_sap` | 17.9 |
| `gjk::gjk_epa_witness` | 10.8 |
| `gjk::_support` / `_support_mesh` | 8.4 |

Counters inside the hill climb (SO-ARM101, per physics step) say it is **not**
walking badly:

| | |
|---|---|
| support queries | 119 (≈30 per GJK call ⇒ ~15 iterations) |
| walk steps per query | **7.05** (cold 20.4, warm 5.95) |
| neighbour dots per query | 55.3 |
| mean degree of the hull graph | **6.0005** (MuJoCo's: 5.995) |
| cold starts | 7.6% of queries, **21% of the work** |

⚠ **THREE PLAUSIBLE CULPRITS DIED HERE.** The edge graph is not over-connected
— 6.0005 against MuJoCo's 5.995, both the Euler value for a triangulation. The
walks are not long — 7 steps. GJK is not running to its cap — ~15 iterations
against `GJK_MAX_ITERATIONS = 100`, so this was *not* another instance of
[the float32 tolerance trap](#) that bit Newton and GJK before. And a
cross-step warm cache is capped at 21% of the walk, not the 30% the
warm-start-off experiment (16.45 → 21.38 s) suggested.

**The actual difference is that MuJoCo never computes the distance.**
`engine_collision_convex.c:106` sets `config.dist_cutoff = 0` — *"no geom
distances needed"* — so `mj_gjk` returns the moment it can bound the pair
apart, in 1–3 iterations. Ours converged all ~15 to produce a `dist` whose only
consumer is `if dist < cm`. Confirmed from the other side: `mj_geomDistance`,
which *must* converge, costs MuJoCo **3.72 µs/call** on these very pairs —
close to our 8.74 — against ~0.6 µs/call for its in-step path. The gap was
never per-iteration speed. It was doing 15 iterations instead of 2.

**The fix** is MuJoCo's `dist_cutoff` arm (`engine_collision_gjk.c:225`): with
`nd = -v/|v|`, `-w_dot` is `dot(w, v)/|v|`, the standard GJK **lower bound** on
the distance. Once that bound reaches `cm`, no further iteration can change
`dist < cm`, so the loop returns.

⚠⚠ **THIS IS SAFE WHERE THE `gi == 0` CERTIFICATE WAS NOT, AND THE DIFFERENCE
IS THE BOUND.** That branch proved "separated" and returned 1e30 — equivalent
to "no contact" only at margin 0, and with a margin it lost every contact in
the band (0 against MuJoCo's 5). The cutoff exits only when a lower bound on
the true distance has reached the exact threshold the caller compares against,
so it can cost iterations, never a contact. A penetrating pair has the origin
inside, hence `dot(w, v) < 0`, so it can never fire on one.

⚠ **DO NOT ALSO COPY MuJoCo'S OTHER EARLY-OUT** (`!get_dist`, one branch up),
which returns on *any* separating hyperplane. That is safe only because
`mjc_penetration` inflates both geoms by margin first — a transformation we
have never ported. See
`feedback_copying_control_flow_without_its_precondition`.

⚠ **THE CUTOFF IS OPT-IN AND ITS DEFAULT MUST STAY DISABLED.** Passing it makes
`gjk_epa_witness` return a *lower bound* rather than the true separation, which
every distance gate in the tree would fail
(`test_gjk_float32_no_phantom_contacts` asserts on separations of 7–17 cm).
Only the two narrow-phase call sites, which read the result solely through
`if dist < cm`, pass one.

### ⚠ 6.2 RETRACTED: `multi_ccd` is ~1%, not ~5.2 µs

The previous revision measured `multi_ccd_extra_contacts` at ~5.2 µs/step by
diffing two **stage-timer** builds (22.90 → 17.74 µs) — and warned in the same
breath that absolutes move between ablation builds because removing a large
inlined block changes register allocation. That warning applied to its own
number. Re-measured on the current build with plain wall clock, MIN of 3
interleaved rounds:

| SO-ARM100 | MIN |
|---|---|
| current | 7.26 s |
| `multi_ccd_extra_contacts` stubbed | 7.30 s |
| `MC_ENABLED = False` (native multicontact) | 7.18 s |

Both are ~1%, i.e. inside the noise. **Neither multicontact path is a lever.**
⚠ Use undistorted wall-clock A/B for attribution whenever the stage can be
stubbed; keep stage timers for finding *which* stage, not *how much*.

### ⚠ 6.3 KILLED: a cross-step warm cache for the hill climb

The obvious read of §6.1's counters is that cold starts dominate: after the
cutoff, support queries fell 119 → 17 per step while **steps per query rose
7.05 → 27**, with cold starts 53% of queries and 40% of walk work. So carry the
last support vertex across steps.

Built as a ceiling probe — one warm slot per mesh parked in the free tail of
`mesh_edges`, so no plumbing — and measured:

| | current | with warm cache |
|---|---|---|
| SO-ARM100 | 7.44 s | 7.41 s (nothing) |
| SO-ARM101 | 10.02 s | **11.19 s (16% SLOWER)** |

A vertex cached from a *different search direction* is a worse seed than vertex
0, and jumping to it thrashes the locality that a consistent start point keeps.
⚠ The probe cost one file edit and one build; the real version would have been
a new `Data` field threaded through both narrow phases and the GPU kernel. Test
the payoff before the implementation.

### 6.4 Newton runs 5 iterations where MuJoCo runs 1 — MEASURED, not fixed

Now the largest single item on SO-ARM100 at **23.3% of the step**. Counted over
20 000 physics steps (elliptic cone — ⚠ the pyramidal loop one branch up is
dead for both arms, and instrumenting it first returned zero calls):

| | |
|---|---|
| mean iterations per solve | **5.13** |
| MuJoCo `solver_niter` on the same model | **1** |
| max | 200 (the cap) |

⚠ THERE IS NO PATHOLOGICAL TAIL, WHICH IS WHAT THE HISTOGRAM IS FOR. From the
mean of 5.13 against a max of 200 it is tempting to infer a few non-converging
solves carrying the cost — arithmetic gives ~1.6% at the cap. Measured: **14
solves (0.07%) hit the cap and account for 2.7% of Newton work**, 15 more sit
in 11–199, and **99.855% of solves take ≤10 iterations and carry 96.6% of the
work**. The cost is the ordinary case, not the tail.

**What MuJoCo does that we do not: warm-start `qacc`.** `mj_warmstart`
(`engine_forward.c:611`) starts from `d->qacc_warmstart` — the previous step's
solution — after picking the better of it and `qacc_smooth` by cost. At steady
state that lands on the answer, hence one iteration. We start cold every step.

The storage already exists: `qacc_constrained` is a per-env `[BATCH, NV]` that
already holds the previous solution. What is missing is the cost comparison and
the choice.

⚠ WORTH ~1.1× AND ON ONE MODEL ONLY — SO-ARM101's solver is 4.6% of its step,
so this is a SO-ARM100 change. It also alters the solver's starting point for
every model in the tree, so it wants the full parity suite. Sized honestly
before building, not after.

⚠ It should be a pure speed change: the constrained problem is convex, so
Newton converges to the same minimum from any start. If a parity gate moves,
that is evidence of a convergence bug, not of the warm start.

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

---

## ⚠⚠ 9. The MuJoCo baseline was carrying a per-step debug memset

**Every MuJoCo number this project has published was inflated**, by between 8%
and 51%, and the correction is not uniform across models — so it changed the
ranking, not just the scale.

`mj_collision` (`engine_collision_driver.c`) opens with

```c
  if (m->vis.global.bvactive) {
    memset(d->bvh_active, 0, m->nbvh);
  }
```

`bvactive` is a **visualisation** flag — it exists so the viewer can highlight
which bounding volumes were touched — and **it defaults to 1**. `nbvh` counts
every node of every mesh BVH, so it is enormous on mesh-heavy scenes. Measured,
`mj_step` with the flag on and off, 20 000 steps, MIN of 5:

| model | `nbvh` | bvactive=1 | bvactive=0 | delta |
|---|---|---|---|---|
| SO-ARM100 | 123 136 | 8.705 µs | 7.720 µs | 0.985 µs (11.3%) |
| **SO-ARM101** | **696 364** | 12.160 µs | **5.987 µs** | **6.173 µs (50.8%)** |
| Sawyer | 133 936 | 17.016 µs | 15.595 µs | 1.421 µs (8.4%) |

⚠ **THE DELTA IS memset BANDWIDTH, WHICH IS HOW YOU KNOW IT IS REAL AND NOT
NOISE.** 123 kB/0.985 µs, 696 kB/6.173 µs and 134 kB/1.421 µs are 125, 113 and
94 GB/s — all three land on M1 Pro's memset rate for that byte count. A
timing artefact would not track the byte count that precisely across a 5.7×
range.

**Consequences, in order of how much they hurt:**

- SO-ARM101 went from **1.78× (our best model) to 4.20× (our worst)**. It has
  by far the largest `nbvh` in the tree, so it was the most flattered.
- The 2026-08-13 revision named a missing mid-phase BVH as the largest prize
  partly from MuJoCo's BVH node count. That claim was already retracted (§4) on
  other grounds; this is a second, independent reason it pointed the wrong way.
  **A number that large sitting in the reference is worth explaining before it
  is used as evidence.**
- `benchmarks/physics3d_mujoco_phases.py` now sets `bvactive = 0` by default
  and prints the flag and `nbvh` on every run. Pass a 4th argument `1` to get
  the old behaviour back.

⚠ **A SECOND, SMALLER INFLATION IN THE SAME SCRIPT, ALSO FIXED.** The
calibration loop appended `d.ncon` / `d.nefc` / `d.solver_niter` to Python lists
**inside the timed region**. That pybind11 traffic is ~0.4 µs/step, and since
every phase is scaled by `wall_us / raw_STEP`, it inflated *every phase
number the script has ever printed* by 3–5%. Work counts now come from their
own untimed loop.

⚠ **IS TURNING IT OFF FAIR?** Yes, and state why rather than assume it: nothing
in the dynamics reads `bvh_active`, our engine has no counterpart, and a
headless benchmark is not drawing anything. But it *is* what a user gets from
stock MuJoCo, so quote it when the question is "what does MuJoCo cost me",
and quote `bvactive=0` when the question is "whose physics is faster".

---

## 10. Where the difference actually is: collision, on every model

Ours from `sample` (exclusive attribution, `benchmarks/physics3d_sample_phases.py`)
scaled onto the interleaved wall-clock totals of §1; MuJoCo from its own
`mjTimerStat` phases with `bvactive=0`. Grouped so the two sides line up —
MuJoCo's `POS_KINEMATICS` covers `mj_kinematics` + `mj_comPos`, which is our
kinematics + `cdof` + `subtree_com`; its `POS_INERTIA` is `mj_crb` +
`mj_factorM`, which is our mass matrix + LDL.

| phase | SO-ARM100 | SO-ARM101 | Sawyer |
|---|---|---|---|
| **collision** (broad + narrow) | 11.74 / 3.61 = **3.25×** | 20.93 / 1.95 = **10.7×** | 12.75 / 6.77 = **1.88×** |
| constraint build + solve | 3.98 / 2.36 = 1.69× | 1.24 / 1.80 = **0.69×** | 11.47 / 6.77 = 1.69× |
| mass matrix + LDL | 0.54 / 0.19 = 2.8× | 0.49 / 0.21 = 2.3× | 1.19 / 0.46 = 2.6× (§11) |
| kinematics + cdof + subtree com | 0.81 / 0.96 = **0.85×** | 0.96 / 1.13 = **0.85×** | 1.58 / 2.05 = **0.77×** |
| our env glue (no MuJoCo counterpart) | 0.99 | 0.89 | ~2.2 |

(µs per physics step, ours / MuJoCo.)

**The finding is that there is only one finding.** Outside collision nothing is
worse than 1.7×; forward kinematics is **faster than MuJoCo on all three
models**, and SO-ARM101's constraint stage is faster too. Collision carries the
entire gap:

- **SO-ARM101: collision is 19.0 µs of excess against a total gap of 18.7 µs.**
  Everything else nets out slightly in our favour. There is no second target on
  this model — it is the narrow phase or nothing.
- SO-ARM100: 8.1 µs of a 10.6 µs gap.
- Sawyer: 6.0 µs of a 17.9 µs gap — the only model where the solver (4.7 µs of
  excess) is in the same league, because it is the only one with real contacts
  (`ncon = 5`, `nefc = 29`).

⚠⚠ **THE NEXT STEP IS A CALL COUNTER, AND THE ARITHMETIC ALREADY SAYS SO.**
MuJoCo's whole narrow phase on SO-ARM101 is **0.41 µs/step**; ours is ~20 µs.
§4 measured both engines selecting the same pairs, **4.0 plane + 1.0 geom-geom
per step on each side** — and if that is also the `gjk_epa_witness` call count,
then one geom-geom call is costing us ~20 µs, which **contradicts §6.1's own
8.74 µs/call before a 1.59× cutoff**. Those two cannot both be right. Either a
pair issues more than one call (multi-contact re-invocation is the obvious
suspect), or the `sample` bucket is charging `gjk_epa_witness` work it does not
own. **Do not design against either story until a counter at the call site says
which.** A Python replay of MuJoCo's filter chain bounds its narrow-phase pairs
at ≤24/step here, but that ignores its body-level broadphase and settles
nothing.

⚠ This probe wants the widened-`smeta` build of §8, and on 2026-08-14 it was
**abandoned mid-setup because the machine ran out of disk** — a `git worktree`
of this repo makes pixi materialise a fresh 1.8 GB environment, and the volume
was already at 99%. Instrument in place and revert, or free space first.

⚠ **`sample` CANNOT SEE A FUNCTION THAT BECAME SMALL ENOUGH TO INLINE.** After
§11 the `mass_matrix` bucket vanished from Sawyer's profile entirely — not
because the work went to zero, but because the tree-walk is small enough that
`compute_mass_matrix` now inlines into the step and its residue is charged to
`env/other` (which "grew" 6.77 → 7.47 µs while the step shrank 4.58 µs). **The
wall clock is the authority; the profile only says where to look.**

---

## 11. LANDED: the CPU CRBA was O(NV²·NBODY) — 1.14× on Sawyer

`dynamics/mass_matrix.mojo` has two algorithms. The **dense** one evaluates
every `(i, j)` DOF pair against every body through a subtree mask; the
**tree-walk** one (`_mm_treewalk_env`) accumulates composite inertia leaf→root
and then walks each DOF's ancestor chain, which is what `mj_crb` does. Their
costs are not a constant apart:

| | dense | tree-walk |
|---|---|---|
| inner iterations, SO-ARM100 (NV=6, NBODY=8) | 168 | ~44 |
| inner iterations, Sawyer (NV=15, NBODY=34) | **4 080** | **~110** |

**The tree-walk was unreachable from the CPU.** `compute_mass_matrix` carried
`comptime assert not (TREEWALK and not PARALLEL)`, and all three integrators
carried the matching `PARALLEL_GPU or not CRBA_TREEWALK`. So the whole CPU side
— the viewer, every test, every single-env rollout — ran the dense kernel.

⚠ **THE REQUIREMENT WAS NEVER REAL.** The "inherently cooperative" tree-walk
kernel's only parallelism is two `range(tid, N, N_THREADS)` loops and two
`barrier()` calls; `N_THREADS = 1, tid = 0` collapses them exactly. The fix is
one shared `@always_inline` helper with the barriers behind a `comptime if GPU`,
called by both the GPU kernel and a new CPU branch — so there is still exactly
one copy of the arithmetic and the GPU path stays bit-identical.

Measured, interleaved, MIN of 3 rounds:

| model | dense | tree-walk | |
|---|---|---|---|
| **Sawyer** (NV=15, NBODY=34) | 37.98 µs | **33.40 µs** | **1.14×** |
| SO-ARM100 (NV=6, NBODY=8) | 18.17 | 18.42 | noise |
| SO-ARM101 (NV=6, NBODY=9) | 24.73 | 24.54 | noise |

⚠ **THIS DEFECT IS INVISIBLE ON SMALL MODELS AND THAT IS THE LESSON.** On the
arms it was worth 0.3 µs and sat inside a 6.9% "kinematics + CRBA + LDL + cdof"
line nobody would ever pick as a target. It took a model with 34 bodies to make
it 13.7% of the step. **A profile taken on one shape of model cannot distinguish
a constant factor from a growth rate** — the arms said "CRBA is 3% of the step",
which was true and useless. Every model larger than the arms was paying:
humanoid, quadruped, dog, and every dm_control manipulation scene.

Gates, all green: `test_crba_treewalk_fields` (bit-exact vs the legacy GPU
tree-walk; tolerance vs dense), `test_sawyer_settle_vs_mujoco`,
`test_sawyer_mesh_rest_vs_mujoco`, `test_euler_fields_vs_mujoco`,
`test_humanoid_limits_fields_vs_mujoco`, `test_constraints_vs_mujoco`.

⚠ The tree-walk is float-tolerance-equal to the dense kernel, **not bit-exact**
— it sums the same terms in a different order. Tests that pin CPU `M` bitwise
against the dense kernel would move; none in the suite do, but a new one should
not be written that way.

---

## 12. What is left, in the order the measurements support

1. **SO-ARM101's narrow phase** — 20 µs against MuJoCo's 0.41. Biggest item in
   the tree by a wide margin, and the *only* item on that model. First step is
   a call counter, not a rewrite (§10).
2. **SO-ARM100's collision** — 11.7 vs 3.6 µs, same shape of problem.
3. **Sawyer's solver** — 11.5 vs 6.8 µs. The only model where the solver is a
   real target, and the one place the Newton warm start of §6.4 would show up
   against a `ncon = 5` workload rather than a contact-free one.
4. Nothing else. Kinematics is already faster than MuJoCo, the mass matrix is
   fixed, and the broadphase sweep is 0.91 µs.
