# physics3d CPU performance — where the time goes, and what is left

Measured 2026-08-13, revised 2026-08-14, on Apple M1 Pro, single env, against
MuJoCo 3.10.0 stepping the same XMLs. Three models: SO-ARM100, SO-ARM101
(`float32` ours vs `float64` MuJoCo) and Sawyer Reach-v3 (`float64` both).

The short version: **the gap to MuJoCo is algorithmic, not Mojo-vs-C, and it is
entirely collision.** Nothing outside collision is worse than 1.7×, and forward
kinematics is faster than MuJoCo's on two of the three models. §10 is that
table; §10.1 splits the narrow phase into calls × cost per call.

⚠⚠ **THAT SENTENCE WAS TRUE OF THREE SMALL ARMS AND IS FALSE OF THE TREE. §13
(2026-09-04) sweeps fourteen models and finds the gap is the CONSTRAINT SOLVER
and a dense `M⁻¹`, growing with `nv` to 10–13× on the 60–80-dof models, while
collision is under 2% of those steps.** §1–§12 stand as the record of the
collision work; §13 is the sweep, and **§13.5 is what landed on it the next
day: humanoid_CMU 10.7× → 3.75×, the park scenes 10.7× → 3.8×, every row
past 20 dofs 1.3–3× faster, two of the three changes bit-exact.**

⚠⚠ **BOTH SIDES OF THE COMPARISON WERE WRONG UNTIL 2026-08-14.** (a) Stock
MuJoCo memsets its whole BVH-active array every step for the *visualiser*, and
every ratio this project published included it — up to **45% of the
reference's step** on SO-ARM101 (§9). (b) SO-ARM101 was being compared against
Menagerie's `robotstudio_so101`, which is **not the model we ported** and
collides with boxes where ours collides with 27 k-vertex meshes (§1). Both
corrections make our numbers worse, not better.

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

(2026-09-06: this section is the August SO-ARM/Sawyer study. For the
fourteen-model standing against MuJoCo after the September solver rounds,
read §13.27 first, then §13.20–§13.26 for how each row got there.)

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

| model | ours | MuJoCo 3.10.0 | ratio | reference XML |
|---|---|---|---|---|
| SO-ARM100 | 18.09 µs | 7.53 µs | **2.40×** | `mujoco_menagerie-main/trs_so_arm100/scene.xml` |
| SO-ARM101 | 25.07 µs | 7.63 µs | **3.29×** | `SO-ARM100-main/Simulation/SO101/scene.xml` |
| Sawyer Reach-v3 | 33.40 µs | 15.49 µs | **2.16×** | `Metaworld-master/.../sawyer_reach_v3.xml` |

⚠⚠ **QUOTE THE REFERENCE XML WITH THE RATIO, BECAUSE SO-ARM101 HAS TWO AND
THEY ARE NOT THE SAME ROBOT.** Menagerie's `robotstudio_so101` collides with
~20 **boxes** plus three 124-vertex gripper hulls — every full-body mesh in it
is `class="visual" contype="0" conaffinity="0"`. The model we ported is The
Robot Studio's own (`references/SO-ARM100-main/`), which collides **13
full-resolution meshes** up to 27 k vertices. Benchmarking our mesh collision
against MuJoCo's box collision is not an engine comparison, and this document
did exactly that earlier on 2026-08-14: it published **4.20×** against
menagerie, where the model we actually run gives **3.29×**.

⚠⚠ **THESE ARE STILL WORSE THAN THE 1.93× / 1.78× PUBLISHED ON 2026-08-13.**
Every MuJoCo baseline this project has ever quoted was measured with `bvactive`
on, i.e. against MuJoCo's physics **plus a per-step 645 kB memset that nothing
in the dynamics reads**. §9 has the proof. That correction is independent of
the XML one above and both point the same way.

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
| **SO-ARM101** | **645 136** | 13.364 µs | **7.372 µs** | **5.992 µs (44.8%)** |
| Sawyer | 133 936 | 17.016 µs | 15.595 µs | 1.421 µs (8.4%) |

⚠ **THE DELTA IS memset BANDWIDTH, WHICH IS HOW YOU KNOW IT IS REAL AND NOT
NOISE.** 123 kB/0.985 µs, 645 kB/5.992 µs and 134 kB/1.421 µs are 125, 108 and
94 GB/s — all three land on M1 Pro's memset rate for that byte count. A
timing artefact would not track the byte count that precisely across a 5.2×
range. (Menagerie's `robotstudio_so101` — the *wrong* so101, see §1 — gives
696 364 nbvh and 6.173 µs, i.e. 113 GB/s. The finding is the same on either.)

**Consequences, in order of how much they hurt:**

- SO-ARM101 went from **1.78× (our best model) to 3.29× (our worst)**. It has
  by far the largest `nbvh`, so it was the most flattered. (An earlier version
  of this section said 4.20×; that also carried the wrong-XML error of §1.)
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
| **collision** (broad + narrow) | 11.74 / 3.61 = **3.25×** | 20.93 / 3.58 = **5.84×** | 12.75 / 6.77 = **1.88×** |
| constraint build + solve | 3.98 / 2.36 = 1.69× | 1.24 / 1.75 = **0.71×** | 11.47 / 6.77 = 1.69× |
| mass matrix + LDL | 0.54 / 0.19 = 2.8× | 0.49 / 0.18 = 2.7× | 1.19 / 0.46 = 2.6× (§11) |
| kinematics + cdof + subtree com | 0.81 / 0.96 = **0.85×** | 0.96 / 0.71 = 1.36× | 1.58 / 2.05 = **0.77×** |
| our env glue (no MuJoCo counterpart) | 0.99 | 0.89 | ~2.2 |

(µs per physics step, ours / MuJoCo.)

**The finding is that there is only one finding.** Outside collision nothing is
worse than 1.7×, forward kinematics is faster than MuJoCo on two of the three
models, and SO-ARM101's constraint stage is faster. Collision carries the gap:

- SO-ARM101: **17.3 µs of a 17.4 µs gap.** Everything else nets out. There is
  no second target on this model — it is the narrow phase or nothing.
- SO-ARM100: 8.1 µs of a 10.6 µs gap.
- Sawyer: 6.0 µs of a 17.9 µs gap — the only model where the solver (4.7 µs of
  excess) is in the same league, because it is the only one with real contacts
  (`ncon = 5`, `nefc = 29`).

### 10.1 The narrow phase, split into calls × cost per call

Counters at the `gjk_epa_witness` and `_plane_mesh_contacts` call sites in
`_detect_contacts_sap_env` (widened `smeta`, 200 000 physics steps, §8):

| | calls/step | µs/step | **µs per call** |
|---|---|---|---|
| SO-ARM100 geom-geom GJK | 2.00 | 2.37 | **1.18** |
| SO-ARM100 plane-mesh | 5.00 | 1.08 | 0.22 |
| SO-ARM100 **narrow total** | 7.00 | **3.45** | vs MuJoCo **2.49** = **1.4×** |
| SO-ARM101 geom-geom GJK | **4.00** | 14.26 | **3.56** |
| SO-ARM101 plane-mesh | 1.00 | 0.05 | 0.05 |
| SO-ARM101 **narrow total** | 5.00 | **14.31** | vs MuJoCo **2.49** = **5.7×** |

**The call counts are MuJoCo's exactly** — §4 measured 2.0 + 5.0 and 4.0 + 1.0
on both sides, and these are 2.00 + 5.00 and 4.00 + 1.00. So the whole
difference is **cost per call**, and it is not uniform: SO-ARM100's narrow
phase is already within **1.4×** of MuJoCo's, while SO-ARM101's is 5.7×.

⚠⚠ **THE INTERESTING NUMBER IS NOT THE RATIO TO MuJoCo, IT IS THE RATIO TO
OURSELVES: 3.56 µs/call on SO-ARM101 against 1.18 µs on SO-ARM100, same code.**
That is a property of the *geometry*, not the algorithm — SO-ARM101 collides 13
full-resolution meshes (up to 27 k vertices; one hull is ~4 000 by §3) where
SO-ARM100's collision meshes are 8–187-vertex `*_Collision_*.stl` proxies plus
4 boxes. **The support walk is doing more work because it is walking a bigger
polytope, and the model is why.**

⚠ **THIS RETRACTS THE CONTRADICTION THIS SECTION FLAGGED HOURS EARLIER.** It
read §4's "4.0 + 1.0" as *plane + geom-geom* when the column order is
*geom-geom + plane*, inferred one 20 µs GJK call from it, and called the result
irreconcilable with §6.1's 8.74 µs/call. There was never a contradiction —
4 calls at 3.56 µs. **The counter cost one build; the misreading cost a
paragraph of confident nonsense in a document whose whole point is that
inference loses to measurement.**

⚠ **§10.1 BELOW IS THAT SPLIT, MEASURED.** It replaces an inference this
section carried for a few hours and which was wrong in both directions.

⚠ The probe wants the widened-`smeta` build of §8. A `git worktree` of this
repo cannot host it — `references/` alone is 5.3 GB and pixi materialises a
fresh multi-GB environment per manifest — so **instrument in place, measure,
`git checkout --` the two files**. Total cost: one build.

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

(2026-09-06: superseded for the solver-bound models by §13.27; items 1–2
here, the convex walk and SO-ARM100's collision, still stand.)

1. **The support walk on big hulls.** SO-ARM101's geom-geom GJK costs
   **3.56 µs/call against SO-ARM100's 1.18 µs — same code, 3× apart** (§10.1).
   The call counts already match MuJoCo exactly on both models, so there is no
   pruning left to win; the cost is per call and it tracks hull size. Two
   candidates, in order of what the evidence supports:

   ⚠⚠ **"OUR HULLS ARE TOO BIG" WAS THE OBVIOUS EXPLANATION AND IT IS FALSE.**
   MuJoCo collides against `mesh_graph` — the convex hull plus its edge graph —
   and the reduction from the raw mesh is large (`wrist_roll_pitch_so101_v2`
   26 967 verts → **7 296** hull). Summed over the collidable meshes MuJoCo
   walks **76 320** hull vertices on SO-ARM101; our whole mesh table is
   `NMESH_VERTS = 33 280`, and `fields_build` RAISES on overflow rather than
   truncating, so that total is real. **We walk a polytope less than half
   MuJoCo's size and are still ~6× slower per call.** That makes the remaining
   gap a code problem, not a data problem — which is the opposite of what this
   list said before the query was run. Cost of the check: one Python call.

   So the target is the walk itself, and §7.2 already says it is latency-bound
   and hostile to SIMD. Expect this to be hard, and re-derive the per-call cost
   on SO-ARM100 (1.18 µs on a 2 746-vertex table) versus SO-ARM101 (3.56 µs on
   33 280) before assuming the walk length is what scales.

2. **SO-ARM100's collision** — 11.7 vs 3.6 µs total, though its *narrow phase*
   is already within 1.4× (§10.1). The excess is in `detect_contacts_sap`
   outside the 0.91 µs sweep and outside GJK, which is not yet split.
3. **Sawyer's solver** — 11.5 vs 6.8 µs. The only model where the solver is a
   real target, and the one place the Newton warm start of §6.4 would show up
   against a `ncon = 5` workload rather than a contact-free one.
4. Nothing else. The mass matrix is fixed (§11), the broadphase sweep is
   0.91 µs, and kinematics is at or better than MuJoCo.

---

## 13. 2026-09-04 — fourteen models: the gap moved to the solver, and it grows with `nv`

Re-measured against MuJoCo 3.10.0 on the M1 Pro, one physics step, same XML,
same protocol on both sides (`scripts/physics3d_cpu_vs_mujoco.sh`; §13.4).
Ours `float32` single env through `Phyics3dEnv`, MuJoCo `float64`, `bvactive=0`,
MIN of 3 interleaved rounds, spread printed.

| model | nv | integ | ours µs | MuJoCo µs | **ratio** | ncon ours/mj | nefc | mj niter |
|---|---|---|---|---|---|---|---|---|
| hopper | 6 | RK4 | 15.00 | 14.67 | **1.02×** | 2.00/2.00 | 11.0 | 0.23 |
| half_cheetah | 9 | Euler | 6.28 | 4.65 | 1.35× | 2.00/2.00 | 8.0 | 1.00 |
| walker2d | 9 | RK4 | 38.77 | 24.22 | 1.60× | 6.00/6.01 | 27.8 | 1.00 |
| sawyer_reach | 15 | Euler | 25.18 | 15.22 | 1.65× | 5.00/5.00 | 30.0 | 1.00 |
| so_arm101 | 6 | Euler | 11.59 | 6.49 | 1.79× | 0/0 | 6.0 | 1.00 |
| so_arm101 **f64** | 6 | Euler | 11.85 | 6.43 | 1.84× | 0/0 | 6.0 | 1.00 |
| park_k0 | 6 | Euler | 11.40 | 6.32 | 1.80× | 0/0 | 6.0 | 1.07 |
| ant | 14 | RK4 | 67.50 | 33.38 | 2.02× | 3.00/3.00 | 20.0 | 0.09 |
| humanoid | 23 | RK4 | 199.2 | 80.3 | 2.48× | 7.00/7.00 | 39.6 | 4.92 (PGS) |
| park_k3 | 24 | Euler | 22.90 | 6.95 | 3.30× | 0/0 | 6.0 | 1.07 |
| park_k6 | 42 | Euler | 47.99 | 7.76 | **6.18×** | 0/0 | 6.0 | 1.07 |
| park_k9 | 60 | Euler | 94.33 | 8.81 | **10.71×** | 0/0 | 6.0 | 1.07 |
| humanoid_cmu | 62 | Euler | 768.3 | 71.95 | **10.68×** | 13.67/13.67 | 77.8 | 1.75 |
| dog_stand | 79 | Euler | 3003 | 231.4 | **12.98×** | 8.98/8.99 | 63.0 | 6.09 |

Three things are visible before any profile:

* **The ratio is a function of `nv`, not of the scene.** Six-dof models sit at
  1.0–1.8×; every model past 40 dofs is 6–13×. The park sweep isolates it: the
  scene is SO-ARM101 plus `k` free props in mid-air, **zero contacts, six
  constraint rows at every k**, and ours goes 11.4 → 94.3 µs while MuJoCo goes
  6.3 → 8.8. The excess over k=0 divided by `Δnv²` is 0.0355 / 0.0282 / 0.0284
  — quadratic in the dofs, the same shape the GPU probe found
  (`docs/BLOCK_DIAGONAL_MASS_MATRIX_PLAN.md` §1.1).
* **The float is worth 2%.** `so_arm101` in `float64` is 1.84× against 1.79× in
  `float32`. Whatever the gap is, it is not precision, and it is not Mojo-vs-C
  either — hopper is at parity.
* **SO-ARM101 is at 1.79×, down from 3.29× on 2026-08-14** (25.07 → 11.59 µs;
  MuJoCo 7.63 → 6.49). The collision campaign of §1–§12 did land.

### 13.1 Where the time goes — ours by `sample`, MuJoCo by its own timers

Ours: `physics3d_sample_top.py` exclusive symbols scaled onto the table's µs.
MuJoCo: `physics3d_mujoco_phases.py`, `bvactive=0`. Grouped as §10 did.

| µs per step | park_k9 | humanoid_cmu | dog_stand | humanoid | ant | so_arm101 |
|---|---|---|---|---|---|---|
| **`solve_newton`** (ours) | **70.9** (75%) | **462** (60%) | **2046** (68%) | **127** (64%) | **38** (56%) | 1.2 (11%) |
| MuJoCo CONSTRAINT (+PROJECT) | 1.5 | 40.1 | 78.9 (+80.6) | 27.4 (+18.9) | 13.6 | 0.95 |
| **`compute_m_inv`** (ours) | 6.4 (7%) | **231** (30%) | **648** (22%) | **48** (24%) | 12 (18%) | 0.3 |
| `ldl_factor` + `ldl_solve` (ours) | 0.5 | 24 | 74 | 4 | 1 | — |
| MuJoCo POS_INERTIA (crb + factorM) | 0.3 | 5.2 | 7.6 | 3.8 | 2.0 | 0.19 |
| collision (ours) | 8.6 | 8 | 93 | 4 | 4.2 | **8.2** (71%) |
| MuJoCo POS_COLLISION | 4.2 | 7.7 | 24.1 | 10.6 | 5.5 | 3.1 |
| MuJoCo POS_MAKE (rows) | 1.0 | 5.9 | 6.6 | 8.6 | 4.9 | 0.64 |

(MuJoCo's `POS_PROJECT` is `mj_projectConstraint`: `AR = J M⁻¹ Jᵀ`, built
only under a dual solver or `noslip_iterations > 0` — dog has `noslip=4`,
humanoid.xml says `solver="PGS"`. ⚠ The humanoid row compares our Newton
against MuJoCo's PGS at 4.9 iterations; every other row is Newton vs Newton.)

**Two functions are the whole story past 20 dofs, and neither is collision.**

1. **`solve_newton` — the CPU Newton, `_newton_solve_env`
   (`solver/newton_solve.mojo:784–3103`) — is DENSE IN `nv`.** Three sites
   call `chol_factor_inline(H, L_chol, nv)` on a full `nv×nv` Hessian
   (`:1029`, `:1756`, `:2223`); `H` is built from an `nv×nv` `M_local` copy
   (`:831`, `:1524`); seventy `range(nv)` loops. On park_k9 that is 70.9 µs
   against MuJoCo's **1.5 µs** for six friction rows — **47×** — on a scene
   where nine of the ten kinematic trees are 6×6 diagonal blocks.

   ⚠ **The block-diagonal campaign never touched this function.** PN2a–e and
   F3 (`docs/BLOCK_DIAGONAL_MASS_MATRIX_IMPLEMENTATION.md` §1) segmented the
   GPU kernel `_newton_blocked_fields_kernel` (`:3394+`) — `build_dof_segments`
   and the per-block Cholesky live there and only there (`:4451`, `:4784`).
   The CPU function takes the `trees` operand (`:843`) and does not read it.
   So the CPU path is the un-segmented twin of a kernel whose segmented
   arithmetic is already written and gated bit-exact on the multi-tree arm
   (`85bd3150`).

   ⚠ But **blocks explain only the park rows.** humanoid_cmu and dog are ONE
   tree each, so segmentation buys them nothing, and they are still 11×/26×
   MuJoCo on this function at 1.75 / 6.1 solver iterations. That excess is
   per-iteration cost — the `H` rebuild, the `M_local` copy, the line search
   — and **it is not attributed inside `solve_newton` on CPU**; `sample`
   sees one inlined 2 300-line function. The GPU bisect found the line search
   at 54% of Newton and the tid-0 setup at 31% (implementation doc §2, "THE
   ANSWER, FOR REAL"); the same serial probes (`NEWTON_SERIAL_PROBE`) are the
   way to split this one before touching it.

2. **`compute_m_inv` builds a full dense `M⁻¹` every step, and MuJoCo's Newton
   never forms one.** 231 µs of humanoid_cmu's 768, 648 of dog's 3003, 48 of
   humanoid's 199 — **24–30% of every big model** — for an `O(nv³)` product
   whose MuJoCo counterpart is `mj_diagApprox`
   (`engine_core_constraint.c:1720`): joint limits and dof friction read
   **`dof_invweight0`**, a MODEL-TIME constant (`:1876`, `:1880`); contacts
   read `body_invweight0`; `M⁻¹` appears only inside `mj_projectConstraint`
   (`:3096`), as solves against the sparse `qLD`, and only when a dual solver
   or noslip asks for `AR`.

   Ours reads `m_inv[dof, dof]` for the limit and friction rows (`:1361`,
   `:1424`, `:1585` — the friction row falls back to it only when
   `dof_invweight0 < 1e-10`, the limit row reads it outright), hands the whole
   matrix to the weld-equality rows (`w_MinvJ`, `:1536`), to `noslip`
   (`:1962`), and to the CG / island-PGS solvers. ⚠ **Grep before replacing:
   thirteen files consume `m_inv`** (`constraints/*`, `solver/*`, the three
   integrators). The diagonal is `dof_invweight0` per MuJoCo — check the limit
   row's gate first, since MuJoCo reads the constant where we read the exact
   per-step diagonal; `w_MinvJ` and noslip are `ldl_solve` per row.

   `_m_inv_env` (`dynamics/ldl.mojo:487`) is ONE body for both targets —
   `compute_m_inv[target]` calls it on CPU and launches
   `_m_inv_fields_kernel` around it on GPU — so removing it removes the GPU
   kernel P2 block-restricted (implementation doc §1, "ldl ×1.41") as well.

3. **Collision is the SO-ARM101 story and only that.** 8.2 of an 11.6 µs
   step against MuJoCo's 3.1 — 2.6×, down from 5.8× (§10) — and the entire
   5.1 µs gap on that model. §12.1's support-walk diagnosis stands; nothing
   here changes it. On every model over 20 dofs collision is under 3%.

### 13.2 What this says about the GPU blocked kernels

The user's question was whether a CPU pass would feed the GPU work. Three ways
it does, one way it does not:

* **The CPU Newton is a replay of PN2a–e with no shared memory, no thread
  count and no `Je` spill** — the same `build_dof_segments` table, the same
  per-block Cholesky, and `test_ldl_blocked`-style byte comparison on every
  single-tree model. It is the cheapest place to find out what the segmented
  arithmetic is worth when nothing else is in the way, and the park rows give
  it a clean `nv`-sweep control that the GPU probe never had (its k=0 was
  launch-bound; here k=0 is 11.4 µs of real work).
* **`compute_m_inv` is one function on both targets.** The GPU campaign spent
  P2 making it `sum(bn³)` instead of `nv³`; the CPU numbers say the right size
  is zero.
* **A single-tree model is 11× MuJoCo on the solver with blocks fully
  applied** (humanoid_cmu: one tree, 1.75 iterations). Whatever that is, the
  blocked kernel has it too — the implementation doc's F3 tail (setup 31%,
  line search 54%) is the same shape. Splitting `_newton_solve_env` with
  serial probes is a CPU measurement that answers a GPU question.
* What it does NOT feed: threadgroup budgets, `THREADS`, the `Je` spill
  boundary. Those are GPU-only and the CPU says nothing about them.

### 13.3 Ranked, with the number each is worth

1. **Kill `compute_m_inv`** — 24–30% of every model past 20 dofs, one body
   for both targets, and the reference says the matrix should not exist.
   Gate: `dof_invweight0` vs our `m_inv` diagonal on the row it feeds, then
   bit-exact trajectories on every shipped model. Risk: thirteen consumers.
2. **Segment `_newton_solve_env`** — the park rows (70.9 µs at k=9 against
   1.5) and every multi-object task scene; the arithmetic exists and is
   gated. Risk: LOW, it is a CPU-gateable loop-bounds change.
3. **Split `solve_newton` on a single tree** (humanoid_cmu) with serial probes
   before optimising anything inside it. The record on guessing this
   function's internals is two probes wrong out of three.
4. The support walk on big hulls (§12.1) — unchanged, and only worth it on
   the arms.

### 13.4 How to reproduce, and the three things that went wrong doing it

```bash
pixi run bash scripts/physics3d_cpu_vs_mujoco.sh                 # build + 3 rounds + table
ROUNDS=5 MODEL_GROUPS=so101 SKIP_BUILD=1 OUT=... pixi run bash scripts/physics3d_cpu_vs_mujoco.sh
pixi run python benchmarks/physics3d_mujoco_phases.py <xml> 20000 "" 0 [warmup]
<bin> <model> 200 60000 & sample $! 10 1 -f s.txt; python3 benchmarks/physics3d_sample_top.py s.txt
```

Ours: `benchmarks/physics3d_cpu/{harness,bench_gym,bench_so101,bench_contact}.mojo`
— one integrator step (`apply_actions` + `integ.step["cpu"]`) through
`Phyics3dEnv`, the production facade. MuJoCo: `benchmarks/physics3d_cpu_vs_mujoco.py`,
timed as ONE `mj_step(m, d, nstep)` call so no interpreter is in the loop.
Both: `mj_resetData` state, `ctrl = 0.1`, 2000 warmup, counters from their own
loop, 20 000 timed steps. Three binaries, ~2 min each to build.

* ⚠ **The park props FALL.** They are parked at `z = 50 m` and the first lands
  at step 1596 (MuJoCo, `ctrl=0.1`). A 20 000-step horizon measured a contact
  scene, and at k ≥ 6 our side pinned at **`MAX_CONTACTS = 16` while MuJoCo
  counted 24 / 36** — the table's `!! ncon differs` flag is what caught it.
  The park rows now run 100 + 700 + 700 steps × 8 in-process resets. Print
  the work counters on both sides; a ratio without them is two problems.
* ⚠ **`GROUPS` is a read-only bash builtin.** The first full sweep produced
  zero rows and an empty table with no error. The variable is `MODEL_GROUPS`.
* ⚠ **The integrator must come from the XML, not the config.**
  `So101ParkProbeConfig` inherits `"rk4"` for a scene whose XML says nothing,
  so MuJoCo steps it with Euler; the harness takes the integrator as a
  parameter and both integrators live on the env.
* The trajectories DIVERGE (ant's `qpos[0]` differs in the second digit after
  20 000 steps) — chaotic, expected, and why the contact count and not the
  state is the equivalence check. Hopper agrees to five digits.
* half_cheetah's 138% spread is one inflated round of three
  (`_a_sporadic_row_inflation_makes_an_n1_bench_undecidable`); the MIN is
  the number.

### 13.5 LANDED (2026-09-05): the two ranked items, and the after table

Three changes, in the order §13.3 ranked them, each gated before the next.

**A. The CPU Newton walks each row's nonzero dofs** (`solver/primal.mojo`,
`SPARSE`; `_newton_solve_env` builds `je_n` / `je_ix` once the rows are
final). The Hessian update `H += D·JᵀJ` was `num_edges × nv²` per iteration —
78 × 62² on humanoid_CMU for rows that touch a dozen dofs — and is now
`Σ nnz²`; `Jv`, the edge forces and the warm-start cost walk the same lists.
**B. The CPU Newton factors `H` per kinematic-tree segment**
(`build_dof_segments_p`, the pointer twin of the blocked kernel's builder;
`chol_factor_seg` / `chol_solve_seg` per segment; `Ma`, `Mv` restricted to the
segment). Both A and B sit behind `TREE_AWARE`, passed `True` by the CPU
dispatcher only — the GPU legs compile the byte-identical dense body they
always did, because a per-thread index list is a frame they cannot afford.
**A + B are BIT-EXACT: all fourteen models print the same final-state
checksum as the baseline binary** (`qsum` in the harness's `RESULT` line),
by the exact-zero argument `cholesky.chol_factor_seg` documents.

**C. The dense `M⁻¹` is no longer formed under Newton.** The joint-limit and
tendon rows read `diag(M⁻¹)` / `J M⁻¹ Jᵀ` only to round-trip R
(`1/(1/(K+R)) - K`); they now set `D = 1/R` outright, which is MuJoCo's
`efc_D` (`engine_core_constraint.c:2259`) priced by `*_invweight0`
(`mj_diagApprox`, `:1720`). With that, `compute_m_inv` is skipped in both
integrators when `CONTACTS and SOLVER == "newton" and NOSLIP_ITER == 0` and
the model has no equality constraints (`d.dims.get_nequality()`, a RUNTIME
read so the studio's dynamic leg decides the same way). **C is not bit-exact
and is not meant to be** — it moves D by the round-trip's rounding, toward
the reference. Gated on 17 MuJoCo / parity tests (below).

| model | nv | before | after | **speedup** | MuJoCo | ratio before → after |
|---|---|---|---|---|---|---|
| humanoid_cmu | 62 | 768.3 | **254.2** | **3.02×** | 67.8 | 10.7× → **3.75×** |
| park_k9 | 60 | 94.3 | **33.3** | **2.83×** | 8.76 | 10.7× → **3.81×** |
| park_k6 | 42 | 48.0 | 23.4 | 2.05× | 7.76 | 6.2× → 3.0× |
| dog_stand | 79 | 3003 | 1937 | 1.55× | 225.7 | 13.0× → 8.6× |
| humanoid | 23 | 199.2 | 119.6 | 1.67× | 77.4 | 2.48× → 1.55× |
| ant | 14 | 67.5 | 51.3 | 1.32× | 32.4 | 2.02× → 1.58× |
| park_k3 | 24 | 22.9 | 15.9 | 1.44× | 6.88 | 3.3× → 2.3× |
| walker2d | 9 | 38.8 | 33.0 | 1.17× | 23.4 | 1.60× → 1.41× |
| hopper / half_cheetah / sawyer / so_arm101 / park_k0 | ≤15 | | | 1.00–1.04× | | unchanged |

(µs per physics step, MIN of 3 interleaved rounds, same protocol as §13;
`scripts/physics3d_cpu_vs_mujoco.sh`. ⚠ The dog row was re-measured alone:
the laptop slept mid-sweep and one round came back at 25× — the `spread`
column is what caught it.)

The park sweep's excess over k=0 is now **0.0149 / 0.0094 / 0.0076 per
Δnv²** and FALLING — it is no longer quadratic in the dofs. What is left at
k=9 (33.3 µs against so_arm101's 11.2) is the `nv²` copies the Newton still
makes (`M_local`, `H = M`, the zeroing of `L`), the collision pass, and the
Euler step's own `nv`-sized passes.

**Which gates ran, all green:** `test_frictionless_contact_pyramidal`
(8e-17), `test_impratio_pyramidal_vs_mujoco`, `test_humanoid_limits_fields_vs_mujoco`,
`test_walker2d_contacts_vs_mujoco`, `test_friction_dof_rows_vs_mujoco`,
`test_limit_solref_per_joint` (5.7e-14), `test_newton_warmstart_vs_mujoco`,
`test_constraints_vs_mujoco`, `test_elliptic_condim46_vs_mujoco`,
`test_rk4_newton_fields`, `test_equality_tendon_fields`,
`test_newton_solve_fields`, `test_newton_float32_tracks_float64`,
`test_newton_blocks`, `test_cholesky_segmented`,
`test_newton_solves_on_runtime_dims`, `test_newton_no_constraint_rows`,
`test_tree_blocks_vs_mujoco`, `test_newton_both_legs`.

⚠⚠ **THE ONE THAT FAILED FIRST, AND WHY IT MATTERS.**
`test_limit_solref_per_joint` came back with `|d qacc| = 153` on a 2-dof
arm — its limit row had gone INERT (ours equalled the smooth acceleration
for two different stiffnesses). Bisected by toggling: `TREE_AWARE=False`
changed nothing; forcing `compute_m_inv` back on fixed it. The step it
drives is `step["cpu", CONTACTS=False]`, and with `CONTACTS=False` the seam
runs the STANDALONE `solve_limits` / `solve_friction` stages — Gauss-Seidel
over `J M⁻¹ Jᵀ`, which read the inverse for real. The predicate now includes
`CONTACTS`. The lesson is §5's: **enumerate the readers by the CODE PATH, not
by the function list** — every `m_inv[` read in the Newton files was gone,
and the reader was in a stage the Newton never calls.

**What is left, re-ranked on the after profile** (`sample`, patched
binaries):

1. **dog_stand keeps its `M⁻¹`** (28% of 1.9 ms) because `noslip_iterations=4`
   needs `M⁻¹ Jᵀ` per row. MuJoCo pays for the same thing (`POS_PROJECT`
   80 µs of its 226). Ours could form `M⁻¹ Jᵀ` by `ldl_solve` per row instead
   of the full inverse — `nrows × nv²` against `nv³`, about 2× on that term.
2. **The dense Cholesky on a single tree** — humanoid_CMU and dog are one
   tree each, so B buys them nothing there, and `chol_factor_seg`'s inner
   dot product is scalar (Mojo does not autovectorise, §7.1). A SIMD dot
   reassociates the sum and is therefore NOT bit-exact; it belongs in a
   batch gated like C.
3. **The `nv²` copies inside the Newton** (`M_local`, `H = M`, `L` zeroing,
   three per iteration) — segment-restrict them as PN2d did for `L_sh`.
   Bit-exact, small.
4. Collision on the arms (§12.1), unchanged at 2.6×.

### 13.6 The stage probe: what is inside `solve_newton`, per model

`sample` sees `_newton_solve_env` as one inlined body, so `_CPU_PROBE`
(`solver/newton_solve.mojo`) now times its stages with `perf_counter_ns` and
prints one `[probe]` line per solve; it is a comptime flag, off and free by
default. Run with it on (build the bench binary, fold with `awk`):

```
    awk '/^\[probe\]/ {n++; for(i=3;i<=NF;i+=2) s[$i]+=$(i+1)} END {...}'   # §13.6 of the tree has the full one-liner
```

µs per SOLVE (one Euler step = one solve), after §13.5, M1 Pro, f32:

| stage | humanoid_cmu | dog_stand | reassemble3 | reassemble5 | park_k9 |
|---|---|---|---|---|---|
| iterations / solve | 3.3 | 3.2 | **18.3** | **35.4** | 2.1 |
| rows | 3.7 | 6.5 | 4.1 | 11.4 | 0.3 |
| setup (`M_local`, `Ma`, warm start) | 31.8 (16%) | 48.9 | 27.5 | 80.8 | **6.2 (38%)** |
| Hessian build | 27.9 (14%) | 32.2 | 25.5 | 84.5 | 3.4 (21%) |
| Cholesky (+solve) | **99.8 (51%)** | 210 (22%) | 14.2 | 56.4 | 5.0 (31%) |
| `M·s`, `J·s` | 16.2 | 27.2 | 165 (12%) | 865 (12%) | 0.7 |
| line search | 8.2 | 9.5 | 184 (13%) | 633 (9%) | 0.2 |
| post-step update | 9.1 | 9.0 | 214 (16%) | 1200 (17%) | 0.5 |
| **H rebuild + refactor** (elliptic) | — | — | **536 (39%)** | **3284 (46%)** | — |
| noslip | — | **617 (64%)** | 207 (15%) | 904 (13%) | — |
| **solve total** | 197 | 961 | 1376 | 7119 | 16.3 |

(MuJoCo, same scenes: 7.0 iterations on both reassemble scenes; whole
CONSTRAINT phase 40 µs on humanoid_cmu, 79 + 81 on dog, 1972 + 592 on
reassemble5.)

**Four models, four different answers — which is why the probe was worth
building before touching anything:**

1. **humanoid_cmu is the dense Cholesky**: 100 µs of 197, one 62×62 factor
   per iteration at ~1.3 GFLOP/s — scalar, `chol_factor_seg`'s inner dot
   product does not vectorise (§7.1). One tree, so segments cannot help.
   ⚠ MuJoCo's `jacobian="auto"` goes SPARSE at nv ≥ 60, so on this model
   and dog the reference factors a sparse H; ours is dense on both.
2. **dog is noslip**: 617 µs of 961 in `noslip_pyramidal`, against MuJoCo's
   whole constraint + projection phases at 160 µs. That routine, not the
   Newton loop, is dog's target; its own stages are not split yet.
3. **The reassemble scenes are the ELLIPTIC path, and the first number is the
   iteration count: 18 and 35 against MuJoCo's 7 on the same scenes.** Each
   iteration rebuilds and refactors the cone Hessian (`cone_live` is true
   whenever any contact sits in its cone zone), and the rebuild walks DENSE
   contact Jacobians — `nefc × nv²`, 900 × 33² on reassemble5 — the same
   shape §13.5-A removed from the pyramidal path. Two separate items: the
   per-iteration cost (sparse rows, bit-exact, same recipe) and the
   iteration count (a convergence question against the reference, not a
   speed one). ⚠ The contact COUNT also differs: ours 69 / 125 to MuJoCo's
   93 / 232 on average (both under our caps of 256 / 512), which is a
   fidelity question outside this pass and is recorded in the table's flag.
4. **park_k9 is the `nv²` copies now**: setup 38% + Hessian build 21% at
   six rows are `M_local = M`, `H = M` and the zeroing of `L` — 3 × 3 600
   entries per iteration on a scene whose arithmetic is now ~300 flops.
   Bit-exact to segment-restrict (PN2d did it for `L_sh`).

The two reassemble rows, added to the sweep (2000 timed steps × 3 rounds,
`!! ncon differs` on both):

| model | nv | ours µs | MuJoCo µs | ratio | ncon ours / mj | mj nefc | mj niter |
|---|---|---|---|---|---|---|---|
| reassemble3 | 21 | 4165 | 482 | **8.6×** | 69.2 / 93.3 | 293 | 7.0 |
| reassemble5 | 33 | 7320 | 2427 | **3.0×** | 124.6 / 232.2 | 710 | 7.0 |

Ours there is 81% `solve_newton` (elliptic), 6–11% `noslip_elliptic`, ~10%
collision.

### 13.7 LANDED (2026-09-05): the elliptic path — sparse rows, and an exit it never had

Two changes on the elliptic Newton, in the order §13.6 ranked them.

**D. Sparse contact rows** (`f6b67b73`, bit-exact). Each contact carries its
nonzero-dof list (the union of its normal and tangent rows' supports); the
cone-Hessian assembly, `J·s`, `Jᵀf`, the jar recomputation and the warm-start
cost walk it. Same recipe as §13.5-A; `_cn_len` / `_cn_dof` make one loop
body serve the dense GPU walk and the sparse CPU one. reassemble5 7320 → 5113
µs/step, reassemble3 4165 → 3142. Checksums identical.

**E. The improvement exit.** The re-probe after D still showed 18 and 35
iterations per solve against MuJoCo's 4–7 Newton iterations (its
`solver_niter` counts the five noslip sweeps too). The per-iteration trace
(`_ELL_TRACE`) showed why: in float32 the scaled gradient falls from 3e6 to
~0.3 in eight iterations and then WANDERS between 0.1 and 0.6 — the float32
rounding of forces of order 1e6 — while the tolerance is 1e-8, so the only
exits left were a vanished `alpha` or the 100-iteration cap, and one solve in
six hit the cap. **The same scene in float64, same loop, converges in 5–8**
(`reassemble3_f64` in the bench), so it is the precision floor and not the
direction. The loop had no `improvement` test — `mj_solPrimal` stops on
`(improvement > 0 && improvement < tol) || gradient < tol`
(engine_solver.c:2279), and the pyramidal path has had one since it was
written. It now prices the total cost once per iteration (one closure,
`_total_cost`, shared with nothing that could drift) and stops on
`improvement < tol`, without MuJoCo's `> 0` guard: a non-positive change is
the noise floor, and in float64 it does not occur before convergence.

| | iterations / solve | `solve_newton` µs | step µs (3 rounds) | vs MuJoCo |
|---|---|---|---|---|
| reassemble3 before | 18.3 | 1376 | 4165 | 8.6× |
| reassemble3 after D | 18.3 | 1012 | 3142 | 6.4× |
| **reassemble3 after E** | **3.1** | **356** | **738** | **1.54×** |
| reassemble5 before | 35.4 | 7119 | 7320 | 3.0× |
| reassemble5 after D | 35.4 | 4631 | 5113 | 2.1× |
| **reassemble5 after E** | **3.9** | **1408** | **2223** | **0.92×** |

(⚠ reassemble5's 0.92× is against a MuJoCo step that carries 232 contacts to
our 129; the contact-count gap is still open and still flagged.) Sawyer,
the other elliptic model, is unchanged at 23.9 µs.

**Gates, all green after E:** `test_elliptic_condim46_vs_mujoco` (7e-11),
`test_noslip_elliptic_vs_mujoco` (9e-12), `test_newton_both_legs`,
`test_newton_float32_tracks_float64`, `test_reassemble_3_bricks_vs_dm_control`
and `test_reassemble_5_bricks_vs_dm_control` (1e-15 on both, contact cases
included), `test_reassemble_5_tower_energy_vs_mujoco`,
`test_sawyer_mesh_rest_vs_mujoco`, `test_constraints_vs_mujoco`,
`test_newton_solve_fields`.

**After E the reassemble solve is noslip**: `noslip_elliptic` is 208 of 356 µs
(58%) on reassemble3 and 947 of 1408 (67%) on reassemble5; the elliptic H
rebuild is 15%, everything else under 8%. With dog's `noslip_pyramidal` at
64% of its solve (§13.6), noslip is now the largest single term on three of
the five heavy models, and it is next.

### 13.8 LANDED (2026-09-05): noslip hoists `M⁻¹Jᵀ` and walks the nonzero dofs — bit-exact

Item 3 of the §13.6 order. Both `noslip_pyramidal` and `noslip_elliptic`
recomputed `M⁻¹J_rowᵀ` — an `nv²` product — for every friction row (or
tangent row) on every sweep, and `_minv_jt`'s own docstring said so: "if the
CPU path ever needs the speed, hoist it: J and M do not change during the
sweep". They also walked dense Jacobians in every dot product and every jar
refresh. Under `CACHE` the CPU path now forms every row's `M⁻¹Jᵀ` once per
solve (`E_CAP × V_CAP` on the pyramidal side, `T_CAP × V_CAP` on the elliptic
one — the slab the note says a per-env GPU frame cannot hold, so the GPU legs
keep both knobs off and pass one-element placeholders); under `SPARSE` the
products run over the Newton's own row lists (`je_n`/`je_ix`, `cn_n`/`cn_ix`).
Bit-exact by the exact-zero argument: dog, both reassemble scenes and Sawyer
print their previous checksums.

| model | before | after | speedup | vs MuJoCo |
|---|---|---|---|---|
| dog_stand | 1937 | **1126** | 1.72× | 13.0× at the start of the day → **5.0×** |
| reassemble3 | 738 | **613** | 1.20× | 8.6× → **1.28×** |
| reassemble5 | 2223 | **1531** | 1.45× | 3.0× → **0.63×** |
| sawyer_reach | 23.9 | 23.9 | — | 1.59× |

(µs per physics step, MIN of 3 interleaved rounds.) Gates green:
`test_noslip_vs_mujoco` (7e-17), `test_noslip_elliptic_vs_mujoco` (9e-12),
`test_noslip_blocked_kernel` (the GPU call site with its placeholders),
`test_friction_dof_rows_vs_mujoco`.

⚠ `tests/dm_control/test_dog_gpu_vs_cpu.mojo` fails on this Mac with "Compute
function exceeds available stack space" — **pre-existing and documented in
its own header**: Apple builds that kernel and cannot run it, dog's NV=79 is
past Metal's per-thread stack ceiling, NVIDIA is its only target. It was run
to check the call-site change compiles, which it does. While there, the two
`V_CAP` segment arrays the tree-aware CPU path allocates are now sized 1 on the
GPU legs, so the shared body did not grow their frames at all.

**The day's ledger, whole tree** (before = §13's table; after = the latest
three-round row for each model):

| model | nv | before µs | after µs | speedup | vs MuJoCo before → after |
|---|---|---|---|---|---|
| dog_stand | 79 | 3003 | 1126 | 2.7× | 13.0× → 5.0× |
| humanoid_cmu | 62 | 768 | 254 | 3.0× | 10.7× → 3.75× |
| park_k9 | 60 | 94.3 | 33.3 | 2.8× | 10.7× → 3.8× |
| reassemble5 | 33 | 7320 | 1531 | 4.8× | 3.0× → 0.63× |
| reassemble3 | 21 | 4165 | 613 | 6.8× | 8.6× → 1.28× |
| humanoid | 23 | 199 | 120 | 1.7× | 2.48× → 1.55× |
| ant | 14 | 67.5 | 51.3 | 1.3× | 2.02× → 1.58× |
| walker2d | 9 | 38.8 | 33.0 | 1.2× | 1.60× → 1.41× |
| ≤ 15 dofs, few contacts | | | | ~1.0× | unchanged |

**What is left, on the after profiles:** humanoid_cmu's dense scalar Cholesky
(51% of its solve; a SIMD dot is not bit-exact and needs its own gate batch),
dog's remaining `M⁻¹` (it still forms the full inverse for noslip — `M⁻¹Jᵀ`
by `ldl_solve` per row would replace an `nv³` with `rows × nv²`), park_k9's
three `nv²` copies per iteration (bit-exact), and the reassemble contact
count (ours 68 / 129 to MuJoCo's 93 / 232 — a fidelity question, not a
speed one, and the reason the 0.63× is not a like-for-like number).

### 13.9 LANDED (2026-09-05): the three nv² copies — bit-exact

Item 4 of §13.6 (park_k9's remaining 59%). Under `TREE_AWARE` the pyramidal
Newton now copies only the in-segment entries of `M` into `M_local` and of
`M_local` into `H` (every reader of either is already restricted to the dof's
segment, and `M` is exactly zero elsewhere), and zeroes `L` once per solve
instead of once per factorisation (`chol_factor_seg` writes every in-segment
lower entry it will read, and nothing reads an off-segment or upper one).
Nothing that was read has changed a bit: ten models — the four park scenes,
SO-ARM101, walker2d, hopper, humanoid, ant, humanoid_cmu, dog — print their
previous checksums.

| model | nv | before µs | after µs | vs MuJoCo |
|---|---|---|---|---|
| park_k3 | 24 | 15.9 | 15.3 | 2.21× |
| park_k6 | 42 | 23.4 | 20.8 | 2.67× |
| park_k9 | 60 | 33.3 | **27.7** | 10.7× at the start of the day → **3.15×** |

The park excess over k=0 is now 4.1 / 9.7 / 16.6 µs at k = 3 / 6 / 9 —
**linear in the added dofs, ~0.3 µs per dof**, against a MuJoCo excess of
0.6 / 1.5 / 2.5. What is left of it sits outside the Newton: the collision
pass (~9 µs at k=9), the still-dense `compute_m_inv` (this model has
`frictionloss` rows, no equalities, no noslip — the inverse is skipped; the
remaining `nv`-sized passes are the Euler step's own), and the per-solve row
scan that builds the sparsity lists.

### 13.10 LANDED (2026-09-05): the tree-ordered LDL — MuJoCo's factorisation, on the CPU

Dog's remaining `M⁻¹` (§13.8: it still forms the full inverse for noslip).
Replacing it with per-row solves against OUR factor would not have helped:
`_ldl_factor_env` eliminates forward, and on a kinematic tree that fills in
between siblings, so `L` is dense within a tree, a solve is O(nv²), and 63
rows of solves cost what the inverse costs. MuJoCo's win is the ORDER:
`mj_factorI` (`engine_core_smooth.c:1973`) eliminates from the last dof
backwards and gets `M = Lᵀ D L` with `L` on M's own sparsity — row `k`
nonzero at `k`'s ancestors only, no fill — so `mj_solveLD` (`:2113`) is O(nC)
and the inverse O(nv² · depth) instead of O(nv³).

Three pieces:

* **`Model.dof_parentid`**, MuJoCo's table, built in `fields_build` beside
  `trees` from the dof→body map (dofs of one body chain in order, a body's
  first dof hangs from the last dof of the nearest ancestor body with any).
  Gated entry for entry against `m.dof_parentid` on the tree-block model
  list: **28 models, 590 dofs, 0 differing**
  (`test_dof_parentid_vs_mujoco`).
* **`_ldl_factor_tree_env` / `_ldl_solve_tree_env` / `_m_inv_tree_env`**
  (`dynamics/ldl.mojo`): the reference's three loops on our dense `[nv*nv]`
  storage, walking the parent table (read into integers once per call — the
  chain walk is the whole cost, and a float→int per hop was a third of it).
  The CPU dispatchers select them when `MODEL_META_IDX_NTREE > 0` and keep
  the dense trio otherwise and on every GPU leg.
* ⚠ **A DIFFERENT CONVENTION IN THE SAME BUFFER.** `scratch.L` now holds
  `LᵀDL`'s L on the CPU and `LDLᵀ`'s L on a GPU. A factor is only ever read
  by the solve behind the same dispatcher, so nothing mixes — but a test
  that compares `L` across the two would be comparing two matrices. The one
  that did compare `L` across legs (`test_dispatchers_both_legs`) compares
  two CPU legs and failed for a different reason: its record copier copied
  `meta` (with `NTREE`) but neither topology table, so the dynamic arm
  claimed a table of all roots. It copies both now.

Not bit-exact against the dense trio (a different rounding of the same
inverse) and gated as such: `dof_invweight0` / `body_invweight0` at qpos0
against MuJoCo (`test_constraints_vs_mujoco`: 1e-16 on ant, 1e-14 on
humanoid), `test_frictionless_contact_pyramidal` (8e-17), `test_noslip_vs_mujoco`
(7e-17), `test_walker2d_contacts_vs_mujoco`, `test_humanoid_limits_fields_vs_mujoco`,
`test_newton_warmstart_vs_mujoco`, `test_elliptic_condim46_vs_mujoco`,
`test_reassemble_3_bricks_vs_dm_control` (1e-15), `test_rk4_newton_fields` and
`test_cfrc_ext_batched_vs_cpu` (CPU vs GPU, the two factorisations against
each other), `test_newton_both_legs`, `test_newton_solves_on_runtime_dims`,
`test_ldl_blocked`, `test_dyn_dims_ldl`, `test_dispatchers_both_legs`.

| model | nv | before µs | after µs | speedup | vs MuJoCo |
|---|---|---|---|---|---|
| dog_stand | 79 | 1126 | **722** | 1.56× | 13.0× at the start of the day → **3.1×** |
| reassemble5 | 33 | 1531 | 1318 | 1.16× | 0.54× |
| reassemble3 | 21 | 613 | 699 | (noise band; both 1.3–1.5×) | 1.43× |
| humanoid_cmu | 62 | 254 | 249 | — | 3.57× |
| everything ≤ 23 dofs | | | | ~1.0× | unchanged |

Dog's profile after: Newton 49%, `compute_m_inv` 23% (≈170 µs for ~75k
chain hops, ≈2 ns a hop — it is a pointer chase now, not arithmetic), Euler
11%, `ldl_factor` under 2.4%.

**The whole tree, start of the day → now** (three interleaved rounds each):

| model | nv | before | after | speedup | vs MuJoCo |
|---|---|---|---|---|---|
| dog_stand | 79 | 3003 | 722 | 4.2× | 13.0× → 3.1× |
| humanoid_cmu | 62 | 768 | 249 | 3.1× | 10.7× → 3.6× |
| park_k9 | 60 | 94.3 | 27.4 | 3.4× | 10.7× → 3.1× |
| reassemble5 | 33 | 7320 | 1318 | 5.6× | 3.0× → 0.54× |
| reassemble3 | 21 | 4165 | 699 | 6.0× | 8.6× → 1.43× |
| humanoid | 23 | 199 | 121 | 1.6× | 2.48× → 1.52× |
| ant | 14 | 67.5 | 51.1 | 1.3× | 2.02× → 1.55× |
| walker2d | 9 | 38.8 | 34.1 | 1.1× | 1.60× → 1.41× |

**Left, on the after profiles:** humanoid_cmu's dense scalar Cholesky (half
its solve; not bit-exact to vectorise, own gate batch), dog's inverse as a
pointer chase (the reference does not form it at all — noslip's `M⁻¹Jᵀ`
as 63 tree solves would be ~4× fewer hops than 79 columns), the reassemble
contact count (68 / 127 to MuJoCo's 93 / 232 — fidelity, and why the 0.54× is
not like-for-like), and porting the tree order to the GPU legs, which would
retire the dense trio and the `LDLᵀ`/`LᵀDL` split.

### 13.11 The reassemble contact-count gap was the PROTOCOL, not the engine

§13.6 flagged ours at 68 / 127 contacts against MuJoCo's 93 / 232 on the
reassemble scenes and kept the caveat on every ratio since. A per-step diff of
the contact sets as body pairs (`benchmarks/physics3d_cpu/contact_probe.mojo`,
`physics3d_contact_probe.py`, `physics3d_contact_diff.py`) settled it in one
run:

* **The sets differ at step 0, and the state differs by 0.5 after one 2 ms
  step** — not drift, an initial-state problem.
* From `mj_resetData` every brick's free joint is at `qpos0 = 0`, so **all
  three bricks sit at the origin INSIDE each other and inside the welded one**.
  MuJoCo reports 28 contacts between each pair of coincident hulls and pushes
  them apart violently; our GJK/EPA reports NONE for coincident identical
  meshes (a zero-measure degenerate case the task never produces) and the two
  runs part company at step 0. Every contact-count and ratio number for these
  two rows before this section compared two different scenes.
* The task poses the scene through its reset — the free bricks through
  `qpos`, the welded brick through the MODEL (`body_pos`), which is why
  `sf.qpos0` cannot pose it. The harness now has `TASK_POSE`: the reassemble
  rows keep the env's own reset and write it (`QPOS`, `QVEL`, and the jointless
  bodies' poses) to a file the MuJoCo twin applies.

From the task pose, float64, ctrl = 0.1 on both sides:

| | step 0 | mean over the run | pairs that differ |
|---|---|---|---|
| reassemble3, 3000 steps | 48 / 48, identical sets | ours 54.7, MuJoCo 51.9 | stud multiplicity on `duplo2x4/~duplo2x4_2/` (+3 ours) and `_2/~_4/` (+1 MuJoCo) |
| reassemble5, 2000 steps | 96 / 96, identical sets | ours 105.3, MuJoCo 116.4 | the same four stacked pairs, MuJoCo ~10% more stud contacts |

The per-dof state after ONE step from the task pose agrees to 2e-9 on every
arm dof and to 2e-7 on the free bricks' quaternions — the remaining fidelity
thread is the multi-contact manifold on the interlocking studs (how many
contact points a stud/tube pair yields), not which pairs collide.

**The bench rows, re-measured from the task pose** (this is the sweep's
protocol for these two rows from now on; `scripts/physics3d_cpu_vs_mujoco.sh`
passes the pose file):

| model | nv | ours µs | MuJoCo µs | ratio | ncon ours / mj | mj nefc | mj niter |
|---|---|---|---|---|---|---|---|
| reassemble3 | 21 | 399 | 219 | **1.82×** | 54.3 / 51.6 | 164 | 6.0 |
| reassemble5 | 33 | 1053 | 645 | **1.63×** | 109.4 / 116.7 | 359 | 6.0 |

⚠ These REPLACE §13.7–§13.10's 0.54–1.43× for the reassemble rows. Those were
not wrong measurements; they were measurements of a scene with the bricks
inside each other, which is cheaper for us (no stud contacts) and dearer for
MuJoCo (28 coincident-hull contacts per pair). MuJoCo's own step on the posed
scene is 219 / 645 µs, not 489 / 2440.

### 13.12 Contact fidelity on the brick piles: at the reference's own tolerance

After §13.11 the remaining differences on the reassemble scenes from the task
pose were small and specific — ~10% fewer stud contacts on reassemble5 and a
7e-6 state difference after ONE float64 step with identical contact sets. Both
were run down with `contact_probe.mojo`'s detail mode (per-contact position,
normal, distance, force, tangent frame; noslip / solver-tolerance / CCD-
tolerance switches; body mass properties) against MuJoCo, and the answer is
that we sit inside MuJoCo's own tolerance and backend spread.

**1. The XML disables MuJoCo's native CCD** (`<flag nativeccd="disable"/>`),
so the reference in those tables was **libccd**, while our GJK/EPA is the port
of MuJoCo's NATIVE routines. With native CCD enabled on the reference:

| step 0, welded brick vs brick 2 | ours | MuJoCo native | MuJoCo libccd |
|---|---|---|---|
| contact distances (µm) | 2.448 / 3.020 / 3.493 | **2.448 / 3.020 / 3.493** | 2.70 / 2.93 / 3.001 / 3.003 |
| witness heights z | 0.02287 / 0.02321 | **0.02287 / 0.02321** | 0.0229 |
| mean contacts, reassemble5, 2000 steps | 105.3 | 109.4 | 116.4 |
| \|Δstate\| after 3000 steps, reassemble3 | ours–native **7e-4** | native–libccd **2.4e-3** | ours–libccd 1.7e-3 |
| \|Δstate\| after 2000 steps, reassemble5 | ours–native 4.4e-3 | native–libccd 5.9e-3 | ours–libccd 1.4e-3 |

Our contact geometry is MuJoCo native's to the printed digit, and our long-run
drift against either backend is no larger than the two backends' drift
against each other.

**2. What was left against native: a yaw-only acceleration on the free
bricks.** With noslip off, every arm dof and every brick translation and tilt
agrees to ≤ 1e-11 after one step; the brick YAW accelerations differ by 0.05
and 0.12 rad/s². Not tolerance (`tolerance = 1e-14` on both sides changed
nothing; MuJoCo converges in ONE Newton iteration), not inertia (mass, inertia
tensor, COM and inertial frame identical to 1e-12), not noslip (off). Per
contact: 18 of 24 contacts agree to 1e-16 in position and to the digit in
distance and force; **six stud/flange contacts** (the 0.55 mm-thick flange
boxes) sit 50 µm apart, with the two distance values 3.02 and 3.49 µm
assigned to different stud/flange pairs and normal forces 0.4% apart. That
0.47 µm is below `ccd_tolerance = 1e-6`, the GJK exit both engines run with.
**With `ccd_tolerance = 1e-12` on both sides the step-0 acceleration
difference falls from 1.2e-1 to 6.6e-5** (2000×), and the yaw dofs carry the
whole of what is left. A 12-gram brick has a yaw inertia of 1.9e-6 kg·m²: the
0.1 rad/s² was a 2e-7 N·m torque, a 2e-5 N force imbalance against a 0.12 N
weight — the size of a GJK tolerance, amplified by a tiny inertia.

**3. Multiplicity.** The base~base box contact between stacked bricks is a
knife edge: resting faces at 1e-9 separation with margin 0, and which of the
box-box candidate points fall on the negative side decides 0 to 3 contacts. Ours
emits 2–3 at depths of 1e-9 where MuJoCo emits 0–1 at 5e-11 — physically
inert either way, and the whole of the "+16k only-ours contacts" over a
3000-step run.

**Conclusion:** no collision defect on these scenes. The remaining
ours-vs-MuJoCo residual is at the reference's own GJK tolerance, and MuJoCo's
two CCD backends disagree with each other by more than either disagrees with
us. Anyone comparing manipulation rollouts against MuJoCo should enable native
CCD on the reference (`m.opt.disableflags &= ~mjDSBL_NATIVECCD`) or accept a
libccd-sized spread.

⚠ Method notes, each of which changed a conclusion here: pose the scene the
way the TASK does (§13.11); diff contact SETS as body pairs before trusting a
count; MuJoCo's `mjcontact.frame[:3]` is the negation of our stored normal;
match contacts by position AND check the distance travels with the match — six
pairs matched to 4 digits and had swapped distances.

### 13.13 LANDED (2026-09-05): the Cholesky's inner product, `W` lanes wide

§7.1 stands — Mojo does not autovectorise, and until today `physics3d` held no
explicit SIMD. The §13.6 probe put half of humanoid_CMU's solve in
`chol_factor_seg`, whose inner loop is a dot product of two contiguous rows of
`L`, and §7.2's objection (gather, loop-carried, short) does not apply to it.

`cholesky.mojo` gains `_dot_seg` / `_dot_rows` / `_axpy_seg` — explicit
`load[width=W]` loops with a scalar tail, `W = 2 × simd_width_of[DTYPE]` — and
a `VEC` flag on `chol_factor_seg`, `chol_solve_seg` and `chol_solve_seg_p`:
the factor's `s = Σ L[i,k] L[j,k]` becomes one SIMD dot per entry, the
solve's forward pass a dot of row `i` against `y`, and its backward pass is
rewritten in AXPY form so it walks a row of `L` instead of gathering a column.
The CPU pyramidal Newton's three matvecs (`Ma`, `Mv`, the warm-start cost)
take the same dot over their segment. `VEC` is set by the CPU pyramidal
Newton only; the GPU legs and the elliptic path compile the scalar loops.

⚠ NOT BIT-EXACT — a `W`-wide accumulator reassociates the sum — and gated
as such: `test_newton_float32_tracks_float64`, `test_frictionless_contact_pyramidal`
(8e-17), `test_noslip_vs_mujoco` (1.5e-16), `test_walker2d_contacts_vs_mujoco`,
`test_humanoid_limits_fields_vs_mujoco`, `test_newton_warmstart_vs_mujoco`,
`test_impratio_pyramidal_vs_mujoco`, `test_cholesky_segmented`,
`test_newton_both_legs`, `test_newton_solve_fields`, `test_rk4_newton_fields`.
⚠ The exclusivity checker refuses the same mutable pointer in two arguments,
which is why the factor's self-dot is a one-pointer, two-offset routine.

| model | nv | before µs | after µs | speedup | vs MuJoCo |
|---|---|---|---|---|---|
| humanoid_cmu | 62 | 249 | **195** | 1.28× | 3.57× → **2.79×** |
| dog_stand | 79 | 722 | **637** | 1.13× | 3.1× → **2.80×** |
| humanoid | 23 | 121 | 96 | 1.26× | 1.52× → 1.22× |
| ant | 14 | 51.1 | 42.6 | 1.20× | 1.55× → 1.31× |
| walker2d | 9 | 34.1 | 29.2 | 1.17× | 1.41× → 1.21× |
| hopper | 6 | 15.8 | 13.5 | 1.17× | 1.07× → **0.93×** |
| park_k9 | 60 | 27.4 | 25.0 | 1.10× | 3.06× → 2.82× |
| reassemble3 / 5, sawyer (elliptic) | | | | — | unchanged |

(Three interleaved rounds; the reassemble rows from the task pose, §13.11.)

**The day, start to finish**, every model past 20 dofs:

| model | nv | 09-04 µs | now µs | speedup | vs MuJoCo, then → now |
|---|---|---|---|---|---|
| dog_stand | 79 | 3003 | 637 | 4.7× | 13.0× → 2.8× |
| humanoid_cmu | 62 | 768 | 195 | 3.9× | 10.7× → 2.8× |
| park_k9 | 60 | 94.3 | 25.0 | 3.8× | 10.7× → 2.8× |
| reassemble5 | 33 | 7320 | 1059 | 6.9× | (degenerate pose) → 1.63× |
| reassemble3 | 21 | 4165 | 399 | 10.4× | (degenerate pose) → 1.83× |
| humanoid | 23 | 199 | 96 | 2.1× | 2.48× → 1.22× |
| ant | 14 | 67.5 | 42.6 | 1.6× | 2.02× → 1.31× |
| walker2d | 9 | 38.8 | 29.2 | 1.3× | 1.60× → 1.21× |
| hopper | 6 | 15.0 | 13.5 | 1.1× | 1.02× → 0.93× |

No model is now more than 2.8× MuJoCo on the CPU; hopper is faster.

**Left:** the elliptic path could take `VEC` too (its Cholesky is ~1%, its
`J·s` and `Jᵀf` walks are sparse gathers — little to gain); the remaining CPU
excess on the big models is the Newton loop's per-iteration passes and noslip,
both now segment- and sparsity-restricted; and everything GPU-side from §13.2
and §13.10 still waits for NVIDIA hardware.

### 13.14 Why MuJoCo is still faster on the CPU, phase by phase, and what would close it

Measured on the final binaries (`sample` for ours, MuJoCo's own `mjTimerStat`
with `bvactive = 0`, the `_CPU_PROBE` stage split inside the Newton). µs per
physics step; RK4 models run four solves a step.

| humanoid_cmu (nv 62, 1 tree, ~14 contacts) | ours | MuJoCo | gap |
|---|---|---|---|
| constraint solve (`solve_newton` / CONSTRAINT + MAKE) | ~95–140 | 46.3 | **~60** |
| Euler finalize: implicit damping refactor + `M·qacc` (/ ADVANCE) | 34 | 4.6 | **29** |
| mass matrix + LDL (/ POS_INERTIA) | 12 | 5.3 | 7 |
| kinematics + cdof + velocities (/ KIN + VEL) | 8 | 4.6 | 3 |
| collision | 7 | 7.7 | 0 |
| **step** | **208** | **70.7** | 137 |

| dog_stand (nv 79, 1 tree, 9 contacts, noslip 4) | ours | MuJoCo | gap |
|---|---|---|---|
| solve incl. noslip | 179 (noslip 58) | 71 + 73 PROJECT | ~35 |
| `compute_m_inv`, kept for noslip | 150 | (inside PROJECT) | ~150 |
| Euler finalize | 71 | 6.8 | 64 |
| collision (meshes) | 72 | 22.7 | 49 |
| **step** | **663** | **200** | 463 |

humanoid (nv 23, RK4): 105 vs 79 — the solve is 58 (4 × 14.4) against 27 + 19.
ant (nv 14, RK4): 47 vs 34 — the solve is 29 against 13.8. park_k9: 25 vs 8.9,
split evenly between solve (8), collision (8.5) and the Euler step (4).

Inside our Newton on humanoid_cmu, per solve (3.3 iterations): Cholesky
38.4 (40%), Hessian build 21.9 (23%), setup 19.4 (20%), line search 5.5,
update 4.8, rows 3.6. MuJoCo's whole CONSTRAINT phase is 40.

**What MuJoCo does that we do not, in the order it costs us:**

1. **It factors the Hessian ONCE per solve and updates it.** Its Newton loop
   (`engine_solver.c:2120`) walks the constraint states after each step and,
   for every row that entered or left the quadratic zone, applies a rank-1
   Cholesky update/downdate with `J_i·√D_i` (`mju_cholUpdate`, O(nv²));
   it refactors only when an update loses rank. On humanoid_cmu the state
   changes are 0–14 rows of 78 per iteration. We rebuild `H = M + JᵀDJ` and
   refactor it from scratch on every iteration: 3.3 factorisations of a
   62×62 where MuJoCo does one plus a handful of rank-1 updates. That is the
   Cholesky's 38 µs AND the Hessian build's 22 µs — 60 of our 95.
2. **Its factor is sparse past 60 dofs.** `jacobian="auto"` flips to sparse at
   nv ≥ 60, so humanoid_cmu and dog get `mju_cholFactorSparse` over H's
   symbolic pattern (the tree plus the contact couplings) — nC-sized, not
   nv²-sized. Ours is dense within a tree, and both big models are one tree.
3. **`mj_Euler`'s implicit damping uses the sparse factor.** Our
   `_finalize_env` (`integrator/euler.mojo:337`) still calls the OLD dense
   `_ldl_factor_env` / `_ldl_solve_env` on `M + h·D` directly — it never went
   through the dispatcher that §13.10 switched — and forms `M·qacc` as a
   dense nv² matvec. 34 µs on humanoid_cmu and 71 on dog against MuJoCo's
   4.6 and 6.8. ⚠ The tree-ordered trio is already in the tree; this is
   plumbing (`dof_parentid` into `_finalize_env`, `use_tree` on `NTREE > 0`).
4. **Its `M·v` is sparse.** `mj_mulM` walks nC entries; our `Ma`, `Mv`, the
   warm-start matvec and the finalize `rhs` are nv² (dense within the
   segment). 3 844 against 952 FMAs each on humanoid_cmu, three or four times
   a solve. Small per call; `dof_parentid` makes it a chain walk.
5. **It allocates nothing per solve.** Ours fills per-solve `Scratch` arrays
   built with `fill=` — `je_ix` alone is `E_CAP × V_CAP` Ints, ~118 kB on
   humanoid_cmu, zeroed EVERY solve (`fill=0`; `Scratch(uninitialized=)` skips
   the fill on the static leg, `fill=` does not), plus `kind_e`, `R_e`,
   `floss_e`, `state_e`, `seg0/seg1`, `cn_ix`. `sample` shows `memset` /
   `bzero` at 2–4% of the step on every model. Bit-exact to remove.
6. **Its C is autovectorised; our Mojo is not** (§7.1). Every dense pass we
   have not hand-vectorised — FK, RNE, CRBA, the row builders, the Hessian's
   gathered rank-1 updates — runs one lane wide against clang's 2–4 lanes.
   This is the diffuse 2× on kinematics and dynamics (8 vs 4.6 on
   humanoid_cmu) and is not one fix.
7. **Its inverse does not exist.** dog keeps `compute_m_inv` (150 µs) for
   noslip; MuJoCo pays 73 µs of `mj_projectConstraint` for the same `AR`, so
   this is a 2× not a 10× — and `M⁻¹Jᵀ` by 63 tree solves would cost about
   what the 79 tree columns do. The saving is in not materialising the dense
   inverse (nv² writes, nv² reads by noslip).
8. **Mesh collision** (dog 72 vs 23, SO-ARM101 8.2 vs 3.1): the §12.1
   support-walk item, unchanged, a separate campaign.

**Ideas, ranked by what they buy per unit of risk:**

| | idea | expected | risk |
|---|---|---|---|
| 1 | `_finalize_env` on the tree LDL + chain-walk `M·qacc` (item 3) | cmu −25, dog −55, every damped Euler model | LOW — the routines exist; numerics change of the kind already gated |
| 2 | stop zero-filling per-solve scratch (item 5) | 2–4% everywhere | NONE — bit-exact |
| 3 | rank-1 Cholesky update/downdate on state change; factor once per solve; drop the per-iteration `H` rebuild (item 1) | cmu −35 to −45 (of 95), dog −40, humanoid −10 | MEDIUM — port `mju_cholUpdate` per segment; not bit-exact; the gate batch |
| 4 | chain-walk `M·v` for `Ma`, `Mv`, warm start, finalize (item 4) | ~3–5 on the big models | LOW |
| 5 | symbolic sparse Cholesky of `H` on one tree (item 2) | cmu's remaining factor 12 → ~4 per iteration; only after 3 | HIGH — a second factorisation kind |
| 6 | hand-vectorise the remaining dense passes (item 6) | diffuse, ≤ 2× on ~15% of the step | MEDIUM, piecemeal |
| 7 | `M⁻¹Jᵀ` for noslip without the dense inverse (item 7) | dog −50 to −80 | MEDIUM — noslip reads rows of `M⁻¹` in three places |

Items 1–4 together are worth roughly 70–80 µs of humanoid_cmu's 208 and
120–150 of dog's 663 — humanoid_cmu at ~1.8× MuJoCo and dog at ~2.6×, from
2.8× and 3.3× today — without a new factorisation kind. Item 5 is what the
last 1.5× on the big single-tree models costs, and it is the one MuJoCo
itself only turns on at 60 dofs.

### 13.15 LANDED (2026-09-05): §13.14's items 5, 3 and 1

Three commits, each gated before the next, each on the CPU path only.

* **No per-solve zero-fill** (`d8a8a66f`, bit-exact, ten checksums identical):
  `je_ix`, `cn_ix`, the noslip caches and the one-time `L` zeroing now use
  `Scratch(uninitialized=)`; the pyramidal factor never read an entry it had
  not written. park_k9 25.3 → 22.9, humanoid 99.8 → 94.2, ant 43 → 41.
* **The Euler finalize on the tree-ordered LDL** (`95f0fc8e`): the one
  caller of the forward LDL that §13.10 had not switched, plus `M·qacc` over
  M's tree sparsity. Interleaved: humanoid_cmu 196 → 185, dog 594 → 536. The
  gain is smaller than item 3's 29 µs estimate because the integrator's
  self-time was NOT mostly that refactor — see §13.16.
* **Factor once, update rank-1** (`bc50fd4f`): `chol_update_seg` is
  `mju_cholUpdate` over one tree segment; the pyramidal loop keeps its factor
  and updates it for the rows whose zone changed, rebuilding `H` only on rank
  loss. humanoid_cmu 185 → 169, dog 535 → 516; the two-iteration models are
  unchanged, since their second factorisation is exactly what this removes.

| model | nv | §13.13 | now | vs MuJoCo |
|---|---|---|---|---|
| humanoid_cmu | 62 | 195 | **171** | 2.79× → **2.45×** |
| dog_stand | 79 | 637 | **542** | 2.80× → **2.40×** |
| park_k9 | 60 | 25.0 | 24.2 | 2.82× → 2.73× |
| humanoid | 23 | 96 | 92 | 1.22× → 1.16× |
| ant | 14 | 42.6 | 40.8 | 1.31× → 1.24× |
| walker2d | 9 | 29.2 | 28.2 | 1.21× → 1.18× |
| hopper | 6 | 13.5 | 13.2 | 0.93× → 0.90× |
| reassemble3 / 5, sawyer (elliptic) | | | | 1.84× / 1.63× / 1.62×, unchanged |

(Three interleaved rounds; `results_final2`.) Two days in: dog 3003 → 542,
humanoid_cmu 768 → 171, park_k9 94 → 24; nothing past 20 dofs is more than
2.45× MuJoCo, and the six-to-nine-dof models are at or under 1.2×.

### 13.16 LANDED (2026-09-05): the step, stage by stage — and a matvec nobody read

`euler.mojo` now has `_EULER_PROBE`, the twin of `newton_solve._CPU_PROBE`:
off and free by default, on it prints one `[eprobe]` line per step with the
nanoseconds of each stage of `step["cpu"]`. The Newton probe gained `init`,
`pre1` and `pre2` marks over the preamble it had never covered.

**Where a step goes** (µs per step, one round each, 2000 timed steps):

| stage | humanoid_cmu (171) | dog_stand (498) |
|---|---|---|
| constraint solve (Newton) | **123.6** (72%) | **216** (43%) |
| dense M⁻¹ (noslip needs it) | 0 (skipped, §13.13) | **141** (28%) |
| collision | 6.1 | 61 |
| Euler finalize | 17.4 | 37 |
| LDL factor | 8.8 | 14 |
| everything else (FK, CRBA, RNE, cdof, …) | 15 | 29 |

So on humanoid_cmu three quarters of the step is the Newton, and on dog the
noslip's dense inverse is the single biggest item after it (§13.14 item 4).

**Inside the Newton, the preamble was half of it.** With the new marks,
humanoid_cmu's 120 µs solve split as: `pre1` (normal precompute) **40.3**,
`pre2` (friction precompute) 14.3, `hbuild` 17.7, `chol` 17.2, `setup` 14.4,
`ls` 5.6, `update` 4.6, `rows` 3.6, `init` 0.6. The 40 µs was
`_precompute_contact_normal` computing `M⁻¹·J_n` for every contact — a dense
nv×nv matvec per contact, 3844 scalar FMAs on nv=62 — to fill `ws_MinvJn` and
`K_n = J M⁻¹ Jᵀ`. Only the PGS family reads either. The Newton takes its
`R` from `diag_n` (`body_invweight0`, as `mj_diagApprox` does) and reads only
`J_n`, `pos_bias` and `c_dist` from that phase; the CG reads neither field
either. Worse, the Newton path had stopped computing `M⁻¹` at all in §13.13,
so on those models the matvec was multiplying a stale inverse.

Two changes, one commit, bit-exact (four checksums identical: humanoid_cmu,
dog, reassemble3, sawyer):

* `_precompute_contact_normal[MINV_J=False]` at the three Newton call sites
  (per-env and blocked kernel alike) skips the matvec and the `K_n` it feeds.
  The GPU per-env kernel shares the body, so it inherits the skip untested
  (no NVIDIA hardware here).
* The workspace init and PHASE 1 run over the `nc` live slots, not
  `max_contacts`: on humanoid_cmu that was 64 slots × 4 × 62 Jacobian zeroes
  plus 64 normal inits per solve, for eleven contacts. Nothing reads a slot
  at or past `nc`.

| model | before | after | Newton/solve | vs MuJoCo |
|---|---|---|---|---|
| humanoid_cmu | 169 | **129** | 120 → 85 (`pre1` 40 → 6) | 2.45× → **1.85×** |
| dog_stand | 494 | **454** | 212 → 176 | 2.40× → 2.01× |
| reassemble3 | 388 | 383 | | |
| sawyer_reach | 24.2 | 23.2 | | |

(Interleaved, three rounds, MIN; the MuJoCo column uses §13.15's reference
times.) What is left in humanoid_cmu's 85 µs solve: `hbuild` 17, `chol` 17,
`setup` 14, `pre2` 14 — the friction precompute is now the largest piece of
the preamble, and the next §13.14 item on this model. On dog the order is
noslip 60, `chol` 28, `hbuild` 23, `setup` 20.

⚠ **A probe that is off must be a no-op, and this one was not.** The first
build of the Euler probe had one timer block indented one level shallower
than the statement it followed; with the flag False, the `comptime if`
swallowed the `comptime assert` and the whole constraint-solve dispatch after
it. The step ran at 46 µs with 63 contacts (the humanoid fell through the
floor), the checksum changed, and a worktree bisect blamed the rank-1 commit —
the worktrees did not carry the uncommitted probe diff, and the one build
that had the flag ON was sane. What settled it: the wrong answer was the same
to the last digit across builds with every scratch zero-filled, so it was not
garbage, and the only difference left between sane and broken trees was the
diff itself. The lesson was already in this file's ancestry (a rule written
twice drifts): a flag-gated block is a statement like any other, and the gate
for "off is a no-op" is the checksum of the flag-off build against the tree
without the diff, not a read of the diff.

### 13.17 LANDED (2026-09-05): the noslip solves against the LDL — dog loses its dense inverse

§13.16's Euler split put dog's dense `M⁻¹` at 141 µs of 498 (28%), computed
only because the noslip reads it: `M⁻¹Jᵀ` per row for the sweep's `A`
entries and `M⁻¹·qfrc` at the end (§13.14 item 4). The step already holds a
tree-ordered `LᵀDL` factor of `M` (§13.10, `mj_factorI`), and `mj_solveLD`
applies `M⁻¹` to a vector in O(nv · depth) from it. MuJoCo never forms the
inverse for this.

**What landed.** `_newton_solve_env` takes the factor (`ldl_L`, `ldl_D`) and
the dof parent table; on the CPU path with `NTREE > 0` the two noslip
functions solve against them (`noslip._minv_apply` for one vector,
`_minv_apply_rows` for a block) and the integrators skip `compute_m_inv`
under the same predicate the LDL dispatcher uses. Every other leg — GPU
kernels, `NTREE == 0` — passes placeholders and keeps the dense product; the
blocked kernel and the per-env kernel both compile on Metal and match the
CPU oracle (`test_newton_freejoint_vs_cpu`).

⚠ **A tree solve per row was SLOWER than the inverse it replaced.** The first
build did one `mj_solveLD` per noslip row: dog went 454 → 557 µs. A single
solve is a serial chain walk — `x[j] -= L[i,j]·x[i]` up the parents, ~3 µs a
row on 79 dofs — and 32 rows of it cost more than the 141 µs it saved. The
reference does not do that either: `mj_solveLD(…, n)` solves `n` vectors in
one pass. `_minv_apply_rows` keeps the vectors column-major (`W[k·n + r]`),
so each chain step is one contiguous `n`-wide SIMD axpy over every row
(`_axpy_self`, the one-pointer twin of `cholesky._axpy_seg`): same flops, no
dependency chain. The noslip's own time rose 60 → 94 µs; the 141 µs is gone.

| model | before | after | vs MuJoCo |
|---|---|---|---|
| dog_stand | 454 | **352** | 2.01× → **1.56×** |
| reassemble3 (elliptic noslip) | 370 | 357 | |
| humanoid_cmu, sawyer_reach (no noslip) | | bit-exact, unchanged | |

(Interleaved, three rounds, MIN.) Not bit-exact — a different summation
order — so the gate is MuJoCo: `test_noslip_vs_mujoco`,
`test_noslip_elliptic_vs_mujoco`, `test_noslip_reaches_the_runtime_path`,
`test_constraints_vs_mujoco`, plus both-legs, dispatchers, RK4 and fields
tests. Checksums moved in the fifth digit (dog 903.639 → 903.671).

**Where dog's 352 µs goes now** (`_EULER_PROBE`): Newton 213 (60%, of
which noslip 94, `chol` 27, `hbuild` 23, `setup` 19, `pre2` 17), collision
59, finalize 38, LDL factor 15. **Reassemble3** (379 µs probed): collision
**204 (54%)**, Newton 167 (noslip 53, `hrebuild` 31, `hbuild` 16). On the
manipulation scenes the solver is no longer the first item — the box-box
narrow phase over the brick pile is.

### 13.18 LANDED (2026-09-05): the manipulation scene is collision-bound — two exact cuts

§13.17 left reassemble3 at 357 µs with collision at 204 (54%). The reference
for the same scene, same task pose, same `ctrl = 0.1`
(`physics3d_mujoco_phases.py`, now with a pose file and a ctrl argument):
212 µs, of which collision 85 (narrow 81, broad 3.7), constraint make +
project + solve 118, everything else 8. So on this scene the solver was
already within 1.4× and collision carried the gap at 2.4×.

`_COLL_PROBE` (broadphase_sap.mojo), the third stage probe, splits a
detector call into the time before the pair loop, the loop's own overhead,
and time × calls per narrow-phase routine, plus the pair counts at each
stage of the loop. Two things it showed, and one it did not:

* **The sweep held every geom, collidable or not.** 267 geoms, 120 with a
  nonzero contype or conaffinity. 10,600 sweep iterations and 2,100
  AABB-passing pairs per step for 79 real candidates; each of the 2,000
  false candidates paid the predefined-pair lookup and the body filter.
  MuJoCo walks only bodies that `canCollide` (engine_collision_driver.c:320)
  and `filterBitmask` (:535) rejects a geom whose two words are zero against
  every partner. Both detectors now drop such geoms before the pair loop
  (`3b97ce19`). Exact, checksums identical. Sweep 10,600 → 3,450 iterations,
  survivors 2,100 → 520. reassemble3 360 → 331, sawyer 23.7 → 21.0, dog
  369 → 354.
* **Every convex primitive pair ran GJK to convergence.** The generic
  `gjk_epa` path (cylinder–box, cylinder–cylinder, …) is 64 calls a step on
  reassemble3 at 2.15 µs — 138 of the 200 µs — and 34 calls on dog at
  1.1 µs, **all 34 separated** (0.02 hits). MuJoCo's `mjc_Convex` sets
  `dist_cutoff = 0` on its margin-inflated shapes so a separated pair exits
  as soon as a lower bound proves it apart (engine_collision_convex.c:104,
  engine_collision_gjk.c:225). Our `gjk_epa_witness` has that exit and the
  mesh path already used it; the primitive wrapper `gjk_epa` hard-coded it
  off. It now forwards `dist_cutoff`, and both detectors pass their margin,
  which is the same test the caller applies. Exact by construction (the exit
  fires only when no contact is possible) and by checksum on four models.
  dog 351 → **316**, reassemble3 333 → **316**, humanoid_cmu 131 → 128.
* **What the probe did NOT show, for 60 µs of it.** The first two builds
  reported 180 µs of "pair-loop overhead" that no hook covered, and the
  counters made it look like per-pair filtering. It was two unhooked
  branches — `gjk_epa` and the `box_box` fallback — plus the probe's own
  timers: at ~40 ns a read, four timer pairs on each of 520 survivors is
  60 µs a step. A residual is only as informative as the hook set is
  complete; hook every call site before reading the remainder.

| model | §13.17 | now | vs MuJoCo |
|---|---|---|---|
| reassemble3 | 357 | **316** | 1.68× → **1.49×** |
| dog_stand | 352 | **316** | 1.56× → **1.40×** |
| sawyer_reach | 23.2 | 21.0 | 1.53× → 1.39× |
| humanoid_cmu | 129 | 128 | 1.85× → 1.83× |

(Interleaved, two rounds, MIN; MuJoCo reassemble3 212 at the task pose.)

**What is left in reassemble3's collision (~130 µs):** 48 touching convex
pairs a step at ~2.1 µs each (GJK + EPA + the polytope), where MuJoCo's
whole narrow phase for 51 contacts is 81 µs — the per-touching-pair cost is
the item, and it needs a probe INSIDE `gjk_epa_witness` (GJK, EPA, polytope
init) before anything is changed; then the 520 surviving pairs' filtering
(~20 ns each for the predefined-pair lookup and the body filter, both once
per geom pair where MuJoCo does them once per body pair); then the 3,450
sweep iterations (MuJoCo sweeps 18 bodies, not 119 geoms).

### 13.19 Where the CPU path stands after the day (2026-09-05, evening)

Every row below is the tree at `7596c6f7`, interleaved, MIN of two or three
rounds; the MuJoCo column is §13.15's reference (§13.1 protocol) except
where noted.

| model | nv | ncon | §13.15 (this morning) | now | vs MuJoCo |
|---|---|---|---|---|---|
| humanoid_cmu | 62 | 11.6 | 171 | **129** | 2.45× → **1.85×** |
| dog_stand | 79 | 7.8 | 542 | **314** | 2.40× → **1.39×** |
| reassemble3 (task pose) | 21 | 55 | 390 | **315** | 1.84× → **1.49×** (1.26× against MuJoCo with native CCD, 249 µs) |
| reassemble5 (task pose) | 33 | 111 | 1057 | **818** | 1.63× → **1.26×** |
| sawyer_reach | 15 | 5 | 24.4 | **21.0** | 1.62× → **1.39×** |
| humanoid | 23 | 7 | 92 | 81 | 1.16× → 1.03× |
| ant | 14 | 3 | 40.8 | 37.7 | 1.24× → 1.14× |
| walker2d | 9 | 6 | 28.2 | 27.2 | 1.18× → 1.14× |
| hopper | 6 | 2 | 13.2 | 12.8 | 0.90× → 0.87× |

Nothing is more than 1.85× MuJoCo now, the manipulation scenes are at
1.26–1.49×, and the gym models sit between 0.87× and 1.14×. Three days ago
dog was 3003 µs, humanoid_cmu 768 and reassemble5 2222.

**What landed today, in order** (§13.15–§13.18 and this section): the
Euler and Newton stage probes and the collision and GJK probes; no per-solve
zero-fills; the finalize on the tree LDL; factor-once + rank-1 on the
pyramidal Newton; the per-contact `M⁻¹J_n` nobody read; live-slot init; the
noslip on the tree LDL (one-pass multi-vector solve); non-collidable geoms
out of the sweep; the GJK cutoff on the primitive path; lower-triangle
Hessian builds. Eight of the ten are bit-exact; the two that are not (tree
LDL solves in the noslip) are gated against MuJoCo.

**What the probes say is left, by model.** humanoid_cmu (85 µs Newton of
129): `hbuild` 18, `chol` 17, `setup` 15, `pre2` 14 — no single item above
15%; the chain-walk `M·v` (§13.14 item 2, now unblocked since the parent
table reaches the Newton) is worth ~8 µs of `setup`. dog (314): noslip 93
inside a 212 µs Newton, collision 57, finalize 38. reassemble3 (315):
collision ~150 of which 48 touching convex pairs at ~2.5 µs (GJK+EPA, at
parity with MuJoCo's native path per pair) and ~35 µs of sweep and
per-geom-pair filtering that a body-level sweep would remove; the elliptic
Newton ~165, of which noslip 56 and the per-iteration Hessian rebuild 35
(MuJoCo pays a comparable per-iteration cone update at this nv).

⚠ **Two measurement lessons from today, both about probes.** A flag-gated
block that is off must be gated by CHECKSUM against the tree without the
diff (§13.16: one indent level made the solve disappear and a bisect blamed
an innocent commit). And a probe's residual is only as good as its hook set
and its own overhead: §13.18's "180 µs of pair-loop overhead" was two
unhooked routines plus 60 µs of timer reads, and §13.19's lower-triangle
change gained a third of what the stage numbers promised because those
numbers carried the probe's cost. Hook every call site, subtract the
timers, and quote stage numbers as a ceiling.

### 13.20 LANDED (2026-09-05, night): the pyramidal noslip reads a precomputed `A` — dog −25%

`d31f8dc0`. §13.19 left dog's noslip at 93 µs inside a 212 µs Newton, the
largest single item on the board. A stage probe inside `noslip_pyramidal`
(a `_NOSLIP_PROBE` flag kept out of the tree) split it, per solve, on dog's
first 3000 steps (ns, probe-inflated — §13.19's caveat applies):

| stage | before | swept rows | + `Z` Gram | + SIMD dot |
|---|---|---|---|---|
| `M⁻¹ Jᵀ` cache | 76 | 25 | 0 | 0 |
| `A` / `b` build | (30, first attempt) | 10 | 44 | **21** |
| sweep (4 iterations) | 14 | 3 | 3 | 3 |
| `dualFinish` | 8 | 4 | 5 | 5 |
| rows / swept rows / contacts | 119 / — / 8.7 | 119 / 27 / 8.7 | | |

**What MuJoCo does.** `solNoSlip` never touches `qacc`. `mj_projectConstraint`
builds `efc_AR = J M⁻¹ Jᵀ + R` once, and the sweep READS it: a residual is
one row of `AR` against the current forces (`residual`, then `- R f`), a
block's `Ac` is four entries (`extractBlock`), the diagonal is `ARdiaginv`.
Ours recomputed `M⁻¹ Jᵀ` for every row, added `d · M⁻¹ Jᵀ` into `qacc` after
every pair and refreshed every row's `jar` — `E · nnz(J)` per pair, four
sweeps of 18 pairs on dog. And the cache it did build was for ALL 119 rows.

**Three cuts, each measured in the table.**

1. *Only the swept rows.* The pass moves dry-friction dof rows and the
   friction edges of condim ≥ 3 contacts — 27 of dog's 119 rows. The limit
   and normal-direction rows carry forces the sweep never changes, so their
   whole contribution to every residual is a constant: `b_S = J_S (qacc_smooth
   + M⁻¹ Jᵀ_¬S f_¬S) + bias_S`, ONE extra `M⁻¹` apply. Cache 76 → 25.
2. *`A_S` as a Gram matrix.* MuJoCo builds `AR` with `mj_solveM2` (the half
   solve `D^-½ L⁻ᵀ`) and `mju_sqrMatTDSparse`. With `M = Lᵀ D L`,
   `A_S = Z_Sᵀ Z_S` for `Z_S = D^-½ L⁻ᵀ J_Sᵀ`; the `L⁻ᵀ` half pushes each dof
   up its ancestors, which never leaves a contact row's ancestor-closed
   support, so it is O(chain²) per row rather than a dense fill. No
   `M⁻¹ Jᵀ` cache exists any more on this leg. Cache 25 → 0 — and the build
   went UP, 10 → 44, because
3. *the Gram's dots were scalar.* 378 dots of 79 floats, each a serial chain
   of 79 dependent adds at ~4 cycles: 38 µs by arithmetic, 44 measured. A
   `W`-lane dot with a single `reduce_add` at the end (`_dot_self`, the
   twin of `_axpy_self`) took the stage to 21. ⚠ **Every scalar reduction
   in this engine is latency-bound, not throughput-bound**; the same
   arithmetic said the previous `J · cache` build (10 µs for 7k adds) was
   the same shape. Anywhere a dot runs over more than ~16 elements, the
   accumulator count is the cost.

**Whole step, interleaved against the tree without this change:**

| run | before | after |
|---|---|---|
| dog, 500 + 3000 steps | 310 | **232** (−25%) |
| dog, 2000 + 6000 | 351 | **265** |
| dog, 2000 + 20000 | 366 | 323 (new run: 9.5 contacts vs 9.0) |

⚠ **The closed-loop bench cannot A/B a rounding-level change over a long
run.** The two binaries agree on cost for ~6000 steps and then part: the
sample profiler showed `solve_newton` at 49% in the new binary against 36%
in the old with the noslip share halved, and a 2000-solve windowing of the
Newton stage probe showed why — from window 3 on, the new run carries 10%
more rows and every Newton stage scales with it, while windows 0–2 are
identical stage for stage. The dog falls either way; where it lands is
chaotic. Quote the matched-regime windows (or a short run), and window the
probe before believing a whole-run number in either direction.

⚠ **`A f + b` is not the Newton's `jar`.** The probe also compared the folded
residual against the `jar` the primal solve left: 0.2% of `|jar|` apart on
average, up to 0.7%. That is the primal solver's convergence slack — the
identity `qacc = qacc_smooth + M⁻¹ Jᵀ f` holds only at the exact optimum —
and it is exactly what MuJoCo's pass sees, since it too works from `f`, not
from the Newton's `qacc`. The tree LDL, the two `M⁻¹` applies and this
change are gated against MuJoCo (`test_noslip_vs_mujoco`,
`test_noslip_reaches_the_runtime_path`, `test_noslip_elliptic_vs_mujoco`,
`test_newton_both_legs`, `test_noslip_blocked_kernel` on Metal), not
against the old checksum. The GPU legs keep the refreshing scheme.

**What is left in the pass** (ns, probe-inflated): `A_S` build 21, of which
the Gram ~4 by arithmetic — the rest is the `Z` half-solve's per-row zero
scan and `LayoutTensor` indexing, and the `b_S` fold's one full solve;
`dualFinish` 5; sweep 3. The elliptic pass (`noslip_elliptic`, reassemble3's
56 µs) still refreshes `qacc` per contact and caches `M⁻¹ Jᵀ` for every
tangential row through `_minv_apply_rows`; it is the same shape and the
next item.

### 13.21 LANDED (2026-09-05, night): `Scratch[i]` cost four pointer accesses — 13–23% on every model

Two changes, one commit each. The second came out of probing the first.

**The elliptic pass** (`noslip_elliptic`, reassemble3's 56 µs). Its shape has
to stay: at nv = 21 with 55 contacts a residual against a precomputed `A`
would be a 110-long gather per row, worse than the 12-nonzero `J · qacc` it
does now, and the pass keeps `qacc` incrementally on purpose (the float32
rebuild note at the end of the routine). The stage probe (per solve, ns,
probe-inflated): cache 17.5, dry-friction columns 1.3, sweep 31.6, `qfrc`
1.5. Three cuts in the sweep and the cache:

- each contact's `Ac` block (`J_t M⁻¹ J_uᵀ`, floored diagonal) is
  loop-invariant and was rebuilt from `nt²` sparse dots on every contact of
  every iteration — now built once, before the loop;
- each contact copied its `nt` rows of `M⁻¹ Jᵀ` out of the cache before
  using them — now read in place;
- the cache build staged the rows row-major, memset the whole
  `T_CAP × V_CAP` slab and transposed twice around the tree solve — the
  rows' nonzeros now scatter straight into the column-major work slab
  (`_tree_solve_cols`, the core of `_minv_apply_rows` exposed), and only the
  live part is zeroed.

Sweep 31.6 → 18.9. The cache build did not move (17.5 → 18.2 with the new
`Ac` precompute inside it), and the arithmetic said ~5. A four-way sub-probe:

| sub-stage | `Scratch[i]` loops | same loops on `unsafe_ptr()` |
|---|---|---|
| zero the live slab + scatter the rows | 4.3 | **1.1** |
| tree solve (`_tree_solve_cols`) | 5.0 | 5.0 |
| transpose out | 5.6 | **1.3** |
| `Ac` precompute | 2.0 | 2.0 |

Zeroing 2.3k floats in 4.3 µs is ~2 ns an element. The pointer rewrite of
the same loops is 4× faster.

**The accessor.** `Scratch.__getitem__` / `__setitem__` forwarded to
`InlineArray.__getitem__` and `List.__getitem__`, which normalise a negative
index (a compare and a select per access, and a branch the optimiser has to
carry through every loop) and hold a bounds `debug_assert`. Nothing in the
engine indexes a `Scratch` from the end. Both now call `unsafe_get`. One
twelve-line change in `fields/scratch.mojo`, every model, interleaved twins
built from the same tree with only that file different:

| model | `Scratch[i]` | `unsafe_get` | |
|---|---|---|---|
| sawyer_reach | 21.2 | **16.4** | −23% |
| humanoid_cmu | 125.5 | **101.0** | −19% |
| dog_stand (3000 steps) | 233 | **188** | −19% |
| reassemble3 | 297 | **252** | −15% |
| reassemble5 | 628 | **507** | −19% |
| hopper / walker2d / ant | 12.8 / 27.0 / 38.1 | **10.9 / 22.5 / 29.6** | −15 / −17 / −22% |
| humanoid / half_cheetah | 81.2 / 5.13 | **70.5 / 4.32** | −13 / −16% |

⚠ **Semantically identical, NOT checksum-stable.** The pyramidal models and
the gym models shifted at rounding level (sawyer 6e-7 relative; dog, being
chaotic, parted after a few thousand steps); reassemble3/5 were
bit-identical. Two things had to be ruled out before believing that was
codegen: (1) a site indexing from the end — an instrumented accessor that
aborts on ANY out-of-range index ran 600 steps of sawyer, humanoid_cmu, dog
and reassemble3 without firing; (2) an uninitialized read whose garbage
moved with the new frame layout — a fills-everywhere twin of the new
accessor matched it bit for bit on all three pyramidal models. What is left
is the compiler contracting multiply-adds differently once the normalisation
branch is gone. The gate for a change of this kind is MuJoCo, not the old
checksum: the curated physics3d manifest (32 files — Newton legs and fields,
warmstart, constraints, friction dofs, impratio, weld and equality rows,
tendons, rolling friction, both noslip passes, condim 4/6, sawyer settle,
walker2d and jaco contacts, mesh manifold, hfield, CCD margin, RNE
sensors, tree blocks, humanoid limits) plus the two Metal kernel gates
(`test_newton_freejoint_vs_cpu`, `test_noslip_blocked_kernel`) all pass.

⚠ **Where else this lesson bites.** `sample` cannot see it — the accessor is
inlined into every caller — and a stage probe only shows it as "a stage that
costs more than its arithmetic". The check is cheap: count the elements a
stage touches, multiply by ~0.3 ns, and if the probe says four times that,
rewrite ONE loop on a pointer before touching the algorithm. The same
family of cost is in §13.20's scalar reductions (one add chain, ~4 cycles a
term): both are the accessor and the accumulator, not the flops.

**The table after this round** (MIN of interleaved rounds, 500 + 3000 steps
for the contact rows, 1000 + 10000 for the gym rows; MuJoCo per §13.15):

| model | §13.19 (evening) | now | vs MuJoCo |
|---|---|---|---|
| humanoid_cmu | 129 | **101** | 1.85× → **1.45×** |
| dog_stand | 314 | **268** (20k steps; 188 over the first 3k) | 1.39× → **1.19×** |
| reassemble3 | 315 | **252** | 1.49× → **1.19×** (1.01× against native-CCD MuJoCo) |
| reassemble5 | 818 | **507** | 1.26× → **0.78×** |
| sawyer_reach | 21.0 | **16.4** | 1.39× → **1.08×** |
| humanoid / ant / walker2d / hopper | 81 / 37.7 / 27.2 / 12.8 | **70.5 / 29.6 / 22.5 / 10.9** | 0.90× / 0.90× / 0.94× / 0.74× |
| half_cheetah | 5.1 | **4.3** | — |

Five of nine rows are now at or below MuJoCo. The three above it are
humanoid_cmu (1.45×, a 62-dof Newton with nothing above 15% of the solve),
dog (1.19×, the pyramidal noslip now ~30 µs of a 268 µs step) and
reassemble3 (1.19× against libccd MuJoCo, parity against native CCD). Both
commits of this section landed after the 32-file gate manifest and the two
Metal kernel gates passed on the new accessor.

### 13.22 Why the small models beat MuJoCo — not the glue, not float32 (2026-09-05)

Asked after §13.21's table put five rows at or below MuJoCo. Two easy
explanations, both checked and both wrong; the real one is in MuJoCo's own
timers.

**Not the Python glue.** The twin (`physics3d_cpu_vs_mujoco.py`) times ONE
`mj_step(m, d, nsteps)` call; the loop runs inside the C library and no
interpreter is in the timed region. (A Python `for` around single steps would
add ~0.3 µs a step — the script's header says so and avoids it.)

**Not float32.** A float64 twin of `bench_gym` (`DT = DType.float64`, nothing
else changed), interleaved with the float32 one, 1000 + 10000 steps:

| model | ours f32 | ours f64 | MuJoCo (f64) |
|---|---|---|---|
| hopper | 11.0 | **9.7** | 15.9 |
| ant | 30.0 | 29.1 | 35.9 |
| humanoid | 71.8 | 73.1 | 81 |

Same speed within noise, hopper slightly faster in float64. So the CPU path
is nowhere bandwidth- or SIMD-width-bound in a way float32 helps, which fits
§13.21 (the costs were the accessor and the accumulator chain, not the
flops). Float32 is the GPU's choice; on the CPU it buys nothing.

**It is MuJoCo's fixed per-forward setup, paid four times a step.** Its phase
timers on hopper (`physics3d_mujoco_phases.py`, `bvactive=0`; 2 contacts, 11
rows, 6 dofs, RK4 = 4 forwards per step, 15.9 µs):

| phase | µs/step | note |
|---|---|---|
| `CONSTRAINT` | 4.67 | Newton at **0.23 iterations** on average — nearly all of it is setup: Hessian factor, row state, dual finish, warmstart |
| `POS_MAKE` | 3.47 | 11 rows through the generic path: arena, per-row impedance/aref, sparse assembly, the `efc_*` arrays |
| `COL_BROAD` | 1.14 | 5 geoms through the BVH machinery |
| `POS_KINEMATICS` + `POS_INERTIA` | 1.93 | the arithmetic |
| everything else | 4.7 | narrow phase, velocity, actuation, the rest |

9.3 of 15.9 µs is setup on a system whose arithmetic is ~2 µs. Ant is the
same shape (`CONSTRAINT` 14.5 at 0.09 iterations, `POS_MAKE` 5.3 of 35.9).
Our engine has compile-time dimensions, fixed-capacity row storage, no
arena, no sparse/dense dual dispatch and no per-row generic instantiation,
so it has no comparable floor. The advantage is a constant, and it vanishes
as `nv` grows and the arithmetic dominates — which is the table's pattern:
we win at 6–23 dofs and MuJoCo still wins at 62 and 79.

### 13.23 LANDED (2026-09-06): the second round — ancestor arrays, chain walks, a compact factor

Three commits (`f88de9db`, `1e3d53b8`, `aefcac0d`, `23918a84`),
all BIT-EXACT by checksum on every model, picked from the stage probes after
§13.21 had moved every stage. The method was §13.21's: count what a stage
touches, and when the probe says several times the arithmetic, look at how
the loop addresses memory before looking at the algorithm.

**1. The tree LDL factor iterated `j = par[j]` chains.** `_ldl_factor_tree_env`
(`mj_factorI` on dense storage) runs twice a step on the Euler path — the
smooth factor, and again inside the finalize on `M + dt·damping` — and its
triple loop visited ~8k entries on dog through parent-chain walks: one
dependent load per hop, then a `LayoutTensor` read-modify-write per entry,
after an element-wise zero of the whole `nv×nv` slab. Ancestor lists built
once per call and iterated as arrays, the slab through a raw pointer, the zero
as `W`-lane stores; the solve and the finalize's sparse `M·qacc` took the
pointer form too. Factor 9.0 → 6.3 (humanoid_cmu) and 15.5 → 10.2 (dog);
finalize 15.1 → 11.6 and 32.7 → 26.0 (ns per step, probe-inflated).

**2. The Jacobian-row builders walked the chain once per JOINT.**
`_contact_jacobian_row` and both `_angular_jacobian_row`s decided "does joint
j move body a" by walking a's parent chain for every joint — `njoint × depth`
dependent loads per row, ~650 on dog, three rows a contact, again for body
b — which is why the two contact precomputes (`pre1` + `pre2`) stood at ~20 µs
against MuJoCo's 6.6 for all of `POS_MAKE`. `mj_jac` walks the chain once.
Now `_body_chain` records the chain in a 32-slot array and `_affects` tests
each joint's body against it; a deeper chain falls back to the walk. The
helper lives once and all three builders use it — the block had been written
inline three times (§13.14's recurring defect shape). Dog 185 → 175 µs.

**3. The factor in a compact row layout.** `mj_factorI`'s `qLD` shape: row
`k` holds `L[k, anc]` root-first, then the diagonal. An ancestor's list is a
PREFIX of its descendant's, so the inner update for `i ∈ anc(k)` is one
contiguous `W`-lane axpy of `dep[i] + 1` entries where the dense form still
gathered `nv`-strided entries. The dense `L` the solves read is rebuilt from
the compact rows (zero, scatter). Factor 6.3 → 4.9 / 10.2 → 7.5; finalize
11.6 → 10.2 / 26.0 → 23.9. The lists are built by copying the parent's
(parents have the smaller index) — the first version walked twice per dof,
and at nv = 6 with four factors a step that fixed cost showed.

**The round, interleaved against §13.21's binaries** (500 + 3000 steps for
the contact rows, 1000 + 10000 for the gym rows, two rounds, MIN):

| model | §13.21 | now | | vs MuJoCo |
|---|---|---|---|---|
| humanoid_cmu | 103 | **93** | −10% | 1.45× → **1.33×** |
| dog_stand (3k / 20k steps) | 192 / 281 | **168 / 248** | −11% | 1.19× → **1.10×** |
| humanoid | 73 | **67.5** | −8% | 0.90× → 0.83× |
| sawyer_reach | 16.4 | 16.6 | +1% | 1.08× |
| reassemble3 / reassemble5 | 257 / 513 | 257 / 514 | 0 | 1.19× / 0.78× |
| hopper / walker2d / ant / half_cheetah | 11.0 / 22.7 / 29.8 / 4.36 | 11.3 / 23.4 / 30.8 / 4.5 | **+2–4%** | 0.71× / 0.94× / 0.86× / — |

⚠ **The small RK4 rows lost 2–4% to the compact factor.** They factor four
times a step at nv = 6–14, where the compact form's fixed costs (the compact
fill, the scatter back, the `W`-lane guards on 6-entry rows) exceed what the
axpy saves. Kept as is: one code path, bit-exact, and those rows already sit
15–30% under MuJoCo; the alternative is a size-gated second path, which is
the drift shape this file keeps recording. If the gym rows ever matter more
than dog and humanoid_cmu, that is the knob.

**What the probes say is left** (dog / humanoid_cmu, ns per step,
probe-inflated, at the end of the round): Newton ~100 / ~62 — of which the
Cholesky with rank-1 updates 26 / 17 (the updates walk COLUMNS of a row-major
`L`, as `mju_cholUpdate` does; inherent unless the updates are batched),
`setup` 15 / 10 (the `M_local` copy could be one memcpy; the `je_ix` nonzero
scan is `E × nv`), the pyramidal noslip 20 / 0, Hessian build 11 / 11; the
two factors 7.5 + ~7.5 / 4.9 + ~4.9 against MuJoCo's ~3 + 3; collision 22 /
4. Reassemble3's remaining structure is unchanged from §13.19: the pair
filters cost ~35 ns a candidate pair (525 of them), the per-iteration elliptic
Hessian rebuild 23, the elliptic noslip 28.

### 13.24 LANDED (2026-09-06): the Newton's setup scanned every row twice; the cone Hessian tested every term

Two commits (`a3d32419`, `338a3cbb`), bit-exact by checksum, from a
sub-probe of the pyramidal Newton's `setup` stage and a re-read of the
elliptic cone Hessian.

**`setup` was two scans of every row over every dof.** The sub-probe split
dog's 15 µs `setup` (humanoid_cmu's 10) as: `je_ix` nonzero scan + segment
builder **9.6** (5.9), `M_local` copy + `M·qacc` 1.7 (1.0), edge forces and
warm-start cost 1.0 (1.0), the rest under 3. `build_dof_segments_p` was
rescanning `Je[e*nv + i] != 0` for every row to find its tree range — the
same scan the caller had just done to build the sparse lists. Three cuts:
the nonzero scan reads `W` entries at a time and skips an all-zero chunk (a
contact row is nonzero on its two chains only, a dozen of dog's 79 dofs);
`build_dof_segments_p` gains `SPARSE` — a row's tree range is its first and
last list entry, since `seg_start` is monotone in the dof index (the blocked
kernel keeps the scanning form through the default); `M_local = M` and
`H = M_local` are one contiguous copy each (`M` is zero between trees), and
the `JᵀDJ` outer product runs `b <= a` over the ascending list instead of
testing `j <= i` per term. Dog 167 → 155 µs, humanoid_cmu 92 → 84.

**The elliptic cone Hessian did the same test.** `ell_add_contact_hessian`
walked the full `nnz × nnz` block per contact row and kept the lower
triangle by `if j <= i`; the contact dof list is ascending, so `b <= a` is
the triangle by construction. Reassemble3 256 → 248, reassemble5 511 → 486.

**Tried and reverted: hoisting the pyramid edges' per-dof `LayoutTensor`
reads** (the normal row and `qvel`, re-read four times per contact) into
locals. Bit-exact, and no measurable change on dog, humanoid_cmu or sawyer —
so the remaining cost of the two contact precomputes (`pre1` + `pre2`, ~15 µs
on dog) is not those reads. It is the joint loop: each Jacobian row visits
all 50 of dog's joints, reading three joint fields for each, when only the
~15 joints on the two contact bodies' chains contribute. Walking the chain's
joints directly needs a body-to-joint map (`body_jntadr` / `body_jntnum`)
the body table does not carry; it can be derived once per solve from the
joint table (joints are stored in body order) and passed to the builders
through a defaulted parameter, as `build_dof_segments_p` now does. Worth
~5 µs on dog and humanoid_cmu; the next item on this path.

| model | §13.23 | now | vs MuJoCo |
|---|---|---|---|
| humanoid_cmu | 93 | **83** | 1.33× → **1.19×** |
| dog_stand (3k steps) | 168 | **155** | |
| reassemble3 / reassemble5 | 257 / 514 | **248 / 486** | 1.19× → **1.15×** / 0.78× → **0.74×** |
| sawyer_reach | 16.6 | 16.3 | 1.08× |

**Left, by the probes** (dog / humanoid_cmu, probe-inflated): Cholesky with
rank-1 updates 26 / 17 (column walks, as `mju_cholUpdate`); the pyramidal
noslip 20 / 0 (its `Z` half-solve still reads `L` through `LayoutTensor`
one hop at a time); Hessian build 11 / 11 (gathered `nnz²/2` updates per
row); the two contact precomputes ~15 / ~13 (the joint loop above); the
two LDL factors 7.5 + 7.5 / 4.9 + 4.9; collision on dog 22. On reassemble3:
collision ~150 (64 GJK calls at per-pair parity), the elliptic noslip 28,
the per-iteration rebuild region 23, the pair filters ~18.

### 13.25 LANDED (2026-09-06): the contact rows visit the chain's joints — and where the day ends

`8ed2153f`. §13.24 named the joint loop as the next item on the row path:
`_contact_jacobian_row` visited all `njoint` joints per row, three fields
each, when only the joints on the two contact bodies' ancestor chains — ~15
of dog's 50 — contribute. `mj_jac` walks the chain. The body table has no
`body_jntadr`, so the CPU Newton derives the map once per solve from the
joint table (joints are stored in body order, so a body's joints are
contiguous; if they ever were not the map is rejected and the scanning form
kept) and passes it through the two contact precomputes via defaulted
parameters, so the GPU kernels and every other caller are untouched. The
per-joint arithmetic is one helper used by both walks.

⚠ **Rounding-level, and not for the reason one would guess.** The checksums
moved. A probe build ran BOTH walks on every row and printed the first
differing dof: last-bit differences on rows where the two walks call the
same helper with the same arguments. A build with the map forced off matched
the old binary bit for bit. So the walk is logically identical, and each
inlined call site of the helper gets its own copy in which the compiler
contracts the multiply-adds differently. Third time today that a change
that is the same arithmetic on paper is not the same bits (§13.21, §13.23
item 3): the gate for such a change is MuJoCo, and the probe that compares
the two paths element-wise is what tells "same algorithm, different
contraction" from "different algorithm". Ten CPU gates and the Metal
free-joint gate pass.

**The day, against the §13.21 binaries** (interleaved, MIN of two or three
rounds; contact rows 500 + 3000 steps, dog also 2000 + 20000, gym rows
1000 + 10000; MuJoCo per §13.15):

| model | §13.21 | now | | vs MuJoCo |
|---|---|---|---|---|
| humanoid_cmu | 103 | **75** | −27% | 1.45× → **1.07×** |
| dog_stand (20k steps) | 281 | **219** | −22% | 1.19× → **0.97×** |
| dog_stand (3k steps) | 192 | **150** | −22% | |
| reassemble3 | 257 | **241** | −6% | 1.19× → **1.14×** |
| reassemble5 | 513 | **486** | −5% | 0.78× → **0.74×** |
| sawyer_reach | 16.4 | 16.1 | −2% | 1.07× |
| humanoid | 70.2 | **54.2** | −23% | 0.87× → **0.67×** |
| walker2d | 22.8 | **20.7** | −9% | 0.73× |
| ant | 29.8 | **28.0** | −6% | 0.78× |
| hopper / half_cheetah | 10.95 / 4.37 | 10.6 / 4.19 | −3 / −4% | 0.67× / — |

Eight of ten rows are now at or below MuJoCo; humanoid_cmu is within 7%
and reassemble3 within 14% (parity against MuJoCo's native-CCD build). The
§13.23 small-model regression is gone: the joint map gave the RK4 rows back
more than the compact factor took.

⚠ **One measurement trap from today's table.** A single interleaved run
had ant at +12% and hopper with a 17.6 µs outlier — a load spike, which the
same binaries could not reproduce three rounds later (ant 29.8 → 28.0).
With a laptop as the bench machine, no row moves on one run: three rounds,
MIN, and a re-run before believing a regression.

**Left, by the probes** (dog / humanoid_cmu, ns per step, probe-inflated):
the Cholesky with rank-1 updates 26 / 17 (column walks of a row-major `L`,
as `mju_cholUpdate`); the pyramidal noslip 20 / 0 (its `Z` half-solve reads
`L` through `LayoutTensor` one hop at a time — the ancestor-table treatment
of §13.23 applies); the Hessian build 11 / 11; the two LDL factors 7.5 + 7.5
/ 4.9 + 4.9 against MuJoCo's ~3 + 3; collision on dog 22. On reassemble3:
collision ~150 with the 64 GJK calls at per-pair parity, the elliptic noslip
28, the per-iteration rebuild region 23, the pair filters ~18.

### 13.26 LANDED (2026-09-06, evening): the fourth round — the Cholesky's chain, the tree walks, and two filters

Seven commits (`e1ea3e8e` … `09cde68c`), every one bit-exact: checksums
identical on all ten bench models at every step, gated by the Cholesky,
noslip, Newton, Euler, sensor and collision tests. The method was the same
as §13.20–§13.25 — sub-probe the largest window, read what the code does
per element, change the loop shape and not the arithmetic.

1. **The Newton's Cholesky ran row-outer** (`e1ea3e8e`). A sub-probe of
   the Cholesky block found the FACTOR at 24 of dog's 28 µs and 14 of
   humanoid_cmu's 17 — the rank-1 updates §13.25 had named were the other
   4, one or two a step. `chol_factor_seg`'s vectorised leg walked `for i:
   for j <= i`, and along a row that is a serial chain: the dot for
   `(i, j+1)` ends on `L[i, j]`, a subtract and a divide away, so each entry
   waited ~18 cycles on the one before — latency, not the 82k flops.
   `mju_cholFactor` is column-outer: the `s1 - j` dots of a column read
   complete rows and are independent, and the core overlaps them. Factor
   24.1 → 11.6 µs on dog, 14.3 → 7.6 on humanoid_cmu, the same operations
   per entry in the same order. The update's two column walks became one.
   MuJoCo's reciprocal-multiply was tried and dropped: it moved the
   checksums and gained nothing — the division was never on the critical
   path once the chain was gone.
2. **Ancestor lists for every tree walk; the noslip's Z half-solve
   column-major** (`5f63e3fb`, `73ade3e5`). `_dof_ancestors` is now shared
   by the LDL factor, the LDL solve, the noslip's two full solves and the
   Euler finalize's `M·qacc`; the `par` chase (a dependent load per hop)
   becomes independent loads, `L` and `M` are read through pointers, and
   each walk keeps its per-entry order (parent-first gathers). Modest: the
   two noslip solves 3.4 → 2.9 µs each on dog, the finalize's product
   4 → 2. The half-solve was the surprise: 7.2 µs a call for 27 rows, and
   the ancestor table did NOT move it — the cost is the read-modify-write
   chain through one row's entries, not the chase. Column-major over all
   swept rows (one 27-wide axpy per hop, `_tree_solve_cols`'s shape) gives
   6.2 including its scatter. A GATHER form (each block collecting from
   its descendants, deepest first) was tried and is SLOWER, 7.5: its
   accumulator is one serial FMA chain per block, where the push form's
   chains run through memory across independent blocks. Kept the push.
3. **The plane loop mixed parameters before rejecting** (`91626a49`). The
   SAP pair loop had hoisted its bounding-sphere test above
   `mix_contact_params` with a note saying why; the plane loop — the third
   copy of the same pair logic — still mixed ~30 tensor reads and the
   priority/solref/solimp rules for every geom against every plane, 391 a
   step on dog, of which the test keeps 58. Same values for every survivor.
4. **The RNE passes scanned every joint per body** (`d9569b49`).
   `_rne_fwd_body` — the forward pass of the bias RNE and of the
   post-constraint RNE dog's sensors need — and the post pass's
   `cdof·qacc` sweep each scanned all `njoint` rows per body for
   `JOINT_IDX_BODY_ID == b`: three O(nbody·njoint) scans a step, 62 × 50,
   a float→int conversion per read. The body→joint map §13.25 built inline
   in the Newton is now `dynamics/body_joint_map.mojo`, shared by the
   Newton and both RNE dispatchers, passed down under a `JMAP` flag only
   the CPU legs set. Dog 135 → 127 µs.
5. **The body-pair filter re-read the exclude table per pair**
   (`03baf1a3`). Dog has 30 `<exclude>`s and ~400 candidate pairs a step:
   24k `LayoutTensor` reads and conversions. MuJoCo keeps
   `exclude_signature` sorted and searches it; each detection call now
   sorts the integer signatures once and the three loops binary-search.
   Dog 127 → 122.
6. **A cap of 0 is the heap leg, and a static zero is also 0**
   (`09cde68c`). The exclude table's `Scratch` used `cap[D.NEXCLUDE]()`,
   which is 0 for a model with no excludes — so hopper, half_cheetah and
   walker2d paid a malloc per detection call, four a step under RK4, and
   the interleaved table showed them +2–4%. `may_exist` tells a static
   zero from a dynamic dim; a one-slot inline array put the three rows
   back on their baselines. ⚠ Any `Scratch` sized by `cap[]` of a
   dimension that can be zero needs this guard.

**Numbers, interleaved against this morning's binaries (`917c3cfd`), MIN of
three rounds**, on a busier machine than §13.25's table — the absolute
values are higher than this morning's, the deltas are what the interleave
measures:

| model | morning | now | | vs MuJoCo (§13.15) |
|---|---|---|---|---|
| dog_stand (3k steps) | 145.8 | **121.8** | −16.5% | |
| dog_stand (20k steps) | 216.1 | **184.8** | −14.5% | 0.97× → **0.82×** |
| humanoid_cmu | 73.5 | **63.3** | −14.0% | 1.07× → **0.91×** |
| humanoid | 53.5 | 52.0 | −2.8% | 0.66× |
| reassemble3 / reassemble5 | 238 / 551 | 241 / 549 | flat | 1.14× / 0.74× |
| sawyer_reach | 16.0 | 15.9 | flat | 1.06× |
| walker2d / ant / hopper / half_cheetah | 20.5 / 27.8 / 10.5 / 4.15 | 20.6 / 27.7 / 10.6 / 4.17 | flat | |

Both large single-tree models are now under MuJoCo. Every row but
reassemble3 (native-CCD parity at 1.14×) and sawyer (1.06×) is at or below.

**Sub-probes this round, for the next one** (dog, µs a step, probe-inflated):
inside the Newton — noslip 20 (Z 5.6, Gram 3.3, two full solves 2.9 each,
four sweeps 2.8), Cholesky 14–16 (factor 11.6: 3160 short dots, throughput
now — a 4-row blocking of `_dot_rows` is the next shape), setup 7.6
(diffuse: the row copies through `LayoutTensor`, the limit loop's 20 reads
a joint), Hessian build 6.3, line search 4.4; outside it — collision
(SAP broad 6.3, 36 GJK-vs-plane calls on the mesh geoms 4.7, and two
per-pair filters whose probe cost is mostly the timer's own), the two LDL
factors 6.6 each (anc 0.7 + gather 0.6 + elimination 4.3 + dense scatter
0.9: compact `qLD` storage read directly by the solves would drop the
scatter and the zero, and the ancestor table is now rebuilt SIX times a
step — factor ×2, solve ×2, noslip, finalize — ~4 µs that one per-step
table would remove), fk 6.2, bodyvel / cdof / crba / rne ~4 each.

### 13.27 Standing after four rounds, and what is left — the consolidated list (2026-09-06)

Four rounds in two days (§13.20–§13.26), twenty-two commits, one method:
sub-probe the largest window, read what the code does per element, change
the loop shape and never the arithmetic — and when the arithmetic has to
change, gate against MuJoCo, not the old checksum. Nineteen of the
twenty-two were bit-exact.

**Where every row stands.** Ratios are §13.25's (measured this morning
against MuJoCo 3.10.0 per §13.15, one `mj_step(m, d, n)` call, no Python
in the loop) carried forward by today's interleaved deltas (§13.26); a
ratio is only ever an interleaved pair, never two sessions' absolutes.

| model | nv | ncon | cone | ours vs MuJoCo | where the rest is |
|---|---|---|---|---|---|
| dog_stand | 79 | ~9 | pyramidal | **0.82×** | noslip 20, Cholesky 14, collision ~22, LDL 2×6.6 |
| humanoid_cmu | 62 | ~13 | pyramidal | **0.91×** | Cholesky 9.5, line search 6.7, Hessian 7.6, setup 6 |
| humanoid | 23 | | pyramidal | 0.66× | |
| walker2d / ant / hopper | 9 / 14 / 6 | | pyramidal | 0.73× / 0.78× / 0.67× | fixed per-forward setup MuJoCo pays and we do not (§13.22) |
| half_cheetah | 9 | | pyramidal | not twinned | |
| reassemble5 | | ~231 | elliptic | 0.74× | |
| reassemble3 | | ~92 | elliptic | **1.14×** | collision ~150 (64 GJK at per-pair parity), elliptic noslip 28, per-iteration rebuild 23 |
| sawyer_reach | 15 | ~5 | elliptic | 1.06× | |

Two rows above MuJoCo. Both are the elliptic, contact-dense corner, and on
reassemble3 the excess is collision at per-pair parity with MuJoCo's
native CCD — an algorithmic gap (MuJoCo's mid-phase and its BVH-free
convex path), not a loop-shape one.

**Tried this fortnight and rejected, so nobody retries them blind:**

- reciprocal-multiply in the Cholesky (MuJoCo's form): rounding-level, no
  gain — the division was never on the critical path (§13.26);
- the noslip half-solve in gather form (each block collecting from its
  descendants): bit-exact and slower, one serial FMA chain per block
  (§13.26);
- hoisting the pyramid-edge reads out of the Hessian build: bit-exact,
  no gain (§13.24);
- ancestor tables for the noslip half-solve on their own: the cost was a
  read-modify-write chain, not the pointer chase (§13.26);
- a dense `M⁻¹ Jᵀ` cache for the noslip: slower than the half-solve it
  replaced (§13.20).

**What is left, ranked by the probes and by what each would take.** The
first block is dog / humanoid_cmu (µs a step, probe-inflated, §13.26); the
second is reassemble3.

1. **One ancestor table a step, not six.** `_dof_ancestors` now runs in
   the LDL factor (×2), the LDL solve (×2), the noslip and the finalize —
   ~0.7 µs each on dog, ~4 µs a step, for a table that depends only on
   `dof_parentid`. It belongs in `DynamicsScratch` (CPU-side, integer,
   built once when the scratch is), threaded to the six consumers. Bit-exact
   by construction. ~3% on dog, ~2% on humanoid_cmu.
2. **Compact `qLD` storage read by the solves.** The tree factor already
   works in MuJoCo's compact row layout and then zeroes and scatters a
   dense `L` for the solves (0.9 + ~0.3 µs of each 6.6 µs factor, twice a
   step). With ancestor lists in every consumer the compact row is the
   natural read (`Lc[k*nv + a]` for slot `a` of the list), so the scatter
   and the zero go, and the solves read contiguous rows. Consumers:
   `_ldl_solve_tree_env`, `_m_inv_tree_env`, `_minv_apply`,
   `_tree_solve_cols`, the noslip's half-solve, the finalize. ⚠ Only the
   TREE path — the dense-block factor and every GPU kernel keep dense `L`,
   so the storage convention becomes path-dependent and must be stated on
   the field. ~3 µs a step on dog.
3. **The Cholesky's 3160 short dots** (11.6 µs on dog, throughput-bound
   now). Each pair is a `W`-lane dot of average length 26 plus a
   `reduce_add`, a tail and a divide; a four-row blocking of `_dot_rows`
   (four rows of `L` against one, sharing its loads and loop control)
   halves the per-pair overhead. Bit-exact if each accumulator keeps its
   order. Perhaps 11.6 → 8.
4. **The pyramidal noslip, 20 µs a step on dog.** Its half-solve (5.6) is
   now a memory chain through ~1260 hops that neither ancestor lists nor a
   gather form shortened; the Gram (3.3) and the two full solves (2.9
   each) are at their arithmetic. The lever left is structural: MuJoCo
   projects `AR` for ALL rows once per solve and the noslip reads it —
   ours projects the swept rows only, which is already less work — so what
   remains is the hop count itself (dog's max dof depth is 37).
5. **The Newton's setup (7.6) and Hessian build (6.3).** Diffuse: the
   contact rows are copied out of the solver workspace through
   `LayoutTensor` (nc·NE·nv reads), each limit row zeroes `nv` entries and
   reads ~20 joint fields, and the Hessian's outer products gather through
   `je_ix`. Pointer forms and a per-row dense segment (the support is
   ancestor-closed, so a contact row's nonzeros lie in one tree segment)
   would take each to about half; ~5 µs a step in total.
6. **Collision on dog** (~22 real): the SAP broad phase 6.3 for 296 geoms,
   36 GJK-vs-plane calls on the mesh geoms near the floor 4.7, and the
   two per-pair filters whose probe cost is mostly the timer's own. The
   plane-mesh path could reject on the mesh's bounding sphere before GJK
   (MuJoCo does); the rest is at MuJoCo's shape.
7. **The kinematics stages** (fk 6.2, bodyvel 4.3, cdof 4.3, crba 4.0,
   rne 4.3 on dog): all `LayoutTensor`-indexed, joint-major, no
   algorithmic defect left after §13.11 and the joint map. Pointer forms
   would give 20–30% of each; ~5 µs a step, and the same on humanoid_cmu.

On reassemble3 (§13.25's probes, unchanged by this round — it has no
excludes and its noslip is elliptic):

8. **Collision ~150 of ~240**, with the 64 GJK calls already at per-pair
   parity. What MuJoCo has and we do not is a cheaper convex path for
   box-box (its `mjc_BoxBox` face/edge routine where we run GJK+EPA) and
   fewer calls from its mid-phase; §13.19 has the call-count comparison.
   This is the one item on the list that is an algorithm, not a loop.
9. **The elliptic noslip (28)** and **the per-iteration rebuild (23)**:
   the elliptic Hessian and cone updates are per-contact 3×3 blocks that
   still go through `Scratch` per element; the §13.21 pointer treatment
   was applied to the cache build only.
10. **The pair filters (~18)**: `find_predefined_pair` scans the pair
    table per candidate; reassemble3 has pairs. The exclude table got its
    sorted signatures this round (§13.26); the pair table wants the same,
    keyed on `(geom1, geom2)`.

Items 1–3 are a morning each and bit-exact; 8 is the only one that would
change results and needs its own MuJoCo gates.

### 13.28 Two regressions from the campaign, found by the Menagerie board and bisected (2026-09-06)

Reported after the fourth round, from the studio: unitree_g1 bouncing off
the floor "like a trampoline", ToddlerBot's arm passing through its chest,
and a folded Jaco in the reassemble viewer. Neither bench checksum nor any
gate that ran during the four rounds had moved. The board did, at ONE step:

```
docs/menagerie_fidelity_harnesses/ab.py 1 <drive_base> <drive_head>   # base = worktree at 58ec75a2
  WORSE   5.941e-17 ->  1.010e-02  unitree_g1/scene.xml
  WORSE   2.442e-15 ->  4.494e-02  kinova_gen3/scene.xml
```

Six builds of `drive.mojo` from worktrees (never `git checkout`) bisected
that to `3bc98c55`; a folded-arm pose on ToddlerBot, compared contact set
for contact set against `mj_forward`, named the second one without a bisect.

**Defect 1 — the noslip read a factor the implicit step never built.**
`3bc98c55` moved the CPU noslip off the dense `M^-1` and onto the step's
tree LDL (`scratch.L`/`scratch.D`, as `mj_solNoSlip` solves on `qLD`). The
Euler and RK4 steps fill those with `ldl_factor` before their Newton, so
every bench row and every noslip gate stayed green: none of them says
`implicitfast`. `ImplicitIntegrator.step` LU-factors M instead, and the LU
keeps its factor in the SAME two slabs — so an implicitfast model with
contacts and `NOSLIP_ITER > 0` (the studio runs 1) solved its noslip
against the LU of M read as an LDL. Fix: factor the plain M's tree LDL in
the implicit step, AFTER the smooth `lu_solve` and BEFORE the constraint
seam (`integrator/implicit.mojo`), gated exactly like the noslip's own
`tree_ok`. Placing it beside `lu_factor` instead overwrote the LU the
smooth solve still had to read — kinova went 4.5e-02 -> 3.3e-01, which is
how the slab sharing was found. MuJoCo keeps `qLD` and the implicit `qH`
apart (engine_forward.c:1812); so do we now.

Gate: `test_noslip_implicitfast_vs_mujoco` — the chain of
`test_noslip_reaches_the_runtime_path` under `integrator="implicitfast"`,
both cones, through `StudioImpFast*`. ON arm 1.3e-13 / 4.1e-13 vs MuJoCo
where the pass is worth 0.09 / 0.18; on a pre-fix worktree it reads 0.35
and 17.8. Board after: g1 2.8e-17, kinova 2.7e-15.

**Defect 2 — the "cannot collide" skip dropped the predefined pairs.**
`3b97ce19` (§13.18) keeps a geom with `contype = conaffinity = 0` out of
the SAP sweep, and out of the naive O(N²) loop, because `filterBitmask`
rejects it against every partner. It does — but MuJoCo's `<pair>` table
never meets `filterBitmask` (engine_collision_driver.c:611–615, :779–780),
and ToddlerBot's 65 torso-to-arm pairs are between geoms whose class sets
both words to zero. Folded-arm pose, contacts against `mj_forward`: MuJoCo
31 (torso × five arm links, pelvis × hand), ours 0 before, 31 after, body
pair for body pair. Fix: a per-call `pair_geom` flag from the pair table
lets a named geom through the mask skip, in both loops. The existing gate
`test_contact_pair_vs_mujoco` (fixture "masks off + pair") had been red
since `3b97ce19` — it is not in the smoke manifest and nobody ran it during
the rounds; it is 7/7 now.

**What did not reproduce.** The reassemble fold: reassemble5 from its task
reset, 3000 steps at zero control, tracks MuJoCo to 1e-5 on every arm joint;
under the viewer's `sweep` drive (600 env steps, frame_skip 20) joints 1–3
agree to 1e-3 and joint 4 to 0.1 rad in plateaus that jump only at brick
collisions, which the pre-campaign build also shows (0.02). Nothing folds
and nothing sticks on the CPU Euler path. ToddlerBot's board residual
(`waist_yaw` 4e-02 at 100 steps, MuJoCo itself stable to 4e-12 under a
1e-12 kick) is IDENTICAL on the pre-campaign build — an older defect,
open. kinova_gen3's 1.7 at 300 steps is chaos: MuJoCo kicked by 1e-12 lands
3.2 away.

**Cost.** Both fixes are bit-exact on every bench row (all eight `qsum`
identical) and inside noise on the timings (dog 121.6 → 122.2, humanoid_cmu
64.6 → 64.4, reassemble3 239.0 → 238.9, reassemble5 548 → 532 µs, MIN of 3
interleaved). Board after both fixes: 84/84 scenes ≤ 1e-9 at N=1; at N=50
the same eight scenes above 1e-3 on both builds and nothing new.

**The lesson the rounds should have carried.** A routine that gains a new
input from a shared scratch has to be checked at EVERY caller that reaches
it — the integrators are three, the bench rows exercised one. And the board
A/B costs seconds once the binaries exist; it belongs after every round,
not after the report.

### 13.29 The toddlerbot residual was two missing rows, and the reassemble5 asset has no pose (2026-09-06)

Follow-up on the two items §13.28 left open, plus the studio question.

**reassemble5 from the studio.** The asset has no `<keyframe>`; the task pose
is built by dm_control's `initialize_episode`, which the viewer replays and
the studio does not. From `qpos0` all five bricks sit at the origin inside
each other: MuJoCo reports 330 contacts on step one, our probe caps at its
128-contact budget, and the two disagree from the first step (finger_1
8.4e-02 at N=1). Not an engine defect; the asset needs a keyframe carrying
the task pose if it is to be opened in the studio (the bench writes that
pose to a file for the MuJoCo twin — `harness.write_pose` — and the same
numbers would do).

**kinova_gen3 at 300 steps is chaos.** MuJoCo itself, kicked by 1e-12 on one
dof, lands 3.2 away at 300 steps; both our builds sit inside that envelope
(9e-05 and 1.7), and at 120 steps both are at 1e-09.

**ToddlerBot: `waist_yaw` 4e-02 at 100 steps, MuJoCo stable to 4e-12.** The
method that found it, since neither the board nor any gate had: step
MuJoCo to state K, hand that exact state (qpos AND qvel) to our runtime
probe, take ONE step on both sides at shared `<option iterations=N
tolerance="0">`, and sweep K. The error is 1e-14 without contacts, 1e-5 to
1e-4 per step with the six foot contacts, and independent of N on both
sides — a converged-but-different answer, not a convergence gap. Ablating
the model one attribute at a time from the SAME state (not from an ablated
trajectory, which confounded the first pass): removing the four neck
`connect`s takes it to 6e-15; softening their `solimp` from 0.9999 to 0.9
takes it to 7e-07; damping, `eulerdamp`, the cone, `frictionloss`, the
joint equalities and the anchor sites change nothing. Then the row itself,
dumped from a traced build against MuJoCo's `efc_*` at that state: J to
1e-17, D, pos, vel, KBIP, invweights, M — all equal; `aref` off by 5e-03.

*Defect 1 — `J̇·v` on connect and weld rows.* MuJoCo 3.10's
`mj_referenceConstraint` ends with `mj_Jdotv`, which subtracts the anchor's
`J̇·qvel` (the centripetal/Coriolis part of its acceleration) from `aref` on
every connect and weld row. None of the three older reference trees has the
routine (3.3.6, 3.5.1, 3.6.0), and neither did `build_weld_equality_rows`.
Zero at rest, so every connect/weld gate written at rest was green. Ported:
an `mj_comVel` pass (per-body `cvel`, per-dof `cdof_dot`) once per call
when a connect or weld exists, `mj_jacDot` folded with the matvec per
anchor, and the weld's three-term quaternion product for its rotational
rows. Gate: `test_equality_jdotv_vs_mujoco` — a four-bar closed by a
connect swinging at 8 rad/s and two free boxes welded and spun; pre-fix
2.9e-02 / 5.6e-02 after one step, rounding after.

*Defect 2 — a `<fixed>` tendon's limit was never a row.*
`build_tendon_limit_rows` had `else: continue` for every non-spatial
tendon, with a comment saying the builder was not given `qpos`. ToddlerBot's
waist is coupled by two fixed tendons with `range="-0.001 0.001"`; the
one-step error jumped from 1e-13 to 3.4e-02 on exactly the step MuJoCo's
`limTen` row appears (K=19), and back to 1e-13 two steps later when the
row is satisfied on MuJoCo's side too. Fixed by threading `qpos` through
the three call sites and giving the builder the same fixed-tendon
length/Jacobian the equality builder has. Gate:
`test_fixed_tendon_limit_vs_mujoco`; pre-fix 2.06 at five steps.

*What is left on ToddlerBot* is a mesh-versus-plane manifold choice: at
K=35 both engines report six foot contacts and agree on five; for the third
vertex of the left foot MuJoCo picks (−0.021, 0.047) and we pick
(0.036, 0.055), both under the plane. That is the hull/polygon column the
board README already names (our exact hull keeps vertices qhull merges) and
it is an event, not a drift: 3.3e-03 at 50 steps on `toddlerbot_2xm`, 3e-14
on `toddlerbot_2xc`.

**Board, the two fixes against the §13.28 build, N=50:** 0 worse, 9 better —
toddlerbot_2xc 1.4e-02 → 3e-14, agility_cassie 1.3e-05 → 3e-15,
ufactory_xarm7 1.3e-08 → 2e-15, robotiq_2f85 / _v4 to 1e-12 / 3e-15,
stanford_tidybot 1.0e-03 → 7e-08, toddlerbot_2xm 1.2e-02 → 3.3e-03. Scenes
at or below 1e-9: 64 → 67; above 1e-3: 8 → 6. At N=1 nothing moves above
1e-12.

**Bench:** every checksum unchanged except sawyer, whose mocap weld now
carries the term (qsum 52.31560 → 52.31570) — the one bench row with an
equality. Timings inside noise; sawyer +0.5 µs of 16 for the velocity pass.

**Two things the gates caught before the commit, worth keeping.** The
weld's rotational term was exact at identity and 0.5% off once the bodies
rotated: the builder's `qrel` is already `q_a * relpose`, and the port had
multiplied by `q_a` again — the jdotv gate's three-step arm (1.7e-06 where
one step read 1e-10) is what showed it, and a row dump at a rotated state
named it. And the `cvel`/`cdof_dot` scratch was first sized on the heap
(`Scratch[.., 0]`), which the CPU path is happy with and the GPU Newton
kernels are not: `test_equality_tendon_fields`' CPU-vs-GPU parity leg went
red. Sized by `cap[D.NBODY]()` / `cap[D.NV]()` now — static on a
compile-time model, the heap leg only on a dynamic one. That gate's
part-B golden is a self-frozen pin on a MOVING weld, so the `J̇·v` term
moves it by design — 23698.22 → 23708.37, 4.3e-04, inside the gate's own
1e-3 — and part A moves 2.5e-06 as its injected tendons gain their limit
rows. Both pins stay; the harvested numbers are recorded in the file, per
its own rule that a moved golden is explained, never bumped.

### 13.30 The three-tree board: Menagerie + Gymnasium + dm_control through the runtime engine (2026-09-06)

The question was whether the speed advantage over MuJoCo holds on every
scene and whether any fidelity gap had been missed, asked of the RUNTIME
engine because one instantiation per model does not scale to 118 scenes.
`docs/menagerie_fidelity_harnesses/bigsweep.py` runs both columns from two
binaries (`drive`, `rbench.mojo`) over every loadable `scene*.xml` in
Menagerie, every Gymnasium MuJoCo asset and every dm_control suite model.

**Fidelity, 50 steps of the board's random controls, |d(qpos)|max vs MuJoCo
3.10.0, 116 scenes compared** (5 not comparable: `ms_human_700` ×3 will not
load in MuJoCo, `lqr` has no joints, `iit_softfoot` attaches a radian model
into a degree scene, which our expander refuses):

| | scenes | ≤1e-9 | 1e-9..1e-6 | 1e-6..1e-3 | >1e-3 |
|---|---|---|---|---|---|
| Menagerie | 84 | 69 | 6 | 4 | 5 |
| Gymnasium | 14 | 10 | 1 | 1 | 2 |
| dm_control | 18 | 17 | 1 | 0 | 0 |
| **all, after this section's fixes** | 116 | **96** | 8 | 5 | 7 |
| all, before | 116 | 92 | | | 10 |

The two trees the board had never covered found four defects in an
afternoon, three of them fixed here and gated:

1. **The inertia-box fluid model rotated the velocity into the BODY frame
   and built the box from the INERTIAL-frame diagonal** (`dynamics/
   fluid_forces.mojo`). `mj_inertiaBoxFluidModel` does both in `ximat`
   (`xquat ⊗ iquat`); a `fromto` capsule along the body's x axis has its
   principal z along that axis, so the pressure-drag faces were paired
   with the wrong velocity components. Gymnasium's swimmer: 9.6e-08 at step
   1, 5.6e-02 at 50; the injection probe with `viscosity=0` and `density=0`
   ablations pinned it to the density term (viscous term symmetric, exact).
   After: 1.1e-18 / 1.2e-15. Also moved by it: `flybody` 2.3e-03 → 1.3e-09
   and `skydio_x2` 3.4e-04 → 3e-16, the two Menagerie scenes with a fluid
   `<option>`, both previously filed as something else.
2. **Both tendon spring sites skipped MuJoCo's REFSAFE clamp**
   (`constraints/tendon_limit.mojo`, limit and equality rows). MuJoCo raises
   `solref[0]` to `2*timestep` before it becomes a stiffness
   (engine_core_constraint.c:2029); the inline `1/(d² tc² dr²)` did not.
   dm_control's quadruped couples each leg with `<equality><tendon
   solref=".005 .5">` at `dt=0.005`, so our row was four times too stiff:
   2.9e-02 at 50 steps, exactly 1e-16 once the equality was ablated to the
   default solref, and 8.9e-16 now that both sites go through
   `solref_spring_damper`, the helper that already carried the clamp for the
   twelve sites it replaced. A rule written inline twice, again.
3. **`<custom><numeric name="init_qpos">` overrides our `qpos0`; MuJoCo
   ignores it.** Gymnasium's ant ships one from the mujoco-py era (z 0.55,
   ankles ±1.0 rad) and our parser applies it (`_fill_qpos0`, step 3,
   mirroring the legacy parser). MuJoCo's ant starts at z 0.75 with every
   ankle at 0 — OUTSIDE its `range="30 70"` — and its first step is a 1300
   rad/s² limit shove. The board row (2.9e-01) is therefore two different
   initial poses, not a solver defect; per-dof dumps had it reading
   `pos=1.0` for the last joint and a joint-address hunt found nothing
   because the address was right and the pose was not. Left as a decision
   in the first cut of this section; taken in §13.31 below.
4. **humanoidstandup diverges 7.2e-05 at step 21 with a two-step history
   dependence, not localised.** From MuJoCo's injected state at K=20 one
   step is exact (4e-14) and so are five; from K=19 two steps are exact;
   from K=18 three steps reproduce the 7.2e-05. Step 19 has five
   margin-band contacts (butt, arms), 20 has none, 21 gains the left foot.
   Not tolerance (both at `tolerance=0`), not `qacc_warmstart` (flag
   disabled, honoured), not MuJoCo's PGS (reference forced to Newton), not
   the per-slot force warm start (the Newton path only writes it). Something
   a contact-free step leaves standing from the step before it. Repro:
   `INJ_STEPS=3 inject1.py <standup_tol0.xml> 18`.

Also read off the board and not defects: `point` (2.5e-04) is a sphere
resting at EXACTLY `dist = 0` with no vertical dof, so `con->exclude = dist
>= includemargin` is a rounding coin on both sides — moving it 1 mm either
way makes the two agree to 1e-17; `humanoidstandup`'s remaining growth to
1e-2 by step 50 and the anymal/spot/go1 rows at 1e-6 are the trajectories
after an event, and hello_robot_stretch_3 / toddlerbot / tetheria are the
rows §9 and §13.29 already own.

**Step time through the runtime engine** — 117 scenes, `rbench` vs MuJoCo,
same keyframe, `ctrl=0.1`, contact capacity 2× MuJoCo's observed max, MIN
of three interleaved rounds:

| | median ours/MuJoCo | faster than MuJoCo | within 2× | above 4× |
|---|---|---|---|---|
| Menagerie (84) | 2.63 | 0 | | |
| Gymnasium (14) | 2.48 | 0 | | |
| dm_control (19) | 2.46 | 1 (quadruped 0.93) | | |
| all | **2.49** | 1 | 27 | 15 |

**That number is the STUDIO's path, not the engine the training loop
builds, and the calibration says so.** The five Gym models through the
compile-time `Phyics3dEnv` (bench_gym) against `rbench` on the same XML
against MuJoCo, interleaved, min of three:

| model | compile-time | runtime (studio) | MuJoCo | runtime / compile-time |
|---|---|---|---|---|
| walker2d | 20.7 | 59.8 | 24.0 | 2.9× |
| hopper | 10.5 | 33.9 | 14.6 | 3.2× |
| half_cheetah | 4.18 | 11.3 | 4.69 | 2.7× |
| ant | 27.3 | 66.7 | 32.6 | 2.4× |
| humanoid | 50.9 | 134.9 | 82.4 | 2.7× |

The compile-time engine still beats MuJoCo by 1.12–1.6× on all five, as
§13.22 had it. The runtime path is 2.4–3.2× slower than it on the same XML,
uniformly, and `sample` on hopper puts a quarter of the step in
`_platform_memset`/`__bzero` and tcmalloc: the dynamic leg of `Scratch`
allocates and fills a `List` at every one of the ~200 scratch sites a step
crosses (126 in `newton_solve.mojo` alone), and the row-sized ones are
sized by the contact CAPACITY (hopper at cap 128 → 8: 42 → 33 µs). The
studio aliases also carry `MAX_CONDIM=6` (12–24%, `studio/stepping.mojo`)
and no `CRBA_TREEWALK`. Dividing the runtime board by that calibration
puts the production engine at roughly parity with MuJoCo on the median
Menagerie scene and behind it on the 15 scenes above 4× — flybody (nv 108),
pal_talos, apollo, aloha, trossen_wxai — which is consistent with §13.22's
"the advantage is a constant and vanishes as nv grows". Whether the
runtime path can be brought to the compile-time one (a step-scoped arena
for the dynamic leg; the studio would gain the same 2.5×) is the next
optimisation item, and it is the prerequisite for reading this board as a
production number.

**⚠ `pndbotics_adam_lite` reads 24.9× and is a measurement artefact**: 68
µs/step when run alone, 644–651 in both sweeps, because a 30 s `mojo build`
overlapped that row both times. A three-round MIN did not save it. Build
nothing while a perf sweep runs.

**Gates, all on the final tree:** PASS — `test_fluid_fields_vs_mujoco`,
`test_swimmer_fk_vs_mujoco`, `test_fluid_wire_fields`,
`test_batched_env_fields_swimmer`, `test_fixed_tendon_limit_vs_mujoco`,
`test_spatial_tendon_equality_vs_mujoco`, `test_equality_tendon_fields`
(goldens untouched), `test_equality_jdotv_vs_mujoco`,
`test_tendon_rows_live_budget_vs_mujoco`, `test_tendon_index_order_vs_mujoco`,
`test_newton_blocked_tendon_fields`, `test_limit_solref_per_joint`.
`test_wrap_tendon_vs_mujoco` passes its 24 wrap poses and then dies on its
`iit_softfoot` leg with the expander's radian-into-degree attach refusal —
red since 8c788511 (2026-08-19), the same reason that scene is not on the
board, and not this section's. Bench checksums: the Gym five bit-identical
before and after (no fluid, no tendon rows in them).

### 13.31 `qpos0` is MuJoCo's: the `init_qpos` numeric is no longer applied, and the Ant env resets like Gymnasium (2026-09-06)

The decision on §13.30's third item: align to MuJoCo. `_fill_qpos0` no
longer copies `<custom><numeric name="init_qpos">` over the pose it builds
from joint `ref`s and free-joint body poses. MuJoCo's `mj_resetData` never
read that numeric; mujoco-py's `MjSim` did, which is where the legacy parser
took it from, and Gymnasium on the current bindings resets from `data.qpos`
after load, i.e. `qpos0`. Gymnasium's ant is the only model in the tree
carrying one.

**Board:** Gymnasium ant 2.856e-01 → 4.4e-16 at 50 steps (1.5e-16 at one);
every other row unchanged. `test_qpos0_vs_mujoco` pins `qpos0` to
`MjModel.qpos0` elementwise on ant, humanoid and dm_control's quadruped, and
on an inline fixture whose numeric says 9 everywhere so the gate cannot pass
by coincidence.

**The Ant env** (`mojo_rl/envs/ant/`) now resets from z 0.75 with every
ankle at 0, which is outside `range="30 70"`, so its first step is MuJoCo's
limit shove — exactly what Gymnasium's Ant-v5 does, and what our env had
been sparing itself with the mujoco-py pose (z 0.55, ankles ±1 rad).
Nothing in the env read the old pose by value: the reset is `qpos0 + 0.1
uniform` on both devices (`reset_data` / the GPU reset kernel through
`pose_meta`), `init_qpos_gpu` is a no-op for Ant, the inverse weights were
already built from the joint records at MuJoCo's `qpos0` rather than the
reset pose (§ `invweight.mojo`), and the healthy band [0.2, 1.0] contains
0.75. `tests/envs/test_ant_env_reset_is_qpos0` pins the reset height and
runs three random episodes from it. A policy trained on the old start pose
will see a different first second of every episode; that is the price of
matching the reference, and it is Gymnasium's first second too.

The ant asset keeps its `<custom>` block: it is a verbatim copy of
Gymnasium's file, and the parser now treats it the way MuJoCo does.
`test_ant_fk_vs_mujoco`'s bent-ankle case stays as an explicit pose on both
sides, no longer labelled the default.

**Gates on the final tree:** PASS — `test_qpos0_vs_mujoco`,
`tests/envs/test_ant_env_reset_is_qpos0` (reset z 0.66–0.85; random
episodes 14 / 123 / 74 steps, where MuJoCo under Gymnasium's own protocol
gives 27–140), `test_ant_fk_vs_mujoco`, `test_constraints_vs_mujoco`,
`test_euler_fields_vs_mujoco`, `test_env_fields_mujoco_roundtrip`,
`test_newton_freejoint_vs_cpu`. `test_validate_vs_mujoco` passes all 35
fixtures and nine real models and then dies on its `iit_softfoot` row with
the same radian-into-degree attach refusal as the wrap-tendon gate (§13.30)
— two gates now name that expander gap, which is worth its own fix.

### 13.32 "Before the campaign every Menagerie scene was below 1e-9" — at ONE step, and it still is (2026-09-06)

The question, on reading §13.30's 69 of 84: was that a regression? No —
two boards were being compared. The board §9 and §13.28–29 report is
`sweepN.py 1`: ONE step from the keyframe, the metric that climbed 50 → 77
→ 80 of 85 through the fidelity campaign. Today it reads **84 of 84 at or
below 1e-9, none above 1e-3** — the best it has read. The 69/84 is the
FIFTY-step board, a harder metric that was never all-green and that §13.29
already reported at 64 → 67.

To close the question properly rather than by recollection, an A/B on the
fifty-step board: `drive` built from `7f4fa8e5` (2026-09-04, the last
commit before anything in this campaign touched `physics3d`) against
today's, both against the same MuJoCo answer, same controls:

| N=50, Menagerie | pre-campaign (7f4fa8e5) | today |
|---|---|---|
| scenes compared | 85 | 84 |
| at or below 1e-9 | 65 | **69** |
| above 1e-3 | 8 | **5** |

Rows that moved more than 10× — eight better (toddlerbot_2xc 1.4e-02 →
3e-14, flybody 2.3e-03 → 1.3e-09, tidybot 1.0e-03 → 7e-08, skydio 3.4e-04 →
3e-16, cassie 1.3e-05 → 3e-15, xarm7, both robotiq), one "worse":
shadow_dexee 2.5e-13 → 1.1e-11, rounding-level and still four orders below
the line, on the scene whose noslip pass §13.28 turned on. And one scene
LOST: `iit_softfoot` loaded on the old binary (3.0e-04) and is refused on
today's, because the expander now reads an absent `<compiler angle>` as
`degree` (the `so101_tabletop` fix) and that scene attaches a `radian`
sub-model into a scene that says nothing. The refusal is honest — a text
splice would reinterpret the sub-model's angles — but it costs one board
row and two gates (`test_wrap_tendon_vs_mujoco`,
`test_validate_vs_mujoco`); converting the sub-model's angles at splice
time is the fix, and it is the next parser item.

The fifteen Menagerie rows above 1e-9 at fifty steps are, in order: the
tetheria hand (§9, 3.7e-01), the three toddlerbot poses and
hello_robot_stretch_3 (§13.29 and §9, the mesh manifold), then the anymal /
spot / unitree / tidybot rows at 1e-6–1e-8 — event-driven trajectories
after a contact — and crazyflie / flybody at 1e-8 / 1e-9. Nothing in that
list is new to this campaign.

### 13.33 `iit_softfoot` is back: the splice converts a sub-model's angle units, and two default-class gaps it exposed (2026-09-06)

§13.32's one lost scene. MuJoCo compiles each attached model under ITS OWN
`<compiler angle>` and attaches the compiled result; we splice text, and
the host's compiler then reads the sub-model's numbers. The expander used
to refuse a unit mismatch. It now scales the sub-model's angles into the
host's units before the splice, using the exact attribute set MuJoCo's
compiler scales by `degree` (`user_objects.cc`, 3.10.0): `euler` (all
components), `axisangle` (the angle only), joint `range` on hinge AND ball,
joint `ref` / `springref` on hinge only. The joint attributes depend on the
joint's TYPE, resolved as MuJoCo resolves it — `class=`, the enclosing
body's `childclass=` (inherited down the tree), parent classes, the root
`<default>` — so a slide joint's range, in metres, does not move. A
`<default>` block is scaled under the type IT resolves to, and a joint that
inherits an attribute from a block of the other kind gets it materialised
on its own tag in its own units (MuJoCo confirms both traps: a slide joint
under a root `<joint ref="0.1">` keeps `qpos0 = 0.1` raw; an explicit hinge
in a slide class has its inherited range scaled). A differing `eulerseq`
still refuses; no model in the tree does that.

The gate (`test_attach_angle_units_vs_mujoco`, fixtures
`fixtures/attach_units/`) compares `parse_xml_full(expand_mjcf(...))`
against `MjModel.from_xml_path` on the SAME scene file — two routes to one
model — for every joint's range / qpos0 / qpos_spring and every body, geom,
site and camera quaternion, on a radian asset attached twice into a degree
scene and a degree asset into a radian scene: 261 checks, worst 1.1e-16.

Writing that fixture found three things that were NOT the conversion, all
reproduced with the asset loaded plain, no attach:

1. **Joint `ref` never consulted the class default.** `springref` and
   `range` did. A `<default><joint ref=...>` gave every inheriting joint
   `qpos0 = 0`. No reference model writes one, which is why no board saw it.
2. **Geom orientation consulted the class default for `quat` only.**
   `euler` / `axisangle` / `xyaxes` / `zaxis` from a class read as identity,
   while sites resolved all five. `anymal_b` carries a class `<geom euler>`;
   the fixture's root `<geom euler="0.3 0 0">` was the catch.
3. **The sub-model's root `<default>` was merged into the host's root**, so
   the scene's FLOOR inherited the foot's `<geom euler>`. MuJoCo attaches a
   model's default tree as a class `<prefix>main` under the host's root
   (`mj_saveLastXML` prints exactly that); the splice now wraps it that way
   and points every top-level spliced element that names no class at it
   (`childclass=` on bodies). What still differs: a class nested under the
   host's root inherits the host's root attributes in a text model where
   MuJoCo's attached tree does not; no scene in the tree attaches under a
   root `<default>` that sets anything.

Boards with the new binary, same protocol as §13.30 / §13.32:

| | before | after |
|---|---|---|
| one step, Menagerie, at or below 1e-9 | 84 of 84 | **85 of 85** |
| fifty steps, three trees, loaded | 116 | 117 |
| fifty steps, at or below 1e-9 | 96 | **98** |
| fifty steps, above 1e-3 | 7 | 6 |

`iit_softfoot`: 1.8e-17 at one step, 3.0e-15 at fifty. No other row moved
by 10× in either direction (the ant row's 0.29 → 4e-16 is §13.31's). The
fifteen attach / include / defaults / composition gates pass, and the two
that had lost the scene (`test_wrap_tendon_vs_mujoco` 56/56,
`test_validate_vs_mujoco`) exercise it again.

### 13.34 The fifty-step board, row by row: one clamp, and five reasons that are not defects (2026-09-06)

The nineteen three-tree rows above 1e-9 at fifty steps, each traced step
by step (`trace.py`: the board's protocol on one scene, both engines'
contact counts and MuJoCo's iteration count per step) and then stepped ONE
step from MuJoCo's own state at the step before the jump (`one.py`, per
dof, both contact lists, §13.29's method without XML surgery). Every row
now has a mechanism. One was a defect; it is fixed below.

**The defect: a `solimp` dmax of exactly 1 reached the friction damper
unclamped.** Seven rows — anymal_b/c, spot, spot_arm, a1, go1, go2 — sat at
1e-7…1e-6 and were already there at five steps, contact counts equal,
converged on both sides (`tolerance="0"` moved nothing). From MuJoCo's
state at step 1 the knee's qvel was off 9e-6 per step; from the keyframe
(rest) the step was exact to 6e-17. Ablating the anymal foot one attribute
at a time from the SAME state: `condim`, `frictionloss`, `impratio`, the
cone, damping, `eulerdamp` changed nothing; removing the foot's
`solimp="0.015 1 0.03"` OR its `priority="1"` took it to 6e-15 — and so
did writing the 1 as 0.9999. MuJoCo clamps `solimp[0]`, `[1]`, `[3]` to
[mjMINIMP, mjMAXIMP] = [0.0001, 0.9999] before deriving anything
(`engine_core_constraint.c:2044`). Ours clamped them before the impedance,
and `solref_spring_damper`'s docstring said the caller clamps the dmax it
receives for `B = 2/(dmax*timeconst)`. The normal-row caller did; the two
friction-row callers in `contact_solve` (pyramidal `_kb_c`, elliptic
`_kb_e`) passed the contact's raw `solimp[1]`. With `priority="1"` the
foot's parameters win unmixed, so the 1 came through where an
equal-priority mix with the plane's 0.95 would have landed at 0.975 and
never touched the clamp. B off by 1e-4 is invisible at rest — the damper
term is B·vel — which is why the one-step board never saw it. The clamp
now lives inside `solref_spring_damper`, idempotently, for all fourteen
callers. Gate: `test_contact_solimp_clamp_vs_mujoco` — a `priority="1"`
sphere with that solimp sliding on a plane; pre-fix 6.7e-06 after one step,
after 4.5e-19 / 1.6e-18 / 3.6e-15 at 1 / 5 / 30. The seven rows:

| N=50 | before | after |
|---|---|---|
| anybotics_anymal_b | 3.2e-06 | 8.5e-11 |
| anybotics_anymal_c | 5.4e-06 | 8.6e-11 |
| boston_dynamics_spot | 3.1e-06 | 1.3e-11 |
| boston_dynamics_spot arm | 1.9e-06 | 2.0e-11 |
| unitree_a1 | 9.2e-07 | 2.1e-11 |
| unitree_go1 | 9.2e-07 | 4.4e-11 |
| unitree_go2 | 7.8e-08 | 1.7e-10 |

The Gym bench checksums are untouched by construction: the change is
`dw = clamp(d_width)`, the identity for every dmax the five models carry
(0.95 default, 0.99).

**The reference solver, twice.** Gymnasium's `humanoid.xml` and
`humanoidstandup.xml` declare `solver="PGS" iterations="50"`; MuJoCo hits
that cap for the first three standup steps and again whenever a contact
appears, and ours runs Newton. Text-ablating `PGS` to `Newton` on both
sides: standup 3.7e-06 → 1.4e-17 at five steps and 6.7e-16 at ten,
humanoid 4.7e-07 → 1.2e-15 at fifty. Those two rows measure MuJoCo's own
unconverged PGS, not us. (These are the only two models in the three trees
that declare a non-Newton solver.)

**The iteration cap, and a convergence-rate difference under it.**
`tetheria_aero_hand_open` (3.7e-01) is 1e-15 through step 7 and 3.7e-01 at
step 8, the first step where MuJoCo's `solver_niter` reaches the scene's
`iterations="5"`. With `iterations="200"` on BOTH sides the scene is 1e-14
for all 25 steps traced, and one step from the same cold state at
iterations 7, 8, 10 agrees to 2e-11. So the physics is right and the row
is two truncated Newton runs stopped at different iterates — MuJoCo's own
5-iteration answer is 1e-2 from its converged one. What is ours: from the
same cold start our fifth iterate is 1700 from the converged thumb
acceleration where MuJoCo's is 5 (iterate 1 differs by 0.5%, iterate 2 is
IDENTICAL to 1e-12, iterate 3 diverges; the line-search cap is not it —
`ls_iterations` 8 and 50 give the same answer). §9.4.3 measured the same
thing and named it: our convergence RATE on this problem, not our answer.
Open; the per-iteration traces are in the harness (`MJ_STATS=1`,
`_PYR_TRACE`).

**RK4 holds the actuator force fixed across its four stages.**
`bitcraze_crazyflie_2` (1.9e-08) has no contacts and a one-step error of
3.3e-13 that the fluid terms do not touch (density and viscosity ablated
to zero: identical) and Euler does not have (5e-23). A Python replica of
`mj_RungeKutta` built on `mj_forward` matches MuJoCo to 0.0; the same
replica with `qfrc_actuator` frozen at the step's start matches OURS to
1.3e-23. MuJoCo re-evaluates actuation at every stage; our driver applies
actions once, before `RK4Integrator.step`, and the stages integrate that
`qfrc`. A body-fixed thrust rotating with the body is exactly the case; a
joint motor's `gear*ctrl` is not, which is why Gym's RK4 models never
showed it. Open: a per-stage actuator hook in the RK4 step (it is the
only Menagerie RK4 scene; tumbling flight amplifies the 3e-13 to 2e-8).

**Rounding under stiff dynamics.** `stanford_tidybot` (7.0e-08): 1.6e-10
at step one, no contacts, seven equality rows, `implicitfast`, and joint
accelerations of 3.6e4 — the per-step qvel error is 8e-8 on a qvel of 71,
1e-9 relative, then ×2 every eight steps. `point` (2.5e-04) is §13.30's
`dist >= includemargin` coin on a sphere resting at exactly zero distance.

**Contact geometry at 1e-9, and a self-colliding keyframe.** dm_control's
`dog` (6.6e-08): exact to step 10 without noslip (with the scene's
`noslip_iterations="4"` a 1e-9 difference appears at step 4 — the noslip
pass is another capped iteration), then 4e-9 at step 11 where the two
pelvis–tail contacts report `dist` 4e-9 apart between the engines — the
convex-collision family of §13.29's toddlerbot manifold, which is where
`toddlerbot_2xm` (3.3e-03 at 50, 9e-15 at 20) and the two `_pos` scenes
still sit. `hello_robot_stretch_3` (2.1e-02): its keyframe has the wrist
4.8 cm inside the base with 6e4 N of self-collision force on both sides,
plus a floor–base contact at exactly zero distance that MuJoCo keeps and we
drop; from there the two engines part at step 6. Not a state a simulation
is meant to start from (§13.29's reassemble5 note).

**Boards after the clamp**, same protocol as §13.33:

| | before | after |
|---|---|---|
| one step, Menagerie, at or below 1e-9 | 85 of 85 | 85 of 85 |
| fifty steps, three trees, at or below 1e-9 | 98 of 117 | **105 of 117** |
| fifty steps, above 1e-3 | 6 | 6 |

The twelve rows left above 1e-9: tetheria (cap), toddlerbot ×3 and dog
(convex manifold), stretch_3 (keyframe), standup and humanoid (PGS),
point (coin), tidybot (stiff rounding), crazyflie (RK4 actuation),
flybody (1.3e-09: 5e-11 at step one under its `noslip_iterations="3"`,
the same capped pass as dog's step 4).

### 13.35 RK4 re-evaluates the actuators at every stage (2026-09-06)

§13.34's open item. `mj_RungeKutta` runs `mj_forwardSkip` at each stage,
which recomputes `qfrc_actuator` at the stage's `qpos`/`qvel`.
`RK4Integrator.step` integrated the `d.qfrc` the driver applied at the
step's start through all four stages. For a joint `<motor>` the two are
the same constant, which is why the Gym suite's RK4 models — walker2d,
hopper, humanoid, ant — were exact; for a `<position>` servo (reads the
stage's `qpos`), a spatial-tendon transmission (its stage moment arm) or a
body-fixed site wrench (the stage's orientation) they are not, and the
Python replica of §13.34 pinned the old entry at 1.3e-23 with the actuator
frozen.

`step` is now a stage loop over three pieces — setup, dynamics, the
per-stage constraint solve — and a finish (warmstart save + combine), and
`step_actuated` is the same loop with `d.qfrc` zeroed and
`apply_actions_fields` + `apply_pose_transmission` re-run at stages 1–3
from the stage state (the driver's application IS stage 0). The activation
state is not advanced by the stages: `apply_actions_fields` integrates a
dyntype's `act` by `timestep` on every call, and the driver's call already
did that once, so the stages work on a copy and a dyntype actuator sees
its start-of-step activation at every stage (MuJoCo carries `act` in the
RK state; no RK4 model in the tree has a dyntype, so this is not yet
measurable). CPU, BATCH=1 — the actuation entry points are. `drive.mojo`
and `Phyics3dEnv`'s RK4 branch call it; a CONFIG that applied its own
actions keeps the frozen entry, since the integrator has nothing to
re-evaluate for it.

Gate: `test_rk4_stage_actuation_vs_mujoco` — a servo pendulum at dt 0.02
and an off-axis site thrust on a tumbling free body, each with a CONTROL
ARM that runs the frozen `step` and must disagree with MuJoCo (it does, by
3e-02 at five steps), then `step_actuated` at 1 / 5 / 30 steps: 7e-15 /
3e-14 / 8e-13.

| N=50 | before | after |
|---|---|---|
| bitcraze_crazyflie_2 | 1.85e-08 | **4.7e-20** |
| the fourteen Gym rows | unchanged to the last digit | |

Fifty-step board: **106 of 117** at or below 1e-9.

**Bench checksums, A/B against the previous commit** (the bench harness
calls `integ_rk4.step` directly, so it exercises the refactored loop, not
the new entry): walker2d, hopper, half_cheetah and ant bit-identical over
20000 steps; humanoid identical for 2010 steps (contacts from step 37) and
then 1 ulp of float32 (1.2e-10) at step 2011, 5e-07 by 3000. A per-substep
probe from an identical state shows `step` and `step_actuated` bit-identical
on humanoid for twenty substeps, so the re-application is the no-op it
should be for motors; the 1-ulp event is the refactor's inlining choosing
a different FMA contraction on a rarely-taken branch — the same shape as
§13.21's `Scratch` change, and gated the same way, against MuJoCo
(`test_humanoid_limits_fields_vs_mujoco`) rather than the old checksum.

### 13.36 LANDED (2026-09-06): the runtime path's `Scratch` takes its blocks from a pool — the three-tree median from 2.49× to 1.65× MuJoCo

§13.30 read the runtime (studio) engine at 2.4–3.2× the compile-time one
on the same XML and named the per-call heap `Scratch` of the dynamic leg as
the first suspect. This section is that item, and it was bigger than the
runtime-dims assessment had it: that assessment measured "heap scratch" at
~1.14× on a CRBA+LDL microbench, a handful of sites, and the number was
carried as the cost of the leg. The step crosses a few hundred.

**Counted, not inferred.** A `_Global` counter in `Scratch.__init__`
(temporary, `rbench` printed it), one step of the board protocol:

| model | heap scratches / step | bytes malloc'd + filled / step | of which `fill=` sites |
|---|---|---|---|
| hopper (RK4) | 314 | 168 kB | 88 |
| half_cheetah | 92 | 60 kB | 22 |
| ant | 329 | 341 kB | 88 |
| humanoid | 429 | 646 kB | 96 |

Every one of those was a `List[T](length=n, fill=...)`: a tcmalloc
`malloc` (its thread cache behind `pthread_getspecific`), a fill — the
`uninitialized=` sites too, since a `List` cannot be indexed without a
length — and a `free`. `sample` on the main thread of the hopper bench:
tcmalloc family 21%, `_platform_memset` 10%; humanoid 7% + 6%.

**What landed — `fields/scratch_pool.mojo`.** A process-wide free list
keyed by size class (four classes per octave; a request rounds up to a
quarter of its leading power of two, so a site whose length varies with
the row count lands on a bounded set of classes). Blocks link through their
own first word. `Scratch`'s heap leg takes a block at construction and
hands it back in `__deinit__`; after the first step nothing mallocs, and
the `uninitialized=` sites no longer fill. The static leg is untouched:
three inert fields (a null pointer, 0, a null pointer) beside the
`InlineArray`.

Not a step-scoped bump arena, which is what the item was filed as. An
arena reset at the top of `step` wants either LIFO release or no release
within the step, and neither holds here: Mojo destroys a value at its LAST
USE, so two scratches in one function are released in whatever order
their last reads fall, and the Newton constructs scratches inside its
iteration loop, so an arena without release would grow with the iteration
count. The free list has neither constraint and needs no reset hook in
four integrators — it is the arena's saving with the `List` leg's lifetime
rules.

Where the handle lives: Mojo has no mutable module-level `var` (nightly
2026-09: "use of unknown declaration"). The stdlib's own globals are
`std.ffi._Global`, a name-keyed slot in the compiler runtime; the named
lookup costs **7.4 ns** (it `memcmp`s the name), the fixed-index variant
`get_or_create_indexed_ptr(2)` **1.1 ns**, but slot 2 is the stdlib's
"reserved for prototyping" and not ours to take. A `Scratch` looks the
pool up ONCE, at construction, and keeps the pointer for the release;
after the change the lookup is 4% of hopper's step (`get_or_create_ptr`
1.8% + the runtime stub + its `memcmp`).

**Results, `rbench` interleaved head vs pool, MIN of 3 × 3000 steps, `qsum`
identical:**

| | before | after | |
|---|---|---|---|
| hopper (Gym, RK4) | 34.9 / 35.3 | **19.1 / 19.0** | 1.84× |
| humanoid (Gym) | 139.3 / 139.3 | **110.4 / 109.4** | 1.27× |

After, on hopper's main thread: `solve_newton` 47%, `compute_mass_matrix`
9%, contact detection 7%, memset 6% (the 88 `fill=` sites), the global
lookup 4%; the allocation family as a whole 30% → 10%.

**The three-tree perf board** (`bigsweep.py perf`, same protocol as
§13.30, MIN of three interleaved rounds, MuJoCo re-timed in the same
sweep and within 0.96–1.04 of §13.30's column):

| | median ours/MuJoCo | faster than MuJoCo | within 2× | above 4× |
|---|---|---|---|---|
| Menagerie (85) | 2.63 → **1.88** | 0 → 1 | 22 → 46 | 12 → 9 |
| Gymnasium (14) | 2.48 → **1.34** | 0 → 4 | 2 → 14 | 0 → 0 |
| dm_control (19) | 2.46 → **1.32** | 1 → 1 | 4 → 18 | 3 → 0 |
| all (118) | 2.49 → **1.65** | 1 → 6 | 28 → 78 | 15 → 9 |

Per row, ours before / ours after: median 1.27×, from 1.01× to 3.03×. The
small models gained the most — pendulum 4.3 → 1.4 µs, cartpole 17.9 →
6.3, inverted_pendulum 18.3 → 6.7, point 15.9 → 6.1 (four rows now at or
under MuJoCo) — because a fixed per-site cost is a larger share of a small
step, which is §13.22's argument seen from the other side. The rows that
did not move are the ones the solver owns: flybody (nv 108) 667 → 664,
aloha 161 → 157, pal_talos 461 → 446; there the malloc share was already
small and the remaining gap is the Newton at `MAX_CONDIM=6` with runtime
bounds. The Gym five through this path against §13.30's compile-time
column: walker2d 40.1 vs 20.7, hopper 18.6 vs 10.5, half_cheetah 6.9 vs
4.2, ant 48.9 vs 27.3, humanoid 105 vs 50.9 — **1.65–2.1×**, from 2.4–3.2×
(⚠ the sweep's rows, not the §13.30 calibration script's; same controls,
different step counts). What is left of that ratio, in the order the
profile supports: `MAX_CONDIM=6` in the studio aliases (12–24%,
`studio/stepping.mojo`), no `CRBA_TREEWALK` (`compute_mass_matrix` 9% on
hopper, 18% on humanoid before this change), the runtime bound itself
(~1.1–1.25× per the assessment's layout split), the 88 `fill=` sites, and
rows still sized by the contact capacity.

**Gates.**
- Checksums: `rbench` `qsum` identical head vs pool on hopper and
  humanoid; the compile-time bench (`bench_gym`) interleaved against the
  §13.35 build, all five Gym models `qsum` identical and 0.99–1.01× in
  time (the inert fields cost nothing).
- The fifty-step three-tree board through the pooled driver: **121 of 121
  rows identical to the previous board to the printed digit**, 106 of 117
  comparable rows at or below 1e-9 as in §13.35 — every uninitialized
  block the heap leg now hands over unfilled is read only after it is
  written, on 117 scenes.
- Tests: `test_dyn_dims_ldl`, `test_studio_honours_option_cone`,
  `test_structural_edit`, `test_noslip_implicitfast_vs_mujoco`,
  `test_tendon_rows_live_budget_vs_mujoco`, `test_dof_parentid_vs_mujoco`,
  `test_hfield_vs_mujoco`, `test_equality_tendon_fields` /
  `test_newton_blocked_tendon_fields` / `test_contact_solve_fields` (GPU
  parity on Metal — the static leg from a kernel's side),
  `test_rk4_stage_actuation_vs_mujoco`, `test_contact_solimp_clamp_vs_mujoco`,
  `test_attach_angle_units_vs_mujoco`, `test_fk_fields`: all PASS.
  `test_studio_honours_option_cone` reads 5 of 6 — its "the two cones
  differ" arm sees 1.2251e-13 between pyramidal and elliptic on its
  contact model — and reads exactly the same 5 of 6 with the same digits
  at `15ed3b37` in a detached worktree: pre-existing, not this change's
  (the pool cannot move arithmetic; a test that asserts two cones DISAGREE
  is measuring the scene, and is left for its owner).
- ⚠ Harness caveat met on the way: `rbench`'s final `qsum` depends on
  `rounds` (humanoid 153.52 at 1, 152.215 at 3, on head and pool alike) —
  something survives its per-round reset (qpos/qvel/warmstart/act are
  reset; the contact slots are not). Compare checksums at equal `rounds`.

**What the `List` was also doing, and its replacement.** A short `n` at a
heap site used to fail loudly (`List` bounds-checks); a pooled block
overruns into the next block silently. `Scratch.BOUNDS = True` (a
comptime flag, off in shipped binaries) keeps `n` on the heap leg and
aborts on any index at or past it, naming the index and the length —
positive control: a four-slot scratch written at index 4 aborts with
`Scratch: index 4 out of bounds for a heap scratch of length 4`; negative
control: hopper, humanoid, unitree_go1 and google_robot run 200 steps
under it without firing. Build with it on when sweeping sites.

⚠ The pool is not thread-safe and lives in one `_Global` slot; nothing in
physics3d steps from two threads, and the module docstring says what a
multi-threaded CPU leg would need. ⚠ `perf` rows are still not production
numbers — the studio aliases' `MAX_CONDIM=6` and missing `CRBA_TREEWALK`
remain — but the calibration is now 1.65–2.1× instead of 2.4–3.2×.

### 13.37 LANDED (2026-09-06): the two studio aliases priced, and the tree-walk CRBA landed on them

§13.36 left the runtime path at 1.65–2.1× the compile-time engine and named
two alias choices in `studio/stepping.mojo` as the next measurable terms:
`MAX_CONDIM=6` and no `CRBA_TREEWALK`. Both priced the same way — a variant
`rbench` per choice, four binaries interleaved on one machine, MIN of three
rounds × 3000 steps, contact capacity 16:

| model | base µs | condim 3 | tree-walk CRBA | both |
|---|---|---|---|---|
| walker2d | 39.8 | 0.85 | 0.95 | 0.81 |
| hopper | 18.8 | 0.89 | 0.97 | 0.86 |
| half_cheetah | 6.9 | 0.92 | 0.94 | 0.87 |
| ant | 48.8 | 0.95 | 0.82 | 0.76 |
| humanoid | 108.7 | 0.92 | 0.81 | 0.73 |
| google_barkour_vb | 58.3 | 0.97 | 0.94 | 0.91 |

They split by model size. Condim 3 is the pyramidal edge count per slot
(4 instead of 10) and is worth 8–15% on the small models, 3–5% on the
big ones; it is bit-identical to the base on all six, as the note in
`stepping.mojo` says (six is a superset of three). The tree walk is
O(NV·depth) against the dense CRBA's O(NV²·NBODY) and is worth 3–6% on
the small models and 18–20% on ant and humanoid. Together 0.73–0.91.

**Landed: `CRBA_TREEWALK=True` on the five studio aliases.** It is a
parameter on the existing instantiations (no build-time cost), and the
training env has run it since `phyics3d_env.mojo` called it a fix rather
than a knob. Condim 3 is NOT landed: 46 models in the tree need 6, and a
condim dispatch axis would double the studio's instantiations for 8–15% on
a tool that renders at 60 Hz — the standing decision in `stepping.mojo`.

**Gate.** The tree walk is a different summation order, so it moves the
last one or two digits of every checksum (1e-15 relative) and the stored
`qsum`s cannot gate it. The fifty-step three-tree board through the
tree-walk driver: **the same 106 / 4 / 1 / 6 rows in the four bins as
§13.36, no row changes bin**; 48 rows identical to the printed digit, 69
moved, the largest move flexiv_rizon4 2.8e-17 → 2.2e-16 and unitree_g1
7.4e-16 → 2.7e-15 — rounding, on scenes that sit at 1e-16.

**The perf board after both changes (§13.36 pool + this section):**
same protocol as §13.30, MIN of three interleaved rounds, MuJoCo re-timed
in the same sweep (median drift 1.01 against the §13.36 sweep):

| | median ours/MuJoCo | faster than MuJoCo | within 2× | above 4× |
|---|---|---|---|---|
| Menagerie (85) | 2.63 → 1.88 → **1.76** | 0 → 1 → 1 | 22 → 46 → 46 | 12 → 9 → **4** |
| Gymnasium (14) | 2.48 → 1.34 → **1.17** | 0 → 4 → 4 | 2 → 14 → 14 | 0 |
| dm_control (19) | 2.46 → 1.32 → **1.24** | 1 → 1 → 2 | 4 → 18 → 19 | 3 → 0 → 0 |
| all (118) | 2.49 → 1.65 → **1.50** | 1 → 6 → 7 | 28 → 78 → 79 | 15 → 9 → **4** |

(§13.30 → §13.36 → here.) Per row against the pooled sweep: median 1.08×,
and the gain sits where the dense CRBA's `NV²·NBODY` was: dog 387 → 200 µs
(1.74× → **0.92×** MuJoCo), humanoid_CMU 169 → 110 (1.54×), flybody 664
→ 462 (3.12×), robot_soccer_kit 225 → 136 (1.06×), the three toddlerbot
scenes 3.4–3.6× → 2.7×, dm_control's quadruped 0.72× → 0.58×. The Gym
five against §13.30's compile-time column: walker2d 37.7 vs 20.7, hopper
17.6 vs 10.5, half_cheetah 6.3 vs 4.2, ant 39.6 vs 27.3, humanoid 82.4 vs
50.9 — **1.45–1.8×**, from 1.65–2.1× (§13.36) and 2.4–3.2× (§13.30);
humanoid now reads 1.05× MuJoCo through the studio path. ⚠ One row in the
sweep is a load spike, not a regression: `unitree_go2` read 32.9 → 41.9 µs
with MuJoCo inflated 35 → 66 in the SAME round; re-measured interleaved at
capacity 32, pool 46.3 / tree-walk 43.4 (min of three) — the tree walk is
6% faster there as everywhere else.

Tests on the tree-walk aliases: `test_structural_edit` 66/66,
`test_validate_vs_mujoco` 83/83, `test_noslip_implicitfast_vs_mujoco` 3/3.

What remains between the runtime and compile-time engines after this is
the runtime bound itself — the loops the compile-time `NV` unrolls into
straight-line code — and the pyramidal edge count at `MAX_CONDIM=6`; the
pricing above puts the first at ~1.5× on every row and the second at
3–15%. Neither is an optimisation inside the engine: the first is a
decision to specialise hot loops on a few compile-time `nv` buckets, the
second the standing decision above.

### 13.38 MEASURED (2026-09-07): the campaign on NVIDIA — six of seven gates, and where the RTX 5090 step went

The September CPU campaign (§13.20–13.37) edited solver code the batched
GPU kernels share and matched every change on Apple only; the PYRAMIDAL +
NVIDIA route goes through the blocked Newton kernel, which never launches
on Metal. `tests/manifests/physics3d-gpu.txt` (`pixi run -e nvidia
test-physics3d-gpu`, commit 99357b9d) names the seven gates. Run on a
rented RTX 5090 box and on Apple, at HEAD a4b22e22 + 99357b9d:

| gate | where | result |
|---|---|---|
| `test_newton_freejoint_vs_cpu` (blocked vs CPU oracle; Ant, Humanoid, ThreeTrees) | NVIDIA | PASS |
| `test_newton_blocked_fields` (golden, walker2d) | NVIDIA | PASS |
| `test_noslip_blocked_kernel` (dog: noslip on the blocked kernel) | NVIDIA | PASS — blocked-GPU vs per-env-GPU **0.0** with the pass off and on; vs per-env-CPU 2.1e-5 off, 4.6e-4 on (float32, the sensitivity arm prices it) |
| `test_ldl_blocked` | CPU | 9/9 |
| `test_fields_mt_parity` | Apple | bit-exact, 3 steps |
| `test_cfrc_ext_batched_vs_cpu` | Apple | PASS |
| `test_newton_blocked_tendon_fields` (ball_in_cup: tendon rows on the blocked kernel) | NVIDIA | **DOES NOT COMPILE** — killed after 30 min; builds on Apple in 106 s and passes there (§13.35). Parked, see below. |

**The parked-slot probe, three sweeps side by side** (`scripts/p0_attrib.sh`,
1024 lanes, per-step ms; Sep 4 is the `p0_lshoist_unpinned/` traces at
377c8360, Sep 7 is HEAD):

| k=13 term | Sep 3 | Sep 4 | Sep 7 |
|---|---|---|---|
| newton | 23.72 | 16.60 | 17.45 |
| collision | 5.86 | 5.80 | **3.43** |
| ldl_pair | 3.15 | 3.08 | **1.48** |
| crba | 3.00 | 2.92 | 2.92 |
| rne | 0.56 | 0.55 | 0.56 |
| wall | 37.53 | 30.30 | **27.20** |

k=0 wall 5.80 → 5.64 → **3.36**. Two wins, both inherited from the CPU
campaign's shared code, both visible at the kernel level rather than as a
statistical claim: the collision kernel per launch 552 → 266 µs at k=0
and 725 → 428 µs at k=13 (§13.19–13.22's sweep, the plane loop and the
body-pair filter), and the `compute_m_inv` launch — second of the three
LDL launches in the Sep 4 trace, 195 µs at k=13 — absent from the Sep 7
trace (§13.13/13.14's M⁻¹ skip, which was "inherited untested" until
today). The LDL pair halved because of it, so the block ledger's F1 on
`ldl_solve` now targets a 112 µs kernel.

Newton did NOT move: 3–5% slower per launch than Sep 4 at k≥6 (2075 →
2181 µs at k=13), inside the box's recorded noise band, and 62% slower at
k=3 (205 → 332 µs) on a row that also carried a +3.5 ms host residual and a
wall time nearly equal to k=6's — a perturbed process, not the kernel. Not
chased: it is the term the block ledger already sends to a redesign
(`THREADS = MAX_CONTACTS = 16`), and rebuilding a Sep 4 arm on a rented
box to settle 3% is the wrong trade. `scripts/p0_ab.sh` exists for the
day it is worth it.

All three sweeps printed "NOT DECIDABLE" at k=9..13: nsys averages over
every launch, warmup included, and with 200 warm + 300 timed the ramping
warmup was 40% of the launches, 2% above the wall clock. `TIMED_STEPS` is
1500 since 99357b9d (warmup share 12%); the next sweep is the clean
baseline and every sweep after it compares to that one.

**The tendon gate.** The blocked kernel compiles for walker2d, Ant and
Humanoid on CUDA (gates 1–2), so the increment is the `NTENDON > 0` block
— the spatial tendon length/Jacobian builders inlined into the cooperative
kernel — through the NVPTX backend. Last known to compile on NVIDIA around
2026-08-21; 35 commits touched the kernel since (§13.24–13.37 plus the
block-diagonal campaign). The discriminating arm is a capped build of the
Aug 27 tree's own copy (`3f3a3763`, the kernel before any of it):

    git worktree add /workspace/mojo-rl-aug27 3f3a3763
    time timeout 600 pixi run -e nvidia mojo build -I /workspace/mojo-rl-aug27 \
        -o /tmp/tbt_aug27 /workspace/mojo-rl-aug27/tests/physics3d/test_newton_blocked_tendon_fields.mojo

Exit 124 there = predates the campaign; a build = bisect the 35 at ten
minutes per arm. Not on the parked-slot path (no tendons), so it gates
nothing measured above; it gates the first tendon model anyone trains on
GPU. Rented-box discipline learned today: a blocked-kernel test is ~15
min of compile there (`test_noslip_blocked_kernel` 924 s), so run only the
NVIDIA-only gates on the box and everything else on Apple.

**Addendum, later the same day — the 1500-step sweep, and why its Newton
column is not a regression.** The rebuilt probe (`TIMED_STEPS = 1500`) came
back decidable (residual +0.2..+4.1 ms, all positive) and with Newton at
**2.8× the morning's cost per launch** on the SAME kernel hash at every k:

| k | newton µs/launch, 500-step run | 1700-step run | collision |
|---|---|---|---|
| 3 | 332 | 543 | 286 → 296 |
| 6 | 524 | 1468 | 388 → 260 |
| 9 | 1234 | 3077 | 414 → 287 |
| 13 | 2181 | 6095 | 428 → 305 |

Same code, same box; the only change is the trajectory length. The probe
drives nothing (action buffer at zero, never resets), so under position
control the arms drift toward the zero pose and settle into resting
contact with the table and each other. Newton's cost is set by constraint
rows and iterations, so it climbs; the collision kernel's is set by the
geometry-pair count, so it does not (it even fell) — the same reason it was
a useless control in the bisect note above. The Sep 4 1000-step k=3 trace
shows the shape (`scripts/p0_drift.py`): flat at ~178 µs to step 700, then
10× spikes. The 500-step sweeps sat entirely in the flat regime by
accident; the 1700-step one averaged the other regime. A per-launch
average over a trajectory that changes character is not a property of the
kernel, and the positive residual is the same fact from the other side —
the warmup steps were CHEAPER than the timed ones this time.

Redefined: the probe resets every `EPISODE_STEPS = 300`, the task's own
`MAX_STEPS`, so the 1500 timed steps cover five whole zero-action episodes
and every episode phase equally. That is the quantity training pays,
reset kernels included. Header prints `episode_steps`; the pre-existing
`RESET_EVERY_STEP` still overrides it for bisects. Consequences: (1) none
of today's three sweeps is the baseline — the next one is; (2) run
`p0_drift.py` on each sweep's k=13 trace before reading its table, a flat
profile is what makes the average a number; (3) the "1500 fixes the
residual" claim above was half right — it fixed the cold-clock bias and
exposed the drift the 500-step runs had been hiding. Same shape as
`_the_sweep_was_not_the_distribution`.

The k=13 trace of that sweep, binned (`p0_drift.py`, 170 steps per span):

    steps     0- 170   2367 µs/launch  (max 5341: the cold first steps)
    steps   170- 680   2035            <- the regime every 500-step sweep measured
    steps   680-1020   5123, 3112      (max 21,156 / 23,750: single launches 10× the plateau)
    steps  1020-1700  11,000 ±30       <- a second plateau, 5.4× the first, dead flat

Two regimes and a transition, not a climb. The second plateau is the arms
at rest on the table with joint limits and contacts all active; its 11 ms
per launch is 88 ms per step at 1024 lanes — a real number for the kernel
at that state, and the state an untrained policy that drives the arm into
the table will visit. The spikes in the transition are single solves at
10× the plateau, which is what an iteration cap being hit looks like; not
chased today. The per-episode probe measures the first regime plus the
reset; whoever trains a policy that lives in the second should know its
Newton costs 5× and that the block ledger's `THREADS = 16` item is what
addresses it.

**THE BASELINE (2026-09-07, RTX 5090, per-episode probe, commit a0eadfee).**
Residual +0.13..+0.22 ms at every k — decidable for the first time; the
k=13 drift profile reads 0.97 last/first with a sawtooth in the max column
(5.35 ms single launches in the first steps after each reset, 2.45 ms
otherwise) and no trend. Wall ms/step at 1024 lanes, and env-steps/s:

| k | nv | ms/step | env-steps/s | newton | collision | crba | ldl_pair | Je |
|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 3.39 | 302,000 | 0.29 | 2.15 | 0.06 | 0.06 | shared |
| 3 | 24 | 5.40 | 190,000 | 1.63 | 2.22 | 0.15 | 0.28 | shared |
| 6 | 42 | 10.34 | 99,000 | 4.45 | 3.16 | 0.57 | 0.59 | shared |
| 9 | 60 | 17.88 | 57,000 | 10.47 | 3.32 | 1.15 | 0.89 | shared |
| 12 | 78 | 26.07 | 39,000 | 16.03 | 3.41 | 2.56 | 1.38 | spilled |
| 13 | 84 | 28.61 | 35,800 | 17.94 | 3.44 | 2.97 | 1.52 | spilled |

Against the block ledger's closing table (0.0.11, 2026-09-03, no-reset
500-step probe): k=13 43.9 → 28.6 ms, k=9 27.4 → 17.9, k=0 5.76 → 3.39.
The two are not the same workload (this one pays five resets and the
cold first steps of each episode; that one measured steps 200–500 of one
drift), so read the ratio as "the training-shaped cost fell by about a
third", not as a kernel A/B. Within THIS workload the shares at k=13:
newton 63%, collision 12%, crba 10%, ldl_pair 5%; `d/dnv² = 0.0029` on
newton. This morning's k=3 row (9.78 ms) is confirmed as a perturbed
process: 5.40 here, newton 204 µs/launch against Sep 4's 205.

Every later sweep compares to this table, at the same probe. What remains
is the block ledger's own list: the Newton block at `THREADS =
MAX_CONTACTS = 16` (63% of the step), CRBA's dense `[BATCH, NV*NV]` write
(10%), `ldl_solve` (the 117 µs kernel), and the second-plateau Newton
(11 ms/launch when the arms rest on the table), which is what the
iteration cap governs.

**BASELINE 2 (2026-09-07 evening, commit e9918f5e — `Je` spilled above 16 KB
of threadgroup memory).** Same probe, same box, residual +0.13..+0.22 at
every k. The first GPU change of the campaign that ships; the block ledger
(experiments 1–3) has the mechanism — the blocked kernel is thread-0
latency-bound and needs co-resident blocks, and the Jacobian rows were
buying them out.

| k | ms/step, baseline 1 | baseline 2 | newton µs/launch | env-steps/s |
|---|---|---|---|---|
| 0 | 3.39 | 3.39 | 36 | 302,000 |
| 3 | 5.40 | 5.10 | 204 → 158 | 201,000 |
| 6 | 10.34 | 10.03 | 556 → 508 | 102,000 |
| 9 | 17.88 | 17.70 | 1308 → 1283 | 57,900 |
| 12 | 26.07 | 26.08 | 2006 | 39,300 |
| 13 | 28.61 | 28.62 | 2244 | 35,800 |

k=12/13 already spilled, k=0 never does. Every later sweep compares to
this table. Next on the same mechanism: one dense array instead of three
in the blocked kernel (block ledger, stage 1), expected 1.1–1.6× on
Newton at k≥9 and measured, not predicted, when it lands.

### 13.39 LANDED (2026-09-07): the SO-101 configs step under Euler, as MuJoCo does

`So101ParkProbeConfig` and `So101TabletopConfig` inherited `INTEGRATOR =
"rk4"` from the trait's default; neither the Menagerie SO-101 model nor the
generated park scenes set `<option integrator>`, so MuJoCo steps them under
Euler. RK4 at frame skip 2 ran the whole pipeline 8 times per env step — 8
collision launches, 8 Newton solves, 8 CRBA passes — against Euler's 2.
Both configs now say `"euler"` and `INTEGRATOR_WS_EXTRA = 0`.

Fidelity, before the switch was made: the studio path under Euler agrees
with MuJoCo to **4.2e-17** over 50 steps on the k=0 and k=13 park scenes
(`trace.py`, the board's protocol: seed 2024, random ctrl); the
compile-time CPU engine under Euler at 300 physics steps, zero control,
matches MuJoCo on every arm dof to ~1e-12 and on the parked free body
(free fall from 50 m) to the same order — see the line pair in the
session log. Gates on Apple: `test_tape_gpu_parity`,
`test_device_placement`, `test_goal_distance`, `test_active_mask`.

⚠ Every parked-slot number above this section was RK4: 8 launches per
step. The next sweep is a new baseline (baseline 3) with 2 launches per
step and Euler's own extra term — the `M_hat = M + dt·diag(damping)`
re-factorisation (`euler.mojo:403`) that the RK4 probe never saw.
`scripts/p0_attrib.py` derives launches per step from the instance count;
`p0_drift.py` now does too (it assumed 8).

**BASELINE 3 (2026-09-07, commit 3c3b480f — Euler, 2 launches per step).**
Same probe, same box, residual +0.05..+0.07 ms at every k, k=13 drift 0.96.

| k | ms/step, baseline 2 | baseline 3 | ratio | env-steps/s | newton | euler finalize | collision | crba |
|---|---|---|---|---|---|---|---|---|
| 0 | 3.39 | 0.93 | 3.6× | 1,100,000 | 0.07 | 0.03 | 0.54 | 0.02 |
| 3 | 5.10 | 1.56 | 3.3× | 656,000 | 0.32 | 0.22 | 0.57 | 0.04 |
| 6 | 10.03 | 3.13 | 3.2× | 327,000 | 1.02 | 0.55 | 0.79 | 0.15 |
| 9 | 17.70 | 5.54 | 3.2× | 185,000 | 2.58 | 1.02 | 0.84 | 0.30 |
| 12 | 26.08 | 8.18 | 3.2× | 125,000 | 4.02 | 1.54 | 0.85 | 0.65 |
| 13 | 28.62 | 9.32 | 3.1× | 110,000 | 4.48 | 2.06 | 0.86 | 0.75 |

Every kernel's per-launch cost is what it was (Newton 2240 µs at k=13,
collision 431); the step is 2 launches instead of 8. The shortfall from 4×
is ONE new term: the Euler integrator's fused finalize kernel
(`integrator…`, 1028 µs per launch at k=13, 22% of the step, second to
Newton, `d/dnv^2` the highest after Newton's — it grows like nv³). That is
the `M_hat = M + dt·diag(damping)` re-factorisation (`euler.mojo:403`) the
RK4 probe never ran, a DENSE LDL where the LDL pair's kernels are
block-restricted and cost 115 + 73 µs on the same matrix. Next item, and
the cheapest on the table: the block-restricted factor on `M_hat` (the
tree table is already in the kernel's reach), ~1.0 → ~0.15 ms per launch,
k=13 9.3 → ~7.6 ms/step. After it the shares at k=13 are Newton 48%,
collision 9%, CRBA 8%.

The real SO-101 tasks live near k=0..3: 0.93–1.56 ms per step, 0.66–1.1 M
env-steps/s at 1024 lanes, where collision is 36–58% of the step and the
warp-cooperative GJK (block ledger §6) is the remaining lever.

**The Euler finalize, split (0a4bf917 + the gate fix after it).** The
single-launch kernel ran the whole finalize on one thread per env — dense
nv² matvec, damping diagonal, `_ldl_factor_env`, `_ldl_solve_env` — 1028 µs
per launch at k=13. It is now a block-per-env rhs kernel (row-parallel
matvec, then the damping diagonal over joints), the step's own
cooperative `ldl_factor` and `ldl_solve` on `scratch.M`/`fnet`/`qacc_ws`,
and a per-env integrate kernel. Same arithmetic as the old kernel (its
GPU-vs-CPU rounding signature reproduced to the digit: 11 values at 2⁻²⁰
on the pendulum, the CPU's tree-ordered leg against the dense one, which
predates all of this). Gate: `test_euler_finalize_gpu_parity` (damped
model, 1e-5, the halved-damping mutant reads 0.32). ⚠ Two green gates had
said nothing — `test_tape_gpu_parity` never steps, `test_ip_fields_env_loop`
compares at 1e-2 — and the first version of the new gate demanded
bit-exactness across two legs that never had it and was committed on the
mutant's verdict alone. Priced on the box next: expect ~1.0 → ~0.2 ms per
launch at k=13, the step 9.3 → ~7.7.

**MEASURED (RTX 5090): the finalize split, k=6 3.13 → 2.77 ms/step (1.13×),
k=13 9.32 → 7.71 (1.21×).** The 277 / 1028 µs launch is gone; the Euler
step now carries two LDL pairs per step (the unconstrained solve's and
`M_hat`'s: 4 launches of 107 + 70 µs at k=13), a 40 µs rhs kernel and the
integrate. Residual +0.06 at both k. Shares at k=13: Newton 58%, collision
11%, CRBA 10%, LDL pair 9%. The step is 3.7× baseline 2 at k=13 and
3.6× at k=0; 133k env-steps/s at k=13, 1.1 M at k=0, 1024 lanes.

Standing levers, in the order the shares set: the Newton kernel's dense
triple (block ledger stage 1, 1.1–1.6× on Newton), `ldl_solve` (F1: the
107 µs kernel, block-restricted like its siblings, ~10×), CRBA's dense
write (F2), and at the small k the real tasks run at, collision — the
warp-cooperative GJK (block ledger §6).

### 13.40 LANDED (2026-09-07): Newton stage 1 — one dense array in threadgroup memory, not three

The blocked Newton kernel kept three `NV*NV` arrays per block: a copy of
`M`, the Hessian `H`, and its factor `L`. At k=13 (nv=84) that is 84,672 B
of the block's ~94 KB, and it is what held the kernel at one block per SM
on the RTX 5090 (§13.38: occupancy is the elastic term — a 44 KB pad cost
2.45×/1.6× at k=3/6, spilling `Je` bought 1.30/1.09/1.02×). Stage 1 of the
block ledger removes two of the three:

- **`M` is not copied.** The two setup matvecs (`M*qacc_smooth`, the
  warmstart trial's `M*qacc_w`) were `NV²` serial loops on thread 0 over
  the copy; they are one cooperative block-restricted matvec each
  (`_block_matvec_coop`, a row per thread, the same ascending inner sum, so
  the same bits), reading `M` from global memory. The Hessian build and the
  loop's `M*search` read global `M` too.
- **`H` is factored in place.** The build writes `L_sh`; `_chol_factor_coop`
  reads `H[i,j]` at the slot it then writes `L[i,j]` into, and every `L[.,k]`
  it reads was finished when column k ran. The zeroing pass went with it
  (nothing reads outside a block or above the diagonal — audited: the factor's
  restricted k loops and `chol_solve_seg_p`'s lower-triangle reads). The
  rank-deficient retry moved to the caller: rebuild `H` with 1e-6 on the
  diagonal (the same bits the helper used to add) and factor again.

Footprint (`newton_shared_elems`, pinned in `test_newton_shared_budget`,
22/22): k=6 48,876 → 34,764 B, k=13 ~94 KB → ~37 KB. The ceiling the three
arrays set moved: k=14 (nv=90) was 74 KB over `0x18C00` and now fits at
41,920 B; the reach is ~k=24 before `L_sh` alone binds. Gates on Apple:
`test_newton_blocked_fields` (golden fingerprint 5707.35403907299, bit-exact),
`test_newton_freejoint_vs_cpu` (ThreeTrees oracle), `test_noslip_blocked_kernel`
4/4, `test_newton_blocked_tendon_fields` 2/2, `test_fields_mt_parity`
BIT-EXACT over two steps. Not priced yet: the box sweep and
`p0_kernel_shape.py` say what the occupancy bought; the block ledger's
estimate is 1.1–1.6× on the Newton launch, more at high k.

⚠ THE DEFECT IN THE MAKING, recorded because it cost two hours and the
analysis could not find it. The edit that removed the `M` copy matched the
`if valid_env:` block around it — and that block also held the cooperative
load of the contact edges (`Je_sh`/`De_sh`/`bias_e_sh`) and the `barrier()`
that publishes them. The golden read −815.75, the smooth acceleration:
thread 0 built every row from unloaded edge arrays, unsynchronised. Every
static suspect (aliasing through the solve, the retry's scope, the global-M
indexing) was exonerated by reading, correctly; what named the race was a
discriminating run that should have changed nothing — giving `H` its own
array again moved the fingerprint, and two runs of that one configuration
differed by 2e-3. A number that moves between identical runs is a race,
and a race after a deletion is a missing barrier: diff the barrier list
before and after any block removed by pattern. `NEWTON_STAGE1_CHECK` (a
knob, off) now recomputes all four moved pieces on thread 0 and poisons
`qacc[0]` with a magnitude per failing check — one run names the cut;
positive control 1.1e21.

**MEASURED (RTX 5090, 2026-09-08): the Newton launch 1.47× at k=6, 2.3× at
k=9, 2.2× at k=12, 1.81× at k=13; the step k=6 2.77 → 2.46 ms (1.13×),
k=13 7.71 → 5.72 (1.35×).** Per launch: k=0 33 µs, k=3 138, k=6 347, k=9
562, k=12 905, k=13 1235 (was 2240). k=0 and k=3 do not move (35/160 µs
before): at nv ≤ 24 the block never was shared-memory bound, so removing
two arrays buys nothing there — the gain is the occupancy the ledger
predicted, and it lands above the 1.1–1.6× estimate from k=9 on. Residual
+0.07 ms at every k (the divisor is right). Whole sweep, ms/step: 0.92 /
1.39 / 2.46 / 3.36 / 4.86 / 5.72 at k = 0/3/6/9/12/13; 1.11 M env-steps/s
at k=0, 179 k at k=13, 1024 lanes; k=13 is 5.0× baseline 2. Shares at
k=13: Newton 44%, collision 15%, CRBA 13%, LDL pair 12.5%. Newton's
`d/dnv²` is 0.00040, still the highest, three times CRBA's: stage 2 (the
thread-0 serial setup and line search, §13.38's latency term) is what is
left in it. Note for the shape script: the trace truncates the kernel name
to `solver_newt…`, so the filter is `newt`, not `newton`.
Kernel shape at k=13 (`p0_kernel_shape.py … newt`): 1024 blocks × 64
threads, **218 registers (was 255), 37.9 KB shared (was ~94)**, blocks per
SM 3 by shared memory (was 1) and 4 by registers — the register bound is
the binding one now, so the next block per SM is a register question, not
a memory one.

### 13.41 LANDED, UNPRICED (2026-09-08): Newton stage 2 — the Hessian factor on one thread per block

The block ledger's own "stage 2" (pack `L_sh` by block) tops out at the
register ceiling: after stage 1 the kernel sits at 3 blocks/SM by shared
memory and 4 by registers, so packing buys at most 4/3 and stage 3 (registers)
would have to come first. The larger term is the block's own latency: a k=13
block-solve takes ~600 µs where one CPU core does it in 4.6 µs, with the
line search at two evaluations (block ledger §5) and the loop at 61% of the
kernel with its internal split unmeasured.

Structurally the loop's cooperative Cholesky (`_chol_factor_coop`) walks
every column of the whole matrix with two `barrier()`s each — 168 barriers
per Newton iteration at nv=84 — to factor fourteen INDEPENDENT 6×6 blocks.
The solve already went to one thread per block (F3b). The factor now does
the same: `chol_factor_seg_p` (cholesky.mojo) factors a diagonal block in
place on one thread, no barrier inside, the same walk and thread assignment
as the solve, whenever every block is at most `NEWTON_FACTOR_PER_BLOCK_MAX_BN
= 12` dofs (decided per solve from the segment table, block-uniform). Wider
blocks keep the cooperative walk for the whole matrix — a 60-dof block on one
thread is the serial floor the walk exists to avoid. Bit-identical by
construction: the same ascending sums per entry, only the production order
changes. Gates on Apple: golden (walker2d, one 9-dof block, per-block path;
a mutant on the inner sum reads fingerprint 0.0), ThreeTrees oracle (three
6-dof blocks), tendon 2/2, noslip dog 4/4 (coop path, nv=79), mt parity
bit-exact. Barriers per iteration at k=13: ~175 → ~7. What it buys is the
box's to say — the record on predicting this kernel's terms is four
over-predictions running; the A/B is stage 1 vs stage 2, interleaved.

**MEASURED (RTX 5090, 2026-09-08, `p0_ab.sh` stage 1 vs stage 2 at
`MAX_BN = 12`, three interleaved rounds, MIN): Newton 1.054× SLOWER at k=6
(347 → 366 µs per launch) and 1.060× at k=13 (1231 → 1305), behind in
every round; every other kernel 1.000 (the A/B is clean). Off by default
(`NEWTON_FACTOR_PER_BLOCK_MAX_BN = 0`, kept as a pricing knob).** What the
negative says is more useful than the change would have been: ~168
barriers per iteration removed and the kernel got slower, so the
cooperative factor was never a term of the loop. Written down, the
arithmetic agrees — a 6×6 factor on one thread is ~2k cycles, the
84-column walk with its barriers ~25k, both under 10 µs against a ~600 µs
block-solve. The loop's 61% (pinned STOP bisect, block ledger) is therefore
in its THREAD-0 passes — gradient, read-back, line search, update — on
per-thread `Scratch` locals in local memory, or the split has moved since
that bisect (before the `Je` spill and stage 1). The block ledger's
prediction record on this kernel is now five for five; the next move is a
measurement of the current split, not a change.

### 13.42 MEASURED (2026-09-08, Apple): the blocked kernel's serial split, priced at home — one small win, one loss, and what the probe cannot tell you

With stage 2's negative in hand and the box priced per build, the split
moved to the laptop: `NEWTON_FORCE_BLOCKED` (new, measurement only) routes
Metal to the blocked kernel for `nv ≤ 60` (its threadgroup footprint fits
Metal's 32 KB there), and the park probe at k=9 (600 timed steps, ~4 min a
build, ~35 s a run) becomes one arm per `NEWTON_SERIAL_PROBE` term. Metal's
latencies are not CUDA's, so the numbers below are a SPLIT, not a cost.
Two new read-only terms: 12 = the warmstart read, 14 = the joint-limit
scan's loads; 11 (the segment build) was measured once and then retired
when the build went cooperative. Arm 9 (the line-search evaluator, an
always-inlined closure) blows the compiler past 16 GB of host memory and
was dropped — the line search is two evaluations here anyway (block ledger
§5).

⚠ Apple wall time drifts 10% between eras (baselines 58.0 / 52.7 / 58.2
ms/step over the afternoon), so each arm reads against the baseline of its
own era, and anything under ~1 ms/step is noise. Marginal cost of one
instance = (arm − baseline)/9, ms per step, k=9, nv=60, 2 launches/step:

| term | what | ms/step |
|---|---|---|
| 2 | the per-block Cholesky solve (`chol_solve_seg_p`, one thread per block) | **5.05** |
| 11 | `build_dof_segments` on thread 0 (`num_edges·nv` reads of spilled `Je`) | **5.0–5.6** |
| 10 | the H build (cooperative) | 1.5–2.1 |
| 3 | the coop factor's `d_j` reduction (thread 0) | 1.58 |
| 14 | the joint-limit scan's loads (thread 0) | 0.4–1.0 |
| 1 | the gradient loop | 0.44 |
| 5 | workspace init + edge zeroing | 0.45 |
| 6 | the contact precompute | 0.12 |
| 4, 12 | the read-back loop, the warmstart read | ~0 |

Two changes followed, both bit-exact by construction, gated on Apple (golden
fingerprint, ThreeTrees oracle, tendon 2/2, noslip 4/4, mt parity, the
segment and Cholesky unit tests 22/22 + 7/7 + 9/9) and priced by
interleaving three binaries at k=9:

- **The solve without its private intermediate** (`chol_solve_seg_p`, the
  scalar path): the forward pass writes `x` directly and the back
  substitution runs in place. The `y` it drops was a per-thread array with
  dynamic indexing — local memory on CUDA, device memory on Apple. **Kept:
  ~1.2 ms/step faster on Apple** (old 57.4 / 52.6, new 58.8 / 54.2 against
  a segment-only arm at 60.4 / 55.1 in the same rounds).
- **The segment build's edge scan on one thread per edge**
  (`newton_blocks.mojo` split into phases A/B/C, the kernel running B
  cooperatively around two barriers): **~2.7 ms/step SLOWER on Apple**,
  three rounds of three. Behind `NEWTON_SEG_BUILD_COOP = False`.

⚠ WHAT THE PROBE CANNOT TELL YOU, and this is the finding. The serial probe
priced the segment build at ~5 ms/step and the solve at ~5; taking the
first off thread 0 cost 2.7 and the second's memory traffic bought 1.2. A
term's REPEAT cost is the work its copies add, and that is paid in full; the
term's own cost in context is mostly hidden by the other blocks resident on
the core, so removing it saves little and re-mapping it across threads with
barriers can cost more than it hides. This is the third structural change to
this kernel in two days and the pattern is now clear: stage 1 won by
occupancy (memory footprint), the in-place solve won by memory traffic, and
both re-mappings of a serial chain across threads lost (the per-block factor
on CUDA, the cooperative scan on Apple). Price a GPU change by an A/B of the
change; a repeat probe ranks terms by work, not by what their removal saves.

Unpriced on CUDA: the in-place solve (the box A/B, stage 1 vs HEAD). What
is left in the kernel by mechanism, not by probe: registers (218/thread,
the 4-block ceiling — stage 3), and the memory traffic of the thread-0
setup (the joint scan's global loads, the per-thread `Scratch` locals).

**MEASURED (RTX 5090, 2026-09-08, `p0_ab.sh` stage 1 vs the in-place solve,
three interleaved rounds, MIN): Newton 1.098× SLOWER at k=6 (346 → 380 µs
per launch) and 1.112× at k=13 (1234 → 1372), behind in every round; every
other kernel 0.97–1.00. REVERTED (7b6c126a's solve change; the phase split
and the knobs stay).** Apple had the same binary pair 1.2 ms/step the other
way. So the laptop proxy does not carry the SIGN for this kernel, not only
the size: on CUDA the forward pass's store into shared `x` and the inner
loop's reads of it back cost a shared round trip per row that the
local-memory `y` did not, and Apple's device-memory private arrays made the
opposite trade. Two lessons on top of §13.42's: (1) Apple prices Apple;
the blocked kernel is a CUDA kernel and only the box's A/B decides; (2) the
ledger's count on this kernel is now three losses in three structural
changes after stage 1 — per-block factor, cooperative scan, in-place solve
— all bit-exact, all gated, all measured, none shipped. The kernel as it
stands after stage 1 (d3a465b5) is the one to beat, and by mechanism the
candidates left are registers (stage 3) and the setup's memory traffic,
each priced on the box before anything is built on it.

### 13.43 MEASURED (2026-09-08): the occupancy lever priced at the operating point — and Newton closed for now

`NEWTON_SHARED_PAD = 2600` (10.4 KB of fake threadgroup memory, bit-identical
arithmetic) against the stage-1 binary, `p0_ab.sh`, three interleaved rounds,
MIN: **k=13 3 → 2 blocks/SM, Newton 1.122× slower (1231 → 1382 µs); k=9
~5 → 3, 1.282× (562 → 721)**; every other kernel 1.00. So occupancy is still
elastic at k=13, but the slope is flattening — 1 → 3 blocks bought 1.8× (stage
1), 3 → 2 costs 1.12×. The block ledger's pack (`L_sh` by diagonal block,
~38 → ~14 KB) plus a register cap (218 → 170, the CUDA-only `nvvm.minctasm`
annotation — it crashes the Metal compile, so it needs its own entry point)
would take k=13 from 3 to ~6 blocks/SM: about 1.2–1.3× on Newton, i.e.
~10% of the step at k=13, less at k=9, nothing at k ≤ 6 where the kernel is
already unbound and the SO-101 tasks live. It needs a comptime cap on the
packed size, a fallback for envs whose coupled blocks overflow it, block-local
indexing at every factor and solve site, and the second entry point.

**Newton is closed at stage 1 (d3a465b5).** The week's ledger on this kernel:
one win by occupancy (1.8× at k=13), three bit-exact re-mappings of serial
chains measured and not shipped (§13.41–13.42), the remaining occupancy priced
at ~10% of the step at the wide end only. Next by share at k=13 — collision
15%, CRBA 13%, the LDL pair 12.5% — and by the mechanisms that won (memory
footprint, memory traffic): `ldl_solve` (block ledger F1) and CRBA's dense
write (F2).

### 13.44 LANDED, UNPRICED (2026-09-08): the LDL solve kernel gets a block per env — F1's launch shape

`ldl_solve`'s GPU kernel ran one THREAD per env: 1024 envs are 16 blocks of
64 on a 170-SM part, each thread a dependent chain of ~600 global loads
(the block-restricted forward/backward substitution over 14 trees at k=13)
with nothing resident to hide it — 106.6 µs per launch, four launches a
step under Euler (the unconstrained solve's and `M_hat`'s), 7.5% of the
k=13 step and the largest of the LDL kernels. Its sibling `ldl_factor`
already had a block-per-env cooperative kernel (`PARALLEL`), 70.5 µs on
the same matrix.

`_ldl_solve_fields_mt_kernel`: a block per env, one warp, ONE THREAD PER
KINEMATIC TREE — the tree blocks are independent systems, so a thread runs
the serial body restricted to its own `[b0, b1)`: the same loops, the same
per-row accumulation order, the same bits (a column-cooperative form would
reverse the back substitution's order per row). `ldl_solve[PARALLEL=True]`
selects it; the Euler and RK4 integrators pass `PARALLEL_GPU` to it as they
do to the factor. Gates on Apple: `test_fields_mt_parity` arms A (walker2d)
and D (ThreeTrees, the multi-tree one) extended with the solve, serial vs
PARALLEL bit-exact on a non-zero right-hand side (a mutant skipping each
block's first row reads 1.088 vs 0.0 at i=0); arm C (RK4 with every
cooperative kernel vs every serial one, 3 steps with contacts) bit-exact;
golden fingerprint; `test_euler_finalize_gpu_parity`; `test_ldl_blocked`
9/9; `test_task_reset_steps`. Mechanism, not probe: the kernel's cost is a
tiny grid on a latency chain (the same shape the ledger's
`_row_per_thread_kernels_are_uncoalesced_and_tiny_grid` names), and the
block-per-env factor is the measured precedent. The box A/B prices it.

### 13.45 MEASURED (2026-09-08): a switched-off knob had moved the Newton kernel 1.11× — and three verdicts with it

The F1 A/B (stage-1 binary vs e913a48b) read the LDL pair at 0.75 as
expected, and **Newton at 1.108× (k=6) / 1.115× (k=13) slower with a
changed kernel hash — in a build whose Newton knobs were all OFF.** Every
A/B since stage 1 was built from HEAD, so the same shift sits inside three
verdicts already recorded:

| A/B (HEAD of the day vs stage 1) | Newton read | of which this shift | the change's own effect |
|---|---|---|---|
| stage 2, per-block factor ON (33a6e774) | 1.054 / 1.060 | ≤ that | ≤ 1.05, not separable |
| in-place solve (7b6c126a) | 1.098 / 1.112 | ~1.11 | **~neutral**, not 1.10× slower |
| pad, k=13 (f2aa8b79) | 1.122 | ~1.11 | **~1.01**: 3 → 2 blocks/SM costs ~nothing at k=13 |
| pad, k=9 | 1.282 | unmeasured at k=9 | ≈ 1.15 if the shift is the k=6 one |

So §13.42's in-place solve was not a loss, §13.43's occupancy elasticity at
k=13 is ~1% rather than 12% (which makes the pack worth even less than
recorded — Newton stays closed), and the k=9 elasticity stands at roughly
1.15. What moved the kernel: with the knobs at their production values the
body still carried the stage-2 decision variable and its dead branches, an
extra flag store before the build barrier, and a segment build whose one
function had become three inlined phases. Which of the three, the box can
say for ~45 minutes; it is not worth it — none of them earns its place. All
three are REMOVED (commit below): the kernel's body is stage 1's again,
the file differing from d3a465b5 only by the Metal routing knob (outside
the kernel) and two `comptime`-elided probe terms. `chol_factor_seg_p` and
the phase split of `build_dof_segments_p` are gone with them; the numbers
stay in §13.41–13.43.

**F1 itself: 106.8 → ~62 µs per launch at k=13, not the ~10 the arithmetic
said.** The first kernel walked `_dof_block` at every tree boundary, and
that helper scans the table from the top — `ntree²` global loads on every
thread before its own 42. The table is now read by index: a thread loads
the two entries of the trees it owns. Priced with the Newton check in the
same A/B.

⚠ THE RULE THIS ADDS. An A/B arm is the WHOLE build, and a knob at its off
value is not a no-op for a GPU kernel's compiled body. (1) Read every
kernel's ratio in an A/B, not the one the change targets — Newton's 1.11
was in the table three times before it was read. (2) A "different kernel
hash" flag on a kernel the change did not touch is the alarm, not a
footnote. (3) Re-baseline after each landed change, and A/B a knob
experiment against the SAME tree with the knob off, not against the last
landed binary.

**MEASURED (RTX 5090, 2026-09-08, `p0_ab.sh` stage 1 vs fab85f0a, three
interleaved rounds, MIN): the LDL pair 0.498× at k=6 (0.287 → 0.143
ms/step) and 0.522× at k=13 (0.710 → 0.371); the solve kernel 44.6 → ~8.6
µs per launch at k=6 and 106.9 → ~22 at k=13 (the factor, 27.2 / 70.5, is
the pair's larger kernel now); the step 2.455 → 2.306 (0.939) and 5.714 →
5.385 (0.942). Newton 1.001 / 1.004 — the §13.45 shift is gone, which is
the check this A/B existed for. Every other kernel 0.97–1.03.** k=13 is now
5.3× baseline 2; 190 k env-steps/s at k=13, 444 k at k=6, 1024 lanes.
fab85f0a is BASELINE 4: the full sweep of it (`p0_attrib.sh`, all six k)
is the table every later A/B's numbers are read against. Shares at k=13
from this A/B: Newton 46%, collision 16%, CRBA 14%, LDL pair 7%.

### 13.46 LANDED, 18.5× on the kernel at k=13 (2026-09-08): the CRBA kernel builds its dof topology once per block, cooperatively

`_mass_matrix_treewalk_fields_mt_kernel` (block per env, `NV` threads, the
SO-101 path's CRBA at 376 µs per launch and 14% of the k=13 step) opened
with a per-thread rebuild of the model's dof topology — dof → body, dof →
parent dof, body → first/last dof — from the joint and body tables: a loop
over every joint with three global loads each (97 joints at k=13) and a
parent walk per dof, on EVERY thread of every env, every step, before any
arithmetic. The block ledger's F2 named the dense zeroing of `M` as the
kernel's cost; the zeroing is 84 coalesced stores per thread and cannot be
376 µs, while this preamble is a ~500-step chain of dependent global loads
in a kernel that runs a quarter of a wave (1024 blocks against ~24 per SM
on 170 SMs) and therefore pays its per-block latency in full — the LDL
solve's shape, which just paid 5× (§13.44).

The tables are model constants. They are now built once per block into
threadgroup memory (`topo`, 2·NV + 2·NBODY int32) by the same rule split
by what can be written without a race: a thread per JOINT writes the dofs
it owns; a thread per BODY scans that map for its first and last dof (the
serial min/max over its joints, read off the map); a thread per DOF walks
its parent; three barriers. The CPU leg runs the same body at one thread,
so its tables — and `M` — are what they were. Gates on Apple: the
treewalk goldens (walker2d, Ant: bit-match; a parent-walk mutant reads
9181.8 vs 5823.4), the parity test's CRBA arms A and D (serial vs
PARALLEL bit-exact, ThreeTrees included) and arm C, the golden
fingerprint, the ThreeTrees oracle. What remains serial in the kernel:
the thread-0 backward accumulation of the body composites (97 × 10
shared read-modify-writes) — the next cut if this one moves the number.
The box A/B prices it.

**Priced (5090, `p0_ab.sh`, ROUNDS=3, MIN over rounds, B = c98948ac vs
A = fab85f0a = baseline 4):**

| k | term | A ms/step | B ms/step | B/A | largest kernel A → B µs |
|---|------|-----------|-----------|-----|-------------------------|
| 6 | crba | 0.154 | 0.018 | **0.116** | 76.9 → 8.9 |
| 6 | wall | 2.309 | 2.065 | 0.894 | |
| 13 | crba | 0.750 | 0.041 | **0.054** | 375.0 → 20.3 |
| 13 | wall | 5.388 | 4.598 | 0.853 | |

The kernel is 18.5× faster at k=13 and 8.6× at k=6: the preamble WAS the
kernel. Newton 0.994 / 0.984, LDL pair 0.990 / 1.008, cdof 0.999 / 0.999,
same kernel hashes throughout. CRBA is now 0.9% of the k=13 step; the
thread-0 composite accumulation named above is not worth a cut.

**Rows that moved without being touched — not booked to the change.**
Three kernels whose code and hash are identical in A and B ran faster in
B, 3/3 rounds: collision at k=6 0.896 (396 → 355 µs; at k=13 it is
0.997), rne at k=13 0.897 (69.4 → 62.3), the warm-start kernel at k=13
0.648 (11.0 → 7.1). At k=6 collision's move is a THIRD of the wall gain
(0.083 of 0.244 ms); with only the CRBA row counted the k=6 wall is
0.941, not 0.894. At k=13 the CRBA row is 0.709 of the 0.790 ms and the
strays sum to 0.05. The three are the kernels that run beside or right
after CRBA in the stream; a plausible mechanism is the box's clock and
cache state after a 375 µs low-occupancy kernel was cut to 20, but that
is a guess and the full baseline-5 sweep (all six k, `p0_attrib.sh`) is
the measurement that settles what those rows cost on their own. Until
then the change is booked as the CRBA row.

**Baseline 5 = c98948ac.** k=13 step 4.598 ms (1024 envs: 223 k
env-steps/s, 6.2× baseline 2's 28.5 ms), k=6 2.065 ms. Shares at k=13:
Newton 53%, collision 18.5%, unlabelled (the batched env kernel, 91 µs)
14%, LDL pair 8%, rne 2.7%, CRBA 0.9%. The levers left are the Newton
(closed at stage 1, §13.43), collision (warp-cooperative GJK, unbuilt),
the env kernel and the LDL factor (70 µs). Before any of those: grep
the rne, cdof and env kernels for the SAME per-thread topology rebuild —
this cut was one grep away for weeks.

### 13.46b Baseline 5, the full sweep (5090, 2026-09-08, `p0_attrib.sh`, 1500 timed steps)

Run on c98948ac (the CRBA row, 20.5 µs and hash `9339336f`, is the tree's
signature; the directory was named `p0_base4` before the CRBA A/B came
back). Per-step ms by term; `unlabelled` is every kernel the labeller
cannot prove a term for (integrator, kinematics, the post-constraint RNE
sensor pair, subtree_com, the env kernel, `parser_mode…`).

| k | nv | wall | GPU | newton | collision | ldl_pair | rne | unlabelled | crba | cdof |
|---|----|------|-----|--------|-----------|----------|-----|------------|------|------|
| 0 | 6 | 0.918 | 0.848 | 0.066 | 0.538 | 0.031 | 0.032 | 0.158 | 0.010 | 0.011 |
| 3 | 24 | 1.311 | 1.233 | 0.277 | 0.561 | 0.077 | 0.044 | 0.247 | 0.012 | 0.011 |
| 6 | 42 | 2.069 | 1.998 | 0.681 | 0.713 | 0.145 | 0.061 | 0.357 | 0.018 | 0.013 |
| 9 | 60 | 2.856 | 2.786 | 1.120 | 0.821 | 0.213 | 0.095 | 0.484 | 0.025 | 0.015 |
| 12 | 78 | 3.883 | 3.809 | 1.814 | 0.843 | 0.328 | 0.119 | 0.632 | 0.037 | 0.017 |
| 13 | 84 | 4.607 | 4.542 | 2.460 | 0.852 | 0.367 | 0.125 | 0.663 | 0.041 | 0.018 |

Residual (wall − GPU) is a flat 0.07 ms at every k: host gaps, not a
divisor. The Newton kernel's span table is flat (last/first 1.00) with a
per-launch max of 1.95 ms against a 1.24 mean — the contact-count tail.

**What the sweep says that the k=13 shares hid.** Collision is nearly
flat in k (269 µs a launch at k=0, 426 at k=13) and is therefore 63% of
the k=0 step and 46% of k=3: it is the step's FIXED cost, and the lever
for every small-k configuration. At k=13 the unlabelled bucket (0.663 ms,
14.6%) is, by kernel: the five integrator kernels 0.232, the two
kinematics kernels 0.135, `parser_mode…` 0.107 (= `apply_actions_kernel_gpu`'s
`apply_kernel` in `parser/model_def_from_xml.mojo`: ctrl → `qfrc`, 53.5 µs
a launch at k=13 for 84 dofs × 1024 envs, 11 µs at k=0 — thread per env,
serial over actuators; a candidate), the sensor RNE pair 0.091,
subtree_com 0.047, the env kernel 0.039. Nothing in it is one kernel
worth a cut on its own; the actuator kernel is the first to look at.

The LDL split reads "no CRBA launch in the trace to anchor on" at every k
although the same report labels the CRBA kernel: a defect of the split
reader (`disambiguate_by_launch_order` anchors on `label(nm) == "crba"`
over the trace's `Name` column), harmless because the factor (69.8 µs at
k=13) and the solve (22.0) are already separate rows above it. To be
looked at with a trace file at hand.

### 13.47 LANDED, 0.97 at k=13 (2026-09-08): the RNE kernel — topology once per block, backward pass level-parallel

The same grep §13.46 ended on. `_rne_fields_mt_kernel` (block per env,
`NV` threads; 62.7 µs a launch at k=13, 2.8% of the step; 16 µs at k=0)
had three pieces of the CRBA kernel's shape:

- every thread rebuilt the body-level table from `NBODY` global parent
  reads before any arithmetic;
- the forward pass called `_rne_fwd_body` with the SCAN form — each body
  read all `NJOINT` rows of the joint table to find its own joints, on
  every level of the level-serial pass (the CPU leg had used
  `body_joint_map` since §13.26; the GPU legs "kept the scan — no
  per-thread table", which was the right call for a per-thread table and
  the wrong one for a per-block table);
- the backward accumulation of `cfrc` ran on thread 0 as `NBODY`
  iterations of six read-modify-writes of GLOBAL memory, each chained
  through the parent's row — a ~100-deep dependent chain in a kernel at
  a quarter of a wave, the LDL solve's shape (§13.44).

Now: one `3·NBODY` int32 table in threadgroup memory — parent, first
joint, joint count (−1 = not one contiguous run, that body scans) —
built by a thread per body (one parent read, one pass over the joint
table) before the first barrier; the level table computed from it; the
forward pass handed each body its `[j_lo, j_hi)`; and the backward pass
level-parallel in GATHER form: a body at level `lvl` adds its children
(all at `lvl+1`, complete after the previous barrier) into its own row
in DECREASING child index, which is the order the serial pass delivered
them to that parent with each child's row final at that moment — same
additions, same order, same rounding. The serial GPU kernel and the CPU
leg are untouched (the forward helper takes the range instead of the
map; the CPU leg passes what its map gave it).

Gates on Apple: `test_fields_mt_parity` — walker2d RNE bias bit-exact,
and a NEW ThreeTrees RNE compare (three bodies hang off the world, two of
them on free joints) bit-exact; the mutant that gathers children in
ASCENDING order fails the walker2d compare at the last bit
(−13.804855 vs −13.804854), so the gate sees the ordering claim; arm C
(RK4 with contacts, three steps) bit-exact; the Newton blocked golden
fingerprint; SO101Tabletop blocked vs the CPU oracle; the ThreeTrees
block oracle 115/115. The box A/B prices it against `probe_crba`.

**Priced (5090, `p0_ab.sh`, ROUNDS=3, MIN over rounds, B = 019c2605 vs
A = c98948ac = baseline 5):** rne kernel 30.7 → 25.8 µs at k=6
(**0.841**, 3/3), 62.2 → 60.4 at k=13 (**0.972**, 3/3); wall 0.996 /
0.999; every other row 0.99–1.005. Kept: bit-exact, gated, and the kernel
no longer scans the joint table per body per level. But the shape did
NOT transfer with its magnitude. The CRBA preamble was ~500 DEPENDENT
global loads per thread in a kernel with little other work; the RNE's
was ~100 independent loads plus a 100-deep backward chain, in a kernel
whose 60 µs are elsewhere — the level-serial forward pass (a barrier
and a global round trip per tree level, ~9 levels on the arm, the same
at every k, which is why the kernel costs 16 µs at k=0 already), cinert,
the projection. I had the bound in hand and reasoned past it: cdof does
the same per-body joint scan and costs 9 µs in total at k=13, so the
scan could not have been more than ~9 of RNE's 62, a ceiling of ~15%
for the whole cut; the thread-0 backward was named by analogy with the
LDL solve, not measured. §13.47's lesson: before landing a sibling of a
cut that paid, bound the term against a kernel that already does it.
RNE is 2.6% of the k=13 step; no further cut here.

**Where the step is now (k=13, 4.60 ms):** Newton 53% (closed at
stage 1, §13.43), collision 18.5%, unlabelled 14% (integrator 5%,
kinematics 3%, actuator apply 2.3%, sensor RNE 2%), LDL pair 8%, RNE
2.6%, CRBA 0.9%. And at k=0 (0.92 ms) collision is 63%: 269 µs a launch
with six dofs and a handful of geoms per env, nearly the same 426 at
84 dofs. A kernel that costs the same at 8 bodies as at 100 is paying
a fixed per-launch cost, not the model's — the next thing to read.

### 13.48 MEASURED (2026-09-08): the mesh hill climb is cold on every call, and the previous step already knows the answer

Context: MuJoCo 3.12 (now the pixi runtime, and `references/mujoco-3.12.0/`)
seeds `mjc_hillclimbSupport` from a per-mesh table of 27 extreme vertices,
one per direction of the (-1,0,1)³ grid (`mesh_extrema`, commit 83e621d7,
"up to 2× on large-mesh convex collision"). The cold start used to be
vertex 0, as ours still is. The park scene's hulls are 772–4,262 vertices
(ten meshes with graphs), and §13.18's block-kernel bisect put 206 of the
k=0 collision kernel's 270 µs in four GJK candidates — walks over these
hulls, each scan of a neighbourhood a dependent chain of global loads on
one thread. So before writing anything: how long are the walks, and what
would each seed buy? `_HILL_PROBE` in `collision/gjk.mojo` counts, and
replays each walk from 3.12's seed and from the previous step's landing;
`benchmarks/physics3d_cpu/hill_probe.mojo` drives it on k=0 (CPU, Euler,
the same code the GPU kernels inline).

| per step, k=0 (500 steps, two windows) | now | 3.12 seed | previous step's landing |
|---|---|---|---|
| hill-climb calls | 9 | 9 | 9 |
| of which cold (no warm vertex) | **9** | 9 | 0 after step 1 |
| scans per call | **13.8** | **5.9** | **1.00** |
| scans per step | 124 | 53 | 9 |
| landing differs from now | — | 500 of 4,500, **all ties** | — |

**Every call is cold.** The within-run warm start (`warm`, MuJoCo's
`meshindex`) never fires on this scene: each candidate's GJK proves the
pair apart on its FIRST support point and exits, so each mesh sees one
support call per step and starts it from vertex 0, 13.8 scans from the
answer. The warm start that pays here is ACROSS steps: the pose moves a
little per step, and the vertex a candidate landed on last step is the
answer this step in 4,491 of 4,491 replays (1.00 scans = the check that no
neighbour improves). 3.12's grid seed cuts the cold walk 2.3×; the
cross-step seed cuts it 14×. The 500 seeded landings that differ are one
call per step landing on a vertex with the SAME dot to 1e-6 relative — a
tie on a face perpendicular to the query direction — and 0 of 4,500 are
not ties. A tie moves the support POINT to another point of the same
face, so a seed can change a witness where the hull is flat; the
goldens, not the argument, decide whether any of ours does. (3.12 accepted
the same nondeterminism.)

**What this bounds.** If the four candidates' 206 µs are their walks
(28 scans each at ~1.8 µs, which is what a ~8-load dependent chain costs),
a cross-step seed removes ~26 of the 28 and the k=0 collision kernel
would read ~100 µs against 270; at k=13 the same candidates are the same
walks, so ~150 µs of the 426. That is a ceiling from a CPU count, not a
GPU time (§13.47's lesson stands): the A/B decides.

**The design, when it is built:** a per-env table of warm vertices in
`Data`, indexed by the candidate's ordinal in the serial emission order
(`[BATCH, 2 · COLL_NCAND_CAP]`), read into `warm1`/`warm2` before
`gjk_epa_witness` and written back after. An ordinal that shifts when the
candidate set changes hands a vertex of another mesh to the walk, which
the existing clamp turns into steps, never a wrong point. 3.12's extrema
are the cold fallback for a new ordinal — 5.9 scans against 13.8 — and a
model field the parser would fill; second, if the cold share after the
first step ever matters. Both legs (serial per-env kernel, block kernel,
CPU) share `_sap_pair_narrow`, so one threading serves all three.

### 13.49 LANDED, collision kernel 0.50 at both k (2026-09-08): the mesh hill climb warm-starts across steps

§13.48's design, built. The CCD workspace row (`Data.ccd_ws`, per env —
per CCD lane in the block kernel) grows a tail of `HILL_WARM_SLOTS = 128`
pairs of vertex indices (`HW_WS_OFF`, `ccd_workspace.mojo`), keyed by a
hash of the geom pair (`(gi·131 + gj) mod 128`) in `_sap_pair_narrow` and
in the SAP plane phase. `gjk_epa_witness` became a thin wrapper over
`_gjk_epa_witness_run`: it seeds `warm1`/`warm2` from the pair's slot
(mesh objects only — a box keeps `warm` as EPA's corner code and must start
at -1), the run starts BOTH of its phases from those seeds instead of -1,
and the wrapper writes the landings back. `_plane_mesh_contacts` takes the
warm vertex in and out (`mut warm`); the SAP plane phase hands it the slot,
the O(N²) detector (ngeom < 16, no scene of ours with large hulls) passes
-1 and is byte-for-byte what it was. Every other caller of the witness
function runs cold through the default `warm_slot = -1`.

**What the probe reads now (k=0, CPU, the same 9 calls a step):**

| | before | after |
|---|---|---|
| cold calls per step | 9 | **0** (after step 1) |
| scans per call | 13.8 | **1.00** |
| scans per step | 124 | **9** |

**Exact where it was measured.** `bench_so101` over 3,000 steps from
qpos0, warm start on vs off (`HILL_WARM_ACROSS_STEPS`): park_k0 `qsum`
2.1029078364372253 both, park_k3 502.05773257916786 both — the
trajectories are bit-identical, so no tie moved a witness on these scenes.
CPU step (Euler, one env): k=0 8.60 → **4.03 µs**, k=3 18.4 → **12.1 µs**
(same session, not interleaved — the direction is not in doubt, the third
digit is). GPU gates on Apple, all green: `test_plane_mesh_fields` (both
detectors vs golden), `test_sap_fields` (humanoid, sawyer mesh leg,
walker2d dispatch goldens), `test_newton_blocked_fields` (fingerprint),
`test_newton_freejoint_vs_cpu` (SO101Tabletop blocked vs CPU oracle),
`test_fields_mt_parity` arm C (contacts, three steps bit-exact),
`test_mesh_manifold_gpu_parity`.

⚠ `test_mesh_manifold_vs_mujoco` FAILS — "mesh contact DEPTH diverges
from MuJoCo by 0.0033" — and fails IDENTICALLY with the warm start off.
That is the runtime moving under the gate, not this change: pixi now
ships MuJoCo 3.12, whose multiccd carries a distance per witness point
(`witnessOnFace`, `status->dist[i]`) where 3.10 wrote one distance for
the whole manifold. To be re-read against 3.12's rule when the tests are
swept for the version.

**The box prices it** against `probe_rne` (= §13.47's tree). What §13.48
bounded: ~26 of 28 dependent scans per GJK candidate, i.e. the k=0
collision kernel from 270 toward ~100 µs and k=13's 426 toward ~250. A
CPU count, not a GPU time: the A/B decides, and the rows to read are
collision and — the unlabelled env kernel aside — nothing else, since
no other kernel touches `ccd_ws`.

**Priced (5090, `p0_ab.sh`, ROUNDS=3, MIN over rounds, B = 89e64dff vs
A = 019c2605 = §13.47's tree):**

| k | term | A ms/step | B ms/step | B/A | largest kernel A → B µs |
|---|------|-----------|-----------|-----|-------------------------|
| 6 | collision | 0.711 | 0.357 | **0.501** | 355.7 → 178.3 |
| 6 | wall | 2.058 | 1.720 | **0.835** | |
| 13 | collision | 0.853 | 0.420 | **0.492** | 426.6 → 209.9 |
| 13 | wall | 4.597 | 4.195 | **0.912** | |

The collision kernel halved at both k, 3/3 rounds, its hash changed (the
code path did). Every other row: Newton 1.010 / 1.003, LDL 0.989 / 0.998,
RNE 1.003 / 1.012, CRBA, cdof, the env kernel 1.00, same hashes. The one
stray is the warm-start kernel at k=13, 7.2 → 11.0 µs (1.538, 3/3): the
same 11-µs kernel that read 11.0 → 7.1 in §13.46's A/B, untouched then
and now, 0.008 ms of the step either way — it moves with something in the
box's state, not with the tree, and it is not booked in either direction.

§13.48's bound was 426 → ~250 at k=13 and 270 → ~100 at k=0; the kernel
read 210 at k=13, past the bound, so the walks were more than the 206 µs
the block-kernel bisect had charged to four candidates, or the serial
kernel's lockstep paid the chain more dearly. What is left in the kernel
at k=13 (210 µs) is the sweep, the AABB and body filters, the primitive
pairs, the GJK setup and the one scan per support call the seed cannot
remove; §13.18's CPU probe had the sweep and the per-pair filtering as
the next items on this scene's cousin, and a 5090 bisect
(`COLL_STOP_AFTER`, off knob, comptime-elided) would say which of them
the GPU pays.

**Baseline 6 = 89e64dff.** k=13 step 4.195 ms (1024 envs: 244 k
env-steps/s, 6.8× baseline 2's 28.5 ms), k=6 1.720 ms. Shares at k=13:
Newton 58.8%, unlabelled 15.8% (integrator ~5.5%, kinematics 3.2%,
actuator apply 2.5%, sensor RNE 2.2%), collision 10.0%, LDL pair 8.7%,
RNE 2.9%, CRBA 1.0%. The k=0 row of the sweep (collision was 63% of that
step) has not been re-run; the full baseline-6 sweep is what gives it.
