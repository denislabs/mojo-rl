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
