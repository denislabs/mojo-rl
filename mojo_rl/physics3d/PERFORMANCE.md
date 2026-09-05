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
