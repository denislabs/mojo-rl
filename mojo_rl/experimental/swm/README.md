# `swm/` — Sheaf World Model with holonomy as an observable (SWM-H)

**Status: Phases 0–8 complete.** With learned encoders on
observations that mix a transported landmark with non-transported texture,
`det H = −1` on Möbius and `+1` on the orientable twin in **24/24 seeds each,
zero false obstructions**, with the frame channel carrying the landmark
(R² ≥ 0.983) and rejecting the texture (R² ≤ 0.009). Hypothesis 4.0 holds on
E1 — and survives a **nonlinear** observation model up to 16 % saturation
(G6b). The v1 cocycle loss is measured to be destructive or inert (G8).
Inference-by-descent, the confidence weights and the classification table land
with Phase 4, and the fault confusion matrix is clean: 40/40 in every cell,
**zero false obstructions**. The planner gets the lap parity right **95.8 %** of
the time where parity-blind baselines sit at chance.

- Design, as measured through Phase 8: [`docs/SHEAF_WORLD_MODELS_V3.md`](../../../docs/SHEAF_WORLD_MODELS_V3.md) (v2, French, is the original specification and is superseded)
- Plan, phases, gate definitions: [`docs/SWM_IMPLEMENTATION_PLAN.md`](../../../docs/SWM_IMPLEMENTATION_PLAN.md)
- Gates: `pixi run test-swm`

## The idea in one paragraph

A world model's local representations are stalks on a graph of places; the
transports between them are **orthogonal** matrices indexed by the *edge of the
base* (action, place, context) — never by the coordinates of the latents they
relate. The **holonomy** around a cycle is then read and never optimized. Its
class `det H ∈ {±1}` is a `Z/2` invariant that neither noise nor a sensor bias
can manufacture: a corridor whose seam mirrors the frame is a Möbius strip, it
is locally consistent everywhere and globally inconsistent, and that fact is a
property of the *cycle*, not of any edge. A high-residual edge is an outlier to
be down-weighted; a nominal-residual cycle with `det H = −1` is a fact about
the world, to be recorded and handed to the planner.

The v1 of this idea minimized a cocycle loss. That cannot work, and the reason
is exact algebra rather than an empirical finding: for any reflection,
`‖H − I‖²_F = 4 − 2 tr H = 4`, so the loss is *constant* on the `det = −1`
component and its tangent gradient is zero. With free morphisms it is instead
destructive — it crushes the frustrated dimension. So: read, never optimize.

## What is here

| File | Role |
|---|---|
| `so_d.mojo` | dense `D×D` algebra; `skew_from_vector`, `cayley`, `expm_skew` (→ SO(D)), `householder` (→ det = −1) |
| `procrustes.mojo` | per-edge `O(D)` fit by **Newton polar decomposition**, and the pre-consensus residual |
| `place_graph.mojo` | runtime place graph, spanning-tree gauge, `holonomy()` in `O(D³)` per cycle |
| `sheaf_laplacian.mojo` | dense `L = δᵀδ` + Jacobi spectrum — **gates only**, on no execution path |
| `reference_io.mojo` | reader for the pinned Phase 0 oracle |
| `rng.mojo` | xorshift64\* — reproducible from a seed the env carries, independent of `std.random` |
| `envs/mobius_ring.mojo` | E1: ring world, transported landmark + non-transported texture |
| `mlp.mojo` | 2-layer MLP, hand-written grads + Adam, float64 (see below) |
| `transport.mojo` | per-(action, place) O(D) transports, both components carried, orientation bit |
| `swm_trainer.mojo` | Phase 3 training loop and the observables it reads |
| `ablations.mojo` | the v1 arms (translations; free/orthogonal morphisms + cocycle) — **gates only** |
| `sheaf_inference.mojo` | Dirichlet-energy descent over the frame channel + an exact solve to gate it |
| `observables.mojo` | pre-consensus residual, GNC confidence, the §1.2 classification table, cross-cycle confirmation, the verdict latch |
| `planner.mojo` | rollouts in edge coordinates, CEM + an exhaustive monotone planner, trust penalty |
| `content.mojo` | the content channel: decoder (anchors both channels) + a free nonlinear transition |
| `envs/klein_grid.mojo` | a non-orientable O(2) bundle over a torus grid — many cycles, one reversing |
| `place_recognition.mojo` | appearance-based place memory, scored against the oracle |
| `observables.mojo` (Phase 7) | root-gauge PCM composition, the bootstrapped Z/2 maximal clique, `classify_cycle` with `dim ker(H − I)` |
| `planner.mojo` (Phase 7) | `plan_exhaustive_with_place_code`: the content channel as a per-cell lookup along the rollout |
| `envs/klein_grid.mojo` (`flat=True`) | the seam as a deck transformation: a FLAT non-orientable bundle, holonomy group `{I, M}` |
| `world.mojo` (Phase 8) | the `SwmWorld` trait: what a world supplies for the one training loop |
| `envs/klein_grid.mojo` `KleinWorld` (Phase 8) | the 2D bundle with an observation model (landmark + texture, aliasing modes) |
| `map_builder.mojo` (Phase 8) | online labels from the content channel, successor conflicts, context splitting, per-clone transports |
| `cscg.mojo` (Phase 8) | Clone-Structured Cognitive Graph by EM — the P5 baseline, gates only |

## What was learned (Phase 0–1)

**Newton polar decomposition replaces the SVD we do not have.** There is no
generic SVD in this repo. The orthogonal Procrustes answer `U Vᵀ` is exactly
the orthogonal polar factor of `M = Y Xᵀ`, reachable with only matmul and
inverse. Measured against numpy's SVD on the same 1920 observation pairs, the
agreement is **5.6e-16** — machine precision, on all 12 edges. It also inherits
`sign(det M)`, so it yields `O(D)` and not `SO(D)`, which is required since the
reflection *is* the signal.

**The spectral claim is confirmed at 13 digits, twice.** `λ₂ = 2(1 − cos(π/N))`
holds in the numpy oracle and independently in the Mojo assembly, for
`N ∈ {12, 24, 48, 96}`, while `‖H − I‖_F` stays at 2.000 out to `N = 192`. This
is the measurement that settles the design question: on long loops the spectrum
drowns and the holonomy does not.

**An orthogonality check cannot gate a rotation generator.** The first version
of G1 asserted that `expm_skew` of a large generator stays on the manifold. A
mutant that dropped the scaling-and-squaring **passed it**, because `exp(S/2ᵏ)`
is still perfectly orthogonal — just the wrong rotation. Membership in `O(D)`
is preserved by a whole family of wrong answers. The gate now pins the *value*
with closed forms (`exp(skew(t)) = rot(t)`, `cayley(skew(t)) = rot(−2 atan t)`)
plus the semigroup property `exp(S) = exp(S/2)²`, and kills the mutant.

**§5 of the design doc has a bug.** Its `holonomy()` pseudo-code returns
`T_dst ᵀ T_src`, omitting `R_e`. The general form `H_e = T_dst ᵀ R_e T_src` is
what is implemented; §2 is right and §4.4 is its `R_e = I` special case.

**A derived parameter count does not fold at a literal call site.** A signature
of `InlineArray[..., D * (D - 1) // 2]` type-checks when `D` is a parameter but
fails when a caller writes `[2]` ("types parameters include unfolded expression
at parser time"). `skew_from_vector` and `householder` take a `Span` instead,
with the length checked at runtime — which is also the natural spelling once
Phase 3 generates the coefficients from `W_a a + W_l l + W_c c`.

**The environment carries nuisance, and that is not decoration.** The design
doc's E1 is `x_i = g_i w` with `g_i ∈ O(2)` and nothing else — the observation
*is* an orthogonal group action, so recovering `O(2)` structure from it would
not test hypothesis 4.0 (that the topologically relevant part is *separable*
from ordinary content). A world with no "rest" cannot falsify that. So each
cell also carries a texture which identifies the cell, is bit-identical at both
lap parities, and is not transported; the observation is an overcomplete
mixing (2 + 6 → 16) of the two. The aliasing is then sharp: texture says "same
cell", landmark says "mirrored", and only the holonomy explains both.

**A tolerance-based reward made the task ill-posed.** "Reach the cell where the
landmark appears on the left" with a fixed goal cell and an angular tolerance
is satisfiable in only ~45% of episodes, because the landmark direction is
redrawn each episode — some episodes have no goal at any parity. The reward is
now an **argmax over all `2N` frames** (the double cover), which always exists,
is generically unique, and is a (cell, *parity*) pair — so standing in the right
cell is not enough. The orientable control ties exactly across parities in
20/20 episodes, which is what shows the difficulty comes from the seam and not
from how the reward is written.

**The transport loss is trivially satisfiable by a place-indexed constant, and
that nearly produced a false P1.** If the encoder learns `u = f(place)`, the
transport for `(action, place)` — itself indexed by place — only has to carry
one fixed vector to one fixed vector. In 2D exactly one rotation and one
reflection do that, both at ~zero residual, so the orientation bit is chosen by
noise and `det H` becomes a **fair coin**. Measured: landmark R² = 0.003,
nuisance R² = 0.5, and `det H = −1` in 2/6 Möbius seeds *and* 2/6 orientable
seeds. The first Möbius run alone looked like a clean P1 pass.

The fix is that the anti-collapse term must be measured **within a place**, not
across places — the degenerate solution has excellent across-place variance,
that is what it *is*. This is not a trick: a frame is precisely what varies
while you stand still, and a place label is precisely what does not. With a
per-place variance hinge the encoder finds the landmark (R² 0.99) and rejects
the texture (R² 0.003).

**Why `mlp.mojo` and not `mojo_rl.nn`.** `nn` is float32
(`nn/constants.mojo`). Phase 3 must distinguish "the frame channel collapsed to
rank one" from "the obstruction is genuinely absent" — the first makes a
reflection act like the identity and reads `+1` for reasons unrelated to the
world. That is a 1e-12-scale distinction, and the networks here are 16→32→10.
Porting to `nn` is graduation work.

**The cocycle loss is inert on the topological class, and it is exact.**
`L = ‖H − I‖²_F = 4 − 2 tr H`; on the `det = −1` component every `H` is a
reflection, so `tr H = 0` and `L = 4` identically. Measured: raw gradient
**13.86**, tangent projection **6.7e-16**. The control that makes that a
measurement rather than an artefact is a *continuously frustrated* orientable
ring (`det = +1`, `H ≠ I`), where the tangent gradient is **7.61** — sixteen
orders of magnitude apart. With free morphisms the same term is destructive
instead: `det H` runs −1.00 → −0.47 → −0.14 → −0.02 as λ grows, the minimum
singular value shrinks 1.0 → 0.64, and the local residual rises 8×.

**The prototype's 7–10× parity gap does not transfer, for an exact reason.** In
the oracle-frame prototype the observations are centred, so model B's fitted
translations are exactly zero: it predicts "nothing moves", which is *right*
after an even number of laps and wrong after an odd one — a clean parity ratio
with B nearly matching A at even `k`. A learned frame carries an offset, so the
translations are non-zero, accumulate, and B fails at *every* lap (10–27× worse
than A, odd/even compressed to 1.8×). A stronger refutation of the constant
sheaf, not a weaker one — so the gate asserts the B-vs-A gap and *reports* the
parity ratio instead of asserting the prototype's number.

**The result survives observation nonlinearity, and fails gracefully past it.**
G6 alone uses a linear mixing, where the split hypothesis 4.0 asks for exists
exactly — the friendliest observation model, and a real bound on what G6 means.
With `obs = tanh(gain · mix @ latent)`:

| gain | saturated | Möbius | orientable | false obstr. | worst landmark R² |
|---|---|---|---|---|---|
| 0.5 | 0 % | 8/8 | 8/8 | 0 | 0.988 |
| 1.0 | 1.8 % | 8/8 | 8/8 | 0 | 0.971 |
| 2.0 | 16.1 % | 8/8 | 8/8 | 0 | 0.933 |
| 4.0 | 38.6 % | 7/8 | 8/8 | 0 | 0.449 |

The breaking point tracks **saturation**, not nonlinearity: at gain 4 two fifths
of the coordinates are pinned at ±1, destroying the landmark rather than
entangling it. And the asymmetry that makes the observable usable — even where
the frame channel *cannot* recover the landmark, there are **zero false
obstructions**. The method loses the signal rather than inventing one.

**The removal rule is necessary, and it is measured as a 2×2.** The doc asserts
that an identification whose cycle carries a non-trivial holonomy must be taken
out of the energy — it is a monodromy, not a constraint. Inferred-frame
anisotropy at the revisited place:

| β (anchor weight) | Möbius +ID | Möbius −ID | orientable +ID | orientable −ID |
|---|---|---|---|---|
| 0.02 | **0.092** | 0.943 | 0.943 | 0.943 |
| 0.10 | 0.157 | 0.944 | 0.943 | 0.944 |
| 1.00 | 0.456 | 0.945 | 0.944 | 0.945 |
| 5.00 | 0.756 | 0.946 | 0.945 | 0.946 |

Only one cell moves. Adding a *consistent* identification on the orientable twin
changes nothing, so the collapse comes from the contradiction and not from
adding a constraint. The β dependence is §4.4's mechanism visible: weak anchors
let the disagreement spread into the frame (a 10× loss of anisotropy — the frame
driven into the cycle's one-dimensional fixed subspace), strong anchors deposit
it on the identification edge instead. Either way the residual must be read
*before* consensus, which is why `observables` never reads it after.

**The fault classification defends an asymmetry, not a rate.** An obstruction is
a fact to record and hand to the planner; a broken sensor is an edge to
down-weight; a constant bias is *neither* — it produces a perfectly coherent
continuous holonomy that one cycle cannot tell from real curvature, so it must
come out UNDECIDED and must **never** be filed as an obstruction. 40 trials each:

| world | NOMINAL | ABERRANT | OBSTRUCTION | UNDECIDED | mean ‖H−I‖ | w[faulty edge] |
|---|---|---|---|---|---|---|
| Möbius | 0 | 0 | **40** | 0 | 2.000 | 0.98 |
| noisy edge | 0 | **40** | 0 | 0 | 0.040 | **0.010** |
| biased edge | 0 | 0 | 0 | **40** | 0.424 | 0.98 |
| clean | **40** | 0 | 0 | 0 | 0.007 | 0.98 |

Zero false obstructions, and zero Möbius obstructions explained away as aberrant
edges. The biased world's 0.424 matches the prototype's 0.430.

**The GNC threshold is the inlier scale, not the typical residual.** Geman–McClure
gives `w = 0.25` exactly at `r = c̄`, so setting `c̄` to the median scores every
nominal edge at 0.25 and leaves nothing to distinguish "fine" from "doubtful"
(measured, before it was fixed). The prototype's `10 × median` puts a
median-residual edge at 0.98 and is kept — which is also why the prototype's own
weights sat at a uniform 0.82: a fixed `c̄` with no schedule.

**The planner's claim is PARITY, not raw goal success.** The task is
goal-conditioned, which makes it gauge-free: the encoder learns some basis, so
"the landmark appears on the left" is a direction the planner cannot name, but
"reach a state that looks like this" is well posed in any gauge. Over 3 training
seeds × 40 episodes:

| model | parity correct | cell | goal |
|---|---|---|---|
| SWM (orthogonal) | **115/120 = 95.8 %** | 99 | 95 |
| translation (constant sheaf) | 58/120 = 48.3 % (chance) | 42 | 25 |
| place lookup (no double cover) | 78/120 = 65.0 % | 12 | 3 |
| SWM + monodromy applied twice | **35/120 = 29.2 %** | 48 | 23 |

Parity is what the double cover buys and a parity-blind model can only guess it,
so it is the headline; cell accuracy is reported beside it and is *not* the
claim. Of SWM's failures, **9 in 10 are the right parity in the wrong cell** —
the frame alone is a weak place code (adjacent cells differ by ~0.3 rad; the
frame identifies the goal in ~93–95 % of episodes against ~98 % for frame +
content). That residue is the concrete argument for the content channel, not a
failure of the frame channel at its own job.

The monodromy-twice ablation lands **below chance** (29.2 %), which is the tell:
`H² = I` predicts the un-reflected frame, so it does not guess the parity — it
systematically inverts it, while cell tracking largely survives (48 vs 23).

**Planning is exhaustive over monotone walks, not CEM, and that was a
measurement.** On a ring, forward `k ∈ [0, 2N)` reaches every double-cover state
exactly once, so the optimum is a scan of `4N` rollouts. The CEM planner's step
penalty turned out to be a path-length prior that trades near goals against far
ones (penalty 0 → 14/14 far and 16/26 near; penalty 0.01 → 11/14 and 22/26).
Those are properties of the search, and a gate on the world model should not be
reading them.

**Frame prediction error does not grow with rollout length** — flat at ~0.11
from 1 to 24 steps, and lower at 24 (two laps, where `H² = I` lets errors
cancel). Orthogonal transports rotate error rather than amplify it, so long
imagined rollouts stay usable. I had assumed the opposite and measured it.

**The content channel localises, but must not be rolled far — and that is the
best evidence for the design's asymmetry.** Phase 5's deficit was cell-level, so
Phase 6a added the content channel. It works at its own job (nearest-centroid
cell accuracy from `h` alone: 0.55 untrained → **0.998** trained) and does not
contaminate the frame (nuisance R² stays ~0.005, landmark R² 0.987, `det H = −1`
with one reflected edge still 3/3). But matching on `(u, h)` **hurts** the
planner, monotonically in how much the content is trusted (weight 0 / 0.05 / 1.0
→ cell 96 / 77 / 75 of 120).

The reason is measured, not guessed:

| rollout steps | frame error | content error |
|---|---|---|
| 1 | 0.092 | 0.405 |
| 6 | 0.117 | 1.709 |
| 12 | 0.107 | **4.398** |

The frame is flat because its transition is an **isometry** — an orthogonal map
rotates error instead of amplifying it — while a free nonlinear transition
drifts 11× over twelve steps. The orthogonal constraint is normally justified by
what it does for the holonomy (it makes `det H` an invariant); this says it also
buys the only channel still trustworthy after a long imagined rollout. So the
content channel's value is localisation *from an observation*, not imagination,
and using it well means observing at the arrival state and re-planning — a
control-loop change rather than a world-model one. Phase 5 plans open-loop on
purpose, to keep the model under test, so the negative result is recorded and
gated rather than engineered around.

**Parity is provably not decodable from a single frame.** Both channels score
~0.67 on absolute within-place parity classification, and they are *supposed*
to be indistinguishable: `u(c,1) = A F_{c,0} H w`, and since `H` is orthogonal
and `w` is uniform on the circle, `Hw` is uniform too — the two parity classes
have identical marginals. Parity is decodable only *relative* to a reference,
which is exactly why the Phase 5 task is goal-conditioned and where the real
parity number lives (95.8% vs 48.3%).

**Many cycles at once, and the reading `det H` misses.** Every gate before 6b
read exactly one holonomy, because a ring has one cycle. A torus grid with a
reflecting x-seam gives 31 fundamental cycles — 5 orientation-reversing, 8
non-trivial rotations (loops crossing the seam twice at different frames), 18
trivial — and the `Z/2` class of **all 31** survives Procrustes recovery from
noisy observations. It is also the first real exercise of
`confirm_by_independent_cycles`, previously gated only on synthetic edge lists:
4 edge-disjoint pairs of non-trivial cycles exist here, which is the situation
the rule was written for.

And it pins the reading the determinant under-reports above two dimensions —
the design doc's own example:

| | `det H` | `dim ker(H − I)` |
|---|---|---|
| O(2) reflection | −1 | 1 (fixes a line) |
| O(3) reflection | −1 | 2 (fixes a plane) |
| O(3) `H = −I` | −1 | **0 (fixes nothing)** |

Same class, different fixed subspace. In 2D the two readings agree; above 2D
they come apart, so a method that only reads the determinant is answering a
coarser question than the one asked.

**Removing the last oracle costs more than the doc expects.** Every result from
Phase 3 on assumed an oracle place identity. Holding the landmark fixed so that
mirroring is the only difference, the doc's predicted failure appears exactly as
described — a whole-latent similarity misses precisely the revisits that create
the informative cycle:

| recogniser | parity 0 | parity 1 |
|---|---|---|
| naive, whole latent | 240/240 | **118/240** |
| content channel only | 240/240 | **240/240** |

The content channel supplies the required frame-invariance, and for a structural
reason: under the reflection `u → Hu`, and the only O(2) invariant of a vector is
its norm, which is near-constant here — the frame *cannot* carry frame-invariant
place information.

That makes E1 too easy unless the textures alias, so `aliased_mobius(2)` makes
cells `c` and `c + 6` perceptually identical. Then: **a false identification
manufactures `det H = −1` exactly as a true one does** (93 of 95 false, 115 of
353 true). The holonomy depends on the *graph span*, not on whether the two
places genuinely match — so `det H` inherits place recognition's reliability
wholesale. The doc's §7 says a false identification "creates an aberrant edge
(handled)"; under perceptual aliasing that is too optimistic, and it is the
largest open risk in the method.

**PCM defends, but its textbook criterion is wrong here.** Asking whether two
composed closures return the *identity* assumes a global frame exists. On a
Möbius ring it does not, and two genuine closures at opposite parities compose
to the *reflection*. Asking instead whether the composition lies in the world's
**holonomy group** recovers the good closures:

| criterion | true–true consistent | true–false |
|---|---|---|
| composition = I (textbook) | 29270/62128 = 47% | 1234/33535 = 3.7% |
| composition ∈ {I, H} (corrected) | 45196/62128 = **73%** | 3781/33535 = 11% |

Both discriminate; the textbook form throws away more than half the loop
closures a map depends on.

## What is deliberately absent

No cocycle loss (only as ablations C/C′ in a later gate). No spectral observable
in any loop. No transport conditioned on latents — `TransportBlock.generate`
will never take `u`, so violating that rule is a compile error rather than a
code review. No GPU: the design's edge-parallel kernel accumulates into vertex
gradients with `atomic_add`, a shape that has already produced a silent
miscompute in this repo, so CPU numbers get frozen as the reference first.

## What is still not established

Place recognition is an oracle throughout. On E1 it is confounded with the
effect being measured: after an odd lap the same place looks *mirrored*, so a
naive encoding similarity would miss exactly the identifications that create the
informative cycle. A learned, frame-invariant recogniser is Phase 6, and nothing
measured here speaks to it.

## What was learned (Phase 7 — identifications without an oracle)

**6c's PCM composition mixed gauges, and the 73 % was an artefact.** In the
spanning-tree gauge the tree is flat, so the PCM composition of two closures is
the product of their holonomies, `H_a H_b^T`. G17 had composed
`T_ta^T T_tb · T_sb^T T_sa`, which treats the tree path between base places as a
transport. Corrected, on the same 353 true and 95 false closures:

| criterion | true–true | true–false | false–false |
|---|---|---|---|
| G17 as first written, `= I` | 47 % | 3.7 % | not measured |
| G17 as first written, `∈ {I, H}` | 73 % | 11 % | not measured |
| root gauge, `= I` | 56 % | 4.8 % | 16 % |
| root gauge, `∈ {I, M}` | **100 %** | 12.6 % | 16.5 % |

The corrected textbook figure is exactly the same-parity pair count
(28203 + 6555 of 62128), which is the check that the fix is right. False
closures agree with each other only when they share a base cell, so they do
not form a competing clique.

**The clique, with the group bootstrapped from the closures (G18).** No lap
length, no true monodromy: each `det = −1` closure is tried as `M`, the largest
set within tolerance of `{I, M}` wins, and `M` is refined to the polar factor
of the members' sum — the seed closure sat 0.285 from the true monodromy
(between the true reflections and a cluster of gauge-coincident false ones),
the refined one 0.071. The clique keeps **353/353** true closures and
**10/95** false ones, from two base cells. Those ten are the residual no
consistency test can see: their spurious holonomy coincides with `I` or `M`.

**A false identification cannot invent a reflection (G18).** The control 6c
did not run: on the ORIENTABLE aliased twin, with confusion modelled (the
learned content channel disambiguates aliased cells there on its own — it is
free to absorb frame information when nothing flips — 1 false match in 480),
**zero** of 960 closures read `det = −1`. A closure's holonomy is the transport
along the walk it spans, and on an orientable world every walk is a rotation.
So 6c's "det H inherits place recognition's reliability wholesale" narrows to:
**the graph inherits it; the Z/2 class of a walk is a fact about the walk.** A
false identification mis-attributes a reflection the world has to a spurious
cycle; it never manufactures one.

**The oracle leak, closed (G18).** Phases 3–6c indexed the transport table by
the oracle cell even while the recogniser was under test. Indexed by the
texture label instead — what a content recogniser can deliver — a merged entry
must fit two rotations with one matrix. The design's §7 promised an aberrant
edge; what happens is co-adaptation: the residual is elevated **2.1–9.0×** the
floor, below the 10× ABERRANT threshold, the frame channel's landmark R² drops
0.998 → 0.752, the per-cell residual under a merged transport is *not* bimodal
(worst pair ratio 1.01, so a residual-norm splitting rule has no signal), and
the seam becomes unrepresentable — one entry would have to be a rotation for
one cell and a reflection for its alias. **The obstruction is lost, never
manufactured**: 0 OBSTRUCTION and 0 ABERRANT of 410 closures, against 115
true odd-lap obstructions with oracle labels.

**Two dimensions, on a flat bundle (G19).** 6b's bundle is curved along its
seam (a square straddling it composes `refl r refl⁻¹ = r⁻¹`), which is why it
had 8 rotation cycles. With the seam inserted as a deck transformation the
bundle is flat and every root-gauge holonomy lies in `{I, M}`. There the Z/2
clique is exact (2166/2166 true closures, `M` within 0.003 of the seam); on
the curved bundle it keeps **18 %** — the flatness assumption, gated so it
cannot be forgotten. And the plan's prediction that the second dimension
resolves gauge coincidence is **refuted**: 41 % of false closures survive on
the grid against 10 % on the ring, because on a flat bundle the transport
between two aliased places is the same along every homotopic path and differs
by `M` across classes, so coincidence in one class is coincidence in all. The
driver is the angular spread of frames between aliased places relative to the
tolerance, not the dimension. What refutes a *local* false identification is
the graph: a merged label has two successor labels under one action (2
conflicts on the one merged pair; 0 without aliasing; 0 under a global
symmetry, whose quotient is a consistent world). That is the signal a
clone-splitting rule runs on.

**E3 through the encoder, and the encoder compresses the fault (G20).** Same
four worlds as G9, faults now in the observation. Verdicts 3/3 in every cell,
zero false obstructions, biased edge `‖H − I‖ = 0.41–0.48` against the
prototype's 0.43. But a sensor with 30× the noise gives a residual **4.7×**
the median through the encoder, where G9 saw **920×** at frame level: an MLP
over 16 mixed coordinates averages the noise down. Clean worlds sit at
1.5–1.7×. The ABERRANT band is one order of magnitude wide here, not three;
the outlier factor is 3 and gated from both sides.

**The content channel is a place code, not a dynamics (G21).** Re-planning
every step does not fix 6a: with rolled content in the score, cell 103 and
parity down to 106; frame-only, the loop oscillates (28 cells, budget
exhausted — "stay when the best plan is to stay" is unstable on a noisy frame
match). A ±1 probe at arrival helps nothing (97 vs 96) because the failures
are 2–3 cells away, never adjacent. What works: the rollout's content term is
a **lookup** of the stored centroid of the cell the rollout stands in, matched
against the goal's observed `h`. Cell accuracy **120/120** (open-loop 96,
rolled content 103; the same planner at content weight 0 is exactly the
open-loop arm, 96).

**The parity residue is the fixed subspace (G21 ⇄ G22).** With the cell
pinned the frame chooses between two states one lap apart, which differ by
`M`. Every one of the 10/120 parity failures falls in the lowest third of the
frame's own margin `|u_k − u_{k+N}|²`, none in the upper two thirds: where the
goal's landmark lies along `M`'s axis — `dim ker(H − I) = 1`, the direction
that admits a global frame — the frame carries no parity. `classify_cycle`
now returns that dimension beside the class (O(3) reflection fixes a plane,
`−I` fixes nothing, both `det = −1`; det-only classification files them
identically, shown).

## What was learned (Phase 8 — a map with no oracle in it)

**One training loop, two worlds (G23).** The loop now runs over the
`SwmWorld` trait; the ring's gates pass through it unchanged. On the flat
Klein world with learned encoders the transports reproduce the planted Z/2
class on **31/31** fundamental cycles in every seed (5 reversing), the torus
reads 0, landmark R² ≥ 0.98, texture R² ≤ 0.003, content cell accuracy 1.0.
Phase 3's result holds in 2D with two actions.

**The graph resolves local aliasing, and repairs a weak labeller (G24).**
With the content channel labelling visits online (29 labels for 30 cells, the
aliased pair merged), the merged label and no other shows a successor
conflict; splitting by context gives **30 clones at purity 1.0**, and the
clone graph with re-fitted transports reads **5 reversing cycles** — the
planted number, with no oracle anywhere from encoding onward. On the
orientable twin the content channel is a weaker place code (G18) and the
labeller under-segments to 26 labels; the same rule recovers 30 clones at
purity 1.0 with 0 reversing. A global symmetry gives 15 labels, 0 conflicts
and nothing to split: the quotient is a consistent world, and its holonomy
reading (1 reversing of 16) is not the truth's (5 of 31) — the frame
disagreement G18 measured, now as a reading.

**P5 is not confirmed (G25).** A CSCG learned by EM on the same label and
action sequences is level with the context rule: aliasing resolved at 121
visits (both over-split there), the exact 30-clone map at 242 visits, both.
What separates them is stability: a fixed-budget EM keeps
splitting places into extra pure clones as data grows (30 → 39 → 46 → 56
while purity stays 1.0), the context rule is pinned at 30 because a label
with one context is never split. The frame channel needs 484 visits for
enough pairs per edge to read all 5 reversing cycles.

| visits | SWM clones | CSCG clones | aliasing resolved | exact map |
|---|---|---|---|---|
| 121 | 34 | 35 | both, over-split | neither |
| 242 | 30 | 30 | both | both |
| 484 | 30 | 39 | both | SWM only |
| 1936 | 30 | 56 | both | SWM only |

## Open questions after Phase 8

1. **Global symmetry is a genuine ambiguity.** Nothing in the graph refutes a
   quotient; only the frame transports disagree, and G18 showed the encoder
   co-adapts rather than flags. Whether a *transport residual vector* (not its
   norm) is bimodal on merged entries is unmeasured and is the only lead.
2. **CSCG with a merge step.** The fragmentation G25 records is what a
   fixed-budget EM does without pruning; the published method prunes by
   usage, which would not merge pure duplicates either. A fair extension is
   context-based merging on CSCG's side, and then the comparison is between
   two graph rules.
3. **Curved bundles** and the deferred items (GPU, float32 `nn` port, E2)
   as before.

