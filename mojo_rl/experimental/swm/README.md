# `swm/` — Sheaf World Model with holonomy as an observable (SWM-H)

**Status: Phases 0–4 complete — P1, P2 and P4 answered.** With learned encoders on
observations that mix a transported landmark with non-transported texture,
`det H = −1` on Möbius and `+1` on the orientable twin in **24/24 seeds each,
zero false obstructions**, with the frame channel carrying the landmark
(R² ≥ 0.983) and rejecting the texture (R² ≤ 0.009). Hypothesis 4.0 holds on
E1 — and survives a **nonlinear** observation model up to 16 % saturation
(G6b). The v1 cocycle loss is measured to be destructive or inert (G8).
Inference-by-descent, the confidence weights and the classification table land
with Phase 4, and the fault confusion matrix is clean: 40/40 in every cell,
**zero false obstructions**.

- Design: [`docs/SHEAF_WORLD_MODELS_V2.md`](../../../docs/SHEAF_WORLD_MODELS_V2.md)
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

## Next

Phase 5: the planner in edge coordinates (CEM/MPPI, `u ← R_e u` with the content
block alongside), a trust penalty on low-confidence edges, an intrinsic reward
for closing unconfirmed cycles, and the E1 parity task against a translation
baseline and a recurrent one. Note G14 in advance: the planner must **not** apply
the monodromy twice — the edge transports already carry the reflection at the
seam.

A caveat on how far G6 reaches. E1's observation is a *linear* mixing of
landmark and texture, so the split hypothesis 4.0 asks for genuinely exists and
is linearly recoverable. G6 establishes that the mechanism finds it and that
`det H` survives the encoder's gauge. It does **not** establish that the method
works when the split is only approximate.
