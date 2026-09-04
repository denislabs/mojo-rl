# `swm/` — Sheaf World Model with holonomy as an observable (SWM-H)

**Status: Phases 0–2 of 6 complete.** Algebra core, place graph, analytic
gates, and the E1 environment land. Nothing is learned yet, and the central
hypothesis is still untested — that is Phase 3.

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

## What is deliberately absent

No cocycle loss (only as ablations C/C′ in a later gate). No spectral observable
in any loop. No transport conditioned on latents — `TransportBlock.generate`
will never take `u`, so violating that rule is a compile error rather than a
code review. No GPU: the design's edge-parallel kernel accumulates into vertex
gradients with `atomic_add`, a shape that has already produced a silent
miscompute in this repo, so CPU numbers get frozen as the reference first.

## Next

Phase 3 — the first phase that can refute the thesis. A learned encoder must
find the transported subspace under the mixing, and a learned transport must
keep the orientation bit stable. Does `det H = −1` survive? If not, hypothesis
4.0 is false as stated, and that is the result: write it up here and stop,
rather than build Phases 4–6 on it.
