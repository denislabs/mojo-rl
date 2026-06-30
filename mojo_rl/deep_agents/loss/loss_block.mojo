"""LossBlock — marker trait for self-contained loss blocks (Block E-3).

A "loss block" is a self-contained struct that owns:
  1. Its own forward / backward scratch buffers (CPU + GPU mirrors)
  2. Its own `ts: TargetStorage` field
  3. A factory pair `make[target]()` / `make[target](ctx)`
  4. A domain-specific `step[target, ...]` (or `forward_backward[target, ...]`)
     method that runs the forward + loss + backward + optimizer pass
     and returns the scalar loss

Concrete impls (current): `CriticUpdateBlock`, `TwinCriticUpdateBlock`,
`TargetYBlock`, `SACActorLoss`.

**Why a marker, not a method-prescribing trait.** Step signatures vary
per block — `CriticUpdateBlock.step(critic, opt, sa_t, y_t)` is 4-input,
`TwinCriticUpdateBlock.step(c1, c1_opt, c2, c2_opt, s_ptr, a_ptr, y_t)`
is 7-input, `SACActorLoss.forward_backward(actor, opt, c1, c2, s_ptr,
alpha)` is 6-input and returns a 2-field record. No single trait method
covers them. The marker form still adds value:

  * **Trainer field type-grouping**: a field declared as
    `var blocks: SomeLossBlockBundle[...]` is meaningfully typed.
  * **Storage in `LossBlockBundle[*BLOCKS]`**: variadic Tuple of any
    `LossBlock` conformers, lifecycle-uniform.
  * **Lifecycle invariants**: every block gets the same
    `Defaultable & Movable & ImplicitlyDeletable` constraints, which
    happens to match all today's blocks' bases.

DreamerV3 / TD-MPC2 trainers will add 3–5 more loss blocks (reward head,
done head, RSSM dynamics, world-model decoder, value distributional head)
— having a single `LossBlock` marker + `LossBlockBundle` field will keep
those trainers compact.

If a future block has a step signature that *does* fit a uniform shape
(e.g. `step[target, OPT](inputs_packed) -> Scalar[DT]`), we can later
extend this trait with a method default — Mojo nightly supports default
trait method bodies (`feedback_mojo_trait_default_impls`), so existing
conformers stay green.
"""


trait LossBlock(Defaultable & Movable & ImplicitlyDeletable):
    pass
