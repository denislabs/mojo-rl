"""OptimizerBundle[*OPTS: Optimizer] — variadic container of N optimizers.

Block E-1. Wraps any number of `Optimizer`-conforming instances behind a
single trainer-side field, so an agent with K nets+optimizers does not
need K parallel field declarations + K parallel `make` blocks.

Direct savings are modest on N=3 (SAC actor + 2 critics); the abstraction
pays back as N grows — DreamerV3's world-model decomposition usually has
6 or more optimizers (RSSM + decoder + reward head + done head + actor
+ critic), and a single OptimizerBundle field collapses the bookkeeping.

Composition:

    var bundle = OptimizerBundle[Adam, Adam, Adam].make_default()
    bundle.items[0] = Adam.make[target="cpu", M=ACTOR](actor)
    bundle.items[1] = Adam.make[target="cpu", M=CRITIC](critic1)
    bundle.items[2] = Adam.make[target="cpu", M=CRITIC](critic2)
    bundle.items[0].lr = Scalar[DT](3e-4)
    ...

Per-call dispatch (heterogeneous models — Mojo nightly rejects variadic
value-level packs of TileTensor-bearing receivers, so the bundle exposes
fixed-arity helpers instead):

    bundle.zero_grad_at["cpu", i=0, M=ACTOR](actor)
    bundle.step_at["cpu", i=0, M=ACTOR](actor)

Or — when N optimizers all bind to a single model — a homogeneous
convenience method:

    bundle.zero_grad_all_uniform["cpu", M=NET](shared_model)
    bundle.step_all_uniform["cpu", M=NET](shared_model)

That helper applies every optimizer to the same model — useful when N
schedules act on the same parameter set (e.g. SGD-on-warmup +
Adam-on-main).

CPU + GPU dispatch flows through `target: StaticString` exactly like the
underlying `Optimizer` trait. Bundle owns its own `ts: TargetStorage` so
build-time vs runtime errors point at the bundle, not the leaves.

Heterogeneous mixes (e.g. Adam + AdamW + SGD) work transparently because
`*OPTS: Optimizer` is a variadic generic — each tuple slot has its
declared concrete type at the call site.

**Mojo nightly aliasing constraint (2026-05-22, observed in TD3Trainer).**
Passing two `mut bundle.items[i]` / `mut bundle.items[j]` arguments to
the *same* call (e.g. `TwinCriticUpdateBlock.step(c1, c1_opt, c2, c2_opt,
...)`) is rejected by Mojo's alias analyzer — it treats both indexed
borrows as touching the same `bundle.items` Tuple field. Workarounds:

  1. Keep simultaneously-passed optimizers as separate trainer fields,
     bundle only optimizers that are passed individually.
  2. Split the multi-mut call into N single-mut calls (loses fused-block
     amortization).

DDPGTrainer (single-critic single-opt-per-call) bundles both opts
cleanly. TD3Trainer (twin-critic call passes both at once) bundles only
the actor opt and keeps critic1_opt + critic2_opt as bare fields. The
same limit applies if a future trainer's loss block takes 2+ optimizers
per call.
"""

from std.gpu.host import DeviceContext

from ..constants import DT
from ..core.module import Module
from ..core.optimizer import Optimizer
from ..core.target_storage import TargetStorage, assert_tag_for


struct OptimizerBundle[*OPTS: Optimizer](
    Defaultable & Movable & ImplicitlyDeletable
):
    comptime N = Self.OPTS.size

    var items: Tuple[*Self.OPTS]
    var ts: TargetStorage

    # ----- Defaultable -----------------------------------------------------

    def __init__(out self):
        comptime assert Self.N >= 1, "OptimizerBundle: at least one optimizer"
        self.items = Tuple[*Self.OPTS]()
        self.ts = TargetStorage.make_uninit()

    def __init__(out self, var *opts: *Self.OPTS):
        """Variadic consume — accepts pre-built optimizers.

        Tag is set to `cpu` since this constructor does not take a ctx.
        Use `OptimizerBundle[*](ctx, *opts)` for GPU bundles."""
        comptime assert Self.N >= 1, "OptimizerBundle: at least one optimizer"
        self.items = Tuple(*opts^)
        self.ts = TargetStorage.make_cpu()

    def __init__(out self, ctx: DeviceContext, var *opts: *Self.OPTS) raises:
        """GPU constructor — bundle tag set to gpu + ctx stored."""
        comptime assert Self.N >= 1, "OptimizerBundle: at least one optimizer"
        self.items = Tuple(*opts^)
        self.ts = TargetStorage.make_gpu(ctx)

    # ----- Factories -------------------------------------------------------

    @staticmethod
    def make_default[target: StaticString]() raises -> Self:
        """CPU factory: default-init each optimizer. Caller assigns real
        per-model optimizers into `items[i]` post-construction."""
        comptime assert target == "cpu", (
            "OptimizerBundle.make_default[target='gpu'] requires a DeviceContext"
        )
        var b = Self()
        b.ts = TargetStorage.make_cpu()
        return b^

    @staticmethod
    def make_default[target: StaticString](ctx: DeviceContext) raises -> Self:
        """GPU factory: default-init each optimizer + record ctx."""
        comptime assert target == "gpu", (
            "OptimizerBundle.make_default[target='cpu'](ctx) — drop ctx for CPU"
        )
        var b = Self()
        b.ts = TargetStorage.make_gpu(ctx)
        return b^

    # ----- Indexed per-optimizer dispatch ---------------------------------

    def zero_grad_at[
        target: StaticString,
        i: Int,
        M: Module,
    ](mut self, mut model: M) raises:
        """Apply `items[i].zero_grad[target, M](model)`.

        Heterogeneous packs need per-index dispatch because the caller
        only knows the concrete `M` at the call site (not at bundle-
        declaration time). The comptime `i` avoids a runtime branch."""
        comptime assert 0 <= i < Self.N, "OptimizerBundle.zero_grad_at: i out of range"
        assert_tag_for["OptimizerBundle", target](self.ts.target_tag)
        self.items[i].zero_grad[target, M=M](model)

    def step_at[
        target: StaticString,
        i: Int,
        M: Module,
    ](mut self, mut model: M) raises:
        """Apply `items[i].step[target, M](model)`."""
        comptime assert 0 <= i < Self.N, "OptimizerBundle.step_at: i out of range"
        assert_tag_for["OptimizerBundle", target](self.ts.target_tag)
        self.items[i].step[target, M=M](model)

    # ----- Homogeneous-model dispatch (every optimizer hits same model) ---

    def zero_grad_all_uniform[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        """Apply zero_grad to `model` through every optimizer in the bundle.

        Useful when multiple schedules act on a single parameter set
        (e.g. warmup-SGD + Adam-main, or LR-schedule overlay). For the
        more common case where each optimizer owns its own model, use
        `zero_grad_at[i, ...]` per index instead."""
        assert_tag_for["OptimizerBundle", target](self.ts.target_tag)
        comptime for i in range(Self.N):
            self.items[i].zero_grad[target, M=M](model)

    def step_all_uniform[
        target: StaticString,
        M: Module,
    ](mut self, mut model: M) raises:
        """Apply step to `model` through every optimizer in the bundle."""
        assert_tag_for["OptimizerBundle", target](self.ts.target_tag)
        comptime for i in range(Self.N):
            self.items[i].step[target, M=M](model)

    # Per-optimizer LR scheduling is done at the call site via
    # `bundle.items[i].lr = ...` — the `Optimizer` trait does not declare
    # `lr` (slim trait, see `core/optimizer.mojo`), so a uniform helper
    # would have to break trait abstraction. Callers can write a 2-line
    # comptime-for at the call site if they need it.
