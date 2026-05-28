"""Dropout[DIM, p, SEED] — inverted-dropout regularisation.

Phase 2 of `nn2/PORTING_PLAN.md`. First nn2 leaf to carry runtime
state beyond `TargetStorage`. Design choice (see PORTING_PLAN.md
Phase 2 train/eval section):

  - **Per-instance runtime `training: Bool` field**, default `True`.
    Set directly (`dropout.training = False`) or through
    `set_attr["training"](v)` where `v > 0.5` ⇒ True. We deliberately
    do NOT thread a comptime mode through the `Module` trait —
    nn2 explicitly removed an old `inference: Bool` flag from
    `TargetStorage` because no consumer ever used it. Dropout is the
    only leaf that needs train/eval, so we keep the surface local.
  - **Per-instance counter `call_counter: UInt64`**, bumped on every
    forward to give each call a unique PhiloxRandom offset. Mirrors
    the legacy `STATE_SIZE=1` GPU counter slot, just on the host.

Math (inverted dropout, identical to PyTorch):
    training:  mask ~ Bernoulli(1 - p), y = x · mask / (1 - p)
    eval:      y = x  (identity)

Backward (mask cached from forward):
    training:  grad_x = grad_y · mask
    eval:      grad_x = grad_y

Cache: leaf-owned `[BATCH, DIM]` slab — we cache the scaled mask
(0 or 1/(1-p)) so backward is a single elementwise multiply.

CPU-only at landing — no nn2 consumer needs Dropout yet; GPU is a
follow-up once one does (Phase 5 CNN agents typically don't use it
either, so this is genuinely on-demand work).
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.random.philox import Random as PhiloxRandom
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import (
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
)


struct Dropout[DIM: Int, p: Float64, SEED: UInt64](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    # Runtime state.
    var training: Bool
    var call_counter: UInt64
    # Mask cache [BATCH, DIM] — scaled (0 or 1/(1-p)).
    var cache_mask: List[Scalar[DT]]
    var ts: TargetStorage

    def __init__(out self):
        self.training = True
        self.call_counter = 0
        self.cache_mask = List[Scalar[DT]]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (no params). GPU path
        raises — no nn2 consumer needs Dropout yet (PORTING_PLAN.md
        Phase 2)."""
        comptime assert target == "cpu" or target == "gpu", (
            "Dropout: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.p >= 0.0 and Self.p < 1.0, (
            "Dropout: p must be in [0, 1)"
        )
        var d = Self()
        comptime if target == "cpu":
            d.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Dropout.make[target='gpu']: ctx required")
            raise Error(
                "Dropout: GPU path not implemented yet (see"
                " PORTING_PLAN.md Phase 2)"
            )
        return d^

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["Dropout", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            if not self.training:
                for b in range(BATCH):
                    for i in range(Self.DIM):
                        output_v[b, i] = input[b, i]
                # Eval pass doesn't bump the counter — keeps training
                # determinism cleanly separated from eval calls.
                return
            ensure_cpu_buffer(self.cache_mask, BATCH * Self.DIM)
            var cache_v = TileTensor(
                self.cache_mask, row_major[BATCH, Self.DIM](),
            )
            var scale = Scalar[DT](1.0 / (1.0 - Self.p))
            var threshold = Scalar[DT](Self.p)
            var zero = Scalar[DT](0.0)
            var base_offset = self.call_counter * UInt64(BATCH * Self.DIM)
            for b in range(BATCH):
                for i in range(Self.DIM):
                    var rng = PhiloxRandom(
                        seed=Self.SEED,
                        offset=base_offset
                        + UInt64(b * Self.DIM + i),
                    )
                    var rand = Scalar[DT](rng.step_uniform()[0])
                    var mask: Scalar[DT] = scale if rand >= threshold else zero
                    cache_v[b, i] = mask
                    output_v[b, i] = input[b, i] * mask
            self.call_counter += 1
        else:
            raise Error("Dropout.forward[target='gpu']: not implemented")

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Dropout", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](
            grad_inputs[0]
        )

        comptime if target == "cpu":
            if not self.training:
                for b in range(BATCH):
                    for i in range(Self.DIM):
                        grad_input_v[b, i] = grad_output_v[b, i]
                return
            var cache_v = TileTensor(
                self.cache_mask, row_major[BATCH, Self.DIM](),
            )
            for b in range(BATCH):
                for i in range(Self.DIM):
                    grad_input_v[b, i] = (
                        grad_output_v[b, i] * cache_v[b, i]
                    )
        else:
            raise Error("Dropout.vjp[target='gpu']: not implemented")

    # `set_attr["training"]` lets ComputeGraph / training-loop callers
    # flip the train/eval flag without naming the field directly. Value
    # convention: > 0.5 ⇒ True, else False (matches Clamp's set_attr
    # `Scalar[DT]` interface — there is no Bool-valued set_attr on the
    # trait).
    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "training":
            self.training = value > Scalar[DT](0.5)
