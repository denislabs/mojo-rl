"""Tokenwise[SEQ_LEN, Inner] — apply a shared-weight Module per token.

A sequence sample is laid out `(SEQ_LEN, Inner.IN)` row-major (token-
major). Tokenwise reinterprets the `(BATCH, SEQ_LEN*Inner.IN)` slab as
`(BATCH*SEQ_LEN, Inner.IN)` and runs `Inner` once over that flattened
batch, so the same weights are applied at every position. The reshape is
pure pointer reinterpretation (row-major flat index is identical), so
there is no mid-slab and no extra kernel: forward/vjp delegate straight
to `Inner` at batch `BATCH*SEQ_LEN`.

  IN_DIM  = SEQ_LEN * Inner.IN_DIMS[0]
  OUT_DIM = SEQ_LEN * Inner.OUT_DIM

`Inner` owns its own params (shared across positions — exactly the
tokenwise contract) and its own cache (lazily sized to `BATCH*SEQ_LEN`
rows on first forward). Walkers / set_attr / zero_grad delegate to Inner.

Used throughout the transformer to wrap the per-token Q/K/V projection,
output projection, FFN Linears, LayerNorm, and the token Embedding.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import TargetStorage, assert_tag_for


struct Tokenwise[SEQ_LEN: Int, Inner: Module](Module):
    comptime ARITY: Int = 1
    comptime IN_INNER: Int = Self.Inner.IN_DIMS[0]
    comptime OUT_INNER: Int = Self.Inner.OUT_DIM
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.SEQ_LEN * Self.IN_INNER)
    comptime OUT_DIM = Self.SEQ_LEN * Self.OUT_INNER

    var inner: Self.Inner
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.SEQ_LEN > 0, "Tokenwise: SEQ_LEN must be > 0"
        self.inner = Self.Inner()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "Tokenwise: target must be 'cpu' or 'gpu'"
        )
        var t = Self()
        t.inner = Self.Inner.make[target, INIT](ctx=ctx)
        comptime if target == "cpu":
            t.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Tokenwise.make[target='gpu']: ctx required")
            t.ts = TargetStorage.make_gpu(ctx.value())
        return t^

    @staticmethod
    def display_label() -> String:
        return String("Tokenwise")

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["Tokenwise", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)
        var ip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
        var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)
        # Reinterpret (BATCH, SEQ_LEN*IN_INNER) → (BATCH*SEQ_LEN, IN_INNER).
        var in_r = TileTensor(
            ip, row_major[BATCH * Self.SEQ_LEN, Self.IN_INNER]()
        )
        var out_r = TileTensor(
            op, row_major[BATCH * Self.SEQ_LEN, Self.OUT_INNER]()
        )
        self.inner.forward[target, BATCH * Self.SEQ_LEN, POLICY=POLICY](
            in_r, output=out_r
        )

    # ----- Backward --------------------------------------------------------

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
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Tokenwise", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_output_v.ptr
        )
        var gip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            grad_input_v.ptr
        )
        var go_r = TileTensor(
            gop, row_major[BATCH * Self.SEQ_LEN, Self.OUT_INNER]()
        )
        var gi_r = TileTensor(
            gip, row_major[BATCH * Self.SEQ_LEN, Self.IN_INNER]()
        )
        self.inner.vjp[
            target, BATCH * Self.SEQ_LEN, POLICY=POLICY, mode=mode
        ](go_r, gi_r)

    # ----- Walkers / attrs (delegate to Inner) -----------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Tokenwise", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.inner.for_each_param[target, V](prefix + sep + "inner", visitor)

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Tokenwise", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.inner.for_each_state[target, V](prefix + sep + "inner", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Tokenwise", target](self.ts.target_tag)
        self.inner.zero_grad[target]()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.inner.set_attr[ATTR](value)

    def set_attr_ptr[ATTR: StaticString](
        mut self, p: UnsafePointer[Scalar[DT], MutAnyOrigin]
    ):
        self.inner.set_attr_ptr[ATTR](p)
