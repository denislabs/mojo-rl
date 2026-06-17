"""Repeat[N, Inner, shared=False] — chain N independent copies of `Inner`.

`y = Inner_{N-1}(… Inner_1(Inner_0(x)) …)`, each copy with its OWN
parameters (shared=False). Sugar for writing the same dim-preserving
block N times in a `Sequential`; makes deep stacks (e.g. ResNet stages)
read as `Repeat[3, ResBlockConv2DBN[16, 3, 1, 32, 32], shared=False]`.

Requires `Inner.IN_DIMS[0] == Inner.OUT_DIM` (the block is chained into
itself). Internally this is exactly `Sequential` over N homogeneous
children, so it inherits the same mid-slab reuse + backward-order safety
(each child reads its own cache; the inter-child slab carries forward
activations on the way down and gradients on the way back).

`shared=True` (one weight set reused N times) is NOT supported: it would
need per-application caches or a forward recompute in backward. Pass
`shared=False` (the default). The `shared` parameter exists so call sites
read the same as the legacy `mojo_rl.nn` Repeat.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


struct Repeat[N: Int, Inner: Module, shared: Bool = False](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.OUT_DIM
    comptime D = Self.Inner.OUT_DIM  # per-boundary slab width

    var children: List[Self.Inner]
    var mid_cpu: List[List[Scalar[DT]]]
    var mid_dev: List[DeviceBuffer[DT]]
    var mid_caps: List[Int]
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.N >= 1, "Repeat requires N >= 1"
        comptime assert not Self.shared, (
            "Repeat: shared=True (shared weights) not supported in nn —"
            " use shared=False (independent copies)"
        )
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Inner.OUT_DIM
        ), "Repeat requires Inner.IN_DIMS[0] == Inner.OUT_DIM"
        self.children = List[Self.Inner]()
        self.mid_cpu = List[List[Scalar[DT]]]()
        self.mid_dev = List[DeviceBuffer[DT]]()
        self.mid_caps = List[Int]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        """Unified CPU/GPU factory — builds N independent `Inner` copies."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "Repeat: target must be 'cpu' or 'gpu'"
        var r = Self()
        comptime for _i in range(Self.N):
            r.children.append(Self.Inner.make[target, INIT](ctx=ctx))
        comptime if target == "cpu":
            comptime if Self.N >= 2:
                for _ in range(Self.N - 1):
                    r.mid_cpu.append(List[Scalar[DT]]())
                    r.mid_caps.append(0)
            r.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["Repeat.make[target='gpu']"](ctx)
            comptime if Self.N >= 2:
                for _ in range(Self.N - 1):
                    r.mid_dev.append(ctx_v.enqueue_create_buffer[DT](1))
                    r.mid_caps.append(0)
            r.ts = TargetStorage.make_gpu(ctx_v)
        return r^

    def _ensure_mid_cpu[i: Int](mut self, needed: Int):
        # List owns the storage (RAII): grow in place, no manual alloc/free.
        if self.mid_caps[i] < needed:
            self.mid_cpu[i].resize(needed, Scalar[DT](0))
            self.mid_caps[i] = needed

    def _ensure_mid_gpu[i: Int](mut self, needed: Int) raises:
        if self.mid_caps[i] < needed:
            self.mid_dev[i] = self.ts.ctx.value().enqueue_create_buffer[DT](
                needed
            )
            self.mid_caps[i] = needed

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True,
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises:
        assert_tag_for["Repeat", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if Self.N == 1:
            self.children[0].forward[target, BATCH, POLICY=POLICY](
                input, output=output_v
            )
        else:
            comptime D = Self.D
            # Resolve a MutAnyOrigin pointer per mid slab (CPU or GPU).
            var midp = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
            comptime for k in range(Self.N - 1):
                comptime if target == "cpu":
                    self._ensure_mid_cpu[k](BATCH * D)
                    midp.append(mptr(self.mid_cpu[k].unsafe_ptr()))
                else:
                    self._ensure_mid_gpu[k](BATCH * D)
                    midp.append(mptr(self.mid_dev[k].unsafe_ptr()))

            comptime for i in range(Self.N):
                comptime if i == 0:
                    var om = TileTensor(midp[0], row_major[BATCH, D]())
                    self.children[0].forward[target, BATCH, POLICY=POLICY](
                        input, output=om
                    )
                elif i == Self.N - 1:
                    var im = TileTensor(midp[Self.N - 2], row_major[BATCH, D]())
                    self.children[i].forward[target, BATCH, POLICY=POLICY](
                        im, output=output_v
                    )
                else:
                    var im = TileTensor(midp[i - 1], row_major[BATCH, D]())
                    var om = TileTensor(midp[i], row_major[BATCH, D]())
                    self.children[i].forward[target, BATCH, POLICY=POLICY](
                        im, output=om
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
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Repeat", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if Self.N == 1:
            self.children[0].vjp[
                target,
                BATCH,
                POLICY=POLICY,
                mode=mode,
            ](grad_output_v, grad_input_v)
        else:
            comptime D = Self.D
            var midp = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
            comptime for k in range(Self.N - 1):
                comptime if target == "cpu":
                    self._ensure_mid_cpu[k](BATCH * D)
                    midp.append(mptr(self.mid_cpu[k].unsafe_ptr()))
                else:
                    self._ensure_mid_gpu[k](BATCH * D)
                    midp.append(mptr(self.mid_dev[k].unsafe_ptr()))

            comptime for ri in range(Self.N):
                comptime i = Self.N - 1 - ri
                comptime if i == Self.N - 1:
                    var gim = TileTensor(
                        midp[Self.N - 2], row_major[BATCH, D]()
                    )
                    self.children[i].vjp[
                        target,
                        BATCH,
                        POLICY=POLICY,
                        mode=mode,
                    ](grad_output_v, gim)
                elif i == 0:
                    var gom = TileTensor(midp[0], row_major[BATCH, D]())
                    self.children[0].vjp[
                        target,
                        BATCH,
                        POLICY=POLICY,
                        mode=mode,
                    ](gom, grad_input_v)
                else:
                    var gom = TileTensor(midp[i], row_major[BATCH, D]())
                    var gim = TileTensor(midp[i - 1], row_major[BATCH, D]())
                    self.children[i].vjp[
                        target,
                        BATCH,
                        POLICY=POLICY,
                        mode=mode,
                    ](gom, gim)

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Repeat", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.children[i].for_each_param[target, V](
                prefix + sep + String(i), visitor
            )

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Repeat", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.children[i].for_each_state[target, V](
                prefix + sep + String(i), visitor
            )

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Repeat", target](self.ts.target_tag)
        comptime for i in range(Self.N):
            self.children[i].zero_grad[target]()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime for i in range(Self.N):
            self.children[i].set_attr[ATTR](value)
