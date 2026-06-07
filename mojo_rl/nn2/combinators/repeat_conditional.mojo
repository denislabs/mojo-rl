"""RepeatConditional[N, Inner] — chain N copies of a 2-input conditional block.

Stacks `Inner` (ARITY=2, `forward(x, c)`, dim-preserving) N times: the main
stream `x` chains block→block, while the conditioning `c` is **broadcast** to
every block (same input). On backward, grad_x chains in reverse and grad_c is
**accumulated** across all N blocks (c fans out to every layer, so its
gradient is the sum of each layer's contribution).

    x_0 = x;  x_{i+1} = Inner_i(x_i, c);  out = x_N
    grad_c = sum_i (∂Inner_i/∂c)·grad_{x_{i+1}}

The LeWM AR-predictor stack: `RepeatConditional[DEPTH,
ConditionalTransformerBlock[...]]`. Each block has its own params
(shared=False, like `Repeat`). Mirrors `Repeat`'s mid-slab reuse; adds a
single grad_c scratch + accumulation.
"""

from std.memory import alloc
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


def _rc_accum_kernel[NN: Int](
    dst: LayoutTensor[DT, Layout.row_major(NN), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(NN), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < NN:
        dst[i] = rebind[Scalar[DT]](dst[i]) + rebind[Scalar[DT]](src[i])


def _rc_zero_kernel[NN: Int](
    dst: LayoutTensor[DT, Layout.row_major(NN), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < NN:
        dst[i] = Scalar[DT](0.0)


struct RepeatConditional[N: Int, Inner: Module](Module):
    comptime ARITY: Int = 2
    comptime D = Self.Inner.OUT_DIM
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.D)
    comptime OUT_DIM = Self.Inner.OUT_DIM

    @staticmethod
    def display_label() -> String:
        return String("RepeatConditional")

    var children: List[Self.Inner]
    var mid_cpu: List[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var mid_dev: List[DeviceBuffer[DT]]
    var mid_caps: List[Int]
    # grad_c scratch (one buffer reused across the reverse pass).
    var gc_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var gc_dev: Optional[DeviceBuffer[DT]]
    var gc_cap: Int
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.N >= 1, "RepeatConditional requires N >= 1"
        comptime assert Self.Inner.ARITY == 2, (
            "RepeatConditional: Inner must be ARITY=2 (forward(x, c))"
        )
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Inner.OUT_DIM
            and Self.Inner.IN_DIMS[1] == Self.Inner.OUT_DIM
        ), "RepeatConditional: Inner must be dim-preserving with IN0==IN1==OUT"
        self.children = List[Self.Inner]()
        self.mid_cpu = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.mid_dev = List[DeviceBuffer[DT]]()
        self.mid_caps = List[Int]()
        self.gc_cpu = alloc[Scalar[DT]](1)
        self.gc_dev = None
        self.gc_cap = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "RepeatConditional: target must be 'cpu' or 'gpu'"
        )
        var r = Self()
        comptime for _i in range(Self.N):
            r.children.append(Self.Inner.make[target, INIT](ctx=ctx))
        comptime if target == "cpu":
            comptime if Self.N >= 2:
                for _ in range(Self.N - 1):
                    r.mid_cpu.append(alloc[Scalar[DT]](1))
                    r.mid_caps.append(0)
            r.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["RepeatConditional.make[target='gpu']"](ctx)
            comptime if Self.N >= 2:
                for _ in range(Self.N - 1):
                    r.mid_dev.append(ctx_v.enqueue_create_buffer[DT](1))
                    r.mid_caps.append(0)
            r.gc_dev = ctx_v.enqueue_create_buffer[DT](1)
            r.ts = TargetStorage.make_gpu(ctx_v)
        return r^

    def __del__(deinit self):
        for p in self.mid_cpu:
            p.free()
        self.gc_cpu.free()

    def _ensure_mid_cpu[i: Int](mut self, needed: Int):
        if self.mid_caps[i] < needed:
            self.mid_cpu[i].free()
            self.mid_cpu[i] = alloc[Scalar[DT]](needed)
            self.mid_caps[i] = needed

    def _ensure_mid_gpu[i: Int](mut self, needed: Int) raises:
        if self.mid_caps[i] < needed:
            self.mid_dev[i] = self.ts.ctx.value().enqueue_create_buffer[DT](
                needed
            )
            self.mid_caps[i] = needed

    def _ensure_gc(mut self, needed: Int) raises:
        if self.gc_cap < needed:
            if self.ts.ctx:
                self.gc_dev = self.ts.ctx.value().enqueue_create_buffer[DT](
                    needed
                )
            else:
                self.gc_cpu.free()
                self.gc_cpu = alloc[Scalar[DT]](needed)
            self.gc_cap = needed

    # ----- Forward ---------------------------------------------------------
    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
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
        assert_tag_for["RepeatConditional", target](self.ts.target_tag)
        comptime D = Self.D
        var x = typed_view[BATCH, D](inputs[0])
        var c = typed_view[BATCH, D](inputs[1])
        var out = typed_view_mut[BATCH, D](output)

        comptime if Self.N == 1:
            self.children[0].forward[target, BATCH, POLICY=POLICY](
                x, c, output=out
            )
        else:
            var midp = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
            comptime for k in range(Self.N - 1):
                comptime if target == "cpu":
                    self._ensure_mid_cpu[k](BATCH * D)
                    midp.append(self.mid_cpu[k])
                else:
                    self._ensure_mid_gpu[k](BATCH * D)
                    midp.append(self.mid_dev[k].unsafe_ptr())

            comptime for i in range(Self.N):
                comptime if i == 0:
                    var om = TileTensor(midp[0], row_major[BATCH, D]())
                    self.children[0].forward[target, BATCH, POLICY=POLICY](
                        x, c, output=om
                    )
                elif i == Self.N - 1:
                    var im = TileTensor(midp[Self.N - 2], row_major[BATCH, D]())
                    self.children[i].forward[target, BATCH, POLICY=POLICY](
                        im, c, output=out
                    )
                else:
                    var im = TileTensor(midp[i - 1], row_major[BATCH, D]())
                    var om = TileTensor(midp[i], row_major[BATCH, D]())
                    self.children[i].forward[target, BATCH, POLICY=POLICY](
                        im, c, output=om
                    )

    # ----- Backward --------------------------------------------------------
    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
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
        assert_tag_for["RepeatConditional", target](self.ts.target_tag)
        comptime D = Self.D
        var go = typed_view[BATCH, D](grad_output)
        var gx = typed_view_mut[BATCH, D](grad_inputs[0])
        var gc = typed_view_mut[BATCH, D](grad_inputs[1])

        comptime if Self.N == 1:
            self.children[0].vjp[target, BATCH, POLICY=POLICY, mode=mode](
                go, gx, gc
            )
            return

        # grad_c accumulator: zero it, then add each block's grad_c.
        var gc_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gc.ptr)
        comptime total = BATCH * D
        self._ensure_gc(total)
        var gc_tmp = (
            self.gc_dev.value().unsafe_ptr() if self.ts.ctx else self.gc_cpu
        )
        # Zero the grad_c accumulator (target-aware).
        comptime if target == "cpu":
            for i in range(total):
                gc_p[i] = Scalar[DT](0.0)
        else:
            comptime zlay = Layout.row_major(total)
            comptime zblocks = (total + TPB - 1) // TPB
            self.ts.ctx.value().enqueue_function[_rc_zero_kernel[total]](
                LayoutTensor[DT, zlay, MutAnyOrigin](gc_p),
                grid_dim=zblocks, block_dim=TPB,
            )

        var midp = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        comptime for k in range(Self.N - 1):
            comptime if target == "cpu":
                self._ensure_mid_cpu[k](BATCH * D)
                midp.append(self.mid_cpu[k])
            else:
                self._ensure_mid_gpu[k](BATCH * D)
                midp.append(self.mid_dev[k].unsafe_ptr())

        comptime for ri in range(Self.N):
            comptime i = Self.N - 1 - ri
            var gc_tmp_t = TileTensor(gc_tmp, row_major[BATCH, D]())
            comptime if i == Self.N - 1:
                var gim = TileTensor(midp[Self.N - 2], row_major[BATCH, D]())
                self.children[i].vjp[
                    target, BATCH, POLICY=POLICY, mode=mode
                ](go, gim, gc_tmp_t)
            elif i == 0:
                var gom = TileTensor(midp[0], row_major[BATCH, D]())
                self.children[0].vjp[
                    target, BATCH, POLICY=POLICY, mode=mode
                ](gom, gx, gc_tmp_t)
            else:
                var gom = TileTensor(midp[i], row_major[BATCH, D]())
                var gim = TileTensor(midp[i - 1], row_major[BATCH, D]())
                self.children[i].vjp[
                    target, BATCH, POLICY=POLICY, mode=mode
                ](gom, gim, gc_tmp_t)
            # grad_c += gc_tmp
            self._accum[target, total](gc_p, gc_tmp)

    def _accum[target: StaticString, NN: Int](
        mut self,
        dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
        src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime if target == "cpu":
            for i in range(NN):
                dst[i] += src[i]
        else:
            comptime lay = Layout.row_major(NN)
            comptime n_blocks = (NN + TPB - 1) // TPB
            comptime kern = _rc_accum_kernel[NN]
            self.ts.ctx.value().enqueue_function[kern](
                LayoutTensor[DT, lay, MutAnyOrigin](dst),
                LayoutTensor[DT, lay, MutAnyOrigin](src),
                grid_dim=n_blocks, block_dim=TPB,
            )

    # ----- Walkers ---------------------------------------------------------
    def for_each_param[
        target: StaticString, V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["RepeatConditional", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.children[i].for_each_param[target, V](
                prefix + sep + String(i), visitor
            )

    def for_each_state[
        target: StaticString, V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["RepeatConditional", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.children[i].for_each_state[target, V](
                prefix + sep + String(i), visitor
            )

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["RepeatConditional", target](self.ts.target_tag)
        comptime for i in range(Self.N):
            self.children[i].zero_grad[target]()
