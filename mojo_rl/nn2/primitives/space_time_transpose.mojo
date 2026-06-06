"""SpaceTimeTranspose[T, S, D] — swap the time and space axes of a token grid.

Dreamer 4 factorizes attention into space layers (attend over S tokens per
frame) and time layers (attend over T frames per token). The two need
different axis groupings, so between them the (T, S) grid of D-vectors is
transposed. This leaf reinterprets each sample's `T*S*D` slab as a row-major
`(T, S, D)` tensor and writes `(S, T, D)`:

    out[b, (s*T + t)*D + d] = in[b, (t*S + s)*D + d]

IN_DIM == OUT_DIM == T*S*D; param-free, cache-free. Backward is the inverse
permutation (the same map with T↔S swapped), so it is self-inverse:

    grad_in[b, (t*S + s)*D + d] = grad_out[b, (s*T + t)*D + d]

This is the block-valued analogue of `Transpose2D` (which transposes scalars,
not D-blocks). Used to drive the time-attention layer with effective batch
`B*S` (or `B*L` for latents-only) and sequence length `T`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


def _stt_kernel[
    BATCH: Int, T: Int, S: Int, D: Int, INVERSE: Bool
](
    src: LayoutTensor[DT, Layout.row_major(BATCH, T * S * D), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, T * S * D), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * T * S * D
    if gid >= total:
        return
    var b = gid // (T * S * D)
    var rem = gid % (T * S * D)
    var d = rem % D
    var pos = rem // D                # destination grid position
    comptime if INVERSE:
        # dst grid is (T, S): pos = t*S + s ; read from src (S, T) at s*T + t
        var t = pos // S
        var s = pos % S
        dst[b, gid % (T * S * D)] = rebind[Scalar[DT]](
            src[b, (s * T + t) * D + d]
        )
    else:
        # dst grid is (S, T): pos = s*T + t ; read from src (T, S) at t*S + s
        var s = pos // T
        var t = pos % T
        dst[b, gid % (T * S * D)] = rebind[Scalar[DT]](
            src[b, (t * S + s) * D + d]
        )


struct SpaceTimeTranspose[T: Int, S: Int, D: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.T * Self.S * Self.D)
    comptime OUT_DIM = Self.T * Self.S * Self.D

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SpaceTimeTranspose: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("SpaceTimeTranspose.make[target='gpu']: ctx required")
            m.ts = TargetStorage.make_gpu(ctx.value())
        return m^

    @staticmethod
    def display_label() -> String:
        return String("SpaceTimeTranspose")

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
        assert_tag_for["SpaceTimeTranspose", target](self.ts.target_tag)
        var inp = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)
        comptime if target == "cpu":
            for b in range(BATCH):
                for t in range(Self.T):
                    for s in range(Self.S):
                        for d in range(Self.D):
                            out[b, (s * Self.T + t) * Self.D + d] = inp[
                                b, (t * Self.S + s) * Self.D + d
                            ]
        else:
            self._run_gpu[BATCH, False](inp, out)

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
        assert_tag_for["SpaceTimeTranspose", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])
        comptime if target == "cpu":
            # inverse permutation: grad_in[(t,s)] = grad_out[(s,t)]
            for b in range(BATCH):
                for t in range(Self.T):
                    for s in range(Self.S):
                        for d in range(Self.D):
                            gi[b, (t * Self.S + s) * Self.D + d] = go[
                                b, (s * Self.T + t) * Self.D + d
                            ]
        else:
            self._run_gpu[BATCH, True](go, gi)

    def _run_gpu[
        BATCH: Int, INVERSE: Bool
    ](
        mut self,
        src: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        dst: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime lay = Layout.row_major(BATCH, Self.T * Self.S * Self.D)
        var src_lt = LayoutTensor[DT, lay, MutAnyOrigin](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](src.ptr)
        )
        var dst_lt = LayoutTensor[DT, lay, MutAnyOrigin](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dst.ptr)
        )
        comptime total = BATCH * Self.T * Self.S * Self.D
        comptime n_blocks = (total + TPB - 1) // TPB
        comptime kernel = _stt_kernel[BATCH, Self.T, Self.S, Self.D, INVERSE]
        self.ts.ctx.value().enqueue_function[kernel](
            src_lt, dst_lt, grid_dim=n_blocks, block_dim=TPB
        )
