"""Name-keyed parameter sync between two ComputeGraphs (CPU).

The DreamerV3 trainer optimizes the RSSM core/prior params inside the
`WMCoreGraph`, but the imagination rollout runs them through a separate
`WMImagineGraph` (no obs/token path). Both graphs name the shared nodes
identically (`x0`/`x1`/`x2`/`dhin`/`h`/`gru` core + `pr0`/`pr1`/`prior`),
so their `for_each_param` walks emit matching names for the shared
sub-modules. `sync_params` copies values for every NAME that exists in
both — the imagine graph becomes a read-only mirror of the trained core
each AC step (the imagine graph is never optimized).

CPU-only (param.ptr is the CPU slab); a GPU variant would enqueue a
device-to-device copy. v1 trains the WM on CPU.
"""

from layout import TileTensor
from std.gpu import global_idx
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.module import mptr
from mojo_rl.nn.core import ParamVisitor, GraphNode
from mojo_rl.nn.combinators.compute_graph import ComputeGraph


def _pcopy_k(
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
):
    """Runtime-length device→device copy (one param slab)."""
    var i = Int(global_idx.x)
    if i < n:
        dst[i] = src[i]


@fieldwise_init
struct _CollectVisitor(ParamVisitor):
    """Records (name, param-ptr, n) for every param in the source graph."""

    var names: UnsafePointer[List[String], MutAnyOrigin]
    var ptrs: UnsafePointer[
        List[UnsafePointer[Scalar[DT], MutAnyOrigin]], MutAnyOrigin
    ]
    var lens: UnsafePointer[List[Int], MutAnyOrigin]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        self.names[].append(name)
        self.ptrs[].append(
            mptr(param.ptr)
        )
        self.lens[].append(n_elems)


@fieldwise_init
struct _CopyByNameVisitor(ParamVisitor):
    """For each destination param, copies values from the matching source
    name (skips names with no source match)."""

    var names: UnsafePointer[List[String], MutAnyOrigin]
    var ptrs: UnsafePointer[
        List[UnsafePointer[Scalar[DT], MutAnyOrigin]], MutAnyOrigin
    ]
    var lens: UnsafePointer[List[Int], MutAnyOrigin]

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var dp = mptr(param.ptr)
        for i in range(len(self.names[])):
            if self.names[][i] == name:
                var sp = self.ptrs[][i]
                var nn = self.lens[][i] if self.lens[][i] < n_elems else n_elems
                for k in range(nn):
                    dp[k] = sp[k]
                return


@fieldwise_init
struct _CopyByNameVisitorGPU(ParamVisitor):
    """GPU mirror of `_CopyByNameVisitor` — enqueues a device copy kernel
    for each destination param whose name matches a collected source."""

    var names: UnsafePointer[List[String], MutAnyOrigin]
    var ptrs: UnsafePointer[
        List[UnsafePointer[Scalar[DT], MutAnyOrigin]], MutAnyOrigin
    ]
    var lens: UnsafePointer[List[Int], MutAnyOrigin]
    var ctx: DeviceContext

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var dp = mptr(param.ptr)
        for i in range(len(self.names[])):
            if self.names[][i] == name:
                var sp = self.ptrs[][i]
                var nn = self.lens[][i] if self.lens[][i] < n_elems else n_elems
                var nb = (nn + TPB - 1) // TPB
                self.ctx.enqueue_function[_pcopy_k](
                    sp, dp, nn, grid_dim=nb, block_dim=TPB
                )
                return


# Mojo forbids two variadic `*NODES` packs in one signature, so the sync
# is two calls (collect from src, apply to dst) — each over one graph.


def collect_params[
    target: StaticString, OUT: Int, *NODES: GraphNode
](
    mut src: ComputeGraph[OUT, *NODES],
    mut names: List[String],
    mut ptrs: List[UnsafePointer[Scalar[DT], MutAnyOrigin]],
    mut lens: List[Int],
) raises:
    """Record (name, param-ptr, n) for every param in `src`. Target-agnostic
    (just stores pointers; the pointer is host on CPU, device on GPU)."""
    var cv = _CollectVisitor(
        names=UnsafePointer(to=names),
        ptrs=UnsafePointer(to=ptrs),
        lens=UnsafePointer(to=lens),
    )
    src.for_each_param[target, _CollectVisitor](String(""), cv)


def apply_params[
    target: StaticString, OUT: Int, *NODES: GraphNode
](
    mut dst: ComputeGraph[OUT, *NODES],
    mut names: List[String],
    mut ptrs: List[UnsafePointer[Scalar[DT], MutAnyOrigin]],
    mut lens: List[Int],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Copy every shared-name param value from the collected source into
    `dst` (skips names with no match). CPU = host slab copy; GPU = enqueue a
    device copy kernel per matched param."""
    comptime if target == "cpu":
        var cp = _CopyByNameVisitor(
            names=UnsafePointer(to=names),
            ptrs=UnsafePointer(to=ptrs),
            lens=UnsafePointer(to=lens),
        )
        dst.for_each_param[target, _CopyByNameVisitor](String(""), cp)
    else:
        var cp = _CopyByNameVisitorGPU(
            names=UnsafePointer(to=names),
            ptrs=UnsafePointer(to=ptrs),
            lens=UnsafePointer(to=lens),
            ctx=ctx.value(),
        )
        dst.for_each_param[target, _CopyByNameVisitorGPU](String(""), cp)
