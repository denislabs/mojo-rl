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
from std.gpu.memory import AddressSpace

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import ParamVisitor, GraphNode
from mojo_rl.nn2.combinators.compute_graph import ComputeGraph


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
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
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
        var dp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        for i in range(len(self.names[])):
            if self.names[][i] == name:
                var sp = self.ptrs[][i]
                var nn = self.lens[][i] if self.lens[][i] < n_elems else n_elems
                for k in range(nn):
                    dp[k] = sp[k]
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
    """Record (name, param-ptr, n) for every param in `src`."""
    comptime assert target == "cpu", "collect_params: CPU-only (v1)"
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
) raises:
    """Copy every shared-name param value from the collected source into
    `dst` (skips names with no match)."""
    comptime assert target == "cpu", "apply_params: CPU-only (v1)"
    var cp = _CopyByNameVisitor(
        names=UnsafePointer(to=names),
        ptrs=UnsafePointer(to=ptrs),
        lens=UnsafePointer(to=lens),
    )
    dst.for_each_param[target, _CopyByNameVisitor](String(""), cp)
