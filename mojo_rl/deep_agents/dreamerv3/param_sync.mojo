"""Name-keyed parameter sync between two ComputeGraphs (storage-native).

The DreamerV3 trainer optimizes the RSSM core/prior params inside the
`WMCoreGraph`, but the imagination rollout runs them through a separate
`WMImagineGraph` (no obs/token path). Both graphs name the shared nodes
identically (`x0`/`x1`/`x2`/`dhin`/`h`/`gru` core + `pr0`/`pr1`/`prior`),
so their `for_each_param` walks emit matching dotted names for the shared
sub-modules.

`collect_graph_params` snapshots the trained source graph's params into a
`name → values` Dict (downloads on GPU). `apply_graph_params` then copies,
by NAME, every value that exists in the snapshot into the destination graph
(uploads on GPU) — the imagine graph becomes a read-only mirror of the
trained core each AC step. The src/dst graphs are DIFFERENT types sharing
some node names; the dict skip-on-miss handles the non-shared params.

Two functions (not one) because Mojo forbids two variadic `*DECLS` packs in
one signature — collect over `src`, apply over `dst`, threaded by the Dict.
"""

from std.collections import Dict
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.combinators.compute_graph import ComputeGraph
from mojo_rl.nn.combinators.graph_decl import GraphDecl


# Snapshot visitor: copy each param's values into the dict by name (downloads
# on GPU first so `param.data` is current).
struct _SnapshotVisitor(Movable, ParamVisitor):
    var d: Dict[String, List[Scalar[DT]]]

    def __init__(out self):
        self.d = Dict[String, List[Scalar[DT]]]()

    def take(deinit self) -> Dict[String, List[Scalar[DT]]]:
        return self.d^

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        var vals = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
        for i in range(N):
            vals[i] = param.data[i]
        self.d[name] = vals^


# Named-import visitor: for each dst param, if its name is in the snapshot,
# copy the values into the slab (uploads on GPU). Names not present are left
# at their current value (the dst-only params with no source match).
struct _NamedImportVisitor(ParamVisitor):
    var d: Dict[String, List[Scalar[DT]]]
    var missing: Int

    def __init__(out self, var d: Dict[String, List[Scalar[DT]]]):
        self.d = d^
        self.missing = 0

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if name not in self.d:
            self.missing += 1
            return
        ref vals = self.d[name]
        param.ensure(N)
        var nn = len(vals) if len(vals) < N else N
        for i in range(nn):
            param.data[i] = vals[i]
        param.n = N
        comptime if target == "gpu":
            param.upload(ctx.value())


def collect_graph_params[
    target: StaticString, *DECLS: GraphDecl
](
    mut src: ComputeGraph[*DECLS],
    ctx: Optional[DeviceContext] = None,
) raises -> Dict[String, List[Scalar[DT]]]:
    """Snapshot every param of `src` into a `name → values` Dict (downloads
    on GPU). Target-agnostic result (host values either way)."""
    var v = _SnapshotVisitor()
    src.for_each_param[target](v, ctx)
    return v^.take()


def apply_graph_params[
    target: StaticString, *DECLS: GraphDecl
](
    mut dst: ComputeGraph[*DECLS],
    read snap: Dict[String, List[Scalar[DT]]],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Copy every shared-name param value from the snapshot into `dst` (skips
    names with no match; uploads on GPU)."""
    var v = _NamedImportVisitor(snap.copy())
    dst.for_each_param[target](v, ctx)
