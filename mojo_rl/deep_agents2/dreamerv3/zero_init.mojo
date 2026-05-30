"""Zero-init the reward-predictor and critic output layers (Finding 4).

DreamerV3 paper (Robust predictions / Critic learning, p.6): *"randomly
initialized reward predictor and critic networks can result in large
predicted rewards [...] that can delay the onset of learning. We thus
initialize the output weight matrix of the reward predictor and critic to
zeros."* Without this the heads emit large biased predictions at init; on a
negative-reward task (e.g. Pendulum) the imagined λ-returns come out
optimistically positive and the actor optimizes a reward landscape that
doesn't exist, so the policy never improves.

The reward / value / slow-value heads are 1-hidden-layer MLPs (`nets.mojo`
pins the head MLP depth to 1), so the output `Linear` is Sequential child
index 3 → params `3.weight` / `3.bias`. Inside the reward `ComputeGraph`
(`RewLossGraph`) the head is node `rew`, so the names are prefixed
`rew.3.weight` / `rew.3.bias`. We zero BOTH weight and bias so the head's
twohot logits start at 0 (≈ neutral / 0 prediction over the symexp bins).

Only the OUTPUT layer is zeroed — zeroing the whole head would zero the
downstream weight and kill the hidden layers' gradient.
"""

from layout import TileTensor
from std.gpu import global_idx
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import ParamVisitor, GraphNode, Module
from mojo_rl.nn2.combinators.compute_graph import ComputeGraph


def _zero_k(dst: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int):
    """Runtime-length device zero-fill (one param slab)."""
    var i = Int(global_idx.x)
    if i < n:
        dst[i] = Scalar[DT](0.0)


@fieldwise_init
struct _ZeroOutVisitorCPU(ParamVisitor):
    """Zeros the two named output-layer params (weight + bias) on the host."""

    var wname: String
    var bname: String

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
        if name == self.wname or name == self.bname:
            var dp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
            for k in range(n_elems):
                dp[k] = Scalar[DT](0.0)


@fieldwise_init
struct _ZeroOutVisitorGPU(ParamVisitor):
    """GPU mirror — enqueues a zero kernel for the matched output params."""

    var wname: String
    var bname: String
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
        if name == self.wname or name == self.bname:
            var dp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
            var nb = (n_elems + TPB - 1) // TPB
            self.ctx.enqueue_function[_zero_k](
                dp, n_elems, grid_dim=nb, block_dim=TPB
            )


def zero_output_module[
    target: StaticString, M: Module
](mut m: M, wname: String, bname: String, ctx: Optional[DeviceContext]) raises:
    """Zero the output-layer params (`wname`/`bname`) of a Sequential head."""
    comptime if target == "cpu":
        var v = _ZeroOutVisitorCPU(wname, bname)
        m.for_each_param[target, _ZeroOutVisitorCPU](String(""), v)
    else:
        var v = _ZeroOutVisitorGPU(wname, bname, ctx.value())
        m.for_each_param[target, _ZeroOutVisitorGPU](String(""), v)


def zero_output_graph[
    target: StaticString, OUT: Int, *NODES: GraphNode
](
    mut g: ComputeGraph[OUT, *NODES],
    wname: String,
    bname: String,
    ctx: Optional[DeviceContext],
) raises:
    """Zero the output-layer params of a head wrapped in a ComputeGraph."""
    comptime if target == "cpu":
        var v = _ZeroOutVisitorCPU(wname, bname)
        g.for_each_param[target, _ZeroOutVisitorCPU](String(""), v)
    else:
        var v = _ZeroOutVisitorGPU(wname, bname, ctx.value())
        g.for_each_param[target, _ZeroOutVisitorGPU](String(""), v)
