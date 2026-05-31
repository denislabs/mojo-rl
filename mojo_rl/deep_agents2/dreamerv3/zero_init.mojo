"""Scale the reward-predictor and critic OUTPUT layers at init (Finding 4).

DreamerV3 paper (Robust predictions / Critic learning, p.6): *"randomly
initialized reward predictor and critic networks can result in large
predicted rewards [...] that can delay the onset of learning. We thus
initialize the output weight matrix of the reward predictor and critic to
zeros."* On a negative-reward task (e.g. Pendulum) a non-neutral head makes
the imagined λ-returns optimistically positive and the actor optimizes a
reward landscape that doesn't exist.

We generalize the paper's hard zero to a SCALE applied to the Kaiming-init
output layer (`scale=0` == the paper's zero-init):
  * `scale = 0.0`  → exact zero (best for negative-reward tasks; Pendulum).
  * `scale` small (e.g. 0.1) → near-neutral but keeps a little of the random
    Kaiming asymmetry. On POSITIVE-reward tasks (CartPole) the pre-zero-init
    optimism actually helped exploration / fast solve; a small scale restores
    some of it without the large-prediction blow-up of full Kaiming.
The output is `scale·Kaiming`, so predictions start ≈ `scale`× the Kaiming
magnitude over the symexp bins.

The reward / value / slow-value heads are 1-hidden-layer MLPs (`nets.mojo`
pins the head MLP depth to 1), so the output `Linear` is Sequential child
index 3 → params `3.weight` / `3.bias`. Inside the reward `ComputeGraph`
(`RewLossGraph`) the head is node `rew`, so the names are prefixed
`rew.3.weight` / `rew.3.bias`. We scale BOTH weight and bias.

Only the OUTPUT layer is scaled — scaling the whole head toward 0 would also
shrink the downstream weight and choke the hidden layers' gradient.
"""

from layout import TileTensor
from std.gpu import global_idx
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import ParamVisitor, GraphNode, Module
from mojo_rl.nn2.combinators.compute_graph import ComputeGraph


def _scale_k(
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, scale: Scalar[DT]
):
    """Runtime-length device in-place scale (one param slab)."""
    var i = Int(global_idx.x)
    if i < n:
        dst[i] = scale * dst[i]


@fieldwise_init
struct _ScaleOutVisitorCPU(ParamVisitor):
    """Scales the two named output-layer params (weight + bias) on the host."""

    var wname: String
    var bname: String
    var scale: Scalar[DT]

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
                dp[k] = self.scale * dp[k]


@fieldwise_init
struct _ScaleOutVisitorGPU(ParamVisitor):
    """GPU mirror — enqueues a scale kernel for the matched output params."""

    var wname: String
    var bname: String
    var scale: Scalar[DT]
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
            self.ctx.enqueue_function[_scale_k](
                dp, n_elems, self.scale, grid_dim=nb, block_dim=TPB
            )


def scale_output_module[
    target: StaticString, M: Module
](
    mut m: M,
    wname: String,
    bname: String,
    scale: Scalar[DT],
    ctx: Optional[DeviceContext],
) raises:
    """Scale the output-layer params (`wname`/`bname`) of a Sequential head by
    `scale` (0.0 == exact zero-init)."""
    comptime if target == "cpu":
        var v = _ScaleOutVisitorCPU(wname, bname, scale)
        m.for_each_param[target, _ScaleOutVisitorCPU](String(""), v)
    else:
        var v = _ScaleOutVisitorGPU(wname, bname, scale, ctx.value())
        m.for_each_param[target, _ScaleOutVisitorGPU](String(""), v)


def scale_output_graph[
    target: StaticString, OUT: Int, *NODES: GraphNode
](
    mut g: ComputeGraph[OUT, *NODES],
    wname: String,
    bname: String,
    scale: Scalar[DT],
    ctx: Optional[DeviceContext],
) raises:
    """Scale the output-layer params of a head wrapped in a ComputeGraph by
    `scale` (0.0 == exact zero-init)."""
    comptime if target == "cpu":
        var v = _ScaleOutVisitorCPU(wname, bname, scale)
        g.for_each_param[target, _ScaleOutVisitorCPU](String(""), v)
    else:
        var v = _ScaleOutVisitorGPU(wname, bname, scale, ctx.value())
        g.for_each_param[target, _ScaleOutVisitorGPU](String(""), v)
