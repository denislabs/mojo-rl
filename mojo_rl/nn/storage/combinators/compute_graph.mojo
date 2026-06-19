"""ComputeGraph[NUM_IN, *NODES] — typed multi-input DAG over the storage design.

The keystone orchestrator for agent loss/target graphs (SAC actor-loss,
target-y, twin critic). Generalises the single-input `graph.mojo` prototype:

  - `NUM_IN` EXTERNAL input slots (0 .. NUM_IN-1) — the graph's leaves (state,
    action, reward, a reparam sample, …), seeded each forward.
  - `*NODES: Module` in topological order; node `i` writes pool slot
    `NUM_IN + i`. `edges[i]` lists the slots feeding node `i` (each in
    `0 .. NUM_IN+i-1`), so any node can read an external input OR an earlier
    node's output (skips / fan-out).
  - ALL activations live in ONE owning `TensorPack` pool → every edge is
    same-origin (§B0), no raw pointers (cf. legacy `GraphNode`'s out_ptr/
    grad_out_ptr raw-pointer model that the storage design replaces).
  - Backward walks reverse-topo; each node's `vjp` scatters its grad_inputs
    into `tmp`, which the graph ACCUMULATES into the fed slots' grad buffers
    (fan-out summation). `vjp` writes the external-input grads back into
    `grad_inputs` for chaining.
  - `node_output(i)` exposes a node's forward output (the `node_out_ptr`
    equivalent — for diagnostics / reading an intermediate like `min_q`).

CPU path (validates the builder API + unblocks SAC blocks; SAC has a CPU
training path). GPU graph execution (device pool-seed / grad-accumulate
kernels) is the mechanical next step — the leaves already run on GPU.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP


def _cg_accum_kernel[
    N: Int
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    """dst[i] += src[i] — fan-out grad accumulation on device."""
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](dst[i]) + rebind[Scalar[DT]](src[i])


struct ComputeGraph[NUM_IN: Int, *NODES: Module](Movable & ImplicitlyDeletable):
    comptime N = Self.NODES.size
    comptime NSLOT = Self.NUM_IN + Self.N
    comptime OUT_DIM = Self.NODES[Self.N - 1].OUT_DIM
    var children: Tuple[*Self.NODES]
    var pool: TensorPack[Self.NSLOT]  # 0..NUM_IN-1 = inputs, then node outs
    var gpool: TensorPack[Self.NSLOT]  # grads (zeroed + fan-out-accumulated)
    var tmp: TensorPack[2]  # grad_inputs scratch (max arity 2)
    var slot_n: List[Int]  # logical elem count per slot (forward)

    def __init__(out self):
        comptime assert Self.N >= 1, "ComputeGraph needs >= 1 node"
        comptime assert Self.NUM_IN >= 1, "ComputeGraph needs >= 1 input"
        self.children = Tuple[*Self.NODES]()
        self.pool = TensorPack[Self.NSLOT]()
        self.gpool = TensorPack[Self.NSLOT]()
        self.tmp = TensorPack[2]()
        self.slot_n = List[Int](length=Self.NSLOT, fill=0)

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var g = Self()
        comptime for i in range(Self.N):
            g.children[i] = Self.NODES[i].make[target, INIT](ctx)
        return g^

    def node_output(mut self, node_i: Int) raises -> ref[MutAnyOrigin] Tensor:
        """The forward output of node `node_i` (pool slot NUM_IN+node_i) — the
        storage `node_out_ptr` equivalent for diagnostics / intermediate reads.
        """
        return self.pool[Self.NUM_IN + node_i]

    def forward[
        B: Int, target: StaticString = "cpu", POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        edges: List[List[Int]],
        mut inputs: TensorPack[Self.NUM_IN],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # Seed external input slots (device-to-device copy on GPU).
        for k in range(Self.NUM_IN):
            var nk = inputs[k].n
            self.slot_n[k] = nk
            comptime if target == "cpu":
                self.pool[k].ensure(nk)
                for q in range(nk):
                    self.pool[k].data[q] = inputs[k].data[q]
            else:
                var c = ctx.value()
                self.pool[k].ensure_gpu(c, nk)
                c.enqueue_copy(self.pool[k].dev.value(), inputs[k].dev.value())
        # Topological forward; gather each node's inputs from the pool by slot.
        comptime for i in range(Self.N):
            self.slot_n[Self.NUM_IN + i] = B * Self.NODES[i].OUT_DIM
            comptime if Self.NODES[i].ARITY == 1:
                self.children[i].forward[target, B, POLICY=POLICY](
                    TensorRefs[Self.NODES[i].ARITY](self.pool[edges[i][0]]),
                    self.pool[Self.NUM_IN + i],
                    ctx,
                )
            elif Self.NODES[i].ARITY == 2:
                self.children[i].forward[target, B, POLICY=POLICY](
                    TensorRefs[Self.NODES[i].ARITY](
                        self.pool[edges[i][0]], self.pool[edges[i][1]]
                    ),
                    self.pool[Self.NUM_IN + i],
                    ctx,
                )
        var dN = B * Self.OUT_DIM
        comptime if target == "cpu":
            out.ensure(dN)
            for q in range(dN):
                out.data[q] = self.pool[Self.NSLOT - 1].data[q]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, dN)
            c.enqueue_copy(
                out.dev.value(), self.pool[Self.NSLOT - 1].dev.value()
            )

    def vjp[
        B: Int, target: StaticString = "cpu", POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        edges: List[List[Int]],
        mut grad_out: Tensor,
        mut grad_inputs: TensorPack[Self.NUM_IN],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # Size + ZERO every grad slot (forward-recorded slot_n), then seed the
        # output slot's grad = grad_out.
        for k in range(Self.NSLOT):
            var dk = self.slot_n[k]
            comptime if target == "cpu":
                self.gpool[k].ensure(dk)
                for q in range(dk):
                    self.gpool[k].data[q] = 0
            else:
                self.gpool[k].ensure_gpu(ctx.value(), dk)
                self.gpool[k].dev.value().enqueue_fill(Scalar[DT](0))
        var dN = B * Self.OUT_DIM
        comptime if target == "cpu":
            for q in range(dN):
                self.gpool[Self.NSLOT - 1].data[q] = grad_out.data[q]
        else:
            ctx.value().enqueue_copy(
                self.gpool[Self.NSLOT - 1].dev.value(), grad_out.dev.value()
            )
        # Reverse-topo: each node scatters grad_inputs into tmp; the graph
        # ACCUMULATES tmp into the fed slots' grad buffers (fan-out sum).
        comptime for jj in range(Self.N):
            comptime i = Self.N - 1 - jj
            comptime if Self.NODES[i].ARITY == 1:
                comptime a0 = B * Self.NODES[i].IN_DIMS[0]
                comptime if target == "cpu":
                    self.tmp[0].ensure(a0)
                else:
                    self.tmp[0].ensure_gpu(ctx.value(), a0)
                self.children[i].vjp[target, B, POLICY=POLICY](
                    TensorRefs[Self.NODES[i].ARITY](self.pool[edges[i][0]]),
                    self.gpool[Self.NUM_IN + i],
                    TensorRefs[Self.NODES[i].ARITY](self.tmp[0]),
                    ctx,
                )
                var e0 = edges[i][0]
                comptime if target == "cpu":
                    for q in range(a0):
                        self.gpool[e0].data[q] += self.tmp[0].data[q]
                else:
                    var c = ctx.value()
                    c.enqueue_function[_cg_accum_kernel[a0]](
                        self.gpool[e0].lt["gpu", Layout.row_major(a0)](),
                        self.tmp[0].lt["gpu", Layout.row_major(a0)](),
                        grid_dim=(a0 + TPB - 1) // TPB,
                        block_dim=TPB,
                    )
            elif Self.NODES[i].ARITY == 2:
                comptime a0 = B * Self.NODES[i].IN_DIMS[0]
                comptime a1 = B * Self.NODES[i].IN_DIMS[1]
                comptime if target == "cpu":
                    self.tmp[0].ensure(a0)
                    self.tmp[1].ensure(a1)
                else:
                    self.tmp[0].ensure_gpu(ctx.value(), a0)
                    self.tmp[1].ensure_gpu(ctx.value(), a1)
                self.children[i].vjp[target, B, POLICY=POLICY](
                    TensorRefs[Self.NODES[i].ARITY](
                        self.pool[edges[i][0]], self.pool[edges[i][1]]
                    ),
                    self.gpool[Self.NUM_IN + i],
                    TensorRefs[Self.NODES[i].ARITY](self.tmp[0], self.tmp[1]),
                    ctx,
                )
                var e0 = edges[i][0]
                var e1 = edges[i][1]
                comptime if target == "cpu":
                    for q in range(a0):
                        self.gpool[e0].data[q] += self.tmp[0].data[q]
                    for q in range(a1):
                        self.gpool[e1].data[q] += self.tmp[1].data[q]
                else:
                    var c = ctx.value()
                    c.enqueue_function[_cg_accum_kernel[a0]](
                        self.gpool[e0].lt["gpu", Layout.row_major(a0)](),
                        self.tmp[0].lt["gpu", Layout.row_major(a0)](),
                        grid_dim=(a0 + TPB - 1) // TPB,
                        block_dim=TPB,
                    )
                    c.enqueue_function[_cg_accum_kernel[a1]](
                        self.gpool[e1].lt["gpu", Layout.row_major(a1)](),
                        self.tmp[1].lt["gpu", Layout.row_major(a1)](),
                        grid_dim=(a1 + TPB - 1) // TPB,
                        block_dim=TPB,
                    )
        # Write external-input grads back (slots 0..NUM_IN-1) for chaining.
        for k in range(Self.NUM_IN):
            var dk = self.slot_n[k]
            comptime if target == "cpu":
                grad_inputs[k].ensure(dk)
                for q in range(dk):
                    grad_inputs[k].data[q] = self.gpool[k].data[q]
            else:
                var c = ctx.value()
                grad_inputs[k].ensure_gpu(c, dk)
                c.enqueue_copy(
                    grad_inputs[k].dev.value(), self.gpool[k].dev.value()
                )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_param[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_state[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        comptime for i in range(Self.N):
            self.children[i].zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Soft-update every node toward `src`'s matching node (target ← online).
        """
        comptime for i in range(Self.N):
            self.children[i].polyak_from[target](src.children[i], tau, ctx)
