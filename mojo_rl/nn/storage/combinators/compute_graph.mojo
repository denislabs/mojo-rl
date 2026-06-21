"""ComputeGraph[*DECLS] — typed, NAME-wired multi-input DAG (storage design).

The keystone orchestrator for agent loss/target graphs (SAC actor-loss,
target-y, twin critic). Nodes are declared by NAME with their predecessors'
NAMES baked in (`InputSlot` / `Node` / `ExternalNode`, see graph_decl.mojo);
the graph resolves every edge to a pool-slot index at COMPILE TIME — no runtime
`List[List[Int]]` edge list to hand-build and keep in sync.

```
comptime G = ComputeGraph[
    InputSlot["s", OBS],                     # slot 0 — external input
    ExternalNode["actor", ACTOR, "s"],       # slot 1 — supplied at forward
    Node["rs", RSample[ACT], "actor"],       # slot 2
    Node["concat", Concat2[OBS, ACT], "s", "act"],  # reads "s" + "act" by name
    ...
]
g.set_input["s", B](state_tensor)            # seed inputs (by COPY)
g.forward[B, target](out, ctx, actor, q1, q2)  # externals threaded in node order
var logp = g.node_output["logp"]()           # read an intermediate by name
g.vjp[B, target](grad_out, ctx, actor, q1, q2)
var ds = g.grad_input["s"]()                 # input grad by name
```

Internals (unchanged from the index-based draft — only edge resolution moved to
comptime + inputs folded into the decl list):
  - One owning `TensorPack` POOL with one slot per decl (slot `i` ↔ decl `i`);
    ALL activations live there → every edge is same-origin (§B0), no raw
    pointers (cf. legacy `GraphNode`'s out_ptr/grad_out_ptr raw-pointer model).
  - `*EXT` externals are threaded as TRACKED `mut *externals` refs (NOT stored)
    — the load-bearing GPU fix (a wildcard pointer field disables arg-
    exclusivity and miscompiles the delegated matmul). Each `ExternalNode`
    (KIND==2 / `IsExternal`) slot dispatches to `externals[ei]` by node order.
  - Backward walks reverse-topo; each node's `vjp` scatters its grad_inputs into
    `tmp`, ACCUMULATED into the fed slots' grad buffers (fan-out summation).
    Input-slot (KIND==0) grad buffers hold the final input gradient, read via
    `grad_input[NAME]`.

CPU + GPU (the leaves + the pool-seed / grad-accumulate run on device).
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
from .graph_decl import GraphDecl


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


struct ComputeGraph[*DECLS: GraphDecl](Movable & ImplicitlyDeletable):
    comptime N = Self.DECLS.size
    comptime OUT_DIM = Self.DECLS[Self.N - 1].OUT_DIM
    var children: Tuple[*Self.DECLS]
    var pool: TensorPack[Self.N]  # one slot per decl (slot i ↔ decl i)
    var gpool: TensorPack[Self.N]  # grads (zeroed + fan-out-accumulated)
    var tmp: TensorPack[3]  # grad_inputs scratch (max arity 3)
    var slot_n: List[Int]  # logical elem count per slot (forward)

    def __init__(out self):
        comptime assert Self.N >= 1, "ComputeGraph needs >= 1 node"
        comptime assert Self.DECLS[Self.N - 1].KIND > 0, (
            "ComputeGraph: the last decl must be a compute/external node"
            " (the graph output), not an InputSlot"
        )
        self.children = Tuple[*Self.DECLS]()
        self.pool = TensorPack[Self.N]()
        self.gpool = TensorPack[Self.N]()
        self.tmp = TensorPack[3]()
        self.slot_n = List[Int](length=Self.N, fill=0)

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var g = Self()
        comptime for i in range(Self.N):
            g.children[i] = Self.DECLS[i].make[target, INIT](ctx)
        return g^

    @staticmethod
    def _slot_of[nm: StaticString]() -> Int:
        """The pool-slot index of the decl named `nm` (slot ↔ decl index).
        Comptime-folded; the named-edge resolution that replaces the runtime
        edge list. Returns -1 if no decl matches (a wiring bug — would index
        out of range)."""
        var s = -1
        comptime for j in range(Self.N):
            comptime if Self.DECLS[j].NAME == nm:
                s = j
        return s

    @staticmethod
    def _ext_before[upto: Int]() -> Int:
        """Count `ExternalNode` (KIND==2) decls in `DECLS[0:upto]` — the index
        of the threaded external arg that supplies node `upto` (externals are
        passed in node order). Comptime-folded."""
        var c = 0
        comptime for j in range(upto):
            comptime if Self.DECLS[j].KIND == 2:
                c += 1
        return c

    def set_input[
        slot_name: StaticString, B: Int
    ](mut self, mut src: Tensor, ctx: Optional[DeviceContext] = None) raises:
        """Seed the named input slot's pool entry with `src` (a COPY — no cached
        pointer). Call once per input before each `forward`. CPU copies element-
        wise; GPU does a device-to-device `enqueue_copy`."""
        comptime slot = Self._slot_of[slot_name]()
        comptime assert slot >= 0, "set_input: no InputSlot named " + slot_name
        comptime assert Self.DECLS[slot].KIND == 0, (
            "set_input: '" + slot_name + "' is not an InputSlot"
        )
        var n = B * Self.DECLS[slot].OUT_DIM
        self.slot_n[slot] = n
        if ctx:
            var c = ctx.value()
            self.pool[slot].ensure_gpu(c, n)
            c.enqueue_copy(self.pool[slot].dev.value(), src.dev.value())
        else:
            self.pool[slot].ensure(n)
            for q in range(n):
                self.pool[slot].data[q] = src.data[q]

    def set_node_attr[
        NAME: StaticString, ATTR: StaticString
    ](mut self, value: Scalar[DT]):
        """Set a runtime scalar attribute `ATTR` on the named node's inner op
        (e.g. `set_node_attr["alogp", "multiplier"](alpha)`). Dispatches via the
        uniform `Module.set_attr[ATTR]` trait method (no-op on ops without that
        attr). The name-wired replacement for `graph.children[i].op.<field> = …`
        (which a runtime tuple subscript can't express — it erases to the
        existential)."""
        comptime for i in range(Self.N):
            comptime if Self.DECLS[i].NAME == NAME:
                self.children[i].set_attr[ATTR](value)

    def node_output[
        name: StaticString
    ](mut self) raises -> ref[MutAnyOrigin] Tensor:
        """The forward output of the named node — for diagnostics / reading an
        intermediate (e.g. `log_prob`). Comptime name → slot."""
        comptime slot = Self._slot_of[name]()
        comptime assert slot >= 0, "node_output: no node named " + name
        return self.pool[slot]

    def grad_input[
        name: StaticString
    ](mut self) raises -> ref[MutAnyOrigin] Tensor:
        """The accumulated gradient flowing back to the named input slot (read
        after `vjp`). Comptime name → grad-pool slot."""
        comptime slot = Self._slot_of[name]()
        comptime assert slot >= 0, "grad_input: no node named " + name
        return self.gpool[slot]

    def forward[
        B: Int,
        target: StaticString = "cpu",
        POLICY: AMPPolicy = NoAMP,
        *EXT: Module,
    ](
        mut self,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
        mut *externals: *EXT,
    ) raises:
        # `externals` supplies the `ExternalNode` (IsExternal) slots, in node
        # order, as TRACKED `mut` refs threaded from the trainer (which owns the
        # actor/critics). Load-bearing: storing them as wildcard pointers
        # miscompiles the GPU matmul (see external_ref.mojo / graph_decl.mojo).
        # Inputs are pre-seeded via set_input — KIND==0 slots are skipped here.
        comptime for i in range(Self.N):
            comptime if Self.DECLS[i].KIND == 0:
                pass  # input slot — already seeded by set_input
            else:
                self.slot_n[i] = B * Self.DECLS[i].OUT_DIM
                comptime s0 = Self._slot_of[Self.DECLS[i].IN_NAMES[0]]()
                comptime if Self.DECLS[i].KIND == 2:
                    # External slot — dispatch to the threaded module.
                    comptime ei = Self._ext_before[i]()
                    comptime if EXT[ei].ARITY == 1:
                        externals[ei].forward[target, B, POLICY=POLICY](
                            TensorRefs[EXT[ei].ARITY](self.pool[s0]),
                            self.pool[i],
                            ctx,
                        )
                    elif EXT[ei].ARITY == 2:
                        comptime s1 = Self._slot_of[Self.DECLS[i].IN_NAMES[1]]()
                        externals[ei].forward[target, B, POLICY=POLICY](
                            TensorRefs[EXT[ei].ARITY](
                                self.pool[s0], self.pool[s1]
                            ),
                            self.pool[i],
                            ctx,
                        )
                    elif EXT[ei].ARITY == 3:
                        comptime s1 = Self._slot_of[Self.DECLS[i].IN_NAMES[1]]()
                        comptime s2 = Self._slot_of[Self.DECLS[i].IN_NAMES[2]]()
                        externals[ei].forward[target, B, POLICY=POLICY](
                            TensorRefs[EXT[ei].ARITY](
                                self.pool[s0], self.pool[s1], self.pool[s2]
                            ),
                            self.pool[i],
                            ctx,
                        )
                elif Self.DECLS[i].ARITY == 1:
                    self.children[i].forward[target, B, POLICY=POLICY](
                        TensorRefs[Self.DECLS[i].ARITY](self.pool[s0]),
                        self.pool[i],
                        ctx,
                    )
                elif Self.DECLS[i].ARITY == 2:
                    comptime s1 = Self._slot_of[Self.DECLS[i].IN_NAMES[1]]()
                    self.children[i].forward[target, B, POLICY=POLICY](
                        TensorRefs[Self.DECLS[i].ARITY](
                            self.pool[s0], self.pool[s1]
                        ),
                        self.pool[i],
                        ctx,
                    )
                elif Self.DECLS[i].ARITY == 3:
                    comptime s1 = Self._slot_of[Self.DECLS[i].IN_NAMES[1]]()
                    comptime s2 = Self._slot_of[Self.DECLS[i].IN_NAMES[2]]()
                    self.children[i].forward[target, B, POLICY=POLICY](
                        TensorRefs[Self.DECLS[i].ARITY](
                            self.pool[s0], self.pool[s1], self.pool[s2]
                        ),
                        self.pool[i],
                        ctx,
                    )
        var dN = B * Self.OUT_DIM
        comptime if target == "cpu":
            out.ensure(dN)
            for q in range(dN):
                out.data[q] = self.pool[Self.N - 1].data[q]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, dN)
            c.enqueue_copy(out.dev.value(), self.pool[Self.N - 1].dev.value())

    def vjp[
        B: Int,
        target: StaticString = "cpu",
        POLICY: AMPPolicy = NoAMP,
        *EXT: Module,
    ](
        mut self,
        mut grad_out: Tensor,
        ctx: Optional[DeviceContext] = None,
        mut *externals: *EXT,
    ) raises:
        # Size + ZERO every grad slot (forward-recorded slot_n), then seed the
        # output slot's grad = grad_out.
        for k in range(Self.N):
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
                self.gpool[Self.N - 1].data[q] = grad_out.data[q]
        else:
            ctx.value().enqueue_copy(
                self.gpool[Self.N - 1].dev.value(), grad_out.dev.value()
            )
        # Reverse-topo: each node scatters grad_inputs into tmp; the graph
        # ACCUMULATES tmp into the fed slots' grad buffers (fan-out sum). Input
        # slots (KIND==0) have no vjp — their grad slot is the accumulator.
        comptime for jj in range(Self.N):
            comptime i = Self.N - 1 - jj
            comptime if Self.DECLS[i].KIND == 0:
                pass
            elif Self.DECLS[i].ARITY == 1:
                comptime a0 = B * Self.DECLS[i].IN_DIMS[0]
                comptime s0 = Self._slot_of[Self.DECLS[i].IN_NAMES[0]]()
                comptime if target == "cpu":
                    self.tmp[0].ensure(a0)
                else:
                    self.tmp[0].ensure_gpu(ctx.value(), a0)
                comptime if Self.DECLS[i].KIND == 2:
                    comptime ei = Self._ext_before[i]()
                    externals[ei].vjp[target, B, POLICY=POLICY](
                        TensorRefs[EXT[ei].ARITY](self.pool[s0]),
                        self.gpool[i],
                        TensorRefs[EXT[ei].ARITY](self.tmp[0]),
                        ctx,
                    )
                else:
                    self.children[i].vjp[target, B, POLICY=POLICY](
                        TensorRefs[Self.DECLS[i].ARITY](self.pool[s0]),
                        self.gpool[i],
                        TensorRefs[Self.DECLS[i].ARITY](self.tmp[0]),
                        ctx,
                    )
                comptime if target == "cpu":
                    for q in range(a0):
                        self.gpool[s0].data[q] += self.tmp[0].data[q]
                else:
                    var c = ctx.value()
                    c.enqueue_function[_cg_accum_kernel[a0]](
                        self.gpool[s0].lt["gpu", Layout.row_major(a0)](),
                        self.tmp[0].lt["gpu", Layout.row_major(a0)](),
                        grid_dim=(a0 + TPB - 1) // TPB,
                        block_dim=TPB,
                    )
            elif Self.DECLS[i].ARITY == 2:
                comptime a0 = B * Self.DECLS[i].IN_DIMS[0]
                comptime a1 = B * Self.DECLS[i].IN_DIMS[1]
                comptime s0 = Self._slot_of[Self.DECLS[i].IN_NAMES[0]]()
                comptime s1 = Self._slot_of[Self.DECLS[i].IN_NAMES[1]]()
                comptime if target == "cpu":
                    self.tmp[0].ensure(a0)
                    self.tmp[1].ensure(a1)
                else:
                    self.tmp[0].ensure_gpu(ctx.value(), a0)
                    self.tmp[1].ensure_gpu(ctx.value(), a1)
                comptime if Self.DECLS[i].KIND == 2:
                    comptime ei = Self._ext_before[i]()
                    externals[ei].vjp[target, B, POLICY=POLICY](
                        TensorRefs[EXT[ei].ARITY](self.pool[s0], self.pool[s1]),
                        self.gpool[i],
                        TensorRefs[EXT[ei].ARITY](self.tmp[0], self.tmp[1]),
                        ctx,
                    )
                else:
                    self.children[i].vjp[target, B, POLICY=POLICY](
                        TensorRefs[Self.DECLS[i].ARITY](
                            self.pool[s0], self.pool[s1]
                        ),
                        self.gpool[i],
                        TensorRefs[Self.DECLS[i].ARITY](
                            self.tmp[0], self.tmp[1]
                        ),
                        ctx,
                    )
                comptime if target == "cpu":
                    for q in range(a0):
                        self.gpool[s0].data[q] += self.tmp[0].data[q]
                    for q in range(a1):
                        self.gpool[s1].data[q] += self.tmp[1].data[q]
                else:
                    var c = ctx.value()
                    c.enqueue_function[_cg_accum_kernel[a0]](
                        self.gpool[s0].lt["gpu", Layout.row_major(a0)](),
                        self.tmp[0].lt["gpu", Layout.row_major(a0)](),
                        grid_dim=(a0 + TPB - 1) // TPB,
                        block_dim=TPB,
                    )
                    c.enqueue_function[_cg_accum_kernel[a1]](
                        self.gpool[s1].lt["gpu", Layout.row_major(a1)](),
                        self.tmp[1].lt["gpu", Layout.row_major(a1)](),
                        grid_dim=(a1 + TPB - 1) // TPB,
                        block_dim=TPB,
                    )
            elif Self.DECLS[i].ARITY == 3:
                comptime a0 = B * Self.DECLS[i].IN_DIMS[0]
                comptime a1 = B * Self.DECLS[i].IN_DIMS[1]
                comptime a2 = B * Self.DECLS[i].IN_DIMS[2]
                comptime s0 = Self._slot_of[Self.DECLS[i].IN_NAMES[0]]()
                comptime s1 = Self._slot_of[Self.DECLS[i].IN_NAMES[1]]()
                comptime s2 = Self._slot_of[Self.DECLS[i].IN_NAMES[2]]()
                comptime if target == "cpu":
                    self.tmp[0].ensure(a0)
                    self.tmp[1].ensure(a1)
                    self.tmp[2].ensure(a2)
                else:
                    self.tmp[0].ensure_gpu(ctx.value(), a0)
                    self.tmp[1].ensure_gpu(ctx.value(), a1)
                    self.tmp[2].ensure_gpu(ctx.value(), a2)
                comptime if Self.DECLS[i].KIND == 2:
                    comptime ei = Self._ext_before[i]()
                    externals[ei].vjp[target, B, POLICY=POLICY](
                        TensorRefs[EXT[ei].ARITY](
                            self.pool[s0], self.pool[s1], self.pool[s2]
                        ),
                        self.gpool[i],
                        TensorRefs[EXT[ei].ARITY](
                            self.tmp[0], self.tmp[1], self.tmp[2]
                        ),
                        ctx,
                    )
                else:
                    self.children[i].vjp[target, B, POLICY=POLICY](
                        TensorRefs[Self.DECLS[i].ARITY](
                            self.pool[s0], self.pool[s1], self.pool[s2]
                        ),
                        self.gpool[i],
                        TensorRefs[Self.DECLS[i].ARITY](
                            self.tmp[0], self.tmp[1], self.tmp[2]
                        ),
                        ctx,
                    )
                comptime if target == "cpu":
                    for q in range(a0):
                        self.gpool[s0].data[q] += self.tmp[0].data[q]
                    for q in range(a1):
                        self.gpool[s1].data[q] += self.tmp[1].data[q]
                    for q in range(a2):
                        self.gpool[s2].data[q] += self.tmp[2].data[q]
                else:
                    var c = ctx.value()
                    c.enqueue_function[_cg_accum_kernel[a0]](
                        self.gpool[s0].lt["gpu", Layout.row_major(a0)](),
                        self.tmp[0].lt["gpu", Layout.row_major(a0)](),
                        grid_dim=(a0 + TPB - 1) // TPB,
                        block_dim=TPB,
                    )
                    c.enqueue_function[_cg_accum_kernel[a1]](
                        self.gpool[s1].lt["gpu", Layout.row_major(a1)](),
                        self.tmp[1].lt["gpu", Layout.row_major(a1)](),
                        grid_dim=(a1 + TPB - 1) // TPB,
                        block_dim=TPB,
                    )
                    c.enqueue_function[_cg_accum_kernel[a2]](
                        self.gpool[s2].lt["gpu", Layout.row_major(a2)](),
                        self.tmp[2].lt["gpu", Layout.row_major(a2)](),
                        grid_dim=(a2 + TPB - 1) // TPB,
                        block_dim=TPB,
                    )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_param[target](
                visitor, ctx, join_name(prefix, String(Self.DECLS[i].NAME))
            )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        comptime for i in range(Self.N):
            self.children[i].for_each_state[target](
                visitor, ctx, join_name(prefix, String(Self.DECLS[i].NAME))
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
