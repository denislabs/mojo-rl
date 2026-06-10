"""ComputeGraph — name-based DAG over GraphNode variadic.

Builds a named DAG by composing `InputSlot` / `Node` / `ExternalNode`
wrappers in topological order. Each node carries `NAME` + predecessor
names; the graph resolves names at compile time via a `comptime for`
double-loop. `Node.KIND` (= `Op.ARITY`) drives forward / backward
dispatch — unary nodes ignore `IN1_NAME`, binary nodes read both.

```mojo
comptime SACActorGraph = ComputeGraph[
    1,                                          # OUT_DIM
    InputSlot["latent",    LATENT_DIM],         # one external input
    InputSlot["action",    ACT_DIM],            # another external input
    Node["enc",       Encoder,    "latent"],            # ARITY=1
    Node["q1",        Critic,     "enc", "action"],     # ARITY=2
    Node["loss",      MSEHead,    "q1"],                # ARITY=1
]
```

Each external input is an `InputSlot[NAME, DIM]` node. The graph has no
fixed `IN_DIM`; per-slot dims live on the slot type. The last node in
`*NODES` is the graph's external output.

Caller API:
  g.set_input["latent", BATCH](latent_tile)
  g.set_input["action", BATCH](action_tile)
  g.forward[target, BATCH](output_tile)
  g.vjp[target, BATCH](grad_output_tile)
  var grad_latent_ptr = g.grad_input_ptr["latent"]()

Memory: every node (slot or compute) owns its own `grad_out_buf`.
InputSlot.grad_out_buf is the gradient accumulator that the caller
reads via `grad_input_ptr[NAME]` after backward. There is no separate
external `_grad_input_buf` on the graph — slots own theirs.

Forward (topo order):
  - comptime for each node i:
      - if KIND == 0 (InputSlot): skip (out_ptr is set externally).
      - else: resolve IN0_NAME / IN1_NAME via name lookup, call
        node_i.forward_via(in0_ptr, in1_ptr).
  - copy `nodes[N-1].out_ptr` → `output_t`.

Backward (reverse topo):
  - zero all `grad_out_buf`s (including slot accumulators).
  - copy `grad_output_t` → `nodes[N-1].grad_out_buf`.
  - comptime for each node i in reverse:
      - if KIND == 0: skip (grad_out_buf already holds the answer).
      - else: vjp_via, then scatter-add grad_in0/1_buf into the
        predecessor (slot or compute) named by IN0_NAME / IN1_NAME.

Fan-out is handled by `+=` accumulation: when one producer feeds two
consumers, each consumer's grad_in_* writes scatter-add into the same
producer.grad_out_buf, naturally summing the gradient contributions.
Slot consumers contribute the same way.

GPU path mirrors the CPU bodies via `_forward_gpu` / `_backward_gpu`;
scatter-add and zero are GPU kernels. Pointer caching on all nodes'
out / grad_out / grad_in fields + SIMD inter-node bookkeeping helpers
below.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, CPU_SIMD_W, TPB
from ..core.module import mptr
from ..core import (
    GraphNode,
    Module,
    ParamVisitor,
    GraphVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
)
from ..core.target_tag import TARGET_GPU
from ..core.target_storage import (
    TargetStorage, assert_tag_for, ensure_gpu_buffer,
)


# ──────────────────────────────────────────────────────────────────────
# CPU SIMD helpers — zero / copy / scatter-add on flat float buffers.
#
# Used to vectorize the inter-node bookkeeping loops in `_forward_cpu`
# and `_backward_cpu` (scatter-add into predecessor grad_out_bufs, zero
# init, last-node output copy). Mojo nightly does not autovectorize
# scalar `ptr[i]` loops — manual `while i + W <= n: ptr.load[width=W]`
# is the project's standard idiom (see Adam/AdamW/MSE).
# ──────────────────────────────────────────────────────────────────────


@always_inline
def _zero_cpu(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int):
    var zero_v = SIMD[DT, CPU_SIMD_W](Scalar[DT](0.0))
    var i = 0
    while i + CPU_SIMD_W <= n:
        p.store(i, zero_v)
        i += CPU_SIMD_W
    while i < n:
        p[i] = Scalar[DT](0.0)
        i += 1


@always_inline
def _copy_cpu(
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
):
    var i = 0
    while i + CPU_SIMD_W <= n:
        dst.store(i, src.load[width=CPU_SIMD_W](i))
        i += CPU_SIMD_W
    while i < n:
        dst[i] = src[i]
        i += 1


@always_inline
def _scatter_add_cpu(
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
):
    var i = 0
    while i + CPU_SIMD_W <= n:
        var d = dst.load[width=CPU_SIMD_W](i)
        var s = src.load[width=CPU_SIMD_W](i)
        dst.store(i, d + s)
        i += CPU_SIMD_W
    while i < n:
        dst[i] += src[i]
        i += 1


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — buffer-wide zero / scatter-add / copy. Used to implement
# the inter-node wiring in `_forward_gpu` / `_backward_gpu`. Free
# module-level functions so `enqueue_function` can bind them.
# ──────────────────────────────────────────────────────────────────────


def _zero_kernel[N: Int](
    buf: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        buf[idx] = Scalar[DT](0.0)


def _add_kernel[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](dst[idx]) + rebind[Scalar[DT]](src[idx])


def _copy_kernel[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](src[idx])


def _enqueue_zero[N: Int](
    ctx: DeviceContext,
    p: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    var lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)
    comptime n_blocks = (N + TPB - 1) // TPB
    comptime kernel = _zero_kernel[N]
    ctx.enqueue_function[kernel](lt, grid_dim=n_blocks, block_dim=TPB)


def _enqueue_add[N: Int](
    ctx: DeviceContext,
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    var dst_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](dst)
    var src_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](src)
    comptime n_blocks = (N + TPB - 1) // TPB
    comptime kernel = _add_kernel[N]
    ctx.enqueue_function[kernel](
        dst_lt, src_lt, grid_dim=n_blocks, block_dim=TPB,
    )


def _enqueue_copy[N: Int](
    ctx: DeviceContext,
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    var dst_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](dst)
    var src_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](src)
    comptime n_blocks = (N + TPB - 1) // TPB
    comptime kernel = _copy_kernel[N]
    ctx.enqueue_function[kernel](
        dst_lt, src_lt, grid_dim=n_blocks, block_dim=TPB,
    )


# ──────────────────────────────────────────────────────────────────────
# ComputeGraph
# ──────────────────────────────────────────────────────────────────────


struct ComputeGraph[
    OUT_DIM_: Int,
    *NODES: GraphNode,
](Movable & ImplicitlyDestructible):
    comptime OUT_DIM = Self.OUT_DIM_
    comptime N = Self.NODES.size

    var nodes: Tuple[*Self.NODES]
    var ts: TargetStorage

    def __init__(out self):
        """Defaultable form — empty placeholders, tag=UNINIT."""
        comptime assert Self.N >= 1, "ComputeGraph requires at least one node"
        comptime assert Self.NODES[Self.N - 1].OUT_DIM == Self.OUT_DIM, (
            "ComputeGraph: last node OUT_DIM must equal graph OUT_DIM"
        )
        comptime assert Self.NODES[Self.N - 1].KIND > 0, (
            "ComputeGraph: last node must be a compute node (KIND>0), "
            "not an InputSlot"
        )
        self.nodes = Tuple[*Self.NODES]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory — recurses via
        `NODES[i].make_via[target, INIT](ctx=ctx)`. `ctx=None` on CPU;
        required on GPU (node ctors raise if missing)."""
        comptime assert target == "cpu" or target == "gpu", (
            "ComputeGraph: target must be 'cpu' or 'gpu'"
        )
        var g = Self()
        comptime for i in range(Self.N):
            g.nodes[i] = Self.NODES[i].make_via[target, INIT](ctx=ctx)
        comptime if target == "cpu":
            g.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("ComputeGraph.make[target='gpu']: ctx required")
            g.ts = TargetStorage.make_gpu(ctx.value())
        return g^

    def _ensure_all_buffers[BATCH: Int](mut self) raises:
        comptime for i in range(Self.N):
            self.nodes[i].ensure_buffers_via[BATCH]()

    # ──────────────────────────────────────────────────────────────────
    # Block B — multi-external-input plumbing.
    #
    # `set_input[NAME, BATCH](tile)` caches `tile.ptr` inside the named
    # InputSlot. Must be called once per slot before each `forward` (the
    # slot's _out_ptr is what the graph's forward loop reads for
    # consumers that reference NAME as IN0/IN1). The cached pointer
    # survives across calls if the caller's tile pointer is stable —
    # so for in-place loops over the same scratch buffer, set_input can
    # be called once at setup.
    #
    # `grad_input_ptr[NAME]()` returns the slot's grad accumulator
    # pointer (its _grad_out_buf). Stable after the first
    # `ensure_buffers_via[BATCH]` of that slot (which fires inside
    # `forward` / `backward`).
    # ──────────────────────────────────────────────────────────────────

    def set_input[
        slot_name: StaticString,
        BATCH: Int,
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises:
        comptime assert input.flat_rank == 2, "set_input: tile must be rank-2"
        # Note: if `slot_name` matches no InputSlot, the comptime loop is
        # a no-op (silent) and the slot's _out_ptr stays at the previous
        # value (null on first call). Caller will see a null deref in the
        # next forward call — debug-only failure mode; not worth the
        # comptime mutable-var workaround.
        #
        # We don't call _ensure_all_buffers here — forward and backward
        # do it. set_input just caches a pointer.
        var p = mptr(input.ptr)
        comptime for i in range(Self.N):
            comptime if (
                Self.NODES[i].KIND == 0 and Self.NODES[i].NAME == slot_name
            ):
                self.nodes[i].set_input_via(p)

    # ──────────────────────────────────────────────────────────────────
    # External-Module dispatch.
    #
    # `set_external[NAME](mut module)` binds an externally-owned Module
    # instance to a named ExternalNode (any arity). The node holds an
    # `UnsafePointer[M, MutAnyOrigin]`; per-call `forward_via` / `vjp_via`
    # derefs and calls `Module.forward` / `Module.vjp` on the supplied
    # instance.
    #
    # Caller MUST keep the bound `module` instance alive (and in the
    # same address) across subsequent `forward`/`backward` calls. The
    # typical pattern is to bind once during trainer setup where the
    # trainer owns the field, then `set_external` is a single pointer
    # store per re-bind.
    # ──────────────────────────────────────────────────────────────────

    def set_external[
        ext_name: StaticString,
        M: Module,
    ](mut self, mut module: M) raises:
        """Bind an external Module to the ExternalNode named `ext_name`.

        The matching node must have been declared as
        `ExternalNode[ext_name, M, ...]` — `M` here is the concrete
        Module type, deduced at the call site. No-op if no node matches.
        The pointer is type-erased to `UnsafePointer[Scalar[DT]]` at the
        trait surface (so it can flow through the GraphNode trait method),
        and rebound to `UnsafePointer[Self.M]` inside the receiving node.
        Works for both ARITY=1 and ARITY=2 modules.
        """
        var typed_ptr = UnsafePointer[M, MutAnyOrigin](to=module)
        var erased_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](typed_ptr)
        comptime for i in range(Self.N):
            comptime if Self.NODES[i].NAME == ext_name:
                self.nodes[i].set_external_via(erased_ptr)

    def node_out_ptr[
        node_name: StaticString,
    ](ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Returns the named node's output buffer pointer.

        Stable after the first `ensure_buffers_via[BATCH]` of that node
        (which fires inside `forward` / `backward`). Raises (via
        `.value()`) if `node_name` matches no node in the graph. Used by
        callers that need access to an intermediate node's output value
        — e.g. the SAC actor loss reads `log_prob`'s out_ptr to compute
        its mean for the α optimizer.
        """
        var out_p: Optional[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ] = None
        comptime for i in range(Self.N):
            comptime if Self.NODES[i].NAME == node_name:
                out_p = self.nodes[i].out_ptr_via()
        return out_p.value()

    def grad_input_ptr[
        slot_name: StaticString,
    ](ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Returns the slot's grad_out_buf pointer.

        The buffer holds the accumulated gradient flowing back to this
        input after `backward` has run. Sized [BATCH, slot.OUT_DIM].
        Stable across the lifetime of the graph (resolved once in
        `ensure_buffers_via`). Raises (via `.value()`) if `slot_name`
        matches no InputSlot in the graph.
        """
        var out_p: Optional[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ] = None
        comptime for i in range(Self.N):
            comptime if (
                Self.NODES[i].KIND == 0 and Self.NODES[i].NAME == slot_name
            ):
                out_p = self.nodes[i].grad_out_ptr_via()
        return out_p.value()

    # ──────────────────────────────────────────────────────────────────
    # Per-call node attribute mutation. Resolves the target node by
    # name at comptime; the named node's `op` must have a struct field
    # matching `ATTR` (e.g. `Scale.multiplier`). Replaces the hard-coded
    # `self.nodes[5].op.multiplier = alpha` pattern in SACActorLoss.
    # ──────────────────────────────────────────────────────────────────

    def set_node_attr[
        NAME: StaticString, ATTR: StaticString,
    ](mut self, value: Scalar[DT]):
        """Set runtime attribute `ATTR` on node `NAME`'s inner op.

        Dispatches via the uniform `GraphNode.set_op_attr_via[ATTR]`
        trait method (InputSlot's default no-op, Node forwards to
        `self.op.set_attr[ATTR](value)`). The matched op
        must implement `Module.set_attr` for the given `ATTR` — e.g.
        `Scale.set_attr["multiplier"]` mutates `self.multiplier`.
        Unrecognised `ATTR` values are silently no-ops on the op.

        Replaces the hard-coded `self.nodes[5].op.multiplier = value`
        index pattern with a name-resolved comptime lookup.
        """
        comptime for i in range(Self.N):
            comptime if Self.NODES[i].NAME == NAME:
                self.nodes[i].set_op_attr_via[ATTR](value)

    def set_node_attr_ptr[
        NAME: StaticString, ATTR: StaticString,
    ](mut self, p: UnsafePointer[Scalar[DT], MutAnyOrigin]):
        """Bind a device-resident attribute source on node `NAME`'s op.

        Pointer variant of `set_node_attr`: instead of baking a host
        scalar into the op's field, points the op's `ATTR` at a device
        buffer (e.g. SAC's on-device α buffer → a `Scale` node's
        `multiplier_ptr`). Dispatches via `GraphNode.set_op_attr_ptr_via`
        (InputSlot/ExternalNode no-op, Node forwards to
        `self.op.set_attr_ptr[ATTR](p)`). One-time wiring at make — the
        pointer is stable for the buffer's lifetime, so no per-step host
        work and CUDA-graph capturable. No-op if `NAME` matches no node."""
        comptime for i in range(Self.N):
            comptime if Self.NODES[i].NAME == NAME:
                self.nodes[i].set_op_attr_ptr_via[ATTR](p)

    # ──────────────────────────────────────────────────────────────────
    # Forward — topological walk, comptime name resolution.
    # ──────────────────────────────────────────────────────────────────

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert output.flat_rank == 2, "output must be rank-2"
        assert_tag_for["ComputeGraph", target](self.ts.target_tag)
        self._ensure_all_buffers[BATCH]()

        comptime if target == "cpu":
            _forward_cpu[
                target, BATCH, POLICY=POLICY,
            ](self, output)
        else:
            _forward_gpu[
                target, BATCH, POLICY=POLICY,
            ](self, output)

    # ──────────────────────────────────────────────────────────────────
    # Backward — reverse topo + scatter-add.
    #
    # The `mode` comptime param mirrors `Module.vjp[mode]` (audit
    # Follow-up #7) for parity with the slim trait, but for a graph it's
    # not a meaningful distinction: each node decides what to accumulate
    # on its own. Callers that want frozen-params behaviour should wrap
    # individual sub-nets in `StopGradParams`. `mode` is accepted and
    # forwarded to every child so leaves can gate their own param-grad
    # work uniformly.
    # ──────────────────────────────────────────────────────────────────

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["ComputeGraph", target](self.ts.target_tag)
        self._ensure_all_buffers[BATCH]()

        comptime if target == "cpu":
            _backward_cpu[
                target, BATCH, POLICY=POLICY,
            ](self, grad_output)
        else:
            _backward_gpu[
                target, BATCH, POLICY=POLICY,
            ](self, grad_output)

    # ──────────────────────────────────────────────────────────────────
    # for_each_param — recurse with `node_name.` prefix.
    # ──────────────────────────────────────────────────────────────────

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        assert_tag_for["ComputeGraph", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.nodes[i].for_each_param_via[target, V](
                prefix + sep + String(Self.NODES[i].NAME),
                visitor,
            )

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        """Walk every node's `State` fields with `node_name.` prefixes
        (mirrors `for_each_param`). Was a no-op until 2026-06-10 — LeWM
        puts BatchNorm1D inside graph nodes (encoder projector, PredProj),
        whose running stats are State and must be exportable for
        eval-mode inference (planning) and name-keyed predictor sync."""
        assert_tag_for["ComputeGraph", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.nodes[i].for_each_state_via[target, V](
                prefix + sep + String(Self.NODES[i].NAME),
                visitor,
            )

    # ──────────────────────────────────────────────────────────────────
    # describe — topology walk into a pluggable GraphVisitor sink.
    #
    # Pure comptime-metadata walk (no runtime node state touched), so it
    # works on a default-constructed graph — no `make` / buffers needed.
    # Mirrors the `comptime for i ... comptime for k` shape of the
    # forward/backward bodies, but emits node + edge events instead of
    # running kernels. See `combinators/graph_export.mojo` for exporters.
    # ──────────────────────────────────────────────────────────────────

    def describe[
        V: GraphVisitor,
    ](self, mut visitor: V, graph_name: String = "") raises:
        visitor.begin(graph_name, Self.N)
        comptime for i in range(Self.N):
            comptime kind = Self.NODES[i].KIND
            var name = String(Self.NODES[i].NAME)
            visitor.node(
                i,
                name,
                Self.NODES[i].display_label_via(),
                kind,
                Self.NODES[i].OUT_DIM,
            )
            # Container nodes (Sequential & its aliases) expand into one
            # inner step per child.
            var steps = Self.NODES[i].display_steps_via()
            for s in range(len(steps)):
                visitor.node_inner(
                    name, s, steps[s].label, steps[s].out_dim,
                )
            comptime for k in range(kind):
                visitor.edge(
                    name,
                    String(Self.NODES[i].IN_NAMES[k]),
                    k,
                    Self.NODES[i].IN_DIMS[k],
                )
        visitor.end()


# ──────────────────────────────────────────────────────────────────────
# Free-function forward / backward bodies (so the comptime double loops
# don't blow up the inliner the way an inline body would — same lesson
# as LeWM trainer extraction in 2026-05-14).
# ──────────────────────────────────────────────────────────────────────


def _forward_cpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    OUT_DIM_: Int,
    *NODES: GraphNode,
](
    mut g: ComputeGraph[OUT_DIM_, *NODES],
    mut output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
) raises:
    comptime N = NODES.size

    # Seed sentinel: a valid pointer (node 0's out_ptr) used purely as
    # InlineArray fill; every slot is overwritten in the loop below.
    var seed_ptr = g.nodes[0].out_ptr_via()

    comptime for i in range(N):
        comptime kind = NODES[i].KIND
        # InputSlot (KIND=0) has no compute and its out_ptr was set
        # externally via `set_input`.
        comptime if kind > 0:
            # I.2.6.k — InlineArray sized at per-node KIND (no padding).
            # Single arg to forward_via; wrapper indexes in_ptrs[k] for
            # k in [0, ARITY). KIND cap dropped on the dispatch layer.
            var ptrs = InlineArray[
                UnsafePointer[Scalar[DT], MutAnyOrigin], kind,
            ](fill=seed_ptr)
            comptime for k in range(kind):
                comptime src_k = NODES[i].IN_NAMES[k]
                comptime for j in range(N):
                    comptime if NODES[j].NAME == src_k:
                        ptrs[k] = g.nodes[j].out_ptr_via()
            g.nodes[i].forward_via[target, BATCH, POLICY=POLICY](ptrs)

    # Copy last node's out_buf into the external output.
    comptime LAST_OUT_DIM = NODES[N - 1].OUT_DIM
    var out_p = mptr(output.ptr)
    var last_out_ptr = g.nodes[N - 1].out_ptr_via()
    var total = BATCH * LAST_OUT_DIM
    _copy_cpu(out_p, last_out_ptr, total)


def _backward_cpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    OUT_DIM_: Int,
    *NODES: GraphNode,
](
    mut g: ComputeGraph[OUT_DIM_, *NODES],
    grad_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
) raises:
    comptime N = NODES.size

    # Zero all node grad_out_bufs (scatter-add targets). For InputSlots
    # this is the gradient accumulator the caller reads after backward;
    # for compute nodes it's the per-call scratch.
    comptime for i in range(N):
        comptime OUT_DIM_i = NODES[i].OUT_DIM
        var go_p_i = g.nodes[i].grad_out_ptr_via()
        _zero_cpu(go_p_i, BATCH * OUT_DIM_i)

    # Seed last node's grad_out_buf from the external grad_output.
    comptime LAST_OUT_DIM = NODES[N - 1].OUT_DIM
    var ext_go_p = mptr(grad_output.ptr)
    var last_go_p = g.nodes[N - 1].grad_out_ptr_via()
    _copy_cpu(last_go_p, ext_go_p, BATCH * LAST_OUT_DIM)

    # Reverse topological backward. Skip InputSlots (KIND=0): their
    # grad_out_buf is already the accumulator that consumers scatter
    # into; no compute and no scatter-add of their own.
    comptime for ridx in range(N):
        comptime i = N - 1 - ridx
        comptime kind = NODES[i].KIND
        comptime if kind > 0:
            # Run backward: reads grad_out_buf, writes grad_in*_buf.
            g.nodes[i].vjp_via[target, BATCH, POLICY=POLICY]()

            # I.2.6.e — collect grad_in pointers into a fixed-4
            # InlineArray of Optionals, then uniform comptime-for over
            # KIND for scatter-add. Slots K >= ARITY stay None.
            var grad_in_ptrs = InlineArray[
                Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]], 4
            ](fill=None)
            grad_in_ptrs[0] = g.nodes[i].grad_in0_ptr_via()
            grad_in_ptrs[1] = g.nodes[i].grad_in1_ptr_via()
            grad_in_ptrs[2] = g.nodes[i].grad_in2_ptr_via()
            grad_in_ptrs[3] = g.nodes[i].grad_in3_ptr_via()

            comptime for k in range(kind):
                comptime src_k = NODES[i].IN_NAMES[k]
                comptime in_dim_k = NODES[i].IN_DIMS[k]
                var total_k = BATCH * in_dim_k
                comptime for j in range(N):
                    comptime if NODES[j].NAME == src_k:
                        var pred_go_p = g.nodes[j].grad_out_ptr_via()
                        _scatter_add_cpu(
                            pred_go_p, grad_in_ptrs[k].value(), total_k,
                        )


# ──────────────────────────────────────────────────────────────────────
# GPU forward / backward — mirror the CPU bodies, swap scalar loops for
# `_enqueue_*` kernel calls. Buffers come from each node's GPU storage
# (chosen via `*_ptr_via()` based on `ts.target_tag`).
# ──────────────────────────────────────────────────────────────────────


def _forward_gpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    OUT_DIM_: Int,
    *NODES: GraphNode,
](
    mut g: ComputeGraph[OUT_DIM_, *NODES],
    mut output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
) raises:
    comptime N = NODES.size
    var ctx = g.ts.ctx.value()
    # Seed sentinel: valid pointer from node 0 used purely as
    # InlineArray fill; every slot is overwritten in the loop below.
    var seed_ptr = g.nodes[0].out_ptr_via()

    comptime for i in range(N):
        comptime kind = NODES[i].KIND
        comptime if kind > 0:
            # I.2.6.k — InlineArray sized at per-node KIND (mirrors _forward_cpu).
            var ptrs = InlineArray[
                UnsafePointer[Scalar[DT], MutAnyOrigin], kind,
            ](fill=seed_ptr)
            comptime for k in range(kind):
                comptime src_k = NODES[i].IN_NAMES[k]
                comptime for j in range(N):
                    comptime if NODES[j].NAME == src_k:
                        ptrs[k] = g.nodes[j].out_ptr_via()
            g.nodes[i].forward_via[target, BATCH, POLICY=POLICY](ptrs)

    # Copy last node's out_buf into the external output.
    comptime LAST_OUT_DIM = NODES[N - 1].OUT_DIM
    comptime last_total = BATCH * LAST_OUT_DIM
    var out_p = mptr(output.ptr)
    var last_out_ptr = g.nodes[N - 1].out_ptr_via()
    _enqueue_copy[last_total](ctx, out_p, last_out_ptr)


def _backward_gpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    OUT_DIM_: Int,
    *NODES: GraphNode,
](
    mut g: ComputeGraph[OUT_DIM_, *NODES],
    grad_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
) raises:
    comptime N = NODES.size
    var ctx = g.ts.ctx.value()

    # Zero all node grad_out_bufs (slot accumulators included).
    comptime for i in range(N):
        comptime OUT_DIM_i = NODES[i].OUT_DIM
        comptime total_i = BATCH * OUT_DIM_i
        var go_p_i = g.nodes[i].grad_out_ptr_via()
        _enqueue_zero[total_i](ctx, go_p_i)

    # Seed last node's grad_out_buf from the external grad_output.
    comptime LAST_OUT_DIM = NODES[N - 1].OUT_DIM
    comptime last_total = BATCH * LAST_OUT_DIM
    var ext_go_p = mptr(grad_output.ptr)
    var last_go_p = g.nodes[N - 1].grad_out_ptr_via()
    _enqueue_copy[last_total](ctx, last_go_p, ext_go_p)

    # Reverse topological backward. Skip InputSlots (KIND=0).
    comptime for ridx in range(N):
        comptime i = N - 1 - ridx
        comptime kind = NODES[i].KIND
        comptime if kind > 0:
            g.nodes[i].vjp_via[target, BATCH, POLICY=POLICY]()

            # I.2.6.e — uniform comptime-for over KIND (mirrors _backward_cpu).
            var grad_in_ptrs = InlineArray[
                Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]], 4
            ](fill=None)
            grad_in_ptrs[0] = g.nodes[i].grad_in0_ptr_via()
            grad_in_ptrs[1] = g.nodes[i].grad_in1_ptr_via()
            grad_in_ptrs[2] = g.nodes[i].grad_in2_ptr_via()
            grad_in_ptrs[3] = g.nodes[i].grad_in3_ptr_via()

            comptime for k in range(kind):
                comptime src_k = NODES[i].IN_NAMES[k]
                comptime in_dim_k = NODES[i].IN_DIMS[k]
                comptime total_k = BATCH * in_dim_k
                comptime for j in range(N):
                    comptime if NODES[j].NAME == src_k:
                        var pred_go_p = g.nodes[j].grad_out_ptr_via()
                        _enqueue_add[total_k](
                            ctx, pred_go_p, grad_in_ptrs[k].value(),
                        )
