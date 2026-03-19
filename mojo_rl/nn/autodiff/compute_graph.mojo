"""ComputeGraph — compile-time differentiable DAG builder.

Composes Model-conforming nodes into an arbitrary directed acyclic graph (DAG)
with automatic fan-out gradient accumulation. Unlike Sequential (linear chain)
or DualPath (same-input fan-out), ComputeGraph handles true DAGs where any
node's output can feed into multiple downstream nodes, and any node can take
inputs from multiple predecessors.

Key features:
- Fixed 2-input arity per node (covers all RL algorithms; chain concats for 3+)
- Automatic gradient accumulation at fan-out points
- All topology resolved at compile time — zero runtime overhead
- Nodes use the Model trait, so existing composed types (Sequential, SkipConcat,
  etc.) work directly as node ops

Usage:
    from mojo_rl.nn.autodiff.compute_graph import ComputeGraph, GNode

    # DDPG actor loss as a DAG (equivalent to Sequential[SkipConcat[Actor], Critic, Negate])
    comptime DDPGGraph = ComputeGraph[
        GNode[ActorModel, -1],          # 0: obs → action
        GNode[ConcatNode, -1, 0],       # 1: [obs, action] (implicit concat)
        GNode[CriticModel, 1],          # 2: → Q
        GNode[Negate[1], 2],            # 3: → -Q
    ]

    # SAC actor loss — true DAG with fan-out
    comptime SACGraph = ComputeGraph[
        GNode[ActorModel, -1],          # 0: obs → [mean, log_std]
        GNode[RSample[6], 0],           # 1: → [action, log_prob]
        GNode[Slice[7, 0, 6], 1],       # 2: → action  (fan-out from 1)
        GNode[ConcatNode, -1, 2],       # 3: [obs, action] = critic_input
        GNode[Critic1, 3],              # 4: → Q1  (fan-out from 3)
        GNode[Critic2, 3],              # 5: → Q2  (fan-out from 3)
        GNode[Min[1], 4, 5],            # 6: → min_Q
        GNode[Slice[7, 6, 7], 1],       # 7: → log_prob  (fan-out from 1)
        GNode[SACLossOp, 6, 7],         # 8: → loss
    ]

Memory layout:
  Cache (per sample): [activations | node_caches]
    activations: node_0_out(OUT_DIM_0) | ... | node_{N-1}_out(OUT_DIM_{N-1})
    node_caches: node_0_cache(CS_0) | ... | node_{N-1}_cache(CS_{N-1})

  Workspace (per sample, temporary during backward):
    grad_activations: same layout as activations
    concat_buffer: max across all dual-input nodes of their concat_dim
    op_workspace: max across all nodes of their WORKSPACE_SIZE_PER_SAMPLE
"""

from ..constants import dtype, TPB
from ..model.model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.builtin.variadics import Variadic
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


# GPU matmul requires 16-byte alignment = 4 float32 elements
@always_inline
fn _align4(x: Int) -> Int:
    """Round up to next multiple of 4 for GPU alignment."""
    return (x + 3) & ~3


# =============================================================================
# GraphNode trait — a node in the compute graph
# =============================================================================


trait GraphNode(Movable & ImplicitlyCopyable):
    """A node in a ComputeGraph DAG.

    Each node wraps a Model and declares its input sources:
      IN0: Index of first predecessor (-1 = graph external input)
      IN1: Index of second predecessor (-2 = unused, single-input node)

    When IN1 != -2, the outputs of IN0 and IN1 are concatenated to form
    this node's input. The concatenated dimension must equal OP_IN_DIM.
    """

    comptime IN0: Int
    comptime IN1: Int

    comptime OP_IN_DIM: Int
    comptime OP_OUT_DIM: Int
    comptime OP_PARAM_SIZE: Int
    comptime OP_CACHE_SIZE: Int
    comptime OP_WORKSPACE_SIZE_PER_SAMPLE: Int

    @staticmethod
    fn initialize_params[INIT: Initializer](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    fn op_forward[
        BATCH: Int
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    fn op_forward_no_cache[
        BATCH: Int
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    fn op_backward[
        BATCH: Int
    ](
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
    ):
        ...

    # --- GPU methods ---

    @staticmethod
    fn op_forward_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        ...

    @staticmethod
    fn op_backward_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        ...


# =============================================================================
# GNode — concrete GraphNode wrapping any Model
# =============================================================================


@fieldwise_init
struct GNode[Op: Model, in0: Int = -1, in1: Int = -2](GraphNode):
    """Concrete graph node wrapping a Model type.

    Args:
        Op: The Model to execute at this node.
        in0: Index of first input source (-1 = graph input, >= 0 = node index).
        in1: Index of second input source (-2 = unused, -1 = graph input,
             >= 0 = node index). When used, in0 and in1 outputs are concatenated.
    """

    comptime IN0: Int = Self.in0
    comptime IN1: Int = Self.in1
    comptime OP_IN_DIM: Int = Self.Op.IN_DIM
    comptime OP_OUT_DIM: Int = Self.Op.OUT_DIM
    comptime OP_PARAM_SIZE: Int = Self.Op.PARAM_SIZE
    comptime OP_CACHE_SIZE: Int = Self.Op.CACHE_SIZE
    comptime OP_WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self.Op.WORKSPACE_SIZE_PER_SAMPLE
    )

    @staticmethod
    fn initialize_params[INIT: Initializer](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
    ):
        Self.Op.initialize_params[INIT](params)

    @staticmethod
    fn op_forward[
        BATCH: Int
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
    ):
        Self.Op.forward[BATCH](input, output, params, cache)

    @staticmethod
    fn op_forward_no_cache[
        BATCH: Int
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
    ):
        Self.Op.forward[BATCH](input, output, params)

    @staticmethod
    fn op_backward[
        BATCH: Int
    ](
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
    ):
        Self.Op.backward[BATCH](
            grad_output, grad_input, params, cache, grads
        )

    @staticmethod
    fn op_forward_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.Op.forward_gpu[BATCH](ctx, output, input, params, cache, workspace)

    @staticmethod
    fn op_backward_gpu[
        BATCH: Int
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.Op.backward_gpu[BATCH](
            ctx, grad_input, grad_output, params, cache, grads, workspace
        )


# =============================================================================
# ComputeGraph — compile-time DAG of GraphNodes
# =============================================================================


@fieldwise_init
struct ComputeGraph[*NODES: GraphNode](Model):
    """Compile-time differentiable DAG.

    Nodes are executed in index order (forward) and reverse order (backward).
    All buffer sizes, offsets, and fan-out points are resolved at compile time.

    Fan-out handling:
      When a node's output feeds into multiple downstream nodes, the backward
      pass naturally accumulates gradients: each consumer's VJP produces a
      grad_input contribution that is ADDED to the predecessor's grad_act buffer.

    Input rules:
      - IN0 == -1: reads from graph external input
      - IN0 >= 0: reads from that node's activation
      - IN1 == -2: single input (no concat)
      - IN1 >= 0 or -1: second input, concatenated with IN0's output
    """

    comptime node_types = Variadic.types[T=GraphNode, *Self.NODES]
    comptime N = Variadic.size(Self.node_types)

    # =========================================================================
    # Compile-time dimension inference
    # =========================================================================

    # Convention: node 0 must have IN0 == -1 and IN1 == -2 (single input
    # from graph external input). This is the natural topological ordering
    # for all RL algorithm graphs.
    comptime IN_DIM: Int = Self.node_types[0].OP_IN_DIM
    comptime OUT_DIM: Int = Self.node_types[Self.N - 1].OP_OUT_DIM

    # =========================================================================
    # Offset helpers
    # =========================================================================

    @staticmethod
    fn _total_act_size() -> Int:
        """Total activation storage per sample (all node outputs)."""
        var total = 0
        comptime for i in range(Self.N):
            total += Self.node_types[i].OP_OUT_DIM
        return total

    @staticmethod
    fn _total_cache_size() -> Int:
        """Total per-node cache storage per sample."""
        var total = 0
        comptime for i in range(Self.N):
            total += Self.node_types[i].OP_CACHE_SIZE
        return total

    @staticmethod
    fn _act_offset[idx: Int]() -> Int:
        """Per-sample offset to node idx's activation output."""
        var total = 0
        comptime for j in range(idx):
            total += Self.node_types[j].OP_OUT_DIM
        return total

    @staticmethod
    fn _node_cache_offset[idx: Int]() -> Int:
        """Per-sample offset to node idx's op cache (after all activations)."""
        var total = Self._total_act_size()
        comptime for j in range(idx):
            total += Self.node_types[j].OP_CACHE_SIZE
        return total

    @staticmethod
    fn _param_offset[idx: Int]() -> Int:
        """Aligned param offset for node idx."""
        var total = 0
        comptime for j in range(idx):
            comptime if Self.node_types[j].OP_PARAM_SIZE > 0:
                total += _align4(Self.node_types[j].OP_PARAM_SIZE)
        return total

    @staticmethod
    fn _sum_param_size() -> Int:
        """Total param size with alignment padding."""
        var total = 0
        comptime for j in range(Self.N - 1):
            comptime if Self.node_types[j].OP_PARAM_SIZE > 0:
                total += _align4(Self.node_types[j].OP_PARAM_SIZE)
        total += Self.node_types[Self.N - 1].OP_PARAM_SIZE
        return total

    @staticmethod
    fn _max_concat_dim() -> Int:
        """Max concat input dimension across all dual-input nodes."""
        var m = 0
        comptime for i in range(Self.N):
            comptime if Self.node_types[i].IN1 != -2:
                if Self.node_types[i].OP_IN_DIM > m:
                    m = Self.node_types[i].OP_IN_DIM
        return m

    @staticmethod
    fn _max_ws() -> Int:
        """Max per-node workspace across all nodes."""
        var m = 0
        comptime for i in range(Self.N):
            if Self.node_types[i].OP_WORKSPACE_SIZE_PER_SAMPLE > m:
                m = Self.node_types[i].OP_WORKSPACE_SIZE_PER_SAMPLE
        return m

    # Cache: activations + per-node caches
    comptime CACHE_SIZE: Int = Self._total_act_size() + Self._total_cache_size()

    comptime PARAM_SIZE: Int = Self._sum_param_size()

    # Workspace: grad_activations + concat buffer + op workspace
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self._total_act_size()  # grad_activations (same size as activations)
        + Self._max_concat_dim()  # reusable concat buffer
        + Self._max_ws()  # reusable op workspace
    )

    # =========================================================================
    # Source dimension helpers
    # =========================================================================

    @staticmethod
    fn _source_dim[src: Int]() -> Int:
        """Get output dimension of source. -1 = graph IN_DIM."""
        comptime if src == -1:
            return Self.IN_DIM
        else:
            return Self.node_types[src].OP_OUT_DIM

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    fn initialize_params[INIT: Initializer](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize all node params, zeroing alignment padding."""
        # Zero everything (covers alignment padding regions)
        for i in range(Self.PARAM_SIZE):
            params.ptr[i] = Scalar[dtype](0.0)

        comptime for i in range(Self.N):
            comptime if Self.node_types[i].OP_PARAM_SIZE > 0:
                var np = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                Self.node_types[i].initialize_params[INIT](np)

    # =========================================================================
    # CPU Forward (with cache)
    # =========================================================================

    @staticmethod
    fn forward[
        BATCH: Int
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass: execute nodes 0..N-1 in topological order.

        For each node:
        1. Gather input(s) — either graph input or predecessor activation(s)
        2. If dual-input: concat into temporary buffer
        3. Forward through node's Model
        4. Store output in activation buffer (in cache)

        Last node's output is also copied to the output tensor.
        """
        comptime for i in range(Self.N):
            # Always write to activation buffer in cache
            var act_ptr = cache.ptr + BATCH * Self._act_offset[i]()
            var node_out = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.node_types[i].OP_OUT_DIM),
                MutAnyOrigin,
            ](act_ptr)

            # Params and cache for this node
            var node_p = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var node_c = LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH, Self.node_types[i].OP_CACHE_SIZE
                ),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._node_cache_offset[i]())

            # --- Gather input and forward ---
            comptime if Self.node_types[i].IN1 == -2:
                # Single input
                comptime if Self.node_types[i].IN0 == -1:
                    # From graph input
                    var node_in = LayoutTensor[
                        dtype,
                        Layout.row_major(
                            BATCH, Self.node_types[i].OP_IN_DIM
                        ),
                        MutAnyOrigin,
                    ](input.ptr)
                    Self.node_types[i].op_forward[BATCH](
                        node_in, node_out, node_p, node_c
                    )
                else:
                    # From predecessor activation
                    comptime src0 = Self.node_types[i].IN0
                    var node_in = LayoutTensor[
                        dtype,
                        Layout.row_major(
                            BATCH, Self.node_types[i].OP_IN_DIM
                        ),
                        MutAnyOrigin,
                    ](cache.ptr + BATCH * Self._act_offset[src0]())
                    Self.node_types[i].op_forward[BATCH](
                        node_in, node_out, node_p, node_c
                    )
            else:
                # Dual input: concat in0 and in1 outputs
                # Use OP_IN_DIM directly (= dim0 + dim1) to avoid
                # type unification issues with computed comptime vars
                comptime src0 = Self.node_types[i].IN0
                comptime src1 = Self.node_types[i].IN1
                comptime dim0 = Self._source_dim[src0]()
                comptime dim1 = Self._source_dim[src1]()
                comptime NI = Self.node_types[i].OP_IN_DIM

                # Build concat buffer using OP_IN_DIM for type
                var concat_buf = InlineArray[
                    Scalar[dtype], BATCH * NI
                ](uninitialized=True)

                # Copy source 0
                comptime if src0 == -1:
                    for b in range(BATCH):
                        for d in range(dim0):
                            concat_buf[b * NI + d] = (
                                input.ptr[b * Self.IN_DIM + d]
                            )
                else:
                    var s0_ptr = cache.ptr + BATCH * Self._act_offset[
                        src0
                    ]()
                    for b in range(BATCH):
                        for d in range(dim0):
                            concat_buf[b * NI + d] = s0_ptr[
                                b * dim0 + d
                            ]

                # Copy source 1
                comptime if src1 == -1:
                    for b in range(BATCH):
                        for d in range(dim1):
                            concat_buf[
                                b * NI + dim0 + d
                            ] = input.ptr[b * Self.IN_DIM + d]
                else:
                    var s1_ptr = cache.ptr + BATCH * Self._act_offset[
                        src1
                    ]()
                    for b in range(BATCH):
                        for d in range(dim1):
                            concat_buf[
                                b * NI + dim0 + d
                            ] = s1_ptr[b * dim1 + d]

                var node_in = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, NI),
                    MutAnyOrigin,
                ](concat_buf.unsafe_ptr())

                Self.node_types[i].op_forward[BATCH](
                    node_in, node_out, node_p, node_c
                )

            # Copy last node's activation to output tensor
            comptime if i == Self.N - 1:
                for k in range(
                    BATCH * Self.node_types[i].OP_OUT_DIM
                ):
                    output.ptr[k] = act_ptr[k]

    # =========================================================================
    # CPU Forward (no cache — inference)
    # =========================================================================

    @staticmethod
    fn forward[
        BATCH: Int
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Inference forward — allocate dummy cache and delegate."""
        var cache_storage = List[Scalar[dtype]](
            capacity=BATCH * Self.CACHE_SIZE if Self.CACHE_SIZE > 0 else 1
        )
        var cap = BATCH * Self.CACHE_SIZE if Self.CACHE_SIZE > 0 else 1
        for _ in range(cap):
            cache_storage.append(0)
        var cache = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.CACHE_SIZE),
            MutAnyOrigin,
        ](cache_storage.unsafe_ptr())
        Self.forward[BATCH](input, output, params, cache)

    # =========================================================================
    # CPU Backward
    # =========================================================================

    @staticmethod
    fn backward[
        BATCH: Int
    ](
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward pass: execute nodes N-1..0 in reverse topological order.

        Algorithm:
        1. Allocate grad_activations buffer (same layout as activations)
        2. Initialize grad_act[last_node] = grad_output
        3. For each node i (reverse order):
           a. Run node_i.backward: grad_act[i] → grad_node_input
           b. For single-input nodes: ADD grad_node_input to grad_act[IN0]
              (or grad_input if IN0 == -1)
           c. For dual-input nodes: SPLIT grad_node_input and ADD each
              portion to the respective predecessor's grad_act buffer
        4. Fan-out accumulation happens naturally — multiple consumers ADD
           to the same grad_act buffer before that node's VJP runs.
        """
        comptime TOTAL_ACT = Self._total_act_size()

        # Allocate gradient activation buffer (zero-initialized)
        var grad_act_storage = List[Scalar[dtype]](
            capacity=BATCH * TOTAL_ACT
        )
        for _ in range(BATCH * TOTAL_ACT):
            grad_act_storage.append(0)
        var ga_ptr = grad_act_storage.unsafe_ptr()

        # Initialize grad_act for last node = grad_output
        comptime LAST = Self.N - 1
        comptime LAST_OFF = Self._act_offset[LAST]()
        comptime LAST_DIM = Self.node_types[LAST].OP_OUT_DIM
        for k in range(BATCH * LAST_DIM):
            ga_ptr[BATCH * LAST_OFF + k] = grad_output.ptr[k]

        # Zero grad_input
        for k in range(BATCH * Self.IN_DIM):
            grad_input.ptr[k] = Scalar[dtype](0.0)

        # Reverse iteration
        comptime for _ri in range(Self.N):
            comptime i = Self.N - 1 - _ri

            # Get this node's accumulated gradient
            var gi_go = LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH, Self.node_types[i].OP_OUT_DIM
                ),
                MutAnyOrigin,
            ](ga_ptr + BATCH * Self._act_offset[i]())

            # Params and cache for this node
            var node_p = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var node_c = LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH, Self.node_types[i].OP_CACHE_SIZE
                ),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._node_cache_offset[i]())
            var node_g = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr + Self._param_offset[i]())

            # Allocate grad_input buffer for this node's VJP
            comptime NODE_IN = Self.node_types[i].OP_IN_DIM
            var gi_buf = InlineArray[Scalar[dtype], BATCH * NODE_IN](
                uninitialized=True
            )
            for k in range(BATCH * NODE_IN):
                gi_buf[k] = Scalar[dtype](0.0)
            var gi_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, NODE_IN),
                MutAnyOrigin,
            ](gi_buf.unsafe_ptr())

            # Run VJP
            Self.node_types[i].op_backward[BATCH](
                gi_go, gi_t, node_p, node_c, node_g
            )

            # --- Scatter grad_input to predecessors ---
            comptime if Self.node_types[i].IN1 == -2:
                # Single input: add entire grad to predecessor
                comptime src0 = Self.node_types[i].IN0
                comptime if src0 == -1:
                    # Add to graph grad_input
                    for k in range(BATCH * Self.IN_DIM):
                        grad_input.ptr[k] = (
                            grad_input.ptr[k] + gi_buf[k]
                        )
                else:
                    # Add to predecessor's grad_act
                    var dst = ga_ptr + BATCH * Self._act_offset[src0]()
                    comptime dst_dim = Self.node_types[src0].OP_OUT_DIM
                    for k in range(BATCH * dst_dim):
                        dst[k] = dst[k] + gi_buf[k]
            else:
                # Dual input: split grad and add to each predecessor
                comptime src0 = Self.node_types[i].IN0
                comptime src1 = Self.node_types[i].IN1
                comptime dim0 = Self._source_dim[src0]()
                comptime dim1 = Self._source_dim[src1]()

                # Scatter to source 0
                comptime if src0 == -1:
                    for b in range(BATCH):
                        for d in range(dim0):
                            grad_input.ptr[b * Self.IN_DIM + d] = (
                                grad_input.ptr[b * Self.IN_DIM + d]
                                + gi_buf[b * NODE_IN + d]
                            )
                else:
                    var dst0 = (
                        ga_ptr + BATCH * Self._act_offset[src0]()
                    )
                    for b in range(BATCH):
                        for d in range(dim0):
                            dst0[b * dim0 + d] = (
                                dst0[b * dim0 + d]
                                + gi_buf[b * NODE_IN + d]
                            )

                # Scatter to source 1
                comptime if src1 == -1:
                    for b in range(BATCH):
                        for d in range(dim1):
                            grad_input.ptr[b * Self.IN_DIM + d] = (
                                grad_input.ptr[b * Self.IN_DIM + d]
                                + gi_buf[
                                    b * NODE_IN + dim0 + d
                                ]
                            )
                else:
                    var dst1 = (
                        ga_ptr + BATCH * Self._act_offset[src1]()
                    )
                    for b in range(BATCH):
                        for d in range(dim1):
                            dst1[b * dim1 + d] = (
                                dst1[b * dim1 + d]
                                + gi_buf[
                                    b * NODE_IN + dim0 + d
                                ]
                            )

    # =========================================================================
    # GPU Forward (with cache)
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU forward pass: execute nodes in topological order.

        Each node's activation is stored in cache (same layout as CPU).
        Dual-input nodes get a temporary concat buffer on device.
        """
        comptime for i in range(Self.N):
            # Activation buffer in cache for this node
            var act_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.node_types[i].OP_OUT_DIM),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._act_offset[i]())

            # Params and op cache
            var node_p = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var node_c = LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH, Self.node_types[i].OP_CACHE_SIZE
                ),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._node_cache_offset[i]())

            # --- Gather input and forward ---
            comptime if Self.node_types[i].IN1 == -2:
                # Single input
                comptime if Self.node_types[i].IN0 == -1:
                    var node_in = LayoutTensor[
                        dtype,
                        Layout.row_major(
                            BATCH, Self.node_types[i].OP_IN_DIM
                        ),
                        MutAnyOrigin,
                    ](input.ptr)
                    Self.node_types[i].op_forward_gpu[BATCH](
                        ctx, act_t, node_in, node_p, node_c, workspace
                    )
                else:
                    comptime src0 = Self.node_types[i].IN0
                    var node_in = LayoutTensor[
                        dtype,
                        Layout.row_major(
                            BATCH, Self.node_types[i].OP_IN_DIM
                        ),
                        MutAnyOrigin,
                    ](
                        cache.ptr + BATCH * Self._act_offset[src0]()
                    )
                    Self.node_types[i].op_forward_gpu[BATCH](
                        ctx, act_t, node_in, node_p, node_c, workspace
                    )
            else:
                # Dual input: concat on GPU using two copy kernels
                comptime src0 = Self.node_types[i].IN0
                comptime src1 = Self.node_types[i].IN1
                comptime dim0 = Self._source_dim[src0]()
                comptime dim1 = Self._source_dim[src1]()
                comptime NI = Self.node_types[i].OP_IN_DIM

                var concat_buf = ctx.enqueue_create_buffer[dtype](
                    BATCH * NI
                )
                var concat_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, NI),
                    MutAnyOrigin,
                ](concat_buf.unsafe_ptr())

                # Copy source 0 into concat[:dim0]
                comptime S0_COPY = BATCH * dim0
                var s0_grid = (S0_COPY + TPB - 1) // TPB

                comptime if src0 == -1:
                    var s0_src = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.IN_DIM),
                        ImmutAnyOrigin,
                    ](input.ptr)

                    @always_inline
                    fn copy_s0_ext(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, NI),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.IN_DIM),
                            ImmutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx >= S0_COPY:
                            return
                        var b = idx // dim0
                        var d = idx % dim0
                        dst.ptr[b * NI + d] = src.ptr[
                            b * Self.IN_DIM + d
                        ]

                    ctx.enqueue_function[copy_s0_ext, copy_s0_ext](
                        concat_t,
                        s0_src,
                        grid_dim=(s0_grid,),
                        block_dim=(TPB,),
                    )
                else:
                    var s0_src = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, dim0),
                        ImmutAnyOrigin,
                    ](
                        cache.ptr
                        + BATCH * Self._act_offset[src0]()
                    )

                    @always_inline
                    fn copy_s0_node(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, NI),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, dim0),
                            ImmutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx >= S0_COPY:
                            return
                        var b = idx // dim0
                        var d = idx % dim0
                        dst.ptr[b * NI + d] = src.ptr[
                            b * dim0 + d
                        ]

                    ctx.enqueue_function[copy_s0_node, copy_s0_node](
                        concat_t,
                        s0_src,
                        grid_dim=(s0_grid,),
                        block_dim=(TPB,),
                    )

                # Copy source 1 into concat[dim0:]
                comptime S1_COPY = BATCH * dim1
                var s1_grid = (S1_COPY + TPB - 1) // TPB

                comptime if src1 == -1:
                    var s1_src = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.IN_DIM),
                        ImmutAnyOrigin,
                    ](input.ptr)

                    @always_inline
                    fn copy_s1_ext(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, NI),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.IN_DIM),
                            ImmutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx >= S1_COPY:
                            return
                        var b = idx // dim1
                        var d = idx % dim1
                        dst.ptr[b * NI + dim0 + d] = src.ptr[
                            b * Self.IN_DIM + d
                        ]

                    ctx.enqueue_function[copy_s1_ext, copy_s1_ext](
                        concat_t,
                        s1_src,
                        grid_dim=(s1_grid,),
                        block_dim=(TPB,),
                    )
                else:
                    var s1_src = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, dim1),
                        ImmutAnyOrigin,
                    ](
                        cache.ptr
                        + BATCH * Self._act_offset[src1]()
                    )

                    @always_inline
                    fn copy_s1_node(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, NI),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, dim1),
                            ImmutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx >= S1_COPY:
                            return
                        var b = idx // dim1
                        var d = idx % dim1
                        dst.ptr[b * NI + dim0 + d] = src.ptr[
                            b * dim1 + d
                        ]

                    ctx.enqueue_function[copy_s1_node, copy_s1_node](
                        concat_t,
                        s1_src,
                        grid_dim=(s1_grid,),
                        block_dim=(TPB,),
                    )

                # Forward through node
                Self.node_types[i].op_forward_gpu[BATCH](
                    ctx, act_t, concat_t, node_p, node_c, workspace
                )

        # Copy last node's activation to output
        comptime LAST_DIM = Self.node_types[Self.N - 1].OP_OUT_DIM
        comptime COPY_TOTAL = BATCH * LAST_DIM
        var copy_grid = (COPY_TOTAL + TPB - 1) // TPB

        var last_act_immut = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, LAST_DIM),
            ImmutAnyOrigin,
        ](cache.ptr + BATCH * Self._act_offset[Self.N - 1]())

        @always_inline
        fn copy_output_kernel(
            dst: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.OUT_DIM),
                MutAnyOrigin,
            ],
            src: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LAST_DIM),
                ImmutAnyOrigin,
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= COPY_TOTAL:
                return
            dst.ptr[idx] = src.ptr[idx]

        ctx.enqueue_function[copy_output_kernel, copy_output_kernel](
            output,
            last_act_immut,
            grid_dim=(copy_grid,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU inference forward — allocate dummy cache from workspace."""
        # Use workspace tail as dummy cache
        var cache_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * Self.CACHE_SIZE if Self.CACHE_SIZE > 0 else 1
        )
        var cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.CACHE_SIZE),
            MutAnyOrigin,
        ](cache_buf.unsafe_ptr())
        Self.forward_gpu[BATCH](
            ctx, output, input, params, cache_t, workspace
        )

    @staticmethod
    fn forward_gpu_no_cache_on_stream[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        """GPU forward on stream — delegates to default."""
        Self.forward_gpu_no_cache[BATCH](
            ctx, output, input, params, workspace
        )

    # =========================================================================
    # GPU Backward
    # =========================================================================

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU backward pass with fan-out gradient accumulation.

        Uses device buffers for gradient activations, elementwise add
        kernels for fan-out accumulation.
        """
        comptime TOTAL_ACT = Self._total_act_size()

        # Allocate zero-initialized grad activation buffer on GPU
        var ga_buf = ctx.enqueue_create_buffer[dtype](BATCH * TOTAL_ACT)
        var ga_ptr = ga_buf.unsafe_ptr()

        # Zero the buffer
        comptime GA_TOTAL = BATCH * TOTAL_ACT
        var ga_grid = (GA_TOTAL + TPB - 1) // TPB

        @always_inline
        fn zero_ga_kernel(
            dst: LayoutTensor[
                dtype,
                Layout.row_major(GA_TOTAL, 1),
                MutAnyOrigin,
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= GA_TOTAL:
                return
            dst.ptr[idx] = Scalar[dtype](0.0)

        var ga_flat = LayoutTensor[
            dtype, Layout.row_major(GA_TOTAL, 1), MutAnyOrigin
        ](ga_ptr)
        ctx.enqueue_function[zero_ga_kernel, zero_ga_kernel](
            ga_flat, grid_dim=(ga_grid,), block_dim=(TPB,)
        )

        # Initialize grad_act[last_node] = grad_output
        comptime LAST = Self.N - 1
        comptime LAST_OFF = Self._act_offset[LAST]()
        comptime LAST_DIM = Self.node_types[LAST].OP_OUT_DIM
        comptime INIT_TOTAL = BATCH * LAST_DIM
        var init_grid = (INIT_TOTAL + TPB - 1) // TPB

        @always_inline
        fn init_last_grad_kernel(
            dst: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LAST_DIM),
                MutAnyOrigin,
            ],
            src: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.OUT_DIM),
                ImmutAnyOrigin,
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= INIT_TOTAL:
                return
            dst.ptr[idx] = src.ptr[idx]

        var ga_last = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, LAST_DIM),
            MutAnyOrigin,
        ](ga_ptr + BATCH * LAST_OFF)
        var go_immut = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.OUT_DIM),
            ImmutAnyOrigin,
        ](grad_output.ptr)
        ctx.enqueue_function[
            init_last_grad_kernel, init_last_grad_kernel
        ](
            ga_last,
            go_immut,
            grid_dim=(init_grid,),
            block_dim=(TPB,),
        )

        # Zero grad_input
        comptime GI_TOTAL = BATCH * Self.IN_DIM
        var gi_grid = (GI_TOTAL + TPB - 1) // TPB

        @always_inline
        fn zero_gi_kernel(
            dst: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.IN_DIM),
                MutAnyOrigin,
            ],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= GI_TOTAL:
                return
            dst.ptr[idx] = Scalar[dtype](0.0)

        ctx.enqueue_function[zero_gi_kernel, zero_gi_kernel](
            grad_input, grid_dim=(gi_grid,), block_dim=(TPB,)
        )

        # Reverse iteration
        comptime for _ri in range(Self.N):
            comptime i = Self.N - 1 - _ri

            # Get this node's accumulated gradient
            var gi_go = LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH, Self.node_types[i].OP_OUT_DIM
                ),
                MutAnyOrigin,
            ](ga_ptr + BATCH * Self._act_offset[i]())

            # Params and cache for this node
            var node_p = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var node_c = LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH, Self.node_types[i].OP_CACHE_SIZE
                ),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._node_cache_offset[i]())
            var node_g = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr + Self._param_offset[i]())

            # Allocate grad_input buffer for this node
            comptime NODE_IN = Self.node_types[i].OP_IN_DIM
            var gi_node_buf = ctx.enqueue_create_buffer[dtype](
                BATCH * NODE_IN
            )
            var gi_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, NODE_IN),
                MutAnyOrigin,
            ](gi_node_buf.unsafe_ptr())

            # Run VJP on GPU
            Self.node_types[i].op_backward_gpu[BATCH](
                ctx, gi_t, gi_go, node_p, node_c, node_g, workspace
            )

            # --- Scatter grad_input to predecessors (GPU add kernels) ---
            comptime if Self.node_types[i].IN1 == -2:
                # Single input
                comptime src0 = Self.node_types[i].IN0
                comptime if src0 == -1:
                    # Add to graph grad_input
                    comptime ADD_N = BATCH * Self.IN_DIM
                    var add_grid = (ADD_N + TPB - 1) // TPB
                    var gi_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.IN_DIM),
                        ImmutAnyOrigin,
                    ](gi_node_buf.unsafe_ptr())

                    @always_inline
                    fn add_to_gi_single(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.IN_DIM),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.IN_DIM),
                            ImmutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx >= ADD_N:
                            return
                        dst.ptr[idx] = dst.ptr[idx] + src.ptr[idx]

                    ctx.enqueue_function[
                        add_to_gi_single, add_to_gi_single
                    ](
                        grad_input,
                        gi_immut,
                        grid_dim=(add_grid,),
                        block_dim=(TPB,),
                    )
                else:
                    # Add to predecessor's grad_act
                    comptime dst_dim = Self.node_types[src0].OP_OUT_DIM
                    comptime ADD_N = BATCH * dst_dim
                    var add_grid = (ADD_N + TPB - 1) // TPB
                    var dst_t = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, dst_dim),
                        MutAnyOrigin,
                    ](ga_ptr + BATCH * Self._act_offset[src0]())
                    var gi_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, dst_dim),
                        ImmutAnyOrigin,
                    ](gi_node_buf.unsafe_ptr())

                    @always_inline
                    fn add_to_pred_single(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, dst_dim),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, dst_dim),
                            ImmutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx >= ADD_N:
                            return
                        dst.ptr[idx] = dst.ptr[idx] + src.ptr[idx]

                    ctx.enqueue_function[
                        add_to_pred_single, add_to_pred_single
                    ](
                        dst_t,
                        gi_immut,
                        grid_dim=(add_grid,),
                        block_dim=(TPB,),
                    )
            else:
                # Dual input: split and scatter
                comptime src0 = Self.node_types[i].IN0
                comptime src1 = Self.node_types[i].IN1
                comptime dim0 = Self._source_dim[src0]()
                comptime dim1 = Self._source_dim[src1]()

                # Scatter source 0 portion
                comptime if src0 == -1:
                    comptime S0_N = BATCH * dim0
                    var s0_grid = (S0_N + TPB - 1) // TPB

                    @always_inline
                    fn scatter_to_gi_s0(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.IN_DIM),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, NODE_IN),
                            ImmutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx >= S0_N:
                            return
                        var b = idx // dim0
                        var d = idx % dim0
                        dst.ptr[b * Self.IN_DIM + d] = (
                            dst.ptr[b * Self.IN_DIM + d]
                            + src.ptr[b * NODE_IN + d]
                        )

                    var gi_node_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, NODE_IN),
                        ImmutAnyOrigin,
                    ](gi_node_buf.unsafe_ptr())
                    ctx.enqueue_function[
                        scatter_to_gi_s0, scatter_to_gi_s0
                    ](
                        grad_input,
                        gi_node_immut,
                        grid_dim=(s0_grid,),
                        block_dim=(TPB,),
                    )
                else:
                    comptime S0_N = BATCH * dim0
                    var s0_grid = (S0_N + TPB - 1) // TPB
                    comptime d0_dim = Self.node_types[src0].OP_OUT_DIM

                    @always_inline
                    fn scatter_to_pred_s0(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, d0_dim),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, NODE_IN),
                            ImmutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx >= S0_N:
                            return
                        var b = idx // dim0
                        var d = idx % dim0
                        dst.ptr[b * d0_dim + d] = (
                            dst.ptr[b * d0_dim + d]
                            + src.ptr[b * NODE_IN + d]
                        )

                    var dst0_t = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, d0_dim),
                        MutAnyOrigin,
                    ](ga_ptr + BATCH * Self._act_offset[src0]())
                    var gi_node_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, NODE_IN),
                        ImmutAnyOrigin,
                    ](gi_node_buf.unsafe_ptr())
                    ctx.enqueue_function[
                        scatter_to_pred_s0, scatter_to_pred_s0
                    ](
                        dst0_t,
                        gi_node_immut,
                        grid_dim=(s0_grid,),
                        block_dim=(TPB,),
                    )

                # Scatter source 1 portion
                comptime if src1 == -1:
                    comptime S1_N = BATCH * dim1
                    var s1_grid = (S1_N + TPB - 1) // TPB

                    @always_inline
                    fn scatter_to_gi_s1(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, Self.IN_DIM),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, NODE_IN),
                            ImmutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx >= S1_N:
                            return
                        var b = idx // dim1
                        var d = idx % dim1
                        dst.ptr[b * Self.IN_DIM + d] = (
                            dst.ptr[b * Self.IN_DIM + d]
                            + src.ptr[b * NODE_IN + dim0 + d]
                        )

                    var gi_node_immut_s1 = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, NODE_IN),
                        ImmutAnyOrigin,
                    ](gi_node_buf.unsafe_ptr())
                    ctx.enqueue_function[
                        scatter_to_gi_s1, scatter_to_gi_s1
                    ](
                        grad_input,
                        gi_node_immut_s1,
                        grid_dim=(s1_grid,),
                        block_dim=(TPB,),
                    )
                else:
                    comptime S1_N = BATCH * dim1
                    var s1_grid = (S1_N + TPB - 1) // TPB
                    comptime d1_dim = Self.node_types[src1].OP_OUT_DIM

                    @always_inline
                    fn scatter_to_pred_s1(
                        dst: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, d1_dim),
                            MutAnyOrigin,
                        ],
                        src: LayoutTensor[
                            dtype,
                            Layout.row_major(BATCH, NODE_IN),
                            ImmutAnyOrigin,
                        ],
                    ):
                        var idx = Int(
                            block_dim.x * block_idx.x + thread_idx.x
                        )
                        if idx >= S1_N:
                            return
                        var b = idx // dim1
                        var d = idx % dim1
                        dst.ptr[b * d1_dim + d] = (
                            dst.ptr[b * d1_dim + d]
                            + src.ptr[b * NODE_IN + dim0 + d]
                        )

                    var dst1_t = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, d1_dim),
                        MutAnyOrigin,
                    ](ga_ptr + BATCH * Self._act_offset[src1]())
                    var gi_node_immut_s1 = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, NODE_IN),
                        ImmutAnyOrigin,
                    ](gi_node_buf.unsafe_ptr())
                    ctx.enqueue_function[
                        scatter_to_pred_s1, scatter_to_pred_s1
                    ](
                        dst1_t,
                        gi_node_immut_s1,
                        grid_dim=(s1_grid,),
                        block_dim=(TPB,),
                    )
