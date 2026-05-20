"""ComputeGraph — compile-time differentiable DAG builder with named nodes.

Instead of opaque integer indices:
    GNode[ActorModel, -1],          # what does -1 mean?
    GNode[RSample[6], 0],           # what is node 0?

Use readable string names:
    GNode["actor",   ActorModel,  "input"],
    GNode["rsample", RSample[6],  "actor"],

Usage:
    from mojo_rl.nn.autodiff.compute_graph import ComputeGraph, GNode

    # DDPG actor loss
    comptime DDPGGraph = ComputeGraph[
        GNode["actor",     Linear[4, 2],  "input"],
        GNode["critic_in", Identity[6],   "input", "actor"],
        GNode["critic",    Linear[6, 1],  "critic_in"],
        GNode["neg",       Negate[1],     "critic"],
    ]

    # SAC actor loss — true DAG with fan-out
    comptime SACGraph = ComputeGraph[
        GNode["actor",    ActorModel,     "input"],
        GNode["rsample",  RSample[6],     "actor"],
        GNode["action",   Slice[7, 0, 6], "rsample"],
        GNode["Q1",       CriticModel,    "input", "action"],
        GNode["Q2",       CriticModel,    "input", "action"],
        GNode["min_q",    Min[1],         "Q1", "Q2"],
        GNode["log_prob", Slice[7, 6, 7], "rsample"],
        GNode["loss",     SACLossOp,      "min_q", "log_prob"],
    ]

Memory layout: activations, per-node caches, workspaces (same as before).
Name resolution happens entirely at compile time via comptime for + comptime if.
"""

from ..constants import dtype, TPB, gpu_align
from ..model.model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream


# GPU matmul requires 16-byte alignment = 4 float32 elements
@always_inline
def _align4(x: Int) -> Int:
    """GPU-aligned element count (16-byte aligned for any dtype)."""
    return gpu_align(x)


# =============================================================================
# GraphNode trait — a node with string-based input references
# =============================================================================


trait GraphNode(Movable & ImplicitlyCopyable):
    """A node in a ComputeGraph DAG with string-based input references.

    Each node has a unique NAME and declares its input sources by name:
      IN0_NAME: Name of first input source ("input" = graph external input)
      IN1_NAME: Name of second input source ("" = unused, single-input node)

    When IN1_NAME != "", the outputs of IN0 and IN1 sources are concatenated
    to form this node's input. The concatenated dimension must equal OP_IN_DIM.
    """

    comptime NAME: String
    comptime IN0_NAME: String
    comptime IN1_NAME: String

    comptime OP_IN_DIM: Int
    comptime OP_OUT_DIM: Int
    comptime OP_PARAM_SIZE: Int
    comptime OP_CACHE_SIZE: Int
    comptime OP_WORKSPACE_SIZE_PER_SAMPLE: Int
    # Persistent non-trainable state size for the wrapped op (0 for most nodes).
    comptime OP_STATE_SIZE: Int = 0

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    def initialize_state[dtype: DType = DType.float32](
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize persistent non-trainable state for the wrapped op.

        Default: no-op (OP_STATE_SIZE == 0). Nodes wrapping stateful Models
        override to recurse into the underlying Model.initialize_state.
        """
        pass

    @staticmethod
    def op_forward[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    def op_forward_no_cache[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
        ],
    ):
        ...

    @staticmethod
    def op_backward[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
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
    def op_forward_gpu[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        ...

    @staticmethod
    def op_backward_gpu[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
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
struct GNode[
    node_name: StringLiteral,
    Op: Model,
    in0_name: StringLiteral = "input",
    in1_name: StringLiteral = "",
](GraphNode):
    """Concrete named graph node wrapping a Model type.

    Args:
        node_name: Unique name for this node.
        Op: The Model to execute at this node.
        in0_name: Name of first input source ("input" = graph external input).
        in1_name: Name of second input source ("" = unused).
                  When used, in0 and in1 outputs are concatenated.
    """

    comptime NAME: String = String(Self.node_name)
    comptime IN0_NAME: String = String(Self.in0_name)
    comptime IN1_NAME: String = String(Self.in1_name)
    comptime OP_IN_DIM: Int = Self.Op.IN_DIM
    comptime OP_OUT_DIM: Int = Self.Op.OUT_DIM
    comptime OP_PARAM_SIZE: Int = Self.Op.PARAM_SIZE
    comptime OP_CACHE_SIZE: Int = Self.Op.CACHE_SIZE
    comptime OP_WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self.Op.WORKSPACE_SIZE_PER_SAMPLE
    )
    comptime OP_STATE_SIZE: Int = Self.Op.STATE_SIZE

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
    ):
        Self.Op.initialize_params[INIT, dtype](params)

    @staticmethod
    def initialize_state[dtype: DType = DType.float32](
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
        ],
    ):
        Self.Op.initialize_state[dtype](state)

    @staticmethod
    def op_forward[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
    ):
        Self.Op.forward[BATCH, dtype](input, output, params, state, cache)

    @staticmethod
    def op_forward_no_cache[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
        ],
    ):
        Self.Op.forward[BATCH, dtype](input, output, params, state)

    @staticmethod
    def op_backward[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
    ):
        Self.Op.backward[BATCH, dtype](grad_output, grad_input, params, state, cache, grads)

    @staticmethod
    def op_forward_gpu[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.Op.forward_gpu[BATCH, dtype](ctx, output, input, params, state, cache, workspace)

    @staticmethod
    def op_backward_gpu[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.OP_STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OP_CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.OP_PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        Self.Op.backward_gpu[BATCH, dtype](
            ctx, grad_input, grad_output, params, state, cache, grads, workspace
        )


# =============================================================================
# ComputeGraph — compile-time DAG with named node references
# =============================================================================


@fieldwise_init
struct ComputeGraph[*NODES: GraphNode](Model):
    """Compile-time differentiable DAG with named node references.

    Functionally identical to ComputeGraph but uses string names instead
    of integer indices for node references. Name resolution happens entirely
    at compile time — zero runtime overhead.

    Input rules:
      - IN0_NAME == "input": reads from graph external input
      - IN0_NAME == <node_name>: reads from that node's activation
      - IN1_NAME == "": single input (no concat)
      - IN1_NAME == <node_name> or "input": second input, concatenated
    """

    comptime node_types = Self.NODES
    comptime N = Self.node_types.size

    # =========================================================================
    # Compile-time dimension inference
    # =========================================================================

    comptime IN_DIM: Int = Self.node_types[0].OP_IN_DIM
    comptime OUT_DIM: Int = Self.node_types[Self.N - 1].OP_OUT_DIM

    # =========================================================================
    # Name resolution helpers (all compile-time evaluable)
    # =========================================================================

    @staticmethod
    def _source_dim_by_name[name: String]() -> Int:
        """Get output dimension of named source. 'input' = graph IN_DIM."""
        comptime if name == "input":
            return Self.IN_DIM
        comptime for j in range(Self.N):
            comptime if Self.node_types[j].NAME == name:
                return Self.node_types[j].OP_OUT_DIM
        return 0

    @staticmethod
    def _source_act_offset_by_name[name: String]() -> Int:
        """Get activation offset for a named source node."""
        comptime for j in range(Self.N):
            comptime if Self.node_types[j].NAME == name:
                return Self._act_offset[j]()
        return 0

    # =========================================================================
    # Offset helpers (same as ComputeGraph)
    # =========================================================================

    @staticmethod
    def _total_act_size() -> Int:
        """Total activation storage per sample (all node outputs)."""
        var total = 0
        comptime for i in range(Self.N):
            total += Self.node_types[i].OP_OUT_DIM
        return total

    @staticmethod
    def _total_cache_size() -> Int:
        """Total per-node cache storage per sample."""
        var total = 0
        comptime for i in range(Self.N):
            total += Self.node_types[i].OP_CACHE_SIZE
        return total

    @staticmethod
    def _act_offset[idx: Int]() -> Int:
        """Per-sample offset to node idx's activation output."""
        var total = 0
        comptime for j in range(idx):
            total += Self.node_types[j].OP_OUT_DIM
        return total

    @staticmethod
    def _node_cache_offset[idx: Int]() -> Int:
        """Per-sample offset to node idx's op cache (after all activations)."""
        var total = Self._total_act_size()
        comptime for j in range(idx):
            total += Self.node_types[j].OP_CACHE_SIZE
        return total

    @staticmethod
    def _param_offset[idx: Int]() -> Int:
        """Aligned param offset for node idx."""
        var total = 0
        comptime for j in range(idx):
            comptime if Self.node_types[j].OP_PARAM_SIZE > 0:
                total += _align4(Self.node_types[j].OP_PARAM_SIZE)
        return total

    @staticmethod
    def _sum_param_size() -> Int:
        """Total param size with alignment padding."""
        var total = 0
        comptime for j in range(Self.N - 1):
            comptime if Self.node_types[j].OP_PARAM_SIZE > 0:
                total += _align4(Self.node_types[j].OP_PARAM_SIZE)
        total += Self.node_types[Self.N - 1].OP_PARAM_SIZE
        return total

    @staticmethod
    def _sum_state_size() -> Int:
        """Total per-node state storage (scalar-indexed — no alignment padding).

        Mirrors Sequential._sum_state_size: state slots hold RNG counters /
        running stats, not matmul'd, so no 4-element GPU alignment is needed.
        """
        var total = 0
        comptime for i in range(Self.N):
            total += Self.node_types[i].OP_STATE_SIZE
        return total

    @staticmethod
    def _state_offset[idx: Int]() -> Int:
        """Unaligned state offset for node idx (scalar-indexed)."""
        var total = 0
        comptime for j in range(idx):
            total += Self.node_types[j].OP_STATE_SIZE
        return total

    @staticmethod
    def _max_concat_dim() -> Int:
        """Max concat input dimension across all dual-input nodes."""
        var m = 0
        comptime for i in range(Self.N):
            comptime if Self.node_types[i].IN1_NAME != "":
                if Self.node_types[i].OP_IN_DIM > m:
                    m = Self.node_types[i].OP_IN_DIM
        return m

    @staticmethod
    def _max_node_in_dim() -> Int:
        """Max OP_IN_DIM across all nodes (for backward gi scratch)."""
        var m = 0
        comptime for i in range(Self.N):
            if Self.node_types[i].OP_IN_DIM > m:
                m = Self.node_types[i].OP_IN_DIM
        return m

    @staticmethod
    def _max_scratch_dim() -> Int:
        """Max scratch per sample: max(concat_dim, node_in_dim)."""
        comptime c = Self._max_concat_dim()
        comptime n = Self._max_node_in_dim()
        return c if c > n else n

    @staticmethod
    def _max_ws() -> Int:
        """Max per-node workspace across all nodes."""
        var m = 0
        comptime for i in range(Self.N):
            if Self.node_types[i].OP_WORKSPACE_SIZE_PER_SAMPLE > m:
                m = Self.node_types[i].OP_WORKSPACE_SIZE_PER_SAMPLE
        return m

    # Cache: activations + per-node caches
    comptime CACHE_SIZE: Int = Self._total_act_size() + Self._total_cache_size()

    comptime PARAM_SIZE: Int = Self._sum_param_size()

    # Persistent non-trainable state: sum of node states (unaligned).
    comptime STATE_SIZE: Int = Self._sum_state_size()

    # Workspace: grad_activations + scratch buffer + op workspace + dummy cache
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = (
        Self._total_act_size()
        + Self._max_scratch_dim()
        + Self._max_ws()
        + Self.CACHE_SIZE
    )

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize all node params, zeroing alignment padding."""
        for i in range(Self.PARAM_SIZE):
            params.ptr[i] = Scalar[dtype](0.0)

        comptime for i in range(Self.N):
            comptime if Self.node_types[i].OP_PARAM_SIZE > 0:
                var np = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                    MutAnyOrigin,
                ](params.ptr + Self._param_offset[i]())
                Self.node_types[i].initialize_params[INIT, dtype](np)

    @staticmethod
    def initialize_state[dtype: DType = DType.float32](
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Recursively initialize each node's persistent state slice."""
        comptime for i in range(Self.N):
            comptime if Self.node_types[i].OP_STATE_SIZE > 0:
                var node_state = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.node_types[i].OP_STATE_SIZE),
                    MutAnyOrigin,
                ](state.ptr + Self._state_offset[i]())
                Self.node_types[i].initialize_state[dtype](node_state)

    # =========================================================================
    # CPU Forward (with cache)
    # =========================================================================

    @staticmethod
    def forward[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass: execute nodes 0..N-1 in topological order."""
        comptime for i in range(Self.N):
            # Always write to activation buffer in cache
            var act_ptr = cache.ptr + BATCH * Self._act_offset[i]()
            var node_out = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.node_types[i].OP_OUT_DIM),
                MutAnyOrigin,
            ](act_ptr)

            # Params, state, and cache for this node
            var node_p = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var node_s = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_STATE_SIZE),
                MutAnyOrigin,
            ](state.ptr + Self._state_offset[i]())
            var node_c = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.node_types[i].OP_CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._node_cache_offset[i]())

            # --- Gather input and forward ---
            comptime if Self.node_types[i].IN1_NAME == "":
                # Single input
                comptime if Self.node_types[i].IN0_NAME == "input":
                    # From graph input
                    var node_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.node_types[i].OP_IN_DIM),
                        MutAnyOrigin,
                    ](input.ptr)
                    Self.node_types[i].op_forward[BATCH, dtype](
                        node_in, node_out, node_p, node_s, node_c
                    )
                else:
                    # From predecessor activation (resolved by name)
                    comptime src0_name = Self.node_types[i].IN0_NAME
                    var node_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.node_types[i].OP_IN_DIM),
                        MutAnyOrigin,
                    ](
                        cache.ptr
                        + BATCH * Self._source_act_offset_by_name[src0_name]()
                    )
                    Self.node_types[i].op_forward[BATCH, dtype](
                        node_in, node_out, node_p, node_s, node_c
                    )
            else:
                # Dual input: concat in0 and in1 outputs
                comptime src0_name = Self.node_types[i].IN0_NAME
                comptime src1_name = Self.node_types[i].IN1_NAME
                comptime dim0 = Self._source_dim_by_name[src0_name]()
                comptime dim1 = Self._source_dim_by_name[src1_name]()
                comptime NI = Self.node_types[i].OP_IN_DIM

                # Build concat buffer
                var concat_buf = InlineArray[Scalar[dtype], BATCH * NI](
                    uninitialized=True
                )

                # Copy source 0
                comptime if src0_name == "input":
                    for b in range(BATCH):
                        for d in range(dim0):
                            concat_buf[b * NI + d] = input.ptr[
                                b * Self.IN_DIM + d
                            ]
                else:
                    var s0_ptr = (
                        cache.ptr
                        + BATCH * Self._source_act_offset_by_name[src0_name]()
                    )
                    for b in range(BATCH):
                        for d in range(dim0):
                            concat_buf[b * NI + d] = s0_ptr[b * dim0 + d]

                # Copy source 1
                comptime if src1_name == "input":
                    for b in range(BATCH):
                        for d in range(dim1):
                            concat_buf[b * NI + dim0 + d] = input.ptr[
                                b * Self.IN_DIM + d
                            ]
                else:
                    var s1_ptr = (
                        cache.ptr
                        + BATCH * Self._source_act_offset_by_name[src1_name]()
                    )
                    for b in range(BATCH):
                        for d in range(dim1):
                            concat_buf[b * NI + dim0 + d] = s1_ptr[b * dim1 + d]

                var node_in = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, NI),
                    MutAnyOrigin,
                ](concat_buf.unsafe_ptr())

                Self.node_types[i].op_forward[BATCH, dtype](
                    node_in, node_out, node_p, node_s, node_c
                )

            # Copy last node's activation to output tensor
            comptime if i == Self.N - 1:
                for k in range(BATCH * Self.node_types[i].OP_OUT_DIM):
                    output.ptr[k] = act_ptr[k]

    # =========================================================================
    # CPU Forward (no cache — inference)
    # =========================================================================

    @staticmethod
    def forward[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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
        Self.forward[BATCH, dtype](input, output, params, state, cache)

    # =========================================================================
    # CPU Backward
    # =========================================================================

    @staticmethod
    def backward[
        BATCH: Int, dtype: DType = DType.float32
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward pass: execute nodes N-1..0 in reverse topological order.

        Fan-out accumulation happens naturally — multiple consumers ADD
        to the same grad_act buffer before that node's VJP runs.
        """
        comptime TOTAL_ACT = Self._total_act_size()

        # Allocate gradient activation buffer (zero-initialized)
        var grad_act_storage = List[Scalar[dtype]](capacity=BATCH * TOTAL_ACT)
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
                Layout.row_major(BATCH, Self.node_types[i].OP_OUT_DIM),
                MutAnyOrigin,
            ](ga_ptr + BATCH * Self._act_offset[i]())

            # Params, state, cache, and grads for this node
            var node_p = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var node_s = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_STATE_SIZE),
                MutAnyOrigin,
            ](state.ptr + Self._state_offset[i]())
            var node_c = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.node_types[i].OP_CACHE_SIZE),
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
            Self.node_types[i].op_backward[BATCH, dtype](
                gi_go, gi_t, node_p, node_s, node_c, node_g
            )

            # --- Scatter grad_input to predecessors ---
            comptime if Self.node_types[i].IN1_NAME == "":
                # Single input: add entire grad to predecessor
                comptime src0_name = Self.node_types[i].IN0_NAME
                comptime if src0_name == "input":
                    # Add to graph grad_input
                    for k in range(BATCH * Self.IN_DIM):
                        grad_input.ptr[k] = grad_input.ptr[k] + gi_buf[k]
                else:
                    # Add to predecessor's grad_act
                    var dst = (
                        ga_ptr
                        + BATCH * Self._source_act_offset_by_name[src0_name]()
                    )
                    comptime dst_dim = Self._source_dim_by_name[src0_name]()
                    for k in range(BATCH * dst_dim):
                        dst[k] = dst[k] + gi_buf[k]
            else:
                # Dual input: split grad and add to each predecessor
                comptime src0_name = Self.node_types[i].IN0_NAME
                comptime src1_name = Self.node_types[i].IN1_NAME
                comptime dim0 = Self._source_dim_by_name[src0_name]()
                comptime dim1 = Self._source_dim_by_name[src1_name]()

                # Scatter to source 0
                comptime if src0_name == "input":
                    for b in range(BATCH):
                        for d in range(dim0):
                            grad_input.ptr[b * Self.IN_DIM + d] = (
                                grad_input.ptr[b * Self.IN_DIM + d]
                                + gi_buf[b * NODE_IN + d]
                            )
                else:
                    var dst0 = (
                        ga_ptr
                        + BATCH * Self._source_act_offset_by_name[src0_name]()
                    )
                    for b in range(BATCH):
                        for d in range(dim0):
                            dst0[b * dim0 + d] = (
                                dst0[b * dim0 + d] + gi_buf[b * NODE_IN + d]
                            )

                # Scatter to source 1
                comptime if src1_name == "input":
                    for b in range(BATCH):
                        for d in range(dim1):
                            grad_input.ptr[b * Self.IN_DIM + d] = (
                                grad_input.ptr[b * Self.IN_DIM + d]
                                + gi_buf[b * NODE_IN + dim0 + d]
                            )
                else:
                    var dst1 = (
                        ga_ptr
                        + BATCH * Self._source_act_offset_by_name[src1_name]()
                    )
                    for b in range(BATCH):
                        for d in range(dim1):
                            dst1[b * dim1 + d] = (
                                dst1[b * dim1 + d]
                                + gi_buf[b * NODE_IN + dim0 + d]
                            )

    # =========================================================================
    # GPU Forward (with cache)
    # =========================================================================

    @staticmethod
    def forward_gpu[
        BATCH: Int, dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU forward pass: execute nodes in topological order."""
        comptime OP_WS_OFF = Self._total_act_size() + Self._max_scratch_dim()
        comptime OP_WS_SIZE = max(1, Self._max_ws())
        var op_ws = DeviceBuffer[dtype](
            ctx,
            workspace.unsafe_ptr() + BATCH * OP_WS_OFF,
            BATCH * OP_WS_SIZE,
            owning=False,
        )
        comptime for i in range(Self.N):
            # Activation buffer in cache for this node
            var act_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.node_types[i].OP_OUT_DIM),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._act_offset[i]())

            # Params, state, and op cache
            var node_p = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var node_s = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_STATE_SIZE),
                MutAnyOrigin,
            ](state.ptr + Self._state_offset[i]())
            var node_c = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.node_types[i].OP_CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._node_cache_offset[i]())

            # --- Gather input and forward ---
            comptime if Self.node_types[i].IN1_NAME == "":
                # Single input
                comptime if Self.node_types[i].IN0_NAME == "input":
                    var node_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.node_types[i].OP_IN_DIM),
                        MutAnyOrigin,
                    ](input.ptr)
                    Self.node_types[i].op_forward_gpu[BATCH, dtype](
                        ctx, act_t, node_in, node_p, node_s, node_c, op_ws
                    )
                else:
                    comptime src0_name = Self.node_types[i].IN0_NAME
                    var node_in = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.node_types[i].OP_IN_DIM),
                        MutAnyOrigin,
                    ](
                        cache.ptr
                        + BATCH * Self._source_act_offset_by_name[src0_name]()
                    )
                    Self.node_types[i].op_forward_gpu[BATCH, dtype](
                        ctx, act_t, node_in, node_p, node_s, node_c, op_ws
                    )
            else:
                # Dual input: concat on GPU using two copy kernels
                comptime src0_name = Self.node_types[i].IN0_NAME
                comptime src1_name = Self.node_types[i].IN1_NAME
                comptime dim0 = Self._source_dim_by_name[src0_name]()
                comptime dim1 = Self._source_dim_by_name[src1_name]()
                comptime NI = Self.node_types[i].OP_IN_DIM

                var concat_ptr = (
                    workspace.unsafe_ptr() + BATCH * Self._total_act_size()
                )
                var concat_t = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH, NI),
                    MutAnyOrigin,
                ](concat_ptr)

                # Copy source 0 into concat[:dim0]
                comptime S0_COPY = BATCH * dim0
                var s0_grid = (S0_COPY + TPB - 1) // TPB

                comptime if src0_name == "input":
                    var s0_src = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.IN_DIM),
                        ImmutAnyOrigin,
                    ](input.ptr)

                    @parameter
                    @always_inline
                    def copy_s0_ext(
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
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                        if idx >= S0_COPY:
                            return
                        var b = idx // dim0
                        var d = idx % dim0
                        dst.ptr[b * NI + d] = src.ptr[b * Self.IN_DIM + d]

                    ctx.enqueue_function[copy_s0_ext](
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
                        + BATCH * Self._source_act_offset_by_name[src0_name]()
                    )

                    @parameter
                    @always_inline
                    def copy_s0_node(
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
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                        if idx >= S0_COPY:
                            return
                        var b = idx // dim0
                        var d = idx % dim0
                        dst.ptr[b * NI + d] = src.ptr[b * dim0 + d]

                    ctx.enqueue_function[copy_s0_node](
                        concat_t,
                        s0_src,
                        grid_dim=(s0_grid,),
                        block_dim=(TPB,),
                    )

                # Copy source 1 into concat[dim0:]
                comptime S1_COPY = BATCH * dim1
                var s1_grid = (S1_COPY + TPB - 1) // TPB

                comptime if src1_name == "input":
                    var s1_src = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.IN_DIM),
                        ImmutAnyOrigin,
                    ](input.ptr)

                    @parameter
                    @always_inline
                    def copy_s1_ext(
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
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                        if idx >= S1_COPY:
                            return
                        var b = idx // dim1
                        var d = idx % dim1
                        dst.ptr[b * NI + dim0 + d] = src.ptr[
                            b * Self.IN_DIM + d
                        ]

                    ctx.enqueue_function[copy_s1_ext](
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
                        + BATCH * Self._source_act_offset_by_name[src1_name]()
                    )

                    @parameter
                    @always_inline
                    def copy_s1_node(
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
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                        if idx >= S1_COPY:
                            return
                        var b = idx // dim1
                        var d = idx % dim1
                        dst.ptr[b * NI + dim0 + d] = src.ptr[b * dim1 + d]

                    ctx.enqueue_function[copy_s1_node](
                        concat_t,
                        s1_src,
                        grid_dim=(s1_grid,),
                        block_dim=(TPB,),
                    )

                # Forward through node
                Self.node_types[i].op_forward_gpu[BATCH, dtype](
                    ctx, act_t, concat_t, node_p, node_s, node_c, op_ws
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

        @parameter
        @always_inline
        def copy_output_kernel(
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

        ctx.enqueue_function[copy_output_kernel](
            output,
            last_act_immut,
            grid_dim=(copy_grid,),
            block_dim=(TPB,),
        )

    @staticmethod
    def forward_gpu_no_cache[
        BATCH: Int, dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """GPU inference forward — use workspace tail as dummy cache."""
        comptime CACHE_OFF = Self._total_act_size() + Self._max_scratch_dim() + Self._max_ws()
        var cache_ptr = workspace.unsafe_ptr() + BATCH * CACHE_OFF
        var cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.CACHE_SIZE),
            MutAnyOrigin,
        ](cache_ptr)
        Self.forward_gpu[BATCH, dtype](ctx, output, input, params, state, cache_t, workspace)

    @staticmethod
    def forward_gpu_no_cache_on_stream[
        BATCH: Int, dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        """GPU forward on stream — delegates to default."""
        Self.forward_gpu_no_cache[BATCH, dtype](ctx, output, input, params, state, workspace)

    # =========================================================================
    # GPU Backward
    # =========================================================================

    @staticmethod
    def backward_gpu[
        BATCH: Int, dtype: DType = DType.float32,
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
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
        """GPU backward pass with fan-out gradient accumulation."""
        comptime TOTAL_ACT = Self._total_act_size()

        # Use workspace offset 0 for grad activation buffer
        var ga_ptr = workspace.unsafe_ptr()

        # Zero the buffer
        comptime GA_TOTAL = BATCH * TOTAL_ACT
        var ga_grid = (GA_TOTAL + TPB - 1) // TPB

        @parameter
        @always_inline
        def zero_ga_kernel(
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
        ctx.enqueue_function[zero_ga_kernel](
            ga_flat, grid_dim=(ga_grid,), block_dim=(TPB,)
        )

        # Initialize grad_act[last_node] = grad_output
        comptime LAST = Self.N - 1
        comptime LAST_OFF = Self._act_offset[LAST]()
        comptime LAST_DIM = Self.node_types[LAST].OP_OUT_DIM
        comptime INIT_TOTAL = BATCH * LAST_DIM
        var init_grid = (INIT_TOTAL + TPB - 1) // TPB

        @parameter
        @always_inline
        def init_last_grad_kernel(
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
        ctx.enqueue_function[init_last_grad_kernel](
            ga_last,
            go_immut,
            grid_dim=(init_grid,),
            block_dim=(TPB,),
        )

        # Zero grad_input
        comptime GI_TOTAL = BATCH * Self.IN_DIM
        var gi_grid = (GI_TOTAL + TPB - 1) // TPB

        @parameter
        @always_inline
        def zero_gi_kernel(
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

        ctx.enqueue_function[zero_gi_kernel](
            grad_input, grid_dim=(gi_grid,), block_dim=(TPB,)
        )

        # Op workspace slice for backward node calls
        comptime OP_WS_OFF = Self._total_act_size() + Self._max_scratch_dim()
        comptime OP_WS_SIZE = max(1, Self._max_ws())
        var op_ws = DeviceBuffer[dtype](
            ctx,
            workspace.unsafe_ptr() + BATCH * OP_WS_OFF,
            BATCH * OP_WS_SIZE,
            owning=False,
        )

        # Scratch pointer for per-node gi buffer (reused each iteration)
        var gi_scratch_ptr = (
            workspace.unsafe_ptr() + BATCH * Self._total_act_size()
        )

        # Reverse iteration
        comptime for _ri in range(Self.N):
            comptime i = Self.N - 1 - _ri

            # Get this node's accumulated gradient
            var gi_go = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.node_types[i].OP_OUT_DIM),
                MutAnyOrigin,
            ](ga_ptr + BATCH * Self._act_offset[i]())

            # Params, state, cache, grads for this node
            var node_p = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            var node_s = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_STATE_SIZE),
                MutAnyOrigin,
            ](state.ptr + Self._state_offset[i]())
            var node_c = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.node_types[i].OP_CACHE_SIZE),
                MutAnyOrigin,
            ](cache.ptr + BATCH * Self._node_cache_offset[i]())
            var node_g = LayoutTensor[
                dtype,
                Layout.row_major(Self.node_types[i].OP_PARAM_SIZE),
                MutAnyOrigin,
            ](grads.ptr + Self._param_offset[i]())

            # Use workspace scratch for grad_input buffer
            comptime NODE_IN = Self.node_types[i].OP_IN_DIM
            var gi_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, NODE_IN),
                MutAnyOrigin,
            ](gi_scratch_ptr)

            # Run VJP on GPU
            Self.node_types[i].op_backward_gpu[BATCH, dtype](
                ctx, gi_t, gi_go, node_p, node_s, node_c, node_g, op_ws
            )

            # --- Scatter grad_input to predecessors (GPU add kernels) ---
            comptime if Self.node_types[i].IN1_NAME == "":
                # Single input
                comptime src0_name = Self.node_types[i].IN0_NAME
                comptime if src0_name == "input":
                    # Add to graph grad_input
                    comptime ADD_N = BATCH * Self.IN_DIM
                    var add_grid = (ADD_N + TPB - 1) // TPB
                    var gi_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, Self.IN_DIM),
                        ImmutAnyOrigin,
                    ](gi_scratch_ptr)

                    @parameter
                    @always_inline
                    def add_to_gi_single(
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
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                        if idx >= ADD_N:
                            return
                        dst.ptr[idx] = dst.ptr[idx] + src.ptr[idx]

                    ctx.enqueue_function[add_to_gi_single](
                        grad_input,
                        gi_immut,
                        grid_dim=(add_grid,),
                        block_dim=(TPB,),
                    )
                else:
                    # Add to predecessor's grad_act
                    comptime dst_dim = Self._source_dim_by_name[src0_name]()
                    comptime ADD_N = BATCH * dst_dim
                    var add_grid = (ADD_N + TPB - 1) // TPB
                    var dst_t = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, dst_dim),
                        MutAnyOrigin,
                    ](
                        ga_ptr
                        + BATCH * Self._source_act_offset_by_name[src0_name]()
                    )
                    var gi_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, dst_dim),
                        ImmutAnyOrigin,
                    ](gi_scratch_ptr)

                    @parameter
                    @always_inline
                    def add_to_pred_single(
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
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                        if idx >= ADD_N:
                            return
                        dst.ptr[idx] = dst.ptr[idx] + src.ptr[idx]

                    ctx.enqueue_function[add_to_pred_single](
                        dst_t,
                        gi_immut,
                        grid_dim=(add_grid,),
                        block_dim=(TPB,),
                    )
            else:
                # Dual input: split and scatter
                comptime src0_name = Self.node_types[i].IN0_NAME
                comptime src1_name = Self.node_types[i].IN1_NAME
                comptime dim0 = Self._source_dim_by_name[src0_name]()
                comptime dim1 = Self._source_dim_by_name[src1_name]()

                # Scatter source 0 portion
                comptime if src0_name == "input":
                    comptime S0_N = BATCH * dim0
                    var s0_grid = (S0_N + TPB - 1) // TPB

                    @parameter
                    @always_inline
                    def scatter_to_gi_s0(
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
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
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
                    ](gi_scratch_ptr)
                    ctx.enqueue_function[scatter_to_gi_s0](
                        grad_input,
                        gi_node_immut,
                        grid_dim=(s0_grid,),
                        block_dim=(TPB,),
                    )
                else:
                    comptime S0_N = BATCH * dim0
                    var s0_grid = (S0_N + TPB - 1) // TPB
                    comptime d0_dim = Self._source_dim_by_name[src0_name]()

                    @parameter
                    @always_inline
                    def scatter_to_pred_s0(
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
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                        if idx >= S0_N:
                            return
                        var b = idx // dim0
                        var d = idx % dim0
                        dst.ptr[b * d0_dim + d] = (
                            dst.ptr[b * d0_dim + d] + src.ptr[b * NODE_IN + d]
                        )

                    var dst0_t = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, d0_dim),
                        MutAnyOrigin,
                    ](
                        ga_ptr
                        + BATCH * Self._source_act_offset_by_name[src0_name]()
                    )
                    var gi_node_immut = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, NODE_IN),
                        ImmutAnyOrigin,
                    ](gi_scratch_ptr)
                    ctx.enqueue_function[scatter_to_pred_s0](
                        dst0_t,
                        gi_node_immut,
                        grid_dim=(s0_grid,),
                        block_dim=(TPB,),
                    )

                # Scatter source 1 portion
                comptime if src1_name == "input":
                    comptime S1_N = BATCH * dim1
                    var s1_grid = (S1_N + TPB - 1) // TPB

                    @parameter
                    @always_inline
                    def scatter_to_gi_s1(
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
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
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
                    ](gi_scratch_ptr)
                    ctx.enqueue_function[scatter_to_gi_s1](
                        grad_input,
                        gi_node_immut_s1,
                        grid_dim=(s1_grid,),
                        block_dim=(TPB,),
                    )
                else:
                    comptime S1_N = BATCH * dim1
                    var s1_grid = (S1_N + TPB - 1) // TPB
                    comptime d1_dim = Self._source_dim_by_name[src1_name]()

                    @parameter
                    @always_inline
                    def scatter_to_pred_s1(
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
                        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
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
                    ](
                        ga_ptr
                        + BATCH * Self._source_act_offset_by_name[src1_name]()
                    )
                    var gi_node_immut_s1 = LayoutTensor[
                        dtype,
                        Layout.row_major(BATCH, NODE_IN),
                        ImmutAnyOrigin,
                    ](gi_scratch_ptr)
                    ctx.enqueue_function[scatter_to_pred_s1](
                        dst1_t,
                        gi_node_immut_s1,
                        grid_dim=(s1_grid,),
                        block_dim=(TPB,),
                    )
