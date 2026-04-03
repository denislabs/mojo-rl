"""NetworkState: consolidated mutable state for a neural network.

Holds params, grads, optimizer_state, and step_num in heap-allocated Lists
and provides zero-copy LayoutTensor views over them.  Both Trainer and Network
delegate all state management here, eliminating duplicated init/view code.

Usage:
    var state = NetworkState[model_type, Adam]()
    state.initialize[Kaiming[]]()

    # LayoutTensor views (zero-copy pointer casts)
    var p = state.params_view()
    var g = state.grads_view()
    var s = state.state_view()

    # Zeroing / stepping
    state.zero_grads()
    state.optimizer_step()       # increments step_num internally

    # Target-network operations
    target_state.copy_params_from(online_state)
    target_state.soft_update_from(online_state, tau=0.005)
"""

from ..model import Model
from ..optimizer import Optimizer
from ..initializer import Initializer, Xavier
from ..constants import dtype as default_dtype
from ..checkpoint import (
    write_checkpoint_header,
    write_float_section_ptr,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_float_section_list,
    read_metadata_section,
    get_metadata_value,
    save_checkpoint_file,
    BinaryCheckpoint,
)

from layout import Layout, LayoutTensor
from std.memory import alloc, memset


struct NetworkState[MODEL: Model, OPTIMIZER: Optimizer, dtype: DType = default_dtype](
    ImplicitlyCopyable, Movable
):
    """Consolidated mutable state for a neural network.

    Contains all persistent data needed by Trainer and Network:
    - params: model weights (heap-allocated to support large models)
    - grads: parameter gradients (zeroed before each backward pass)
    - optimizer_state: optimizer-specific accumulators (e.g. Adam m/v moments)
    - step_num: global optimizer step counter (used for Adam bias correction)

    All lists are heap-allocated to avoid stack overflow with large models.
    LayoutTensor views are created on-demand via zero-copy pointer casts.

    Parameters:
        MODEL: The model architecture (implements Model trait).
        OPTIMIZER: The optimizer (implements Optimizer trait).
        dtype: Data type for all buffers (default: DType.float32).
    """

    comptime PARAM_SIZE: Int = Self.MODEL.PARAM_SIZE
    comptime STATE_SIZE: Int = Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM

    var params: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var grads: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var optimizer_state: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var step_num: Int
    var lr_scale: Float64

    def __init__(out self):
        """Allocate and zero-initialize all state lists."""
        self.step_num = 0
        self.lr_scale = 1.0

        self.params = alloc[Scalar[Self.dtype]](Self.PARAM_SIZE)
        memset(self.params, 0, Self.PARAM_SIZE)

        self.grads = alloc[Scalar[Self.dtype]](Self.PARAM_SIZE)
        memset(self.grads, 0, Self.PARAM_SIZE)

        self.optimizer_state = alloc[Scalar[Self.dtype]](Self.STATE_SIZE)
        memset(self.optimizer_state, 0, Self.STATE_SIZE)

    def __init__(out self, *, copy: Self):
        self.step_num = copy.step_num
        self.lr_scale = copy.lr_scale
        self.params = copy.params.copy()
        self.grads = copy.grads.copy()
        self.optimizer_state = copy.optimizer_state.copy()

    def __init__(out self, *, deinit take: Self):
        self.step_num = take.step_num
        self.lr_scale = take.lr_scale
        self.params = take.params
        self.grads = take.grads
        self.optimizer_state = take.optimizer_state

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize[INITIALIZER: Initializer = Xavier[]](mut self):
        """Initialize params using the given initializer strategy.

        Delegates to MODEL.initialize_params which handles per-layer fan
        dimensions for Sequential models. Leaf layers use their own
        IN_DIM/OUT_DIM directly.

        Parameters:
            INITIALIZER: Weight initialization strategy (Xavier, Kaiming, etc.).

        Example:
            state.initialize[Kaiming[]]()   # for ReLU networks
            state.initialize[Xavier]()    # for tanh/sigmoid networks
        """
        var t = self.params_view()
        Self.MODEL.initialize_params[INITIALIZER](t)

    # =========================================================================
    # LayoutTensor Views (zero-copy)
    # =========================================================================

    def params_view(
        self,
    ) -> LayoutTensor[Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        """Return a LayoutTensor view over params (zero-copy pointer cast)."""

        return LayoutTensor[
            Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ](self.params)

    def grads_view(
        self,
    ) -> LayoutTensor[Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        """Return a LayoutTensor view over grads (zero-copy pointer cast)."""
        return LayoutTensor[
            Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ](self.grads)

    def state_view(
        self,
    ) -> LayoutTensor[
        Self.dtype,
        Layout.row_major(Self.PARAM_SIZE, Self.OPTIMIZER.STATE_PER_PARAM),
        MutAnyOrigin,
    ]:
        """Return a LayoutTensor view over optimizer_state (zero-copy)."""
        return LayoutTensor[
            Self.dtype,
            Layout.row_major(Self.PARAM_SIZE, Self.OPTIMIZER.STATE_PER_PARAM),
            MutAnyOrigin,
        ](self.optimizer_state)

    # =========================================================================
    # Core Mutation
    # =========================================================================

    def zero_grads(mut self):
        """Zero all parameter gradients before a backward pass."""
        for i in range(Self.PARAM_SIZE):
            (self.grads + i)[] = 0

    def set_lr_scale(mut self, scale: Float64):
        """Set the LR multiplier applied at each optimizer step.

        Use for LR annealing (e.g. PPO linear decay):
            state.set_lr_scale(1.0 - progress)  # progress in [0, 1]

        Args:
            scale: Multiplier applied to the compile-time base LR (default 1.0).
        """
        self.lr_scale = scale

    def optimizer_step(mut self):
        """One optimizer step + increment step_num.

        Applies self.lr_scale to the base LR (set via set_lr_scale()).
        Creates lvalue views internally (params and state are mut in step()).
        """
        self.step_num += 1
        var p = self.params_view()
        var s = self.state_view()
        Self.OPTIMIZER.step[Self.PARAM_SIZE](
            p, self.grads_view(), s, self.step_num, self.lr_scale
        )

    # =========================================================================
    # Target Network Operations
    # =========================================================================

    def copy_params_from(mut self, source: Self):
        """Hard copy: self.params = source.params (θ_target ← θ_online).

        Args:
            source: Network state to copy parameters from.
        """
        for i in range(Self.PARAM_SIZE):
            (self.params + i)[] = (source.params + i)[]

    def soft_update_from(mut self, source: Self, tau: Float64):
        """Soft update: self = tau * source + (1 - tau) * self.

        Used for target network updates in DQN, DDPG, TD3, SAC.

        Args:
            source: The online network state to blend from.
            tau: Interpolation factor (e.g. 0.005).
        """
        var tau_s = Scalar[Self.dtype](tau)
        var one_m = Scalar[Self.dtype](1.0 - tau)
        for i in range(Self.PARAM_SIZE):
            (self.params + i)[] = (
                tau_s * (source.params + i)[] + one_m * (self.params + i)[]
            )

    # =========================================================================
    # Section-based helpers (for multi-network single-file checkpoints)
    # =========================================================================

    def write_sections(self, prefix: String) -> String:
        """Serialize params and optimizer_state as prefixed sections.

        Used by agents to build a single checkpoint file containing multiple
        networks. Example prefix "actor_" produces sections:
            actor_params:
            0.123
            ...
            actor_optimizer_state:
            0.0
            ...

        Args:
            prefix: Section name prefix (e.g. "actor_", "critic_").

        Returns:
            String with two sections ready to append to a checkpoint file.
        """
        var content = write_float_section_ptr(
            prefix + "params:", self.params, Self.PARAM_SIZE
        )
        content += write_float_section_ptr(
            prefix + "optimizer_state:",
            self.optimizer_state,
            Self.STATE_SIZE,
        )
        return content

    def read_sections(mut self, content: String, prefix: String) raises:
        """Load params and optimizer_state from prefixed sections.

        Counterpart of write_sections — reads "{prefix}params:" and
        "{prefix}optimizer_state:" from a combined checkpoint file.

        Args:
            content: Full checkpoint file content.
            prefix: Section name prefix used when writing (e.g. "actor_").
        """
        var loaded_params = read_float_section_list[Self.dtype](
            content, prefix + "params:", Self.PARAM_SIZE
        )
        for i in range(Self.PARAM_SIZE):
            (self.params + i)[] = loaded_params[i]

        var loaded_state = read_float_section_list[Self.dtype](
            content, prefix + "optimizer_state:", Self.STATE_SIZE
        )
        for i in range(Self.STATE_SIZE):
            (self.optimizer_state + i)[] = loaded_state[i]

    # =========================================================================
    # Checkpoint Save / Load (single-network file)
    # =========================================================================

    def save_checkpoint(self, filepath: String) raises:
        """Save params, optimizer state, and step_num to a checkpoint file.

        Args:
            filepath: Destination path for the checkpoint file.
        """
        var content = write_checkpoint_header(
            "network_state", Self.PARAM_SIZE, Self.STATE_SIZE
        )
        content += self.write_sections("")
        var metadata = List[String]()
        metadata.append("step_num=" + String(self.step_num))
        content += write_metadata_section(metadata)
        save_checkpoint_file(filepath, content)

    def load_checkpoint(mut self, filepath: String) raises:
        """Load params and optimizer state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
        var content = read_checkpoint_file(filepath)
        var header = parse_checkpoint_header(content)

        if header.param_size != Self.PARAM_SIZE:
            print(
                "Warning: checkpoint param_size ("
                + String(header.param_size)
                + ") != PARAM_SIZE ("
                + String(Self.PARAM_SIZE)
                + ")"
            )

        self.read_sections(content, "")

        var metadata = read_metadata_section(content)
        var step_str = get_metadata_value(metadata, "step_num")
        if len(step_str) > 0:
            self.step_num = Int(atol(step_str))

    # =========================================================================
    # Binary Checkpoint Save / Load (~3x smaller files)
    # =========================================================================

    def write_sections_binary(self, mut ckpt: BinaryCheckpoint[Self.dtype], prefix: String):
        """Add params and optimizer_state as named sections to a binary checkpoint.

        Args:
            ckpt: BinaryCheckpoint to add sections to.
            prefix: Section name prefix (e.g. "actor_", "critic_").
        """
        ckpt.add_float_section_ptr(
            prefix + "params", self.params, Self.PARAM_SIZE
        )
        ckpt.add_float_section_ptr(
            prefix + "optimizer_state", self.optimizer_state, Self.STATE_SIZE
        )

    def read_sections_binary(
        mut self, ckpt: BinaryCheckpoint[Self.dtype], prefix: String
    ) raises:
        """Load params and optimizer_state from a binary checkpoint.

        Args:
            ckpt: BinaryCheckpoint to read sections from.
            prefix: Section name prefix (e.g. "actor_", "critic_").
        """
        var loaded_params = ckpt.get_float_section(
            prefix + "params", Self.PARAM_SIZE
        )
        for i in range(Self.PARAM_SIZE):
            (self.params + i)[] = loaded_params[i]

        var loaded_state = ckpt.get_float_section(
            prefix + "optimizer_state", Self.STATE_SIZE
        )
        for i in range(Self.STATE_SIZE):
            (self.optimizer_state + i)[] = loaded_state[i]

    def save_checkpoint_binary(self, filepath: String) raises:
        """Save params, optimizer state, and step_num to a binary checkpoint.

        Binary format is ~3x smaller than text format.

        Args:
            filepath: Destination path for the binary checkpoint file.
        """
        var ckpt = BinaryCheckpoint[Self.dtype]("network_state")
        self.write_sections_binary(ckpt, "")
        ckpt.add_metadata("step_num", String(self.step_num))
        ckpt.save(filepath)

    def load_checkpoint_binary(mut self, filepath: String) raises:
        """Load params and optimizer state from a binary checkpoint.

        Args:
            filepath: Path to the binary checkpoint file.
        """
        var ckpt = BinaryCheckpoint[Self.dtype].load(filepath)
        self.read_sections_binary(ckpt, "")

        var step_str = ckpt.get_metadata_value("step_num")
        if len(step_str) > 0:
            self.step_num = Int(atol(step_str))
