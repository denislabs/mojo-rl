"""NetworkState: consolidated mutable state for a neural network.

Holds params, grads, optimizer_state, and step_num in heap-allocated Lists
and provides zero-copy LayoutTensor views over them.  Both Trainer and Network
delegate all state management here, eliminating duplicated init/view code.

Usage:
    var state = NetworkState[model_type, Adam]()
    state.initialize[Kaiming]()

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
from ..constants import dtype
from ..checkpoint import (
    write_checkpoint_header,
    write_float_section_list,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_float_section_list,
    read_metadata_section,
    get_metadata_value,
    save_checkpoint_file,
)

from layout import Layout, LayoutTensor


struct NetworkState[MODEL: Model, OPTIMIZER: Optimizer](
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
    """

    comptime PARAM_SIZE: Int = Self.MODEL.PARAM_SIZE
    comptime STATE_SIZE: Int = Self.MODEL.PARAM_SIZE * Self.OPTIMIZER.STATE_PER_PARAM

    var params: List[Scalar[dtype]]
    var grads: List[Scalar[dtype]]
    var optimizer_state: List[Scalar[dtype]]
    var step_num: Int
    var lr_scale: Float64

    fn __init__(out self):
        """Allocate and zero-initialize all state lists."""
        self.step_num = 0
        self.lr_scale = 1.0

        self.params = List[Scalar[dtype]](capacity=Self.PARAM_SIZE)
        for _ in range(Self.PARAM_SIZE):
            self.params.append(Scalar[dtype](0))

        self.grads = List[Scalar[dtype]](capacity=Self.PARAM_SIZE)
        for _ in range(Self.PARAM_SIZE):
            self.grads.append(Scalar[dtype](0))

        self.optimizer_state = List[Scalar[dtype]](capacity=Self.STATE_SIZE)
        for _ in range(Self.STATE_SIZE):
            self.optimizer_state.append(Scalar[dtype](0))

    fn __init__(out self, *, copy: Self):
        self.step_num = copy.step_num
        self.lr_scale = copy.lr_scale
        self.params = copy.params.copy()
        self.grads = copy.grads.copy()
        self.optimizer_state = copy.optimizer_state.copy()

    fn __init__(out self, *, deinit take: Self):
        self.step_num = take.step_num
        self.lr_scale = take.lr_scale
        self.params = take.params^
        self.grads = take.grads^
        self.optimizer_state = take.optimizer_state^

    # =========================================================================
    # Initialization
    # =========================================================================

    fn initialize[INITIALIZER: Initializer = Xavier](mut self):
        """Initialize params using the given initializer strategy.

        Parameters:
            INITIALIZER: Weight initialization strategy (Xavier, Kaiming, etc.).

        Example:
            state.initialize[Kaiming]()   # for ReLU networks
            state.initialize[Xavier]()    # for tanh/sigmoid networks
        """
        var t = self.params_view()
        INITIALIZER.init[
            Self.PARAM_SIZE, Self.MODEL.IN_DIM, Self.MODEL.OUT_DIM
        ](t)

    # =========================================================================
    # LayoutTensor Views (zero-copy)
    # =========================================================================

    fn params_view(
        self,
    ) -> LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        """Return a LayoutTensor view over params (zero-copy pointer cast)."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ](self.params.unsafe_ptr())

    fn grads_view(
        self,
    ) -> LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        """Return a LayoutTensor view over grads (zero-copy pointer cast)."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ](self.grads.unsafe_ptr())

    fn state_view(
        self,
    ) -> LayoutTensor[
        dtype,
        Layout.row_major(Self.PARAM_SIZE, Self.OPTIMIZER.STATE_PER_PARAM),
        MutAnyOrigin,
    ]:
        """Return a LayoutTensor view over optimizer_state (zero-copy)."""
        return LayoutTensor[
            dtype,
            Layout.row_major(Self.PARAM_SIZE, Self.OPTIMIZER.STATE_PER_PARAM),
            MutAnyOrigin,
        ](self.optimizer_state.unsafe_ptr())

    # =========================================================================
    # Core Mutation
    # =========================================================================

    fn zero_grads(mut self):
        """Zero all parameter gradients before a backward pass."""
        for i in range(Self.PARAM_SIZE):
            self.grads[i] = 0

    fn set_lr_scale(mut self, scale: Float64):
        """Set the LR multiplier applied at each optimizer step.

        Use for LR annealing (e.g. PPO linear decay):
            state.set_lr_scale(1.0 - progress)  # progress in [0, 1]

        Args:
            scale: Multiplier applied to the compile-time base LR (default 1.0).
        """
        self.lr_scale = scale

    fn optimizer_step(mut self):
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

    fn copy_params_from(mut self, source: Self):
        """Hard copy: self.params = source.params (θ_target ← θ_online).

        Args:
            source: Network state to copy parameters from.
        """
        for i in range(Self.PARAM_SIZE):
            self.params[i] = source.params[i]

    fn soft_update_from(mut self, source: Self, tau: Float64):
        """Soft update: self = tau * source + (1 - tau) * self.

        Used for target network updates in DQN, DDPG, TD3, SAC.

        Args:
            source: The online network state to blend from.
            tau: Interpolation factor (e.g. 0.005).
        """
        var tau_s = Scalar[dtype](tau)
        var one_m = Scalar[dtype](1.0 - tau)
        for i in range(Self.PARAM_SIZE):
            self.params[i] = tau_s * source.params[i] + one_m * self.params[i]

    # =========================================================================
    # Section-based helpers (for multi-network single-file checkpoints)
    # =========================================================================

    fn write_sections(self, prefix: String) -> String:
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
        var content = write_float_section_list(
            prefix + "params:", self.params
        )
        content += write_float_section_list(
            prefix + "optimizer_state:", self.optimizer_state
        )
        return content

    fn read_sections(mut self, content: String, prefix: String) raises:
        """Load params and optimizer_state from prefixed sections.

        Counterpart of write_sections — reads "{prefix}params:" and
        "{prefix}optimizer_state:" from a combined checkpoint file.

        Args:
            content: Full checkpoint file content.
            prefix: Section name prefix used when writing (e.g. "actor_").
        """
        var loaded_params = read_float_section_list(
            content, prefix + "params:", Self.PARAM_SIZE
        )
        for i in range(Self.PARAM_SIZE):
            self.params[i] = loaded_params[i]

        var loaded_state = read_float_section_list(
            content, prefix + "optimizer_state:", Self.STATE_SIZE
        )
        for i in range(Self.STATE_SIZE):
            self.optimizer_state[i] = loaded_state[i]

    # =========================================================================
    # Checkpoint Save / Load (single-network file)
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
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

    fn load_checkpoint(mut self, filepath: String) raises:
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
