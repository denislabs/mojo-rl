"""NetworkPair: (online, target) pair of NetworkState/GPUNetworkState.

Groups an online + target network with shared operations:
  - initialize: init online with given initializer, then copy → target
  - soft_update: target = tau*online + (1-tau)*target (in-place)
  - copy_target_from_online: hard copy target ← online
  - write_sections / read_sections: checkpoint helpers with prefix support
  - params_view / grads_view / zero_grads / optimizer_step: delegate to online

GPUNetworkPair mirrors the same interface over device memory.

Usage:
    var actor = NetworkPair[ActorModel, Adam[lr]]()
    actor.initialize[Kaiming[]]()

    # Soft update target from online
    actor.soft_update(tau=0.005)

    # Hard copy (initialization)
    actor.copy_target_from_online()

    # GPU variant
    var gpu_actor = GPUNetworkPair[ActorModel, Adam[lr]](ctx)
    gpu_actor.upload_from(actor, ctx)
    gpu_actor.soft_update(tau=0.005, ctx=ctx)
"""

from ..model import Model
from ..optimizer import Optimizer
from ..initializer import Initializer, Kaiming
from ..constants import dtype as default_dtype
from .network_state import NetworkState
from .gpu_network_state import GPUNetworkState

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext


struct NetworkPair[MODEL: Model, OPTIMIZER: Optimizer, dtype: DType = default_dtype](
    ImplicitlyCopyable, Movable
):
    """Pair of (online, target) NetworkState with shared operations.

    Eliminates the repeated boilerplate of keeping actor_online / actor_target
    fields in sync across DDPG, TD3, and SAC agents.

    Parameters:
        MODEL: The model architecture (implements Model trait).
        OPTIMIZER: The optimizer (implements Optimizer trait).
        dtype: Data type for all buffers (default: DType.float32).
    """

    comptime PARAM_SIZE: Int = Self.MODEL.PARAM_SIZE

    var online: NetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype]
    var target: NetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype]

    def __init__(out self):
        """Allocate and zero-initialize both online and target states."""
        self.online = NetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype]()
        self.target = NetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype]()

    def __init__(out self, *, copy: Self):
        self.online = NetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype](copy=copy.online)
        self.target = NetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype](copy=copy.target)

    def __init__(out self, *, deinit take: Self):
        self.online = take.online^
        self.target = take.target^

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize[INITIALIZER: Initializer = Kaiming[]](mut self):
        """Initialize online params with INITIALIZER, then hard-copy to target.

        Parameters:
            INITIALIZER: Weight initialization strategy (default: Kaiming).
        """
        self.online.initialize[INITIALIZER]()
        self.target.copy_params_from(self.online)

    # =========================================================================
    # Target Network Operations
    # =========================================================================

    def soft_update(mut self, tau: Float64):
        """Soft update: target = tau*online + (1-tau)*target.

        Args:
            tau: Interpolation factor (e.g. 0.005).
        """
        self.target.soft_update_from(self.online, tau)

    def copy_target_from_online(mut self):
        """Hard copy: target.params = online.params."""
        self.target.copy_params_from(self.online)

    # =========================================================================
    # Delegates to online
    # =========================================================================

    def params_view(
        self,
    ) -> LayoutTensor[Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin,]:
        """LayoutTensor view over online params (zero-copy)."""
        return self.online.params_view()

    def grads_view(
        self,
    ) -> LayoutTensor[Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin,]:
        """LayoutTensor view over online grads (zero-copy)."""
        return self.online.grads_view()

    def zero_grads(mut self):
        """Zero online parameter gradients."""
        self.online.zero_grads()

    def optimizer_step(mut self):
        """One optimizer step on the online network."""
        self.online.optimizer_step()

    # =========================================================================
    # Checkpoint helpers
    # =========================================================================

    def write_sections(self, prefix: String) -> String:
        """Serialize both online and target as prefixed sections.

        Produces sections: "{prefix}online_{...}" and "{prefix}target_{...}".

        Args:
            prefix: Section name prefix (e.g. "actor_" → "actor_online_params:").

        Returns:
            String containing serialized sections for both networks.
        """
        var content = self.online.write_sections(prefix + "online_")
        content += self.target.write_sections(prefix + "target_")
        return content

    def read_sections(mut self, content: String, prefix: String) raises:
        """Load both online and target from prefixed sections.

        Args:
            content: Full checkpoint file content.
            prefix: Same prefix used when writing.
        """
        self.online.read_sections(content, prefix + "online_")
        self.target.read_sections(content, prefix + "target_")


# =============================================================================
# GPUNetworkPair
# =============================================================================


struct GPUNetworkPair[MODEL: Model, OPTIMIZER: Optimizer, dtype: DType = default_dtype](
    ImplicitlyCopyable, Movable
):
    """GPU-side pair of (online, target) GPUNetworkState.

    Mirrors NetworkPair but lives on the GPU device.

    Parameters:
        MODEL: The model architecture (implements Model trait).
        OPTIMIZER: The optimizer (implements Optimizer trait).
        dtype: Data type for all buffers (default: DType.float32).
    """

    comptime PARAM_SIZE: Int = Self.MODEL.PARAM_SIZE

    var online: GPUNetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype]
    var target: GPUNetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype]

    def __init__(out self, ctx: DeviceContext) raises:
        """Allocate both online and target device buffers.

        Args:
            ctx: GPU device context.
        """
        self.online = GPUNetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype](ctx)
        self.target = GPUNetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype](ctx)

    def __init__(out self, *, copy: Self):
        self.online = GPUNetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype](
            copy=copy.online
        )
        self.target = GPUNetworkState[Self.MODEL, Self.OPTIMIZER, Self.dtype](
            copy=copy.target
        )

    def __init__(out self, *, deinit take: Self):
        self.online = take.online^
        self.target = take.target^

    # =========================================================================
    # Target Network Operations
    # =========================================================================

    def soft_update(mut self, tau: Float64, ctx: DeviceContext) raises:
        """GPU soft update: target = tau*online + (1-tau)*target.

        Runs entirely on device — no CPU synchronization required.

        Args:
            tau: Blending factor (typically 0.001 – 0.01).
            ctx: GPU device context.
        """
        self.target.soft_update_from_gpu(self.online, tau, ctx)

    def copy_target_from_online(mut self, ctx: DeviceContext) raises:
        """Hard copy on GPU: target.params = online.params.

        Args:
            ctx: GPU device context.
        """
        ctx.enqueue_copy(self.target.params_buf, self.online.params_buf)

    # =========================================================================
    # Delegates to online
    # =========================================================================

    def params_view(
        self,
    ) -> LayoutTensor[Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin,]:
        """LayoutTensor view over online params (zero-copy)."""
        return self.online.params_view()

    def grads_view(
        self,
    ) -> LayoutTensor[Self.dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin,]:
        """LayoutTensor view over online grads (zero-copy)."""
        return self.online.grads_view()

    def zero_grads(self, ctx: DeviceContext) raises:
        """Zero online gradients on device."""
        self.online.zero_grads(ctx)

    def optimizer_step(mut self, ctx: DeviceContext) raises:
        """One GPU optimizer step on the online network."""
        self.online.optimizer_step(ctx)

    # =========================================================================
    # CPU ↔ GPU Sync
    # =========================================================================

    def upload_from(
        mut self,
        cpu: NetworkPair[Self.MODEL, Self.OPTIMIZER, Self.dtype],
        ctx: DeviceContext,
    ) raises:
        """Upload both online and target from CPU to GPU.

        Args:
            cpu: Source CPU NetworkPair.
            ctx: GPU device context.
        """
        self.online.upload_from(cpu.online, ctx)
        self.target.upload_from(cpu.target, ctx)

    def download_to(
        mut self,
        mut cpu: NetworkPair[Self.MODEL, Self.OPTIMIZER, Self.dtype],
        ctx: DeviceContext,
    ) raises:
        """Download both online and target from std.gpu to CPU (synchronizes).

        Args:
            cpu: Destination CPU NetworkPair (modified in-place).
            ctx: GPU device context.
        """
        self.online.download_to(cpu.online, ctx)
        self.target.download_to(cpu.target, ctx)
