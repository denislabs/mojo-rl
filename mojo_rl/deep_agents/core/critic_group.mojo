"""CriticGroup: N identical critic networks with shared operations.

Eliminates twin-critic code duplication by providing iteration-based
operations over 1 or N NetworkPair/GPUNetworkPair instances.

Before (scattered throughout agents):
    var critic: NetworkPair[CriticModel, CriticOpt]
    var critic2: NetworkPair[CriticModel, CriticOpt]
    ...
    comptime if NUM_CRITICS == 2:
        critic2.initialize[Kaiming[]]()
    ...
    comptime if NUM_CRITICS == 2:
        critic2.soft_update(tau)
    # Repeated 10+ times per agent

After:
    var critics: CriticGroup[CriticModel, CriticOpt, NUM_CRITICS]
    ...
    critics.soft_update_all(tau)
    for i in range(NUM_CRITICS):
        CriticNet.forward[BS](input, ws.q_out(i), critics.online_params_view(i))
"""

from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.initializer import Initializer, Kaiming
from mojo_rl.nn.training import NetworkState, NetworkPair, GPUNetworkPair
from mojo_rl.nn.training.network_state import NetworkState
from mojo_rl.nn.training.network_pair import NetworkPair, GPUNetworkPair


# =============================================================================
# CriticGroup — CPU
# =============================================================================


struct CriticGroup[
    Net: Model,
    Opt: Optimizer,
    N: Int,
](Movable):
    """N identical (online, target) network pairs.

    Provides batch operations (soft_update_all, zero_all_grads, etc.)
    and indexed access for per-critic forward/backward calls.

    Parameters:
        Net: Critic network architecture.
        Opt: Critic optimizer.
        N: Number of critics (1 for DDPG, 2 for TD3/SAC).
    """

    comptime PARAM_SIZE: Int = Self.Net.PARAM_SIZE

    var pairs: List[NetworkPair[Self.Net, Self.Opt]]

    def __init__(out self):
        """Allocate N network pairs (uninitialized weights)."""
        self.pairs = List[NetworkPair[Self.Net, Self.Opt]](capacity=Self.N)
        for i in range(Self.N):
            self.pairs.append(NetworkPair[Self.Net, Self.Opt]())

    def __init__(out self, *, deinit take: Self):
        self.pairs = take.pairs^

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize[INIT: Initializer = Kaiming[]](mut self):
        """Initialize all N critic pairs with the given strategy."""
        for i in range(Self.N):
            self.pairs[i].initialize[INIT]()

    # =========================================================================
    # Indexed access
    # =========================================================================

    def online_params_view(
        self, idx: Int
    ) -> LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        """Online params for critic `idx`."""
        return self.pairs[idx].online.params_view()

    def target_params_view(
        self, idx: Int
    ) -> LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        """Target params for critic `idx`."""
        return self.pairs[idx].target.params_view()

    def online_grads_view(
        self, idx: Int
    ) -> LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        """Online grads for critic `idx`."""
        return self.pairs[idx].online.grads_view()

    # =========================================================================
    # Batch operations — replace comptime-if twin-critic boilerplate
    # =========================================================================

    def soft_update_all(mut self, tau: Float64):
        """Soft update all N target networks from their online counterparts."""
        for i in range(Self.N):
            self.pairs[i].soft_update(tau)

    def zero_all_grads(mut self):
        """Zero gradients on all N online networks."""
        for i in range(Self.N):
            self.pairs[i].zero_grads()

    def optimizer_step_all(mut self):
        """Run optimizer step on all N online networks."""
        for i in range(Self.N):
            self.pairs[i].optimizer_step()

    def copy_targets_from_online(mut self):
        """Hard copy: target ← online for all N critics."""
        for i in range(Self.N):
            self.pairs[i].copy_target_from_online()

    # =========================================================================
    # Checkpoint helpers
    # =========================================================================

    def write_sections(self, prefix: String) -> String:
        """Serialize all critic pairs with indexed prefixes."""
        var content = String("")
        for i in range(Self.N):
            content += self.pairs[i].write_sections(
                prefix + "critic" + String(i) + "_"
            )
        return content^

    def read_sections(mut self, content: String, prefix: String) raises:
        """Load all critic pairs from indexed prefixes."""
        for i in range(Self.N):
            self.pairs[i].read_sections(
                content, prefix + "critic" + String(i) + "_"
            )


# =============================================================================
# GPUCriticGroup — GPU
# =============================================================================


struct GPUCriticGroup[
    Net: Model,
    Opt: Optimizer,
    N: Int,
](Movable):
    """GPU-side N identical (online, target) network pairs.

    Parameters:
        Net: Critic network architecture.
        Opt: Critic optimizer.
        N: Number of critics (1 for DDPG, 2 for TD3/SAC).
    """

    comptime PARAM_SIZE: Int = Self.Net.PARAM_SIZE

    var pairs: List[GPUNetworkPair[Self.Net, Self.Opt]]

    def __init__(out self, ctx: DeviceContext) raises:
        """Allocate N GPU network pairs."""
        self.pairs = List[GPUNetworkPair[Self.Net, Self.Opt]](
            capacity=Self.N
        )
        for i in range(Self.N):
            self.pairs.append(GPUNetworkPair[Self.Net, Self.Opt](ctx))

    def __init__(out self, *, deinit take: Self):
        self.pairs = take.pairs^

    # =========================================================================
    # Indexed access
    # =========================================================================

    def online_params_view(
        self, idx: Int
    ) -> LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        return self.pairs[idx].online.params_view()

    def target_params_view(
        self, idx: Int
    ) -> LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        return self.pairs[idx].target.params_view()

    def online_grads_view(
        self, idx: Int
    ) -> LayoutTensor[dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin]:
        return self.pairs[idx].online.grads_view()

    # =========================================================================
    # Batch operations
    # =========================================================================

    def soft_update_all(mut self, tau: Float64, ctx: DeviceContext) raises:
        """Soft update all N target networks on GPU."""
        for i in range(Self.N):
            self.pairs[i].soft_update(tau, ctx)

    def zero_all_grads(mut self, ctx: DeviceContext) raises:
        """Zero all N online gradients on GPU."""
        for i in range(Self.N):
            self.pairs[i].online.zero_grads(ctx)

    def optimizer_step_all(mut self, ctx: DeviceContext) raises:
        """Optimizer step on all N online networks on GPU."""
        for i in range(Self.N):
            self.pairs[i].online.optimizer_step(ctx)

    def copy_targets_from_online(mut self, ctx: DeviceContext) raises:
        """Hard copy on GPU: target ← online for all N critics."""
        for i in range(Self.N):
            self.pairs[i].copy_target_from_online(ctx)

    # =========================================================================
    # CPU ↔ GPU transfer
    # =========================================================================

    def upload_from(
        mut self,
        cpu: CriticGroup[Self.Net, Self.Opt, Self.N],
        ctx: DeviceContext,
    ) raises:
        """Upload CPU critic weights to GPU."""
        for i in range(Self.N):
            self.pairs[i].upload_from(cpu.pairs[i], ctx)

    def download_to(
        mut self,
        mut cpu: CriticGroup[Self.Net, Self.Opt, Self.N],
        ctx: DeviceContext,
    ) raises:
        """Download GPU critic weights to CPU."""
        for i in range(Self.N):
            self.pairs[i].download_to(cpu.pairs[i], ctx)
