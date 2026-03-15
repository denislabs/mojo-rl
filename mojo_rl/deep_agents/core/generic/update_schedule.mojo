"""Update schedule strategies for off-policy agents.

Controls when actor updates and target soft-updates happen relative to
critic updates. Each implementation owns its step counter and provides
should_update_actor() and soft_update() methods.

Implementations:
  - EveryStep: update everything every training step (DDPG)
  - DelayedActorAndTargets: delay actor + all targets (TD3)
  - DelayedActorOnly: delay actor, update critic targets every step (SAC)
"""

from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.training import NetworkPair, GPUNetworkPair
from std.gpu.host import DeviceContext


# =============================================================================
# EveryStep — update everything every step (DDPG)
# =============================================================================


struct EveryStep(Movable, Copyable):
    """Update actor and soft-update all targets every training step.

    Used by DDPG: no delay between critic and actor updates.
    """

    var step_count: Int

    fn __init__(out self):
        self.step_count = 0

    fn __init__(out self, *, copy: Self):
        self.step_count = copy.step_count

    fn __init__(out self, *, deinit take: Self):
        self.step_count = take.step_count

    fn should_update_actor(mut self) -> Bool:
        """Always returns True — update actor every step."""
        self.step_count += 1
        return True

    fn soft_update_cpu[
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        self,
        mut actor: NetworkPair[ActorModel, ActorOpt],
        mut critic: NetworkPair[CriticModel, CriticOpt],
        tau: Float64,
    ):
        """Soft-update actor and critic targets."""
        actor.soft_update(tau)
        critic.soft_update(tau)

    fn soft_update_cpu_twin[
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        self,
        mut actor: NetworkPair[ActorModel, ActorOpt],
        mut critic1: NetworkPair[CriticModel, CriticOpt],
        mut critic2: NetworkPair[CriticModel, CriticOpt],
        tau: Float64,
    ):
        """Soft-update actor and both critic targets."""
        actor.soft_update(tau)
        critic1.soft_update(tau)
        critic2.soft_update(tau)

    fn soft_update_gpu[
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        self,
        mut actor: GPUNetworkPair[ActorModel, ActorOpt],
        mut critic: GPUNetworkPair[CriticModel, CriticOpt],
        tau: Float64,
        ctx: DeviceContext,
    ) raises:
        """GPU soft-update actor and critic targets."""
        actor.soft_update(tau, ctx)
        critic.soft_update(tau, ctx)

    fn soft_update_gpu_twin[
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        self,
        mut actor: GPUNetworkPair[ActorModel, ActorOpt],
        mut critic1: GPUNetworkPair[CriticModel, CriticOpt],
        mut critic2: GPUNetworkPair[CriticModel, CriticOpt],
        tau: Float64,
        ctx: DeviceContext,
    ) raises:
        """GPU soft-update actor and both critic targets."""
        actor.soft_update(tau, ctx)
        critic1.soft_update(tau, ctx)
        critic2.soft_update(tau, ctx)


# =============================================================================
# DelayedActorAndTargets — delay actor + all targets (TD3)
# =============================================================================


struct DelayedActorAndTargets(Movable, Copyable):
    """Update actor and soft-update all targets every policy_delay steps.

    Used by TD3: critics update every step, but actor and all target networks
    only update every policy_delay critic updates. This reduces coupling
    between actor and critic and stabilizes training.
    """

    var policy_delay: Int
    var step_count: Int

    fn __init__(out self, policy_delay: Int = 2):
        self.policy_delay = policy_delay
        self.step_count = 0

    fn __init__(out self, *, copy: Self):
        self.policy_delay = copy.policy_delay
        self.step_count = copy.step_count

    fn __init__(out self, *, deinit take: Self):
        self.policy_delay = take.policy_delay
        self.step_count = take.step_count

    fn should_update_actor(mut self) -> Bool:
        """Returns True every policy_delay steps."""
        self.step_count += 1
        return self.step_count % self.policy_delay == 0

    fn soft_update_cpu_twin[
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        self,
        mut actor: NetworkPair[ActorModel, ActorOpt],
        mut critic1: NetworkPair[CriticModel, CriticOpt],
        mut critic2: NetworkPair[CriticModel, CriticOpt],
        tau: Float64,
    ):
        """Soft-update actor + both critic targets (only on delayed steps)."""
        if self.step_count % self.policy_delay == 0:
            actor.soft_update(tau)
            critic1.soft_update(tau)
            critic2.soft_update(tau)

    fn soft_update_gpu_twin[
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        self,
        mut actor: GPUNetworkPair[ActorModel, ActorOpt],
        mut critic1: GPUNetworkPair[CriticModel, CriticOpt],
        mut critic2: GPUNetworkPair[CriticModel, CriticOpt],
        tau: Float64,
        ctx: DeviceContext,
    ) raises:
        """GPU soft-update actor + both critic targets (only on delayed steps)."""
        if self.step_count % self.policy_delay == 0:
            actor.soft_update(tau, ctx)
            critic1.soft_update(tau, ctx)
            critic2.soft_update(tau, ctx)


# =============================================================================
# DelayedActorOnly — delay actor, targets every step (SAC)
# =============================================================================


struct DelayedActorOnly(Movable, Copyable):
    """Update actor every policy_delay steps, critic targets every step.

    Used by SAC: no target actor network. Critic targets soft-update every
    step unconditionally. Actor updates are delayed for stability.
    """

    var policy_delay: Int
    var step_count: Int

    fn __init__(out self, policy_delay: Int = 2):
        self.policy_delay = policy_delay
        self.step_count = 0

    fn __init__(out self, *, copy: Self):
        self.policy_delay = copy.policy_delay
        self.step_count = copy.step_count

    fn __init__(out self, *, deinit take: Self):
        self.policy_delay = take.policy_delay
        self.step_count = take.step_count

    fn should_update_actor(mut self) -> Bool:
        """Returns True every policy_delay steps."""
        self.step_count += 1
        return self.step_count % self.policy_delay == 0

    fn soft_update_cpu_twin[
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        self,
        mut actor: NetworkPair[ActorModel, ActorOpt],
        mut critic1: NetworkPair[CriticModel, CriticOpt],
        mut critic2: NetworkPair[CriticModel, CriticOpt],
        tau: Float64,
    ):
        """Soft-update both critic targets every step. No target actor."""
        critic1.soft_update(tau)
        critic2.soft_update(tau)

    fn soft_update_gpu_twin[
        ActorModel: Model,
        ActorOpt: Optimizer,
        CriticModel: Model,
        CriticOpt: Optimizer,
    ](
        self,
        mut actor: GPUNetworkPair[ActorModel, ActorOpt],
        mut critic1: GPUNetworkPair[CriticModel, CriticOpt],
        mut critic2: GPUNetworkPair[CriticModel, CriticOpt],
        tau: Float64,
        ctx: DeviceContext,
    ) raises:
        """GPU soft-update both critic targets every step. No target actor."""
        critic1.soft_update(tau, ctx)
        critic2.soft_update(tau, ctx)
