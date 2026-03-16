"""Update schedule strategies for off-policy agents.

Stateless strategy types with @staticmethod methods, following the
nn/Model pattern. Runtime state (step_count, policy_delay) lives on
the agent — these strategies are pure decision functions.

Controls when actor updates and target soft-updates happen relative
to critic updates. The agent calls should_update_actor() and
should_update_targets() to decide what to do, then performs the
actual soft_update calls itself based on Config.NUM_CRITICS and
Config.HAS_TARGET_ACTOR.

Implementations:
  - EveryStep: update everything every step (DDPG)
  - DelayedAll: delay actor + all targets (TD3)
  - DelayedActorOnly: delay actor, targets every step (SAC)
"""


trait Schedule:
    """Trait for update schedule strategies."""

    comptime DELAYS_ACTOR: Bool
    comptime DELAYS_TARGETS: Bool
    comptime DEFAULT_POLICY_DELAY: Int

    @staticmethod
    fn should_update_actor(step_count: Int, policy_delay: Int) -> Bool:
        ...

    @staticmethod
    fn should_update_targets(step_count: Int, policy_delay: Int) -> Bool:
        ...


# =============================================================================
# EveryStep — update everything every step (DDPG)
# =============================================================================


struct EveryStep(Schedule):
    """Update actor and soft-update all targets every training step.

    Used by DDPG: no delay between critic and actor updates.
    """

    comptime DELAYS_ACTOR: Bool = False
    comptime DELAYS_TARGETS: Bool = False
    comptime DEFAULT_POLICY_DELAY: Int = 1

    @staticmethod
    fn should_update_actor(step_count: Int, policy_delay: Int) -> Bool:
        """Always returns True — update actor every step."""
        return True

    @staticmethod
    fn should_update_targets(step_count: Int, policy_delay: Int) -> Bool:
        """Always returns True — soft-update targets every step."""
        return True


# =============================================================================
# DelayedAll — delay actor + all targets (TD3)
# =============================================================================


struct DelayedAll(Schedule):
    """Update actor and soft-update all targets every policy_delay steps.

    Used by TD3: critics update every step, but actor and all target
    networks only update every policy_delay critic updates. This
    reduces coupling between actor and critic and stabilizes training.
    """

    comptime DELAYS_ACTOR: Bool = True
    comptime DELAYS_TARGETS: Bool = True
    comptime DEFAULT_POLICY_DELAY: Int = 2

    @staticmethod
    fn should_update_actor(step_count: Int, policy_delay: Int) -> Bool:
        """Returns True every policy_delay steps."""
        return step_count % policy_delay == 0

    @staticmethod
    fn should_update_targets(step_count: Int, policy_delay: Int) -> Bool:
        """Returns True every policy_delay steps (coupled with actor)."""
        return step_count % policy_delay == 0


# =============================================================================
# DelayedActorOnly — delay actor, targets every step (SAC)
# =============================================================================


struct DelayedActorOnly(Schedule):
    """Update actor every policy_delay steps, critic targets every step.

    Used by SAC: no target actor network. Critic targets soft-update every
    step unconditionally. Actor updates are delayed for stability.
    """

    comptime DELAYS_ACTOR: Bool = True
    comptime DELAYS_TARGETS: Bool = False
    comptime DEFAULT_POLICY_DELAY: Int = 2

    @staticmethod
    fn should_update_actor(step_count: Int, policy_delay: Int) -> Bool:
        """Returns True every policy_delay steps."""
        return step_count % policy_delay == 0

    @staticmethod
    fn should_update_targets(step_count: Int, policy_delay: Int) -> Bool:
        """Always returns True — critic targets update every step."""
        return True
