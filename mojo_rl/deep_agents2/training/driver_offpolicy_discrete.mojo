"""Off-policy discrete-action training + eval drivers.

Parallel to `driver_offpolicy.mojo` (continuous actions), this module
provides training drivers for discrete-action agents (DQN family:
DQN, Double DQN, Dueling DQN, DQN+PER, Noisy DQN, C51, Rainbow).

Key differences from the continuous off-policy driver:
  - Env bound: `BoxDiscreteActionEnv` (`step_obs` takes Int action)
  - Action is a single Int index per env, not ACT_DIM floats
  - No actor network / log_prob — no ao_scratch / alp_scratch
  - Epsilon-greedy exploration managed by the agent internally

  env_target | train_target | N_ENVS | driver
  -----------|--------------|--------|----------------------------------
  cpu        | cpu          | 1      | run_offpolicy_discrete_train
  cpu        | gpu          | 1      | run_offpolicy_discrete_train

Trait surface
  - `OffPolicyDiscreteAgent` — minimal: select_action_batched[N_ENVS],
    record, train_step, episode tracker accessors, batched CPU record
    + add_complete_return.

Storage: action indices stored as Scalar[DT] in driver scratches and
replay buffers (cast to/from Int at the env boundary). This keeps the
pointer-based API consistent with the continuous driver and avoids a
separate Int-buffer path.

Batched driver (Tier-3) deferred until a consumer needs it.
"""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from .driver_scratch import DriverScratch


# ──────────────────────────────────────────────────────────────────────
# OffPolicyDiscreteAgent — trait for the discrete off-policy drivers.
# ──────────────────────────────────────────────────────────────────────


trait OffPolicyDiscreteAgent(Movable, ImplicitlyDestructible):
    """Single-trait surface for the discrete off-policy drivers.

    Mirrors `OffPolicyAgent` (continuous) but adapted for discrete
    action spaces:

      - `select_action_batched` writes action INDICES (as Scalar[DT])
        into `action_ptr`, not continuous action vectors.
      - `record` takes `action_idx: Int`, not `action: List[Scalar[DT]]`.
      - No ao_scratch / alp_scratch parameters (no actor network).

    DQN / Double DQN / Dueling / PER / Noisy / C51 / Rainbow all
    conform. Epsilon-greedy is agent-internal — the driver does not
    manage exploration."""

    comptime AGENT_TRAIN_TARGET: StaticString
    comptime AGENT_OBS_DIM: Int
    comptime AGENT_NUM_ACTIONS: Int

    def select_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        """Write N_ENVS discrete action indices into action_ptr.

        obs_ptr: N_ENVS * AGENT_OBS_DIM scalars on train_target.
        action_ptr: N_ENVS scalars on train_target (one index per env,
                    stored as Scalar[DT]).

        The agent handles epsilon-greedy / noisy-net exploration
        internally based on step_idx."""
        ...

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        """Deterministic, exploration-free action selection for eval.

        Host-list signature; trainers dispatch internally on
        AGENT_TRAIN_TARGET (CPU trainers run native, GPU trainers
        H2D the obs and D2H the Q-values under the hood)."""
        ...

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        action_idx: Int,
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        """Store one transition. Action is an Int index; the trainer
        converts to Scalar[DT] internally for replay storage."""
        ...

    def end_episode(mut self):
        ...

    def train_step(mut self, step_idx: Int) raises -> Bool:
        ...

    def mean_return(self) -> Scalar[DT]:
        ...

    def ep_count(self) -> Int:
        ...

    # ─── Batched support (for future Tier-3 driver) ─────────────────

    def add_complete_return(mut self, ret: Scalar[DT]):
        """Push an externally-tracked complete-episode return into the
        trainer's episode tracker. For future N_ENVS batched drivers."""
        ...

    def record_batch_cpu[
        N_ENVS: Int
    ](
        mut self,
        prev_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Push N transitions from host pointer slabs. action_ptr
        stores N_ENVS action indices as Scalar[DT]. Does NOT update
        the episode tracker."""
        ...

    # ─── Optional cadence hooks (default no-op) ──────────────────────
    #
    # Mirror the continuous `OffPolicyAgent` surface so the discrete
    # driver can call into the trainer at `diag_every` / `checkpoint_every`
    # cadences. Each has a `pass` default; DQNTrainer overrides both with
    # real bodies that drain its `DQNMetrics` bundle and write a one-file
    # v2 checkpoint envelope.

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        pass

    def save_state(mut self, path: String) raises:
        pass


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_discrete_train — single-env, env_target="cpu".
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_discrete_train[
    A: OffPolicyDiscreteAgent,
    E: BoxDiscreteActionEnv,
    L: Logger = NoOpLogger,
](
    mut trainer: A,
    mut env: E,
    total_timesteps: Int,
    *,
    ctx: Optional[DeviceContext] = None,
    print_every: Int = 1_000,
    verbose: Bool = True,
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    base_step: Int = 0,
) raises -> List[Scalar[DT]]:
    """Single-env discrete off-policy training driver.

    Covers (env_target=cpu, train_target=cpu) and
    (env_target=cpu, train_target=gpu). The env is always CPU-side
    (`BoxDiscreteActionEnv`); when train_target="gpu" the driver
    stages obs H2D and action D2H around the agent's
    `select_action_batched` call.

    Loop: one env step + one `train_step` per iteration.

    Args:
        trainer: Any nn2 discrete off-policy trainer (DQN family).
        env: Any `BoxDiscreteActionEnv`.
        total_timesteps: Number of env steps to run.
        ctx: Required for train_target="gpu"; ignored for "cpu".
        print_every: Verbose status-line cadence (env steps). 0 disables.
        verbose: Print a per-cadence status line.

    Returns:
        List of `trainer.mean_return()` snapshots at each completed
        episode boundary.
    """
    comptime env_target: StaticString = "cpu"
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM

    comptime assert (
        train_target == "cpu" or train_target == "gpu"
    ), "run_offpolicy_discrete_train: train_target must be 'cpu' or 'gpu'"
    comptime if train_target == "gpu":
        if not ctx:
            raise Error(
                "run_offpolicy_discrete_train[train_target='gpu']:"
                " ctx required for env→trainer H2D/D2H staging"
            )

    comptime needs_boundary_copy: Bool = env_target != train_target
    var obs_scratch = DriverScratch["obs", 1, OBS].make[train_target](
        ctx=ctx, with_host_mirror=needs_boundary_copy,
    )
    var action_scratch = DriverScratch["action", 1, 1].make[train_target](
        ctx=ctx, with_host_mirror=needs_boundary_copy,
    )

    var obs_list = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var next_obs_list = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var env_obs = env.reset_obs_list()

    var ep_returns = List[Scalar[DT]]()
    var current_ep_count = trainer.ep_count()

    var t_start = perf_counter_ns()
    var step: Int = 0
    while step < total_timesteps:
        var obs_scratch_h = obs_scratch.host_ptr()
        for d in range(OBS):
            var v = Scalar[DT](env_obs[d])
            obs_list[d] = v
            obs_scratch_h[d] = v

        comptime if needs_boundary_copy:
            var c = ctx.value()
            c.enqueue_copy(obs_scratch.dev.value(), obs_scratch_h)

        # `base_step + step` — cumulative env-step counter for the
        # trainer's warmup gating. Equivalent to `step` when base_step=0.
        trainer.select_action_batched[1](
            obs_scratch.target_ptr[train_target](),
            action_scratch.target_ptr[train_target](),
            base_step + step,
        )

        comptime if needs_boundary_copy:
            var c = ctx.value()
            c.enqueue_copy(
                action_scratch.host_ptr(), action_scratch.dev.value()
            )
            c.synchronize()

        var action_idx = Int(action_scratch.host_ptr()[0])

        var step_res = env.step_obs(action_idx)
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        # `done` (terminated OR truncated) drives reset/episode tracking; the
        # replay buffer stores `terminated` ONLY so the DQN/C51 TD bootstrap
        # `(1 − done)·γ·max Q'` is kept on time-limit truncation and dropped
        # on natural termination.
        var terminated = env.was_terminated()
        for d in range(OBS):
            next_obs_list[d] = Scalar[DT](nxt[d])

        trainer.record(
            obs_list,
            action_idx,
            Scalar[DT](reward),
            next_obs_list,
            Scalar[DT](1.0) if terminated else Scalar[DT](0.0),
        )

        if done:
            trainer.end_episode()
            env_obs = env.reset_obs_list()
            var new_ep_count = trainer.ep_count()
            if new_ep_count > current_ep_count:
                ep_returns.append(trainer.mean_return())
                current_ep_count = new_ep_count
        else:
            env_obs = nxt^

        step += 1
        _ = trainer.train_step(base_step + step)

        var abs_step = base_step + step

        if verbose and print_every > 0 and abs_step % print_every == 0:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ",
                abs_step,
                "] mean_ret(10)=",
                trainer.mean_return(),
                " ep=",
                trainer.ep_count(),
                " elapsed=",
                elapsed,
                "s",
            )

        # Logger emit at the same cadence. Comptime-elided when
        # L=NoOpLogger (default).
        comptime if L.ENABLED:
            if (
                print_every > 0
                and abs_step % print_every == 0
                and Bool(logger)
            ):
                logger.value()[].log_scalar(
                    "avg_reward",
                    Float64(trainer.mean_return()),
                    abs_step,
                )
                logger.value()[].log_scalar(
                    "episodes",
                    Float64(trainer.ep_count()),
                    abs_step,
                )
                # No forced flush — `log_scalar` auto-flushes when the
                # logger's buffer fills; user controls cadence via
                # `buffer_size`. Final residual sent by `logger.close()`.

        # `diag_every` — drain the trainer's metric bundle through the
        # logger at its own cadence. Default trait impl is no-op for
        # trainers that haven't wired this up yet.
        comptime if L.ENABLED:
            if (
                diag_every > 0
                and abs_step % diag_every == 0
                and Bool(logger)
            ):
                trainer.flush_metrics_through_logger[L](logger, abs_step)

        # `checkpoint_every` — overwrite `checkpoint_path` with the
        # trainer's one-file v2 envelope. Default trait impl is no-op.
        if (
            checkpoint_every > 0
            and abs_step % checkpoint_every == 0
            and checkpoint_path.byte_length() > 0
        ):
            trainer.save_state(checkpoint_path)

    # Always overwrite the final checkpoint at end so resume gets the
    # freshest weights regardless of cadence alignment.
    if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
        trainer.save_state(checkpoint_path)

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# run_offpolicy_discrete_eval — single-env greedy eval.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_discrete_eval[
    A: OffPolicyDiscreteAgent,
    E: BoxDiscreteActionEnv,
](
    mut trainer: A,
    mut env: E,
    num_episodes: Int,
    *,
    max_steps_per_episode: Int = 1_000,
    verbose: Bool = False,
) raises -> Scalar[DT]:
    """Non-mutating greedy eval driver for discrete agents.

    Uses `select_greedy_action` (argmax Q, no epsilon). Does not
    touch the trainer's replay buffer, optimizers, or episode tracker.

    Returns:
        Mean return over `num_episodes` evaluation episodes.
    """
    comptime OBS = A.AGENT_OBS_DIM

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var total_return = Scalar[DT](0.0)
    var t_start = perf_counter_ns()

    for ep in range(num_episodes):
        var obs_list = env.reset_obs_list()
        var ep_return = Scalar[DT](0.0)
        var ep_steps: Int = 0

        for _ in range(max_steps_per_episode):
            for d in range(OBS):
                obs[d] = Scalar[DT](obs_list[d])
            var action_idx = trainer.select_greedy_action(obs)
            var step_res = env.step_obs(action_idx)
            var nxt = step_res[0].copy()
            var reward = step_res[1]
            var done = step_res[2]
            ep_return += Scalar[DT](reward)
            ep_steps += 1
            if done:
                break
            obs_list = nxt^

        total_return += ep_return
        if verbose:
            print(
                "  [eval ep ",
                ep + 1,
                "/",
                num_episodes,
                "] return=",
                ep_return,
                " steps=",
                ep_steps,
            )

    var mean = total_return / Scalar[DT](num_episodes)
    if verbose:
        var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
        print(
            "eval: mean_return=",
            mean,
            " (",
            num_episodes,
            " episodes, ",
            elapsed,
            " s)",
        )
    return mean
