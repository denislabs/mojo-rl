"""On-policy discrete-action training + eval drivers.

Discrete-action sibling of `driver_onpolicy.mojo` (continuous PPO).
Single-env driver for categorical on-policy trainers (discrete PPO
today, possibly discrete A2C later).

Key differences from the continuous on-policy driver:
  - Env bound: `BoxDiscreteActionEnv` (`step_obs` takes an Int action).
  - Action is a single Int index per step, not ACT_DIM floats.
  - `select_action` returns the sampled index; `record_transition`
    takes `action_idx: Int`.

  env_target | train_target | driver
  -----------|--------------|--------------------------------
  cpu        | cpu          | run_onpolicy_discrete_train
  cpu        | gpu          | run_onpolicy_discrete_train (trainer
                              H2Ds obs internally inside the act step)

The batched (Tier-3) discrete on-policy driver is deferred until a
consumer needs it — single-env covers CartPole / classic-control.
"""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.core.env_traits import BoxDiscreteActionEnv


# ──────────────────────────────────────────────────────────────────────
# OnPolicyDiscreteAgent — trait for the discrete on-policy driver.
# ──────────────────────────────────────────────────────────────────────


trait OnPolicyDiscreteAgent(Movable, ImplicitlyDestructible):
    """Single-env host-list surface for the discrete on-policy driver.

    Mirrors `OnPolicyAgent` (continuous PPO) but adapted for discrete
    action spaces: `select_action` / `select_greedy_action` return an
    Int action index, and `record_transition` takes that index.

    The trainer caches `(action index, log_prob, value)` between
    `select_action` and `record_transition`, exactly like the
    continuous on-policy trainer caches `(unbounded sample, log_prob,
    value)`. Callers invoke the pair in order.
    """

    comptime AGENT_TRAIN_TARGET: StaticString
    comptime AGENT_OBS_DIM: Int
    comptime AGENT_NUM_ACTIONS: Int

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        step_idx: Int,
    ) raises -> Int:
        """Sample an action index from the categorical policy and cache
        the (index, log_prob, value) triple for the upcoming record."""
        ...

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        """Deterministic argmax over logits — no sampling, no cache."""
        ...

    def record_transition(
        mut self,
        ref obs: List[Scalar[DT]],
        action_idx: Int,
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        ...

    def mark_terminal(mut self) raises:
        """Mark the just-recorded transition as a TRUE terminal so GAE
        zeroes its V bootstrap. The driver calls this only when the env
        reports `was_terminated()` — time-limit truncation keeps the
        bootstrap."""
        ...

    def end_episode(mut self):
        ...

    def train_step(mut self, step_idx: Int) raises -> Bool:
        ...

    def mean_return(self) -> Scalar[DT]:
        ...

    def ep_count(self) -> Int:
        ...

    # ─── Optional cadence hooks (default no-op) ──────────────────────

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        pass

    def save_state(mut self, path: String) raises:
        pass


# ──────────────────────────────────────────────────────────────────────
# run_onpolicy_discrete_train — single-env discrete on-policy training.
# ──────────────────────────────────────────────────────────────────────


def run_onpolicy_discrete_train[
    A: OnPolicyDiscreteAgent,
    E: BoxDiscreteActionEnv,
    L: Logger = NoOpLogger,
](
    mut trainer: A,
    mut env: E,
    total_timesteps: Int,
    *,
    print_every: Int = 1_000,
    verbose: Bool = True,
    logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    checkpoint_every: Int = 0,
    checkpoint_path: String = "",
    base_step: Int = 0,
) raises -> List[Scalar[DT]]:
    """Step-based discrete on-policy single-env training driver.

    One env step + one `train_step` call per iteration. The rollout
    accumulation and K-epoch update fire inside `trainer.train_step`
    whenever a rollout-length boundary is crossed (most steps return
    False). Covers (env=cpu, train=cpu) and (env=cpu, train=gpu) —
    the trainer H2Ds obs internally on the GPU path.
    """
    comptime OBS = A.AGENT_OBS_DIM

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var obs_list = env.reset_obs_list()

    var ep_returns = List[Scalar[DT]]()
    var current_ep_count = trainer.ep_count()

    var t_start = perf_counter_ns()
    var step: Int = 0
    while step < total_timesteps:
        for d in range(OBS):
            obs[d] = Scalar[DT](obs_list[d])
        var action_idx = trainer.select_action(obs, base_step + step)
        var step_res = env.step_obs(action_idx)
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(OBS):
            next_obs[d] = Scalar[DT](nxt[d])
        trainer.record_transition(
            obs, action_idx, Scalar[DT](reward), next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
        # TRUE terminal (V(s')=0 in GAE) only on natural termination;
        # time-limit truncation keeps the value bootstrap.
        if env.was_terminated():
            trainer.mark_terminal()
        if done:
            trainer.end_episode()
            obs_list = env.reset_obs_list()
            var new_ep_count = trainer.ep_count()
            if new_ep_count > current_ep_count:
                ep_returns.append(trainer.mean_return())
                current_ep_count = new_ep_count
        else:
            obs_list = nxt^
        step += 1
        _ = trainer.train_step(base_step + step)

        var abs_step = base_step + step

        if verbose and print_every > 0 and abs_step % print_every == 0:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ", abs_step, "] mean_ret(10)=", trainer.mean_return(),
                " ep=", trainer.ep_count(),
                " elapsed=", elapsed, "s",
            )

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

        comptime if L.ENABLED:
            if (
                diag_every > 0
                and abs_step % diag_every == 0
                and Bool(logger)
            ):
                trainer.flush_metrics_through_logger[L](logger, abs_step)

        if (
            checkpoint_every > 0
            and abs_step % checkpoint_every == 0
            and checkpoint_path.byte_length() > 0
        ):
            trainer.save_state(checkpoint_path)

    if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
        trainer.save_state(checkpoint_path)

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# run_onpolicy_discrete_eval — single-env greedy eval.
# ──────────────────────────────────────────────────────────────────────


def run_onpolicy_discrete_eval[
    A: OnPolicyDiscreteAgent,
    E: BoxDiscreteActionEnv,
](
    mut trainer: A,
    mut env: E,
    num_episodes: Int,
    *,
    max_steps_per_episode: Int = 1_000,
    verbose: Bool = False,
) raises -> Scalar[DT]:
    """Non-mutating greedy eval driver for discrete on-policy agents.

    Uses `select_greedy_action` (argmax logits). Does not touch the
    rollout buffer, optimizers, or episode tracker.
    """
    comptime OBS = A.AGENT_OBS_DIM
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var total_return = Scalar[DT](0.0)
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
                "  [eval ep ", ep + 1, "/", num_episodes,
                "] return=", ep_return, " steps=", ep_steps,
            )

    var mean = total_return / Scalar[DT](num_episodes)
    if verbose:
        print("eval: mean_return=", mean, " (", num_episodes, " episodes)")
    return mean
