"""On-policy training driver — namespace twin of `driver_offpolicy.mojo`.

Mirrors the symbol shape of the off-policy driver so on-policy
trainers (PPO today, possibly A2C later) plug into a consistent surface:

  - `OnPolicyAgent` — N=1 host-list trait for single-env trainers.
  - `OnPolicyAgentBatched` — N_ENVS-wide pointer trait for batched
    trainers (PPOTrainerV2R conforms).
  - `run_onpolicy_train` — single-env on-policy training driver.
  - `run_onpolicy_train_batched` — BatchedEnv driver covering same-
    target (env_target == train_target) combinations × any N_ENVS,
    via host-staging scratches (D2H for GPU env).

Cross-target (cpu env, gpu train) and the degenerate (gpu env, cpu
train) combination are not exposed by `run_onpolicy_train_batched` —
single-env users get `run_onpolicy_train`; everyone else uses one
target consistently across env + trainer.
"""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer

from ..constants import DT
from mojo_rl.core.env_traits import BoxContinuousActionEnv
from .batched_env import BatchedEnv
from .driver_scratch import DriverScratch


trait OnPolicyAgent(Movable, ImplicitlyDestructible):
    """Surface every nn2 on-policy trainer (PPO / future A2C) exposes
    for the on-policy training driver.

    Per-step contract mirrors the off-policy driver so the loop stays
    almost identical (collect transition → record → call `train_step`
    once per env step). The only behavioural difference is that
    on-policy `train_step` returns False on the vast majority of steps
    and True only when a rollout-length boundary is hit and the
    K-epoch minibatch updates fire.

    Internal state ownership: the trainer caches `(unbounded action,
    log_prob, value)` between `select_action` and `record_transition`.
    Callers must invoke them in pairs — same as the off-policy driver's
    select-then-record pattern. The driver does NOT pass log_prob /
    value back to the trainer; the trainer caches them itself.

    `select_action` writes the *env-ready* action (already action-scaled
    and clamped). The trainer's internal cache holds the *unbounded*
    sample used for the log_prob during the upcoming update.
    """

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        ...

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        ...

    def record_transition(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        ...

    def end_episode(mut self):
        ...

    def train_step(mut self, step_idx: Int) raises -> Bool:
        ...

    def mean_return(self) -> Scalar[DT]:
        ...

    def ep_count(self) -> Int:
        ...


def run_onpolicy_train[
    A: OnPolicyAgent,
    E: BoxContinuousActionEnv,
](
    mut trainer: A,
    mut env: E,
    total_timesteps: Int,
    *,
    obs_dim: Int,
    act_dim: Int,
    print_every: Int = 1_000,
    verbose: Bool = True,
) raises -> List[Scalar[DT]]:
    """Step-based on-policy single-env training driver.

    One env step + one `train_step` call per iteration. PPO's rollout
    accumulation and K-epoch update fire inside `trainer.train_step`
    whenever a rollout-length boundary is crossed (most steps return
    False).

    Args:
        trainer: Any nn2 on-policy trainer (PPO today).
        env: Any `BoxContinuousActionEnv`.
        total_timesteps: Number of env steps to run.
        obs_dim, act_dim: Observation / action dimensionalities.
        print_every: Verbose status-line cadence (env-steps). 0 disables.
        verbose: Print a per-cadence status line.

    Returns:
        List of `trainer.mean_return()` snapshots taken at each completed
        episode boundary (same shape as the off-policy driver).
    """
    var obs = List[Scalar[DT]](length=obs_dim, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=obs_dim, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=act_dim, fill=Scalar[DT](0.0))

    var obs_list = env.reset_obs_list()
    var action_list = List[Scalar[E.dtype]](capacity=act_dim)
    for _ in range(act_dim):
        action_list.append(Scalar[E.dtype](0.0))

    var ep_returns = List[Scalar[DT]]()
    var current_ep_count = trainer.ep_count()

    var t_start = perf_counter_ns()
    var step: Int = 0
    while step < total_timesteps:
        for d in range(obs_dim):
            obs[d] = Scalar[DT](obs_list[d])
        trainer.select_action(obs, action, step)
        for j in range(act_dim):
            action_list[j] = Scalar[E.dtype](action[j])
        var step_res = env.step_continuous_vec[E.dtype](action_list)
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(obs_dim):
            next_obs[d] = Scalar[DT](nxt[d])
        trainer.record_transition(
            obs, action, Scalar[DT](reward), next_obs,
            Scalar[DT](1.0) if done else Scalar[DT](0.0),
        )
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
        _ = trainer.train_step(step)

        if verbose and print_every > 0 and step % print_every == 0:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ", step, "] mean_ret(10)=", trainer.mean_return(),
                " ep=", trainer.ep_count(),
                " elapsed=", elapsed, "s",
            )

    return ep_returns^


# ──────────────────────────────────────────────────────────────────────
# OnPolicyAgentBatched — trait for the Tier-3 BatchedEnv driver.
# ──────────────────────────────────────────────────────────────────────


trait OnPolicyAgentBatched(Movable, ImplicitlyDestructible):
    """N_ENVS-wide pointer-based trait for on-policy trainers consumed
    by `run_onpolicy_train_batched`.

    All pointer args are HOST-side. For GPU envs the driver D2Hs
    env-side obs/reward/done into host scratches before calling. The
    trainer is responsible for any internal H2D of obs into device-
    side scratches (PPOTrainerV2R does this inside PPOActStep).

    Conforming trainers also expose comptime aliases so the driver
    can comptime-assert dimensional consistency with the env adapter
    and gate H2D/D2H around the env step.
    """

    comptime AGENT_TRAIN_TARGET: StaticString
    comptime AGENT_OBS_DIM: Int
    comptime AGENT_ACT_DIM: Int
    comptime AGENT_N_ENVS: Int

    def select_action_batched(
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        """Reads AGENT_N_ENVS * AGENT_OBS_DIM from `obs_ptr`, writes
        AGENT_N_ENVS * AGENT_ACT_DIM into `action_ptr`. Caches per-env
        (sample, log_prob, value) internally for the upcoming
        `record_batch_cpu`. Both pointers must be host-side."""
        ...

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Single-env greedy eval — list-based. Always BATCH=1 even
        when the trainer is configured for N_ENVS > 1."""
        ...

    def record_batch_cpu(
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Push AGENT_N_ENVS transitions into the rollout buffer. All
        pointers host-side. Maintains per-env running returns and
        pushes completed episodes into the EpisodeTracker on done."""
        ...

    def train_step(mut self, step_idx: Int) raises -> Bool:
        ...

    def mean_return(self) -> Scalar[DT]:
        ...

    def ep_count(self) -> Int:
        ...


def run_onpolicy_train_batched[
    A: OnPolicyAgentBatched,
    E: BatchedEnv,
](
    ctx: Optional[DeviceContext],
    mut trainer: A,
    mut env: E,
    total_env_steps: Int,
    *,
    rng_seed: UInt64 = UInt64(42),
    print_every: Int = 5_000,
    verbose: Bool = True,
) raises -> List[Scalar[DT]]:
    """Tier-3 on-policy driver covering same-target combinations
    (env_target == train_target) × any N_ENVS through the `BatchedEnv`
    trait:

      env_target | train_target | N_ENVS | covered
      -----------|--------------|--------|--------
      cpu        | cpu          | >=1    | yes  (via BatchedCpuEnv)
      gpu        | gpu          | >=1    | yes  (via BatchedGpuEnv)

    Cross-target combinations are NOT covered here:
      - (cpu env, gpu train) → use `run_onpolicy_train` (single-env)
      - (gpu env, cpu train) → degenerate (D2H every obs)

    Unlike the off-policy driver, the on-policy trainer always wants
    host-side pointers (PPO V2R's rollout buffer lives host-only on
    both targets; the trainer itself does H2D of obs internally
    inside PPOActStep). The driver therefore stages env outputs
    through host scratches — a no-op pointer alias for CPU envs and
    a D2H copy for GPU envs.

    Loop per iteration (N_ENVS env steps):
      1. Snapshot env.obs_ptr()                  → prev_obs_h
      2. trainer.select_action_batched(prev_obs_h, action_h, step_idx)
      3. (gpu env) H2D action_h → env.action_ptr()
      4. env.step_batch[N_ENVS]
      5. Snapshot env.obs/reward/done             → next_obs_h/reward_h/done_h
      6. trainer.record_batch_cpu(prev_obs_h, reward_h, next_obs_h, done_h)
      7. env.selective_reset_batch[N_ENVS]
      8. trainer.train_step (fires the K-epoch update at rollout
         boundary)
    """
    comptime env_target: StaticString = E.ENV_TARGET
    comptime train_target: StaticString = A.AGENT_TRAIN_TARGET
    comptime OBS = A.AGENT_OBS_DIM
    comptime ACT = A.AGENT_ACT_DIM
    comptime N_ENVS = A.AGENT_N_ENVS

    comptime assert (
        env_target == "cpu" or env_target == "gpu"
    ), "env_target must be 'cpu' or 'gpu'"
    comptime assert (
        train_target == "cpu" or train_target == "gpu"
    ), "train_target must be 'cpu' or 'gpu'"
    comptime assert env_target == train_target, (
        "run_onpolicy_train_batched: env_target must equal train_target."
        " Cross-target (cpu env, gpu train) → use run_onpolicy_train;"
        " (gpu env, cpu train) → rejected as degenerate."
    )
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime assert (
        E.OBS_DIM == OBS and E.ACT_DIM == ACT
    ), "BatchedEnv dimensions must match trainer dimensions"
    comptime if env_target == "gpu":
        if not ctx:
            raise Error(
                "run_onpolicy_train_batched: ctx required when"
                " env_target is 'gpu'"
            )

    # Host-side staging scratches. The trainer always reads/writes
    # host pointers; on GPU env we D2H env outputs into these.
    var prev_obs_h = DriverScratch["prev_obs", N_ENVS, OBS].make["cpu"](
        ctx=None
    )
    var action_h   = DriverScratch["action",   N_ENVS, ACT].make["cpu"](
        ctx=None
    )
    var next_obs_h = DriverScratch["next_obs", N_ENVS, OBS].make["cpu"](
        ctx=None
    )
    var reward_h   = DriverScratch["reward",   N_ENVS, 1].make["cpu"](
        ctx=None
    )
    var done_h     = DriverScratch["done",     N_ENVS, 1].make["cpu"](
        ctx=None
    )

    env.reset_batch[N_ENVS](ctx=ctx, rng_seed=rng_seed)

    var ep_returns = List[Scalar[DT]]()
    var t_start = perf_counter_ns()
    var step_idx: Int = 0
    var iter_idx: Int = 0
    var next_print: Int = print_every
    var last_ep_count = trainer.ep_count()

    while step_idx < total_env_steps:
        # ── 1. Snapshot env.obs_ptr() → prev_obs_h.
        var po_p = prev_obs_h.host_ptr()
        comptime if env_target == "cpu":
            var ob_p = env.obs_ptr()
            for k in range(N_ENVS * OBS):
                po_p[k] = ob_p[k]
        else:
            var c = ctx.value()
            var env_obs_view = DeviceBuffer[DT](
                c, env.obs_ptr(), N_ENVS * OBS, owning=False,
            )
            var po_host = c.enqueue_create_host_buffer[DT](N_ENVS * OBS)
            c.enqueue_copy(po_host, env_obs_view)
            c.synchronize()
            var ph = po_host.unsafe_ptr()
            for k in range(N_ENVS * OBS):
                po_p[k] = ph[k]

        # ── 2. Trainer writes action into host scratch.
        trainer.select_action_batched(
            po_p,
            action_h.host_ptr(),
            step_idx,
        )

        # ── 3. (gpu env) H2D action into env.action_ptr().
        comptime if env_target == "gpu":
            var c = ctx.value()
            var env_act_view = DeviceBuffer[DT](
                c, env.action_ptr(), N_ENVS * ACT, owning=False,
            )
            c.enqueue_copy(env_act_view, action_h.host_ptr())
        else:
            # CPU env: copy action_h → env.action_ptr() (same target side).
            var ap = action_h.host_ptr()
            var ea = env.action_ptr()
            for k in range(N_ENVS * ACT):
                ea[k] = ap[k]

        # ── 4. Env step.
        env.step_batch[N_ENVS](
            ctx=ctx,
            rng_seed=rng_seed + UInt64(iter_idx + 1),
        )

        # ── 5. Snapshot env outputs → host scratches.
        var no_p  = next_obs_h.host_ptr()
        var rew_p = reward_h.host_ptr()
        var dn_p  = done_h.host_ptr()
        comptime if env_target == "cpu":
            var ob_p = env.obs_ptr()
            var er_p = env.reward_ptr()
            var ed_p = env.done_ptr()
            for k in range(N_ENVS * OBS):
                no_p[k] = ob_p[k]
            for e in range(N_ENVS):
                rew_p[e] = er_p[e]
                dn_p[e]  = ed_p[e]
        else:
            var c = ctx.value()
            var env_obs_view = DeviceBuffer[DT](
                c, env.obs_ptr(), N_ENVS * OBS, owning=False,
            )
            var env_rew_view = DeviceBuffer[DT](
                c, env.reward_ptr(), N_ENVS, owning=False,
            )
            var env_done_view = DeviceBuffer[DT](
                c, env.done_ptr(), N_ENVS, owning=False,
            )
            var no_host  = c.enqueue_create_host_buffer[DT](N_ENVS * OBS)
            var rew_host = c.enqueue_create_host_buffer[DT](N_ENVS)
            var dn_host  = c.enqueue_create_host_buffer[DT](N_ENVS)
            c.enqueue_copy(no_host,  env_obs_view)
            c.enqueue_copy(rew_host, env_rew_view)
            c.enqueue_copy(dn_host,  env_done_view)
            c.synchronize()
            var nh = no_host.unsafe_ptr()
            var rh = rew_host.unsafe_ptr()
            var dh = dn_host.unsafe_ptr()
            for k in range(N_ENVS * OBS):
                no_p[k] = nh[k]
            for e in range(N_ENVS):
                rew_p[e] = rh[e]
                dn_p[e]  = dh[e]

        # ── 6. Trainer push.
        trainer.record_batch_cpu(po_p, rew_p, no_p, dn_p)

        # ── 7. Selective env reset (env handles per-env done internally).
        env.selective_reset_batch[N_ENVS](
            ctx=ctx,
            rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
        )

        step_idx += N_ENVS
        iter_idx += 1

        # ── 8. Trainer update (returns True at K-epoch boundary).
        _ = trainer.train_step(step_idx)

        # Snapshot mean_return whenever an episode completes.
        var new_ep_count = trainer.ep_count()
        if new_ep_count > last_ep_count:
            ep_returns.append(trainer.mean_return())
            last_ep_count = new_ep_count

        if verbose and print_every > 0 and step_idx >= next_print:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ", step_idx, "] mean_ret(10)=",
                trainer.mean_return(),
                " ep=", trainer.ep_count(),
                " elapsed=", elapsed, "s",
            )
            next_print += print_every

    return ep_returns^
