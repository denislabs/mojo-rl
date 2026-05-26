"""Off-policy GPU training + eval drivers — Phase B.5.

Single-env GPU drivers. Mirror the CPU drivers (`driver_cpu.mojo`) but
route through the `OffPolicyTrainableGpu` trait's `_gpu` methods
(`select_action_gpu`, `train_step_gpu`, `select_greedy_action_gpu`).

Scope: single-env. The env stays on the CPU side (e.g. `PendulumEnv`);
only the trainer's compute path runs on GPU. Replay buffer is also CPU
in nn2 today (`CPUReplay`). The trainer's GPU path uploads replay
minibatches device-side per `train_step["gpu"]`.

N_ENVS vectorization is a future extension blocked on Phase C.1 (GPU
replay buffer). When that lands, `select_action_gpu` gets a batched
variant + the driver loops over `N_ENVS` env instances per step.

Trainer contract — `OffPolicyTrainableGpu`. Only SAC conforms today;
DDPG/TD3 are CPU-only.

Env contract — same `BoxContinuousActionEnv` as the CPU driver.

DeviceContext: the driver does NOT take `ctx` as an arg. The trainer
already holds it internally (via `target_y_block.ts.ctx`), having been
constructed through the `make["gpu"](ctx, ...)` factory. Threading
`ctx` through every driver call would duplicate the trainer's
ownership of it. Apple Metal: do NOT construct a new `DeviceContext()`
inside the driver — the trainer's queue pool would exhaust within ~1k
steps (see `feedback_apple_metal_devicecontext_per_call`).
"""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from ..constants import DT
from ..data.n_step_replay import GPUNStepBuffer
from mojo_rl.core.env_traits import BoxContinuousActionEnv, GPUContinuousEnv
from .driver_cpu import OffPolicyTrainableGpu, OffPolicyTrainableGpuBatched


def run_offpolicy_train_gpu[
    A: OffPolicyTrainableGpu,
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
    """Step-based off-policy GPU training driver. Single-env.

    Args / returns: same shape as `run_offpolicy_train_cpu`. The only
    visible differences are:
      * `A` must conform to `OffPolicyTrainableGpu` (currently SAC only).
      * Trainer must have been constructed via `make["gpu"](ctx, ...)`.
      * Trainer.{select_action_gpu, train_step_gpu, select_greedy_action_gpu}
        are called instead of the CPU variants.

    Bit-identity vs hand-rolled GPU loop: the existing manual GPU
    example (`examples/pendulum/pendulum_sac_nn2_trainer_gpu.mojo`)
    consumes RNG in the same order — env.reset → select_action_gpu →
    env.step → record → end_episode → train_step_gpu — so converting
    that example to use this driver preserves the convergence number
    (whatever it is on the user's GPU; CPU-baseline -167.572 is for
    the CPU SAC path).
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
        trainer.select_action_gpu(obs, action, step)
        for j in range(act_dim):
            action_list[j] = Scalar[E.dtype](action[j])
        var step_res = env.step_continuous_vec[E.dtype](action_list)
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        for d in range(obs_dim):
            next_obs[d] = Scalar[DT](nxt[d])
        trainer.record(
            obs,
            action,
            Scalar[DT](reward),
            next_obs,
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
        _ = trainer.train_step_gpu(step)

        if verbose and print_every > 0 and step % print_every == 0:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ",
                step,
                "] mean_ret(10)=",
                trainer.mean_return(),
                " ep=",
                trainer.ep_count(),
                " elapsed=",
                elapsed,
                "s",
            )

    return ep_returns^


def run_offpolicy_eval_gpu[
    A: OffPolicyTrainableGpu,
    E: BoxContinuousActionEnv,
](
    mut trainer: A,
    mut env: E,
    num_episodes: Int,
    *,
    obs_dim: Int,
    act_dim: Int,
    max_steps_per_episode: Int = 1_000,
    verbose: Bool = False,
) raises -> Scalar[DT]:
    """GPU mirror of `run_offpolicy_eval_cpu` — uses
    `select_greedy_action_gpu` on the trainer. See the CPU eval driver
    docstring for the non-mutation guarantee + RNG semantics. The only
    GPU-specific behaviour is one extra D2H sync per env step (the
    actor's device output is downloaded so the tanh+clamp can run on
    host for the single-step path)."""
    var obs = List[Scalar[DT]](length=obs_dim, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=act_dim, fill=Scalar[DT](0.0))

    var action_list = List[Scalar[E.dtype]](capacity=act_dim)
    for _ in range(act_dim):
        action_list.append(Scalar[E.dtype](0.0))

    var total_return = Scalar[DT](0.0)
    var t_start = perf_counter_ns()

    for ep in range(num_episodes):
        var obs_list = env.reset_obs_list()
        var ep_return = Scalar[DT](0.0)
        var ep_steps: Int = 0

        for _ in range(max_steps_per_episode):
            for d in range(obs_dim):
                obs[d] = Scalar[DT](obs_list[d])
            trainer.select_greedy_action_gpu(obs, action)
            for j in range(act_dim):
                action_list[j] = Scalar[E.dtype](action[j])
            var step_res = env.step_continuous_vec[E.dtype](action_list)
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


# ──────────────────────────────────────────────────────────────────────
# Phase B.5b — N_ENVS-vectorized GPU off-policy training driver.
# ──────────────────────────────────────────────────────────────────────


def run_offpolicy_train_gpu_n_envs[
    A: OffPolicyTrainableGpuBatched,
    E: GPUContinuousEnv,
    N_ENVS: Int,
    NS: Int = 1,
](
    ctx: DeviceContext,
    mut trainer: A,
    mut env: E,
    total_env_steps: Int,
    *,
    rng_seed: UInt64 = UInt64(42),
    updates_per_step: Int = 1,
    print_every: Int = 5_000,
    verbose: Bool = True,
    nstep_gamma: Scalar[DT] = Scalar[DT](0.99),
) raises -> List[Scalar[DT]]:
    """Multi-env GPU off-policy training driver (Phase B.5b).

    Spins up `N_ENVS` env instances on GPU, runs a batched policy
    forward per step, batched env step, batched record into the
    device-resident replay (`GPUReplay.add_batch[N_ENVS]`), and one or
    more SAC train_steps per iteration. The driver owns all the
    N_ENVS-sized device buffers (states / prev_obs / obs / actions /
    rewards / dones / terminated / ao_scratch / alp_scratch).

    Loop semantics:
      `step_idx` counts env-step transitions (not iterations): with
      `total_env_steps = 30_000` and `N_ENVS = 8`, the loop runs 3750
      iterations and yields ~3750 train_step calls at default
      `updates_per_step = 1`. To match single-env UTD = 1 (one update
      per transition), pass `updates_per_step = N_ENVS`.

    DeviceContext: the driver takes `ctx` explicitly because it
    allocates N_ENVS-sized device buffers itself. The user MUST pass
    the same `ctx` used to construct the trainer (otherwise device
    pointers cross contexts and Apple Metal's queue pool exhausts).

    Args:
        ctx: GPU device context (same one used to build the trainer).
        trainer: Any nn2 off-policy trainer conforming to
            `OffPolicyTrainableGpuBatched`. SAC only today.
        env: GPU continuous env (only used as a type carrier — the
            static `step_kernel_gpu` / `reset_kernel_gpu` /
            `extract_obs_kernel_gpu` / `selective_reset_kernel_gpu`
            methods are called on the type, not the instance).
        total_env_steps: Number of env-step transitions to collect.
            Loop runs `ceil(total_env_steps / N_ENVS)` iterations.
        rng_seed: Seed for env reset + step RNG. Both are advanced per
            iteration so successive steps draw fresh randomness.
        updates_per_step: SAC train_steps per loop iteration. Default
            1 (UTD = 1 / N_ENVS). Pass `N_ENVS` for full UTD = 1
            parity with single-env.
        print_every: Status cadence (env-step counter). 0 disables.
        verbose: Print status lines.
        nstep_gamma: γ for the n-step buffer. Only matters when NS > 1.

    Returns:
        List of `trainer.mean_return()` snapshots, one per completed
        episode across all envs.

    Tracker semantics: the driver maintains its own per-env reward
    accumulators (host-side, fed by a single D2H of `rewards` per
    iteration) and pushes complete-episode returns to the trainer via
    `add_complete_return`. The trainer's per-step `current_return`
    field stays at 0 in N_ENVS mode — `mean_return()` reflects only
    the completed-episode rolling window, which is the correct
    semantics for vectorized training.

    N-step (`NS` > 1): allocates a `GPUNStepBuffer[NS, A.OBS_DIM,
    A.ACT_DIM, N_ENVS]` and routes the batched record through
    `trainer.record_batch_gpu_nstep[N_ENVS, NS](...)` so transitions
    accumulate into n-step compressed returns before landing in the
    replay. The trainer's `target_y_block` must already be configured
    for `γ^NS` bootstrap (i.e. the trainer's `N_STEP` comptime + the
    SACConfig `use_n_step=True` flag — see C.2b). `NS` must equal the
    trainer's `N_STEP`; mismatch is caught by the trainer-side
    comptime assert. `NS=1` (default) bypasses the n-step buffer
    entirely — `comptime if NS > 1` gates the allocation and dispatch
    so the default path stays bit-identical to the pre-NS driver.

    `nstep_gamma` controls the per-step γ accumulated by the
    GPUNStepBuffer. It only matters when NS > 1; default 0.99 matches
    SAC's default `gamma`.
    """
    comptime assert N_ENVS > 0, "N_ENVS must be > 0"
    comptime assert NS > 0, "NS must be > 0"
    comptime STATE_SIZE = E.STATE_SIZE
    comptime OBS_DIM = E.OBS_DIM
    comptime ACT_DIM = E.ACTION_DIM

    # Device buffers — one big alloc up front, reused every iteration.
    var states = ctx.enqueue_create_buffer[DT](N_ENVS * STATE_SIZE)
    var prev_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS_DIM)
    var obs_buf = ctx.enqueue_create_buffer[DT](N_ENVS * OBS_DIM)
    var actions = ctx.enqueue_create_buffer[DT](N_ENVS * ACT_DIM)
    var rewards = ctx.enqueue_create_buffer[DT](N_ENVS)
    var dones = ctx.enqueue_create_buffer[DT](N_ENVS)
    var terminated = ctx.enqueue_create_buffer[DT](N_ENVS)
    var ao_scratch = ctx.enqueue_create_buffer[DT](N_ENVS * 2 * ACT_DIM)
    var alp_scratch = ctx.enqueue_create_buffer[DT](N_ENVS * (ACT_DIM + 1))
    ctx.enqueue_memset(actions, 0)
    ctx.enqueue_memset(rewards, 0)
    ctx.enqueue_memset(dones, 0)
    ctx.enqueue_memset(terminated, 0)

    # Host scratch — per-iteration D2H of rewards + dones for episode
    # tracking. Tiny (N_ENVS * 2 scalars), Apple-friendly.
    var host_rewards = alloc[Scalar[DT]](N_ENVS)
    var host_dones = alloc[Scalar[DT]](N_ENVS)
    var per_env_returns = List[Scalar[DT]](
        length=N_ENVS,
        fill=Scalar[DT](0.0),
    )

    # N-step buffer (NS > 1 only). `comptime if` elides the
    # allocation entirely when NS == 1 — the default path is bit-
    # identical to the pre-NS driver.
    # N-step buffer — allocated unconditionally so the value is in
    # scope at the dispatch site (comptime-if doesn't bleed bindings
    # into the enclosing scope in Mojo nightly). When NS=1 the
    # GPUNStepBuffer holds a few hundred bytes of device memory and
    # is never touched by `.process()`, so the cost is negligible and
    # the numerics are unaffected (no `.process` calls means no
    # kernels run against this buffer).
    var nstep_buf = GPUNStepBuffer[
        NS,
        A.AGENT_OBS_DIM,
        A.AGENT_ACT_DIM,
        N_ENVS,
    ].new(ctx, gamma=nstep_gamma)

    # Initial reset + obs extraction.
    E.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states, rng_seed=rng_seed)
    E.extract_obs_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM](
        ctx,
        states,
        obs_buf,
    )

    var ep_returns = List[Scalar[DT]]()
    var t_start = perf_counter_ns()
    var step_idx: Int = 0
    var iter_idx: Int = 0
    var next_print: Int = print_every

    while step_idx < total_env_steps:
        # Save the pre-step obs as the transition's `prev_obs`. D2D copy
        # — no host involvement.
        ctx.enqueue_copy(prev_obs, obs_buf)

        # Batched policy (warmup uniform if step_idx < learning_starts).
        trainer.select_action_gpu_batched[N_ENVS](
            ctx,
            obs_buf,
            actions,
            ao_scratch,
            alp_scratch,
            step_idx,
        )

        # Env step writes next-obs into `obs_buf` and rewards/dones/
        # terminated as outputs. RNG seed advances per iteration so
        # stochastic envs see fresh randomness.
        E.step_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM, ACT_DIM](
            ctx,
            states,
            actions,
            rewards,
            dones,
            terminated,
            obs_buf,
            rng_seed=rng_seed + UInt64(iter_idx + 1),
        )

        # Push N_ENVS transitions into the device-resident replay.
        # NS=1 → direct uniform/PER routing. NS>1 → wrap through the
        # n-step buffer so transitions accumulate into γ^NS-compressed
        # returns before landing in the replay.
        comptime if NS > 1:
            trainer.record_batch_gpu_nstep[N_ENVS, NS](
                ctx,
                nstep_buf,
                prev_obs,
                actions,
                rewards,
                obs_buf,
                dones,
            )
        else:
            trainer.record_batch_gpu[N_ENVS](
                ctx,
                prev_obs,
                actions,
                rewards,
                obs_buf,
                dones,
            )

        # D2H of rewards + dones — small (N_ENVS * 2 scalars), needed
        # synchronously to update host-side per-env return accumulators
        # and to decide which envs need a selective reset.
        ctx.enqueue_copy(host_rewards, rewards)
        ctx.enqueue_copy(host_dones, dones)
        ctx.synchronize()

        for e in range(N_ENVS):
            per_env_returns[e] = per_env_returns[e] + host_rewards[e]
            if host_dones[e] > Scalar[DT](0.5):
                trainer.add_complete_return(per_env_returns[e])
                per_env_returns[e] = Scalar[DT](0.0)
                ep_returns.append(trainer.mean_return())

        # Selective reset: only resets state for envs with `done`
        # currently 1.0. `extract_obs_kernel_gpu` then refreshes the
        # obs for those envs (state for not-done envs is unchanged so
        # their obs is identical to what step_kernel_gpu wrote).
        E.selective_reset_kernel_gpu[N_ENVS, STATE_SIZE](
            ctx,
            states,
            dones,
            rng_seed=rng_seed + UInt64(iter_idx + 1) * UInt64(7),
        )
        E.extract_obs_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM](
            ctx,
            states,
            obs_buf,
        )

        step_idx += N_ENVS
        iter_idx += 1

        # SAC train_steps. Default UTD = 1 / N_ENVS (1 update per
        # iteration); user can pass `updates_per_step=N_ENVS` to match
        # single-env UTD = 1 per transition.
        for _ in range(updates_per_step):
            _ = trainer.train_step_gpu(step_idx)

        if verbose and print_every > 0 and step_idx >= next_print:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ",
                step_idx,
                "] mean_ret(10)=",
                trainer.mean_return(),
                " ep=",
                trainer.ep_count(),
                " elapsed=",
                elapsed,
                "s",
            )
            next_print += print_every

    return ep_returns^
