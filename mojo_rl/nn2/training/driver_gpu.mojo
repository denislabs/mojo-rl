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

from std.memory import alloc
from std.time import perf_counter_ns

from ..constants import DT
from mojo_rl.core.env_traits import BoxContinuousActionEnv
from .driver_cpu import OffPolicyTrainableGpu


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
    var obs = alloc[Scalar[DT]](obs_dim)
    var next_obs = alloc[Scalar[DT]](obs_dim)
    var action = alloc[Scalar[DT]](act_dim)

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
        _ = trainer.train_step_gpu(step)

        if verbose and print_every > 0 and step % print_every == 0:
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "[step ", step, "] mean_ret(10)=", trainer.mean_return(),
                " ep=", trainer.ep_count(),
                " elapsed=", elapsed, "s",
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
    var obs = alloc[Scalar[DT]](obs_dim)
    var action = alloc[Scalar[DT]](act_dim)

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
                "  [eval ep ", ep + 1, "/", num_episodes,
                "] return=", ep_return, " steps=", ep_steps,
            )

    var mean = total_return / Scalar[DT](num_episodes)
    if verbose:
        var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
        print(
            "eval: mean_return=", mean, " (",
            num_episodes, " episodes, ", elapsed, " s)",
        )
    return mean
