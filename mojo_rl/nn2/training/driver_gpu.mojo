"""Off-policy GPU eval driver — single remaining function post-migration.

Both `run_offpolicy_train_gpu` (Phase B.5 single-env CPU env + GPU train)
and `run_offpolicy_train_gpu_n_envs` (Phase B.5b N_ENVS GPU env + GPU
train) have been deleted. Their consumers migrated to:

  - `run_offpolicy_train_unified` (Tier-1 Phase 3) — cross-target
    single-env (cpu env, gpu train, N=1).
  - `run_offpolicy_train_batched` (Tier-3) — same-target at any
    N_ENVS, via BatchedEnv trait (BatchedCpuEnv / BatchedGpuEnv).

The eval function stays here as a separate concern from training; it
operates on a CPU env (since N=1 eval doesn't benefit from GPU env
batching) and uses the trainer's `select_greedy_action_gpu`.
"""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from ..constants import DT
from mojo_rl.core.env_traits import BoxContinuousActionEnv
from .driver_cpu import OffPolicyTrainableGpu


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
