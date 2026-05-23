"""Off-policy CPU evaluation driver — Phase B.2.

Non-mutating eval loop: greedy action selection, no replay record, no
train_step, no end_episode tracker side-effects. Used after training to
measure converged-policy performance.

Trainer contract — uses the `OffPolicyTrainable` trait declared in
`driver_cpu.mojo`. The only trait method invoked here is
`select_greedy_action`. `record` / `train_step` / `end_episode` /
`mean_return` / `ep_count` are intentionally skipped — eval must not
touch the trainer's replay buffer, optimizers, or episode tracker.

The trainer struct is still `mut` (the greedy action call uses internal
scratch buffers `_ob1`, `_ao1`) but the call sequence guarantees no
training-state mutation: no grads accumulated, no Adam.step, no Polyak
update, no replay insertion.

Env contract: same `BoxContinuousActionEnv` as the train driver.
"""

from std.memory import alloc
from std.time import perf_counter_ns

from ..constants import DT
from mojo_rl.core.env_traits import BoxContinuousActionEnv
from .driver_cpu import OffPolicyTrainable


def run_offpolicy_eval_cpu[
    A: OffPolicyTrainable,
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
    """Run `num_episodes` deterministic eval episodes against `env`.

    Args:
        trainer: Any `OffPolicyTrainable` (SAC / DDPG / TD3). Mutates
            only its single-step scratch buffers — no training side
            effects (no replay record, no optimizer step, no Polyak,
            no tracker update).
        env: Any `BoxContinuousActionEnv`. Reset between episodes.
        num_episodes: How many episodes to roll out.
        obs_dim: Observation dimensionality.
        act_dim: Action dimensionality.
        max_steps_per_episode: Hard cap on per-episode length (the env
            may end the episode sooner via `done`). Default 1000; for
            Pendulum (200-step truncation) this is essentially unused.
        verbose: Print one line per episode with the return + step count.

    Returns:
        Mean episode return across `num_episodes`. Caller decides what
        "good enough" means (e.g., > -200 for Pendulum swing-up).

    Notes:
        - Greedy action only: SAC uses `tanh(mean) * action_scale`;
          DDPG/TD3 use the deterministic actor output (no Gaussian
          exploration noise). See `OffPolicyTrainable.select_greedy_action`.
        - Per-call dtype cast at the env boundary, mirroring the train
          driver — typically a no-op (DT == f32, Pendulum dtype == f32).
        - No RNG consumed by the trainer's action path (no warmup, no
          rsample sampling, no Gaussian noise). Env RNG (e.g. Pendulum's
          reset randomization) still advances std.random's global state.
    """
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
            trainer.select_greedy_action(obs, action)
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
