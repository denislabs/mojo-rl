"""On-policy training driver — namespace twin of `driver_offpolicy.mojo`.

Mirrors the symbol shape of the off-policy driver so on-policy
trainers (PPO today, possibly A2C later) plug into a consistent surface:

  - `OnPolicyAgent` — trait every on-policy nn2 trainer conforms to.
  - `run_onpolicy_train` — single-env on-policy training driver.

The Tier-3 BatchedEnv + GPU variants land in PPO V2R (P.2/P.3); for now
PPO is CPU + N=1 only.
"""

from std.time import perf_counter_ns

from ..constants import DT
from mojo_rl.core.env_traits import BoxContinuousActionEnv


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
