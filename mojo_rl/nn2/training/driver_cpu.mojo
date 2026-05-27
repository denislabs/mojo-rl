"""Off-policy CPU training driver — Phase B.1.

One generic loop that runs SACTrainer / DDPGTrainer / TD3Trainer end-to-end
against any `BoxContinuousActionEnv`. Mirrors the production
`mojo_rl/deep_agents/core/training/offpolicy_train.mojo` surface (warmup
handled inside the trainer's `select_action`, log/print cadences, optional
Logger pointer) but keeps the existing nn2 step-based shape so the
bit-identity baseline (CPU SAC Pendulum 30k seed=42 → mean_ret(10) = −167.572)
is preserved by construction.

Trainer contract — duck-typed via the local `OffPolicyTrainable` trait:

  trait OffPolicyTrainable:
      def select_action(obs_ptr, act_ptr, step_idx) raises
      def record(obs_ptr, act_ptr, reward, next_obs_ptr, done)
      def end_episode()
      def train_step(step_idx) raises -> Bool
      def mean_return() -> Scalar[DT]
      def ep_count() -> Int

SAC's `select_action[target="cpu"]` / `train_step[target="cpu"]` satisfy
the non-parametric trait signatures via comptime-param default resolution
(verified at compile time when the driver instantiates).

`flush_metrics` is *not* part of the trait — its return type is per-trainer
(SACMetrics / DDPGMetrics / TD3Metrics). The driver calls it through the
struct's own surface; users who want logging plumb a non-NoOp Logger.
For B.1 the driver invokes `flush_metrics` only on SAC (via a comptime
branch). For DDPG/TD3 the per-cadence metric flush is the caller's
responsibility.
"""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer

from ..constants import DT
from ..data.n_step_replay import GPUNStepBuffer
from mojo_rl.core.env_traits import BoxContinuousActionEnv
from mojo_rl.core.logger import Logger, NoOpLogger


# `OffPolicyTrainableGpuBatched` was deleted after the M5 + MBPO migrations.
# The last consumer (`run_offpolicy_train_gpu_n_envs`) was removed in M5;
# the trait surface is fully subsumed by `OffPolicyAgentUnifiedGpu` in
# `driver_unified.mojo`. SAC's conformance was dropped at the same time.


trait OffPolicyTrainableGpu(Movable, ImplicitlyDestructible):
    """Surface every nn2 off-policy trainer that supports a GPU train
    path must expose for the GPU train + eval drivers (Phase B.5).

    Mirrors `OffPolicyTrainable` (the CPU surface) but with explicitly
    `_gpu` non-parametric methods that forward to the trainer's
    parametric `["gpu"]` path. Same conformance rationale as the CPU
    trait: Mojo nightly's trait conformance is strict about comptime-
    param signatures, so we ship a non-parametric wrapper per method.

    DDPG/TD3 are CPU-only in nn2 today and do NOT conform — only SAC
    does. When DDPG/TD3 GPU paths land, they'll add their own
    conformance.

    `select_greedy_action_gpu` exists for symmetry with the CPU eval
    driver; the GPU eval driver uses it. SAC's GPU greedy is the same
    tanh(mean)*action_scale math as CPU but executed against device
    buffers.
    """

    def select_action_gpu(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        ...

    def select_greedy_action_gpu(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        ...

    def record(
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

    def train_step_gpu(mut self, step_idx: Int) raises -> Bool:
        ...

    def mean_return(self) -> Scalar[DT]:
        ...

    def ep_count(self) -> Int:
        ...


trait OffPolicyTrainable(Movable, ImplicitlyDestructible):
    """Surface every nn2 off-policy trainer (SAC / DDPG / TD3) exposes
    for the CPU training + eval drivers.

    Non-parametric. SAC's `select_action[target: StaticString = "cpu"]`
    and `train_step[target: StaticString = "cpu"]` satisfy this trait
    via thin non-parametric wrappers on SACTrainer that forward to the
    parametric `["cpu"]` path. DDPG/TD3 already have non-parametric
    signatures so they conform directly.

    `select_greedy_action` (Phase B.2): deterministic, exploration-free
    action selection used by `run_offpolicy_eval_cpu`. SAC uses
    tanh(mean) * action_scale (skip rsample's stochastic head); DDPG/TD3
    forward the actor and clamp without adding Gaussian noise.
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

    def record(
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


trait OnPolicyTrainable(Movable, ImplicitlyDestructible):
    """Surface every nn2 on-policy trainer (PPO / future A2C) exposes
    for the on-policy CPU training driver (Phase I.2.d).

    Per-step contract mirrors `OffPolicyTrainable` so the driver loop
    stays almost identical (collect transition → record → call
    `train_step` once per env step). The only behavioural difference
    is that on-policy `train_step` returns False on the vast majority
    of steps and True only when a rollout-length boundary is hit and
    the K-epoch minibatch updates fire.

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


def run_onpolicy_train_cpu[
    A: OnPolicyTrainable,
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
    """Step-based on-policy CPU training driver (Phase I.2.d).

    Mirrors `run_offpolicy_train_cpu` exactly: one env step + one
    `train_step` call per iteration. PPO's rollout accumulation and
    K-epoch update fire inside `trainer.train_step` whenever a
    rollout-length boundary is crossed (most steps return False).

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


# `run_offpolicy_train_cpu` was deleted after the MBPO migration to
# `run_offpolicy_train_batched` (Tier-3). All callers now go through the
# unified driver in `driver_unified.mojo`. The `OffPolicyTrainable` trait
# above is kept because the CPU eval driver (`run_offpolicy_eval_cpu`)
# still uses it, as does `run_offpolicy_eval_gpu` via `OffPolicyTrainableGpu`.
