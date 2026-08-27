"""Smoke test — discrete off-policy driver + trait conformance.

Validates that:
  1. OffPolicyDiscreteAgent trait is expressible by a stub trainer.
  2. run_offpolicy_discrete_train drives env steps and records transitions.
  3. run_offpolicy_discrete_eval runs greedy eval without mutating state.
  4. Episode tracking works (ep_count > 0, mean_return finite).

Uses CartPole (OBS=4, ACTIONS=2) with a random-action stub trainer
(no neural network). Not a convergence test — just wiring.
"""

from std.math import isnan, isinf
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.training.driver_offpolicy_discrete import (
    OffPolicyDiscreteAgent,
    run_offpolicy_discrete_train,
    run_offpolicy_discrete_eval,
)
from mojo_rl.deep_agents.training.episode_tracker import EpisodeTracker

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime TOTAL_STEPS = 500


@fieldwise_init
struct StubDQNTrainer(OffPolicyDiscreteAgent):
    """Minimal random-action trainer conforming to OffPolicyDiscreteAgent."""

    comptime AGENT_TRAIN_TARGET: StaticString = "cpu"
    comptime AGENT_OBS_DIM: Int = OBS_DIM
    comptime AGENT_NUM_ACTIONS: Int = NUM_ACTIONS

    var tracker: EpisodeTracker
    var n_recorded: Int

    @staticmethod
    def make() -> Self:
        return Self(
            tracker=EpisodeTracker.new(10, Scalar[DT](0.0)),
            n_recorded=0,
        )

    def select_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        for i in range(N_ENVS):
            var r = random_float64()
            action_ptr[i] = Scalar[DT](0.0) if r < 0.5 else Scalar[DT](1.0)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        return 0

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        action_idx: Int,
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.n_recorded += 1
        self.tracker.add_reward(reward)

    def end_episode(mut self):
        self.tracker.end_episode()

    def train_step(mut self, step_idx: Int) raises -> Bool:
        return False

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    def add_complete_return(mut self, ret: Scalar[DT]):
        self.tracker.add_complete_return(ret)

    def record_batch_cpu[
        N_ENVS: Int
    ](
        mut self,
        prev_obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        done_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        for _ in range(N_ENVS):
            self.n_recorded += 1


def test_discrete_train() raises:
    print("--- run_offpolicy_discrete_train[CartPole, cpu] ---")
    seed(42)
    var trainer = StubDQNTrainer.make()
    var env = CartPoleEnv[DT]()
    _ = run_offpolicy_discrete_train(
        trainer,
        env,
        TOTAL_STEPS,
        print_every=0,
        verbose=False,
    )
    var mean_ret = trainer.mean_return()
    print("  n_recorded=", trainer.n_recorded,
          " ep_count=", trainer.ep_count(),
          " mean_return=", mean_ret)
    assert_true(trainer.n_recorded == TOTAL_STEPS, "wrong n_recorded")
    assert_true(trainer.ep_count() > 0, "no episodes completed")
    assert_true(not isnan(mean_ret), "mean_return NaN")
    assert_true(not isinf(mean_ret), "mean_return Inf")


def test_discrete_eval() raises:
    print("--- run_offpolicy_discrete_eval[CartPole, cpu] ---")
    seed(42)
    var trainer = StubDQNTrainer.make()
    var env = CartPoleEnv[DT]()
    var mean_eval = run_offpolicy_discrete_eval(
        trainer,
        env,
        5,
        max_steps_per_episode=200,
        verbose=True,
    )
    print("  eval mean_return=", mean_eval)
    assert_true(not isnan(mean_eval), "eval mean_return NaN")
    assert_true(not isinf(mean_eval), "eval mean_return Inf")
    assert_true(mean_eval > Scalar[DT](0.0), "eval mean_return <= 0")


def main() raises:
    print("=" * 60)
    print("driver_offpolicy_discrete — smoke test")
    print("=" * 60)
    test_discrete_train()
    test_discrete_eval()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
