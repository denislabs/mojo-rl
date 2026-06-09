"""MultiTaskEnv — a 2-task lighthouse harness for multi-task TD-MPC2 (item C).

Holds PendulumV2 (obs3/act1, non-episodic) + InvertedPendulum (obs4/act1,
episodic). Round-robins the task each episode and exposes the unified contract
the multi-task agent expects:

  * `MAX_OBS = 4`, `MAX_ACT = 1`, `NUM_TASKS = 2`.
  * obs zero-padded to MAX_OBS (Pendulum fills [0:3], leaves [3]=0).
  * a single MAX_ACT=1 action, scaled per task to the env's native torque range
    (Pendulum ±2, InvertedPendulum ±3) — the agent outputs tanh∈[-1,1].
  * `task_id()`, `was_terminated()` (Pendulum non-episodic → False), and a
    per-task `action_mask` (≡1 here — both ACT=1; the mask machinery exists for
    heterogeneous-action suites, validated by the synthetic-mask test).

Obs padding lives HERE so the replay/agent always see `[MAX_OBS]` frames.
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.envs.inverted_pendulum import InvertedPendulum


struct MultiTaskEnv:
    comptime MAX_OBS = 4
    comptime MAX_ACT = 1
    comptime NUM_TASKS = 2

    var pendulum: PendulumV2[DT]
    var inverted: InvertedPendulum[DT, True]
    var cur_task: Int

    def __init__(out self):
        self.pendulum = PendulumV2[DT]()
        self.inverted = InvertedPendulum[DT, True]()
        self.cur_task = 1  # so the first reset() advances to task 0

    def task_id(self) -> Int:
        return self.cur_task

    def _scale(self) -> Scalar[DT]:
        return Scalar[DT](2.0) if self.cur_task == 0 else Scalar[DT](3.0)

    def _pad(self, obs: List[Scalar[DT]]) -> List[Scalar[DT]]:
        var o = List[Scalar[DT]]()
        for _ in range(Self.MAX_OBS):
            o.append(Scalar[DT](0.0))
        for i in range(len(obs)):
            if i < Self.MAX_OBS:
                o[i] = obs[i]
        return o^

    def reset(mut self) raises -> List[Scalar[DT]]:
        """Advance to the next task (round-robin) and reset it; return padded obs."""
        self.cur_task = (self.cur_task + 1) % Self.NUM_TASKS
        if self.cur_task == 0:
            return self._pad(self.pendulum.reset_obs_list())
        return self._pad(self.inverted.reset_obs_list())

    def reset_current(mut self) raises -> List[Scalar[DT]]:
        """Reset the CURRENT task (no round-robin) — used on episode boundary."""
        if self.cur_task == 0:
            return self._pad(self.pendulum.reset_obs_list())
        return self._pad(self.inverted.reset_obs_list())

    def step(
        mut self, action: List[Scalar[DT]]
    ) raises -> Tuple[List[Scalar[DT]], Scalar[DT], Bool]:
        var a = List[Scalar[DT]]()
        a.append(action[0] * self._scale())
        if self.cur_task == 0:
            var r = self.pendulum.step_continuous_vec[DT](a)
            return (self._pad(r[0]), r[1], r[2])
        var r2 = self.inverted.step_continuous_vec[DT](a)
        return (self._pad(r2[0]), r2[1], r2[2])

    def was_terminated(self) -> Bool:
        if self.cur_task == 0:
            return False
        return self.inverted.was_terminated()

    def action_mask(
        self, task: Int, dst: UnsafePointer[Scalar[DT], MutAnyOrigin]
    ):
        # Both tasks use all MAX_ACT=1 dims → mask ≡ 1.
        for j in range(Self.MAX_ACT):
            dst[j] = Scalar[DT](1.0)
