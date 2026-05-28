"""AlphaUpdateStep — ScalarAdam step on log_alpha.

Reads state.log_prob_mean → `alpha_opt.step(-(log_prob_mean + H_target))`.
Holds target_entropy as a small hyperparam.
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.optimizer.scalar_adam import ScalarAdam
from ..trainer_block import TrainerState


struct AlphaUpdateStep[
    OBS_: Int, ACT_: Int, BATCH_: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var target_entropy: Scalar[DT]

    def __init__(out self):
        self.target_entropy = Scalar[DT](-1.0)

    @staticmethod
    def make(target_entropy: Scalar[DT]) -> Self:
        var b = Self()
        b.target_entropy = target_entropy
        return b^

    def step(
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut alpha_opt: ScalarAdam,
    ) raises:
        alpha_opt.step(-(state.log_prob_mean + self.target_entropy))
