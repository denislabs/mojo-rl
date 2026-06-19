"""AlphaUpdateStep — ScalarAdam step on log_alpha (storage).

Reads `state.log_prob_mean` → `alpha_opt.step(-(log_prob_mean + H_target))`.

STORAGE migration (Stage 5): α is a HOST scalar on BOTH targets (the device-α /
on-device-lp_mean / CUDA-graph path is deferred — capture is deferred project-
wide). On GPU the actor-loss block D2Hs `log_prob_mean` once per step and stores
it in `state.log_prob_mean`, so this step is target-agnostic host ScalarAdam.
Holds `target_entropy` as a small hyperparam.
"""

from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.optimizer.scalar_adam import ScalarAdam
from ...training.trainer_block import TrainerState


struct AlphaUpdateStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
](Defaultable & Movable & ImplicitlyDeletable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var target_entropy: Scalar[DT]

    def __init__(out self):
        self.target_entropy = -1.0

    @staticmethod
    def make(target_entropy: Scalar[DT]) -> Self:
        var b = Self()
        b.target_entropy = target_entropy
        return b^

    def step[
        target: StaticString,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut alpha_opt: ScalarAdam,
    ) raises:
        # grad = -(E[log π] + H_target); host ScalarAdam on log_alpha.
        alpha_opt.step(-(state.log_prob_mean + self.target_entropy))
