"""AlphaUpdateStep — ScalarAdam step on log_alpha.

CPU: reads state.log_prob_mean → `alpha_opt.step(-(log_prob_mean + H_target))`.
GPU (Slice 4): the entropy grad lives in the actor-loss `lp_mean` device
buffer; `alpha_opt.step_device` reads it on-device and refreshes the device
α the `Scale` nodes consume — no host scalar, CUDA-graph capturable.
Holds target_entropy as a small hyperparam.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.nn.constants import DT
from mojo_rl.nn.optimizer.scalar_adam import ScalarAdam
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
        lp_mean: Optional[DeviceBuffer[DT]] = None,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            # Host scalar grad (SAC CPU bit-identity path).
            alpha_opt.step(-(state.log_prob_mean + self.target_entropy))
        else:
            # Device grad: ScalarAdam reads `lp_mean_ptr[0]` on-device,
            # forms `-(lp_mean + H_target)`, and writes the device α in
            # place. No host work, no D2H.
            alpha_opt.step_device(
                ctx.value(),
                lp_mean.value(),
                self.target_entropy,
            )
