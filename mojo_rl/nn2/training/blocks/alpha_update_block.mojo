"""AlphaUpdateBlock — ScalarAdam step on log_alpha.

Reads state.log_prob_mean → does `alpha_opt.step(-(log_prob_mean + H_target))`.
Holds a pointer to the trainer's alpha_opt + a copy of target_entropy.
Carries OBS/ACT/BATCH markers only to satisfy the TrainerBlock trait shape."""

from ...constants import DT
from ...optimizer.scalar_adam import ScalarAdam
from ..trainer_block import TrainerBlock, TrainerState


struct AlphaUpdateBlock[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
](TrainerBlock):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var alpha_opt_ptr:  UnsafePointer[ScalarAdam, MutAnyOrigin]
    var target_entropy: Scalar[DT]

    def __init__(out self):
        self.alpha_opt_ptr = UnsafePointer[ScalarAdam, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.target_entropy = Scalar[DT](-1.0)

    def bind(
        mut self,
        alpha_opt_ptr: UnsafePointer[ScalarAdam, MutAnyOrigin],
        target_entropy: Scalar[DT],
    ):
        self.alpha_opt_ptr = alpha_opt_ptr
        self.target_entropy = target_entropy

    def step_via[target: StaticString = "cpu"](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        self.alpha_opt_ptr[].step(
            -(state.log_prob_mean + self.target_entropy)
        )
