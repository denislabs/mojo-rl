"""PolyakBlock — twin-critic target polyak update.

Holds pointers to the trainer's pair1 + pair2 (OnlineTargetPair[CRITIC])
and the τ rate. No state reads/writes."""

from ...constants import DT
from ...core.module import Module
from ...core.online_target_pair import OnlineTargetPair
from ..trainer_block import TrainerBlock, TrainerState


struct PolyakBlock[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    CRITIC: Module,
](TrainerBlock):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    alias Pair = OnlineTargetPair[Self.CRITIC]

    var pair1_ptr: UnsafePointer[Self.Pair, MutAnyOrigin]
    var pair2_ptr: UnsafePointer[Self.Pair, MutAnyOrigin]
    var tau: Scalar[DT]

    def __init__(out self):
        self.pair1_ptr = UnsafePointer[Self.Pair, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.pair2_ptr = UnsafePointer[Self.Pair, MutAnyOrigin](
            unsafe_from_address=0
        )
        self.tau = Scalar[DT](0.005)

    def bind(
        mut self,
        pair1_ptr: UnsafePointer[Self.Pair, MutAnyOrigin],
        pair2_ptr: UnsafePointer[Self.Pair, MutAnyOrigin],
        tau: Scalar[DT],
    ):
        self.pair1_ptr = pair1_ptr
        self.pair2_ptr = pair2_ptr
        self.tau = tau

    def step_via[target: StaticString = "cpu"](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        comptime if target == "cpu":
            self.pair1_ptr[].polyak_step["cpu"](self.tau)
            self.pair2_ptr[].polyak_step["cpu"](self.tau)
        else:
            self.pair1_ptr[].polyak_step["gpu"](self.tau, state.ctx)
            self.pair2_ptr[].polyak_step["gpu"](self.tau, state.ctx)
