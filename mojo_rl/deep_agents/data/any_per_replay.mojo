"""AnyPerReplay[target, OBS, ACT, CAP] — target-selecting PER buffer.

PER counterpart of `AnyReplay`: maps `target` to either
`CPUPrioritizedReplay` or `GPUPrioritizedReplay`, carrying both as
`Optional` and dispatching each `ReplayBuffer` trait method (including
the PER hooks `configure_per` / `set_beta` / `update_priorities`) via
method-body `comptime if`. Only the selected backend is constructed.

Lets the Rainbow config preset plug into a single
`NStepSampleStep[N, AnyPerReplay[target, …], BATCH]` covering cpu and
gpu with no per-target block duplication.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from ..training.replay_buffer import ReplayBuffer
from ..training.trainer_block import TrainerState
from .cpu_per_replay import CPUPrioritizedReplay
from .per_replay import GPUPrioritizedReplay


@fieldwise_init
struct AnyPerReplay[
    target: StaticString, OBS_: Int, ACT_: Int, CAP_: Int,
    OBS_STORE_DT_: DType = DT,
    DEVICE_TREE_: Bool = True,
](ReplayBuffer):
    """`OBS_STORE_DT_` (default `DT`) selects the GPU backend's obs
    storage dtype (`uint8` = pixel-obs capacity option).
    `DEVICE_TREE_` (default True) selects the GPU PER sum-tree backend
    — device-resident (capture-ready) vs the host-tree oracle. CPU
    backend ignores both."""

    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    var cpu: Optional[CPUPrioritizedReplay[Self.OBS_, Self.ACT_, Self.CAP_]]
    var gpu: Optional[
        GPUPrioritizedReplay[
            Self.OBS_, Self.ACT_, Self.CAP_,
            Self.OBS_STORE_DT_, Self.DEVICE_TREE_,
        ]
    ]

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        batch_capacity: Int = 4096,
    ) raises -> Self:
        comptime assert (
            Self.target == "cpu" or Self.target == "gpu"
        ), "AnyPerReplay: target must be 'cpu' or 'gpu'"
        comptime if Self.target == "cpu":
            comptime assert Self.OBS_STORE_DT_ == DT, (
                "AnyPerReplay[cpu]: OBS_STORE_DT is a GPU-backend option"
            )
            return Self(
                cpu=CPUPrioritizedReplay[
                    Self.OBS_, Self.ACT_, Self.CAP_
                ].make(batch_capacity=batch_capacity),
                gpu=None,
            )
        else:
            return Self(
                cpu=None,
                gpu=GPUPrioritizedReplay[
                    Self.OBS_, Self.ACT_, Self.CAP_,
                    Self.OBS_STORE_DT_, Self.DEVICE_TREE_,
                ].make(ctx=ctx, batch_capacity=batch_capacity),
            )

    def configure_per(
        mut self,
        alpha: Scalar[DT] = Scalar[DT](0.6),
        beta: Scalar[DT] = Scalar[DT](0.4),
        epsilon: Scalar[DT] = Scalar[DT](1e-6),
    ):
        comptime if Self.target == "cpu":
            self.cpu.value().configure_per(alpha=alpha, beta=beta, epsilon=epsilon)
        else:
            self.gpu.value().configure_per(alpha=alpha, beta=beta, epsilon=epsilon)

    def set_beta(mut self, beta: Scalar[DT]):
        comptime if Self.target == "cpu":
            self.cpu.value().set_beta(beta)
        else:
            self.gpu.value().set_beta(beta)

    def add(
        mut self,
        ref s: List[Scalar[DT]],
        ref a: List[Scalar[DT]],
        r: Scalar[DT],
        ref sp: List[Scalar[DT]],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if Self.target == "cpu":
            self.cpu.value().add(s, a, r, sp, d, ctx=ctx)
        else:
            self.gpu.value().add(s, a, r, sp, d, ctx=ctx)

    def sample_into[BATCH: Int](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        comptime if Self.target == "cpu":
            self.cpu.value().sample_into[BATCH](state)
        else:
            self.gpu.value().sample_into[BATCH](state)

    def update_priorities[BATCH: Int](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        comptime if Self.target == "cpu":
            self.cpu.value().update_priorities[BATCH](state)
        else:
            self.gpu.value().update_priorities[BATCH](state)

    def count(self) -> Int:
        comptime if Self.target == "cpu":
            return self.cpu.value().count()
        else:
            return self.gpu.value().count()

    def add_batch[N_ENVS: Int](
        mut self,
        ctx: DeviceContext,
        src_obs: DeviceBuffer[DT],
        src_act: DeviceBuffer[DT],
        src_rew: DeviceBuffer[DT],
        src_nxt: DeviceBuffer[DT],
        src_dne: DeviceBuffer[DT],
    ) raises:
        comptime if Self.target == "cpu":
            raise Error("AnyPerReplay[cpu].add_batch: device path unavailable")
        else:
            self.gpu.value().add_batch[N_ENVS](
                ctx, src_obs, src_act, src_rew, src_nxt, src_dne
            )

    def configure_ere(
        mut self,
        enable: Bool = False,
        eta: Scalar[DT] = Scalar[DT](0.996),
        c_min: Int = 1,
        k_max: Int = 1000,
    ) raises:
        # GPU PER has no ERE path; inherits the no-op trait default.
        comptime if Self.target == "cpu":
            pass
        else:
            self.gpu.value().configure_ere(
                enable=enable, eta=eta, c_min=c_min, k_max=k_max
            )
