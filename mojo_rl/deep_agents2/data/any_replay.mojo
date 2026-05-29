"""AnyReplay[target, OBS, ACT, CAP] — target-selecting ReplayBuffer leaf.

The single place where `target` (a `StaticString`) is mapped to a
concrete buffer backend. Mojo can't select a field *type* by target
(no type-ternary, no struct-body `comptime if`), so this leaf carries
BOTH backends as `Optional` and dispatches each trait method via
method-body `comptime if` — the nn2 carry-both idiom, confined to one
~60-line shim instead of smeared across every sample block.

Only the selected backend is constructed (`make`); the other stays
`None`. GPUReplay can't exist without a ctx (its DeviceBuffers need
one), which is why both backends are `Optional` rather than values.

This is the conformer the C51 config presets plug into so a single
`ReplaySampleStep[AnyReplay[target, …], BATCH]` covers cpu and gpu with
no per-target block duplication.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from ..training.replay_buffer import ReplayBuffer
from ..training.trainer_block import TrainerState
from .cpu_replay import CPUReplay
from .gpu_replay import GPUReplay


@fieldwise_init
struct AnyReplay[
    target: StaticString, OBS_: Int, ACT_: Int, CAP_: Int
](ReplayBuffer):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    var cpu: Optional[CPUReplay[Self.OBS_, Self.ACT_, Self.CAP_]]
    var gpu: Optional[GPUReplay[Self.OBS_, Self.ACT_, Self.CAP_]]

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        batch_capacity: Int = 4096,
    ) raises -> Self:
        comptime assert (
            Self.target == "cpu" or Self.target == "gpu"
        ), "AnyReplay: target must be 'cpu' or 'gpu'"
        comptime if Self.target == "cpu":
            return Self(
                cpu=CPUReplay[Self.OBS_, Self.ACT_, Self.CAP_].make(),
                gpu=None,
            )
        else:
            return Self(
                cpu=None,
                gpu=GPUReplay[Self.OBS_, Self.ACT_, Self.CAP_].make(
                    ctx=ctx, batch_capacity=batch_capacity
                ),
            )

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
            raise Error("AnyReplay[cpu].add_batch: device path unavailable")
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
        comptime if Self.target == "cpu":
            pass
        else:
            self.gpu.value().configure_ere(
                enable=enable, eta=eta, c_min=c_min, k_max=k_max
            )
