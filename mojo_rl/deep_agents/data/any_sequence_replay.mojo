"""AnySequenceReplay[target, OBS, ACT, CAP] — target-selecting sequence replay.

The sequence-replay sibling of `AnyReplay` (single-transition). The single
place where `target` (a `StaticString`) maps to a concrete window-replay
backend. Mojo can't select a field *type* by target (no type-ternary — a
conditional `A if c else B` over two distinct struct types fails to unify —
and no struct-body `comptime if`), so this leaf carries BOTH backends as
`Optional` and dispatches each method via method-body `comptime if`: the nn
carry-both idiom, confined to one shim.

Only the selected backend is constructed (`make`); the other stays `None`.
`GPUSequenceReplay` can't exist without a ctx (its DeviceBuffers need one), so
both backends are `Optional` rather than values.

Surface = the subset of `SequenceReplay` / `GPUSequenceReplay` the DreamerV3
trainer uses: `record` / `record_terminal` / `count` (window collection),
`sample_batch_fst` (CPU host draw → `state.mb_*`) and `sample_batch_fst_dev`
(GPU device draw → the WM device buffers). Each `sample_batch_*` raises on the
wrong backend, but the trainer only ever calls the one matching its target
(both call sites sit under `comptime if train_target == ...`).
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from .sequence_replay import SequenceReplay
from .gpu_sequence_replay import GPUSequenceReplay


@fieldwise_init
struct AnySequenceReplay[
    target: StaticString, OBS_: Int, ACT_: Int, CAP_: Int,
](Movable & ImplicitlyDeletable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    var cpu: Optional[SequenceReplay[Self.OBS_, Self.ACT_, Self.CAP_]]
    var gpu: Optional[GPUSequenceReplay[Self.OBS_, Self.ACT_, Self.CAP_]]

    @staticmethod
    def make(ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert (
            Self.target == "cpu" or Self.target == "gpu"
        ), "AnySequenceReplay: target must be 'cpu' or 'gpu'"
        comptime if Self.target == "cpu":
            return Self(
                cpu=SequenceReplay[Self.OBS_, Self.ACT_, Self.CAP_].make["cpu"](),
                gpu=None,
            )
        else:
            return Self(
                cpu=None,
                gpu=GPUSequenceReplay[Self.OBS_, Self.ACT_, Self.CAP_].make[
                    "gpu"
                ](ctx=ctx),
            )

    def count(self) -> Int:
        comptime if Self.target == "cpu":
            return self.cpu.value().count()
        else:
            return self.gpu.value().count()

    def set_online(mut self, every: Int):
        """Enable the reference `online: True` queue on the active backend
        (every fresh length-`every` window is sampled exactly once, promptly).
        Pass the training window length T; 0 disables (the default)."""
        comptime if Self.target == "cpu":
            self.cpu.value().set_online(every)
        else:
            self.gpu.value().set_online(every)

    def record(
        mut self,
        s: UnsafePointer[Scalar[DT], MutAnyOrigin],
        a: UnsafePointer[Scalar[DT], MutAnyOrigin],
        r: Scalar[DT],
        d: Scalar[DT],
    ) raises:
        comptime if Self.target == "cpu":
            self.cpu.value().record(s, a, r, d)
        else:
            self.gpu.value().record(s, a, r, d)

    def record_terminal(
        mut self, s: UnsafePointer[Scalar[DT], MutAnyOrigin]
    ) raises:
        comptime if Self.target == "cpu":
            self.cpu.value().record_terminal(s)
        else:
            self.gpu.value().record_terminal(s)

    def sample_batch_fst[
        B: Int, T: Int,
    ](
        mut self,
        obs_out: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, T+1, OBS]
        act_out: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, T, ACT]
        rew_out: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, T]
        dne_out: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, T]
        fst_out: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, T+1]
    ) raises:
        comptime if Self.target == "cpu":
            self.cpu.value().sample_batch_fst[B, T](
                obs_out, act_out, rew_out, dne_out, fst_out
            )
        else:
            raise Error(
                "AnySequenceReplay[gpu].sample_batch_fst: host draw"
                " unavailable — use sample_batch_fst_dev"
            )

    def sample_batch_fst_dev[
        B: Int, T: Int,
    ](
        mut self,
        ctx: DeviceContext,
        obs_dev: DeviceBuffer[DT],
        act_dev: DeviceBuffer[DT],
        rew_dev: DeviceBuffer[DT],
        dne_dev: DeviceBuffer[DT],
        fst_dev: DeviceBuffer[DT],
    ) raises:
        comptime if Self.target == "cpu":
            raise Error(
                "AnySequenceReplay[cpu].sample_batch_fst_dev: device draw"
                " unavailable — use sample_batch_fst"
            )
        else:
            self.gpu.value().sample_batch_fst_dev[B, T](
                ctx, obs_dev, act_dev, rew_dev, dne_dev, fst_dev
            )
