# +--------------------------------------------------------------------------+ #
# | StoreReplayGpu — device-resident ring behind the ReplayBuffer seam
# +--------------------------------------------------------------------------+ #
"""GPU counterpart of `StoreReplay`, using the Stage-3 device index policy.

Kernels here are deliberately NOT imported from `gpu_replay.mojo`. Depending
on the module we intend to delete would make the deletion impossible, so this
carries its own store/gather kernels — ports of the legacy ones, gated
bit-identical against them.

Dims are comptime at this seam (`OBS`/`ACT`/`CAP` are trait parameters), so
these use static `Layout.row_major(...)` rather than the runtime-layout idiom
that `resident.mojo` needs for store-metadata dims. Both spellings exist on
purpose: static where the dims are known, runtime where they are not.

⚠ **ERE is not supported.** `GPUReplay.configure_ere` enables Emphasizing
Recent Experience — a *different index policy* (sample from a shrinking recent
window), not a storage feature. It belongs beside uniform/PER in
`sampler.mojo` and is not ported yet, so `configure_ere(enable=True)` raises
here instead of silently sampling uniformly. Call sites that use ERE must stay
on `GPUReplay` until that policy lands.

⚠ **PER is not supported yet either.** `GPUPrioritizedReplay` carries a device
sum-tree; porting it is the remaining piece of 4b.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.deep_agents.training.replay_buffer import ReplayBuffer
from mojo_rl.deep_agents.training.trainer_block import TrainerState
from .resident import IDX_DT
from .sampler import UniformDeviceSampler


# ── kernels ───────────────────────────────────────────────────────────────

def _store_one_kernel[
    OBS: Int, ACT: Int, CAP: Int
](
    stage_s: LayoutTensor[DT, Layout.row_major(OBS), MutAnyOrigin],
    stage_a: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
    stage_r: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    stage_sp: LayoutTensor[DT, Layout.row_major(OBS), MutAnyOrigin],
    stage_d: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    buf_s: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_a: LayoutTensor[DT, Layout.row_major(CAP, ACT), MutAnyOrigin],
    buf_r: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    buf_sp: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_d: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    slot: Int32,
):
    """One transition, staged host→device, written into `buf[slot]`."""
    var d = Int(thread_idx.x)
    var s = Int(slot)
    if d < OBS:
        buf_s[s, d] = stage_s[d]
        buf_sp[s, d] = stage_sp[d]
    if d < ACT:
        buf_a[s, d] = stage_a[d]
    if d == 0:
        buf_r[s] = stage_r[0]
        buf_d[s] = stage_d[0]


def _store_batch_kernel[
    N_ENVS: Int, OBS: Int, ACT: Int, CAP: Int
](
    src_s: LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin],
    src_a: LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin],
    src_r: LayoutTensor[DT, Layout.row_major(N_ENVS), MutAnyOrigin],
    src_sp: LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin],
    src_d: LayoutTensor[DT, Layout.row_major(N_ENVS), MutAnyOrigin],
    buf_s: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_a: LayoutTensor[DT, Layout.row_major(CAP, ACT), MutAnyOrigin],
    buf_r: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    buf_sp: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_d: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    start_pos: Int32,
):
    """`N_ENVS` transitions written to consecutive ring slots from
    `start_pos`, wrapping. Element-parallel over (env, obs element)."""
    comptime assert OBS >= ACT, "_store_batch_kernel assumes OBS >= ACT"
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= N_ENVS * OBS:
        return
    var e = t // OBS
    var d = t % OBS
    var slot = (Int(start_pos) + e) % CAP
    buf_s[slot, d] = src_s[e, d]
    buf_sp[slot, d] = src_sp[e, d]
    if d < ACT:
        buf_a[slot, d] = src_a[e, d]
    if d == 0:
        buf_r[slot] = src_r[e]
        buf_d[slot] = src_d[e]


def _gather_batch_kernel[
    BATCH: Int, OBS: Int, ACT: Int, CAP: Int
](
    mb_s: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    mb_a: LayoutTensor[DT, Layout.row_major(BATCH, ACT), MutAnyOrigin],
    mb_r: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    mb_sp: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    mb_d: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    buf_s: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_a: LayoutTensor[DT, Layout.row_major(CAP, ACT), MutAnyOrigin],
    buf_r: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    buf_sp: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    buf_d: LayoutTensor[DT, Layout.row_major(CAP), MutAnyOrigin],
    indices: LayoutTensor[IDX_DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    """One thread per (lane, obs element). Element-parallel rather than
    one-thread-per-lane: a per-lane kernel serialises the row copy and
    launches only BATCH threads, which cost ~73% of GPU time on pixel-wide
    rows (`project_rainbow_pong_pixel_replay_gather_bottleneck`)."""
    comptime assert OBS >= ACT, "_gather_batch_kernel assumes OBS >= ACT"
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= BATCH * OBS:
        return
    var i = t // OBS
    var d = t % OBS
    var idx = Int(indices[i])
    mb_s[i, d] = buf_s[idx, d]
    mb_sp[i, d] = buf_sp[idx, d]
    if d < ACT:
        mb_a[i, d] = buf_a[idx, d]
    if d == 0:
        mb_r[i] = buf_r[idx]
        mb_d[i] = buf_d[idx]


# ── the buffer ────────────────────────────────────────────────────────────

struct StoreReplayGpu[OBS_: Int, ACT_: Int, CAP_: Int](ReplayBuffer):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime CAP = Self.CAP_

    var obs: DeviceBuffer[DT]
    var act: DeviceBuffer[DT]
    var rew: DeviceBuffer[DT]
    var nxt: DeviceBuffer[DT]
    var dne: DeviceBuffer[DT]

    var stage_obs: DeviceBuffer[DT]
    var stage_act: DeviceBuffer[DT]
    var stage_rew: DeviceBuffer[DT]
    var stage_nxt: DeviceBuffer[DT]
    var stage_dne: DeviceBuffer[DT]
    var idx_buf: DeviceBuffer[IDX_DT]

    var _h_rew: List[Scalar[DT]]
    var _h_dne: List[Scalar[DT]]

    var size: Int
    var pos: Int
    var sampler: UniformDeviceSampler
    var batch_capacity: Int

    def __init__(
        out self,
        var obs: DeviceBuffer[DT],
        var act: DeviceBuffer[DT],
        var rew: DeviceBuffer[DT],
        var nxt: DeviceBuffer[DT],
        var dne: DeviceBuffer[DT],
        var stage_obs: DeviceBuffer[DT],
        var stage_act: DeviceBuffer[DT],
        var stage_rew: DeviceBuffer[DT],
        var stage_nxt: DeviceBuffer[DT],
        var stage_dne: DeviceBuffer[DT],
        var idx_buf: DeviceBuffer[IDX_DT],
        var _h_rew: List[Scalar[DT]],
        var _h_dne: List[Scalar[DT]],
        size: Int,
        pos: Int,
        var sampler: UniformDeviceSampler,
        batch_capacity: Int,
    ):
        self.obs = obs^
        self.act = act^
        self.rew = rew^
        self.nxt = nxt^
        self.dne = dne^
        self.stage_obs = stage_obs^
        self.stage_act = stage_act^
        self.stage_rew = stage_rew^
        self.stage_nxt = stage_nxt^
        self.stage_dne = stage_dne^
        self.idx_buf = idx_buf^
        self._h_rew = _h_rew^
        self._h_dne = _h_dne^
        self.size = size
        self.pos = pos
        self.sampler = sampler^
        self.batch_capacity = batch_capacity

    def __init__(out self, *, deinit move: Self):
        self.obs = move.obs^
        self.act = move.act^
        self.rew = move.rew^
        self.nxt = move.nxt^
        self.dne = move.dne^
        self.stage_obs = move.stage_obs^
        self.stage_act = move.stage_act^
        self.stage_rew = move.stage_rew^
        self.stage_nxt = move.stage_nxt^
        self.stage_dne = move.stage_dne^
        self.idx_buf = move.idx_buf^
        self._h_rew = move._h_rew^
        self._h_dne = move._h_dne^
        self.size = move.size
        self.pos = move.pos
        self.sampler = move.sampler^
        self.batch_capacity = move.batch_capacity

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        batch_capacity: Int = 4096,
    ) raises -> Self:
        if not ctx:
            raise Error("StoreReplayGpu.make: ctx required (GPU backend)")
        var c = ctx.value()
        var s = c.enqueue_create_buffer[DT](Self.CAP * Self.OBS)
        var a = c.enqueue_create_buffer[DT](Self.CAP * Self.ACT)
        var r = c.enqueue_create_buffer[DT](Self.CAP)
        var sp = c.enqueue_create_buffer[DT](Self.CAP * Self.OBS)
        var d = c.enqueue_create_buffer[DT](Self.CAP)
        s.enqueue_fill(Scalar[DT](0))
        a.enqueue_fill(Scalar[DT](0))
        r.enqueue_fill(Scalar[DT](0))
        sp.enqueue_fill(Scalar[DT](0))
        d.enqueue_fill(Scalar[DT](0))

        return Self(
            obs=s^, act=a^, rew=r^, nxt=sp^, dne=d^,
            stage_obs=c.enqueue_create_buffer[DT](Self.OBS),
            stage_act=c.enqueue_create_buffer[DT](Self.ACT),
            stage_rew=c.enqueue_create_buffer[DT](1),
            stage_nxt=c.enqueue_create_buffer[DT](Self.OBS),
            stage_dne=c.enqueue_create_buffer[DT](1),
            idx_buf=c.enqueue_create_buffer[IDX_DT](batch_capacity),
            _h_rew=List[Scalar[DT]](length=1, fill=Scalar[DT](0)),
            _h_dne=List[Scalar[DT]](length=1, fill=Scalar[DT](0)),
            size=0,
            pos=0,
            # Seed matches GPUReplay so a migrated call site draws the SAME
            # sequence — the parity gate depends on it.
            sampler=UniformDeviceSampler(
                0, seed=UInt64(0xC0FFEE_DECADE_0042), offset=UInt64(0)
            ),
            batch_capacity=batch_capacity,
        )

    def count(self) -> Int:
        return self.size

    def add(
        mut self,
        ref s: List[Scalar[DT]],
        ref a: List[Scalar[DT]],
        r: Scalar[DT],
        ref sp: List[Scalar[DT]],
        d: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        if not ctx:
            raise Error("StoreReplayGpu.add: ctx required")
        var c = ctx.value()
        self._h_rew[0] = r
        self._h_dne[0] = d
        c.enqueue_copy(self.stage_obs, s.unsafe_ptr())
        c.enqueue_copy(self.stage_act, a.unsafe_ptr())
        c.enqueue_copy(self.stage_rew, self._h_rew.unsafe_ptr())
        c.enqueue_copy(self.stage_nxt, sp.unsafe_ptr())
        c.enqueue_copy(self.stage_dne, self._h_dne.unsafe_ptr())

        comptime tpb = Self.OBS if Self.OBS > Self.ACT else Self.ACT
        comptime kern = _store_one_kernel[Self.OBS, Self.ACT, Self.CAP]
        c.enqueue_function[kern](
            LayoutTensor[DT, Layout.row_major(Self.OBS)](self.stage_obs),
            LayoutTensor[DT, Layout.row_major(Self.ACT)](self.stage_act),
            LayoutTensor[DT, Layout.row_major(1)](self.stage_rew),
            LayoutTensor[DT, Layout.row_major(Self.OBS)](self.stage_nxt),
            LayoutTensor[DT, Layout.row_major(1)](self.stage_dne),
            LayoutTensor[DT, Layout.row_major(Self.CAP, Self.OBS)](self.obs),
            LayoutTensor[DT, Layout.row_major(Self.CAP, Self.ACT)](self.act),
            LayoutTensor[DT, Layout.row_major(Self.CAP)](self.rew),
            LayoutTensor[DT, Layout.row_major(Self.CAP, Self.OBS)](self.nxt),
            LayoutTensor[DT, Layout.row_major(Self.CAP)](self.dne),
            Int32(self.pos),
            grid_dim=1,
            block_dim=tpb,
        )
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1

    def add_batch[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        src_obs: DeviceBuffer[DT],
        src_act: DeviceBuffer[DT],
        src_rew: DeviceBuffer[DT],
        src_nxt: DeviceBuffer[DT],
        src_dne: DeviceBuffer[DT],
    ) raises:
        """Device multi-env store — the GPU-batched driver path."""
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        comptime n_blocks = (N_ENVS * Self.OBS + TPB - 1) // TPB
        comptime kern = _store_batch_kernel[
            N_ENVS, Self.OBS, Self.ACT, Self.CAP
        ]
        ctx.enqueue_function[kern](
            LayoutTensor[DT, Layout.row_major(N_ENVS, Self.OBS)](src_obs),
            LayoutTensor[DT, Layout.row_major(N_ENVS, Self.ACT)](src_act),
            LayoutTensor[DT, Layout.row_major(N_ENVS)](src_rew),
            LayoutTensor[DT, Layout.row_major(N_ENVS, Self.OBS)](src_nxt),
            LayoutTensor[DT, Layout.row_major(N_ENVS)](src_dne),
            LayoutTensor[DT, Layout.row_major(Self.CAP, Self.OBS)](self.obs),
            LayoutTensor[DT, Layout.row_major(Self.CAP, Self.ACT)](self.act),
            LayoutTensor[DT, Layout.row_major(Self.CAP)](self.rew),
            LayoutTensor[DT, Layout.row_major(Self.CAP, Self.OBS)](self.nxt),
            LayoutTensor[DT, Layout.row_major(Self.CAP)](self.dne),
            Int32(self.pos),
            grid_dim=n_blocks,
            block_dim=TPB,
        )
        self.pos = (self.pos + N_ENVS) % Self.CAP
        self.size = self.size + N_ENVS
        if self.size > Self.CAP:
            self.size = Self.CAP

    def sample_into[
        BATCH: Int
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, BATCH],
    ) raises:
        if BATCH > self.batch_capacity:
            raise Error(
                "StoreReplayGpu.sample_into: BATCH=" + String(BATCH)
                + " exceeds batch_capacity=" + String(self.batch_capacity)
            )
        var ctx = state.ctx.value()
        self.sampler.n_rows = self.size
        self.sampler.draw_into_device(ctx, self.idx_buf, BATCH)

        comptime n_blocks = (BATCH * Self.OBS + TPB - 1) // TPB
        comptime kern = _gather_batch_kernel[
            BATCH, Self.OBS, Self.ACT, Self.CAP
        ]
        ctx.enqueue_function[kern](
            LayoutTensor[DT, Layout.row_major(BATCH, Self.OBS)](
                state.mb_s.dev.value()
            ),
            LayoutTensor[DT, Layout.row_major(BATCH, Self.ACT)](
                state.mb_a.dev.value()
            ),
            LayoutTensor[DT, Layout.row_major(BATCH)](state.mb_r.dev.value()),
            LayoutTensor[DT, Layout.row_major(BATCH, Self.OBS)](
                state.mb_sp.dev.value()
            ),
            LayoutTensor[DT, Layout.row_major(BATCH)](state.mb_d.dev.value()),
            LayoutTensor[DT, Layout.row_major(Self.CAP, Self.OBS)](self.obs),
            LayoutTensor[DT, Layout.row_major(Self.CAP, Self.ACT)](self.act),
            LayoutTensor[DT, Layout.row_major(Self.CAP)](self.rew),
            LayoutTensor[DT, Layout.row_major(Self.CAP, Self.OBS)](self.nxt),
            LayoutTensor[DT, Layout.row_major(Self.CAP)](self.dne),
            LayoutTensor[IDX_DT, Layout.row_major(BATCH)](self.idx_buf),
            grid_dim=n_blocks,
            block_dim=TPB,
        )

    def configure_ere(
        mut self,
        enable: Bool = False,
        eta: Scalar[DT] = Scalar[DT](0.996),
        c_min: Int = 1,
        k_max: Int = 1000,
    ) raises:
        """ERE is an index POLICY, not a storage feature, and is not ported.

        Raising beats the trait's silent no-op default: a call site that asked
        for recency-biased sampling and got uniform would train differently
        with no signal at all.
        """
        if enable:
            raise Error(
                "StoreReplayGpu: ERE is not supported yet. ERE is a distinct"
                " index policy (sample from a shrinking recent window); it"
                " belongs in data/sampler.mojo beside uniform/PER. Keep this"
                " call site on GPUReplay until it lands."
            )
