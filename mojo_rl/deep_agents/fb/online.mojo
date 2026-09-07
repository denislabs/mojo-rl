"""`FBOnlineAgent` — Forward-Backward trained ONLINE, off-policy, on a batched
GPU env. `docs/BFM_ZERO_SHOT_RL.md` §18.3 step A3.

## Why this exists

The offline `FBTrainer` runs (§13, M2) recover ~19 / 5 / 5 % of the SAC experts
in their own dataset and the curve is flat past 300 k steps. §16.2 names the
constraint: FB run offline is bounded by the COVERAGE of a frozen dataset, and
the field's answer (FB-CPR, BFM-Zero) is no dataset at all — `pi_z` rolls out
in many parallel envs and generates its own coverage. This agent is that
change and ONLY that change: no discriminator, no style critic, no BC term by
default, so that a result is attributable to the coverage effect and nothing
else (§16.7's "isolate one change at a time").

It wraps the unchanged `FBTrainer` (one step body, already capture-safe) in
the `OffPolicyAgentGpu` surface that `run_offpolicy_train_batched` drives.
Nothing in the FB step is re-implemented here; what is new is what an online
loop needs around it:

  * a DEVICE replay ring `[CAP] x (obs | act | next_obs | z | terminated)`,
    written by `record_batch_gpu` and gathered by the kernels FB already has;
  * a per-lane `z` held for `z_hold` env steps then resampled from a mixture
    of the sphere and a FIFO `ZBuffer` of recently trained `z` — BFM-Zero's
    rollout rule (`use_mix_rollout`, buffer of 10 k, hold 150);
  * `relabel_ratio`: 80 % of each minibatch's STORED `z` is overwritten by a
    fresh draw from the training mixture (uniform sphere / `B(s+)`), the
    remaining 20 % keeps the `z` the action was taken under;
  * `s+` drawn INDEPENDENTLY from the ring — the invariant `loss.mojo` states
    and cannot enforce.

## Capture safety — the two counters that must live on device

The driver captures `train_device_kernels` ONCE past `learning_starts` and
replays it. Two host quantities would be baked in at that point and silently
frozen forever:

  * the replay FILL. The uniform index draw reads `size` from a 1-element
    device buffer written by `record_batch_gpu` (`_set_size_kernel`), exactly
    as `data/replay_gpu.mojo` does — its header calls the host form "the
    catastrophic-divergence bug": sampling pinned to the warmup rows for the
    rest of the run, loss still descending.
  * every Philox OFFSET used inside the step: the index draws, the relabel
    uniforms, the mixture Gaussians, and `FBTrainer`'s own smoothing noise.
    All read a device offset and bump it with a kernel in-sequence.

`select_action_batched` and `record_batch_gpu` are EAGER (the driver never
captures them), so their RNG offsets stay host scalars.

⚠ `train_device_kernels_on(gctx)` is NOT overridden: `FBTrainer` enqueues on
its own stored context and cannot be redirected without changing it. Use the
default backend (`MOJO_RL_GRAPH_BACKEND=stream`); the trait default RAISES on
the device-graph backend rather than recording a partial step.

## What is deliberately NOT here

  * Observation normalisation. BFM-Zero runs a `BatchNorm1d` on every input;
    the offline sweep's `obsnorm` arm is what decides whether it matters, and
    a running normaliser under a TD bootstrap is a moving target that needs
    its own gate. Inputs are raw, as every §13 number was.
  * The discriminator / `Q_D` (A4). By design — see the top of this file.
  * Termination masking in the measure target. `FBTrainer` has none, because
    dm_control tasks never terminate; `terminated` is stored in the ring so a
    terminating env (the task family) can add it without a ring change.
"""

from std.gpu import global_idx, thread_idx
from std.math import sqrt, abs
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.call import call_forward
from mojo_rl.nn.core.initializer import Initializer, Xavier
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.nn.random.box_muller import (
    box_muller_normal_gpu,
    advance_rng_offset_kernel,
)
from mojo_rl.data.resident import IDX_DT
from mojo_rl.data.replay_gpu import (
    _uniform_indices_dev_kernel,
    _incr_offset_kernel,
    _set_size_kernel,
)

from ..data.n_step_replay import GPUNStepBuffer
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.blocks.action_select import warmup_uniform_batched
from .trainer import FBTrainer
from .loss import fb_measure_loss_into, fb_ortho_loss_into
from .kernels import (
    gather_rows_kernel,
    pack2_kernel,
    project_sphere_kernel,
    gaussian_dev_t,
    uniform01_kernel,
    mean_sq_t,
    ensure_t,
    _blocks,
)


# ══════════════════════════════════════════════════════════════════════
# Kernels
# ══════════════════════════════════════════════════════════════════════


def uniform01_dev_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    seed: UInt64,
    offset_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """`dst[i] ~ U[0, 1)`, Philox, offset read FROM DEVICE (capture-safe).
    Device-offset twin of `kernels.uniform01_kernel` — see its docstring for
    why the mixture kernels must be fed UNIFORMS."""
    var i = Int(global_idx.x)
    if i >= N:
        return
    var philox = PhiloxRandom(
        seed=seed + UInt64(i), offset=rebind[UInt64](offset_buf[0])
    )
    dst[unsafe_offset=i] = Scalar[DT](Float32(philox.step_uniform()[0]))


def ring_store_kernel[
    OBS: Int, ACT: Int, D: Int, CAP: Int, LANES: Int
](
    obs_src: Pointer[Scalar[DT], MutAnyOrigin],
    act_src: Pointer[Scalar[DT], MutAnyOrigin],
    nxt_src: Pointer[Scalar[DT], MutAnyOrigin],
    term_src: Pointer[Scalar[DT], MutAnyOrigin],
    z_src: Pointer[Scalar[DT], MutAnyOrigin],
    r_obs: Pointer[Scalar[DT], MutAnyOrigin],
    r_act: Pointer[Scalar[DT], MutAnyOrigin],
    r_nxt: Pointer[Scalar[DT], MutAnyOrigin],
    r_term: Pointer[Scalar[DT], MutAnyOrigin],
    r_z: Pointer[Scalar[DT], MutAnyOrigin],
    pos: Int32,
):
    """Append `LANES` transitions at rows `(pos + lane) % CAP`, one launch.

    Element-parallel over the concatenated row width so a wide `z` (128) does
    not serialise inside a per-lane thread. `pos` is a host scalar because
    `record_batch_gpu` is eager — the driver never captures it.
    """
    comptime W = OBS + ACT + OBS + D + 1
    var t = Int(global_idx.x)
    if t >= LANES * W:
        return
    var lane = t // W
    var k = t % W
    var row = (Int(pos) + lane) % CAP
    if k < OBS:
        r_obs[unsafe_offset=row * OBS + k] = obs_src[unsafe_offset=lane * OBS + k]
    elif k < OBS + ACT:
        var j = k - OBS
        r_act[unsafe_offset=row * ACT + j] = act_src[unsafe_offset=lane * ACT + j]
    elif k < OBS + ACT + OBS:
        var j = k - OBS - ACT
        r_nxt[unsafe_offset=row * OBS + j] = nxt_src[unsafe_offset=lane * OBS + j]
    elif k < OBS + ACT + OBS + D:
        var j = k - OBS - ACT - OBS
        r_z[unsafe_offset=row * D + j] = z_src[unsafe_offset=lane * D + j]
    else:
        r_term[unsafe_offset=row] = term_src[unsafe_offset=lane]


def z_lane_resample_kernel[D: Int, LANES: Int, ZBUF: Int](
    z: Pointer[Scalar[DT], MutAnyOrigin],
    gauss: Pointer[Scalar[DT], MutAnyOrigin],
    zbuf: Pointer[Scalar[DT], MutAnyOrigin],
    zbuf_fill: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
    pick: Pointer[Scalar[DT], MutAnyOrigin],
    iter: Int32,
    hold: Int32,
    zbuf_frac: Scalar[DT],
    force: Int32,
):
    """Per lane: if this lane's `z` is due, replace it.

    A lane is due every `hold` iterations, STAGGERED by lane so the whole
    population does not switch task on the same step (`force != 0` makes
    every lane due — used once to initialise). With probability `zbuf_frac`
    the new `z` is a row of the ZBuffer (a `z` recently trained on, so the
    rollout exercises the region the losses are shaping); otherwise a
    Gaussian, to be projected by `project_sphere_kernel` right after.

    ⚠ NOT renormalised here. The projection runs unconditionally on every
    lane after this kernel, whether or not it was resampled — the invariant
    `z_sampler.mojo` states is "every producer renormalises", and a lane
    that skipped the projection because it was not due is one refactor from
    a lane that skipped it because of a bug.
    """
    var e = Int(global_idx.x)
    if e >= LANES:
        return
    var h = Int(hold)
    if h < 1:
        h = 1
    var stagger = (e * h) // LANES
    var due = (Int(force) != 0) or (((Int(iter) + stagger) % h) == 0)
    if not due:
        return
    var fill = Int(zbuf_fill[0])
    var base = e * D
    if fill > 0 and pick[unsafe_offset=2 * e] < zbuf_frac:
        var src = Int(pick[unsafe_offset=2 * e + 1] * Scalar[DT](fill))
        if src >= fill:
            src = fill - 1
        if src < 0:
            src = 0
        for k in range(D):
            z[unsafe_offset=base + k] = zbuf[unsafe_offset=src * D + k]
    else:
        for k in range(D):
            z[unsafe_offset=base + k] = gauss[unsafe_offset=base + k]


def z_relabel_kernel[D: Int, BATCH: Int](
    z: Pointer[Scalar[DT], MutAnyOrigin],
    gauss: Pointer[Scalar[DT], MutAnyOrigin],
    b_states: Pointer[Scalar[DT], MutAnyOrigin],
    pick: Pointer[Scalar[DT], MutAnyOrigin],
    keep_frac: Scalar[DT],
    uniform_frac: Scalar[DT],
):
    """BFM-Zero's `relabel_ratio`, one thread per ROW of the gathered batch.

    `z` arrives holding the STORED `z` of each transition. With probability
    `keep_frac` a row keeps it; otherwise it is overwritten by the training
    mixture — `uniform_frac` of the time a Gaussian (→ sphere), else the
    `B(s+)` embedding of a random row of the batch. Three uniforms per row in
    `pick`; drawn outside so this kernel holds no RNG state.
    """
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var base = i * D
    if pick[unsafe_offset=3 * i] < keep_frac:
        return
    if pick[unsafe_offset=3 * i + 1] < uniform_frac:
        for k in range(D):
            z[unsafe_offset=base + k] = gauss[unsafe_offset=base + k]
    else:
        var src = Int(pick[unsafe_offset=3 * i + 2] * Scalar[DT](BATCH))
        if src >= BATCH:
            src = BATCH - 1
        if src < 0:
            src = 0
        for k in range(D):
            z[unsafe_offset=base + k] = b_states[unsafe_offset=src * D + k]


def zbuf_push_kernel[D: Int, BATCH: Int, ZBUF: Int](
    zbuf: Pointer[Scalar[DT], MutAnyOrigin],
    z: Pointer[Scalar[DT], MutAnyOrigin],
    head: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
):
    """Append the batch's `z` rows to the FIFO at `(head + i) % ZBUF`. The
    head is read from device (captured path); `zbuf_advance_kernel` bumps it
    AFTER this launch so every thread reads the same head."""
    var t = Int(global_idx.x)
    if t >= BATCH * D:
        return
    var i = t // D
    var k = t % D
    var row = (Int(head[0]) + i) % ZBUF
    zbuf[unsafe_offset=row * D + k] = z[unsafe_offset=t]


def zbuf_advance_kernel[BATCH: Int, ZBUF: Int](
    head: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
    fill: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
):
    if Int(thread_idx.x) != 0:
        return
    head[0] = Int32((Int(head[0]) + BATCH) % ZBUF)
    var f = Int(fill[0]) + BATCH
    if f > ZBUF:
        f = ZBUF
    fill[0] = Int32(f)


def explore_action_kernel[N: Int](
    action: Pointer[Scalar[DT], MutAnyOrigin],
    pi: Pointer[Scalar[DT], MutAnyOrigin],
    noise: Pointer[Scalar[DT], MutAnyOrigin],
    std: Scalar[DT],
    scale: Scalar[DT],
):
    """`action = clamp(pi + std·n, ±1) · scale` — BFM-Zero's `actor_std`
    exploration on a deterministic actor. `std = 0` is the greedy path."""
    var t = Int(global_idx.x)
    if t >= N:
        return
    var v = pi[unsafe_offset=t] + std * noise[unsafe_offset=t]
    if v > Scalar[DT](1.0):
        v = Scalar[DT](1.0)
    elif v < Scalar[DT](-1.0):
        v = Scalar[DT](-1.0)
    action[unsafe_offset=t] = v * scale


# ══════════════════════════════════════════════════════════════════════
# The agent
# ══════════════════════════════════════════════════════════════════════


struct FBOnlineAgent[
    FNET: Module,
    BNET: Module,
    ANET: Module,
    OBS: Int,
    ACT: Int,
    D: Int,
    BATCH: Int,
    CAP: Int,
    LANES: Int,
    ZBUF: Int = 10_000,
](OffPolicyAgentGpu):
    """`OffPolicyAgentGpu` conformer around `FBTrainer[..., "gpu"]`.

    `LANES` is the env count the rollout state (`z` per lane) is sized for;
    the driver's `N_ENVS` must equal it, asserted at each entry point.
    """

    comptime AGENT_TRAIN_TARGET: StaticString = "gpu"
    comptime AGENT_OBS_DIM: Int = Self.OBS
    comptime AGENT_ACT_DIM: Int = Self.ACT
    comptime TrainerT = FBTrainer[
        Self.FNET, Self.BNET, Self.ANET,
        Self.OBS, Self.ACT, Self.D, Self.BATCH, "gpu",
    ]
    comptime A_IN: Int = Self.OBS + Self.D

    var t: Self.TrainerT
    var ctx: Optional[DeviceContext]
    var tracker: EpisodeTracker

    # ── replay ring (device only — no host mirror of CAP x D floats) ────
    var r_obs: Tensor
    var r_act: Tensor
    var r_nxt: Tensor
    var r_z: Tensor
    var r_term: Tensor
    var size: Int
    var pos: Int
    var size_dev: Optional[DeviceBuffer[DType.int32]]
    var samp_off: Optional[DeviceBuffer[DType.uint64]]
    var idx_s: Optional[DeviceBuffer[IDX_DT]]
    var idx_sp: Optional[DeviceBuffer[IDX_DT]]

    # ── training-side RNG scratch (device offsets: captured path) ───────
    var rng_dev: Optional[DeviceBuffer[DType.uint64]]
    var gauss: Tensor      # BATCH * D
    var pick: Tensor       # BATCH * 3

    # ── ZBuffer ─────────────────────────────────────────────────────────
    var zbuf: Tensor       # ZBUF * D
    var zbuf_head: Optional[DeviceBuffer[DType.int32]]
    var zbuf_fill: Optional[DeviceBuffer[DType.int32]]

    # ── rollout (eager) ─────────────────────────────────────────────────
    var z_lane: Tensor     # LANES * D
    var _gauss_lane: Tensor  # LANES * D
    var _pick_lane: Tensor   # LANES * 2
    var _ain: Tensor       # LANES * A_IN
    var _pi: Tensor        # LANES * ACT
    var _noise: Tensor     # LANES * ACT
    var _act_iter: Int
    var _roll_seed: UInt64
    var _roll_off: UInt64
    var _warmup_seed: UInt64
    var _warmup_off: UInt64

    # ── hyperparameters ─────────────────────────────────────────────────
    var learning_starts: Int
    var action_scale: Scalar[DT]
    var expl_std: Float64
    var z_hold: Int
    var zbuf_frac: Float64
    var keep_frac: Float64
    var uniform_frac: Float64
    var _train_seed: UInt64

    # ── bookkeeping ─────────────────────────────────────────────────────
    var _update_count: Int
    var _total_train_steps: Int
    var _mean_abs_action_dev: DeviceMeanAccum
    var _z_lane_resamples: Int

    def __init__(out self):
        self.t = Self.TrainerT()
        self.ctx = None
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](), window_size=0, idx=0,
            current_return=Scalar[DT](0.0), ep_count=0,
        )
        self.r_obs = Tensor()
        self.r_act = Tensor()
        self.r_nxt = Tensor()
        self.r_z = Tensor()
        self.r_term = Tensor()
        self.size = 0
        self.pos = 0
        self.size_dev = None
        self.samp_off = None
        self.idx_s = None
        self.idx_sp = None
        self.rng_dev = None
        self.gauss = Tensor()
        self.pick = Tensor()
        self.zbuf = Tensor()
        self.zbuf_head = None
        self.zbuf_fill = None
        self.z_lane = Tensor()
        self._gauss_lane = Tensor()
        self._pick_lane = Tensor()
        self._ain = Tensor()
        self._pi = Tensor()
        self._noise = Tensor()
        self._act_iter = 0
        self._roll_seed = UInt64(0xF0B0_0011)
        self._roll_off = UInt64(0)
        self._warmup_seed = UInt64(0xF0B0_0022)
        self._warmup_off = UInt64(0)
        self.learning_starts = 0
        self.action_scale = Scalar[DT](1.0)
        self.expl_std = 0.2
        self.z_hold = 150
        self.zbuf_frac = 0.5
        self.keep_frac = 0.2
        self.uniform_frac = 0.5
        self._train_seed = UInt64(0xF0B0_0033)
        self._update_count = 0
        self._total_train_steps = 0
        self._mean_abs_action_dev = DeviceMeanAccum()
        self._z_lane_resamples = 0

    @staticmethod
    def make[
        INIT: Initializer = Xavier
    ](
        ctx: DeviceContext,
        *,
        lr: Float64 = 3e-4,
        lr_b: Float64 = -1.0,
        gamma: Float64 = 0.98,
        tau: Float64 = 0.01,
        ortho_weight: Float64 = 1.0,
        max_grad_norm: Float64 = 1.0,
        bc_weight: Float64 = 0.0,
        learning_starts: Int = 10_000,
        action_scale: Float64 = 1.0,
        expl_std: Float64 = 0.2,
        z_hold: Int = 150,
        zbuf_frac: Float64 = 0.5,
        keep_frac: Float64 = 0.2,
        uniform_frac: Float64 = 0.5,
        window_size: Int = 100,
        initial_episode_fill: Float64 = 0.0,
        seed: UInt64 = UInt64(0x5EED_0B),
    ) raises -> Self:
        """Defaults are BFM-Zero's rollout / relabel settings on top of
        `FBTrainer.make`'s (`gamma` 0.98, `tau` 0.01, Adam 3e-4).

        `bc_weight = 0` deliberately: the offline BC term was the stand-in for
        CPR against extrapolation on a FROZEN dataset. Online, `F` is fitted on
        the policy's own actions, so the argmax corner is visited and
        corrected rather than extrapolated to. It stays a knob because that is
        a prediction, not a measurement — `mean|a|` at flush is the check.
        """
        comptime assert Self.CAP >= Self.BATCH, (
            "FBOnlineAgent: CAP must be >= BATCH"
        )
        comptime assert Self.ZBUF >= Self.BATCH, (
            "FBOnlineAgent: ZBUF must hold at least one training batch"
        )
        if learning_starts < Self.BATCH:
            raise Error(
                "FBOnlineAgent.make: learning_starts must be >= BATCH — the"
                " first captured step samples from the ring at that fill"
            )
        var a = Self()
        var octx = Optional[DeviceContext](ctx)
        a.ctx = octx
        a.t = Self.TrainerT.make[INIT](
            lr=lr, gamma=gamma, tau=tau, ortho_weight=ortho_weight,
            ctx=octx, seed=seed + 13, max_grad_norm=max_grad_norm,
            bc_weight=bc_weight, lr_b=lr_b,
        )
        a.t.ensure_sized()
        a.tracker = EpisodeTracker.new(
            window_size=window_size,
            initial_fill=Scalar[DT](initial_episode_fill),
        )
        a.learning_starts = learning_starts
        a.action_scale = Scalar[DT](action_scale)
        a.expl_std = expl_std
        a.z_hold = z_hold
        a.zbuf_frac = zbuf_frac
        a.keep_frac = keep_frac
        a.uniform_frac = uniform_frac
        a._train_seed = seed
        a._roll_seed = seed + 101
        a._warmup_seed = seed + 202

        # Ring: device only. A host mirror of CAP x (2·OBS + ACT + D + 1)
        # floats at CAP = 1 M would be ~700 MB of host RAM nothing reads.
        a.r_obs.ensure_gpu(ctx, Self.CAP * Self.OBS)
        a.r_act.ensure_gpu(ctx, Self.CAP * Self.ACT)
        a.r_nxt.ensure_gpu(ctx, Self.CAP * Self.OBS)
        a.r_z.ensure_gpu(ctx, Self.CAP * Self.D)
        a.r_term.ensure_gpu(ctx, Self.CAP)
        var sz = ctx.enqueue_create_buffer[DType.int32](1)
        sz.enqueue_fill(Int32(0))
        a.size_dev = sz^
        var so = ctx.enqueue_create_buffer[DType.uint64](1)
        so.enqueue_fill(UInt64(0))
        a.samp_off = so^
        a.idx_s = ctx.enqueue_create_buffer[IDX_DT](Self.BATCH)
        a.idx_sp = ctx.enqueue_create_buffer[IDX_DT](Self.BATCH)

        var ro = ctx.enqueue_create_buffer[DType.uint64](1)
        ro.enqueue_fill(UInt64(0))
        a.rng_dev = ro^
        ensure_t["gpu"](a.gauss, Self.BATCH * Self.D, octx)
        ensure_t["gpu"](a.pick, Self.BATCH * 3, octx)

        a.zbuf.ensure_gpu(ctx, Self.ZBUF * Self.D)
        var zh = ctx.enqueue_create_buffer[DType.int32](1)
        zh.enqueue_fill(Int32(0))
        a.zbuf_head = zh^
        var zf = ctx.enqueue_create_buffer[DType.int32](1)
        zf.enqueue_fill(Int32(0))
        a.zbuf_fill = zf^

        ensure_t["gpu"](a.z_lane, Self.LANES * Self.D, octx)
        ensure_t["gpu"](a._gauss_lane, Self.LANES * Self.D, octx)
        ensure_t["gpu"](a._pick_lane, Self.LANES * 2, octx)
        ensure_t["gpu"](a._ain, Self.LANES * Self.A_IN, octx)
        ensure_t["gpu"](a._pi, Self.LANES * Self.ACT, octx)
        ensure_t["gpu"](a._noise, Self.LANES * Self.ACT, octx)

        a._mean_abs_action_dev = DeviceMeanAccum.make["gpu"](ctx=octx)

        # Every lane gets a z before the first action — forced resample, then
        # the unconditional projection.
        a._resample_lanes(force=True)
        ctx.synchronize()
        return a^

    # ── rollout ──────────────────────────────────────────────────────────

    def _resample_lanes(mut self, force: Bool) raises:
        """Draw the per-lane mixture inputs and run the resample + projection.
        Eager path: host RNG offsets."""
        var c = self.ctx.value()
        comptime NG = Self.LANES * Self.D
        comptime NP = Self.LANES * 2
        box_muller_normal_gpu[NG](
            c, mptr(self._gauss_lane.dev.value().unsafe_ptr()),
            self._roll_seed, self._roll_off,
        )
        self._roll_off += UInt64(NG + (NG % 2))
        c.enqueue_function[uniform01_kernel[NP]](
            mptr(self._pick_lane.dev.value().unsafe_ptr()),
            self._roll_seed + 7, self._roll_off,
            grid_dim=_blocks(NP), block_dim=TPB,
        )
        self._roll_off += UInt64(2 * NP)
        c.enqueue_function[
            z_lane_resample_kernel[Self.D, Self.LANES, Self.ZBUF]
        ](
            mptr(self.z_lane.dev.value().unsafe_ptr()),
            mptr(self._gauss_lane.dev.value().unsafe_ptr()),
            mptr(self.zbuf.dev.value().unsafe_ptr()),
            LayoutTensor[DType.int32, Layout.row_major(1)](
                self.zbuf_fill.value()
            ),
            mptr(self._pick_lane.dev.value().unsafe_ptr()),
            Int32(self._act_iter),
            Int32(self.z_hold),
            Scalar[DT](self.zbuf_frac),
            Int32(1) if force else Int32(0),
            grid_dim=_blocks(Self.LANES), block_dim=TPB,
        )
        c.enqueue_function[project_sphere_kernel[Self.D, Self.LANES]](
            mptr(self.z_lane.dev.value().unsafe_ptr()),
            Scalar[DT](sqrt(Float64(Self.D))),
            grid_dim=_blocks(Self.LANES), block_dim=TPB,
        )
        self._z_lane_resamples += 1

    def _policy_into(
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        std: Float64,
    ) raises:
        """`action = clamp(pi_z(obs, z_lane) + std·n) · scale` for all lanes."""
        var c = self.ctx.value()
        comptime NA = Self.LANES * Self.ACT
        c.enqueue_function[pack2_kernel[Self.OBS, Self.D, Self.LANES]](
            obs_ptr,
            mptr(self.z_lane.dev.value().unsafe_ptr()),
            mptr(self._ain.dev.value().unsafe_ptr()),
            grid_dim=_blocks(Self.LANES * Self.A_IN), block_dim=TPB,
        )
        call_forward["gpu", Self.LANES](
            self.t.actor.online, TensorRefs[1, MutAnyOrigin](self._ain),
            self._pi, self.ctx,
        )
        if std > 0.0:
            box_muller_normal_gpu[NA](
                c, mptr(self._noise.dev.value().unsafe_ptr()),
                self._roll_seed + 3, self._roll_off,
            )
            self._roll_off += UInt64(NA + (NA % 2))
        c.enqueue_function[explore_action_kernel[NA]](
            action_ptr,
            mptr(self._pi.dev.value().unsafe_ptr()),
            mptr(self._noise.dev.value().unsafe_ptr()),
            Scalar[DT](std), self.action_scale,
            grid_dim=_blocks(NA), block_dim=TPB,
        )

    def select_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_OBS_DIM), MutAnyOrigin
        ],
        action: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        ao_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, 2 * Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        alp_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM + 1), MutAnyOrigin
        ],
        step_idx: Int,
    ) raises:
        comptime assert N_ENVS == Self.LANES, (
            "FBOnlineAgent: the driver's N_ENVS must equal the agent's LANES"
        )
        _ = ao_scratch
        _ = alp_scratch
        # The lane z advances on EVERY iteration, warmup included, so the
        # stored z under random actions is a real draw and not a constant.
        self._resample_lanes(force=False)
        self._act_iter += 1
        if step_idx < self.learning_starts:
            warmup_uniform_batched["gpu", N_ENVS, Self.ACT](
                action, self.action_scale, self.ctx,
                self._warmup_seed, self._warmup_off,
            )
            return
        self._policy_into(obs.ptr, action.ptr, self.expl_std)

    def select_greedy_action_batched[
        N_ENVS: Int,
    ](
        mut self,
        ctx: Optional[DeviceContext],
        obs: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_OBS_DIM), MutAnyOrigin
        ],
        action: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        ao_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, 2 * Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
    ) raises:
        """Greedy under each lane's CURRENT `z` — no noise, no resample.
        ⚠ The env reward this scores is the env's own task; FB is not trained
        on it. Useful as a coverage signal, not as a zero-shot number — that
        needs `z_from_reward`, see `examples/fb/fb_eval_walker_online.mojo`."""
        comptime assert N_ENVS == Self.LANES, (
            "FBOnlineAgent: the driver's N_ENVS must equal the agent's LANES"
        )
        _ = ctx
        _ = ao_scratch
        self._policy_into(obs.ptr, action.ptr, 0.0)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Single-row greedy action under lane 0's `z`. Host-list surface for
        `run_offpolicy_eval`; allocates per call — never on a hot path."""
        var c = self.ctx.value()
        var zl = Tensor.alloc(Self.LANES * Self.D)
        zl.ensure_gpu(c, Self.LANES * Self.D)
        c.enqueue_copy(zl.dev.value(), self.z_lane.dev.value())
        zl.download(c)
        var x = Tensor.alloc(Self.A_IN)
        for k in range(Self.OBS):
            x.data[k] = obs[k]
        for k in range(Self.D):
            x.data[Self.OBS + k] = zl.data[k]
        x.upload(c)
        var y = Tensor.alloc_gpu(c, Self.ACT)
        call_forward["gpu", 1](
            self.t.actor.online, TensorRefs[1, MutAnyOrigin](x), y, self.ctx
        )
        y.download(c)
        for j in range(Self.ACT):
            var v = y.data[j]
            if v > Scalar[DT](1.0):
                v = Scalar[DT](1.0)
            elif v < Scalar[DT](-1.0):
                v = Scalar[DT](-1.0)
            action_out[j] = v * self.action_scale

    # ── record ───────────────────────────────────────────────────────────

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        raise Error(
            "FBOnlineAgent.record: single-env host path not supported — this"
            " agent is driven by run_offpolicy_train_batched on a GPU env"
        )

    def _replay_add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        raise Error("FBOnlineAgent._replay_add: host path not supported")

    def _tracker_ptr(self) -> Pointer[EpisodeTracker, MutAnyOrigin]:
        return rebind[Pointer[EpisodeTracker, MutAnyOrigin]](
            Pointer(to=self.tracker)
        )

    def record_batch_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """Append the lanes' transitions, each with the `z` its action was
        taken under. `reward` is NOT stored: FB never reads it."""
        comptime assert N_ENVS == Self.LANES, (
            "FBOnlineAgent: the driver's N_ENVS must equal the agent's LANES"
        )
        _ = reward_dev
        comptime W = Self.OBS + Self.ACT + Self.OBS + Self.D + 1
        ctx.enqueue_function[
            ring_store_kernel[Self.OBS, Self.ACT, Self.D, Self.CAP, Self.LANES]
        ](
            mptr(prev_obs_dev.unsafe_ptr()),
            mptr(action_dev.unsafe_ptr()),
            mptr(obs_dev.unsafe_ptr()),
            mptr(done_dev.unsafe_ptr()),
            mptr(self.z_lane.dev.value().unsafe_ptr()),
            mptr(self.r_obs.dev.value().unsafe_ptr()),
            mptr(self.r_act.dev.value().unsafe_ptr()),
            mptr(self.r_nxt.dev.value().unsafe_ptr()),
            mptr(self.r_term.dev.value().unsafe_ptr()),
            mptr(self.r_z.dev.value().unsafe_ptr()),
            Int32(self.pos),
            grid_dim=_blocks(Self.LANES * W), block_dim=TPB,
        )
        self.pos = (self.pos + Self.LANES) % Self.CAP
        self.size = self.size + Self.LANES
        if self.size > Self.CAP:
            self.size = Self.CAP
        # Device mirror of the fill — what the captured index draw reads.
        ctx.enqueue_function[_set_size_kernel](
            LayoutTensor[DType.int32, Layout.row_major(1)](
                self.size_dev.value()
            ),
            Int32(self.size),
            grid_dim=1, block_dim=1,
        )

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS, Self.AGENT_OBS_DIM, Self.AGENT_ACT_DIM, N_ENVS,
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        raise Error(
            "FBOnlineAgent: n-step is meaningless for FB (no reward in the"
            " loss); run the driver with NS=1"
        )

    # ── the training step ────────────────────────────────────────────────

    def _sample_batch(mut self) raises:
        """Two INDEPENDENT uniform draws over the ring's fill, then the
        gathers into `FBTrainer`'s owned batch. All device; capture-safe."""
        var c = self.ctx.value()
        var size_lt = LayoutTensor[DType.int32, Layout.row_major(1)](
            self.size_dev.value()
        )
        var off_lt = LayoutTensor[DType.uint64, Layout.row_major(1)](
            self.samp_off.value()
        )
        var is_lt = LayoutTensor[IDX_DT, Layout.row_major(Self.BATCH)](
            self.idx_s.value()
        )
        var isp_lt = LayoutTensor[IDX_DT, Layout.row_major(Self.BATCH)](
            self.idx_sp.value()
        )
        comptime nb = _blocks(Self.BATCH)
        c.enqueue_function[_uniform_indices_dev_kernel[Self.BATCH]](
            is_lt, size_lt, self._train_seed, off_lt,
            grid_dim=nb, block_dim=TPB,
        )
        c.enqueue_function[_incr_offset_kernel[Self.BATCH]](
            off_lt, grid_dim=1, block_dim=1,
        )
        c.enqueue_function[_uniform_indices_dev_kernel[Self.BATCH]](
            isp_lt, size_lt, self._train_seed, off_lt,
            grid_dim=nb, block_dim=TPB,
        )
        c.enqueue_function[_incr_offset_kernel[Self.BATCH]](
            off_lt, grid_dim=1, block_dim=1,
        )

        var ip_s = mptr(self.idx_s.value().unsafe_ptr())
        var ip_sp = mptr(self.idx_sp.value().unsafe_ptr())
        c.enqueue_function[gather_rows_kernel[Self.OBS, Self.BATCH]](
            mptr(self.r_obs.dev.value().unsafe_ptr()), ip_s,
            mptr(self.t.bs.dev.value().unsafe_ptr()),
            grid_dim=_blocks(Self.BATCH * Self.OBS), block_dim=TPB,
        )
        c.enqueue_function[gather_rows_kernel[Self.ACT, Self.BATCH]](
            mptr(self.r_act.dev.value().unsafe_ptr()), ip_s,
            mptr(self.t.ba.dev.value().unsafe_ptr()),
            grid_dim=_blocks(Self.BATCH * Self.ACT), block_dim=TPB,
        )
        c.enqueue_function[gather_rows_kernel[Self.OBS, Self.BATCH]](
            mptr(self.r_nxt.dev.value().unsafe_ptr()), ip_s,
            mptr(self.t.bsn.dev.value().unsafe_ptr()),
            grid_dim=_blocks(Self.BATCH * Self.OBS), block_dim=TPB,
        )
        c.enqueue_function[gather_rows_kernel[Self.D, Self.BATCH]](
            mptr(self.r_z.dev.value().unsafe_ptr()), ip_s,
            mptr(self.t.bz.dev.value().unsafe_ptr()),
            grid_dim=_blocks(Self.BATCH * Self.D), block_dim=TPB,
        )
        c.enqueue_function[gather_rows_kernel[Self.OBS, Self.BATCH]](
            mptr(self.r_obs.dev.value().unsafe_ptr()), ip_sp,
            mptr(self.t.bsp.dev.value().unsafe_ptr()),
            grid_dim=_blocks(Self.BATCH * Self.OBS), block_dim=TPB,
        )

    def _relabel_z(mut self) raises:
        """`bz` holds the stored z; keep `keep_frac` of rows, overwrite the
        rest from the mixture over `B(bsp)`; project every row."""
        var c = self.ctx.value()
        self.t.embed_sp()
        gaussian_dev_t["gpu", Self.BATCH * Self.D](
            self.gauss, self._train_seed + 5, self.rng_dev.value(), self.ctx
        )
        var off_lt = LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin](
            mptr(self.rng_dev.value().unsafe_ptr())
        )
        comptime NP = Self.BATCH * 3
        c.enqueue_function[uniform01_dev_kernel[NP]](
            mptr(self.pick.dev.value().unsafe_ptr()),
            self._train_seed + 9, off_lt,
            grid_dim=_blocks(NP), block_dim=TPB,
        )
        c.enqueue_function[advance_rng_offset_kernel[2 * NP]](
            off_lt, grid_dim=1, block_dim=1
        )
        c.enqueue_function[z_relabel_kernel[Self.D, Self.BATCH]](
            mptr(self.t.bz.dev.value().unsafe_ptr()),
            mptr(self.gauss.dev.value().unsafe_ptr()),
            mptr(self.t.b_sp.dev.value().unsafe_ptr()),
            mptr(self.pick.dev.value().unsafe_ptr()),
            Scalar[DT](self.keep_frac), Scalar[DT](self.uniform_frac),
            grid_dim=_blocks(Self.BATCH), block_dim=TPB,
        )
        c.enqueue_function[project_sphere_kernel[Self.D, Self.BATCH]](
            mptr(self.t.bz.dev.value().unsafe_ptr()),
            Scalar[DT](sqrt(Float64(Self.D))),
            grid_dim=_blocks(Self.BATCH), block_dim=TPB,
        )

    def _push_zbuf(mut self) raises:
        var c = self.ctx.value()
        var head = LayoutTensor[DType.int32, Layout.row_major(1)](
            self.zbuf_head.value()
        )
        var fill = LayoutTensor[DType.int32, Layout.row_major(1)](
            self.zbuf_fill.value()
        )
        c.enqueue_function[zbuf_push_kernel[Self.D, Self.BATCH, Self.ZBUF]](
            mptr(self.zbuf.dev.value().unsafe_ptr()),
            mptr(self.t.bz.dev.value().unsafe_ptr()),
            head,
            grid_dim=_blocks(Self.BATCH * Self.D), block_dim=TPB,
        )
        c.enqueue_function[zbuf_advance_kernel[Self.BATCH, Self.ZBUF]](
            head, fill, grid_dim=1, block_dim=1,
        )

    def _train_kernels(mut self) raises:
        """The pure device sequence: sample → relabel → FB step → ZBuffer."""
        self._sample_batch()
        self._relabel_z()
        _ = self.t.train_step(want_loss=False)
        self._push_zbuf()
        comptime lba = Layout.row_major(Self.BATCH * Self.ACT)
        self._mean_abs_action_dev.accumulate_gpu_abs_lt[Self.BATCH * Self.ACT](
            self.t.ba.lt["gpu", lba]()
        )

    def train_step(mut self, step_idx: Int) raises -> Bool:
        if step_idx < self.learning_starts or self.size < Self.BATCH:
            return False
        self._train_kernels()
        self._update_count += 1
        self._total_train_steps += 1
        return True

    def train_device_kernels(mut self) raises:
        """Capture body. The driver enters only past `learning_starts_count`,
        and `make` requires `learning_starts >= BATCH`, so the ring is never
        sampled below one batch."""
        self._train_kernels()

    def note_train_update(mut self):
        self._update_count += 1
        self._total_train_steps += 1

    def learning_starts_count(self) -> Int:
        return self.learning_starts

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    # ── diagnostics ──────────────────────────────────────────────────────

    def peek_losses(
        mut self,
        mut measure: Float64, mut ortho: Float64, mut actor: Float64,
        mut f_norm: Float64, mut b_norm: Float64,
    ) raises:
        """The last step's losses, RECOMPUTED from the trainer's live buffers.

        `train_step(want_loss=False)` skips the reductions and the D2H; the
        activations it leaves behind (`f1o`, `b_sp`, `b_sn`, `m_target`,
        `b_s`, `acc_lam`) are the pre-update ones of that step, so re-running
        the two loss reductions over them returns exactly what
        `want_loss=True` would have — without ever putting a sync inside the
        captured sequence. The gradient outputs go to the scratch the next
        step overwrites. D2Hs several buffers: FLUSH CADENCE ONLY.
        """
        measure = 0.0
        ortho = 0.0
        actor = 0.0
        f_norm = 0.0
        b_norm = 0.0
        if self.t.steps == 0:
            return
        var c = self.ctx.value()
        var l1 = fb_measure_loss_into["gpu", Self.D, Self.BATCH](
            self.t.ws1, self.t.f1o, self.t.b_sp, self.t.b_sn, self.t.m_target,
            self.t.g_f1, self.t.g_bsp1, self.t.g_bsn1, True, self.ctx,
        )
        var l2 = fb_measure_loss_into["gpu", Self.D, Self.BATCH](
            self.t.ws2, self.t.f2o, self.t.b_sp, self.t.b_sn, self.t.m_target,
            self.t.g_f2, self.t.g_bsp2, self.t.g_bsn2, True, self.ctx,
        )
        measure = 0.5 * (l1 + l2)
        ortho = fb_ortho_loss_into["gpu", Self.D, Self.BATCH](
            self.t.wso, self.t.b_s, self.t.b_sp, self.t.g_bs_o, self.t.g_bsp_o,
            True, self.ctx,
        )
        self.t.acc_lam.download(c)
        actor = -Float64(self.t.acc_lam.data[0])
        var fn2 = mean_sq_t["gpu", Self.BATCH * Self.D](
            self.t.f1o, self.t.acc, self.ctx
        )
        var bn2 = mean_sq_t["gpu", Self.BATCH * Self.D](
            self.t.b_s, self.t.acc, self.ctx
        )
        f_norm = sqrt(fn2 * Float64(Self.D))
        b_norm = sqrt(bn2 * Float64(Self.D))

    def flush_metrics_through_logger[
        L: Logger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        var measure = Float64(0)
        var ortho = Float64(0)
        var actor = Float64(0)
        var fnorm = Float64(0)
        var bnorm = Float64(0)
        self.peek_losses(measure, ortho, actor, fnorm, bnorm)
        var gf1 = Float64(0)
        var gf2 = Float64(0)
        var gb = Float64(0)
        self.t.read_grad_norms(gf1, gf2, gb)
        var maa = Float64(self._mean_abs_action_dev.read["gpu"]())
        self._mean_abs_action_dev.reset["gpu"]()
        var n_upd = self._update_count
        self._update_count = 0
        if Bool(logger):
            var names = List[String]()
            var vals = List[Float64]()
            names.append(String("fb/measure")); vals.append(measure)
            names.append(String("fb/ortho")); vals.append(ortho)
            names.append(String("fb/actor")); vals.append(actor)
            names.append(String("fb/f_norm")); vals.append(fnorm)
            names.append(String("fb/b_norm")); vals.append(bnorm)
            names.append(String("fb/b_norm_deficit"))
            vals.append(sqrt(Float64(Self.D)) - bnorm)
            names.append(String("fb/ortho_Q"))
            vals.append(ortho + 2.0 * bnorm * bnorm)
            names.append(String("fb/grad_norm_f1")); vals.append(gf1)
            names.append(String("fb/grad_norm_f2")); vals.append(gf2)
            names.append(String("fb/grad_norm_b")); vals.append(gb)
            names.append(String("fb/mean_abs_action")); vals.append(maa)
            names.append(String("fb/replay_size")); vals.append(Float64(self.size))
            names.append(String("fb/train_steps"))
            vals.append(Float64(self._total_train_steps))
            names.append(String("fb/updates_since_flush"))
            vals.append(Float64(n_upd))
            logger.value()[].log_scalars(names, vals, step)
        print(
            "   [fb] step", step, " measure", measure, " ortho", ortho,
            " actor", actor, " |F|", fnorm, " |B|", bnorm, " gF", gf1,
            " mean|a|", maa, " replay", self.size,
        )

    # ── checkpoint ───────────────────────────────────────────────────────

    def save_state(mut self, path: String) raises:
        """`FBTrainer`'s own format — the offline eval scripts load it."""
        self.t.save_state(path)

    def load_state(mut self, path: String) raises:
        self.t.load_state(path)
