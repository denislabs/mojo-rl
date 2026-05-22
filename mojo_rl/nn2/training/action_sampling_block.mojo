"""ActionSamplingBlock — env-interaction policy head (Block E-2).

Encapsulates the warmup-vs-policy `select_action` flow used by every
off-policy continuous-control trainer:

  if step_idx < learning_starts:
      # Uniform exploration on [-action_scale, +action_scale]^ACT_DIM
  else:
      # actor.forward(obs) → optional sampler.forward(.) → clamp to ±scale

The block owns all single-step scratch (obs / actor-out / sampler-out)
on both CPU and GPU, so trainers stop carrying these as parallel fields.

Three call shapes, picked by which the trainer's policy needs:

  block.select_stochastic[target, ACTOR, SAMPLER](
      actor, sampler, obs_ptr, action_out_ptr,
      step_idx, learning_starts, action_scale,
  )
      SAC pattern. `actor` emits packed `[mu | log_std]`; `sampler`
      (typically `RSample`) maps that to packed `[action | log_prob]`.
      The block writes the first ACT_DIM entries of `_sampler_out`
      (clamped) into `action_out_ptr`.

  block.select_deterministic[target, ACTOR](
      actor, obs_ptr, action_out_ptr,
      step_idx, learning_starts, action_scale,
  )
      DDPG-style deterministic policy. `actor.OUT_DIM` must equal
      `ACT_DIM`. Action read directly from actor output, clamped.

  block.select_deterministic_with_noise[target, ACTOR](
      actor, obs_ptr, action_out_ptr,
      step_idx, learning_starts, action_scale,
      noise_scale,
  )
      DDPG / TD3 pattern: deterministic actor + zero-mean Gaussian noise
      with stddev `noise_scale * action_scale` per element. Noise is
      drawn on CPU via `random_float64()` and added before the clamp.

Scratch sizing: caller provides `SAMPLER_OUT_DIM` at type level. SAC
uses `ACT_DIM + 1` (action + log_prob); deterministic agents pass
`ACT_DIM` and ignore the sampler-out buffer.

Target dispatch is uniform: CPU/GPU branches inside the method bodies,
mirroring `SACTrainer.select_action`'s original structure. GPU path
uploads obs once, runs actor (+ sampler) on device, downloads the
sampler-out buffer to host, then clamps + writes the action on host.
"""

from std.memory import alloc
from std.random import random_float64
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module
from ..core.target_storage import TargetStorage, assert_tag_for
from ..random.box_muller import box_muller_normal


struct ActionSamplingBlock[
    ACTOR: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    SAMPLER_OUT_DIM: Int,
](Movable & ImplicitlyDestructible):
    comptime ACTOR_OUT_DIM = Self.ACTOR.OUT_DIM

    # ─── CPU scratch ──────────────────────────────────────────────────
    var _ob1: UnsafePointer[Scalar[DT], MutAnyOrigin]             # [OBS_DIM]
    var _actor_out: UnsafePointer[Scalar[DT], MutAnyOrigin]       # [ACTOR_OUT_DIM]
    var _sampler_out: UnsafePointer[Scalar[DT], MutAnyOrigin]     # [SAMPLER_OUT_DIM]
    var _noise: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [ACT_DIM] (DDPG/TD3 only)

    # ─── GPU scratch (None on CPU bundles) ────────────────────────────
    var _ob1_dev: Optional[DeviceBuffer[DT]]
    var _actor_out_dev: Optional[DeviceBuffer[DT]]
    var _sampler_out_dev: Optional[DeviceBuffer[DT]]

    var ts: TargetStorage

    # ─── Construction ─────────────────────────────────────────────────

    def __init__(out self):
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)
        self._ob1 = null_p
        self._actor_out = null_p
        self._sampler_out = null_p
        self._noise = null_p
        self._ob1_dev = None
        self._actor_out_dev = None
        self._sampler_out_dev = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", (
            "ActionSamplingBlock.make[target='gpu'] requires a DeviceContext"
        )
        var b = Self()
        b._ob1 = alloc[Scalar[DT]](Self.OBS_DIM)
        b._actor_out = alloc[Scalar[DT]](Self.ACTOR_OUT_DIM)
        b._sampler_out = alloc[Scalar[DT]](Self.SAMPLER_OUT_DIM)
        b._noise = alloc[Scalar[DT]](Self.ACT_DIM)
        b.ts = TargetStorage.make_cpu()
        return b^

    @staticmethod
    def make[target: StaticString](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "ActionSamplingBlock.make[target='cpu'](ctx) — drop ctx for CPU"
        )
        var b = Self()
        # Keep CPU scratch alongside GPU: obs lands here before upload,
        # sampler-out lands here after download.
        b._ob1 = alloc[Scalar[DT]](Self.OBS_DIM)
        b._actor_out = alloc[Scalar[DT]](Self.ACTOR_OUT_DIM)
        b._sampler_out = alloc[Scalar[DT]](Self.SAMPLER_OUT_DIM)
        b._noise = alloc[Scalar[DT]](Self.ACT_DIM)
        b._ob1_dev = ctx.enqueue_create_buffer[DT](Self.OBS_DIM)
        b._actor_out_dev = ctx.enqueue_create_buffer[DT](Self.ACTOR_OUT_DIM)
        b._sampler_out_dev = ctx.enqueue_create_buffer[DT](Self.SAMPLER_OUT_DIM)
        b.ts = TargetStorage.make_gpu(ctx)
        return b^

    def __del__(deinit self):
        if Int(self._ob1) != 0:
            self._ob1.free()
        if Int(self._actor_out) != 0:
            self._actor_out.free()
        if Int(self._sampler_out) != 0:
            self._sampler_out.free()
        if Int(self._noise) != 0:
            self._noise.free()

    # ─── Warmup uniform sampling ──────────────────────────────────────

    def _write_warmup(
        mut self,
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_scale: Scalar[DT],
    ):
        for j in range(Self.ACT_DIM):
            var u = Scalar[DT](2.0 * random_float64() - 1.0)
            action_out[j] = u * action_scale

    # ─── Clamp helper ─────────────────────────────────────────────────

    def _clamp_into(
        self,
        src: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_scale: Scalar[DT],
    ):
        for j in range(Self.ACT_DIM):
            var a = src[j]
            if a > action_scale:
                a = action_scale
            elif a < -action_scale:
                a = -action_scale
            action_out[j] = a

    # ─── Stochastic (SAC) ─────────────────────────────────────────────

    def select_stochastic[
        target: StaticString,
        SAMPLER: Module,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut sampler: SAMPLER,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
        learning_starts: Int,
        action_scale: Scalar[DT],
    ) raises:
        """SAC-style: warmup vs (actor → sampler → clamp first ACT_DIM).

        `sampler` is expected to take `[ACTOR_OUT_DIM]` and emit
        `[SAMPLER_OUT_DIM]` where the first ACT_DIM entries are the
        action samples. Log-prob (if any) is in the trailing entries
        and ignored by this method."""
        assert_tag_for["ActionSamplingBlock", target](self.ts.target_tag)
        if step_idx < learning_starts:
            self._write_warmup(action_out, action_scale)
            return

        comptime if target == "cpu":
            for d in range(Self.OBS_DIM):
                self._ob1[d] = obs[d]
            var ob1_t = TileTensor(self._ob1, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(self._actor_out, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["cpu", 1](ob1_t, ao_t)
            var sp_t = TileTensor(self._sampler_out, row_major[1, Self.SAMPLER_OUT_DIM]())
            sampler.forward["cpu", 1](ao_t, sp_t)
            self._clamp_into(self._sampler_out, action_out, action_scale)
        else:
            var ctx = self.ts.ctx.value()
            for d in range(Self.OBS_DIM):
                self._ob1[d] = obs[d]
            var ob1_dev = self._ob1_dev.value()
            ctx.enqueue_copy(ob1_dev, self._ob1)
            var ob1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                ob1_dev.unsafe_ptr()
            )
            var ao_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._actor_out_dev.value().unsafe_ptr()
            )
            var sp_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._sampler_out_dev.value().unsafe_ptr()
            )
            var ob1_t = TileTensor(ob1_p, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(ao_p, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["gpu", 1](ob1_t, ao_t)
            var sp_t = TileTensor(sp_p, row_major[1, Self.SAMPLER_OUT_DIM]())
            sampler.forward["gpu", 1](ao_t, sp_t)
            ctx.enqueue_copy(self._sampler_out, self._sampler_out_dev.value())
            ctx.synchronize()
            self._clamp_into(self._sampler_out, action_out, action_scale)

    # ─── Deterministic (DDPG no-noise / TD3 eval) ─────────────────────

    def select_deterministic[
        target: StaticString,
    ](
        mut self,
        mut actor: Self.ACTOR,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
        learning_starts: Int,
        action_scale: Scalar[DT],
    ) raises:
        """Deterministic policy. `ACTOR.OUT_DIM == ACT_DIM` enforced."""
        comptime assert Self.ACTOR_OUT_DIM == Self.ACT_DIM, (
            "select_deterministic requires ACTOR.OUT_DIM == ACT_DIM"
        )
        assert_tag_for["ActionSamplingBlock", target](self.ts.target_tag)
        if step_idx < learning_starts:
            self._write_warmup(action_out, action_scale)
            return

        comptime if target == "cpu":
            for d in range(Self.OBS_DIM):
                self._ob1[d] = obs[d]
            var ob1_t = TileTensor(self._ob1, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(self._actor_out, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["cpu", 1](ob1_t, ao_t)
            self._clamp_into(self._actor_out, action_out, action_scale)
        else:
            var ctx = self.ts.ctx.value()
            for d in range(Self.OBS_DIM):
                self._ob1[d] = obs[d]
            var ob1_dev = self._ob1_dev.value()
            ctx.enqueue_copy(ob1_dev, self._ob1)
            var ob1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                ob1_dev.unsafe_ptr()
            )
            var ao_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._actor_out_dev.value().unsafe_ptr()
            )
            var ob1_t = TileTensor(ob1_p, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(ao_p, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["gpu", 1](ob1_t, ao_t)
            ctx.enqueue_copy(self._actor_out, self._actor_out_dev.value())
            ctx.synchronize()
            self._clamp_into(self._actor_out, action_out, action_scale)

    # ─── Deterministic + Gaussian exploration noise (DDPG/TD3 acting) ─

    def select_deterministic_with_noise[
        target: StaticString,
    ](
        mut self,
        mut actor: Self.ACTOR,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
        learning_starts: Int,
        action_scale: Scalar[DT],
        noise_scale: Scalar[DT],
    ) raises:
        """DDPG/TD3-style: actor(obs) + ε, ε~N(0, (noise_scale*action_scale)^2)
        per element, clamp to ±action_scale.

        Warmup branch uses uniform random (same as select_deterministic).
        Noise drawn on CPU via box_muller for determinism — GPU path
        still uploads obs, runs actor on device, downloads, then adds
        noise on host. The host-side noise injection keeps RNG state
        identical to CPU runs at fixed seed."""
        comptime assert Self.ACTOR_OUT_DIM == Self.ACT_DIM, (
            "select_deterministic_with_noise requires ACTOR.OUT_DIM == ACT_DIM"
        )
        assert_tag_for["ActionSamplingBlock", target](self.ts.target_tag)
        if step_idx < learning_starts:
            self._write_warmup(action_out, action_scale)
            return

        # Forward into _actor_out (CPU buffer, possibly via GPU detour).
        comptime if target == "cpu":
            for d in range(Self.OBS_DIM):
                self._ob1[d] = obs[d]
            var ob1_t = TileTensor(self._ob1, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(self._actor_out, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["cpu", 1](ob1_t, ao_t)
        else:
            var ctx = self.ts.ctx.value()
            for d in range(Self.OBS_DIM):
                self._ob1[d] = obs[d]
            var ob1_dev = self._ob1_dev.value()
            ctx.enqueue_copy(ob1_dev, self._ob1)
            var ob1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                ob1_dev.unsafe_ptr()
            )
            var ao_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._actor_out_dev.value().unsafe_ptr()
            )
            var ob1_t = TileTensor(ob1_p, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(ao_p, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["gpu", 1](ob1_t, ao_t)
            ctx.enqueue_copy(self._actor_out, self._actor_out_dev.value())
            ctx.synchronize()

        # CPU-side noise sample + add + clamp.
        var sigma = noise_scale * action_scale
        box_muller_normal(self._noise, Self.ACT_DIM)
        for j in range(Self.ACT_DIM):
            var a = self._actor_out[j] + self._noise[j] * sigma
            if a > action_scale:
                a = action_scale
            elif a < -action_scale:
                a = -action_scale
            action_out[j] = a
