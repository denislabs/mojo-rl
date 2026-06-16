"""ActionSamplingBlock — env-interaction policy head (Block E-2).

Phase 2.5 migration: pre-Phase-2.5 the block declared 4 raw
`UnsafePointer` (CPU scratches) + 3 `Optional[DeviceBuffer]` (GPU
staging) + manual `alloc/free` in `make[cpu]` / `make[gpu]` + `__del__`
deallocator. Post-migration, every scratch is a `Scratch[NAME, SIZE,
STAGING]` field; staging scratches use the `STAGING=True` flag so
`init_with["gpu"]` allocates BOTH the device buffer AND a CPU mirror
in one walker call.

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

from std.random import random_float64
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core import Module
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn.random.box_muller import box_muller_normal


struct ActionSamplingBlock[
    ACTOR: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    SAMPLER_OUT_DIM: Int,
](Movable & ImplicitlyDestructible):
    comptime ACTOR_OUT_DIM = Self.ACTOR.OUT_DIM

    # Staging scratches (CPU mirror always allocated; dev allocated on
    # GPU runs). Used as upload/download buffers + on-host postprocessing.
    var _ob1: Scratch["ob1", Self.OBS_DIM, True]
    var _actor_out: Scratch["actor_out", Self.ACTOR_OUT_DIM, True]
    var _sampler_out: Scratch["sampler_out", Self.SAMPLER_OUT_DIM, True]

    # CPU-only scratch (used identically on both targets — noise sampled
    # host-side for determinism).
    var _noise: Scratch["noise", Self.ACT_DIM, True]

    var ts: TargetStorage

    def __init__(out self):
        self._ob1 = Scratch["ob1", Self.OBS_DIM, True]()
        self._actor_out = Scratch["actor_out", Self.ACTOR_OUT_DIM, True]()
        self._sampler_out = Scratch["sampler_out", Self.SAMPLER_OUT_DIM, True]()
        self._noise = Scratch["noise", Self.ACT_DIM, True]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "ActionSamplingBlock: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "ActionSamplingBlock.make[target='gpu']: ctx required"
                )
        var b = Self()
        b.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](b, ctx)
        return b^

    # ─── Warmup uniform sampling ──────────────────────────────────────

    def _write_warmup(
        mut self,
        mut action_out: List[Scalar[DT]],
        action_scale: Scalar[DT],
    ):
        for j in range(Self.ACT_DIM):
            var u = Scalar[DT](2.0 * random_float64() - 1.0)
            action_out[j] = u * action_scale

    # ─── Clamp helper ─────────────────────────────────────────────────

    def _clamp_into(
        self,
        src: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mut action_out: List[Scalar[DT]],
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
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
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

        var ob1_cpu_p = self._ob1.cpu_ptr()
        var actor_out_cpu_p = self._actor_out.cpu_ptr()
        var sampler_out_cpu_p = self._sampler_out.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]

        comptime if target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(actor_out_cpu_p, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["cpu", 1](ob1_t, output=ao_t)
            var sp_t = TileTensor(sampler_out_cpu_p, row_major[1, Self.SAMPLER_OUT_DIM]())
            sampler.forward["cpu", 1](ao_t, output=sp_t)
            self._clamp_into(sampler_out_cpu_p, action_out, action_scale)
        else:
            var ctx = self.ts.ctx.value()
            var ob1_dev_p = self._ob1.dev_ptr()
            var ao_dev_p = self._actor_out.dev_ptr()
            var sp_dev_p = self._sampler_out.dev_ptr()
            # Upload obs through the staging cpu buffer.
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
            var ob1_t = TileTensor(ob1_dev_p, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(ao_dev_p, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["gpu", 1](ob1_t, output=ao_t)
            var sp_t = TileTensor(sp_dev_p, row_major[1, Self.SAMPLER_OUT_DIM]())
            sampler.forward["gpu", 1](ao_t, output=sp_t)
            ctx.enqueue_copy(sampler_out_cpu_p, self._sampler_out.dev.value())
            ctx.synchronize()
            self._clamp_into(sampler_out_cpu_p, action_out, action_scale)

    # ─── Deterministic (DDPG no-noise / TD3 eval) ─────────────────────

    def select_deterministic[
        target: StaticString,
    ](
        mut self,
        mut actor: Self.ACTOR,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
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

        var ob1_cpu_p = self._ob1.cpu_ptr()
        var actor_out_cpu_p = self._actor_out.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]

        comptime if target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(actor_out_cpu_p, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["cpu", 1](ob1_t, output=ao_t)
            self._clamp_into(actor_out_cpu_p, action_out, action_scale)
        else:
            var ctx = self.ts.ctx.value()
            var ob1_dev_p = self._ob1.dev_ptr()
            var ao_dev_p = self._actor_out.dev_ptr()
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
            var ob1_t = TileTensor(ob1_dev_p, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(ao_dev_p, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["gpu", 1](ob1_t, output=ao_t)
            ctx.enqueue_copy(actor_out_cpu_p, self._actor_out.dev.value())
            ctx.synchronize()
            self._clamp_into(actor_out_cpu_p, action_out, action_scale)

    # ─── Deterministic + Gaussian exploration noise (DDPG/TD3 acting) ─

    def select_deterministic_with_noise[
        target: StaticString,
    ](
        mut self,
        mut actor: Self.ACTOR,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
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

        var ob1_cpu_p = self._ob1.cpu_ptr()
        var actor_out_cpu_p = self._actor_out.cpu_ptr()
        var noise_cpu_p = self._noise.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]

        # Forward into actor_out_cpu (possibly via GPU detour).
        comptime if target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(actor_out_cpu_p, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["cpu", 1](ob1_t, output=ao_t)
        else:
            var ctx = self.ts.ctx.value()
            var ob1_dev_p = self._ob1.dev_ptr()
            var ao_dev_p = self._actor_out.dev_ptr()
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
            var ob1_t = TileTensor(ob1_dev_p, row_major[1, Self.OBS_DIM]())
            var ao_t = TileTensor(ao_dev_p, row_major[1, Self.ACTOR_OUT_DIM]())
            actor.forward["gpu", 1](ob1_t, output=ao_t)
            ctx.enqueue_copy(actor_out_cpu_p, self._actor_out.dev.value())
            ctx.synchronize()

        # CPU-side noise sample + add + clamp.
        var sigma = noise_scale * action_scale
        box_muller_normal(noise_cpu_p, Self.ACT_DIM)
        for j in range(Self.ACT_DIM):
            var a = actor_out_cpu_p[j] + noise_cpu_p[j] * sigma
            if a > action_scale:
                a = action_scale
            elif a < -action_scale:
                a = -action_scale
            action_out[j] = a
