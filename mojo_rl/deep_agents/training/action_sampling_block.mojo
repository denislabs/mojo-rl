"""ActionSamplingBlock — env-interaction policy head (single-env, storage).

Encapsulates the warmup-vs-policy `select_action` flow used by every off-policy
continuous-control trainer:

  if step_idx < learning_starts:  # uniform exploration on [-scale, +scale]^ACT
  else:                           # actor.forward(obs) → (sampler) → clamp ±scale

Three call shapes:
  select_stochastic[SAMPLER]          — SAC (actor [mu|ls] → sampler [a|logp]).
  select_deterministic                — DDPG eval (actor.OUT_DIM == ACT_DIM).
  select_deterministic_with_noise     — DDPG/TD3 acting (actor + N(0,σ²), clamp).

STORAGE migration (Stage 5): single-step scratch is owned storage `Tensor`s
(CPU `data` + lazy device buffer via upload/download); actor/sampler use the
storage `forward[target,1](TensorRefs, mut out, ctx)` surface; no Scratch/
TargetStorage/TileTensor/unsafe_ptr. Exploration noise is host Box-Muller
(inline, no raw-pointer RNG helper).
"""

from std.random import random_float64
from std.math import sqrt as fsqrt, log as flog, cos as fcos, sin as fsin, pi
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs


struct ActionSamplingBlock[
    ACTOR: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    SAMPLER_OUT_DIM: Int,
](Movable & ImplicitlyDeletable):
    comptime ACTOR_OUT_DIM = Self.ACTOR.OUT_DIM

    var _ob1: Tensor         # [OBS_DIM]
    var _actor_out: Tensor   # [ACTOR_OUT_DIM]
    var _sampler_out: Tensor # [SAMPLER_OUT_DIM]
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self._ob1 = Tensor()
        self._actor_out = Tensor()
        self._sampler_out = Tensor()
        self.ctx = None

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "ActionSamplingBlock: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error("ActionSamplingBlock.make[target='gpu']: ctx required")
        var b = Self()
        # CPU `data` lists for all (write obs / read back actor/sampler out);
        # device buffers are created lazily by upload()/forward() on GPU.
        b._ob1 = Tensor.alloc(Self.OBS_DIM)
        b._actor_out = Tensor.alloc(Self.ACTOR_OUT_DIM)
        b._sampler_out = Tensor.alloc(Self.SAMPLER_OUT_DIM)
        b.ctx = ctx
        return b^

    # ─── helpers ──────────────────────────────────────────────────────

    def _write_warmup(
        mut self, mut action_out: List[Scalar[DT]], action_scale: Scalar[DT]
    ):
        for j in range(Self.ACT_DIM):
            var u = Scalar[DT](2.0 * random_float64() - 1.0)
            action_out[j] = u * action_scale

    @staticmethod
    def _clamp_into(
        ref src: List[Scalar[DT]],
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
        if step_idx < learning_starts:
            self._write_warmup(action_out, action_scale)
            return
        for d in range(Self.OBS_DIM):
            self._ob1.data[d] = obs[d]
        comptime if target == "cpu":
            actor.forward["cpu", 1](TensorRefs[Self.ACTOR.ARITY](self._ob1), self._actor_out)
            sampler.forward["cpu", 1](
                TensorRefs[SAMPLER.ARITY](self._actor_out), self._sampler_out
            )
            Self._clamp_into(self._sampler_out.data, action_out, action_scale)
        else:
            var c = self.ctx.value()
            self._ob1.upload(c)
            actor.forward["gpu", 1](
                TensorRefs[Self.ACTOR.ARITY](self._ob1), self._actor_out, Optional(c)
            )
            sampler.forward["gpu", 1](
                TensorRefs[SAMPLER.ARITY](self._actor_out), self._sampler_out, Optional(c)
            )
            self._sampler_out.download(c)
            Self._clamp_into(self._sampler_out.data, action_out, action_scale)

    # ─── Deterministic (DDPG eval) ────────────────────────────────────

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
        comptime assert Self.ACTOR_OUT_DIM == Self.ACT_DIM, (
            "select_deterministic requires ACTOR.OUT_DIM == ACT_DIM"
        )
        if step_idx < learning_starts:
            self._write_warmup(action_out, action_scale)
            return
        for d in range(Self.OBS_DIM):
            self._ob1.data[d] = obs[d]
        comptime if target == "cpu":
            actor.forward["cpu", 1](TensorRefs[Self.ACTOR.ARITY](self._ob1), self._actor_out)
            Self._clamp_into(self._actor_out.data, action_out, action_scale)
        else:
            var c = self.ctx.value()
            self._ob1.upload(c)
            actor.forward["gpu", 1](
                TensorRefs[Self.ACTOR.ARITY](self._ob1), self._actor_out, Optional(c)
            )
            self._actor_out.download(c)
            Self._clamp_into(self._actor_out.data, action_out, action_scale)

    # ─── Deterministic + Gaussian noise (DDPG/TD3 acting) ─────────────

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
        comptime assert Self.ACTOR_OUT_DIM == Self.ACT_DIM, (
            "select_deterministic_with_noise requires ACTOR.OUT_DIM == ACT_DIM"
        )
        if step_idx < learning_starts:
            self._write_warmup(action_out, action_scale)
            return
        for d in range(Self.OBS_DIM):
            self._ob1.data[d] = obs[d]
        comptime if target == "cpu":
            actor.forward["cpu", 1](TensorRefs[Self.ACTOR.ARITY](self._ob1), self._actor_out)
        else:
            var c = self.ctx.value()
            self._ob1.upload(c)
            actor.forward["gpu", 1](
                TensorRefs[Self.ACTOR.ARITY](self._ob1), self._actor_out, Optional(c)
            )
            self._actor_out.download(c)

        # Host Box-Muller exploration noise, add, clamp.
        var sigma = noise_scale * action_scale
        var k = 0
        while k < Self.ACT_DIM:
            var u1 = random_float64()
            if u1 < 1e-12:
                u1 = 1e-12
            var u2 = random_float64()
            var radius = fsqrt(-2.0 * flog(u1))
            var ang = 2.0 * pi * u2
            var z0 = Scalar[DT](radius * fcos(ang))
            var a0 = self._actor_out.data[k] + z0 * sigma
            if a0 > action_scale:
                a0 = action_scale
            elif a0 < -action_scale:
                a0 = -action_scale
            action_out[k] = a0
            k += 1
            if k < Self.ACT_DIM:
                var z1 = Scalar[DT](radius * fsin(ang))
                var a1 = self._actor_out.data[k] + z1 * sigma
                if a1 > action_scale:
                    a1 = action_scale
                elif a1 < -action_scale:
                    a1 = -action_scale
                action_out[k] = a1
                k += 1
