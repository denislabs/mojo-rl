"""TD-MPC2 world-model BPTT step (CPU; GPU is P4).

Mirrors DreamerV3's `WMStep` carry-passthrough BPTT and the validated
`tests/nn2/spike_wm_bptt.mojo` scan. The world model's dynamics, reward, and
5 Q heads are **trainer-owned standalone Modules** referenced by the WM
ComputeGraph as `ExternalNode`s (so the policy graph + MPPI planner can call
the same instances). This block binds them, runs the forward scan, reverse
BPTT scan, encoder backward, and steps one Adam per module.

  z_0 = encode(obs[0])
  for t in 0..H-1:
      out_t = WMGraph(z=carry_t, a=a_t, z_enc_next=sg·encode(obs[t+1]),
                      r=r_t, td=td_t)            # dyn/rew/Q bound as externals
      carry_{t+1} = out_t[:, 7:]                 # znext (dynamics output)
  reverse scan seeds loss cols (coef·ρ^t/norm) + znext cols (carry grad),
  vjp accumulates into the external modules, grad_input["z"] threads back.

This is the multi-step gradient flow the legacy CPU path skipped
(`deep_agents/tdmpc2/tdmpc2.mojo:861-867`). Normalization matches reference
`_update` (consistency = MSE mean over batch×latent; reward = CE mean over
batch; value = CE mean over batch×num_q; all ·ρ^t and ÷horizon).

Inputs (t-major, contiguous per step):
  obs [(H+1),B,OBS], act [H,B,ACT], r [H,B], td [H,B] (stop-grad targets).
"""

from std.memory import alloc
from layout import TileTensor, row_major
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.optimizer.adam import Adam

from .nets import (
    TDMPC2Encoder, TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet,
)
from .wm_graph import TDMPC2WMGraph, NQ


@always_inline
def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


struct WMStep[
    OBS: Int,
    ENC: Int,
    ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    H: Int,
](Movable & ImplicitlyDestructible):
    comptime EncT = TDMPC2Encoder[Self.OBS, Self.ENC, Self.LATENT, Self.SN]
    comptime DynT = TDMPC2Dynamics[Self.LATENT, Self.ACT, Self.MLP, Self.SN]
    comptime RewT = TDMPC2Reward[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime GraphT = TDMPC2WMGraph[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.SN, Self.VMIN, Self.VMAX
    ]
    comptime OUTW = 7 + Self.LATENT

    var consistency_coef: Scalar[DT]
    var reward_coef: Scalar[DT]
    var value_coef: Scalar[DT]
    var rho: Scalar[DT]

    def __init__(out self):
        self.consistency_coef = Scalar[DT](20.0)
        self.reward_coef = Scalar[DT](0.1)
        self.value_coef = Scalar[DT](0.1)
        self.rho = Scalar[DT](0.5)

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu", (
            "WMStep: only the CPU path is implemented (GPU is port-plan P4)"
        )
        return Self()

    def step[target: StaticString](
        mut self,
        mut graph: Self.GraphT,
        mut enc: Self.EncT,
        mut dyn: Self.DynT,
        mut rew_net: Self.RewT,
        mut q: List[Self.QNetT],
        mut enc_opt: Adam,
        mut dyn_opt: Adam,
        mut rew_opt: Adam,
        mut q_opt: List[Adam],
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [(H+1),B,OBS]
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [H,B,ACT]
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [H,B]
        td: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [H,B]
    ) raises -> Scalar[DT]:
        comptime assert target == "cpu", "WMStep.step: CPU only (P4 = GPU)"
        comptime LAT = Self.LATENT
        comptime OW = Self.OUTW

        # ── Bind external modules (stable for this step). ──────────────
        graph.set_external["znext", Self.DynT](dyn)
        graph.set_external["rlog", Self.RewT](rew_net)
        graph.set_external["q0", Self.QNetT](q[0])
        graph.set_external["q1", Self.QNetT](q[1])
        graph.set_external["q2", Self.QNetT](q[2])
        graph.set_external["q3", Self.QNetT](q[3])
        graph.set_external["q4", Self.QNetT](q[4])

        var carry = _alloc((Self.H + 1) * Self.B * LAT)
        var zen = _alloc(Self.H * Self.B * LAT)
        var out = _alloc(Self.B * OW)

        # ── 1. Consistency targets: encode(obs[t+1]), stop-grad. ───────
        for t in range(Self.H):
            var src = obs + (t + 1) * Self.B * Self.OBS
            var dst = zen + t * Self.B * LAT
            var dst_t = TileTensor(dst, row_major[Self.B, LAT]())
            enc.forward[target, Self.B](
                TileTensor(src, row_major[Self.B, Self.OBS]()),
                output=dst_t,
            )

        # ── 2. z_0 = encode(obs[0]) — last enc.forward → cache = obs[0]. ─
        var z0_t = TileTensor(carry, row_major[Self.B, LAT]())
        enc.forward[target, Self.B](
            TileTensor(obs, row_major[Self.B, Self.OBS]()),
            output=z0_t,
        )

        # ── 3. Forward scan: roll latents, accumulate weighted loss. ───
        var total: Scalar[DT] = 0.0
        var rho_t: Scalar[DT] = 1.0
        var inv_b = Scalar[DT](1.0) / Scalar[DT](Self.B)
        var inv_h = Scalar[DT](1.0) / Scalar[DT](Self.H)
        var inv_lat = Scalar[DT](1.0) / Scalar[DT](LAT)
        var inv_nq = Scalar[DT](1.0) / Scalar[DT](NQ)
        for t in range(Self.H):
            self._set_step_inputs[target](graph, carry, zen, act, rew, td, t)
            var ot = TileTensor(out, row_major[Self.B, OW]())
            graph.forward[target, Self.B](ot)
            var nxt = carry + (t + 1) * Self.B * LAT
            for b in range(Self.B):
                for k in range(LAT):
                    nxt[b * LAT + k] = out[b * OW + 7 + k]
                var cons = out[b * OW + 0]
                var rl = out[b * OW + 1]
                var vl: Scalar[DT] = 0.0
                for qq in range(NQ):
                    vl += out[b * OW + 2 + qq]
                total += rho_t * inv_h * (
                    self.consistency_coef * inv_b * inv_lat * cons
                    + self.reward_coef * inv_b * rl
                    + self.value_coef * inv_b * inv_nq * vl
                )
            rho_t *= self.rho

        # ── 4. Zero grads (encoder + all external WM modules). ─────────
        enc_opt.zero_grad[target, Self.EncT](enc)
        dyn_opt.zero_grad[target, Self.DynT](dyn)
        rew_opt.zero_grad[target, Self.RewT](rew_net)
        for i in range(NQ):
            q_opt[i].zero_grad[target, Self.QNetT](q[i])

        # ── 5. Reverse-scan BPTT. ──────────────────────────────────────
        var gz = _alloc(Self.B * LAT)
        for i in range(Self.B * LAT):
            gz[i] = 0.0
        var seed = _alloc(Self.B * OW)
        var scratch = _alloc(Self.B * OW)

        var rho_rev: Scalar[DT] = 1.0
        for _ in range(Self.H - 1):
            rho_rev *= self.rho

        for rev in range(Self.H):
            var t = Self.H - 1 - rev
            self._set_step_inputs[target](graph, carry, zen, act, rew, td, t)
            var sct = TileTensor(scratch, row_major[Self.B, OW]())
            graph.forward[target, Self.B](sct)

            var sc_cons = self.consistency_coef * rho_rev * inv_b * inv_lat * inv_h
            var sc_rew = self.reward_coef * rho_rev * inv_b * inv_h
            var sc_val = self.value_coef * rho_rev * inv_b * inv_nq * inv_h
            for b in range(Self.B):
                seed[b * OW + 0] = sc_cons
                seed[b * OW + 1] = sc_rew
                for qq in range(NQ):
                    seed[b * OW + 2 + qq] = sc_val
                for k in range(LAT):
                    seed[b * OW + 7 + k] = gz[b * LAT + k]
            graph.vjp[target, Self.B](TileTensor(seed, row_major[Self.B, OW]()))

            var gzin = graph.grad_input_ptr["z"]()
            for i in range(Self.B * LAT):
                gz[i] = gzin[i]
            rho_rev /= self.rho

        # ── 6. Encoder backward from the t=0 carry grad. ───────────────
        var z0r_t = TileTensor(carry, row_major[Self.B, LAT]())
        enc.forward[target, Self.B](
            TileTensor(obs, row_major[Self.B, Self.OBS]()),
            output=z0r_t,
        )
        var gobs = _alloc(Self.B * Self.OBS)
        var gobs_t = TileTensor(gobs, row_major[Self.B, Self.OBS]())
        enc.vjp[target, Self.B](
            TileTensor(gz, row_major[Self.B, LAT]()),
            gobs_t,
        )

        # ── 7. Optimizer steps (one Adam per module). ──────────────────
        enc_opt.step[target, Self.EncT](enc)
        dyn_opt.step[target, Self.DynT](dyn)
        rew_opt.step[target, Self.RewT](rew_net)
        for i in range(NQ):
            q_opt[i].step[target, Self.QNetT](q[i])

        carry.free(); zen.free(); out.free()
        gz.free(); seed.free(); scratch.free(); gobs.free()
        return total

    def _set_step_inputs[target: StaticString](
        self,
        mut graph: Self.GraphT,
        carry: UnsafePointer[Scalar[DT], MutAnyOrigin],
        zen: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],
        td: UnsafePointer[Scalar[DT], MutAnyOrigin],
        t: Int,
    ) raises:
        comptime LAT = Self.LATENT
        graph.set_input["z", Self.B](
            TileTensor(carry + t * Self.B * LAT, row_major[Self.B, LAT]())
        )
        graph.set_input["a", Self.B](
            TileTensor(act + t * Self.B * Self.ACT, row_major[Self.B, Self.ACT]())
        )
        graph.set_input["z_enc_next", Self.B](
            TileTensor(zen + t * Self.B * LAT, row_major[Self.B, LAT]())
        )
        graph.set_input["r", Self.B](
            TileTensor(rew + t * Self.B, row_major[Self.B, 1]())
        )
        graph.set_input["td", Self.B](
            TileTensor(td + t * Self.B, row_major[Self.B, 1]())
        )
