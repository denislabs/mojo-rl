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
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.optimizer.adam import Adam

from .nets import (
    TDMPC2Encoder, TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Termination,
)
from .wm_graph import TDMPC2WMGraph, NQ, NLOSS, TERM_COL


@always_inline
def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


@fieldwise_init
struct WMLossOut(Copyable & Movable):
    """Per-component world-model losses (already coef·ρ^t/norm weighted).
    `termination` is the BCE head loss (item B); 0 unless bce_coef > 0."""
    var consistency: Scalar[DT]
    var reward: Scalar[DT]
    var value: Scalar[DT]
    var termination: Scalar[DT]

    @always_inline
    def total(self) -> Scalar[DT]:
        return self.consistency + self.reward + self.value + self.termination


# ── GPU marshalling helpers + kernels (mirror dreamerv3/blocks + the
#    spike_wm_bptt_gpu device-buffer orchestration). ─────────────────────
@always_inline
def _dp(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


@always_inline
def _lt[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


def _upload(
    ctx: DeviceContext, src: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int
) raises -> DeviceBuffer[DT]:
    """Host raw pointer → fresh device buffer (one H2D)."""
    var d = ctx.enqueue_create_buffer[DT](n)
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr()[i] = src[i]
    ctx.enqueue_copy(d, h)
    ctx.synchronize()
    return d^


def _copyk[N: Int](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](src[i])


def _extract_carry_k[B_: Int, LAT_: Int, OW_: Int](
    ob: LayoutTensor[DT, Layout.row_major(B_ * OW_), MutAnyOrigin],
    carry_next: LayoutTensor[DT, Layout.row_major(B_ * LAT_), MutAnyOrigin],
):
    """carry_next[b,k] = out[b, NLOSS+k]  (znext passthrough columns)."""
    var i = Int(global_idx.x)
    if i < B_ * LAT_:
        var b = i // LAT_
        var k = i % LAT_
        carry_next[i] = rebind[Scalar[DT]](ob[b * OW_ + NLOSS + k])


def _seed_wm_k[B_: Int, OW_: Int, LAT_: Int, NQ_: Int](
    seed: LayoutTensor[DT, Layout.row_major(B_ * OW_), MutAnyOrigin],
    gz: LayoutTensor[DT, Layout.row_major(B_ * LAT_), MutAnyOrigin],
    sc_cons: Scalar[DT],
    sc_rew: Scalar[DT],
    sc_val: Scalar[DT],
    sc_term: Scalar[DT],
):
    """seed[b] = [sc_cons, sc_rew, sc_val×NQ, sc_term, gz[b]]."""
    var b = Int(global_idx.x)
    if b < B_:
        seed[b * OW_ + 0] = sc_cons
        seed[b * OW_ + 1] = sc_rew
        for q in range(NQ_):
            seed[b * OW_ + 2 + q] = sc_val
        seed[b * OW_ + TERM_COL] = sc_term
        for k in range(LAT_):
            seed[b * OW_ + NLOSS + k] = rebind[Scalar[DT]](gz[b * LAT_ + k])


def _accum_metric_k[B_: Int, OW_: Int, NQ_: Int](
    ob: LayoutTensor[DT, Layout.row_major(B_ * OW_), MutAnyOrigin],
    acc: LayoutTensor[DT, Layout.row_major(4), MutAnyOrigin],
    sc_cons: Scalar[DT],
    sc_rew: Scalar[DT],
    sc_val: Scalar[DT],
    sc_term: Scalar[DT],
):
    """acc += [Σ_b sc_cons·cons, Σ_b sc_rew·rew, Σ_b sc_val·Σ_q v_q,
    Σ_b sc_term·tloss]. One thread."""
    var t = Int(global_idx.x)
    if t == 0:
        var sc: Scalar[DT] = 0.0
        var sr: Scalar[DT] = 0.0
        var sv: Scalar[DT] = 0.0
        var st: Scalar[DT] = 0.0
        for b in range(B_):
            sc += sc_cons * rebind[Scalar[DT]](ob[b * OW_ + 0])
            sr += sc_rew * rebind[Scalar[DT]](ob[b * OW_ + 1])
            var v: Scalar[DT] = 0.0
            for q in range(NQ_):
                v += rebind[Scalar[DT]](ob[b * OW_ + 2 + q])
            sv += sc_val * v
            st += sc_term * rebind[Scalar[DT]](ob[b * OW_ + TERM_COL])
        acc[0] = rebind[Scalar[DT]](acc[0]) + sc
        acc[1] = rebind[Scalar[DT]](acc[1]) + sr
        acc[2] = rebind[Scalar[DT]](acc[2]) + sv
        acc[3] = rebind[Scalar[DT]](acc[3]) + st


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
    QP: Float64 = 0.0,
](Movable & ImplicitlyDestructible):
    comptime EncT = TDMPC2Encoder[Self.OBS, Self.ENC, Self.LATENT, Self.SN]
    comptime DynT = TDMPC2Dynamics[Self.LATENT, Self.ACT, Self.MLP, Self.SN]
    comptime RewT = TDMPC2Reward[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]
    comptime TermT = TDMPC2Termination[Self.LATENT, Self.ACT, Self.MLP]
    comptime GraphT = TDMPC2WMGraph[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
        Self.VMAX, Self.QP,
    ]
    comptime OUTW = NLOSS + Self.LATENT

    var consistency_coef: Scalar[DT]
    var reward_coef: Scalar[DT]
    var value_coef: Scalar[DT]
    # Termination BCE coefficient (item B). 0.0 → non-episodic: the term head
    # gets zero gradient (Adam no-op) → other nets bit-identical. Reference
    # `termination_coef` (e.g. 1.0) for episodic envs.
    var termination_coef: Scalar[DT]
    var rho: Scalar[DT]

    # Persistent GPU scratch (allocated once in make[gpu]; None on CPU). Reused
    # across train steps to avoid per-step device allocation + per-upload syncs.
    var d_obs: Optional[DeviceBuffer[DT]]
    var d_act: Optional[DeviceBuffer[DT]]
    var d_rew: Optional[DeviceBuffer[DT]]
    var d_td: Optional[DeviceBuffer[DT]]
    var d_done: Optional[DeviceBuffer[DT]]
    var d_carry: Optional[DeviceBuffer[DT]]
    var d_zen: Optional[DeviceBuffer[DT]]
    var d_out: Optional[DeviceBuffer[DT]]
    var d_scratch: Optional[DeviceBuffer[DT]]
    var d_seed: Optional[DeviceBuffer[DT]]
    var d_gz: Optional[DeviceBuffer[DT]]
    var d_acc: Optional[DeviceBuffer[DT]]
    var d_gobs: Optional[DeviceBuffer[DT]]
    var h_obs: Optional[HostBuffer[DT]]
    var h_act: Optional[HostBuffer[DT]]
    var h_rew: Optional[HostBuffer[DT]]
    var h_td: Optional[HostBuffer[DT]]
    var h_done: Optional[HostBuffer[DT]]
    var h_acc: Optional[HostBuffer[DT]]

    def __init__(out self):
        self.consistency_coef = Scalar[DT](20.0)
        self.reward_coef = Scalar[DT](0.1)
        self.value_coef = Scalar[DT](0.1)
        self.termination_coef = Scalar[DT](0.0)
        self.rho = Scalar[DT](0.5)
        self.d_obs = None; self.d_act = None; self.d_rew = None
        self.d_td = None; self.d_done = None; self.d_carry = None
        self.d_zen = None
        self.d_out = None; self.d_scratch = None; self.d_seed = None
        self.d_gz = None; self.d_acc = None; self.d_gobs = None
        self.h_obs = None; self.h_act = None; self.h_rew = None
        self.h_td = None; self.h_done = None; self.h_acc = None

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        termination_coef: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "WMStep: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        s.termination_coef = termination_coef
        comptime if target == "gpu":
            var c = ctx.value()
            comptime LAT = Self.LATENT
            comptime OW = Self.OUTW
            s.d_obs = c.enqueue_create_buffer[DT]((Self.H + 1) * Self.B * Self.OBS)
            s.d_act = c.enqueue_create_buffer[DT](Self.H * Self.B * Self.ACT)
            s.d_rew = c.enqueue_create_buffer[DT](Self.H * Self.B)
            s.d_td = c.enqueue_create_buffer[DT](Self.H * Self.B)
            s.d_done = c.enqueue_create_buffer[DT](Self.H * Self.B)
            s.d_carry = c.enqueue_create_buffer[DT]((Self.H + 1) * Self.B * LAT)
            s.d_zen = c.enqueue_create_buffer[DT](Self.H * Self.B * LAT)
            s.d_out = c.enqueue_create_buffer[DT](Self.B * OW)
            s.d_scratch = c.enqueue_create_buffer[DT](Self.B * OW)
            s.d_seed = c.enqueue_create_buffer[DT](Self.B * OW)
            s.d_gz = c.enqueue_create_buffer[DT](Self.B * LAT)
            s.d_acc = c.enqueue_create_buffer[DT](4)
            s.d_gobs = c.enqueue_create_buffer[DT](Self.B * Self.OBS)
            s.h_obs = c.enqueue_create_host_buffer[DT](
                (Self.H + 1) * Self.B * Self.OBS
            )
            s.h_act = c.enqueue_create_host_buffer[DT](Self.H * Self.B * Self.ACT)
            s.h_rew = c.enqueue_create_host_buffer[DT](Self.H * Self.B)
            s.h_td = c.enqueue_create_host_buffer[DT](Self.H * Self.B)
            s.h_done = c.enqueue_create_host_buffer[DT](Self.H * Self.B)
            s.h_acc = c.enqueue_create_host_buffer[DT](4)
            c.synchronize()
        return s^

    def step[target: StaticString](
        mut self,
        mut graph: Self.GraphT,
        mut enc: Self.EncT,
        mut dyn: Self.DynT,
        mut rew_net: Self.RewT,
        mut q: List[Self.QNetT],
        mut term_net: Self.TermT,
        mut enc_opt: Adam,
        mut dyn_opt: Adam,
        mut rew_opt: Adam,
        mut q_opt: List[Adam],
        mut term_opt: Adam,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [(H+1),B,OBS] (host)
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [H,B,ACT]
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [H,B]
        td: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [H,B]
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [H,B] BCE target
        ctx: Optional[DeviceContext] = None,
    ) raises -> WMLossOut:
        comptime if target == "cpu":
            return self._wm_cpu[target](
                graph, enc, dyn, rew_net, q, term_net,
                enc_opt, dyn_opt, rew_opt, q_opt, term_opt,
                obs, act, rew, td, done,
            )
        else:
            return self._wm_gpu[target](
                graph, enc, dyn, rew_net, q, term_net,
                enc_opt, dyn_opt, rew_opt, q_opt, term_opt,
                obs, act, rew, td, done,
                ctx.value(),
            )

    def _wm_cpu[target: StaticString](
        mut self,
        mut graph: Self.GraphT,
        mut enc: Self.EncT,
        mut dyn: Self.DynT,
        mut rew_net: Self.RewT,
        mut q: List[Self.QNetT],
        mut term_net: Self.TermT,
        mut enc_opt: Adam,
        mut dyn_opt: Adam,
        mut rew_opt: Adam,
        mut q_opt: List[Adam],
        mut term_opt: Adam,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [(H+1),B,OBS]
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [H,B,ACT]
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [H,B]
        td: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [H,B]
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [H,B] BCE target
    ) raises -> WMLossOut:
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
        graph.set_external["term", Self.TermT](term_net)

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
        var cons_t: Scalar[DT] = 0.0
        var rew_t: Scalar[DT] = 0.0
        var val_t: Scalar[DT] = 0.0
        var term_t: Scalar[DT] = 0.0
        var rho_t: Scalar[DT] = 1.0
        var inv_b = Scalar[DT](1.0) / Scalar[DT](Self.B)
        var inv_h = Scalar[DT](1.0) / Scalar[DT](Self.H)
        var inv_lat = Scalar[DT](1.0) / Scalar[DT](LAT)
        var inv_nq = Scalar[DT](1.0) / Scalar[DT](NQ)
        for t in range(Self.H):
            self._set_step_inputs[target](graph, carry, zen, act, rew, td, done, t)
            var ot = TileTensor(out, row_major[Self.B, OW]())
            graph.forward[target, Self.B](ot)
            var nxt = carry + (t + 1) * Self.B * LAT
            for b in range(Self.B):
                for k in range(LAT):
                    nxt[b * LAT + k] = out[b * OW + NLOSS + k]
                var cons = out[b * OW + 0]
                var rl = out[b * OW + 1]
                var vl: Scalar[DT] = 0.0
                for qq in range(NQ):
                    vl += out[b * OW + 2 + qq]
                var tl = out[b * OW + TERM_COL]
                cons_t += rho_t * inv_h * self.consistency_coef * inv_b * inv_lat * cons
                rew_t += rho_t * inv_h * self.reward_coef * inv_b * rl
                val_t += rho_t * inv_h * self.value_coef * inv_b * inv_nq * vl
                term_t += rho_t * inv_h * self.termination_coef * inv_b * tl
            rho_t *= self.rho

        # ── 4. Zero grads (encoder + all external WM modules). ─────────
        enc_opt.zero_grad[target, Self.EncT](enc)
        dyn_opt.zero_grad[target, Self.DynT](dyn)
        rew_opt.zero_grad[target, Self.RewT](rew_net)
        for i in range(NQ):
            q_opt[i].zero_grad[target, Self.QNetT](q[i])
        term_opt.zero_grad[target, Self.TermT](term_net)

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
            self._set_step_inputs[target](graph, carry, zen, act, rew, td, done, t)
            var sct = TileTensor(scratch, row_major[Self.B, OW]())
            graph.forward[target, Self.B](sct)

            var sc_cons = self.consistency_coef * rho_rev * inv_b * inv_lat * inv_h
            var sc_rew = self.reward_coef * rho_rev * inv_b * inv_h
            var sc_val = self.value_coef * rho_rev * inv_b * inv_nq * inv_h
            var sc_term = self.termination_coef * rho_rev * inv_b * inv_h
            for b in range(Self.B):
                seed[b * OW + 0] = sc_cons
                seed[b * OW + 1] = sc_rew
                for qq in range(NQ):
                    seed[b * OW + 2 + qq] = sc_val
                seed[b * OW + TERM_COL] = sc_term
                for k in range(LAT):
                    seed[b * OW + NLOSS + k] = gz[b * LAT + k]
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
        term_opt.step[target, Self.TermT](term_net)

        carry.free(); zen.free(); out.free()
        gz.free(); seed.free(); scratch.free(); gobs.free()
        return WMLossOut(cons_t, rew_t, val_t, term_t)

    def _wm_gpu[target: StaticString](
        mut self,
        mut graph: Self.GraphT,
        mut enc: Self.EncT,
        mut dyn: Self.DynT,
        mut rew_net: Self.RewT,
        mut q: List[Self.QNetT],
        mut term_net: Self.TermT,
        mut enc_opt: Adam,
        mut dyn_opt: Adam,
        mut rew_opt: Adam,
        mut q_opt: List[Adam],
        mut term_opt: Adam,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [(H+1),B,OBS] host
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [H,B,ACT] host
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [H,B] host
        td: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [H,B] host
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [H,B] host BCE target
        ctx: DeviceContext,
    ) raises -> WMLossOut:
        comptime LAT = Self.LATENT
        comptime OW = Self.OUTW
        comptime BB = Self.B
        comptime OBSD = Self.OBS

        # ── bind persistent scratch + stage inputs (fill host buffers
        #    in-place, async H2D, single sync at the end) ────────────────
        var d_obs = self.d_obs.value()
        var d_act = self.d_act.value()
        var d_rew = self.d_rew.value()
        var d_td = self.d_td.value()
        var d_done = self.d_done.value()
        var d_carry = self.d_carry.value()
        var d_zen = self.d_zen.value()
        var d_out = self.d_out.value()
        var d_scratch = self.d_scratch.value()
        var d_seed = self.d_seed.value()
        var d_gz = self.d_gz.value()
        var d_acc = self.d_acc.value()
        var d_gobs = self.d_gobs.value()
        var ho = self.h_obs.value()
        var ha = self.h_act.value()
        var hr = self.h_rew.value()
        var htd = self.h_td.value()
        var hdone = self.h_done.value()
        for i in range((Self.H + 1) * BB * OBSD):
            ho.unsafe_ptr()[i] = obs[i]
        for i in range(Self.H * BB * Self.ACT):
            ha.unsafe_ptr()[i] = act[i]
        for i in range(Self.H * BB):
            hr.unsafe_ptr()[i] = rew[i]
            htd.unsafe_ptr()[i] = td[i]
            hdone.unsafe_ptr()[i] = done[i]
        ctx.enqueue_copy(d_obs, ho)
        ctx.enqueue_copy(d_act, ha)
        ctx.enqueue_copy(d_rew, hr)
        ctx.enqueue_copy(d_td, htd)
        ctx.enqueue_copy(d_done, hdone)
        d_acc.enqueue_fill(0.0)

        graph.set_external["znext", Self.DynT](dyn)
        graph.set_external["rlog", Self.RewT](rew_net)
        graph.set_external["q0", Self.QNetT](q[0])
        graph.set_external["q1", Self.QNetT](q[1])
        graph.set_external["q2", Self.QNetT](q[2])
        graph.set_external["q3", Self.QNetT](q[3])
        graph.set_external["q4", Self.QNetT](q[4])
        graph.set_external["term", Self.TermT](term_net)

        comptime nb_lat = (BB * LAT + TPB - 1) // TPB
        comptime nb_b = (BB + TPB - 1) // TPB
        comptime ext_k = _extract_carry_k[BB, LAT, OW]
        comptime acc_k = _accum_metric_k[BB, OW, NQ]
        comptime seed_k = _seed_wm_k[BB, OW, LAT, NQ]
        comptime cp_k = _copyk[BB * LAT]

        var inv_b = Scalar[DT](1.0) / Scalar[DT](BB)
        var inv_h = Scalar[DT](1.0) / Scalar[DT](Self.H)
        var inv_lat = Scalar[DT](1.0) / Scalar[DT](LAT)
        var inv_nq = Scalar[DT](1.0) / Scalar[DT](NQ)

        # ── 1. consistency targets enc(obs[t+1]) → d_zen[t] (stop-grad) ─
        for t in range(Self.H):
            var dst = _dp(d_zen) + t * BB * LAT
            var dst_t = TileTensor(dst, row_major[BB, LAT]())
            enc.forward[target, BB](
                TileTensor(
                    _dp(d_obs) + (t + 1) * BB * OBSD, row_major[BB, OBSD]()
                ),
                output=dst_t,
            )
        # ── 2. z_0 = enc(obs[0]) → d_carry[0] (cache = obs[0]) ─────────
        var z0_t = TileTensor(_dp(d_carry), row_major[BB, LAT]())
        enc.forward[target, BB](
            TileTensor(_dp(d_obs), row_major[BB, OBSD]()), output=z0_t,
        )

        # ── 3. forward scan ────────────────────────────────────────────
        var rho_t = Scalar[DT](1.0)
        for t in range(Self.H):
            self._set_step_inputs[target](
                graph, _dp(d_carry), _dp(d_zen), _dp(d_act), _dp(d_rew),
                _dp(d_td), _dp(d_done), t,
            )
            var ot = TileTensor(_dp(d_out), row_major[BB, OW]())
            graph.forward[target, BB](ot)
            ctx.enqueue_function[ext_k](
                _lt[BB * OW](_dp(d_out)),
                _lt[BB * LAT](_dp(d_carry) + (t + 1) * BB * LAT),
                grid_dim=nb_lat, block_dim=TPB,
            )
            ctx.enqueue_function[acc_k](
                _lt[BB * OW](_dp(d_out)), _lt[4](_dp(d_acc)),
                self.consistency_coef * rho_t * inv_b * inv_lat * inv_h,
                self.reward_coef * rho_t * inv_b * inv_h,
                self.value_coef * rho_t * inv_b * inv_nq * inv_h,
                self.termination_coef * rho_t * inv_b * inv_h,
                grid_dim=1, block_dim=1,
            )
            rho_t *= self.rho

        # ── 4. zero grads + gz ─────────────────────────────────────────
        enc_opt.zero_grad[target, Self.EncT](enc)
        dyn_opt.zero_grad[target, Self.DynT](dyn)
        rew_opt.zero_grad[target, Self.RewT](rew_net)
        for i in range(NQ):
            q_opt[i].zero_grad[target, Self.QNetT](q[i])
        term_opt.zero_grad[target, Self.TermT](term_net)
        d_gz.enqueue_fill(0.0)

        # ── 5. reverse-scan BPTT ───────────────────────────────────────
        var rho_rev = Scalar[DT](1.0)
        for _ in range(Self.H - 1):
            rho_rev *= self.rho
        for rev in range(Self.H):
            var t = Self.H - 1 - rev
            self._set_step_inputs[target](
                graph, _dp(d_carry), _dp(d_zen), _dp(d_act), _dp(d_rew),
                _dp(d_td), _dp(d_done), t,
            )
            var sct = TileTensor(_dp(d_scratch), row_major[BB, OW]())
            graph.forward[target, BB](sct)
            ctx.enqueue_function[seed_k](
                _lt[BB * OW](_dp(d_seed)), _lt[BB * LAT](_dp(d_gz)),
                self.consistency_coef * rho_rev * inv_b * inv_lat * inv_h,
                self.reward_coef * rho_rev * inv_b * inv_h,
                self.value_coef * rho_rev * inv_b * inv_nq * inv_h,
                self.termination_coef * rho_rev * inv_b * inv_h,
                grid_dim=nb_b, block_dim=TPB,
            )
            graph.vjp[target, BB](
                TileTensor(_dp(d_seed), row_major[BB, OW]())
            )
            ctx.enqueue_function[cp_k](
                _lt[BB * LAT](graph.grad_input_ptr["z"]()),
                _lt[BB * LAT](_dp(d_gz)),
                grid_dim=nb_lat, block_dim=TPB,
            )
            rho_rev /= self.rho

        # ── 6. encoder backward from t=0 carry grad ────────────────────
        var z0r_t = TileTensor(_dp(d_carry), row_major[BB, LAT]())
        enc.forward[target, BB](
            TileTensor(_dp(d_obs), row_major[BB, OBSD]()), output=z0r_t,
        )
        var gobs_t = TileTensor(_dp(d_gobs), row_major[BB, OBSD]())
        enc.vjp[target, BB](
            TileTensor(_dp(d_gz), row_major[BB, LAT]()), gobs_t,
        )

        # ── 7. optimizer steps ─────────────────────────────────────────
        enc_opt.step[target, Self.EncT](enc)
        dyn_opt.step[target, Self.DynT](dyn)
        rew_opt.step[target, Self.RewT](rew_net)
        for i in range(NQ):
            q_opt[i].step[target, Self.QNetT](q[i])
        term_opt.step[target, Self.TermT](term_net)

        var h = self.h_acc.value()
        ctx.enqueue_copy(h, d_acc)
        ctx.synchronize()
        return WMLossOut(
            h.unsafe_ptr()[0], h.unsafe_ptr()[1], h.unsafe_ptr()[2],
            h.unsafe_ptr()[3],
        )

    def _set_step_inputs[target: StaticString](
        self,
        mut graph: Self.GraphT,
        carry: UnsafePointer[Scalar[DT], MutAnyOrigin],
        zen: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],
        td: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],
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
        graph.set_input["done", Self.B](
            TileTensor(done + t * Self.B, row_major[Self.B, 1]())
        )
