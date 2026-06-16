"""TD-MPC2 multi-task world-model BPTT step (CPU + GPU) — item C, §14.3.

Clone of `wm_step.mojo` adapted for the task-conditioned world model:

  * Encoder input is `MAX_OBS + TASK_EMB` (obs zero-padded to MAX_OBS by the env
    wrapper, then the gathered task embedding concatenated) — built once per step
    into an augmented obs buffer.
  * The WM graph gains a `task_emb` input (set per step) feeding the 3-way `za`
    concat → dynamics/reward/Q/term.
  * Embedding gradient collection (sites 1 + 2 of three): after EACH reverse-scan
    `vjp` the `task_emb` input-slot grad (`grad_input_ptr["task_emb"]`) is
    scatter-added into the table by window task; after the encoder backward the
    encoder input-grad's last `TASK_EMB` columns are likewise accumulated. (The
    policy graph is site 3, in `policy_step_mt`.) The table's `zero_grad`/`step`
    are driven by the agent around the whole train step.

The metric/seed/carry GPU kernels and `WMLossOut` are reused from `wm_step.mojo`
(the graph output layout `8 + LATENT` is unchanged). See `wm_step.mojo` for the
forward/reverse scan documentation.
"""

from std.memory import alloc
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.optimizer.adam import Adam

from .nets_mt import (
    TDMPC2EncoderMT, TDMPC2DynamicsMT, TDMPC2RewardMT, TDMPC2QNetMT,
    TDMPC2TerminationMT,
)
from .wm_graph_mt import TDMPC2WMGraphMT
from .wm_graph import NQ, NLOSS, TERM_COL
from .wm_step import (
    WMLossOut, _dp, _lt, _alloc, _copyk, _extract_carry_k, _seed_wm_k,
    _accum_metric_k,
)
from .task_embedding import TaskEmbedding


def _build_oaug_k[H1B: Int, MAXOBS: Int, EMB: Int, B: Int](
    obs: LayoutTensor[DT, Layout.row_major(H1B * MAXOBS), MutAnyOrigin],
    tem: LayoutTensor[DT, Layout.row_major(B * EMB), MutAnyOrigin],
    oaug: LayoutTensor[DT, Layout.row_major(H1B * (MAXOBS + EMB)), MutAnyOrigin],
):
    """oaug[row] = [obs[row] | tem[row % B]] — broadcast the per-window task
    embedding across the (H+1) frames (row = t*B + b)."""
    var i = Int(global_idx.x)
    var AOBS = MAXOBS + EMB
    if i < H1B * AOBS:
        var row = i // AOBS
        var c = i % AOBS
        if c < MAXOBS:
            oaug[i] = rebind[Scalar[DT]](obs[row * MAXOBS + c])
        else:
            var b = row % B
            oaug[i] = rebind[Scalar[DT]](tem[b * EMB + (c - MAXOBS)])


struct WMStepMT[
    MAX_OBS: Int,
    ENC: Int,
    MAX_ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    H: Int,
    NUM_TASKS: Int,
    TASK_EMB: Int,
    QP: Float64 = 0.0,
](Movable & ImplicitlyDeletable):
    comptime AOBS = Self.MAX_OBS + Self.TASK_EMB
    comptime EncT = TDMPC2EncoderMT[
        Self.MAX_OBS, Self.ENC, Self.LATENT, Self.SN, Self.TASK_EMB
    ]
    comptime DynT = TDMPC2DynamicsMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.SN, Self.TASK_EMB
    ]
    comptime RewT = TDMPC2RewardMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB
    ]
    comptime QNetT = TDMPC2QNetMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB, Self.QP
    ]
    comptime TermT = TDMPC2TerminationMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.TASK_EMB
    ]
    comptime GraphT = TDMPC2WMGraphMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
        Self.VMAX, Self.TASK_EMB, Self.QP,
    ]
    comptime EmbT = TaskEmbedding[Self.NUM_TASKS, Self.TASK_EMB]
    comptime OUTW = NLOSS + Self.LATENT

    var consistency_coef: Scalar[DT]
    var reward_coef: Scalar[DT]
    var value_coef: Scalar[DT]
    var termination_coef: Scalar[DT]
    var rho: Scalar[DT]

    # Persistent GPU scratch.
    var d_obs: Optional[DeviceBuffer[DT]]
    var d_oaug: Optional[DeviceBuffer[DT]]
    var d_tem: Optional[DeviceBuffer[DT]]
    var d_tids: Optional[DeviceBuffer[DT]]
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
    var h_tids: Optional[HostBuffer[DT]]
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
        self.d_obs = None; self.d_oaug = None; self.d_tem = None
        self.d_tids = None; self.d_act = None; self.d_rew = None
        self.d_td = None; self.d_done = None; self.d_carry = None
        self.d_zen = None
        self.d_out = None; self.d_scratch = None; self.d_seed = None
        self.d_gz = None; self.d_acc = None; self.d_gobs = None
        self.h_obs = None; self.h_tids = None; self.h_act = None
        self.h_rew = None; self.h_td = None; self.h_done = None; self.h_acc = None

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        termination_coef: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "WMStepMT: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        s.termination_coef = termination_coef
        comptime if target == "gpu":
            var c = ctx.value()
            comptime LAT = Self.LATENT
            comptime OW = Self.OUTW
            comptime AOBS = Self.AOBS
            s.d_obs = c.enqueue_create_buffer[DT](
                (Self.H + 1) * Self.B * Self.MAX_OBS
            )
            s.d_oaug = c.enqueue_create_buffer[DT]((Self.H + 1) * Self.B * AOBS)
            s.d_tem = c.enqueue_create_buffer[DT](Self.B * Self.TASK_EMB)
            s.d_tids = c.enqueue_create_buffer[DT](Self.B)
            s.d_act = c.enqueue_create_buffer[DT](Self.H * Self.B * Self.MAX_ACT)
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
            s.d_gobs = c.enqueue_create_buffer[DT](Self.B * AOBS)
            s.h_obs = c.enqueue_create_host_buffer[DT](
                (Self.H + 1) * Self.B * Self.MAX_OBS
            )
            s.h_tids = c.enqueue_create_host_buffer[DT](Self.B)
            s.h_act = c.enqueue_create_host_buffer[DT](
                Self.H * Self.B * Self.MAX_ACT
            )
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
        mut task_emb: Self.EmbT,
        mut enc_opt: Adam,
        mut dyn_opt: Adam,
        mut rew_opt: Adam,
        mut q_opt: List[Adam],
        mut term_opt: Adam,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [(H+1),B,MAX_OBS] host
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [H,B,MAX_ACT]
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [H,B]
        td: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [H,B]
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [H,B] BCE target
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B] DT-encoded
        ctx: Optional[DeviceContext] = None,
    ) raises -> WMLossOut:
        comptime if target == "cpu":
            return self._wm_cpu[target](
                graph, enc, dyn, rew_net, q, term_net, task_emb,
                enc_opt, dyn_opt, rew_opt, q_opt, term_opt,
                obs, act, rew, td, done, task_ids,
            )
        else:
            return self._wm_gpu[target](
                graph, enc, dyn, rew_net, q, term_net, task_emb,
                enc_opt, dyn_opt, rew_opt, q_opt, term_opt,
                obs, act, rew, td, done, task_ids, ctx.value(),
            )

    def _wm_cpu[target: StaticString](
        mut self,
        mut graph: Self.GraphT,
        mut enc: Self.EncT,
        mut dyn: Self.DynT,
        mut rew_net: Self.RewT,
        mut q: List[Self.QNetT],
        mut term_net: Self.TermT,
        mut task_emb: Self.EmbT,
        mut enc_opt: Adam,
        mut dyn_opt: Adam,
        mut rew_opt: Adam,
        mut q_opt: List[Adam],
        mut term_opt: Adam,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],
        td: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> WMLossOut:
        comptime LAT = Self.LATENT
        comptime OW = Self.OUTW
        comptime AOBS = Self.AOBS
        comptime EMB = Self.TASK_EMB
        comptime MO = Self.MAX_OBS

        # gather per-window embeddings + build augmented obs.
        var tem = _alloc(Self.B * EMB)
        task_emb.gather[target, Self.B](task_ids, tem)
        var oaug = _alloc((Self.H + 1) * Self.B * AOBS)
        for row in range((Self.H + 1) * Self.B):
            var b = row % Self.B
            for i in range(MO):
                oaug[row * AOBS + i] = obs[row * MO + i]
            for e in range(EMB):
                oaug[row * AOBS + MO + e] = tem[b * EMB + e]

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

        # 1. consistency targets enc(obs[t+1]) (stop-grad).
        for t in range(Self.H):
            var src = oaug + (t + 1) * Self.B * AOBS
            var dst = zen + t * Self.B * LAT
            var dst_t = TileTensor(dst, row_major[Self.B, LAT]())
            enc.forward[target, Self.B](
                TileTensor(src, row_major[Self.B, AOBS]()), output=dst_t,
            )
        # 2. z_0 = enc(obs[0]).
        var z0_t = TileTensor(carry, row_major[Self.B, LAT]())
        enc.forward[target, Self.B](
            TileTensor(oaug, row_major[Self.B, AOBS]()), output=z0_t,
        )

        # 3. forward scan.
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
            self._set_step_inputs[target](
                graph, carry, tem, zen, act, rew, td, done, t
            )
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

        # 4. zero grads.
        enc_opt.zero_grad[target, Self.EncT](enc)
        dyn_opt.zero_grad[target, Self.DynT](dyn)
        rew_opt.zero_grad[target, Self.RewT](rew_net)
        for i in range(NQ):
            q_opt[i].zero_grad[target, Self.QNetT](q[i])
        term_opt.zero_grad[target, Self.TermT](term_net)

        # 5. reverse-scan BPTT.
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
            self._set_step_inputs[target](
                graph, carry, tem, zen, act, rew, td, done, t
            )
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
            # site 1: accumulate the task_emb input-slot grad into the table.
            task_emb.accumulate[target, Self.B, EMB, 0](
                task_ids, graph.grad_input_ptr["task_emb"]()
            )
            rho_rev /= self.rho

        # 6. encoder backward from t=0 carry grad → gobs [B, AOBS].
        var z0r_t = TileTensor(carry, row_major[Self.B, LAT]())
        enc.forward[target, Self.B](
            TileTensor(oaug, row_major[Self.B, AOBS]()), output=z0r_t,
        )
        var gobs = _alloc(Self.B * AOBS)
        var gobs_t = TileTensor(gobs, row_major[Self.B, AOBS]())
        enc.vjp[target, Self.B](
            TileTensor(gz, row_major[Self.B, LAT]()), gobs_t,
        )
        # site 2: accumulate encoder input-grad tail (last TASK_EMB cols).
        task_emb.accumulate[target, Self.B, AOBS, MO](task_ids, gobs)

        # 7. optimizer steps (nets only; table step is the agent's job).
        enc_opt.step[target, Self.EncT](enc)
        dyn_opt.step[target, Self.DynT](dyn)
        rew_opt.step[target, Self.RewT](rew_net)
        for i in range(NQ):
            q_opt[i].step[target, Self.QNetT](q[i])
        term_opt.step[target, Self.TermT](term_net)

        carry.free(); zen.free(); out.free()
        gz.free(); seed.free(); scratch.free(); gobs.free()
        tem.free(); oaug.free()
        return WMLossOut(cons_t, rew_t, val_t, term_t)

    def _wm_gpu[target: StaticString](
        mut self,
        mut graph: Self.GraphT,
        mut enc: Self.EncT,
        mut dyn: Self.DynT,
        mut rew_net: Self.RewT,
        mut q: List[Self.QNetT],
        mut term_net: Self.TermT,
        mut task_emb: Self.EmbT,
        mut enc_opt: Adam,
        mut dyn_opt: Adam,
        mut rew_opt: Adam,
        mut q_opt: List[Adam],
        mut term_opt: Adam,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],
        td: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ctx: DeviceContext,
    ) raises -> WMLossOut:
        comptime LAT = Self.LATENT
        comptime OW = Self.OUTW
        comptime BB = Self.B
        comptime MO = Self.MAX_OBS
        comptime AOBS = Self.AOBS
        comptime EMB = Self.TASK_EMB

        var d_obs = self.d_obs.value()
        var d_oaug = self.d_oaug.value()
        var d_tem = self.d_tem.value()
        var d_tids = self.d_tids.value()
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
        var htid = self.h_tids.value()
        var ha = self.h_act.value()
        var hr = self.h_rew.value()
        var htd = self.h_td.value()
        var hdone = self.h_done.value()
        for i in range((Self.H + 1) * BB * MO):
            ho.unsafe_ptr()[i] = obs[i]
        for i in range(BB):
            htid.unsafe_ptr()[i] = task_ids[i]
        for i in range(Self.H * BB * Self.MAX_ACT):
            ha.unsafe_ptr()[i] = act[i]
        for i in range(Self.H * BB):
            hr.unsafe_ptr()[i] = rew[i]
            htd.unsafe_ptr()[i] = td[i]
            hdone.unsafe_ptr()[i] = done[i]
        ctx.enqueue_copy(d_obs, ho)
        ctx.enqueue_copy(d_tids, htid)
        ctx.enqueue_copy(d_act, ha)
        ctx.enqueue_copy(d_rew, hr)
        ctx.enqueue_copy(d_td, htd)
        ctx.enqueue_copy(d_done, hdone)
        d_acc.enqueue_fill(0.0)

        # gather embeddings + build augmented obs on device.
        task_emb.gather[target, BB](_dp(d_tids), _dp(d_tem), ctx=ctx)
        comptime H1B = (Self.H + 1) * BB
        comptime oaug_k = _build_oaug_k[H1B, MO, EMB, BB]
        comptime nb_oaug = (H1B * AOBS + TPB - 1) // TPB
        ctx.enqueue_function[oaug_k](
            _lt[H1B * MO](_dp(d_obs)), _lt[BB * EMB](_dp(d_tem)),
            _lt[H1B * AOBS](_dp(d_oaug)), grid_dim=nb_oaug, block_dim=TPB,
        )

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

        # 1. consistency targets enc(obs[t+1]).
        for t in range(Self.H):
            var dst = _dp(d_zen) + t * BB * LAT
            var dst_t = TileTensor(dst, row_major[BB, LAT]())
            enc.forward[target, BB](
                TileTensor(
                    _dp(d_oaug) + (t + 1) * BB * AOBS, row_major[BB, AOBS]()
                ),
                output=dst_t,
            )
        # 2. z_0 = enc(obs[0]).
        var z0_t = TileTensor(_dp(d_carry), row_major[BB, LAT]())
        enc.forward[target, BB](
            TileTensor(_dp(d_oaug), row_major[BB, AOBS]()), output=z0_t,
        )

        # 3. forward scan.
        var rho_t = Scalar[DT](1.0)
        for t in range(Self.H):
            self._set_step_inputs[target](
                graph, _dp(d_carry), _dp(d_tem), _dp(d_zen), _dp(d_act),
                _dp(d_rew), _dp(d_td), _dp(d_done), t,
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

        # 4. zero grads + gz.
        enc_opt.zero_grad[target, Self.EncT](enc)
        dyn_opt.zero_grad[target, Self.DynT](dyn)
        rew_opt.zero_grad[target, Self.RewT](rew_net)
        for i in range(NQ):
            q_opt[i].zero_grad[target, Self.QNetT](q[i])
        term_opt.zero_grad[target, Self.TermT](term_net)
        d_gz.enqueue_fill(0.0)

        # 5. reverse-scan BPTT.
        var rho_rev = Scalar[DT](1.0)
        for _ in range(Self.H - 1):
            rho_rev *= self.rho
        for rev in range(Self.H):
            var t = Self.H - 1 - rev
            self._set_step_inputs[target](
                graph, _dp(d_carry), _dp(d_tem), _dp(d_zen), _dp(d_act),
                _dp(d_rew), _dp(d_td), _dp(d_done), t,
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
            # site 1: accumulate task_emb input-slot grad into the table.
            task_emb.accumulate[target, BB, EMB, 0](
                _dp(d_tids), graph.grad_input_ptr["task_emb"](), ctx=ctx
            )
            rho_rev /= self.rho

        # 6. encoder backward → d_gobs [B, AOBS].
        var z0r_t = TileTensor(_dp(d_carry), row_major[BB, LAT]())
        enc.forward[target, BB](
            TileTensor(_dp(d_oaug), row_major[BB, AOBS]()), output=z0r_t,
        )
        var gobs_t = TileTensor(_dp(d_gobs), row_major[BB, AOBS]())
        enc.vjp[target, BB](
            TileTensor(_dp(d_gz), row_major[BB, LAT]()), gobs_t,
        )
        # site 2: accumulate encoder input-grad tail.
        task_emb.accumulate[target, BB, AOBS, MO](
            _dp(d_tids), _dp(d_gobs), ctx=ctx
        )

        # 7. optimizer steps.
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
        tem: UnsafePointer[Scalar[DT], MutAnyOrigin],
        zen: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],
        td: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],
        t: Int,
    ) raises:
        comptime LAT = Self.LATENT
        comptime A = Self.MAX_ACT
        comptime EMB = Self.TASK_EMB
        graph.set_input["z", Self.B](
            TileTensor(carry + t * Self.B * LAT, row_major[Self.B, LAT]())
        )
        graph.set_input["a", Self.B](
            TileTensor(act + t * Self.B * A, row_major[Self.B, A]())
        )
        graph.set_input["task_emb", Self.B](
            TileTensor(tem, row_major[Self.B, EMB]())
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
