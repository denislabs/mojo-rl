"""Dreamer 4 imagination rollout (Phase 4 / paper §3.3 RL).

Generates an on-policy trajectory *inside the world model* — no environment
interaction. The transformer (dynamics) is FROZEN; this is pure forward
inference used to produce states/actions/rewards/values that the policy and
value heads then train on (`imag_rl_loss.mojo`).

Rollout (one window of length T, NCTX clean context frames from the dataset,
all B sequences in parallel):

  for the acting state i = NCTX−1 .. T−2  (generating frame tgt = i+1):
    1. forward the dynamics over the window (frames 0..i clean, future = 0;
       causal time attention ⇒ future frames don't affect h_i) → read the
       agent token h_i;
    2. policy head(h_i) → categorical logits → SAMPLE action a_i (the action
       token of frame tgt — by the dynamics' convention the action token at a
       frame conditions the transition INTO it, so a_i = "action taken at
       state i leading to frame i+1");
    3. reward head(h_i) dist-0 → r_i ; value head(h_i) → v_i  (annotation);
    4. ODE-denoise frame tgt over K flow-matching steps conditioned on a_i and
       the clean context 0..i → the next latent z_{i+1};
    5. mark frame tgt clean (σ_idx = KMAX−1) for the next iteration.
  finally read h_{T−1} → r_{T−1}, v_{T−1} (the bootstrap state, no action).

Outputs (caller-owned):
  • out_h   [B,T,AGD]   agent tokens (clean) per state
  • out_act [B,T-1]     sampled action class at state i (i = 0..T-2)
  • out_rew [B,T]       reward-head dist-0 prediction per state
  • out_val [B,T]       value-head prediction per state

Sampling is the CALLER's job (deterministic): `u01[B,T]` uniforms for the
categorical action sample and `z_noise[B,T,ND]` for each frame's τ=0 ODE seed.
`agent_in[B*T,AGD]` is the (already broadcast) task embedding. CPU; the
transformer forward runs on whatever target the module was built for.

STORAGE: the dynamics + head forwards go through the storage `Module` surface.
`_fwd_window` bridges the host-scratch `packed`/`zhat` buffers to two boundary
`Tensor`s (mirror `shortcut_loss._run_fwd`): FWD="cpu" fills `out_t.data`
directly; FWD="gpu" uploads `in_t` → device forward → downloads `out_t`. The
agent tokens h_t come from `dyn.agent_out_ptr_cpu()` (CPU) or a D2H copy of
`dyn.agent_out_dev()` into a host staging `HostBuffer` (GPU). The head outputs
feed the raw-pointer host helpers (`cat_sample`/`twohot_pred`), so the head I/O
stays raw scratch and is bridged at the head forward the same way.

NOTE: this requires an action-conditioned, agent-capable dynamics
(`Dreamer4Dynamics` with ADIM = NACT one-hot and NAGENT > 0).
"""

from std.math import max
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.call import call_forward, call_vjp
from .shortcut_loss import _dyn_fwd, ShortcutDynamics, AgentDynamics, _ilog2, _mao
from ..dreamerv3.dists_discrete import cat_sample, UNIMIX
from ..dreamerv3.twohot import twohot_pred


# Module-level helpers (nested `def`s can't capture mutable outer state in
# Mojo nightly — "could not infer capture convention"; pass everything in).
#
# One dynamics forward over the window. The frozen transformer is the heavy
# compute, so FWD="gpu" runs it on device (upload packed → forward[gpu] →
# download zhat + the agent tokens h); the small head + loss arithmetic stays on
# host — the same split as `shortcut_loss._run_fwd`. Either way `h_host` is
# filled with h_t [BF, AGD] for the host-side heads.
def _fwd_window[
    M: AgentDynamics, FWD: StaticString, BF: Int, ND: Int, AGD: Int
](
    mut dyn: M,
    sig_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    step_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    act_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mask_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    agent_in: UnsafePointer[Scalar[DT], MutAnyOrigin],
    packed_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    zhat_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    h_host: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mut in_t: Tensor,
    mut out_t: Tensor,
    ctx: Optional[DeviceContext],
    h_ag: Optional[HostBuffer[DT]],
) raises:
    dyn.set_indices(sig_p, step_p, BF)
    dyn.set_actions(act_p, mask_p, BF)
    dyn.set_agent_in(agent_in, BF)
    # copy host-scratch `packed` into the boundary input Tensor
    for i in range(BF * ND):
        in_t.data[i] = packed_p[i]
    comptime if FWD == "cpu":
        _dyn_fwd["cpu", BF](dyn, TensorRefs[M.ARITY](in_t), out_t, None)
        for i in range(BF * ND):
            zhat_p[i] = out_t.data[i]
        var ao = dyn.agent_out_ptr_cpu()
        for i in range(BF * AGD):
            h_host[i] = ao[i]
    else:
        var c = ctx.value()
        in_t.upload(c)
        _dyn_fwd["gpu", BF](dyn, TensorRefs[M.ARITY](in_t), out_t, ctx)
        out_t.download(c)
        for i in range(BF * ND):
            zhat_p[i] = out_t.data[i]
        # agent tokens h_t: D2H copy of dyn.agent_out_dev() into host staging
        var hb = h_ag.value()
        c.enqueue_copy(hb, dyn.agent_out_dev())
        c.synchronize()
        for i in range(BF * AGD):
            h_host[i] = hb.unsafe_ptr()[i]


def _annotate[
    PH: Module, VH: Module, RH: Module,
    B: Int, T: Int, AGD: Int, PLOG: Int, NBINS: Int, RLOG: Int,
](
    mut ph: PH, mut vh: VH, mut rh: RH,
    state_i: Int,
    h_host: UnsafePointer[Scalar[DT], MutAnyOrigin],
    hg: UnsafePointer[Scalar[DT], MutAnyOrigin],
    pl: UnsafePointer[Scalar[DT], MutAnyOrigin],
    vl: UnsafePointer[Scalar[DT], MutAnyOrigin],
    rl: UnsafePointer[Scalar[DT], MutAnyOrigin],
    bins: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_h: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_rew: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_val: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mut hin_t: Tensor,
    mut plout_t: Tensor,
    mut vlout_t: Tensor,
    mut rlout_t: Tensor,
) raises:
    """Gather h_state_i (from the host-side `h_host` produced by `_fwd_window`)
    for all B, run the heads, store out_h/out_rew/out_val and leave the policy
    logits in `pl` for sampling. The heads are storage Sequential Modules: copy
    the gathered h into a boundary input `Tensor`, run the forward, then copy the
    output `Tensor` back into the raw `pl`/`vl`/`rl` scratch the host helpers
    (`twohot_pred`/`cat_sample`) index."""
    for b in range(B):
        var src = (b * T + state_i) * AGD
        for k in range(AGD):
            hg[b * AGD + k] = h_host[src + k]
            out_h[(b * T + state_i) * AGD + k] = h_host[src + k]
    # bridge gathered h → boundary input Tensor (heads run CPU)
    for i in range(B * AGD):
        hin_t.data[i] = hg[i]
    call_forward["cpu", B](ph, TensorRefs[PH.ARITY](hin_t), plout_t, None)
    call_forward["cpu", B](vh, TensorRefs[VH.ARITY](hin_t), vlout_t, None)
    call_forward["cpu", B](rh, TensorRefs[RH.ARITY](hin_t), rlout_t, None)
    # copy head outputs back into the raw scratch the host helpers index
    for i in range(B * PLOG):
        pl[i] = plout_t.data[i]
    for i in range(B * NBINS):
        vl[i] = vlout_t.data[i]
    for i in range(B * RLOG):
        rl[i] = rlout_t.data[i]
    for b in range(B):
        out_val[b * T + state_i] = twohot_pred[NBINS](vl, b * NBINS, bins)
        out_rew[b * T + state_i] = twohot_pred[NBINS](rl, b * RLOG, bins)


def imagine_rollout[
    M: AgentDynamics,
    PH: Module,
    VH: Module,
    RH: Module,
    B: Int, T: Int, NSP: Int, DSP: Int, KMAX: Int, K: Int, NCTX: Int,
    AGD: Int, NACT: Int, NBINS: Int, NMTP: Int,
    FWD: StaticString = "cpu",
](
    mut dyn: M,
    mut ph: PH,
    mut vh: VH,
    mut rh: RH,
    ctx: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [B, NCTX, ND] clean
    agent_in: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B*T, AGD] task embed
    u01: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [B, T] action uniforms
    z_noise: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, T, ND] ODE τ=0 seeds
    bins: UnsafePointer[Scalar[DT], MutAnyOrigin],      # [NBINS]
    out_h: UnsafePointer[Scalar[DT], MutAnyOrigin],     # OUT [B, T, AGD]
    out_act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # OUT [B, T-1] class ids
    out_rew: UnsafePointer[Scalar[DT], MutAnyOrigin],   # OUT [B, T]
    out_val: UnsafePointer[Scalar[DT], MutAnyOrigin],   # OUT [B, T]
    dctx: Optional[DeviceContext] = None,               # FWD="gpu": the device
) raises:
    """FWD="gpu" runs the (frozen) dynamics transformer forward on `dctx` — the
    heavy compute — while the heads and all rollout orchestration stay on host
    (the dynamics is forward-only here, so no GPU vjp is needed). The heads are
    plain host modules either way. CPU↔GPU parity is validated in
    `test_dreamer4_imag_rollout_gpu`."""
    comptime assert FWD == "cpu" or FWD == "gpu", "FWD must be 'cpu' or 'gpu'"
    comptime ND = NSP * DSP
    comptime BF = B * T
    comptime EMAX = _ilog2(KMAX)
    comptime E = _ilog2(K)
    comptime SCALE = KMAX // K
    comptime ADIM = NACT                       # one-hot discrete action
    comptime PLOG = NMTP * NACT
    comptime RLOG = NMTP * NBINS
    var dt = 1.0 / Float64(K)

    # ── window buffers ──────────────────────────────────────────────────
    var packed = List[Scalar[DT]]()
    packed.resize(BF * ND, 0.0)
    var zhat = List[Scalar[DT]]()
    zhat.resize(BF * ND, 0.0)
    var h_host = List[Scalar[DT]]()            # h_t [BF, AGD] (host mirror)
    h_host.resize(BF * AGD, 0.0)

    # boundary Tensors for the dynamics forward (mirror shortcut_loss._run_fwd):
    # `_fwd_window` writes host `.data` then (gpu) uploads / downloads, so these
    # need BOTH a host buffer AND (gpu) a device one — `Tensor.make["gpu"]` is
    # device-ONLY, so alloc host then add the device buffer for the gpu path.
    var in_t = Tensor.alloc(BF * ND)
    var out_t = Tensor.alloc(BF * ND)
    # agent-out D2H staging (GPU only)
    var h_ag = Optional[HostBuffer[DT]](None)
    comptime if FWD == "gpu":
        var c = dctx.value()
        in_t.ensure_gpu(c, BF * ND)
        out_t.ensure_gpu(c, BF * ND)
        h_ag = c.enqueue_create_host_buffer[DT](BF * AGD)

    # boundary Tensors for the heads (always CPU)
    var hin_t = Tensor.alloc(B * AGD)
    var plout_t = Tensor.alloc(B * PLOG)
    var vlout_t = Tensor.alloc(B * NBINS)
    var rlout_t = Tensor.alloc(B * RLOG)

    var act_oh = List[Scalar[DT]]()
    act_oh.resize(BF * ADIM, 0.0)
    var act_mask = List[Scalar[DT]]()
    act_mask.resize(BF * ADIM, 1.0)             # all-ones (no masking)
    var sig = List[Scalar[DT]]()
    sig.resize(BF, 0.0)
    var step = List[Scalar[DT]]()
    step.resize(BF, Scalar[DT](Float64(EMAX)))

    # head I/O scratch (batch B, one state at a time)
    var hgather = List[Scalar[DT]]()
    hgather.resize(B * AGD, 0.0)
    var plog = List[Scalar[DT]]()
    plog.resize(B * PLOG, 0.0)
    var vlog = List[Scalar[DT]]()
    vlog.resize(B * NBINS, 0.0)
    var rlog = List[Scalar[DT]]()
    rlog.resize(B * RLOG, 0.0)

    # context frames 0..NCTX-1 clean; rest start at 0
    for b in range(B):
        for c in range(NCTX):
            var bt = b * T + c
            sig[bt] = Scalar[DT](Float64(KMAX - 1))
            for i in range(ND):
                packed[bt * ND + i] = ctx[(b * NCTX + c) * ND + i]

    # ── autoregressive generation ───────────────────────────────────────
    for tgt in range(NCTX, T):
        var state_i = tgt - 1
        # 1+3. read h_state_i (frames 0..state_i clean) and annotate r/v
        _fwd_window[M, FWD, BF, ND, AGD](
            dyn, _mao(sig.unsafe_ptr()), _mao(step.unsafe_ptr()),
            _mao(act_oh.unsafe_ptr()), _mao(act_mask.unsafe_ptr()), agent_in,
            _mao(packed.unsafe_ptr()), _mao(zhat.unsafe_ptr()),
            _mao(h_host.unsafe_ptr()), in_t, out_t, dctx, h_ag,
        )
        _annotate[PH, VH, RH, B, T, AGD, PLOG, NBINS, RLOG](
            ph, vh, rh, state_i, _mao(h_host.unsafe_ptr()),
            _mao(hgather.unsafe_ptr()), _mao(plog.unsafe_ptr()),
            _mao(vlog.unsafe_ptr()), _mao(rlog.unsafe_ptr()), bins,
            out_h, out_rew, out_val,
            hin_t, plout_t, vlout_t, rlout_t,
        )
        # 2. sample action a_state_i from the dist-0 policy block
        for b in range(B):
            var k = cat_sample[NACT](
                _mao(plog.unsafe_ptr()), b * PLOG, UNIMIX, u01[b * T + state_i]
            )
            out_act[b * (T - 1) + state_i] = Scalar[DT](Float64(k))
            # set frame tgt's action token = one-hot(a_state_i)
            for a in range(ADIM):
                act_oh[(b * T + tgt) * ADIM + a] = Scalar[DT](
                    1.0 if a == k else 0.0
                )
        # 4. ODE-denoise frame tgt over K substeps (seed = z_noise[b,tgt])
        for b in range(B):
            for kk in range(ND):
                packed[(b * T + tgt) * ND + kk] = z_noise[(b * T + tgt) * ND + kk]
        for b in range(B):
            step[b * T + tgt] = Scalar[DT](Float64(E))
        for isub in range(K):
            var tau = Float64(isub) / Float64(K)
            var sig_i = isub * SCALE
            for b in range(B):
                sig[b * T + tgt] = Scalar[DT](Float64(sig_i))
            _fwd_window[M, FWD, BF, ND, AGD](
                dyn, _mao(sig.unsafe_ptr()), _mao(step.unsafe_ptr()),
                _mao(act_oh.unsafe_ptr()), _mao(act_mask.unsafe_ptr()), agent_in,
                _mao(packed.unsafe_ptr()), _mao(zhat.unsafe_ptr()),
                _mao(h_host.unsafe_ptr()), in_t, out_t, dctx, h_ag,
            )
            var denom = max(1e-4, 1.0 - tau)
            for b in range(B):
                var fb = (b * T + tgt) * ND
                for kk in range(ND):
                    var x1 = Float64(zhat[fb + kk])
                    var zv = Float64(packed[fb + kk])
                    packed[fb + kk] = Scalar[DT](zv + (x1 - zv) / denom * dt)
        # 5. frame tgt is now clean for subsequent reads
        for b in range(B):
            sig[b * T + tgt] = Scalar[DT](Float64(KMAX - 1))
            step[b * T + tgt] = Scalar[DT](Float64(EMAX))

    # ── final state T-1: read h, annotate (bootstrap value/reward) ──────
    _fwd_window[M, FWD, BF, ND, AGD](
        dyn, _mao(sig.unsafe_ptr()), _mao(step.unsafe_ptr()),
        _mao(act_oh.unsafe_ptr()), _mao(act_mask.unsafe_ptr()), agent_in,
        _mao(packed.unsafe_ptr()), _mao(zhat.unsafe_ptr()),
        _mao(h_host.unsafe_ptr()), in_t, out_t, dctx, h_ag,
    )
    _annotate[PH, VH, RH, B, T, AGD, PLOG, NBINS, RLOG](
        ph, vh, rh, T - 1, _mao(h_host.unsafe_ptr()),
        _mao(hgather.unsafe_ptr()), _mao(plog.unsafe_ptr()),
        _mao(vlog.unsafe_ptr()), _mao(rlog.unsafe_ptr()), bins,
        out_h, out_rew, out_val,
        hin_t, plout_t, vlout_t, rlout_t,
    )
