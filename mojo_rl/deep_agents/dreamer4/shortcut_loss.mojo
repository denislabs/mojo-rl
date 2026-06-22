"""Shortcut-forcing flow-matching loss (train_dynamics.py:dynamics_pretrain_loss).

Dreamer 4's dynamics is trained with *shortcut forcing* (paper eq. 7): a
flow-matching x-prediction objective plus a two-step self-consistency
(bootstrap) term that lets the model take large ODE steps at inference.

Per training step, on packed clean latents z1 (B,T,Sz,Dz):
  • split the batch B into B_emp empirical rows (finest step e_max) + B_self
    self rows (random coarser power-of-two step d);
  • corrupt z̃ = (1−σ)·z0 + σ·z1  (z0 ~ N(0,1)), σ sampled on the step grid;
  • **empirical flow loss**  ‖ẑ1 − z1‖² · w(σ),  w(σ)=0.9σ+0.1  (emp rows);
  • **bootstrap loss**  (1−σ)²·‖v̂ − v̄‖² · w(σ)  (self rows), where
      v̂  = (ẑ1 − z̃)/(1−σ)            (from the MAIN forward)
      v̄  = sg((b′ + b″)/2)            (stop-grad target, two half-step passes)
      b′  = (ẑ1_half1 − z̃)/(1−σ),  z′ = z̃ + b′·d/2
      b″  = (ẑ1_half2 − z′)/(1−(σ+d/2))
  • combine  L = (loss_emp·B_emp + loss_self·B_self)/B.

Because v̄ is detached, **gradients flow only through the MAIN forward** — the
two half-step passes are forward-only (they just produce the target). This
lets us run the half passes FIRST (they clobber the module's activation
caches), then the main forward LAST, so the caller's `dyn.vjp(grad_zhat)`
sees the main pass's caches intact.

CONTRACT: `dynamics_pretrain_loss(...)` runs all forwards and returns the
scalar loss + fills `grad_zhat` (= dL/dẑ for the main pass). The caller must
then call `dyn.vjp["cpu", B*T](grad_zhat)` with **nothing else touching `dyn`
in between**, followed by `optim.step`.

Sampling (σ, step indices, z0 noise) is the CALLER's job and is passed in as
host buffers — this keeps the loss pure/deterministic so it can be checked
against a PyTorch fixture to ≈1e-5.

STORAGE: the dynamics forward goes through the storage `Module` surface
(`dyn.forward[target, BATCH](TensorRefs[1](in_t), out_t, ctx)`). `_run_fwd` bridges
the loss's host-scratch buffers to two boundary `Tensor`s — on GPU it uploads
the input + downloads the output (the transformer's heavy compute runs on
device, the element-wise loss arithmetic stays on host, bit-identical to CPU).
"""

from std.memory import alloc
from std.math import max
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs


def _ilog2(n: Int) -> Int:
    var k = 0
    var v = n
    while v > 1:
        v //= 2
        k += 1
    return k


# Erase a host-scratch pointer's origin to MutAnyOrigin. The dynamics control
# inputs (`set_indices`/`set_actions`/`set_agent_in`) take `MutAnyOrigin` host
# pointers by contract (a deliberate raw-pointer boundary for host control I/O);
# the loss's small index/host buffers are passed through here. This is the local
# replacement for the legacy `nn.core.module.mptr` — no legacy-framework import.
def _mao[
    dt: DType, o: Origin
](p: UnsafePointer[Scalar[dt], o]) -> UnsafePointer[Scalar[dt], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[dt], MutAnyOrigin]](p)


trait ShortcutDynamics(Module):
    """A dynamics module whose per-sample signal/step indices are pushed via
    `set_indices` before each forward (Dreamer4Dynamics). Optionally also
    accepts per-sample actions via `set_actions` for conditioned (labeled)
    dynamics pretrain — modules without action conditioning inherit the no-op
    default."""

    def set_indices(
        mut self,
        sig: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step: UnsafePointer[Scalar[DT], MutAnyOrigin],
        batch: Int,
    ):
        ...

    def set_actions(
        mut self,
        actions: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act_mask: UnsafePointer[Scalar[DT], MutAnyOrigin],
        batch: Int,
    ):
        """Default: unconditional (ignore actions). Conditioned dynamics
        (Dreamer4Dynamics with ADIM>0) override this."""
        pass

    def set_agent_in(
        mut self,
        agent_in: UnsafePointer[Scalar[DT], MutAnyOrigin],
        batch: Int,
    ):
        """Default: no agent tokens (ignore). Agent-capable dynamics
        (Dreamer4Dynamics with NAGENT>0) override this — used by the BC agent
        to inject the per-sequence task embedding into every forward so the
        MAIN pass produces the task output embeddings h_t."""
        pass


trait AgentDynamics(ShortcutDynamics):
    """A `ShortcutDynamics` that also exposes the agent task-output embeddings
    h_t after a forward (Dreamer4Dynamics). Used by the imagination rollout
    (`imag_rollout.mojo`), which reads h_t per state to drive the policy /
    value / reward heads."""

    def agent_out_ptr_cpu(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        ...

    def agent_out_dev(self) -> DeviceBuffer[DT]:
        ...


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return _mao(alloc[Scalar[DT]](n))


def _run_fwd[
    M: ShortcutDynamics, FWD: StaticString, BATCH: Int, ND: Int
](
    mut dyn: M,
    in_host: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_host: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ctx: Optional[DeviceContext],
) raises:
    """Run one dynamics forward through the storage Module surface. Copies the
    host input into a boundary `Tensor`, runs `dyn.forward[FWD, BATCH]`, and
    copies the output `Tensor` back to `out_host`. FWD="cpu": the forward fills
    `out_t.data` directly. FWD="gpu": upload input → device forward → download
    output. Either way the loss's element-wise arithmetic stays on host (tiny
    latent buffers), so the GPU path is bit-identical to CPU."""
    comptime N = BATCH * ND
    var in_t = Tensor.alloc(N)
    for i in range(N):
        in_t.data[i] = in_host[i]
    comptime if FWD == "cpu":
        var out_t = Tensor.alloc(N)
        dyn.forward["cpu", BATCH](TensorRefs[M.ARITY](in_t), out_t, None)
        for i in range(N):
            out_host[i] = out_t.data[i]
    else:
        var c = ctx.value()
        in_t.upload(c)
        var out_t = Tensor.alloc_gpu(c, N)
        dyn.forward["gpu", BATCH](TensorRefs[M.ARITY](in_t), out_t, ctx)
        out_t.download(c)
        for i in range(N):
            out_host[i] = out_t.data[i]


def dynamics_pretrain_loss[
    M: ShortcutDynamics,
    B: Int,
    T: Int,
    B_SELF: Int,
    NSP: Int,
    DSP: Int,
    KMAX: Int,
    FWD: StaticString = "cpu",
    ADIM: Int = 0,
    AGDIM: Int = 0,
](
    mut dyn: M,
    z1: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BF, ND] clean targets
    z0: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BF, ND] noise
    sigma: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BF] signal level
    sigma_idx: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BF] signal index
    step_idx: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BF] step index
    do_boot: Bool,
    grad_zhat: UnsafePointer[Scalar[DT], MutAnyOrigin],  # OUT [BF, ND]  dL/dẑ
    zhat: UnsafePointer[Scalar[DT], MutAnyOrigin],  # OUT [BF, ND] main pred
    ctx: Optional[DeviceContext] = None,  # FWD="gpu" only
    actions: Optional[
        UnsafePointer[Scalar[DT], MutAnyOrigin]
    ] = None,  # [BF,ADIM]
    act_mask: Optional[
        UnsafePointer[Scalar[DT], MutAnyOrigin]
    ] = None,  # [ADIM]
    agent_in: Optional[
        UnsafePointer[Scalar[DT], MutAnyOrigin]
    ] = None,  # [BF,AGDIM]
) raises -> Float64:
    """Shortcut-forcing loss. FWD="cpu" (default): all on host. FWD="gpu": the
    dynamics forwards run on device (caller provides `ctx`); the element-wise
    loss arithmetic stays on host (tiny latent buffers), so the GPU path is
    bit-identical to CPU.

    CONDITIONED pretrain: with `ADIM>0` and `actions` (+`act_mask`) provided,
    each forward is preceded by `dyn.set_actions(...)` — the two half passes
    use the self-row subset (matching the reference's `actions[B_emp:]`), the
    main pass the full batch. `ADIM=0` / `actions=None` is unconditional and
    byte-for-byte the original loss."""
    comptime assert FWD == "cpu" or FWD == "gpu", "FWD must be 'cpu' or 'gpu'"
    comptime BF = B * T
    comptime BS = B_SELF * T
    comptime B_EMP = B - B_SELF
    comptime ND = NSP * DSP
    comptime EMAX = _ilog2(KMAX)
    comptime COND = ADIM > 0
    comptime AGCOND = AGDIM > 0

    # ── corrupt: z̃ = (1−σ)·z0 + σ·z1 ────────────────────────────────────
    var ztil = _alloc(BF * ND)
    for bt in range(BF):
        var s = Float64(sigma[bt])
        for i in range(ND):
            var idx = bt * ND + i
            ztil[idx] = Scalar[DT](
                (1.0 - s) * Float64(z0[idx]) + s * Float64(z1[idx])
            )

    # ── bootstrap target v̄ (forward-only; clobbers caches → run FIRST) ──
    var vbar = _alloc(BS * ND)  # detached target
    var zts = _alloc(BS * ND)  # z̃ self subset (reused as v̂ base)
    # self rows are the last BS rows (b ∈ [B_EMP, B))
    var SELF0 = B_EMP * T
    for j in range(BS):
        for i in range(ND):
            zts[j * ND + i] = ztil[(SELF0 + j) * ND + i]

    if do_boot and B_SELF > 0:
        var sig_self = _alloc(BS)
        var step_half = _alloc(BS)
        var sig_plus = _alloc(BS)
        var zprime = _alloc(BS * ND)
        var zh1 = _alloc(BS * ND)
        var zh2 = _alloc(BS * ND)
        var d_half = List[Float64]()
        d_half.resize(BS, 0.0)
        var sig_self_val = List[Float64]()
        sig_self_val.resize(BS, 0.0)
        var sig_plus_val = List[Float64]()
        sig_plus_val.resize(BS, 0.0)

        for j in range(BS):
            var sb = SELF0 + j
            var stp = Int(Float64(step_idx[sb]) + 0.5)
            var dd = 1.0 / Float64(1 << stp)  # d = 1/2^step
            var dh = dd / 2.0
            d_half[j] = dh
            sig_self[j] = sigma_idx[sb]  # σ_idx (unchanged)
            step_half[j] = Scalar[DT](Float64(stp + 1))
            sig_self_val[j] = Float64(sigma[sb])
            sig_plus_val[j] = sig_self_val[j] + dh
            sig_plus[j] = Scalar[DT](
                Float64(sig_self[j]) + Float64(Int(Float64(KMAX) * dh))
            )

        # half1: ẑ1_half1 = dyn(z̃_self; σ_idx, step+1) ; b′ ; z′
        # (conditioned: self-row actions = the reference's actions[B_emp:])
        dyn.set_indices(sig_self, step_half, BS)
        comptime if COND:
            if actions:
                dyn.set_actions(
                    _mao(actions.value() + (B_EMP * T) * ADIM),
                    act_mask.value(),
                    BS,
                )
        comptime if AGCOND:
            if agent_in:
                dyn.set_agent_in(
                    _mao(agent_in.value() + (B_EMP * T) * AGDIM), BS
                )
        _run_fwd[M, FWD, BS, ND](dyn, zts, zh1, ctx)
        for j in range(BS):
            var denom = max(1.0 - sig_self_val[j], 1e-6)
            for i in range(ND):
                var idx = j * ND + i
                var bp = (Float64(zh1[idx]) - Float64(zts[idx])) / denom
                zprime[idx] = Scalar[DT](Float64(zts[idx]) + bp * d_half[j])
                vbar[idx] = Scalar[DT](bp)  # vbar := b′ (add b″ next)

        # half2: ẑ1_half2 = dyn(z′; σ_idx+Δ, step+1) ; b″ ; v̄ = (b′+b″)/2
        dyn.set_indices(sig_plus, step_half, BS)
        comptime if COND:
            if actions:
                dyn.set_actions(
                    _mao(actions.value() + (B_EMP * T) * ADIM),
                    act_mask.value(),
                    BS,
                )
        comptime if AGCOND:
            if agent_in:
                dyn.set_agent_in(
                    _mao(agent_in.value() + (B_EMP * T) * AGDIM), BS
                )
        _run_fwd[M, FWD, BS, ND](dyn, zprime, zh2, ctx)
        for j in range(BS):
            var denom = max(1.0 - sig_plus_val[j], 1e-6)
            for i in range(ND):
                var idx = j * ND + i
                var bpp = (Float64(zh2[idx]) - Float64(zprime[idx])) / denom
                vbar[idx] = Scalar[DT]((Float64(vbar[idx]) + bpp) / 2.0)

        zh1.free()
        zh2.free()
        zprime.free()
        sig_self.free()
        step_half.free()
        sig_plus.free()

    # ── MAIN forward (LAST forward → caches valid for the caller's vjp) ──
    dyn.set_indices(sigma_idx, step_idx, BF)
    comptime if COND:
        if actions:
            dyn.set_actions(actions.value(), act_mask.value(), BF)
    comptime if AGCOND:
        if agent_in:
            dyn.set_agent_in(agent_in.value(), BF)
    _run_fwd[M, FWD, BF, ND](dyn, ztil, zhat, ctx)

    # ── losses + grad_zhat (= dL/dẑ for the main pass) ──────────────────
    # Both terms share a 1/(B*T) prefactor after the (B_emp/B)·(1/(B_emp·T))
    # and (B_self/B)·(1/(B_self·T)) means collapse.
    var loss_emp: Float64 = 0.0
    var loss_self: Float64 = 0.0
    var pref = 1.0 / Float64(BF)

    for i in range(BF * ND):
        grad_zhat[i] = Scalar[DT](0.0)

    # empirical flow loss over emp rows
    for bt in range(B_EMP * T):
        var w_emp = 0.9 * Float64(sigma[bt]) + 0.1
        var sse: Float64 = 0.0
        for i in range(ND):
            var idx = bt * ND + i
            var diff = Float64(zhat[idx]) - Float64(z1[idx])
            sse += diff * diff
            grad_zhat[idx] = Scalar[DT](
                pref * w_emp * (2.0 / Float64(ND)) * diff
            )
        var flow_per = sse / Float64(ND)
        loss_emp += w_emp * flow_per
    loss_emp /= Float64(B_EMP * T)

    # bootstrap loss over self rows
    if do_boot and B_SELF > 0:
        for j in range(BS):
            var sb = SELF0 + j
            var sv = Float64(sigma[sb])
            var w_self = 0.9 * sv + 0.1
            var denom = max(1.0 - sv, 1e-6)
            var one_minus = 1.0 - sv
            var sse: Float64 = 0.0
            for i in range(ND):
                var idx = j * ND + i  # self subset index
                var midx = sb * ND + i  # main-pass index
                var vhat = (Float64(zhat[midx]) - Float64(zts[idx])) / denom
                var d = vhat - Float64(vbar[idx])
                sse += d * d
                grad_zhat[midx] = Scalar[DT](
                    pref
                    * w_self
                    * (one_minus * one_minus)
                    * (2.0 / Float64(ND))
                    * d
                    / denom
                )
            var boot_per = (one_minus * one_minus) * (sse / Float64(ND))
            loss_self += w_self * boot_per
        loss_self /= Float64(BS)

    var loss = (
        loss_emp * Float64(B_EMP) + loss_self * Float64(B_SELF)
    ) / Float64(B)

    ztil.free()
    vbar.free()
    zts.free()
    return loss
