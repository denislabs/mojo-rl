"""Autoregressive ODE sampler (train_dynamics.py:sample_one_timestep_packed).

Inference for the shortcut-forcing dynamics: predict the next frame's packed
latents from a clean context window via a K-step flow-matching ODE.

Our nn dynamics has a *fixed* compile-time window T, so the sampler runs at
that window: frames `0 .. T-2` hold the clean context and frame `T-1` is the
frame being denoised. Starting from noise z at τ=0, K Euler steps integrate

    x̂1 = dyn(packed ; σ_idx, step_idx)            (x-prediction, last frame)
    b   = (x̂1 − z) / max(1e-4, 1−τ)               (flow velocity)
    z   = z + b·dt,   dt = 1/K,   τ = i/K

Conditioning indices (per the reference):
  • step_idx: every frame = e_max (= log2(KMAX)) EXCEPT the denoised last
    frame = e (= log2(K)) — only it uses the shortcut step;
  • σ_idx: context frames = KMAX−1 (treated near-clean), last frame = i·scale
    (scale = KMAX//K) over the K substeps.

K must be a power of two ≤ KMAX and divide KMAX. Pure host loop calling the
dynamics forward; returns the predicted frame [B, NSP·DSP].

STORAGE: the dynamics forward goes through the storage `Module` surface
(`dyn.forward["cpu", BF](TensorRefs[M.ARITY](in_t), out_t, None)`). The host-scratch
working buffers (`packed`/`zhat`/`z`/`sig_idx`/`step_idx`) stay `List`s; we
bridge `packed` ↔ a boundary input `Tensor` and read the output `Tensor` back
into `zhat` at each forward, mirroring `shortcut_loss._run_fwd` (CPU branch).

PHASE 2.5: CPU.
"""

from std.math import max

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from .shortcut_loss import ShortcutDynamics, _ilog2, _mao


def sample_one_timestep[
    M: ShortcutDynamics,
    B: Int, T: Int, NSP: Int, DSP: Int, KMAX: Int, K: Int,
](
    mut dyn: M,
    context: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B*(T-1), NSP*DSP] clean
    z_init: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, NSP*DSP] noise (τ=0)
    out_frame: UnsafePointer[Scalar[DT], MutAnyOrigin],  # OUT [B, NSP*DSP]
) raises:
    comptime ND = NSP * DSP
    comptime BF = B * T
    comptime EMAX = _ilog2(KMAX)
    comptime E = _ilog2(K)
    comptime SCALE = KMAX // K
    var dt = 1.0 / Float64(K)

    var packed = List[Scalar[DT]]()
    packed.resize(BF * ND, 0.0)
    var zhat = List[Scalar[DT]]()
    zhat.resize(BF * ND, 0.0)
    var sig_idx = List[Scalar[DT]]()
    sig_idx.resize(BF, 0.0)
    var step_idx = List[Scalar[DT]]()
    step_idx.resize(BF, 0.0)
    var z = List[Scalar[DT]]()        # current frame estimate [B, ND]
    z.resize(B * ND, 0.0)

    # context → packed frames 0..T-2 ; conditioning indices
    for b in range(B):
        for t in range(T):
            var bt = b * T + t
            step_idx[bt] = Scalar[DT](Float64(EMAX))
            sig_idx[bt] = Scalar[DT](Float64(KMAX - 1))
            if t < T - 1:
                for i in range(ND):
                    packed[bt * ND + i] = context[(b * (T - 1) + t) * ND + i]
        # the denoised (last) frame uses the shortcut step
        step_idx[b * T + (T - 1)] = Scalar[DT](Float64(E))
    for i in range(B * ND):
        z[i] = z_init[i]

    # Boundary tensors bridging the host-scratch buffers to the storage Module
    # surface (mirror shortcut_loss._run_fwd, CPU branch): copy `packed` into
    # `in_t.data` before each forward, run forward, read `out_t.data` → `zhat`.
    var in_t = Tensor.alloc(BF * ND)
    var out_t = Tensor.alloc(BF * ND)

    for i in range(K):
        var tau = Float64(i) / Float64(K)
        var sig_i = i * SCALE
        # write z into the last frame; set its signal index for this substep
        for b in range(B):
            var last_bt = b * T + (T - 1)
            sig_idx[last_bt] = Scalar[DT](Float64(sig_i))
            for k in range(ND):
                packed[last_bt * ND + k] = z[b * ND + k]
        dyn.set_indices(_mao(sig_idx.unsafe_ptr()), _mao(step_idx.unsafe_ptr()), BF)
        for j in range(BF * ND):
            in_t.data[j] = packed[j]
        dyn.forward["cpu", BF](TensorRefs[M.ARITY](in_t), out_t, None)
        for j in range(BF * ND):
            zhat[j] = out_t.data[j]
        var denom = max(1e-4, 1.0 - tau)
        for b in range(B):
            var last_bt = b * T + (T - 1)
            for k in range(ND):
                var x1 = Float64(zhat[last_bt * ND + k])
                var zv = Float64(z[b * ND + k])
                var bvel = (x1 - zv) / denom
                z[b * ND + k] = Scalar[DT](zv + bvel * dt)

    for i in range(B * ND):
        out_frame[i] = z[i]
