"""NoisyLinear storage — reference-noise math gate (CPU + GPU).

NoisyLinear draws fresh factorized noise each forward, so a legacy↔storage
bit-identity test is impossible across RNG. Instead: run forward, read back the
ACTUAL noise the leaf drew (ε_in/ε_out), and verify the deterministic math —
materialize W_eff/b_eff, output, all 4 param grads (µ_W/σ_W/µ_b/σ_b) and
grad_input — against an independent reference computed from that noise. This
gates the porting risk (the math/plumbing), which is what can break; the noise
SAMPLER (host Box-Muller / shared Philox LT kernel) is carried over verbatim.

Run: pixi run -e apple mojo run -I . tests/nn/test_noisy_linear_storage_parity.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.noisy_linear import NoisyLinear


comptime IN = 5
comptime OUT = 4
comptime B = 6


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](2e-4)
    var nl = NoisyLinear[IN, OUT].make_cpu() if target == "cpu" else NoisyLinear[IN, OUT].make_gpu(ctx.value())
    # deterministic params (override the default init)
    for k in range(IN * OUT):
        nl.mu_w.val.data[k] = Scalar[DT]((k % 9) - 4) * 0.05
        nl.sigma_w.val.data[k] = Scalar[DT](0.02 + 0.003 * Float64(k % 5))
    for k in range(OUT):
        nl.mu_b.val.data[k] = Scalar[DT](k + 1) * 0.1
        nl.sigma_b.val.data[k] = Scalar[DT](0.05 + 0.01 * Float64(k))

    var x = Tensor.alloc(B * IN)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        x.data[i] = Scalar[DT]((i % 11) - 5) * 0.13
    for i in range(B * OUT):
        go.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    var out = Tensor.alloc(B * OUT)
    var gi = Tensor.alloc(B * IN)

    comptime if target == "cpu":
        nl.forward["cpu", B](TensorRefs[1].of1(x), out, None)
        nl.zero_grad["cpu"](None)
        nl.vjp["cpu", B](TensorRefs[1].of1(x), go, TensorRefs[1].of1(gi), None)
    else:
        var c = ctx.value()
        nl.mu_w.val.upload(c); nl.sigma_w.val.upload(c)
        nl.mu_b.val.upload(c); nl.sigma_b.val.upload(c)
        x.upload(c); go.upload(c)
        nl.forward["gpu", B](TensorRefs[1].of1(x), out, ctx)
        nl.zero_grad["gpu"](ctx)
        nl.vjp["gpu", B](TensorRefs[1].of1(x), go, TensorRefs[1].of1(gi), ctx)
        out.download(c); gi.download(c)
        nl.noise_in.download(c); nl.noise_out.download(c)
        nl.mu_w.grd.download(c); nl.sigma_w.grd.download(c)
        nl.mu_b.grd.download(c); nl.sigma_b.grd.download(c)

    # ---- reference from the ACTUAL drawn noise ----
    ref ni = nl.noise_in.data
    ref no = nl.noise_out.data
    # W_eff / b_eff
    var we = List[Scalar[DT]](length=IN * OUT, fill=Scalar[DT](0))
    for i in range(IN):
        for j in range(OUT):
            we[i * OUT + j] = nl.mu_w.val.data[i * OUT + j] + nl.sigma_w.val.data[
                i * OUT + j
            ] * ni[i] * no[j]
    var ok = True
    # forward
    for b in range(B):
        for j in range(OUT):
            var acc = nl.mu_b.val.data[j] + nl.sigma_b.val.data[j] * no[j]
            for i in range(IN):
                acc += x.data[b * IN + i] * we[i * OUT + j]
            if abs(out.data[b * OUT + j] - acc) > TOL:
                ok = False
    # grads
    for j in range(OUT):
        var sb: Scalar[DT] = 0
        for b in range(B):
            sb += go.data[b * OUT + j]
        if abs(nl.mu_b.grd.data[j] - sb) > TOL: ok = False
        if abs(nl.sigma_b.grd.data[j] - sb * no[j]) > TOL: ok = False
    for i in range(IN):
        for j in range(OUT):
            var dw: Scalar[DT] = 0
            for b in range(B):
                dw += x.data[b * IN + i] * go.data[b * OUT + j]
            if abs(nl.mu_w.grd.data[i * OUT + j] - dw) > TOL: ok = False
            if abs(nl.sigma_w.grd.data[i * OUT + j] - dw * ni[i] * no[j]) > TOL:
                ok = False
    for b in range(B):
        for i in range(IN):
            var gx: Scalar[DT] = 0
            for j in range(OUT):
                gx += go.data[b * OUT + j] * we[i * OUT + j]
            if abs(gi.data[b * IN + i] - gx) > TOL: ok = False
    return ok


def main() raises:
    print("=" * 70)
    print("NoisyLinear storage reference-noise math gate")
    print("=" * 70)
    var ok_cpu = _check["cpu"](None)
    print("  CPU:", "OK" if ok_cpu else "FAIL")
    var c = DeviceContext()
    var ok_gpu = _check["gpu"](Optional(c))
    print("  GPU:", "OK" if ok_gpu else "FAIL")
    if ok_cpu and ok_gpu:
        print("NOISY LINEAR MATH OK")
    else:
        print("NOISY LINEAR MATH FAIL")
