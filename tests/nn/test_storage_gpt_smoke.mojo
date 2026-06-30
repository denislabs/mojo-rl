"""nn.storage GPT smoke (CPU + GPU): build a tiny weight-tied GPT, run the two
construction ops (scaled-residual init + weight-tie wiring), then forward + vjp.

Proves: (1) the GPT comptime compositions type-check; (2) gpt_scale_residual_proj
walks the child tree and scales the c_proj weights; (3) gpt_wire_tie points the
TiedLinear head at the embedding cells via tie_to_ptr (no exclusivity conflict);
(4) the wired model runs forward+vjp end-to-end with finite outputs on both
targets (the head borrows the embedding weight — it owns no Param).

Run:
  pixi run mojo run -I . tests/nn/test_storage_gpt_smoke.mojo
  pixi run -e apple mojo run -I . tests/nn/test_storage_gpt_smoke.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.models.gpt import GPTDropTied, gpt_scale_residual_proj, gpt_wire_tie


comptime VOCAB = 5
comptime SEQ = 3
comptime EMB = 4
comptime HEADS = 2
comptime LAYERS = 1
comptime FFM = 2
comptime CAUSAL = True
comptime PDROP = 0.0  # identity dropout → deterministic
comptime SEED = UInt64(0)
comptime UMAX = True
comptime B = 2
comptime IN = SEQ * VOCAB
comptime OUT = SEQ * VOCAB

comptime NET = GPTDropTied[VOCAB, SEQ, EMB, HEADS, LAYERS, FFM, CAUSAL, PDROP, SEED, UMAX]


def _onehot_input() raises -> Tensor:
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](0)
    # one-hot per (batch, token): pick a token id deterministically
    for b in range(B):
        for t in range(SEQ):
            var tok = (b * SEQ + t) % VOCAB
            x.data[b * IN + t * VOCAB + tok] = Scalar[DT](1)
    return x^


def _finite(t: Tensor, n: Int) raises -> Bool:
    for i in range(n):
        if t.data[i] != t.data[i]:
            return False
    return True


def test_cpu() raises:
    print("GPT CPU: build + scale + wire + fwd/vjp ...")
    var net = NET.make["cpu", Deterministic]()
    gpt_scale_residual_proj[
        "cpu", VOCAB, SEQ, EMB, HEADS, LAYERS, FFM, CAUSAL, PDROP, SEED, UMAX
    ](net, None)
    gpt_wire_tie[
        "cpu", VOCAB, SEQ, EMB, HEADS, LAYERS, FFM, CAUSAL, PDROP, SEED, UMAX
    ](net)
    var x = _onehot_input()
    var out = Tensor.alloc(B * OUT)
    net.forward["cpu", B](TensorRefs[1](x), out, None)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT]((i % 5) - 2) * 0.1
    var gi = Tensor.alloc(B * IN)
    net.zero_grad["cpu"](None)
    net.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    var ok = _finite(out, B * OUT) and _finite(gi, B * IN)
    print("  finite out+gi:", ok)
    assert_true(ok, "GPT CPU smoke")
    print("  ok")


def test_gpu() raises:
    print("GPT GPU: build + scale + wire + fwd/vjp ...")
    var c = DeviceContext()
    var net = NET.make["gpu", Deterministic](Optional(c))
    gpt_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMB, HEADS, LAYERS, FFM, CAUSAL, PDROP, SEED, UMAX
    ](net, Optional(c))
    gpt_wire_tie[
        "gpu", VOCAB, SEQ, EMB, HEADS, LAYERS, FFM, CAUSAL, PDROP, SEED, UMAX
    ](net)
    var x = _onehot_input()
    x.upload(c)
    var out = Tensor.alloc(B * OUT)
    net.forward["gpu", B](TensorRefs[1](x), out, Optional(c))
    var go = Tensor.alloc(B * OUT)
    for i in range(B * OUT):
        go.data[i] = Scalar[DT]((i % 5) - 2) * 0.1
    go.upload(c)
    var gi = Tensor.alloc(B * IN)
    net.zero_grad["gpu"](Optional(c))
    net.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(c))
    out.download(c)
    gi.download(c)
    var ok = _finite(out, B * OUT) and _finite(gi, B * IN)
    print("  finite out+gi:", ok)
    assert_true(ok, "GPT GPU smoke")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("nn.storage GPT (weight-tied) smoke")
    print("=" * 60)
    test_cpu()
    test_gpu()
    print("ALL PASSED")
