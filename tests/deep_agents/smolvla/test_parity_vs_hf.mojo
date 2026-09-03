"""Numerical parity against HuggingFace's OWN SmolVLM, on the real weights.

Everything else in this suite gates a BRICK against an outside reference. This
gates the WIRING: whether the bricks assembled into a tower reproduce the
published model. A torch forward written by the same hand as the port could not
show that — it would restate the same reading. `transformers`' SmolVLM can.

Two towers, both loaded from `model.safetensors` and fed the identical seeded
input the Python dumper used:

  * **vision** — `SigLIPVisionTower` vs `SmolVLMVisionTransformer`.
  * **text** — `SmolVLAPrefill` over `SmolVLMTextLayers` vs `LlamaModel` at
    SmolVLA's 16 layers. This is the strong one: it exercises RoPE, the GQA head
    broadcast, SwiGLU, RMSNorm, the prefix-LM block mask, both residuals and the
    layer ordering, all at once, against an implementation that shares no code
    and no author with ours.

⚠ **Judged by fraction-outside-tolerance and a relative norm, never a global
max.** These outputs reach ±25 and are sums over hundreds of fp32 terms in a
different order from torch's; a single worst element says nothing, while "how
many elements are off, and by how much relative to the signal" says everything.

⚠ **This gate runs on CPU, deliberately.** One SigLIP layer, same weights and
same input, gives rel L2 **9.5e-07 on CPU** and **1.6e-03 on Metal** — 1700x
worse for identical arithmetic on paper. That gap is a property of the GPU
kernels (MAX's matmul/BMM precision on Apple), not of the port: a wiring error
would show up on both. Gating on GPU would therefore need a tolerance loose
enough to hide real bugs. The CPU number is the correctness statement; the GPU
number is reported alongside as an observation and is worth re-measuring on the
5090 before reading anything into it.

⚠ Needs the ~907 MB checkpoint and a reference dump:
    pixi run -e act-ref python tools/vla/dump_smolvla_reference.py --out /tmp/vla_ref
    pixi run -e apple test-vla-parity

Run:
  VLA_REF=/tmp/vla_ref pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_parity_vs_hf.mojo
"""

from std.math import abs, sqrt
from std.os import getenv
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.torch_names import (
    LoadTorchNamed, TorchNameMap, TN_PLAIN,
)
from mojo_rl.io.safetensors import SafeTensors
from mojo_rl.io.hf import hf_download_file, HF_MODEL
from mojo_rl.deep_agents.act.refload import RefDump
from mojo_rl.deep_agents.smolvla.vision import (
    SigLIPVisionTower, SigLIPEmbeddings, SigLIPLayer, SIGLIP_IMG, SIGLIP_PATCH,
    SIGLIP_DIM, SIGLIP_GRID, SIGLIP_TOKENS, SIGLIP_HEADS, SIGLIP_FF,
)
from mojo_rl.deep_agents.smolvla.text import (
    SmolVLMTextLayers, SMOLLM_DIM, SMOLLM_FF, SMOLLM_KV_W, SMOLLM_LAYERS,
    SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM,
)
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from mojo_rl.deep_agents.smolvla.fused import SmolVLAPrefill
from mojo_rl.deep_agents.smolvla.names import (
    vision_name_map, text_name_map, SMOLVLA_VISION,
)
from mojo_rl.deep_agents.smolvla.attn_mask import att_2d_mask_square, smolvla_ar

comptime REPO = String("lerobot/smolvla_base")
comptime TS = 32                      # text_seq in the dumper
comptime P = TS
comptime SUFFIX = 4
comptime B = 1
comptime W = SMOLLM_DIM
comptime VIS_IN = 3 * 512 * 512
comptime VIS_OUT = 1024 * 768
comptime TXT_N = TS * W

comptime Tower = SmolVLMTextLayers[SMOLLM_LAYERS, W, SMOLLM_FF, SMOLLM_KV_W]
comptime Cache = SmolVLAKVCache[
    SMOLLM_LAYERS, P, SUFFIX, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, B
]
comptime Pre = SmolVLAPrefill[P, SUFFIX, B]
comptime EmbNet = SigLIPEmbeddings[
    SIGLIP_IMG, SIGLIP_PATCH, SIGLIP_DIM, SIGLIP_GRID, SIGLIP_TOKENS
]
comptime LayerNet = SigLIPLayer[
    SIGLIP_TOKENS, SIGLIP_DIM, SIGLIP_HEADS, SIGLIP_FF
]


def _compare(
    mut got: Tensor, ref want: List[Scalar[DT]], n: Int, label: String,
    tol: Float64,
) raises:
    """Fraction outside tolerance + relative norm — not a global max."""
    var num = Float64(0)
    var den = Float64(0)
    var maxref = Float64(0)
    var outside = 0
    var worst = Float64(0)
    for i in range(n):
        var g = Float64(got.data[i])
        var w = Float64(want[i])
        var d = abs(g - w)
        num += d * d
        den += w * w
        if abs(w) > maxref:
            maxref = abs(w)
        if d > worst:
            worst = d
    var scale = maxref if maxref > 1e-12 else 1.0
    for i in range(n):
        if abs(Float64(got.data[i]) - Float64(want[i])) / scale > tol:
            outside += 1
    var rel = sqrt(num) / (sqrt(den) + 1e-30)
    var frac = Float64(outside) / Float64(n)
    print("  ", label, ": compared", n)
    print("      rel L2 =", rel, "  outside", tol, "of |ref|max:", outside,
          "(", frac * 100.0, "% )  worst abs", worst, " |ref|max", maxref)
    assert_true(n > 0, label + ": compared nothing")
    assert_true(maxref > 1e-6, label + ": the reference is ~zero, so any"
                               " output would pass — the gate is vacuous")
    assert_true(rel < 2e-3, label + ": relative L2 error too large")
    assert_true(frac < 1e-3, label + ": too many elements outside tolerance")


def main() raises:
    print("=" * 72)
    print("SmolVLA — parity vs HuggingFace's own SmolVLM, on the real weights")
    print("=" * 72)
    var root = getenv("VLA_REF")
    if root == "":
        root = String("/tmp/vla_ref")
    var dump = RefDump(root)
    print("  reference dump:", root)
    assert_true(dump.has(String("vision_emb")), "no vision_emb in the dump —"
                " run tools/vla/dump_smolvla_reference.py first")

    var path = hf_download_file(REPO, String("model.safetensors"), HF_MODEL)

    # ── [1] vision embeddings: patch conv + transpose + position table ───
    var enet = EmbNet.make["cpu", Deterministic]()
    var em = TorchNameMap()
    var cs: List[Int] = [768, 3, 16, 16]
    em.add(String("0.weight"),
           SMOLVLA_VISION + "embeddings.patch_embedding.weight", cs, TN_PLAIN)
    var cb: List[Int] = [768]
    em.add(String("0.bias"),
           SMOLVLA_VISION + "embeddings.patch_embedding.bias", cb, TN_PLAIN)
    var ps: List[Int] = [1024, 768]
    em.add(String("2.bias"),
           SMOLVLA_VISION + "embeddings.position_embedding.weight", ps,
           TN_PLAIN)
    var el = LoadTorchNamed[""](SafeTensors(path), em^)
    enet.for_each_param["cpu"](el, None)
    el.report(String("vision embeddings"))
    var vin = dump.get(String("vision_in"))
    var vx = Tensor.alloc(VIS_IN)
    for i in range(VIS_IN):
        vx.data[i] = vin[i]
    var vemb = Tensor.alloc(VIS_OUT)
    enet.forward["cpu", 1](TensorRefs[1](vx), vemb, None)
    _compare(vemb, dump.get(String("vision_emb")), VIS_OUT,
             String("[1] vision embeddings"), 1e-5)

    # ── [2] one SigLIP encoder layer ─────────────────────────────────────
    # ⚠ Position ids: SmolVLM buckets FRACTIONAL coordinates to support ragged
    # images. At a full 32x32 grid `bucketize` returns exactly [0..31] per axis,
    # so pos_ids == h*32 + w and a flat position table is right. At a PARTIAL
    # grid it would not be — worth knowing before anyone feeds a padded image.
    var lnet = LayerNet.make["cpu", Deterministic]()
    var lm = TorchNameMap()
    var t = SMOLVLA_VISION + "encoder.layers.0."
    var d1: List[Int] = [768]
    var ff1: List[Int] = [3072]
    lm.add(String("0.0.0.0.gamma"), t + "layer_norm1.weight", d1)
    lm.add(String("0.0.0.0.beta"), t + "layer_norm1.bias", d1)
    lm.add_linear(String("0.0.1.q.0.weight"), t + "self_attn.q_proj.weight", 768, 768)
    lm.add(String("0.0.1.q.0.bias"), t + "self_attn.q_proj.bias", d1)
    lm.add_linear(String("0.0.1.k.0.weight"), t + "self_attn.k_proj.weight", 768, 768)
    lm.add(String("0.0.1.k.0.bias"), t + "self_attn.k_proj.bias", d1)
    lm.add_linear(String("0.0.1.v.0.weight"), t + "self_attn.v_proj.weight", 768, 768)
    lm.add(String("0.0.1.v.0.bias"), t + "self_attn.v_proj.bias", d1)
    lm.add_linear(String("0.0.1.o.0.weight"), t + "self_attn.out_proj.weight", 768, 768)
    lm.add(String("0.0.1.o.0.bias"), t + "self_attn.out_proj.bias", d1)
    lm.add(String("1.0.0.0.gamma"), t + "layer_norm2.weight", d1)
    lm.add(String("1.0.0.0.beta"), t + "layer_norm2.bias", d1)
    lm.add_linear(String("1.0.1.0.0.weight"), t + "mlp.fc1.weight", 3072, 768)
    lm.add(String("1.0.1.0.0.bias"), t + "mlp.fc1.bias", ff1)
    lm.add_linear(String("1.0.1.2.0.weight"), t + "mlp.fc2.weight", 768, 3072)
    lm.add(String("1.0.1.2.0.bias"), t + "mlp.fc2.bias", d1)
    var ll = LoadTorchNamed[""](SafeTensors(path), lm^)
    lnet.for_each_param["cpu"](ll, None)
    ll.report(String("vision layer 0"))
    var lout = Tensor.alloc(VIS_OUT)
    lnet.forward["cpu", 1](TensorRefs[1](vemb), lout, None)
    _compare(lout, dump.get(String("vision_l0")), VIS_OUT,
             String("[2] vision layer 0  "), 1e-4)

    # ── [3] the whole text tower, through the fused prefill ──────────────
    # The strong one: RoPE, the GQA head broadcast, SwiGLU, RMSNorm, the
    # prefix-LM mask, both residuals and the layer ordering, sixteen deep,
    # against an implementation sharing no code and no author with ours.
    var tower = Tower.make["cpu", Deterministic]()
    var tl = LoadTorchNamed[""](SafeTensors(path), text_name_map())
    tower.for_each_param["cpu"](tl, None)
    tl.report(String("text"))
    assert_equal(len(tl.loaded), 145, "text should load 145 tensors")
    assert_equal(len(tl.zeroed), 112, "text should zero-fill 112 biases")

    var ar = smolvla_ar(TS - 1, 0, 1, 0)
    assert_equal(len(ar), TS, "ar length")
    var mask = att_2d_mask_square(ar)
    var cache = Cache.make["cpu"]()
    var pre = Pre.make["cpu"](mask)
    var tin = dump.get(String("text_in"))
    var tx = Tensor.alloc(TXT_N)
    for i in range(TXT_N):
        tx.data[i] = tin[i]
    var tout = Tensor.alloc(TXT_N)
    pre.run["cpu"](tower, cache, tx, tout, None)
    _compare(tout, dump.get(String("text_out")), TXT_N,
             String("[3] text tower x16  "), 1e-4)
    assert_equal(cache.n_filled(), SMOLLM_LAYERS, "prefill filled every layer")

    print()
    print("PASSED — the port reproduces HuggingFace's SmolVLM")
