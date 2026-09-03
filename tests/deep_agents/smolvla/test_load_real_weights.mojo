"""Load the PUBLISHED SmolVLA weights into the port, for real.

Everything before this gated names and shapes against the 73 KB manifest. That
cannot catch a wrong `TN_TRANSPOSE` flag on a SQUARE matrix — `[320, 320]`
transposed is still `[320, 320]`, the element count matches, the load succeeds,
and the weights are silently transposed. Only a VALUE at a known (row, col)
settles it.

So this downloads `model.safetensors` (906,712,520 bytes, cached), loads each
component through `LoadTorchNamed`, and then checks individual values against
`tools/vla/smolvla_samples.tsv` — produced by a Python script that seeks into
the file and decodes BF16 by hand, sharing NO code with the Mojo reader it is
checking.

The sample set is chosen for what shapes cannot separate:

  * `lm_expert.layers.1.self_attn.k_proj` — F32 and SQUARE: the transpose-blind case.
  * `text_model.layers.0.self_attn.k_proj` — BF16, non-square, transposed.
  * `vision_model...q_proj` — BF16, transposed.
  * `patch_embedding.weight` — 4-D conv, NOT transposed.
  * `state_proj.weight` — F32, transposed, sits beside a real bias.

Coverage note: the two SQUARE tensors above (expert `[320,320]` F32, vision
`[768,768]` BF16) are the only transpose-BLIND ones — for every non-square
weight a wrong flag already fails `LoadTorchNamed`'s size check. Both squares are
probed here, at asymmetric coordinates, so the discriminating cases are covered
even though the text/state samples in the TSV belong to components this gate does
not instantiate.

⚠ Not in the default manifest: it needs a ~907 MB download. `pixi run test-vla-weights`.

Run:
  pixi run test-vla-weights
"""

from std.math import abs
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.io.fileio import read_file_bytes
from mojo_rl.io.hf import hf_download_file, HF_MODEL
from mojo_rl.io.safetensors import SafeTensors
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.torch_names import (
    TorchNameMap, LoadTorchNamed, TN_TRANSPOSE, TN_ZEROS,
)
from mojo_rl.deep_agents.smolvla.vision import SigLIPVisionTower
from mojo_rl.deep_agents.smolvla.expert import SmolVLAExpert
from mojo_rl.deep_agents.smolvla.names import (
    vision_name_map, expert_name_map,
)

comptime REPO = String("lerobot/smolvla_base")
comptime SAMPLES = String("tools/vla/smolvla_samples.tsv")


struct Sample(Movable):
    var name: String
    var rows: Int
    var cols: Int
    var r: Int
    var c: Int
    var value: Float32

    def __init__(out self, var name: String, rows: Int, cols: Int, r: Int,
                 c: Int, value: Float32):
        self.name = name^
        self.rows = rows
        self.cols = cols
        self.r = r
        self.c = c
        self.value = value

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.rows = move.rows
        self.cols = move.cols
        self.r = move.r
        self.c = move.c
        self.value = move.value


def load_samples() raises -> List[Sample]:
    var out = List[Sample]()
    var text = String(from_utf8=Span(read_file_bytes(SAMPLES)))
    for line in text.split(String("\n")):
        if line.byte_length() == 0 or line.startswith(String("#")):
            continue
        var p = line.split(String("\t"))
        if len(p) < 7:
            continue
        var dims = p[2].split(String(","))
        var rows = Int(dims[0])
        var cols = 1
        for i in range(1, len(dims)):
            cols *= Int(dims[i])
        out.append(
            Sample(String(p[0]), rows, cols, Int(p[3]), Int(p[4]),
                   Float32(Float64(p[6])))
        )
    return out^


struct Probe(ParamVisitor):
    """Pull one flat element out of a named parameter after loading."""
    var want: String
    var index: Int
    var found: Bool
    var value: Float32

    def __init__(out self, var want: String, index: Int):
        self.want = want^
        self.index = index
        self.found = False
        self.value = Float32(0)

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if name != self.want:
            return
        if self.index < 0 or self.index >= N:
            raise Error("Probe: index " + String(self.index) + " out of "
                        + String(N) + " for '" + name + "'")
        self.found = True
        self.value = Float32(param.data[self.index])


def _our_index(ref map: TorchNameMap, mi: Int, ref sm: Sample) -> Int:
    """Where their (row, col) lands in OUR flat layout.

    `nn.Linear.weight` is `[out, in]` and ours is `[in, out]`, so their
    `(r=out, c=in)` is our `c*rows + r`. Everything else (Conv2D's
    `[OC, IC, KH, KW]`, the 1-D norms) shares their layout, so the flat index is
    unchanged. Reading this backwards is exactly the bug the square F32 sample
    exists to catch.
    """
    if map.kind[mi] == TN_TRANSPOSE:
        return sm.c * sm.rows + sm.r
    return sm.r * sm.cols + sm.c


def main() raises:
    print("=" * 72)
    print("SmolVLA — loading the PUBLISHED weights")
    print("=" * 72)
    var path = hf_download_file(REPO, String("model.safetensors"), HF_MODEL)
    var f = SafeTensors(path)
    print("  file:", len(f.names), "tensors")
    assert_equal(len(f.names), 500, "expected the 500-tensor checkpoint")
    var smp = load_samples()
    print("  samples:", len(smp), "values read independently in Python")
    assert_true(len(smp) > 0, "no samples — the gate would be vacuous")

    var checked = 0

    # ── vision tower ─────────────────────────────────────────────────────
    var vmap = vision_name_map()
    var vnet = SigLIPVisionTower[].make["cpu", Deterministic]()
    var vl = LoadTorchNamed[""](SafeTensors(path), vision_name_map())
    vnet.for_each_param["cpu"](vl, None)
    vl.report(String("vision"))
    print("  vision: loaded", len(vl.loaded), "+ zeroed", len(vl.zeroed))
    assert_equal(len(vl.loaded), 197, "vision should load 197 tensors")
    for si in range(len(smp)):
        var mi = -1
        for k in range(vmap.size()):
            if vmap.kind[k] != TN_ZEROS and vmap.theirs[k] == smp[si].name:
                mi = k
                break
        if mi < 0:
            continue
        var idx = _our_index(vmap, mi, smp[si])
        var pr = Probe(String(vmap.ours[mi]), idx)
        vnet.for_each_param["cpu"](pr, None)
        assert_true(pr.found, "probe did not reach '" + vmap.ours[mi] + "'")
        var d = abs(pr.value - smp[si].value)
        print("    vision", smp[si].name[byte=48:], "(", smp[si].r, ",",
              smp[si].c, ") ->", pr.value, " want", smp[si].value)
        assert_true(d == Float32(0), "value mismatch — the layout or the"
                                     " transpose flag is wrong")
        checked += 1

    # ── action expert (holds the transpose-blind square case) ────────────
    var emap = expert_name_map()
    var enet = SmolVLAExpert[].make["cpu", Deterministic]()
    var el = LoadTorchNamed[""](SafeTensors(path), expert_name_map())
    enet.for_each_param["cpu"](el, None)
    el.report(String("expert"))
    print("  expert: loaded", len(el.loaded), "+ zeroed", len(el.zeroed))
    assert_equal(len(el.loaded), 145, "expert should load 145 tensors")
    assert_equal(len(el.zeroed), 112, "expert should zero-fill 112 biases")
    for si in range(len(smp)):
        var mi = -1
        for k in range(emap.size()):
            if emap.kind[k] != TN_ZEROS and emap.theirs[k] == smp[si].name:
                mi = k
                break
        if mi < 0:
            continue
        var idx = _our_index(emap, mi, smp[si])
        var pr = Probe(String(emap.ours[mi]), idx)
        enet.for_each_param["cpu"](pr, None)
        assert_true(pr.found, "probe did not reach '" + emap.ours[mi] + "'")
        var d = abs(pr.value - smp[si].value)
        print("    expert", emap.ours[mi], "(", smp[si].r, ",", smp[si].c,
              ") ->", pr.value, " want", smp[si].value)
        assert_true(d == Float32(0), "value mismatch — the layout or the"
                                     " transpose flag is wrong")
        checked += 1

    print()
    print("  values checked against the independent Python read:", checked)
    assert_true(checked >= 8, "too few sampled values compared — the transpose"
                              " check would be near-vacuous")
    print()
    print("PASSED")
