"""The action expert's 145 tensors, and the parity that distinguishes its layers.

Same both-directions discipline as the tower gates, plus the one thing unique to
the expert: **k and v change SHAPE with the layer index.**

    even layers (0,2,…)  k,v : [320, 720]   project the expert's own stream
    odd  layers (1,3,…)  k,v : [320, 320]   project the VLM's cached K/V

Both produce a 320-wide result, so a map applying one rule everywhere is right
about the output width and wrong about the input on half the layers. That is
why the map declares full 2-D shapes and why this test asserts the parity
directly rather than trusting a total: a count of elements could be made to
balance by two compensating errors, a per-layer shape cannot.

    257 walked params  =  145 from the file  +  112 zero-filled
    98,360,016 elems   =  98,245,840         +  114,176

Offline: reads the manifest, never the weight file.

Run:
  pixi run mojo run -I . tests/deep_agents/smolvla/test_expert_name_map.mojo
"""

from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.torch_names import TN_ZEROS
from mojo_rl.deep_agents.smolvla.expert import (
    SmolVLAExpert, EXPERT_W, EXPERT_LAYERS, VLM_KV_W,
)
from mojo_rl.deep_agents.smolvla.names import expert_name_map, SMOLVLA_EXPERT
from mojo_rl.deep_agents.smolvla.manifest import Manifest, shape_str

comptime N_FILE = 145
comptime N_ZEROS = 112
comptime N_WALK = N_FILE + N_ZEROS
comptime ELEMS_FILE = 98245840
comptime ELEMS_TOTAL = 98360016


struct WalkCollect(ParamVisitor):
    var names: List[String]
    var sizes: List[Int]
    var total: Int

    def __init__(out self):
        self.names = List[String]()
        self.sizes = List[Int]()
        self.total = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.names.append(name)
        self.sizes.append(N)
        self.total += N

    def index_of(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1


def main() raises:
    print("=" * 70)
    print("SmolVLA action expert — name map coverage and layer parity")
    print("=" * 70)

    var man = Manifest()
    var map = expert_name_map()
    var nz = 0
    for i in range(map.size()):
        if map.kind[i] == TN_ZEROS:
            nz += 1
    print("map entries:", map.size(), " = from-file", map.size() - nz,
          " + zero-filled", nz)
    assert_equal(map.size(), N_WALK, "map should hold 257 entries")
    assert_equal(nz, N_ZEROS, "expected 112 TN_ZEROS biases")

    # ── 1. theirs -> file, with the SHAPE ────────────────────────────────
    var checked = 0
    var elems = 0
    for i in range(map.size()):
        if map.kind[i] == TN_ZEROS:
            continue
        var key = String(map.theirs[i])
        var j = man.index_of(key)
        assert_true(j >= 0, "map names '" + key + "', absent from checkpoint")
        var want = map.their_shape(i)
        assert_true(
            man.same_shape(j, want),
            "'" + key + "': map declares " + shape_str(want)
            + " but the checkpoint has " + shape_str(man.shapes[j]),
        )
        checked += 1
        elems += map.numel(i)
    print("  [1] theirs -> file :", checked, "matched WITH shape,", elems,
          "elements")
    assert_equal(checked, N_FILE, "must have checked all 145")
    assert_equal(elems, ELEMS_FILE, "from-file elements should be 98,245,840")

    # ── 2. file -> theirs ────────────────────────────────────────────────
    var seen = 0
    var unclaimed = 0
    for i in range(len(man.names)):
        if not man.names[i].startswith(SMOLVLA_EXPERT):
            continue
        seen += 1
        var claimed = False
        for j in range(map.size()):
            if map.kind[j] != TN_ZEROS and map.theirs[j] == man.names[i]:
                claimed = True
                break
        if not claimed:
            unclaimed += 1
    print("  [2] file -> theirs :", seen, "expert tensors,", unclaimed,
          "unclaimed")
    assert_equal(seen, N_FILE, "the checkpoint should hold 145 expert tensors")
    assert_true(unclaimed == 0, "some expert tensors are claimed by nothing")

    # ── 3. the parity, asserted per layer ────────────────────────────────
    # Even: k in-dim is the expert's own 720. Odd: it is the VLM's 320.
    var even_ok = 0
    var odd_ok = 0
    for i in range(EXPERT_LAYERS):
        var nm = String("layers.") + String(i) + ".self_attn.k.weight"
        var mi = map.index_of_ours(nm)
        assert_true(mi >= 0, "no map entry for " + nm)
        var sh = map.their_shape(mi)
        var want_in = EXPERT_W if (i % 2 == 0) else VLM_KV_W
        assert_true(
            len(sh) == 2 and sh[0] == VLM_KV_W and sh[1] == want_in,
            nm + ": expected [" + String(VLM_KV_W) + ", " + String(want_in)
            + "] for a " + ("self" if i % 2 == 0 else "cross")
            + "-attention layer, map declares " + shape_str(sh),
        )
        if i % 2 == 0:
            even_ok += 1
        else:
            odd_ok += 1
    print("  [3] parity         :", even_ok, "self layers k=[320,720],",
          odd_ok, "cross layers k=[320,320]")
    assert_equal(even_ok, 8, "expected 8 self-attention layers")
    assert_equal(odd_ok, 8, "expected 8 cross-attention layers")

    # ── 4. ours -> model ─────────────────────────────────────────────────
    var e = SmolVLAExpert[].make["cpu", Deterministic]()
    var w = WalkCollect()
    e.for_each_param["cpu"](w, None)
    print("  [4] walked params  :", len(w.names), " elements", w.total)
    assert_equal(len(w.names), N_WALK, "the expert should expose 257 params")
    assert_equal(w.total, ELEMS_TOTAL, "should hold 98,360,016 elements")

    for i in range(map.size()):
        var ours = String(map.ours[i])
        var j = w.index_of(ours)
        assert_true(j >= 0, "map names our '" + ours + "' but the walk has no"
                            " such parameter")
        assert_equal(map.numel(i), w.sizes[j], "'" + ours + "': size mismatch")
    var un = 0
    for i in range(len(w.names)):
        if map.index_of_ours(w.names[i]) < 0:
            un += 1
            print("      UNMAPPED PARAM:", w.names[i])
    assert_true(un == 0, "some walked parameters are in no map entry")
    print("      ours -> model  :", map.size(), "matched with element counts")

    assert_equal(ELEMS_TOTAL - ELEMS_FILE, w.total - elems,
                 "zero-filled elements do not reconcile")
    print("      reconciles     :", elems, "from file +", w.total - elems,
          "zero-filled =", w.total)

    print()
    print("ALL PASSED —", N_FILE, "from the checkpoint,", N_ZEROS,
          "zero-filled, 8 self + 8 cross layers")
