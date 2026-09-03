"""The SmolLM2 text tower's name map covers the checkpoint EXACTLY, both ways.

Same discipline as `test_vision_name_map.mojo`, plus the thing the vision tower
did not have: **112 zero-filled biases**. SmolLM2 is bias-free while our
`Linear` always carries a `bias` Param, so seven per layer have no counterpart
in the file. `TN_ZEROS` fills them; left at their random initialisation the
loaded model would be a different function from the published one, quietly.

That makes the arithmetic the real assertion here:

    257 walked params  =  145 from the file  +  112 zero-filled
    157,456,320 elems  =  157,318,080        +  138,240

Both halves are checked, and both are printed, because "no errors" is what a
map covering 130 of 145 also reports.

Offline: reads the manifest, never the 906,712,520-byte weight file.

Run:
  pixi run mojo run -I . tests/deep_agents/smolvla/test_text_name_map.mojo
"""

from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.torch_names import TN_ZEROS
from mojo_rl.deep_agents.smolvla.text import SmolLMTextTower
from mojo_rl.deep_agents.smolvla.names import text_name_map, SMOLVLA_TEXT
from mojo_rl.deep_agents.smolvla.manifest import Manifest, shape_str

comptime SEQ = 8  # names do not depend on SEQ; keep the walk cheap
comptime TOWER = SmolLMTextTower[SEQ]
comptime N_FILE = 145      # layers.* + norm.weight  (embed_tokens is separate)
comptime N_ZEROS = 112     # 7 bias Params per layer x 16
comptime N_WALK = N_FILE + N_ZEROS
comptime ELEMS_FILE = 157318080
comptime ELEMS_TOTAL = 157456320
comptime EMBED = SMOLVLA_TEXT + "embed_tokens.weight"


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
    print("SmolVLA SmolLM2 text tower — name map coverage")
    print("=" * 70)

    var man = Manifest()
    var map = text_name_map()
    var n_zero = 0
    for i in range(map.size()):
        if map.kind[i] == TN_ZEROS:
            n_zero += 1
    var n_mapped = map.size() - n_zero
    print("map entries:", map.size(), " = from-file", n_mapped,
          " + zero-filled", n_zero)
    assert_equal(map.size(), N_WALK, "map should hold 257 entries")
    assert_equal(n_zero, N_ZEROS, "expected 112 TN_ZEROS biases")
    assert_equal(n_mapped, N_FILE, "expected 145 from-file entries")

    # ── 1. theirs -> file, with the SHAPE (skip the zero-filled) ─────────
    var checked = 0
    var elems = 0
    for i in range(map.size()):
        if map.kind[i] == TN_ZEROS:
            continue
        var key = String(map.theirs[i])
        var j = man.index_of(key)
        assert_true(j >= 0, "map names '" + key + "' but the checkpoint has no"
                            " such tensor")
        var want = map.their_shape(i)
        assert_true(
            man.same_shape(j, want),
            "'" + key + "': map declares " + shape_str(want)
            + " but the checkpoint has " + shape_str(man.shapes[j]),
        )
        checked += 1
        elems += map.numel(i)
    print("  [1] theirs -> file :", checked, "names matched WITH shape,",
          elems, "elements")
    assert_equal(checked, N_FILE, "must have checked all 145")
    assert_equal(elems, ELEMS_FILE, "the from-file elements should total"
                                    " 157,318,080")

    # ── 2. file -> theirs. embed_tokens is a separate module, not the tower ──
    var seen = 0
    var unclaimed = 0
    var first = String("")
    for i in range(len(man.names)):
        ref n = man.names[i]
        if not n.startswith(SMOLVLA_TEXT):
            continue
        if n == EMBED:
            continue
        seen += 1
        var claimed = False
        for j in range(map.size()):
            if map.kind[j] != TN_ZEROS and map.theirs[j] == n:
                claimed = True
                break
        if not claimed:
            unclaimed += 1
            if first.byte_length() == 0:
                first = String(n)
    print("  [2] file -> theirs :", seen, "text tensors (embed_tokens"
          " excluded),", unclaimed, "unclaimed")
    assert_equal(seen, N_FILE, "the checkpoint should hold 145 tower tensors")
    assert_true(unclaimed == 0, String(unclaimed) + " tensor(s) claimed by no"
                " map entry, first '" + first + "' — they would never be read")

    # ── 3. ours -> model, both directions ────────────────────────────────
    var net = TOWER.make["cpu", Deterministic]()
    var w = WalkCollect()
    net.for_each_param["cpu"](w, None)
    print("  [3] walked params  :", len(w.names), " elements", w.total)
    assert_equal(len(w.names), N_WALK, "the tower should expose 257 params")
    assert_equal(w.total, ELEMS_TOTAL, "the tower should hold 157,456,320"
                                       " elements")

    var matched = 0
    for i in range(map.size()):
        var ours = String(map.ours[i])
        var j = w.index_of(ours)
        assert_true(j >= 0, "map names our '" + ours + "' but the tower walk"
                            " has no such parameter — map and module tree have"
                            " drifted")
        assert_equal(
            map.numel(i), w.sizes[j],
            "'" + ours + "': map declares " + String(map.numel(i))
            + " elements, the parameter holds " + String(w.sizes[j]),
        )
        matched += 1
    print("      ours -> model  :", matched, "matched with element counts")

    var un = 0
    for i in range(len(w.names)):
        if map.index_of_ours(w.names[i]) < 0:
            un += 1
            print("      UNMAPPED PARAM:", w.names[i])
    assert_true(un == 0, String(un) + " walked parameter(s) are in no map"
                " entry — they would keep their random initialisation")

    # The arithmetic that ties the two halves together.
    assert_equal(ELEMS_TOTAL - ELEMS_FILE, w.total - elems,
                 "zero-filled elements do not reconcile")
    print("      reconciles     :", elems, "from file +",
          w.total - elems, "zero-filled =", w.total)

    print()
    print("ALL PASSED —", N_FILE, "from the checkpoint,", N_ZEROS,
          "zero-filled, covered in BOTH directions")
