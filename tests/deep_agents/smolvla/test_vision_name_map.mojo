"""The SigLIP tower's name map covers the checkpoint EXACTLY — both directions.

Offline: it reads `tools/vla/smolvla_base_manifest.tsv` (the 500-tensor
checklist, obtained by two HTTP Range requests) and never touches the
906,712,520-byte weight file, so it runs in CI with no network and no download.

Three checks, because there are three ways a 197-entry map goes wrong:

  1. **theirs -> file.** Every name the map claims exists in the checkpoint,
     WITH THE DECLARED SHAPE. A `[768, 3072]` where the file has `[3072, 768]`
     holds the same float count and is the exact error `TN_TRANSPOSE` exists to
     prevent, so the shape is compared, never the count.
  2. **file -> theirs.** Every vision tensor in the checkpoint is claimed.
     ⚠ This is the direction `LoadTorchNamed.report()` cannot see: a map
     covering 190 of 197 has no unmapped param and no missing tensor, loads
     plenty, and quietly leaves seven published weights unread.
  3. **ours -> model.** Every name the map claims on OUR side is a real walked
     parameter, with a matching element count, and every walked parameter is
     claimed. This is what catches the map drifting from the module tree — the
     positional paths (`1.0.1.0.1.2.0.weight`) are unreadable by design and
     nothing but a machine check will notice an index slipping.

Run:
  pixi run mojo run -I . tests/deep_agents/smolvla/test_vision_name_map.mojo
"""

from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.io.fileio import read_file_bytes
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.torch_names import TorchNameMap
from mojo_rl.deep_agents.smolvla.vision import SigLIPVisionTower
from mojo_rl.deep_agents.smolvla.names import vision_name_map, SMOLVLA_VISION

comptime MANIFEST = String("tools/vla/smolvla_base_manifest.tsv")
comptime N_VISION = 197
comptime TOWER = SigLIPVisionTower[]


struct Manifest(Movable):
    """`<name>\\t<dtype>\\t<d0,d1,...>` per line, `#` comments skipped."""

    var names: List[String]
    var shapes: List[List[Int]]

    def __init__(out self, path: String) raises:
        self.names = List[String]()
        self.shapes = List[List[Int]]()
        var raw = read_file_bytes(path)
        var text = String(from_utf8=Span(raw))
        for line in text.split(String("\n")):
            if line.byte_length() == 0 or line.startswith(String("#")):
                continue
            var parts = line.split(String("\t"))
            if len(parts) < 3:
                continue
            var dims = List[Int]()
            for d in parts[2].split(String(",")):
                dims.append(Int(d))
            self.names.append(String(parts[0]))
            self.shapes.append(dims^)

    def index_of(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1


struct WalkCollect(ParamVisitor):
    var names: List[String]
    var sizes: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.sizes = List[Int]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.names.append(name)
        self.sizes.append(N)

    def index_of(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1


def _shape_str(ref s: List[Int]) -> String:
    var out = String("[")
    for i in range(len(s)):
        if i > 0:
            out += ", "
        out += String(s[i])
    return out + "]"


def main() raises:
    print("=" * 68)
    print("SmolVLA SigLIP vision tower — name map coverage")
    print("=" * 68)

    var man = Manifest(MANIFEST)
    var map = vision_name_map()
    print("manifest tensors:", len(man.names), " map entries:", map.size())
    assert_equal(map.size(), N_VISION, "map should hold 197 entries")

    # ── 1. theirs -> file, with the SHAPE ────────────────────────────────
    var checked = 0
    for i in range(map.size()):
        var key = String(map.theirs[i])
        var j = man.index_of(key)
        assert_true(j >= 0, "map names '" + key + "' but the checkpoint has no"
                            " such tensor")
        var want = map.their_shape(i)
        ref got = man.shapes[j]
        var same = len(want) == len(got)
        if same:
            for k in range(len(want)):
                if want[k] != got[k]:
                    same = False
                    break
        assert_true(
            same,
            "'" + key + "': map declares " + _shape_str(want)
            + " but the checkpoint has " + _shape_str(got),
        )
        checked += 1
    print("  [1] theirs -> file :", checked, "names matched WITH shape")
    assert_equal(checked, N_VISION, "must have checked all 197")

    # ── 2. file -> theirs (the direction report() cannot see) ────────────
    var vision_seen = 0
    var unclaimed = 0
    var first_unclaimed = String("")
    for i in range(len(man.names)):
        ref n = man.names[i]
        if not n.startswith(SMOLVLA_VISION):
            continue
        vision_seen += 1
        var claimed = False
        for j in range(map.size()):
            if map.theirs[j] == n:
                claimed = True
                break
        if not claimed:
            unclaimed += 1
            if first_unclaimed.byte_length() == 0:
                first_unclaimed = String(n)
    print("  [2] file -> theirs :", vision_seen, "vision tensors,",
          unclaimed, "unclaimed")
    assert_equal(vision_seen, N_VISION,
                 "the checkpoint should hold 197 vision tensors")
    assert_true(unclaimed == 0, String(unclaimed) + " vision tensor(s) claimed"
                " by no map entry, first '" + first_unclaimed + "' — they would"
                " never be read")

    # ── 3. ours -> model ─────────────────────────────────────────────────
    var net = TOWER.make["cpu", Deterministic]()
    var w = WalkCollect()
    net.for_each_param["cpu"](w, None)
    print("  [3] walked params  :", len(w.names))
    assert_equal(len(w.names), N_VISION,
                 "the tower should expose 197 parameter tensors")

    var matched = 0
    for i in range(map.size()):
        var ours = String(map.ours[i])
        var j = w.index_of(ours)
        assert_true(j >= 0, "map names our '" + ours + "' but the tower walk"
                            " has no such parameter — the map and the module"
                            " tree have drifted")
        assert_equal(
            map.numel(i), w.sizes[j],
            "'" + ours + "': map declares " + String(map.numel(i))
            + " elements, the tower's parameter holds " + String(w.sizes[j]),
        )
        matched += 1
    print("      ours -> model  :", matched, "matched with element counts")

    var our_unclaimed = 0
    for i in range(len(w.names)):
        if map.index_of_ours(w.names[i]) < 0:
            our_unclaimed += 1
            print("      UNMAPPED PARAM:", w.names[i])
    assert_true(our_unclaimed == 0,
                String(our_unclaimed) + " walked parameter(s) are in no map"
                " entry — they would keep their random initialisation")

    print()
    print("ALL PASSED —", N_VISION, "tensors covered in BOTH directions")
