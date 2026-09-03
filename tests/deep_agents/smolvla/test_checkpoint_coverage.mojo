"""How much of `lerobot/smolvla_base` the port names, across ALL maps at once.

The per-component gates each check their own slice. This checks the slices
together, which catches two things none of them can:

  1. **Double claims.** Two maps naming the same checkpoint tensor is a real
     error — the tensor would be loaded into two different parameters and one of
     them is wrong — and it is invisible to a per-map gate, where each looks
     locally correct.
  2. **The running total**, stated as a number rather than a feeling. 500 is the
     whole file; anything not claimed is named here so the remaining work is a
     list, not an estimate.

Offline: reads the manifest, never the 906,712,520-byte weight file.

Run:
  pixi run mojo run -I . tests/deep_agents/smolvla/test_checkpoint_coverage.mojo
"""

from std.testing import assert_true, assert_equal

from mojo_rl.nn.core.torch_names import TorchNameMap, TN_ZEROS
from mojo_rl.deep_agents.smolvla.names import (
    vision_name_map, text_name_map, misc_name_map,
)
from mojo_rl.deep_agents.smolvla.manifest import Manifest, shape_str

comptime N_TOTAL = 500
comptime N_VISION = 197
comptime N_TEXT = 145
comptime N_MISC = 13
comptime N_CLAIMED = N_VISION + N_TEXT + N_MISC     # 355
comptime N_EXPERT = 145                              # still to port
comptime EXPERT_PREFIX = String("model.vlm_with_expert.lm_expert.")


def _claim(
    mut owner: List[Int], ref man: Manifest, ref m: TorchNameMap, tag: Int,
    label: String,
) raises -> Int:
    """Mark every file tensor this map names; raise on a second claim."""
    var n = 0
    for i in range(m.size()):
        if m.kind[i] == TN_ZEROS:
            continue
        var key = String(m.theirs[i])
        var j = man.index_of(key)
        assert_true(j >= 0, label + " names '" + key + "', absent from the"
                            " checkpoint")
        var want = m.their_shape(i)
        assert_true(
            man.same_shape(j, want),
            label + " '" + key + "': declares " + shape_str(want)
            + ", checkpoint has " + shape_str(man.shapes[j]),
        )
        assert_true(
            owner[j] == 0,
            "'" + key + "' is claimed by TWO maps (" + String(owner[j])
            + " and " + String(tag) + ") — it would be loaded into two"
            " different parameters, and one of them is wrong",
        )
        owner[j] = tag
        n += 1
    print("  ", label, "claims", n, "tensors")
    return n


def main() raises:
    print("=" * 70)
    print("SmolVLA checkpoint coverage — all maps together")
    print("=" * 70)
    var man = Manifest()
    assert_equal(man.size(), N_TOTAL, "the manifest should hold 500 tensors")

    var owner = List[Int](unsafe_uninit_length=man.size())
    for i in range(man.size()):
        owner[i] = 0

    var vm = vision_name_map()
    var tm = text_name_map()
    var mm = misc_name_map()
    var nv = _claim(owner, man, vm, 1, String("vision"))
    var nt = _claim(owner, man, tm, 2, String("text  "))
    var nm = _claim(owner, man, mm, 3, String("misc  "))
    assert_equal(nv, N_VISION, "vision should claim 197")
    assert_equal(nt, N_TEXT, "text should claim 145")
    assert_equal(nm, N_MISC, "misc should claim 13")

    var claimed = 0
    var unclaimed = 0
    var expert = 0
    var other = List[String]()
    for i in range(man.size()):
        if owner[i] != 0:
            claimed += 1
            continue
        unclaimed += 1
        if man.names[i].startswith(EXPERT_PREFIX):
            expert += 1
        else:
            other.append(String(man.names[i]))

    print()
    print("  claimed        ", claimed, "/", N_TOTAL)
    print("  unclaimed      ", unclaimed, " of which action expert:", expert)
    assert_equal(claimed, N_CLAIMED, "expected 355 claimed")
    assert_equal(expert, N_EXPERT, "expected the 145 expert tensors to be the"
                                   " remainder")

    # Anything unclaimed that is NOT the expert is an unaccounted-for tensor —
    # the failure this gate exists to make impossible to overlook.
    if len(other) > 0:
        for i in range(len(other)):
            print("  UNACCOUNTED:", other[i])
    assert_true(len(other) == 0, String(len(other)) + " unclaimed tensor(s)"
                " outside the action expert — the remaining work is not just"
                " the expert")

    print()
    print("PASSED —", claimed, "of", N_TOTAL, "claimed, 0 double-claimed;")
    print("         the remaining", expert, "are exactly the action expert")
