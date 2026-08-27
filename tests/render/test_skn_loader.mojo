"""Gate for `load_skn` against dm_control's `dog_skin.skn`.

    pixi run mojo run -I . tests/render/test_skn_loader.mojo

⚠ THE NUMBERS BELOW WERE MEASURED, NOT COPIED FROM THIS LOADER. They come from
an independent transcription of `mjCSkin::LoadSKN` in Python, run over the same
file; pinning a loader against its own output gates nothing. The strongest of
them is `_EXPECT_WEIGHTS` together with `load_skn`'s own end-of-file check —
between them, a bone stride read even one field wrong cannot survive.

⚠ NEEDS THE REFERENCE TREE. `references/dm_control-main/` is not vendored with
the package, so this test SKIPS rather than fails when the asset is absent —
otherwise it would go red on any checkout that did not clone the references.
"""

from std.testing import assert_true, assert_equal, TestSuite
from std.math import abs

from mojo_rl.render.skn_loader import load_skn


comptime SKN_PATH = String(
    "references/dm_control-main/dm_control/suite/dog_assets/dog_skin.skn"
)

# From the independent Python parse of the same file — see the header.
comptime _EXPECT_NVERT: Int = 24065
comptime _EXPECT_NFACE: Int = 33900
comptime _EXPECT_NBONE: Int = 57
comptime _EXPECT_WEIGHTS: Int = 63008
comptime _EXPECT_BYTES: Int = 1396284


def _available() -> Bool:
    try:
        var f = open(SKN_PATH, "r")
        f.close()
        return True
    except:
        return False


def test_dog_skin_header_and_bone_stride() raises:
    """Counts, and that the parse consumed the file exactly."""
    if not _available():
        print("SKIP: no", SKN_PATH, "(reference tree not cloned)")
        return

    var skin = load_skn(SKN_PATH)

    assert_equal(skin.nvert, _EXPECT_NVERT, "nvert")
    assert_equal(skin.nface, _EXPECT_NFACE, "nface")
    assert_equal(len(skin.bones), _EXPECT_NBONE, "nbone")
    assert_equal(len(skin.vert), 3 * _EXPECT_NVERT, "vert array")
    assert_equal(len(skin.face), 3 * _EXPECT_NFACE, "face array")
    assert_true(skin.has_texcoords(), "dog's skin carries texcoords")

    # ⚠ THE STRIDE GATE. `load_skn` already raises unless the cursor lands on
    # the last byte, so reaching here means the 57 bone records were walked at
    # exactly the right width. This pins the total it walked as well, so a
    # compensating pair of errors cannot hide.
    assert_equal(skin.total_weights(), _EXPECT_WEIGHTS, "vertex-weight entries")
    print(
        "  dog skin:", skin.nvert, "verts,", skin.nface, "faces,",
        len(skin.bones), "bones,", skin.total_weights(), "weights",
        "(", _EXPECT_BYTES, "bytes )",
    )


def test_dog_skin_first_bone_is_torso() raises:
    """The first bone names a real body, and the name field is not truncated.

    An off-by-one in the 40-byte name field shows up here before it shows up as
    a limb following the wrong body.
    """
    if not _available():
        print("SKIP: no", SKN_PATH)
        return

    var skin = load_skn(SKN_PATH)
    assert_equal(skin.bones[0].body_name, String("torso"), "first bone")
    assert_equal(skin.bones[1].body_name, String("L_1"), "second bone")

    # Every name must be non-empty and plausibly short — a desynchronised
    # stride produces binary garbage here, which is the loudest cheap signal.
    for i in range(len(skin.bones)):
        var n = skin.bones[i].body_name
        assert_true(
            n.byte_length() > 0 and n.byte_length() < 40,
            "bone " + String(i) + " has an implausible name",
        )


def test_dog_skin_weights_are_a_partition() raises:
    """Each vertex's weights sum to 1 — the property LBS actually depends on.

    ⚠ THIS IS THE ONE THAT WOULD CATCH SWAPPED ARRAYS. `vertid` and
    `vertweight` are the same length and adjacent in the file, so reading them
    in the wrong order still parses and still lands on the last byte. Weights
    read as vertex ids would not sum to anything near 1.
    """
    if not _available():
        print("SKIP: no", SKN_PATH)
        return

    var skin = load_skn(SKN_PATH)

    var acc = List[Float32]()
    for _ in range(skin.nvert):
        acc.append(Float32(0))
    for b in range(len(skin.bones)):
        # Indexed in place — binding `skin.bones[b].vert_ids` to a local would
        # copy 63k entries' worth of Lists across the loop.
        var n_k = len(skin.bones[b].vert_ids)
        assert_equal(
            n_k, len(skin.bones[b].weights),
            "bone " + String(b) + ": id/weight length mismatch",
        )
        for k in range(n_k):
            acc[Int(skin.bones[b].vert_ids[k])] += skin.bones[b].weights[k]

    var worst = Float32(0)
    var n_touched = 0
    for v in range(skin.nvert):
        if acc[v] > 0:
            n_touched += 1
            var d = abs(acc[v] - Float32(1.0))
            if d > worst:
                worst = d

    print("  vertices with weight:", n_touched, "/", skin.nvert)
    print("  max |sum(w) - 1| =", worst)
    assert_equal(n_touched, skin.nvert, "every vertex must be bound to a bone")
    assert_true(worst < Float32(1e-3), "weights do not sum to 1 per vertex")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
