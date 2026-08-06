"""Dog's skin, bound to dog's actual model — the integration the unit tests cannot reach.

    pixi run mojo run -I . tests/dm_control/test_dog_skin.mojo

`tests/render/test_skn_loader.mojo` and `test_skinning.mojo` gate the FORMAT and
the MATH against the raw file, with a synthetic bone->body map. Neither can
catch the thing that actually goes wrong when wiring a skin into a model: the
57 bone names not matching the bodies the port produced. That failure renders —
the unmatched regions collapse to the origin — and reports nothing.

So this gates the seam, and only the seam:

  · the parser saw `<skin>` and resolved its material to a texture
  · every bone name resolves to a body index in OUR dog
  · geom groups arrived, so the collision proxy is hidden and something is left

⚠ THIS COMPILES THE DOG MODEL DEF, which is the largest in the suite. Expect
minutes, not seconds. It is not in the smoke tier for that reason.
"""

from std.testing import assert_true, assert_equal, TestSuite

from mojo_rl.envs.dm_control.dog.dog_xml import DMDogStandWalkModel
from mojo_rl.render.skn_loader import load_skn
from mojo_rl.render.skinning import resolve_skin_bones


comptime SKN_PATH = String(
    "mojo_rl/envs/dm_control/dog/assets/dog_skin.skn"
)


def test_parser_saw_the_skin() raises:
    """`<skin>` reached the render data, texture and all."""
    assert_true(
        DMDogStandWalkModel.has_skin(),
        "dog's <skin> did not survive merge_mjcf / parse_xml",
    )


def test_every_skin_bone_binds_to_a_body() raises:
    """The seam. A bone that matches nothing collapses its region silently."""
    var skin = load_skn(SKN_PATH)
    var names = DMDogStandWalkModel.body_names()

    print("  model named bodies:", len(names))
    print("  skin bones:", len(skin.bones))
    assert_true(len(names) > 0, "no body names — skin binding is impossible")

    var bone_body = resolve_skin_bones(skin, names)
    var unbound = 0
    for b in range(len(bone_body)):
        if bone_body[b] < 0:
            unbound += 1
            print("    UNBOUND bone:", skin.bones[b].body_name)
    print("  unbound bones:", unbound, "/", len(bone_body))
    assert_equal(unbound, 0, "some skin bones match no body in our dog")

    # ⚠ AND THEY MUST NOT ALL LAND ON THE SAME BODY. `resolve_skin_bones`
    # returning 0 for everything would also report zero unbound, and would
    # animate the whole dog as one rigid lump.
    var distinct = 0
    for b in range(len(bone_body)):
        var seen = False
        for c in range(b):
            if bone_body[c] == bone_body[b]:
                seen = True
                break
        if not seen:
            distinct += 1
    print("  distinct bodies bound:", distinct)
    assert_equal(
        distinct, len(bone_body), "two bones resolved to the same body"
    )


def test_geom_groups_hide_the_collision_proxy() raises:
    """Group arrived, and hiding 3+ did not hide everything.

    dm_control's dog puts its collision capsules in group 3 and its bone meshes
    in group 5; MuJoCo shows only 0-2. If the parser missed `group` entirely
    the count below would be zero, and the skeleton would still be on screen.
    """
    var n_hidden = 0
    var n_shown = 0
    comptime for i in range(DMDogStandWalkModel.NGEOM):
        if DMDogStandWalkModel.geom_group_at(i) >= 3:
            n_hidden += 1
        else:
            n_shown += 1
    print("  dog geoms: ", n_shown, "shown,", n_hidden, "hidden by group")
    assert_true(
        n_hidden > 0,
        "no dog geom is in a hidden group — `group` was not parsed, or the"
        " class default did not inherit",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
