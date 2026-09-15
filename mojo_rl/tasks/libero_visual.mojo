"""What a LIBERO camera shows that the MJCF alone does not say.

    var conds = libero_site_conditions(family)
    var vis = build_visual_model[DT, D](fmd, m, 1 << 1, conditions=conds)

LIBERO changes the picture at runtime in exactly one place: object classes
with a `vis_site_names` entry get their sites' alpha toggled every step
(`problems/*.py:set_visualization`, after `bddl_base_domain._post_process`
has run `update_state()` on every tracked object state). In the asset pack only
`FlatStove` has one — its red `burner` site, visible while `turn_on(qpos)`
holds, i.e. while the knob joint `button` is at 0.5 or more
(`articulated_objects.py`: `default_turnon_ranges = [0.5, 2.1]`,
`turn_on: qpos >= min(ranges)`). `turn_on_the_stove`'s recorded frames end
with it lit; its first frame does not show it.

⚠ IT IS EVERY STEP, NOT THE GOAL'S. `update_state` calls `turn_on` for any
object state with the affordance, so the burner lights in `put_the_bowl_on_the_stove`
too if the arm happens to turn the knob. The rule is the object's, and this
reads it from the object, not from the task.

⚠ THE RESET FRAME IS THE ONE EXCEPTION, AND IT IS NOT MODELLED. robosuite
renders the reset observation before any `_post_action`, so that single frame
shows the XML's own alpha (1: lit) whatever the knob says. Every recorded
`obs/agentview_rgb[i]` is taken after a step (`states[i + 1]`), so no dataset
frame sees it.

⚠ THE ROBOT'S FOUR VISIBLE SITES ARE NOT HERE, and correctly: robosuite hides
them at construction (`visualize(vis_settings={...: False})` sets their alpha
to 0), so a LIBERO picture never shows them. The tracer draws no site that is
not listed.
"""

from mojo_rl.physics3d.raytrace.visual import SiteCondition

from .spec import FamilySpec


comptime _STOVE_ON_MIN: Float64 = 0.5
"""`FlatStove.default_turnon_ranges[0]`."""


def _basename_stem(path: String) -> String:
    var slash = path.rfind("/")
    var base = String(path[byte = slash + 1 :]) if slash >= 0 else path
    if base.endswith(".xml"):
        var stem = String(base[byte = 0 : base.byte_length() - 4])
        return stem^
    return base^


def libero_site_conditions(f: FamilySpec) -> List[SiteCondition]:
    """One `SiteCondition` per flat stove in the family's composed scene.

    A slot's compiled names carry its `<attach prefix>`, which
    `family.compose_family` sets to `slot name + "_"`. Both the fixture
    (`flat_stove.xml`) and the movable (`flat_stove_free.xml`) variants carry
    the same `burner` site and `button` joint."""
    var out = List[SiteCondition]()
    for i in range(len(f.slots)):
        var stem = _basename_stem(f.slots[i].asset)
        if stem == "flat_stove" or stem == "flat_stove_free":
            var prefix = f.slots[i].name + "_"
            out.append(
                SiteCondition(
                    prefix + "burner", prefix + "button", _STOVE_ON_MIN
                )
            )
    return out^
