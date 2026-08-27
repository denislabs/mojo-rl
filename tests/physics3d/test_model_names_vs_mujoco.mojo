"""`FlatModelDef`'s NAME TABLES against MuJoCo's own — element for element.

WHY THIS EXISTS
===============
The parser resolves names into indices and used to DROP the strings. That is
fine for simulating a model and fatal for editing one: an outliner can only
say "body 7", a selection cannot survive an insert, a state remap across a
rebuild has no key, and an MJCF writer would have to synthesise `body0` —
a flattened export is acceptable, a NAMELESS one is not, because keyframes,
sensors, `<contact>` pairs and user code all key on names. See
`docs/PHYSICS3D_STUDIO_PLAN.md` §1.3, where this is a hard prerequisite for
S1.

⚠⚠ THE HARD PART IS NOT THE STRINGS, IT IS THE ORDER. MuJoCo emits joints,
sites and geoms GROUPED BY BODY — all of body 0's, then body 1's, declaration
order preserved inside each — which is NOT document order. `_fill_model` ends
by applying that grouping to the record arrays, and a name table built by
counting tags would label a permuted array. This tree has already paid for
that exact mistake twice: `_find_joint_index_by_name` was "a plain text
count", and `_find_site_index_by_name` carried a docstring asserting document
order that had silently stopped being true when the sort was added.

So the tables come from `names_in_element_order`, which is now THE walk —
`_index_by_name_grouped` is a lookup into it rather than a second copy of the
rule. And bodies go through `body_names_in_order` instead, because bodies are
NOT regrouped and are 1-based off the worldbody.

⚠ THE GOLDEN IS MuJoCo, NOT US. Nothing in our tree can check an element
ORDER against itself — that is the "a gate sharing its reference
implementation is blind" failure, which cost this tree two parsers agreeing on
one wrong default for months. Regenerate with:

    pixi run python scripts/dump_mujoco_names.py

and paste its output over the block below. The runtime is MuJoCo 3.10.0; the
`references/` trees are older and are NOT the oracle here.

⚠ AN UNNAMED ELEMENT IS "" ON BOTH SIDES. `mj_id2name` returns NULL for one,
and we store the empty string; most geoms in this tree have no `name=`, so a
gate that skipped them would test almost nothing on walker2d. The counts are
asserted separately for exactly that reason.

Run: pixi run mojo run -I . tests/physics3d/test_model_names_vs_mujoco.mojo
"""

from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, read_model_source,
)


def golden() -> List[String]:
    """MuJoCo 3.10.0's `mj_id2name` for five models. `model|kind|index|name`."""
    var g = List[String]()
    g.append(String("walker2d|body|0|world"))
    g.append(String("walker2d|body|1|torso"))
    g.append(String("walker2d|body|2|thigh"))
    g.append(String("walker2d|body|3|leg"))
    g.append(String("walker2d|body|4|foot"))
    g.append(String("walker2d|body|5|thigh_left"))
    g.append(String("walker2d|body|6|leg_left"))
    g.append(String("walker2d|body|7|foot_left"))
    g.append(String("walker2d|joint|0|rootx"))
    g.append(String("walker2d|joint|1|rootz"))
    g.append(String("walker2d|joint|2|rooty"))
    g.append(String("walker2d|joint|3|thigh_joint"))
    g.append(String("walker2d|joint|4|leg_joint"))
    g.append(String("walker2d|joint|5|foot_joint"))
    g.append(String("walker2d|joint|6|thigh_left_joint"))
    g.append(String("walker2d|joint|7|leg_left_joint"))
    g.append(String("walker2d|joint|8|foot_left_joint"))
    g.append(String("walker2d|geom|0|floor"))
    g.append(String("walker2d|geom|1|torso_geom"))
    g.append(String("walker2d|geom|2|thigh_geom"))
    g.append(String("walker2d|geom|3|leg_geom"))
    g.append(String("walker2d|geom|4|foot_geom"))
    g.append(String("walker2d|geom|5|thigh_left_geom"))
    g.append(String("walker2d|geom|6|leg_left_geom"))
    g.append(String("walker2d|geom|7|foot_left_geom"))
    g.append(String("walker2d|actuator|0|"))
    g.append(String("walker2d|actuator|1|"))
    g.append(String("walker2d|actuator|2|"))
    g.append(String("walker2d|actuator|3|"))
    g.append(String("walker2d|actuator|4|"))
    g.append(String("walker2d|actuator|5|"))
    g.append(String("humanoid|body|0|world"))
    g.append(String("humanoid|body|1|torso"))
    g.append(String("humanoid|body|2|lwaist"))
    g.append(String("humanoid|body|3|pelvis"))
    g.append(String("humanoid|body|4|right_thigh"))
    g.append(String("humanoid|body|5|right_shin"))
    g.append(String("humanoid|body|6|right_foot"))
    g.append(String("humanoid|body|7|left_thigh"))
    g.append(String("humanoid|body|8|left_shin"))
    g.append(String("humanoid|body|9|left_foot"))
    g.append(String("humanoid|body|10|right_upper_arm"))
    g.append(String("humanoid|body|11|right_lower_arm"))
    g.append(String("humanoid|body|12|left_upper_arm"))
    g.append(String("humanoid|body|13|left_lower_arm"))
    g.append(String("humanoid|joint|0|root"))
    g.append(String("humanoid|joint|1|abdomen_z"))
    g.append(String("humanoid|joint|2|abdomen_y"))
    g.append(String("humanoid|joint|3|abdomen_x"))
    g.append(String("humanoid|joint|4|right_hip_x"))
    g.append(String("humanoid|joint|5|right_hip_z"))
    g.append(String("humanoid|joint|6|right_hip_y"))
    g.append(String("humanoid|joint|7|right_knee"))
    g.append(String("humanoid|joint|8|left_hip_x"))
    g.append(String("humanoid|joint|9|left_hip_z"))
    g.append(String("humanoid|joint|10|left_hip_y"))
    g.append(String("humanoid|joint|11|left_knee"))
    g.append(String("humanoid|joint|12|right_shoulder1"))
    g.append(String("humanoid|joint|13|right_shoulder2"))
    g.append(String("humanoid|joint|14|right_elbow"))
    g.append(String("humanoid|joint|15|left_shoulder1"))
    g.append(String("humanoid|joint|16|left_shoulder2"))
    g.append(String("humanoid|joint|17|left_elbow"))
    g.append(String("humanoid|geom|0|floor"))
    g.append(String("humanoid|geom|1|torso1"))
    g.append(String("humanoid|geom|2|head"))
    g.append(String("humanoid|geom|3|uwaist"))
    g.append(String("humanoid|geom|4|lwaist"))
    g.append(String("humanoid|geom|5|butt"))
    g.append(String("humanoid|geom|6|right_thigh1"))
    g.append(String("humanoid|geom|7|right_shin1"))
    g.append(String("humanoid|geom|8|right_foot"))
    g.append(String("humanoid|geom|9|left_thigh1"))
    g.append(String("humanoid|geom|10|left_shin1"))
    g.append(String("humanoid|geom|11|left_foot"))
    g.append(String("humanoid|geom|12|right_uarm1"))
    g.append(String("humanoid|geom|13|right_larm"))
    g.append(String("humanoid|geom|14|right_hand"))
    g.append(String("humanoid|geom|15|left_uarm1"))
    g.append(String("humanoid|geom|16|left_larm"))
    g.append(String("humanoid|geom|17|left_hand"))
    g.append(String("humanoid|actuator|0|abdomen_y"))
    g.append(String("humanoid|actuator|1|abdomen_z"))
    g.append(String("humanoid|actuator|2|abdomen_x"))
    g.append(String("humanoid|actuator|3|right_hip_x"))
    g.append(String("humanoid|actuator|4|right_hip_z"))
    g.append(String("humanoid|actuator|5|right_hip_y"))
    g.append(String("humanoid|actuator|6|right_knee"))
    g.append(String("humanoid|actuator|7|left_hip_x"))
    g.append(String("humanoid|actuator|8|left_hip_z"))
    g.append(String("humanoid|actuator|9|left_hip_y"))
    g.append(String("humanoid|actuator|10|left_knee"))
    g.append(String("humanoid|actuator|11|right_shoulder1"))
    g.append(String("humanoid|actuator|12|right_shoulder2"))
    g.append(String("humanoid|actuator|13|right_elbow"))
    g.append(String("humanoid|actuator|14|left_shoulder1"))
    g.append(String("humanoid|actuator|15|left_shoulder2"))
    g.append(String("humanoid|actuator|16|left_elbow"))
    g.append(String("ant|body|0|world"))
    g.append(String("ant|body|1|torso"))
    g.append(String("ant|body|2|front_left_leg"))
    g.append(String("ant|body|3|aux_1"))
    g.append(String("ant|body|4|"))
    g.append(String("ant|body|5|front_right_leg"))
    g.append(String("ant|body|6|aux_2"))
    g.append(String("ant|body|7|"))
    g.append(String("ant|body|8|back_leg"))
    g.append(String("ant|body|9|aux_3"))
    g.append(String("ant|body|10|"))
    g.append(String("ant|body|11|right_back_leg"))
    g.append(String("ant|body|12|aux_4"))
    g.append(String("ant|body|13|"))
    g.append(String("ant|joint|0|root"))
    g.append(String("ant|joint|1|hip_1"))
    g.append(String("ant|joint|2|ankle_1"))
    g.append(String("ant|joint|3|hip_2"))
    g.append(String("ant|joint|4|ankle_2"))
    g.append(String("ant|joint|5|hip_3"))
    g.append(String("ant|joint|6|ankle_3"))
    g.append(String("ant|joint|7|hip_4"))
    g.append(String("ant|joint|8|ankle_4"))
    g.append(String("ant|geom|0|floor"))
    g.append(String("ant|geom|1|torso_geom"))
    g.append(String("ant|geom|2|aux_1_geom"))
    g.append(String("ant|geom|3|left_leg_geom"))
    g.append(String("ant|geom|4|left_ankle_geom"))
    g.append(String("ant|geom|5|aux_2_geom"))
    g.append(String("ant|geom|6|right_leg_geom"))
    g.append(String("ant|geom|7|right_ankle_geom"))
    g.append(String("ant|geom|8|aux_3_geom"))
    g.append(String("ant|geom|9|back_leg_geom"))
    g.append(String("ant|geom|10|third_ankle_geom"))
    g.append(String("ant|geom|11|aux_4_geom"))
    g.append(String("ant|geom|12|rightback_leg_geom"))
    g.append(String("ant|geom|13|fourth_ankle_geom"))
    g.append(String("ant|actuator|0|"))
    g.append(String("ant|actuator|1|"))
    g.append(String("ant|actuator|2|"))
    g.append(String("ant|actuator|3|"))
    g.append(String("ant|actuator|4|"))
    g.append(String("ant|actuator|5|"))
    g.append(String("ant|actuator|6|"))
    g.append(String("ant|actuator|7|"))
    g.append(String("so_arm100|body|0|world"))
    g.append(String("so_arm100|body|1|Base"))
    g.append(String("so_arm100|body|2|Rotation_Pitch"))
    g.append(String("so_arm100|body|3|Upper_Arm"))
    g.append(String("so_arm100|body|4|Lower_Arm"))
    g.append(String("so_arm100|body|5|Wrist_Pitch_Roll"))
    g.append(String("so_arm100|body|6|Fixed_Jaw"))
    g.append(String("so_arm100|body|7|Moving_Jaw"))
    g.append(String("so_arm100|body|8|target"))
    g.append(String("so_arm100|joint|0|Rotation"))
    g.append(String("so_arm100|joint|1|Pitch"))
    g.append(String("so_arm100|joint|2|Elbow"))
    g.append(String("so_arm100|joint|3|Wrist_Pitch"))
    g.append(String("so_arm100|joint|4|Wrist_Roll"))
    g.append(String("so_arm100|joint|5|Jaw"))
    g.append(String("so_arm100|geom|0|floor"))
    g.append(String("so_arm100|geom|1|"))
    g.append(String("so_arm100|geom|2|"))
    g.append(String("so_arm100|geom|3|"))
    g.append(String("so_arm100|geom|4|"))
    g.append(String("so_arm100|geom|5|"))
    g.append(String("so_arm100|geom|6|"))
    g.append(String("so_arm100|geom|7|"))
    g.append(String("so_arm100|geom|8|"))
    g.append(String("so_arm100|geom|9|"))
    g.append(String("so_arm100|geom|10|"))
    g.append(String("so_arm100|geom|11|"))
    g.append(String("so_arm100|geom|12|"))
    g.append(String("so_arm100|geom|13|"))
    g.append(String("so_arm100|geom|14|"))
    g.append(String("so_arm100|geom|15|"))
    g.append(String("so_arm100|geom|16|"))
    g.append(String("so_arm100|geom|17|"))
    g.append(String("so_arm100|geom|18|"))
    g.append(String("so_arm100|geom|19|"))
    g.append(String("so_arm100|geom|20|fixed_jaw_pad_1"))
    g.append(String("so_arm100|geom|21|fixed_jaw_pad_2"))
    g.append(String("so_arm100|geom|22|fixed_jaw_pad_3"))
    g.append(String("so_arm100|geom|23|fixed_jaw_pad_4"))
    g.append(String("so_arm100|geom|24|"))
    g.append(String("so_arm100|geom|25|"))
    g.append(String("so_arm100|geom|26|"))
    g.append(String("so_arm100|geom|27|"))
    g.append(String("so_arm100|geom|28|moving_jaw_pad_1"))
    g.append(String("so_arm100|geom|29|moving_jaw_pad_2"))
    g.append(String("so_arm100|geom|30|moving_jaw_pad_3"))
    g.append(String("so_arm100|geom|31|moving_jaw_pad_4"))
    g.append(String("so_arm100|geom|32|target"))
    g.append(String("so_arm100|actuator|0|Rotation"))
    g.append(String("so_arm100|actuator|1|Pitch"))
    g.append(String("so_arm100|actuator|2|Elbow"))
    g.append(String("so_arm100|actuator|3|Wrist_Pitch"))
    g.append(String("so_arm100|actuator|4|Wrist_Roll"))
    g.append(String("so_arm100|actuator|5|Jaw"))
    g.append(String("quadruped_walk|body|0|world"))
    g.append(String("quadruped_walk|body|1|torso"))
    g.append(String("quadruped_walk|body|2|hip_front_left"))
    g.append(String("quadruped_walk|body|3|knee_front_left"))
    g.append(String("quadruped_walk|body|4|ankle_front_left"))
    g.append(String("quadruped_walk|body|5|toe_front_left"))
    g.append(String("quadruped_walk|body|6|hip_front_right"))
    g.append(String("quadruped_walk|body|7|knee_front_right"))
    g.append(String("quadruped_walk|body|8|ankle_front_right"))
    g.append(String("quadruped_walk|body|9|toe_front_right"))
    g.append(String("quadruped_walk|body|10|hip_back_right"))
    g.append(String("quadruped_walk|body|11|knee_back_right"))
    g.append(String("quadruped_walk|body|12|ankle_back_right"))
    g.append(String("quadruped_walk|body|13|toe_back_right"))
    g.append(String("quadruped_walk|body|14|hip_back_left"))
    g.append(String("quadruped_walk|body|15|knee_back_left"))
    g.append(String("quadruped_walk|body|16|ankle_back_left"))
    g.append(String("quadruped_walk|body|17|toe_back_left"))
    g.append(String("quadruped_walk|joint|0|root"))
    g.append(String("quadruped_walk|joint|1|yaw_front_left"))
    g.append(String("quadruped_walk|joint|2|pitch_front_left"))
    g.append(String("quadruped_walk|joint|3|knee_front_left"))
    g.append(String("quadruped_walk|joint|4|ankle_front_left"))
    g.append(String("quadruped_walk|joint|5|yaw_front_right"))
    g.append(String("quadruped_walk|joint|6|pitch_front_right"))
    g.append(String("quadruped_walk|joint|7|knee_front_right"))
    g.append(String("quadruped_walk|joint|8|ankle_front_right"))
    g.append(String("quadruped_walk|joint|9|yaw_back_right"))
    g.append(String("quadruped_walk|joint|10|pitch_back_right"))
    g.append(String("quadruped_walk|joint|11|knee_back_right"))
    g.append(String("quadruped_walk|joint|12|ankle_back_right"))
    g.append(String("quadruped_walk|joint|13|yaw_back_left"))
    g.append(String("quadruped_walk|joint|14|pitch_back_left"))
    g.append(String("quadruped_walk|joint|15|knee_back_left"))
    g.append(String("quadruped_walk|joint|16|ankle_back_left"))
    g.append(String("quadruped_walk|geom|0|floor"))
    g.append(String("quadruped_walk|geom|1|eye_r"))
    g.append(String("quadruped_walk|geom|2|eye_l"))
    g.append(String("quadruped_walk|geom|3|torso"))
    g.append(String("quadruped_walk|geom|4|thigh_front_left"))
    g.append(String("quadruped_walk|geom|5|shin_front_left"))
    g.append(String("quadruped_walk|geom|6|foot_front_left"))
    g.append(String("quadruped_walk|geom|7|toe_front_left"))
    g.append(String("quadruped_walk|geom|8|thigh_front_right"))
    g.append(String("quadruped_walk|geom|9|shin_front_right"))
    g.append(String("quadruped_walk|geom|10|foot_front_right"))
    g.append(String("quadruped_walk|geom|11|toe_front_right"))
    g.append(String("quadruped_walk|geom|12|thigh_back_right"))
    g.append(String("quadruped_walk|geom|13|shin_back_right"))
    g.append(String("quadruped_walk|geom|14|foot_back_right"))
    g.append(String("quadruped_walk|geom|15|toe_back_right"))
    g.append(String("quadruped_walk|geom|16|thigh_back_left"))
    g.append(String("quadruped_walk|geom|17|shin_back_left"))
    g.append(String("quadruped_walk|geom|18|foot_back_left"))
    g.append(String("quadruped_walk|geom|19|toe_back_left"))
    g.append(String("quadruped_walk|site|0|pupil_r"))
    g.append(String("quadruped_walk|site|1|pupil_l"))
    g.append(String("quadruped_walk|site|2|workspace"))
    g.append(String("quadruped_walk|site|3|rf_00"))
    g.append(String("quadruped_walk|site|4|rf_01"))
    g.append(String("quadruped_walk|site|5|rf_02"))
    g.append(String("quadruped_walk|site|6|rf_03"))
    g.append(String("quadruped_walk|site|7|rf_04"))
    g.append(String("quadruped_walk|site|8|rf_10"))
    g.append(String("quadruped_walk|site|9|rf_11"))
    g.append(String("quadruped_walk|site|10|rf_12"))
    g.append(String("quadruped_walk|site|11|rf_13"))
    g.append(String("quadruped_walk|site|12|rf_14"))
    g.append(String("quadruped_walk|site|13|rf_20"))
    g.append(String("quadruped_walk|site|14|rf_21"))
    g.append(String("quadruped_walk|site|15|rf_22"))
    g.append(String("quadruped_walk|site|16|rf_23"))
    g.append(String("quadruped_walk|site|17|rf_24"))
    g.append(String("quadruped_walk|site|18|rf_30"))
    g.append(String("quadruped_walk|site|19|rf_31"))
    g.append(String("quadruped_walk|site|20|rf_32"))
    g.append(String("quadruped_walk|site|21|rf_33"))
    g.append(String("quadruped_walk|site|22|rf_34"))
    g.append(String("quadruped_walk|site|23|torso_touch"))
    g.append(String("quadruped_walk|site|24|torso"))
    g.append(String("quadruped_walk|site|25|toe_front_left"))
    g.append(String("quadruped_walk|site|26|toe_front_right"))
    g.append(String("quadruped_walk|site|27|toe_back_right"))
    g.append(String("quadruped_walk|site|28|toe_back_left"))
    g.append(String("quadruped_walk|actuator|0|yaw_front_left"))
    g.append(String("quadruped_walk|actuator|1|lift_front_left"))
    g.append(String("quadruped_walk|actuator|2|extend_front_left"))
    g.append(String("quadruped_walk|actuator|3|yaw_front_right"))
    g.append(String("quadruped_walk|actuator|4|lift_front_right"))
    g.append(String("quadruped_walk|actuator|5|extend_front_right"))
    g.append(String("quadruped_walk|actuator|6|yaw_back_right"))
    g.append(String("quadruped_walk|actuator|7|lift_back_right"))
    g.append(String("quadruped_walk|actuator|8|extend_back_right"))
    g.append(String("quadruped_walk|actuator|9|yaw_back_left"))
    g.append(String("quadruped_walk|actuator|10|lift_back_left"))
    g.append(String("quadruped_walk|actuator|11|extend_back_left"))
    return g^


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def eq(mut self, got: String, want: String, msg: String):
        self.checks += 1
        if got == want:
            return
        self.fails += 1
        print("  FAIL:", msg, "— MuJoCo says '", want, "', we say '", got, "'")


def _want(g: List[String], model: String, kind: String) -> List[String]:
    """The golden's rows for one model+family, in index order."""
    var pre = model + "|" + kind + "|"
    var out = List[String]()
    var i = 0
    while True:
        var key = pre + String(i) + "|"
        var found = False
        for row in g:
            if row.startswith(key):
                out.append(String(row[byte=key.byte_length():]))
                found = True
                break
        if not found:
            return out^
        i += 1


def family(
    mut t: Tally, g: List[String], model: String, kind: String,
    got: List[String],
) raises:
    var want = _want(g, model, kind)
    # ⚠ THE COUNT IS ITS OWN ARM. Comparing only the overlap would let a table
    # that stops early pass every element it does have — and a name table one
    # short is exactly what a dropped `<body>` looks like.
    t.checks += 1
    if len(want) != len(got):
        t.fails += 1
        print("  FAIL:", model, kind, "count — MuJoCo", len(want),
              "we", len(got))
        return
    if len(want) == 0:
        # ⚠ NON-VACUITY: a family with no elements compares nothing. Say so
        # rather than printing a green line that means "not tested".
        print("    ", model, kind, ": 0 elements (nothing compared)")
        return
    for i in range(len(want)):
        t.eq(got[i], want[i], String(model, " ", kind, "[", i, "]"))
    print("    ", model, kind, ":", len(want), "compared")


def check(mut t: Tally, g: List[String], model: String, path: String) raises:
    print("---", model)
    var fmd = parse_model_runtime(path)
    family(t, g, model, "body", fmd.body_names)
    family(t, g, model, "joint", fmd.joint_names)
    family(t, g, model, "geom", fmd.geom_names)
    family(t, g, model, "site", fmd.site_names)
    family(t, g, model, "actuator", fmd.actuator_names)


def main() raises:
    var t = Tally()
    var g = golden()
    print("=== FlatModelDef name tables vs MuJoCo mj_id2name ===")
    print("   ", len(g), "golden rows")
    check(t, g, "walker2d", "mojo_rl/envs/walker2d/assets/walker2d.xml")
    check(t, g, "humanoid", "mojo_rl/envs/humanoid/assets/humanoid.xml")
    check(t, g, "ant", "mojo_rl/envs/ant/assets/ant.xml")
    check(t, g, "so_arm100", "mojo_rl/envs/robots/assets/so_arm100.xml")
    # ⚠ THE ONE THAT ACTUALLY TESTS SITES — 29 of them, and its `workspace`
    # site is declared on the WORLDBODY, so it must land at index 2 (body 0)
    # rather than wherever the text puts it. The other four models have no
    # sites at all, which the `family` arm reports rather than passing
    # quietly.
    check(t, g, "quadruped_walk",
          "mojo_rl/envs/dm_control/assets/quadruped_walk.xml")
    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_model_names_vs_mujoco: " + String(t.fails) + " failed"
        )
