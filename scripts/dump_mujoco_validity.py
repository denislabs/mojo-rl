"""Generate `tests/physics3d/validity_goldens.mojo` — the fixture table AND
MuJoCo's verdict on each fixture, from ONE source.

⚠⚠ WHY GENERATED AND NOT HAND-WRITTEN. `studio.validate` defines SEV_ERROR as
"MuJoCo refuses this model", so the gate has to compare our verdict against the
runtime's on the SAME text. If the fixtures lived in the Mojo test and the
verdicts in a table beside them, a fixture could be edited on one side only and
the gate would then be checking our validator against a stale opinion — the
"golden table that encodes another module's enum" failure, one level up.
Emitting both from this file makes that unrepresentable.

Run: pixi run python scripts/dump_mujoco_validity.py
"""

import mujoco


def M(body: str, tail: str = "") -> str:
    return f"<mujoco><worldbody>{body}</worldbody>{tail}</mujoco>"


B = (
    '<body name="b" pos="0 0 1"><joint name="j" type="hinge" axis="0 1 0"/>'
    '<geom name="g" type="sphere" size="0.1"/></body>'
)

# (name, xml, expected code — "" for a model that must come back clean).
#
# ⚠⚠ A CODE IS NOT THE SAME AS "MuJoCo REFUSES". Almost every ERROR below is
# one MuJoCo also refuses, and the check at the bottom enforces that. The
# exception is `_UNSUPPORTED_HERE`: `<replicate>` is a feature MuJoCo HAS and
# this engine does not, so the honest verdict is "will not load HERE". Tagging
# it explicitly is what stops that exemption from quietly growing to cover a
# check that is simply wrong.
CASES = [
    # ── the control ───────────────────────────────────────────────────────
    ("ok_baseline", M(B), ""),
    # ── things MuJoCo REFUSES ─────────────────────────────────────────────
    ("dangling_actuator_joint",
     M(B, '<actuator><motor name="m" joint="nope"/></actuator>'),
     "dangling-ref"),
    ("dangling_pair_geom",
     M(B, '<contact><pair geom1="g" geom2="nope"/></contact>'),
     "dangling-ref"),
    ("dangling_exclude_body",
     M(B, '<contact><exclude body1="b" body2="nope"/></contact>'),
     "dangling-ref"),
    ("dangling_equality_body",
     M(B, '<equality><weld body1="b" body2="nope"/></equality>'),
     "dangling-ref"),
    ("dangling_class",
     M('<body name="b"><joint name="j" type="hinge" axis="0 1 0"/>'
       '<geom name="g" class="nope" type="sphere" size="0.1"/></body>'),
     "dangling-ref"),
    ("zero_mass_moving_body",
     M('<body name="b"><joint name="j" type="hinge" axis="0 1 0"/>'
       '<inertial pos="0 0 0" mass="0" diaginertia="0 0 0"/>'
       '<geom name="g" type="sphere" size="0.1"/></body>'),
     "zero-mass-moving-body"),
    ("moving_body_no_geom",
     M('<body name="b"><joint name="j" type="hinge" axis="0 1 0"/>'
       '<site name="s"/></body>'),
     "zero-mass-moving-body"),
    ("free_joint_nested",
     M('<body name="p"><geom type="sphere" size="0.1"/>'
       '<body name="c"><joint name="f" type="free"/>'
       '<geom type="sphere" size="0.1"/></body></body>'),
     "free-joint-nested"),
    ("two_free_joints",
     M('<body name="b"><joint name="f1" type="free"/>'
       '<joint name="f2" type="free"/><geom type="sphere" size="0.1"/></body>'),
     "body-too-many-dofs"),
    ("inverted_joint_range",
     M('<body name="b"><joint name="j" type="hinge" axis="0 1 0"'
       ' limited="true" range="1 -1"/><geom type="sphere" size="0.1"/></body>'),
     "inverted-joint-range"),
    ("equal_joint_range",
     M('<body name="b"><joint name="j" type="hinge" axis="0 1 0"'
       ' limited="true" range="1 1"/><geom type="sphere" size="0.1"/></body>'),
     "inverted-joint-range"),
    ("zero_size_sphere",
     M('<body name="b"><geom name="g" type="sphere" size="0"/></body>'),
     "nonpositive-geom-size"),
    ("zero_len_capsule",
     M('<body name="b"><geom name="g" type="capsule" size="0.1 0"/></body>'),
     "nonpositive-geom-size"),
    ("plane_zero_grid",
     M('<geom name="g" type="plane" size="1 1 0"/>'),
     "nonpositive-geom-size"),
    ("invalid_ctrlrange",
     M(B, '<actuator><motor name="m" joint="j" ctrllimited="true"'
          ' ctrlrange="0 0"/></actuator>'),
     "invalid-ctrlrange"),
    ("zero_hinge_axis",
     M('<body name="b"><joint name="j" type="hinge" axis="0 0 0"/>'
       '<geom type="sphere" size="0.1"/></body>'),
     "zero-joint-axis"),
    ("duplicate_body_name",
     M('<body name="b"><geom type="sphere" size="0.1"/></body>'
       '<body name="b"><geom type="sphere" size="0.1"/></body>'),
     "duplicate-name"),
    ("duplicate_geom_name",
     M('<body name="b"><geom name="g" type="sphere" size="0.1"/>'
       '<geom name="g" type="sphere" size="0.1"/></body>'),
     "duplicate-name"),
    ("plane_in_moving_body",
     M('<body name="b"><joint name="j" type="hinge" axis="0 1 0"/>'
       '<geom type="sphere" size="0.1"/>'
       '<geom name="p" type="plane" size="1 1 1"/></body>'),
     "plane-in-moving-body"),
    ("mocap_with_joint",
     M('<body name="b" mocap="true"><joint name="j" type="hinge"'
       ' axis="0 1 0"/><geom type="sphere" size="0.1"/></body>'),
     "mocap-not-world-child"),
    ("nested_mocap",
     M('<body name="p"><geom type="sphere" size="0.1"/>'
       '<body name="c" mocap="true"><geom type="sphere" size="0.1"/>'
       '</body></body>'),
     "mocap-not-world-child"),
    ("rotation_after_ball",
     M('<body name="b"><joint name="q" type="ball"/>'
       '<joint name="h" type="hinge" axis="0 1 0"/>'
       '<geom type="sphere" size="0.1"/></body>'),
     "rotation-after-ball"),
    # ⚠ MuJoCo ACCEPTS this one — see _UNSUPPORTED_HERE.
    ("generator_replicate",
     M('<replicate count="8" offset="0 0 0.1">'
       '<body name="b"><geom type="sphere" size="0.1"/></body></replicate>'),
     "generator-unsupported"),
    # ── things MuJoCo ACCEPTS, and we must NOT call errors ─────────────────
    ("accept_self_pair",
     M(B, '<contact><pair geom1="g" geom2="g"/></contact>'), ""),
    ("accept_self_exclude",
     M(B, '<contact><exclude body1="b" body2="b"/></contact>'), ""),
    ("accept_zero_gear",
     M(B, '<actuator><motor name="m" joint="j" gear="0"/></actuator>'), ""),
    ("accept_negative_damping",
     M('<body name="b"><joint name="j" type="hinge" axis="0 1 0"'
       ' damping="-1"/><geom type="sphere" size="0.1"/></body>'), ""),
    ("accept_noncollide_geom",
     M('<body name="b"><joint name="j" type="hinge" axis="0 1 0"/>'
       '<geom name="g" type="sphere" size="0.1" contype="0"'
       ' conaffinity="0"/></body>'), ""),
    ("accept_unlimited_inverted_range",
     M('<body name="b"><joint name="j" type="hinge" axis="0 1 0"'
       ' limited="false" range="1 -1"/><geom type="sphere"'
       ' size="0.1"/></body>'), ""),
    ("accept_infinite_plane",
     M('<geom name="g" type="plane" size="0 0 1"/>'), ""),
    ("accept_zero_mass_static_body",
     M('<body name="b"><inertial pos="0 0 0" mass="0" diaginertia="0 0 0"/>'
       '<geom name="g" type="sphere" size="0.1"/></body>'), ""),
    ("accept_mass_from_static_child",
     M('<body name="p"><joint name="j" type="hinge" axis="0 1 0"/>'
       '<body name="c"><geom type="sphere" size="0.1"/></body></body>'), ""),
    ("accept_body_and_geom_share_a_name",
     M('<body name="x"><joint name="j" type="hinge" axis="0 1 0"/>'
       '<geom name="x" type="sphere" size="0.1"/></body>'), ""),
    ("accept_ball_zero_axis",
     M('<body name="b"><joint name="j" type="ball" axis="0 0 0"/>'
       '<geom type="sphere" size="0.1"/></body>'), ""),
]


# Fixtures this engine refuses even though MuJoCo loads them, because the
# feature is not implemented and building a fraction of the model silently is
# the worse answer. KEEP THIS LIST SHORT AND EXPLICIT.
_UNSUPPORTED_HERE = {"generator_replicate"}


def mojo_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> None:
    rows = []
    n_refuse = 0
    for name, xml, code in CASES:
        try:
            mujoco.MjModel.from_xml_string(xml)
            refuses = False
        except Exception:
            refuses = True
        n_refuse += refuses
        unsupported = name in _UNSUPPORTED_HERE
        if bool(code) != (refuses or unsupported):
            raise SystemExit(
                f"FIXTURE DISAGREES WITH ITS OWN INTENT: {name} — MuJoCo "
                f"{'refuses' if refuses else 'accepts'} it but the expected "
                f"code is {code!r}. Fix the fixture, not the table."
            )
        rows.append((name, xml, code, refuses))

    out = [
        '"""MuJoCo 3.10.0\'s verdict on each validator fixture — GENERATED.',
        "",
        "⚠ DO NOT EDIT. Regenerate with:",
        "    pixi run python scripts/dump_mujoco_validity.py",
        "",
        "Each row is (name, xml, expected_code, mujoco_refuses). The XML here",
        "is the exact text MuJoCo judged, which is the point of generating it:",
        "a fixture edited in the test but not re-judged would leave the gate",
        "comparing our validator against a stale verdict.",
        '"""',
        "",
        "",
        "def validity_case_count() -> Int:",
        f"    return {len(rows)}",
        "",
        "",
        "def validity_name(i: Int) -> String:",
    ]
    for k, (name, _, _, _) in enumerate(rows):
        out.append(f"    if i == {k}:")
        out.append(f"        return String({mojo_str(name)})")
    out += ['    return String("")', "", "", "def validity_xml(i: Int) -> String:"]
    for k, (_, xml, _, _) in enumerate(rows):
        out.append(f"    if i == {k}:")
        out.append(f"        return String({mojo_str(xml)})")
    out += ['    return String("")', "", "", "def validity_expected_code(i: Int) -> String:"]
    for k, (_, _, code, _) in enumerate(rows):
        out.append(f"    if i == {k}:")
        out.append(f"        return String({mojo_str(code)})")
    out += ['    return String("")', "", "", "def validity_mujoco_refuses(i: Int) -> Bool:"]
    for k, (_, _, _, refuses) in enumerate(rows):
        out.append(f"    if i == {k}:")
        out.append(f"        return {refuses}")
    out += ["    return False", ""]

    path = "tests/physics3d/validity_goldens.mojo"
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {path}: {len(rows)} cases, "
          f"{n_refuse} refused by MuJoCo, {len(rows) - n_refuse} accepted, "
          f"{len(_UNSUPPORTED_HERE)} unsupported-here")


if __name__ == "__main__":
    main()
