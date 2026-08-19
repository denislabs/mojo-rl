"""MuJoCo 3.10.0's verdict on each validator fixture — GENERATED.

⚠ DO NOT EDIT. Regenerate with:
    pixi run python scripts/dump_mujoco_validity.py

Each row is (name, xml, expected_code, mujoco_refuses). The XML here
is the exact text MuJoCo judged, which is the point of generating it:
a fixture edited in the test but not re-judged would leave the gate
comparing our validator against a stale verdict.
"""


def validity_case_count() -> Int:
    return 35


def validity_name(i: Int) -> String:
    if i == 0:
        return String("ok_baseline")
    if i == 1:
        return String("dangling_actuator_joint")
    if i == 2:
        return String("dangling_pair_geom")
    if i == 3:
        return String("dangling_exclude_body")
    if i == 4:
        return String("dangling_equality_body")
    if i == 5:
        return String("dangling_class")
    if i == 6:
        return String("zero_mass_moving_body")
    if i == 7:
        return String("moving_body_no_geom")
    if i == 8:
        return String("free_joint_nested")
    if i == 9:
        return String("two_free_joints")
    if i == 10:
        return String("inverted_joint_range")
    if i == 11:
        return String("equal_joint_range")
    if i == 12:
        return String("zero_size_sphere")
    if i == 13:
        return String("zero_len_capsule")
    if i == 14:
        return String("plane_zero_grid")
    if i == 15:
        return String("invalid_ctrlrange")
    if i == 16:
        return String("zero_hinge_axis")
    if i == 17:
        return String("duplicate_body_name")
    if i == 18:
        return String("duplicate_geom_name")
    if i == 19:
        return String("plane_in_moving_body")
    if i == 20:
        return String("mocap_with_joint")
    if i == 21:
        return String("nested_mocap")
    if i == 22:
        return String("rotation_after_ball")
    if i == 23:
        return String("generator_replicate")
    if i == 24:
        return String("accept_self_pair")
    if i == 25:
        return String("accept_self_exclude")
    if i == 26:
        return String("accept_zero_gear")
    if i == 27:
        return String("accept_negative_damping")
    if i == 28:
        return String("accept_noncollide_geom")
    if i == 29:
        return String("accept_unlimited_inverted_range")
    if i == 30:
        return String("accept_infinite_plane")
    if i == 31:
        return String("accept_zero_mass_static_body")
    if i == 32:
        return String("accept_mass_from_static_child")
    if i == 33:
        return String("accept_body_and_geom_share_a_name")
    if i == 34:
        return String("accept_ball_zero_axis")
    return String("")


def validity_xml(i: Int) -> String:
    if i == 0:
        return String("<mujoco><worldbody><body name=\"b\" pos=\"0 0 1\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 1:
        return String("<mujoco><worldbody><body name=\"b\" pos=\"0 0 1\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody><actuator><motor name=\"m\" joint=\"nope\"/></actuator></mujoco>")
    if i == 2:
        return String("<mujoco><worldbody><body name=\"b\" pos=\"0 0 1\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody><contact><pair geom1=\"g\" geom2=\"nope\"/></contact></mujoco>")
    if i == 3:
        return String("<mujoco><worldbody><body name=\"b\" pos=\"0 0 1\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody><contact><exclude body1=\"b\" body2=\"nope\"/></contact></mujoco>")
    if i == 4:
        return String("<mujoco><worldbody><body name=\"b\" pos=\"0 0 1\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody><equality><weld body1=\"b\" body2=\"nope\"/></equality></mujoco>")
    if i == 5:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"g\" class=\"nope\" type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 6:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><inertial pos=\"0 0 0\" mass=\"0\" diaginertia=\"0 0 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 7:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><site name=\"s\"/></body></worldbody></mujoco>")
    if i == 8:
        return String("<mujoco><worldbody><body name=\"p\"><geom type=\"sphere\" size=\"0.1\"/><body name=\"c\"><joint name=\"f\" type=\"free\"/><geom type=\"sphere\" size=\"0.1\"/></body></body></worldbody></mujoco>")
    if i == 9:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"f1\" type=\"free\"/><joint name=\"f2\" type=\"free\"/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 10:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\" limited=\"true\" range=\"1 -1\"/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 11:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\" limited=\"true\" range=\"1 1\"/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 12:
        return String("<mujoco><worldbody><body name=\"b\"><geom name=\"g\" type=\"sphere\" size=\"0\"/></body></worldbody></mujoco>")
    if i == 13:
        return String("<mujoco><worldbody><body name=\"b\"><geom name=\"g\" type=\"capsule\" size=\"0.1 0\"/></body></worldbody></mujoco>")
    if i == 14:
        return String("<mujoco><worldbody><geom name=\"g\" type=\"plane\" size=\"1 1 0\"/></worldbody></mujoco>")
    if i == 15:
        return String("<mujoco><worldbody><body name=\"b\" pos=\"0 0 1\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody><actuator><motor name=\"m\" joint=\"j\" ctrllimited=\"true\" ctrlrange=\"0 0\"/></actuator></mujoco>")
    if i == 16:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"j\" type=\"hinge\" axis=\"0 0 0\"/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 17:
        return String("<mujoco><worldbody><body name=\"b\"><geom type=\"sphere\" size=\"0.1\"/></body><body name=\"b\"><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 18:
        return String("<mujoco><worldbody><body name=\"b\"><geom name=\"g\" type=\"sphere\" size=\"0.1\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 19:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom type=\"sphere\" size=\"0.1\"/><geom name=\"p\" type=\"plane\" size=\"1 1 1\"/></body></worldbody></mujoco>")
    if i == 20:
        return String("<mujoco><worldbody><body name=\"b\" mocap=\"true\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 21:
        return String("<mujoco><worldbody><body name=\"p\"><geom type=\"sphere\" size=\"0.1\"/><body name=\"c\" mocap=\"true\"><geom type=\"sphere\" size=\"0.1\"/></body></body></worldbody></mujoco>")
    if i == 22:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"q\" type=\"ball\"/><joint name=\"h\" type=\"hinge\" axis=\"0 1 0\"/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 23:
        return String("<mujoco><worldbody><replicate count=\"8\" offset=\"0 0 0.1\"><body name=\"b\"><geom type=\"sphere\" size=\"0.1\"/></body></replicate></worldbody></mujoco>")
    if i == 24:
        return String("<mujoco><worldbody><body name=\"b\" pos=\"0 0 1\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody><contact><pair geom1=\"g\" geom2=\"g\"/></contact></mujoco>")
    if i == 25:
        return String("<mujoco><worldbody><body name=\"b\" pos=\"0 0 1\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody><contact><exclude body1=\"b\" body2=\"b\"/></contact></mujoco>")
    if i == 26:
        return String("<mujoco><worldbody><body name=\"b\" pos=\"0 0 1\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody><actuator><motor name=\"m\" joint=\"j\" gear=\"0\"/></actuator></mujoco>")
    if i == 27:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\" damping=\"-1\"/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 28:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\" contype=\"0\" conaffinity=\"0\"/></body></worldbody></mujoco>")
    if i == 29:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\" limited=\"false\" range=\"1 -1\"/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 30:
        return String("<mujoco><worldbody><geom name=\"g\" type=\"plane\" size=\"0 0 1\"/></worldbody></mujoco>")
    if i == 31:
        return String("<mujoco><worldbody><body name=\"b\"><inertial pos=\"0 0 0\" mass=\"0\" diaginertia=\"0 0 0\"/><geom name=\"g\" type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 32:
        return String("<mujoco><worldbody><body name=\"p\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><body name=\"c\"><geom type=\"sphere\" size=\"0.1\"/></body></body></worldbody></mujoco>")
    if i == 33:
        return String("<mujoco><worldbody><body name=\"x\"><joint name=\"j\" type=\"hinge\" axis=\"0 1 0\"/><geom name=\"x\" type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    if i == 34:
        return String("<mujoco><worldbody><body name=\"b\"><joint name=\"j\" type=\"ball\" axis=\"0 0 0\"/><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>")
    return String("")


def validity_expected_code(i: Int) -> String:
    if i == 0:
        return String("")
    if i == 1:
        return String("dangling-ref")
    if i == 2:
        return String("dangling-ref")
    if i == 3:
        return String("dangling-ref")
    if i == 4:
        return String("dangling-ref")
    if i == 5:
        return String("dangling-ref")
    if i == 6:
        return String("zero-mass-moving-body")
    if i == 7:
        return String("zero-mass-moving-body")
    if i == 8:
        return String("free-joint-nested")
    if i == 9:
        return String("body-too-many-dofs")
    if i == 10:
        return String("inverted-joint-range")
    if i == 11:
        return String("inverted-joint-range")
    if i == 12:
        return String("nonpositive-geom-size")
    if i == 13:
        return String("nonpositive-geom-size")
    if i == 14:
        return String("nonpositive-geom-size")
    if i == 15:
        return String("invalid-ctrlrange")
    if i == 16:
        return String("zero-joint-axis")
    if i == 17:
        return String("duplicate-name")
    if i == 18:
        return String("duplicate-name")
    if i == 19:
        return String("plane-in-moving-body")
    if i == 20:
        return String("mocap-not-world-child")
    if i == 21:
        return String("mocap-not-world-child")
    if i == 22:
        return String("rotation-after-ball")
    if i == 23:
        return String("generator-unsupported")
    if i == 24:
        return String("")
    if i == 25:
        return String("")
    if i == 26:
        return String("")
    if i == 27:
        return String("")
    if i == 28:
        return String("")
    if i == 29:
        return String("")
    if i == 30:
        return String("")
    if i == 31:
        return String("")
    if i == 32:
        return String("")
    if i == 33:
        return String("")
    if i == 34:
        return String("")
    return String("")


def validity_mujoco_refuses(i: Int) -> Bool:
    if i == 0:
        return False
    if i == 1:
        return True
    if i == 2:
        return True
    if i == 3:
        return True
    if i == 4:
        return True
    if i == 5:
        return True
    if i == 6:
        return True
    if i == 7:
        return True
    if i == 8:
        return True
    if i == 9:
        return True
    if i == 10:
        return True
    if i == 11:
        return True
    if i == 12:
        return True
    if i == 13:
        return True
    if i == 14:
        return True
    if i == 15:
        return True
    if i == 16:
        return True
    if i == 17:
        return True
    if i == 18:
        return True
    if i == 19:
        return True
    if i == 20:
        return True
    if i == 21:
        return True
    if i == 22:
        return True
    if i == 23:
        return False
    if i == 24:
        return False
    if i == 25:
        return False
    if i == 26:
        return False
    if i == 27:
        return False
    if i == 28:
        return False
    if i == 29:
        return False
    if i == 30:
        return False
    if i == 31:
        return False
    if i == 32:
        return False
    if i == 33:
        return False
    if i == 34:
        return False
    return False
