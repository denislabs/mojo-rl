"""The dm_control manipulation reset path — the task-level half.

`manipulation/reach.py::Reach.initialize_episode` is three statements:

    self._hand.set_grasp(physics, close_factors=random_state.uniform())
    self._tcp_initializer(physics, random_state)
    physics.bind(self._target).pos = self._target_placer(random_state)

The middle one is inverse kinematics and lives in
`physics3d/dynamics/ik_site.mojo`, because it is generic engine maths. The
other two are TASK POLICY — which joints count as fingers, which bounding box
a target is sampled from — so they live here rather than pushing dm_control's
task structure down into the physics engine.

⚠ WITH `use_site=True` (the `reach_site_features` task) THE PROP BRANCH IS
DEAD. `initialize_episode` picks `_prop_placer` or `_target_placer` on
`self._prop`, and `_reach(..., use_site=True)` passes `prop=None`. So the
first manipulation task needs NO `PropPlacer` and no settling — a fact worth
stating, because scoping "the reset path" as one lump overstates it
considerably.

⚠ RANDOM DRAWS ARE INJECTED, NOT GENERATED. Every sampler here takes its
values as arguments. dm_control draws them from a numpy `RandomState`, whose
bit stream cannot be reproduced in Mojo; passing them in lets a gate drive
both implementations from the SAME numbers instead of comparing
distributions. Same reasoning as `set_site_to_xpos`'s `retry_poses`.
"""

from std.collections import InlineArray


def set_grasp[
    DTYPE: DType, NHAND: Int
](
    mut qpos: List[Scalar[DTYPE]],
    qpos_adr: InlineArray[Int, NHAND],
    range_min: InlineArray[Float64, NHAND],
    range_max: InlineArray[Float64, NHAND],
    close_factors: InlineArray[Float64, NHAND],
):
    """`kinova/jaco_hand.py::JacoHand.set_grasp`.

    Each finger goes to `min + (max - min) * factor`, so 0 is fully open and
    1 fully closed.

    ⚠ THE REFERENCE TAKES A SCALAR OR A SEQUENCE. `reach.py` passes a scalar
    (`random_state.uniform()`), which it broadcasts to every finger:
    `close_factors = (close_factors,) * len(self.joints)`. This always takes
    the per-finger form; a caller reproducing `reach` fills all NHAND entries
    with the same draw. Keeping the broadcast at the call site means a task
    that DOES vary its fingers needs no second entry point.

    ⚠ NOT DONE HERE, and deliberately: the reference also calls
    `physics.after_reset()` and zeroes the hand actuators' `ctrl`. Those touch
    actuator and derived state that this function is not given, so they belong
    to the env's reset. The `ctrl = 0` in particular is not cosmetic — the
    hand runs `<velocity>` actuators, so a stale non-zero ctrl commands a
    finger velocity from the first step.
    """
    for i in range(NHAND):
        var f = close_factors[i]
        qpos[qpos_adr[i]] = Scalar[DTYPE](
            range_min[i] + (range_max[i] - range_min[i]) * f
        )


def sample_bbox_uniform[
    DTYPE: DType
](
    lower: InlineArray[Float64, 3],
    upper: InlineArray[Float64, 3],
    draws: InlineArray[Float64, 3],
) -> InlineArray[Scalar[DTYPE], 3]:
    """`distributions.Uniform(*bbox)` — one point in an axis-aligned box.

    `draws` are three independent uniforms on [0, 1); the reference obtains
    the same thing as `random_state.uniform(lower, upper)`, which is
    `lower + (upper - lower) * u` componentwise.

    Used twice by `reach` with DIFFERENT boxes, and they are easy to confuse:
    `tcp_bbox` positions the gripper (through the IK initializer) and
    `target_bbox` positions the goal site. For `reach_site_features` both are
    `(-0.2, -0.2, 0.02) .. (0.2, 0.2, 0.4)`, so a test on that task alone
    CANNOT tell them apart — `reach_duplo` is where they differ.
    """
    var out = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    for k in range(3):
        out[k] = Scalar[DTYPE](
            lower[k] + (upper[k] - lower[k]) * draws[k]
        )
    return out^
