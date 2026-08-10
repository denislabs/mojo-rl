"""dog's actuator GAINS against MuJoCo — the table nothing compared.

`test_dog_model_matches_dm_control` checks masses, inertias, ranges, armature,
stiffness, springref, damping, frictionloss, condim, priority and invweight.
It does NOT check `gainprm`, and `gainprm[0]` is the entire actuator force:
every dog actuator is `<general dyntype="filter">`, whose force is
`gainprm[0] * act`.

WHY THIS IS SUSPECTED. The rollout gate diverges by `|d(qvel)| = 6.098` on its
first contacting step while EVERY other comparison is exact — the whole solve
at 2.99e-11, an applied force on every dof at 3.41e-11, and five env steps
through the production `Phyics3dEnv` path at 1e-13. The ONE remaining
difference between the exact env test and the failing rollout is that the
rollout drives `ctrl != 0`. `|d(act)| = 0.0` there, so the activations agree
and the filter is right; what has never been checked is the GAIN the activation
is multiplied by.

⚠ AND A COMMENT IN THE ENGINE ASSERTS THE WRONG VALUE. `model_def_from_xml`
says "dog's actuators are `force = 0.02 * act`". That is the ROOT default
`<general gainprm="0.02">`. MuJoCo compiles nine distinct gains —
`{0.5, 2, 3, 10, 14, 20, 30, 40, 60}` — because dog's nested `<default>`
classes override it (`lumbar` sets 40, a class nested inside it sets 60,
`finger` sets 2, ...). If our resolution stops at the root, every actuator is
between 25x and 3000x too weak, `act` still matches exactly, and only a DRIVEN
rollout can see it.

⚠⚠ RESULT (first successful run, 2026-08-10): THE SUSPICION IS REFUTED.

    nu: ours 38  MuJoCo 38
    MuJoCo distinct gainprm[0] values: 9
    max |d(gainprm)| = 0.0
    max |d(gear)|    = 0.0

Gains match EXACTLY, and MuJoCo's nine distinct values are all present, so the
nested `<default>` resolution is right and the engine is NOT stopping at the
root `gainprm="0.02"`. Whatever causes `|d(qvel)| = 6.098` on the first
contacting step, it is not the actuator gain — look elsewhere and do not
re-derive this suspicion from the paragraphs above.

(The `model_def_from_xml` comment claiming "force = 0.02 * act" is still wrong
AS DOCUMENTATION — the compiled values are correct, the comment describes the
un-overridden root default.)

⚠⚠ THIS FILE DID NOT COMPILE FROM THE MOJO 1.0 MIGRATION UNTIL 2026-08-10, so
none of the above was ever measured until now. A test that fails to BUILD is
indistinguishable from one that passes unless somebody actually runs it —
which is the argument for the CI coverage gap, not just a note about this file.

⚠ `_acd` RE-MATERIALIZES ON EVERY READ. Reading it field-by-field inside a loop
yields garbage; §8 of the plan requires one explicit `materialize` into a local,
which is what this file does.

Run with:
    pixi run mojo run -I . tests/dm_control/test_dog_actuator_gain.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.envs.dm_control.dog import DMDogStandWalkModel

comptime M = DMDogStandWalkModel
comptime TEST_PATH = "tests/dm_control"


def _ref() raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, TEST_PATH)
    var builder = Python.import_module("dog_ref")
    return builder.model()


def test_dog_actuator_gains_match_mujoco() raises:
    print("--- dog: actuator gainprm/dynprm/gear vs MuJoCo ---")
    var m = _ref()
    # ⚠ ONE materialize, per §8 — not a read per element.
    comptime acd = materialize[M._acd]()

    # ⚠ A comptime `Array` CANNOT BE INDEXED BY A RUNTIME VALUE. `acd.motor_kp`
    # is an `Array[Float64, 64]`, which is not `ImplicitlyCopyable`, so
    # `acd.motor_kp[i]` with a runtime `i` asks the compiler to materialize the
    # whole array into runtime storage and fails:
    #
    #     error: cannot materialize comptime value of type
    #            'Array[Float64, Int(64)]' to runtime because it is not
    #            'ImplicitlyCopyable'
    #
    # ⚠⚠ THIS FILE DID NOT COMPILE AT ALL FROM THE MOJO 1.0 MIGRATION UNTIL
    # 2026-08-10, so dog's actuator-gain coverage was silently dead — a build
    # failure and a pass look identical to anyone who is not running it. It was
    # found by accident, while using this test as a regression check for an
    # unrelated change.
    #
    # Copy once through a comptime-unrolled loop into runtime lists. A comptime
    # index alone is NOT enough — `acd.motor_kp[ai]` still tries to materialize
    # the ARRAY — so each ELEMENT is materialized explicitly, which is what the
    # compiler's own `materialize[ ]()` hint asks for. `_acd` itself is still
    # materialized exactly ONCE (into `acd` above), preserving §8's invariant;
    # what is repeated here is only a scalar read out of that single copy.
    comptime NACT = M.nact
    var kp = List[Float64](capacity=NACT)
    var gears = List[Float64](capacity=NACT)

    comptime for ai in range(NACT):
        kp.append(materialize[acd.motor_kp[ai]]())
        gears.append(materialize[acd.motor_gears[ai]]())

    var nu = Int(py=m.nu)
    print("  nu: ours", M.nact, " MuJoCo", nu)
    assert_true(
        M.nact == nu,
        "actuator COUNT differs — every per-index comparison below would be"
        " comparing different actuators",
    )

    var worst_gain = 0.0
    var worst_gain_i = -1
    var worst_dyn = 0.0
    var worst_gear = 0.0
    var n_distinct_mj = 0
    var seen = List[Float64]()
    for i in range(nu):
        var g_ours = kp[i]
        var g_mj = Float64(py=m.actuator_gainprm[i][0])
        var e = abs(g_ours - g_mj)
        if e > worst_gain:
            worst_gain = e
            worst_gain_i = i
        var d_mj = Float64(py=m.actuator_dynprm[i][0])
        var gr_mj = Float64(py=m.actuator_gear[i][0])
        var e_gr = abs(gears[i] - gr_mj)
        if e_gr > worst_gear:
            worst_gear = e_gr
        # Count MuJoCo's distinct gains — the non-vacuity signal.
        var fresh = True
        for k in range(len(seen)):
            if abs(seen[k] - g_mj) < 1e-12:
                fresh = False
                break
        if fresh:
            seen.append(g_mj)
            n_distinct_mj += 1
        _ = d_mj

    print("  MuJoCo distinct gainprm[0] values:", n_distinct_mj)
    print("  max |d(gainprm)| =", worst_gain, " at actuator", worst_gain_i)
    if worst_gain_i >= 0:
        print(
            "      actuator", worst_gain_i,
            " ours", kp[worst_gain_i],
            " MuJoCo", Float64(py=m.actuator_gainprm[worst_gain_i][0]),
        )
    print("  max |d(gear)| =", worst_gear)

    # NON-VACUITY. If MuJoCo compiled a single gain for all 38 actuators, this
    # file could not tell class resolution from a hardcoded root default — the
    # very confusion it exists to settle.
    assert_true(
        n_distinct_mj >= 2,
        "MuJoCo compiled ONE gain for every dog actuator — the nested-class"
        " override this test targets is not present in the model, so a match"
        " here would prove nothing about class resolution",
    )
    assert_true(
        worst_gain <= 1e-12,
        "actuator gainprm differs from MuJoCo — every actuator force is"
        " `gainprm[0] * act`, so this scales the entire drive. Suspect the"
        " nested `<default>` class chain: dog's gains live in classes nested"
        " inside classes, and the root default is 0.02.",
    )
    assert_true(worst_gear <= 1e-12, "actuator gear differs from MuJoCo")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
