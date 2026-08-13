"""Generate `mojo_rl/envs/robots/*_xml.mojo`'s XML from the reference MJCF.

    pixi run python tests/robots/so_arm_bake.py            # print both
    pixi run python tests/robots/so_arm_bake.py so_arm100  # print one

WHY A GENERATOR AND NOT A TRANSCRIPTION. Our parser does not implement three
MJCF attributes these two models use, and all three are *derived* values that
MuJoCo resolves at compile time:

    <position inheritrange="1">   ->  an explicit ctrlrange   (SO-100)
    <position dampratio="1">      ->  an explicit kv          (SO-100)
    <inertial fullinertia="...">  ->  quat + diaginertia      (SO-101)

Every substituted number is READ BACK OUT OF `mjModel` at full precision
(`repr(float)`, 17 significant digits) rather than retyped, so the bake cannot
introduce a rounding difference. ⚠ MuJoCo's own `mj_saveLastXML` would do the
whole job in one call and was tried first — it writes **6 significant digits**
(`biasprm="0 -50 -5.12815"` for a true `-5.12815011462096`), which is a 1e-6
relative error on every derived constant and would fail the layer-1 gate by
construction. That is why this file does targeted substitutions on the
reference TEXT instead.

⚠⚠ THE SUBSTITUTIONS ARE DEVIATIONS AND ARE LABELLED AS SUCH IN THE OUTPUT.
`tests/robots/so_arm_ref.py` is the gate that proves they are *only*
re-spellings: it compiles the reference MJCF and our generated string with the
same MuJoCo and diffs all 97 `mjModel` tables. If any substitution is wrong the
gate fails — that is the whole point of generating rather than hand-writing.
Re-run this after ANY change to the reference tree and diff the output against
the checked-in `.mojo`.

⚠ `<keyframe>` is left in the XML verbatim. Our parser ignores it (a real gap —
`docs/TODDLERBOT_PORT_PLAN.md` §4.6), so it costs nothing here and keeps the
layer-1 diff honest; the poses are ALSO baked into `so_arm100_config.mojo` as
`SO_ARM100_KEY_*` so the env can actually reset to them.

Provenance of the two reference models:
  SO-100  references/mujoco_menagerie-main/trs_so_arm100/     (vendored)
  SO-101  references/SO-ARM100-main/Simulation/SO101/         (vendored from
          https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)
⚠ `references/` is gitignored, so this script is a local tool. The GENERATED
strings are what ships.
"""

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SO100_REF = os.path.join(
    REPO, "references", "mujoco_menagerie-main", "trs_so_arm100"
)
SO101_REF = os.path.join(
    REPO, "references", "SO-ARM100-main", "Simulation", "SO101"
)

SO100_MESHDIR = "mojo_rl/envs/robots/assets/so_arm100/"
SO101_MESHDIR = "mojo_rl/envs/robots/assets/so_arm101/"


def _f(x):
    """Full-precision literal. `repr` round-trips a float64 exactly."""
    return repr(float(x))


def _vec(v):
    return " ".join(_f(x) for x in v)


def _need(path, what):
    if not os.path.isdir(path):
        raise SystemExit(
            "missing reference tree: {}\n"
            "  {} — references/ is gitignored, fetch it locally.".format(path, what)
        )


# ----------------------------------------------------------------------------
# The floor + visual block from each model's scene.xml, inlined.
#
# ⚠ INLINED RATHER THAN <include>d. Our merge path (`merge_mjcf`) exists and
# works, but scene.xml carries a skybox and a `<global azimuth/elevation>` that
# only the renderer reads; splitting one 120-line model across two Mojo string
# constants to mirror an upstream file layout buys nothing and costs a second
# place for the mesh paths to drift. The layer-1 gate compiles the reference's
# scene.xml, so the floor IS compared.
# ----------------------------------------------------------------------------




def _explicit_actuators(src, m, which):
    """Write the named attributes onto every `<position>` element.

    ⚠⚠ FIFTH DEVIATION, AND THE ONE THAT ACTUALLY BROKE BOTH ARMS.
    `<position>`'s `kp` and `kv` are read with `_extract_attr(tag, ...)` — the
    ELEMENT ONLY — while every attribute around them (`gear`, `ctrlrange`,
    `forcerange`, `gainprm`, and `<velocity>`'s own `kv`) goes through
    `_attr_3way_cached`. So a gain declared in ANY `<default>` class is
    missed, and MuJoCo's defaults take over: kp 1, kv 0.

    ⚠ NOT a class-CHAIN problem — chain walking works. Measured with a
    one-class fixture, no nesting: kp still parses as 1.0. An earlier note here
    blamed the chain, and a four-line fixture refutes that.

        SO-100  kp 50 -> 1.0            servo 50x weak; the arm still moved
                                        TOWARD its target, just far too slowly.
        SO-101  kp 998.22 -> 1.0,       torque ~1 N.m short of the gravity
                kv 2.731 -> 0.0         load, so the arm FELL to its joint
                                        limits, 1.26 rad off on shoulder_pan.

    ⚠ An earlier version of this note also claimed SO-101's actuators were
    "demoted to plain <motor> (motor_kind = 0)". That is FALSE — all six are
    kind 1. The 0 came from a probe that materialized four comptime arrays and
    read them afterwards. One root cause, not two.

    Both are SILENT: the model builds, the env steps, the numbers look like a
    badly tuned controller. Only diffing against MuJoCo's own 2 000-step
    rollout showed it, which is why `test_so_arm10x_vs_mujoco.mojo` gates the
    gains explicitly and not just the trajectory.

    Making the element self-describing sidesteps the gap and is invisible to
    MuJoCo (an element attribute overrides a class default), so the layer-1
    diff stays exact.

    ⚠⚠ `kp` IS NO LONGER WRITTEN — the parser resolves it now (fixed
    2026-08-13, `tests/physics3d/test_position_gain_defaults.mojo`). It is
    dropped rather than left in place ON PURPOSE: a workaround kept after its
    cause is fixed silently stops being tested, and `test_actuator_law` on both
    arms is now a live check that the class lookup works on a REAL model
    instead of only on a synthetic fixture. SO-100 needs the chain (`kp` sits
    in the grandparent class `so_arm100`); SO-101 needs only its own class.

    What remains is genuinely underivable by our parser today:
      kv         only on SO-100, where MuJoCo DERIVES it from `dampratio`
      ctrlrange  only on SO-100, where it comes from `inheritrange="1"`
    SO-101 needs neither — upstream writes both on the element — so it no
    longer calls this at all.
    """
    import re as _re

    out, n = src, 0
    for i in range(m.nu):
        name = mujoco_name(m, i)
        pat = _re.compile(r'<position ([^>]*?)name="%s"([^>]*?)/>' % _re.escape(name))
        mo = pat.search(out)
        assert mo is not None, "actuator {} not found".format(name)
        kp = m.actuator_gainprm[i][0]
        kv = -m.actuator_biasprm[i][2]
        flo, fhi = m.actuator_forcerange[i]
        clo, chi = m.actuator_ctrlrange[i]
        # Keep `class=` and `joint=`: the class still supplies joint defaults,
        # and dropping it would change more than the actuator.
        keep = (mo.group(1) + mo.group(2)).strip()
        keep = _re.sub(r'\s*inheritrange="[^"]*"', "", keep)
        keep = _re.sub(r'\s*(kp|kv|forcerange|ctrlrange)="[^"]*"', "", keep)
        parts = ["<position", keep, 'name="{}"'.format(name)]
        if "kp" in which:
            parts.append('kp="{}"'.format(_f(kp)))
        if "kv" in which:
            parts.append('kv="{}"'.format(_f(kv)))
        if "forcerange" in which:
            parts.append('forcerange="{} {}"'.format(_f(flo), _f(fhi)))
        if "ctrlrange" in which:
            parts.append('ctrlrange="{} {}"'.format(_f(clo), _f(chi)))
        new = " ".join(parts) + "/>"
        out = out[: mo.start()] + new + out[mo.end() :]
        n += 1
    assert n == m.nu, "rewrote {} of {} actuators".format(n, m.nu)
    return out


def mujoco_name(m, i):
    import mujoco

    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)


def bake_so_arm100():
    import mujoco

    _need(SO100_REF, "Menagerie trs_so_arm100")
    m = mujoco.MjModel.from_xml_path(os.path.join(SO100_REF, "scene.xml"))
    src = open(os.path.join(SO100_REF, "so_arm100.xml")).read()

    # 1. meshdir REPOINTED at our vendored copy — the attribute stays.
    #
    # ⚠ This used to drop `meshdir` and bake a full path onto every `file=`,
    # because `<compiler meshdir>` was UNPARSED and every collidable mesh
    # silently failed to open (`fields_build` printed a warning and carried
    # on, leaving the arm with no mesh collision at all). The parser resolves
    # it as of 2026-08-13 — `tests/physics3d/test_compiler_meshdir.mojo` —
    # so the workaround is GONE and these two models are now a live check of
    # it on real geometry, rather than a fixture-only guarantee.
    #
    # ⚠ The path is REPO-ROOT RELATIVE: MuJoCo resolves `meshdir` against the
    # model FILE, we resolve it against the process CWD. Anything loading
    # these models must run from the repo root.
    src = src.replace('meshdir="assets/"', 'meshdir="{}"'.format(SO100_MESHDIR))
    assert SO100_MESHDIR in src, "meshdir substitution missed"

    # 2. dampratio -> nothing here; kv is per-actuator (it depends on each
    #    joint's effective inertia), so the class default only keeps kp and
    #    forcerange and every actuator gets its own kv below.
    before = src
    src = src.replace(
        '<position kp="50" dampratio="1" forcerange="-3.5 3.5"/>',
        '<position kp="50" forcerange="-3.5 3.5"/>'
        "  <!-- DEVIATION: dampratio -> per-actuator kv, see so_arm_bake.py -->",
    )
    assert src != before, "dampratio substitution missed"
    # ⚠ the ATTRIBUTE, not the word — the DEVIATION comment names it too.
    assert 'dampratio="' not in src

    # 3. Only what our parser still cannot derive: `kv` (MuJoCo computes it
    #    from `dampratio`) and `ctrlrange` (from `inheritrange="1"`). `kp` is
    #    left in the class on purpose — see `_explicit_actuators`.
    src = _explicit_actuators(src, m, ("kv", "ctrlrange"))
    assert 'inheritrange="' not in src

    # 4. Inline scene.xml's floor + ground material.
    src = src.replace(
        "  </worldbody>",
        "\n".join(
            [
                "    <!-- from trs_so_arm100/scene.xml -->",
                '    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>',
                '    <geom name="floor" size="0 0 0.05" type="plane"'
                ' material="groundplane"/>',
                "  </worldbody>",
            ]
        ),
    )
    src = src.replace(
        "  </asset>",
        "\n".join(
            [
                "    <!-- from trs_so_arm100/scene.xml -->",
                '    <texture type="2d" name="groundplane" builtin="checker"'
                ' mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"',
                '      markrgb="0.8 0.8 0.8" width="300" height="300"/>',
                '    <material name="groundplane" texture="groundplane"'
                ' texuniform="true" texrepeat="5 5" reflectance="0.2"/>',
                "  </asset>",
            ]
        ),
    )
    return src


def bake_so_arm101(calib="new"):
    import mujoco

    _need(SO101_REF, "TheRobotStudio SO-ARM100/Simulation/SO101")
    m = mujoco.MjModel.from_xml_path(os.path.join(SO101_REF, "scene.xml"))
    src = open(
        os.path.join(SO101_REF, "so101_{}_calib.xml".format(calib))
    ).read()

    # 1. meshdir repointed. See `bake_so_arm100` step 1 — the full-path
    #    workaround is gone now that the parser resolves the attribute.
    src = src.replace('meshdir="assets"', 'meshdir="{}"'.format(SO101_MESHDIR))
    assert SO101_MESHDIR in src, "meshdir substitution missed"

    # 2. fullinertia -> quat + diaginertia, per body, at full precision.
    #
    # ⚠⚠ THIS IS THE DEVIATION THAT MATTERS. MuJoCo diagonalises the symmetric
    # 3x3 with `mjuu_eig3`; the values below ARE that routine's output, read
    # back out of mjModel. `body_iquat` is stored (w, x, y, z) — MJCF's `quat`
    # order — so it is written straight through.
    #
    # ⚠ Gate the QUATERNION, not just the moments. A wrong iquat with correct
    # diaginertia leaves total mass and every scalar moment right while
    # silently rotating each body's inertia frame. `so_arm_ref.py` diffs
    # `body_iquat` and `body_inertia` per body for exactly this reason.
    n_sub = 0
    for b in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        pat = re.compile(
            r'<inertial pos="([^"]*)" mass="([^"]*)" fullinertia="([^"]*)"/>'
        )
        # Bodies appear in document order; substitute the FIRST remaining
        # match each time and assert we consumed exactly one per body.
        mo = pat.search(src)
        if mo is None:
            break
        new = (
            '<inertial pos="{}" quat="{}" mass="{}" diaginertia="{}"/>'
            "<!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->"
        ).format(
            _vec(m.body_ipos[b]),
            _vec(m.body_iquat[b]),
            _f(m.body_mass[b]),
            _vec(m.body_inertia[b]),
        )
        src = src[: mo.start()] + new + src[mo.end() :]
        n_sub += 1
    assert n_sub == m.nbody - 1, "substituted {} of {} bodies".format(
        n_sub, m.nbody - 1
    )
    assert 'fullinertia="' not in src

    # 3. Two top-level <default> blocks -> one. The reference emits the
    #    `so101_new_calib` classes and then a second block holding `sts3215`
    #    and `backlash`; MuJoCo merges them, and relying on our parser to do
    #    the same would be an untested assumption inside an untested port.
    #
    # ⚠ This comment used to read "...SECOND top-level <default>", and that
    # literal `<default>` inside a COMMENT made `merge_mjcf` delete the whole
    # `<default>` section. Fixed 2026-08-13 (comments are stripped before
    # scanning), so the wording no longer matters — kept angle-bracket free
    # anyway, because nothing else in the tree depends on that fix holding.
    src = src.replace(
        "  </default>\n  <!-- Additional joints_properties.xml -->\n  <default>\n",
        "    <!-- merged from the reference's SECOND top-level default block;"
        " see so_arm_bake.py -->\n",
        1,
    )

    # 3b. NOTHING to rewrite on the actuators any more. `kp`/`kv`/`forcerange`
    #     live in the `sts3215` class and the parser resolves all three;
    #     `ctrlrange` is already on the element upstream. This call used to
    #     write all four — dropping it is the PROOF that the class lookup
    #     works on a real model, and `test_so_arm101_vs_mujoco::
    #     test_actuator_law` is what would catch a regression.

    # 4. Explicit mesh names. The reference writes `<mesh file="x.stl"/>` and
    #    leans on MuJoCo naming the asset after the stem.
    def _name_mesh(mo):
        stem = os.path.splitext(os.path.basename(mo.group(1)))[0]
        return '<mesh name="{}" file="{}"/>'.format(stem, mo.group(1))

    src = re.sub(r'<mesh file="([^"]*)"/>', _name_mesh, src)

    # 5. Inline scene.xml's floor.
    src = src.replace(
        "  </worldbody>",
        "\n".join(
            [
                "    <!-- from SO101/scene.xml -->",
                '    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>',
                '    <geom name="floor" size="0 0 0.05" pos="0 0 0"'
                ' type="plane" material="groundplane"/>',
                "  </worldbody>",
            ]
        ),
    )
    src = src.replace(
        "  </asset>",
        "\n".join(
            [
                "    <!-- from SO101/scene.xml -->",
                '    <texture type="2d" name="groundplane" builtin="checker"'
                ' mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"',
                '      markrgb="0.8 0.8 0.8" width="300" height="300"/>',
                '    <material name="groundplane" texture="groundplane"'
                ' texuniform="true" texrepeat="5 5" reflectance="0.2"/>',
                "  </asset>",
            ]
        ),
    )
    return src


# ---------------------------------------------------------------------------
# The reach target, inlined into the robot XML to produce the FULL model.
#
# ⚠ NOT `merge_mjcf` — kept for simplicity, no longer for correctness.
# That call did mangle SO-101 once (`<default>` vanished; MuJoCo refused the
# model with "unknown default class name 'sts3215'"), but NOT because the
# defaults are nested. `_extract_section_inner` depth-counted raw text without
# stripping comments, and the comment this bake inserted contained the literal
# `<default>`. FIXED 2026-08-13 — `merge_mjcf` strips comments first, gated by
# `tests/physics3d/test_merge_mjcf_comments.mojo`. Emitting the finished model
# directly is simply fewer moving parts. ⚠ Do not inherit "merge_mjcf cannot
# do nested defaults" from this comment — it never could not.
#
# The target is a MOCAP body, for the reason `reacher_xml` documents: the
# per-episode target must be per-ENV state, and `Model` is not batched while
# `Data.mocap_pos` is. Appended last, so its body/geom indices are the final
# ones in both MuJoCo's ordering and our parser's — `test_so_arm10x_vs_mujoco`
# pins that by NAME rather than trusting it.
TASK_BODY = """    <body name="target" mocap="true" pos="{}">
      <geom name="target" type="sphere" size="0.012" rgba="0.9 0.1 0.1 0.6"
            contype="0" conaffinity="0" group="1"/>
    </body>
"""

TARGET_POS = {"so_arm100": "0 -0.25 0.15", "so_arm101": "0.25 0 0.2"}


def with_task(src, which):
    """Robot XML -> full model XML, by inserting the target before </worldbody>."""
    marker = "  </worldbody>"
    assert src.count(marker) == 1, "expected exactly one </worldbody>"
    return src.replace(
        marker, TASK_BODY.format(TARGET_POS[which]) + marker
    )


BAKERS = {"so_arm100": bake_so_arm100, "so_arm101": bake_so_arm101}

# The generated XML lives between these markers in `*_xml.mojo`. `--inject`
# rewrites only that region, so the module's prose survives regeneration and
# `so_arm_ref.py` can extract exactly what ships.
BEGIN = "# --- BEGIN GENERATED XML (tests/robots/so_arm_bake.py) ---"
END = "# --- END GENERATED XML ---"


def module_path(which):
    return os.path.join(REPO, "mojo_rl", "envs", "robots", which + "_xml.mojo")


def extract(which):
    """The XML string as it is CHECKED IN — what `so_arm_ref.py` gates."""
    text = open(module_path(which)).read()
    i = text.index(BEGIN) + len(BEGIN)
    j = text.index(END)
    body = text[i:j]
    key = "_ROBOT_XML = \"\"\""
    a = body.index(key) + len(key)
    b = body.index('"""', a)
    return body[a:b]


def extract_full(which):
    """The FULL model string (robot + task) as checked in."""
    text = open(module_path(which)).read()
    i = text.index(BEGIN) + len(BEGIN)
    j = text.index(END)
    body = text[i:j]
    key = "_XML = \"\"\""
    # ⚠ `_ROBOT_XML` also ends in `_XML`; search past the robot constant.
    a = body.index(key, body.index("_ROBOT_XML") + 10) + len(key)
    b = body.index('"""', a)
    return body[a:b]


def inject(which):
    """Write BOTH generated constants: the gated robot, and the full model."""
    path = module_path(which)
    text = open(path).read()
    i = text.index(BEGIN) + len(BEGIN)
    j = text.index(END)
    stem = which.upper().replace("SO_ARM", "SO_ARM")
    robot = BAKERS[which]()
    full = with_task(robot, which)
    new = (
        '\ncomptime {0}_ROBOT_XML = """{1}"""\n'
        '\ncomptime {0}_XML = """{2}"""\n'
    ).format(stem, robot, full)
    open(path, "w").write(text[:i] + new + text[j:])
    return path


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--inject":
        for w in args[1:] or list(BAKERS):
            print("wrote", inject(w))
    else:
        for w in args or list(BAKERS):
            print("=" * 78)
            print("==  {}".format(w))
            print("=" * 78)
            print(BAKERS[w]())
