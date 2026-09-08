"""Bake the Unitree G1 (29-DoF) env assets from the BFM-Zero reference tree.

    pixi run python tests/robots/g1_bake.py            # write the assets
    pixi run python tests/robots/g1_bake.py --check    # CI: fail on any diff

WHAT IT EMITS, and from where:

    mojo_rl/envs/robots/assets/unitree_g1.xml
        The reference's sim-to-sim MuJoCo model — the file BFM-Zero's own
        `--simulator mujoco` path loads and the one its "MuJoCo (DR)" table
        row was produced with —
            references/BFM-Zero-main/humanoidverse/data/robots/g1/
                scene_29dof_freebase_noadditional_actuators.xml
                g1_29dof_old_freebase_noadditional_actuators.xml
        with the `<include>` inlined, the two `<sensor>` blocks dropped, the
        `<compiler meshdir>` pointed at our copy of the meshes, and ONE
        deviation stated below.
    mojo_rl/envs/robots/assets/unitree_g1/*.STL
        The 36 meshes the model names (18 MB), copied verbatim.
    mojo_rl/envs/robots/unitree_g1_pd.mojo
        The PD controller tables the reference computes torques with, read
        out of `config/robot/g1/g1_29dof_hard_waist.yaml` by the reference's
        own substring rule and emitted per DoF in the model's joint order:
        kp, kd, effort limit, default joint angle, position limits.

⚠ THE ONE DEVIATION: `<option timestep="0.005"/>`. The XML carries MuJoCo's
default 0.002, and the reference OVERRIDES it at runtime —
`simulator/mujoco/mujoco.py:47` sets `opt.timestep = 1 / sim.fps` with
`fps: 200`, then steps `control_decimation: 4` substeps per 50 Hz control
step. Baking the override into the asset makes the file our env loads say
what the reference actually simulates, and `g1_ref.py` applies the same
override to the reference model before diffing, so the gate compares like
with like rather than skipping `opt.timestep`.

⚠ `<sensor>` IS DROPPED, NOT PORTED. 104 sensors, none read by the training
loop or by the observation (the obs is built from `qpos`, `qvel`, the root
quaternion and body FK). `g1_ref.py` skips the four sensor tables for that
reason and says so.

⚠ NOTHING ELSE IS RE-SPELLED. Unlike the SO-ARM bakes there is no
`inheritrange`, `dampratio` or `fullinertia` here: every actuator is a plain
`<motor ctrlrange>` and every inertial is `diaginertia` + `quat`. The PD
controller is NOT in the XML at all — the reference applies it as a torque
law in the env — which is why the gains live in a generated `.mojo` table
and not in the model.

⚠ `references/` IS GITIGNORED. This is a local tool; the generated files are
what ships.
"""

import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(REPO, "references", "BFM-Zero-main", "humanoidverse")
REF_G1 = os.path.join(REF, "data", "robots", "g1")
REF_YAML = os.path.join(REF, "config", "robot", "g1", "g1_29dof_hard_waist.yaml")
SCENE = "scene_29dof_freebase_noadditional_actuators.xml"
ROBOT = "g1_29dof_old_freebase_noadditional_actuators.xml"

ASSET_DIR = os.path.join(REPO, "mojo_rl", "envs", "robots", "assets")
ASSET_XML = os.path.join(ASSET_DIR, "unitree_g1.xml")
MESH_DIR = os.path.join(ASSET_DIR, "unitree_g1")
PD_MOJO = os.path.join(REPO, "mojo_rl", "envs", "robots", "unitree_g1_pd.mojo")

# `simulator/mujoco/mujoco.py:47` — `opt.timestep = 1 / sim.fps`, fps 200.
SIM_TIMESTEP = 1.0 / 200.0
# `config/simulator/mujoco.yaml` — `control_decimation: 4`.
CONTROL_DECIMATION = 4


def _need(path, what):
    if not os.path.exists(path):
        raise SystemExit(
            "missing reference file: {}\n  {} — references/ is gitignored,"
            " fetch it locally.".format(path, what)
        )


def _f(x):
    """Full-precision literal. `repr` round-trips a float64 exactly."""
    s = repr(float(x))
    return s if ("." in s or "e" in s) else s + ".0"


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def bake_xml():
    _need(os.path.join(REF_G1, SCENE), "BFM-Zero sim-to-sim scene")
    _need(os.path.join(REF_G1, ROBOT), "BFM-Zero G1 robot MJCF")
    scene = open(os.path.join(REF_G1, SCENE)).read()
    robot = open(os.path.join(REF_G1, ROBOT)).read()

    # Drop the sensor blocks (there are two).
    robot, n_sensor = re.subn(r"\s*<sensor>.*?</sensor>", "", robot, flags=re.S)
    assert n_sensor == 2, n_sensor
    # Our copy of the meshes lives beside the asset.
    robot, n_dir = re.subn(r'meshdir="meshes"', 'meshdir="unitree_g1"', robot)
    assert n_dir == 1, n_dir
    # The runtime timestep override, baked (see the module docstring).
    assert "<option" not in robot and "<option" not in scene
    robot = robot.replace(
        '<compiler angle="radian" meshdir="unitree_g1" />',
        '<compiler angle="radian" meshdir="unitree_g1" />\n\n'
        '  <!-- DEVIATION: the reference sets this at runtime -->\n'
        '  <option timestep="%s"/>' % _f(SIM_TIMESTEP),
    )
    assert "<option" in robot
    # Inline the robot into the scene at the <include>, as MuJoCo would.
    inner = robot.strip()
    assert inner.startswith('<mujoco model="g1_29dof">') and inner.endswith("</mujoco>")
    inner = inner[len('<mujoco model="g1_29dof">'):-len("</mujoco>")]
    scene, n_inc = re.subn(
        r'\s*<include file="%s"/>' % re.escape(ROBOT),
        "\n  <!-- inlined from %s -->" % ROBOT + inner,
        scene,
    )
    assert n_inc == 1, n_inc
    header = (
        "<!-- GENERATED by tests/robots/g1_bake.py from the BFM-Zero reference"
        " tree — do not edit. -->\n"
    )
    return header + scene


def mesh_files(xml):
    return sorted(set(re.findall(r'<mesh [^>]*file="([^"]+)"', xml)))


# ---------------------------------------------------------------------------
# The PD tables
# ---------------------------------------------------------------------------


def _match_gain(table, dof_name):
    """The reference's rule: the table key that is a SUBSTRING of the dof name.

    `legged_robot_base._setup_robot_body_indices`-style lookup; asserting
    exactly one hit is ours, so an ambiguous key cannot pick silently.
    """
    hits = [k for k in table if k in dof_name]
    assert len(hits) == 1, (dof_name, hits)
    return table[hits[0]]


def read_pd():
    import yaml

    _need(REF_YAML, "BFM-Zero G1 robot config")
    y = yaml.safe_load(open(REF_YAML))
    robot = y["robot"]
    names = robot["dof_names"]
    assert len(names) == 29, len(names)
    ctrl = robot["control"]
    assert ctrl["control_type"] == "P"
    assert ctrl["action_rescale"] is True and ctrl["clip_torques"] is True
    assert ctrl["normalize_action"] is True
    defaults = robot["init_state"]["default_joint_angles"]
    # The yaml spells three keys with a space before the colon.
    defaults = {k.strip(): v for k, v in defaults.items()}
    rows = []
    for i, n in enumerate(names):
        rows.append(dict(
            name=n,
            kp=float(_match_gain(ctrl["stiffness"], n)),
            kd=float(_match_gain(ctrl["damping"], n)),
            effort=float(robot["dof_effort_limit_list"][i]),
            vel_limit=float(robot["dof_vel_limit_list"][i]),
            default=float(defaults[n]),
            lower=float(robot["dof_pos_lower_limit_list"][i]),
            upper=float(robot["dof_pos_upper_limit_list"][i]),
        ))
    scal = dict(
        action_scale=float(ctrl["action_scale"]),
        action_clip=float(ctrl["action_clip_value"]),
        normalize_from=float(ctrl["normalize_action_from"]),
        normalize_to=float(ctrl["normalize_action_to"]),
        root_z=float(robot["init_state"]["pos"][2]),
    )
    return rows, scal


def _table(name, rows, key, doc):
    out = ["\n\n@always_inline\ndef %s(i: Int) -> Float64:\n" % name,
           '    """%s"""\n' % doc]
    for i, r in enumerate(rows):
        kw = "if" if i == 0 else "elif"
        out.append("    %s i == %d:  # %s\n        return %s\n" % (kw, i, r["name"], _f(r[key])))
    out.append("    return 0.0\n")
    return "".join(out)


def bake_pd():
    rows, scal = read_pd()
    out = [
        '"""Unitree G1 PD controller tables — GENERATED, DO NOT EDIT.\n',
        "\n",
        "Regenerate with:  pixi run python tests/robots/g1_bake.py\n",
        "CI checks it with: pixi run python tests/robots/g1_bake.py --check\n",
        "\n",
        "Source of truth is BFM-Zero's `config/robot/g1/g1_29dof_hard_waist.yaml`,\n",
        "read by the reference's own substring rule (a gain key such as\n",
        "`hip_yaw` applies to every dof whose name contains it) and emitted per\n",
        "dof in the model's joint order, which `test_unitree_g1_vs_mujoco`\n",
        "pins against `mjModel`'s joint names.\n",
        "\n",
        "The torque law these feed (`legged_robot_base._compute_torques`):\n",
        "\n",
        "    a       in [-1, 1]                      (the policy's tanh output)\n",
        "    a       *= NORMALIZE_TO / NORMALIZE_FROM   then clipped to +-ACTION_CLIP\n",
        "    target  = a * ACTION_SCALE * effort / kp + default_pos\n",
        "    tau     = kp * (target - q) - kd * qdot,  clipped to +-effort\n",
        "\n",
        "evaluated at EVERY physics substep, and applied through the model's\n",
        "`<motor>` actuators (gear 1), whose `ctrlrange` clamps it AGAIN.\n",
        "\n",
        "!! `ctrlrange` IS NOT `effort` ON FOUR JOINTS: the sim-to-sim XML says\n",
        "+-88 N m on hip pitch/roll where this table (the yaml) says 139. The\n",
        "env applies both, in the reference's order; see `unitree_g1_config`.\n",
        '"""\n',
        "\n",
        "comptime G1_N_DOF: Int = %d\n" % len(rows),
        "comptime G1_ACTION_SCALE: Float64 = %s\n" % _f(scal["action_scale"]),
        "comptime G1_ACTION_CLIP: Float64 = %s\n" % _f(scal["action_clip"]),
        "comptime G1_NORMALIZE_FROM: Float64 = %s\n" % _f(scal["normalize_from"]),
        "comptime G1_NORMALIZE_TO: Float64 = %s\n" % _f(scal["normalize_to"]),
        "# `init_state.pos[2]` — the pelvis height the reference resets to.\n",
        "comptime G1_INIT_ROOT_Z: Float64 = %s\n" % _f(scal["root_z"]),
        "# `1 / sim.fps` and `control_decimation`, from the simulator config.\n",
        "comptime G1_SIM_TIMESTEP: Float64 = %s\n" % _f(SIM_TIMESTEP),
        "comptime G1_CONTROL_DECIMATION: Int = %d\n" % CONTROL_DECIMATION,
    ]
    out.append("\n\n@always_inline\ndef g1_dof_name(i: Int) -> StaticString:\n")
    out.append('    """`dof_names`, the reference\'s dof order — pinned against `mjModel` joint names."""\n')
    for i, r in enumerate(rows):
        kw = "if" if i == 0 else "elif"
        out.append('    %s i == %d:\n        return "%s"\n' % (kw, i, r["name"]))
    out.append('    return ""\n')
    out.append(_table("g1_kp", rows, "kp", "`control.stiffness`, N*m/rad."))
    out.append(_table("g1_kd", rows, "kd", "`control.damping`, N*m*s/rad."))
    out.append(_table("g1_effort", rows, "effort",
                      "`dof_effort_limit_list`, N*m — the torque clip AND the `<motor ctrlrange>`."))
    out.append(_table("g1_default_pos", rows, "default",
                      "`init_state.default_joint_angles`, rad — the PD target at action 0."))
    out.append(_table("g1_pos_lower", rows, "lower", "`dof_pos_lower_limit_list`, rad."))
    out.append(_table("g1_pos_upper", rows, "upper", "`dof_pos_upper_limit_list`, rad."))
    out.append(_table("g1_vel_limit", rows, "vel_limit", "`dof_vel_limit_list`, rad/s."))
    return "".join(out)


# ---------------------------------------------------------------------------
# The reference's torque law in Python, for the layer-2 gate. Independent of
# the generated Mojo tables on purpose: it reads the yaml again and applies
# `_compute_torques` with numpy, so a wrong table shows up as a rollout
# residual rather than being copied onto both sides of the comparison.
# ---------------------------------------------------------------------------


class ReferencePD:
    def __init__(self):
        import numpy as np

        rows, scal = read_pd()
        self.kp = np.array([r["kp"] for r in rows])
        self.kd = np.array([r["kd"] for r in rows])
        self.effort = np.array([r["effort"] for r in rows])
        self.default = np.array([r["default"] for r in rows])
        self.scal = scal

    def target(self, a):
        """`a` in [-1, 1] -> joint targets, `legged_robot_base.py:222-297`."""
        import numpy as np

        a = np.asarray(a, dtype=np.float64)
        a = a * (self.scal["normalize_to"] / self.scal["normalize_from"])
        a = np.clip(a, -self.scal["action_clip"], self.scal["action_clip"])
        scaled = a * self.scal["action_scale"] * self.effort / self.kp
        return scaled + self.default

    def torque(self, target, q, qd):
        """`_compute_torques`, control_type P, clip_torques True."""
        import numpy as np

        tau = self.kp * (target - np.asarray(q)) - self.kd * np.asarray(qd)
        return np.clip(tau, -self.effort, self.effort)

    def control_step(self, mujoco, m, d, action, decimation=CONTROL_DECIMATION):
        """One 50 Hz control step of the reference: `decimation` MuJoCo
        substeps, the torque re-evaluated from the CURRENT `qpos`/`qvel` at
        each one (`legged_robot_base._pre_physics_step` computes torques
        inside the decimation loop). Returns MuJoCo's max `ncon` seen."""
        target = self.target(action)
        worst_ncon = 0
        for _ in range(decimation):
            d.ctrl[:] = self.torque(target, d.qpos[7:], d.qvel[6:])
            mujoco.mj_step(m, d)
            worst_ncon = max(worst_ncon, int(d.ncon))
        return worst_ncon

    def stand_state(self, mujoco, m, d):
        """The reference's `init_state`: pelvis at `pos`, identity
        orientation, default joint angles, at rest."""
        mujoco.mj_resetData(m, d)
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[2] = self.scal["root_z"]
        d.qpos[3] = 1.0
        d.qpos[7:] = self.default
        mujoco.mj_forward(m, d)


def main(argv):
    check = "--check" in argv
    xml = bake_xml()
    pd = bake_pd()
    meshes = mesh_files(xml)
    assert len(meshes) == 36, len(meshes)
    if check:
        bad = []
        if not os.path.exists(ASSET_XML) or open(ASSET_XML).read() != xml:
            bad.append(ASSET_XML)
        if not os.path.exists(PD_MOJO) or open(PD_MOJO).read() != pd:
            bad.append(PD_MOJO)
        for m in meshes:
            dst = os.path.join(MESH_DIR, m)
            src = os.path.join(REF_G1, "meshes", m)
            if not os.path.exists(dst) or open(dst, "rb").read() != open(src, "rb").read():
                bad.append(dst)
        if bad:
            print("STALE — re-run tests/robots/g1_bake.py:")
            for b in bad:
                print("  ", os.path.relpath(b, REPO))
            return 1
        print("g1_bake --check: up to date (%d meshes)" % len(meshes))
        return 0
    os.makedirs(MESH_DIR, exist_ok=True)
    for m in meshes:
        shutil.copyfile(os.path.join(REF_G1, "meshes", m), os.path.join(MESH_DIR, m))
    open(ASSET_XML, "w").write(xml)
    open(PD_MOJO, "w").write(pd)
    print("wrote", os.path.relpath(ASSET_XML, REPO))
    print("wrote", os.path.relpath(PD_MOJO, REPO))
    print("copied %d meshes to %s" % (len(meshes), os.path.relpath(MESH_DIR, REPO)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
