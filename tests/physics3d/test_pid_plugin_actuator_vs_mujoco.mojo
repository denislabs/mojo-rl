"""`<plugin plugin="mujoco.pid">` — a force law whose gains are not in the model.

WHY THIS EXISTS
===============
MuJoCo lets an actuator's force law come from a PLUGIN. `shadow_dexee` drives
all twelve of its joints that way, and this parser modelled none of them: the
`<actuator>` scan looked for four spellings (`<motor>`, `<position>`,
`<velocity>`, `<general>`) and `<plugin>` is a fifth. The model still loaded
and still stepped — with `nact` **0** against MuJoCo's **12**.

⚠⚠ A SHORT `nact` IS WORSE THAN A WRONG FORCE. Every index of a control vector
sized for MuJoCo's `nu` lands on a different actuator from the one the caller
meant, or on nothing at all. dexee sat at #2 on the Menagerie board at
5.373e-03; measured, the whole of it was this: driving both engines with
ctrl == 0 everywhere (where a PID whose setpoint and `qpos0` are both zero
produces no force) collapses the scene to 1.860e-04.

⚠ THE GAINS ARE NOT IN `mjModel` IN ANY NUMERIC FIELD. They are NUL-separated
TEXT in `plugin_attr`, keyed by `plugin_attradr[instance]`, while
`gaintype`/`biastype` stay at their defaults — so an engine reading only the
gain/bias record computes `force = 1 * ctrl` and a `mjModel` field diff shows
nothing missing. `mjs_setToAdhesion`-style numeric records are the exception
here, not the rule.

FOUR ARMS:

  1. the RECORD — the four `<instance>` configs land on the right twelve
     actuators. dexee declares FOUR instances for TWELVE actuators (the three
     fingers share gains per joint index), so this also gates the
     instance→actuator mapping and not just the parse.

  2. the FORCE at step 0, against MuJoCo's `qfrc_actuator`. ⚠ IT IS ALSO THE
     `previous_ctrl_exists` ARM: at `t == 0` the slew limiter must NOT clamp,
     and if it did every control here (0.07 .. 0.40) would be squeezed to
     `slewmax*dt` = 6.3e-03 and the forces would collapse.

  3. ⚠⚠ the SLEW LIMITER, WITH ITS OWN CONTROL. A control that JUMPS from 0 to
     1.2 is clamped to +-`slewmax*dt` per step, so by step 2 the forces are
     ~3.5e-02; with the limiter absent the same step SATURATES every actuator
     against its `forcerange` (0.53, 1.2, 0.7, 0.3). Both vectors are stored,
     and the test asserts we match the first and are nowhere near the second.
     Without that second vector this arm would pass on a build with no slew
     limiting at all, because arm 2 cannot see it.

  4. the TRAJECTORY — 60 steps at a constant control, `qpos` against MuJoCo.
     The integral, its `imax` clamp and the slew limiter all compound here in
     a way a single-step force comparison cannot show.

  5. an `instance=` NAMING NOTHING must RAISE. `-1` would be a PID with zero
     gains: a live slot in `nact` that consumes a control and applies no
     force, which is exactly the failure `joint=` was fixed for.

Run: pixi run mojo run -I . tests/physics3d/test_pid_plugin_actuator_vs_mujoco.mojo
"""

from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.flat_model import ACT_KIND_PID
from mojo_rl.physics3d.fields import Data, Model, DynDims, SpecFields
from mojo_rl.physics3d.fields.dynamics_scratch import DynamicsScratch
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.types import ConeType, IntegratorType
from mojo_rl.physics3d.studio.stepping import (
    StudioImpFastEll, studio_cone_of, studio_integrator_of, STUDIO_DT,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    ACT_IDX_KIND, ACT_IDX_KP, ACT_IDX_PID_KI, ACT_IDX_PID_KD,
    ACT_IDX_PID_IMAX, ACT_IDX_PID_SLEW, ACT_IDX_ACT_ADR,
    META_IDX_SIM_TIME,
)


comptime DT = STUDIO_DT
comptime DEXEE = String(
    "references/mujoco_menagerie-main/shadow_dexee/scene.xml"
)


def _mj_ctrl0() -> List[Float64]:
    """The board's own step-0 controls (`default_rng(2024)`, x0.4)."""
    return [
        +1.40665070385025459e-01, -2.28541439009393887e-01,
        -1.52438375294646655e-01, +2.39572877419866548e-01,
        +3.96641679092373467e-01, -2.86214547775958561e-01,
        -3.37019572990400862e-01, -2.55340949042516308e-01,
        -1.12282486648519253e-01, -2.64304600234361364e-01,
        +7.10074524317841210e-02, +9.34460110590224630e-02,
    ]


def _mj_qfrc0() -> List[Float64]:
    """MuJoCo `qfrc_actuator` at `qpos0` under `_mj_ctrl0`."""
    return [
        +3.94987517641151420e-01, -3.49999999999999978e-01,
        +0.00000000000000000e+00, +1.45181163716439132e-01,
        +5.30000000000000027e-01, -3.49999999999999978e-01,
        +0.00000000000000000e+00, -1.54736615119764886e-01,
        -3.15289222509042011e-01, -3.49999999999999978e-01,
        +7.85342423895532410e-02, +5.66282827017676113e-02,
    ]


def _mj_qpos_1() -> List[Float64]:
    """MuJoCo `qpos` at step 1 of the 0 -> 1.2 jump."""
    return [
        +2.86519500726101034e-07, -1.12895171580096429e-03,
        +1.74000255187757471e-03, +1.29015237846890327e-03,
        +2.01847975424870613e-03, -9.81229469138243573e-04,
        +1.12915963293099279e-03, +8.10136060470816177e-06,
        -2.03319442751861892e-03, -9.58038816447574076e-04,
        +1.06667032546428359e-03, +7.61112810402637761e-06,
    ]

def _mj_qvel_1() -> List[Float64]:
    """MuJoCo `qvel` at step 1."""
    return [
        +1.43259750363050522e-04, -5.64475857900482159e-01,
        +8.70001275938787288e-01, +6.45076189234451647e-01,
        +1.00923987712435315e+00, -4.90614734569121802e-01,
        +5.64579816465496376e-01, +4.05068030235408109e-03,
        -1.01659721375930934e+00, -4.79019408223787024e-01,
        +5.33335162732141765e-01, +3.80556405201318889e-03,
    ]

def _mj_act_1() -> List[Float64]:
    """MuJoCo `d.act` at step 1: (integral, previous ctrl) x 12."""
    return [
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
    ]

def _mj_qfrc_1() -> List[Float64]:
    """MuJoCo `qfrc_actuator` at step 1, from the state above."""
    return [
        +1.76380671007310731e-02, +2.98643192378068613e-02,
        -3.67525850176447138e-03, -2.13483477522776821e-03,
        -1.83019180236609590e-02, +2.80169048210428748e-02,
        +5.45483613233591180e-05, +3.77029221305471424e-03,
        +5.38502958052515593e-02, +2.77268825184933607e-02,
        +4.36108072715085085e-04, +3.77255022395285404e-03,
    ]

def _mj_qpos_5() -> List[Float64]:
    """MuJoCo `qpos` at step 5 of the 0 -> 1.2 jump."""
    return [
        -1.67709414105338856e-05, -7.80429557446968465e-03,
        +1.23434913256722234e-02, +8.11667162350481960e-03,
        +1.98295631177917084e-02, -7.14374607619350493e-03,
        +6.39540337083790641e-03, +3.77159531421336256e-04,
        -1.94027785215547983e-02, -7.53391473401755600e-03,
        +6.88200231775610433e-03, +3.60614451668611028e-04,
    ]

def _mj_qvel_5() -> List[Float64]:
    """MuJoCo `qvel` at step 5."""
    return [
        -8.99175393151756784e-03, -7.65802782442190688e-01,
        +9.56189214118717490e-01, +6.77689422520358131e-01,
        +2.52635129606929754e+00, -7.57102634490254056e-01,
        +3.96955698008148838e-01, +1.83562901363591241e-01,
        -2.43017305665113792e+00, -8.43525470493501461e-01,
        +5.28600682536857103e-01, +1.90850367477516492e-01,
    ]

def _mj_act_5() -> List[Float64]:
    """MuJoCo `d.act` at step 5: (integral, previous ctrl) x 12."""
    return [
        +1.25651284653902209e-04, +2.51327200000000008e-02,
        +1.55538846135410314e-04, +2.51327200000000008e-02,
        +7.59074050756362184e-05, +2.51327200000000008e-02,
        +9.31053131426919903e-05, +2.51327200000000008e-02,
        +6.12685807324600556e-05, +2.51327200000000008e-02,
        +1.51868666222635372e-04, +2.51327200000000008e-02,
        +9.77151937509050043e-05, +2.51327200000000008e-02,
        +1.26010196619066373e-04, +2.51327200000000008e-02,
        +1.89350989020554061e-04, +2.51327200000000008e-02,
        +1.52514040118962132e-04, +2.51327200000000008e-02,
        +9.71498661744182293e-05, +2.51327200000000008e-02,
        +1.26174478466054295e-04, +2.51327200000000008e-02,
    ]

def _mj_qfrc_5() -> List[Float64]:
    """MuJoCo `qfrc_actuator` at step 5, from the state above."""
    return [
        +8.90352977600419176e-02, +1.14068482296871082e-01,
        +1.17599140678462596e-02, +8.97713295542129121e-03,
        -4.30110305939081991e-02, +1.12228131755413907e-01,
        +2.39962578730245069e-02, +1.77190041029071411e-02,
        +2.16361444944142223e-01, +1.14936287253674926e-01,
        +2.21399336097164376e-02, +1.76712235378668550e-02,
    ]

def _mj_qpos_30() -> List[Float64]:
    """MuJoCo `qpos` at step 30 of the 0 -> 1.2 jump."""
    return [
        -3.68232924998914760e-02, -7.05687388877463285e-03,
        +2.73857071114824192e-02, +1.00060479367974922e-01,
        +1.50991733677852041e-01, +1.09563092340128482e-01,
        +9.22692329986763565e-02, +9.42798595399736428e-02,
        -2.92670854177040536e-02, -2.45078303509914906e-02,
        +1.70748387886304059e-02, +1.00706344287820576e-01,
    ]

def _mj_qvel_30() -> List[Float64]:
    """MuJoCo `qvel` at step 30."""
    return [
        -2.02944273291449617e+00, +7.75456874071303481e-01,
        +1.23893811435147372e+00, +2.98119328218409496e+00,
        +2.44336170532363672e+00, +5.81155263894971341e+00,
        +4.77001699559532977e+00, +3.15810839688510159e+00,
        +1.93776406644101828e+00, -2.65786075914951514e-01,
        +5.97429991265243876e-01, +3.10310560270375202e+00,
    ]

def _mj_act_30() -> List[Float64]:
    """MuJoCo `d.act` at step 30: (integral, previous ctrl) x 12."""
    return [
        +5.88945332856706874e-03, +1.82212220000000008e-01,
        +6.19126430120360907e-03, +1.82212220000000008e-01,
        +4.70333780471118585e-03, +1.82212220000000008e-01,
        +3.39890548171291069e-03, +1.82212220000000008e-01,
        +1.22826710769984482e-03, +1.82212220000000008e-01,
        +4.65337102139250572e-03, +1.82212220000000008e-01,
        +4.21085508572152482e-03, +1.82212220000000008e-01,
        +3.78092974665511592e-03, +1.82212220000000008e-01,
        +7.69533327477248900e-03, +1.82212220000000008e-01,
        +6.35190524181878345e-03, +1.82212220000000008e-01,
        +4.96625757092024420e-03, +1.82212220000000008e-01,
        +3.64977176007397599e-03, +1.82212220000000008e-01,
    ]

def _mj_qfrc_30() -> List[Float64]:
    """MuJoCo `qfrc_actuator` at step 30, from the state above."""
    return [
        +5.30000000000000027e-01, +4.93118653787453953e-01,
        +1.79907952605319305e-01, +3.99387320906731680e-02,
        +3.69225123036817718e-02, +9.55334232808212891e-02,
        +7.13585360046752465e-02, +4.31725395836605047e-02,
        +5.30000000000000027e-01, +5.58157532503340104e-01,
        +1.98515613499883115e-01, +3.93246382201726460e-02,
    ]

def _mj_c_qpos_30() -> List[Float64]:
    """MuJoCo `qpos` at step 30 of a CONSTANT ctrl 1.2."""
    return [
        -2.83593896102955385e-02, +7.83675742679914245e-01,
        +1.11798747868735737e+00, +1.09291476695202383e+00,
        +9.88591977987180348e-01, +8.91538279588762528e-01,
        +1.14698977276594705e+00, +1.07245838274080763e+00,
        -2.11110738440447199e-01, +2.21515547604157231e-01,
        +8.65262067205606167e-01, +1.07072391487000029e+00,
    ]

def _mj_c_qvel_30() -> List[Float64]:
    """MuJoCo `qvel` there."""
    return [
        +6.39059715035021014e+00, +8.53800115942195781e+00,
        +6.07673663814755916e+00, +5.20575009753455209e+00,
        +6.48733247803850066e+00, +4.09453046686785438e+00,
        +2.89319941158651295e+00, +6.38658267590320872e+00,
        -6.09954247880370737e+00, +1.68962287358413099e+00,
        -1.25605298900472429e+00, +6.45055333984460599e+00,
    ]

def _mj_c_act_30() -> List[Float64]:
    """MuJoCo `d.act` there — actuators 0/3 (mod 4) are PINNED at `imax/ki`
    (0.025 and 0.03333...), which is what makes the clamp arm bite."""
    return [
        +2.50000000000000014e-02, +8.72700000000000031e-01,
        +3.47324757765534592e-02, +7.85399999999999987e-01,
        +4.25265363239129002e-02, +1.19999999999999996e+00,
        +3.33333333333333329e-02, +1.19999999999999996e+00,
        +2.45315084870386842e-02, +8.72700000000000031e-01,
        +2.89767925621984661e-02, +7.85399999999999987e-01,
        +4.00976800319895119e-02, +1.19999999999999996e+00,
        +3.33333333333333329e-02, +1.19999999999999996e+00,
        +2.50000000000000014e-02, +8.72700000000000031e-01,
        +4.51081751175538906e-02, +7.85399999999999987e-01,
        +4.47349998528793688e-02, +1.19999999999999996e+00,
        +3.33333333333333329e-02, +1.19999999999999996e+00,
    ]

def _mj_c_qfrc_30() -> List[Float64]:
    """MuJoCo `qfrc_actuator` there."""
    return [
        +5.30000000000000027e-01, -6.22416070146439085e-02,
        +1.57518091162045820e-01, +1.22605139048509254e-01,
        -4.21918614581002549e-01, -2.60942760300200638e-01,
        +1.49990357300965937e-01, +1.25432308948289717e-01,
        +5.30000000000000027e-01, +1.19999999999999996e+00,
        +5.16985683119284944e-01, +1.25961224359242951e-01,
    ]

struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def eq(mut self, got: Int, want: Int, msg: String):
        self.checks += 1
        if got == want:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "got", got, "want", want)

    def close(mut self, got: Float64, want: Float64, tol: Float64, msg: String):
        self.checks += 1
        if abs(got - want) <= tol:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "got", got, "want", want)

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _worst(got: List[Float64], want: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(len(want)):
        var e = abs(got[i] - want[i])
        if e > w:
            w = e
    return w


def _force_at(
    sf: SpecFields[DT, DynDims],
    mut dd: Data[DT, DynDims, 1],
    mut aa: List[Scalar[DT]],
    qpos: List[Float64],
    qvel: List[Float64],
    acts: List[Float64],
    ctrl: List[Float64],
    want: List[Float64],
    timestep: Float64,
) raises -> Float64:
    """`|d qfrc_actuator|` with the state LOADED from MuJoCo, not integrated.

    ⚠⚠ THE FIRST DRAFT OF THIS GATE STEPPED OUR OWN INTEGRATOR AND FAILED, AND
    THE PID WAS NOT WHY. dexee carries a SEPARATE, pre-existing defect: driven
    with ctrl == 0 everywhere — where a PID whose setpoint and `qpos0` are both
    zero produces no force at all — our trajectory already parts from MuJoCo's
    by 1.860e-04 after one step and 5.03e-03 after sixty. Its `nefc` is 15
    `mjCNSTR_FRICTION_DOF` rows plus one elliptic contact, so it is a
    `frictionloss` question and not this one. A gate for a FORCE LAW that
    integrates for sixty steps first is measuring whatever else is wrong with
    the model.
    """
    var nq = sf.dims.get_nq()
    var nv = sf.dims.get_nv()
    for i in range(nq):
        dd.qpos.data[i] = Scalar[DT](qpos[i])
    for i in range(nv):
        dd.qvel.data[i] = Scalar[DT](qvel[i])
    while len(aa) < len(acts):
        aa.append(Scalar[DT](0))
    for i in range(len(acts)):
        aa[i] = Scalar[DT](acts[i])
    # ⚠ `time > 0`, because every state here is past step 0 and the slew
    # limiter must be live. The step-0 arm is the one that gates the other
    # side of that branch.
    dd.meta.data[META_IDX_SIM_TIME] = Scalar[DT](1.0)
    apply_actions_fields[DT](sf, dd, ctrl, aa, timestep)
    var got = List[Float64]()
    for i in range(nv):
        got.append(Float64(dd.qfrc.data[i]))
    return _worst(got, want)


def main() raises:
    var t = Tally()
    print("=== <plugin plugin='mujoco.pid'> vs MuJoCo 3.10.0 ===")

    var src = read_model_source(DEXEE)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=65536)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var n_act = dims.get_nact()

    # ── 1. the record ────────────────────────────────────────────────────
    print("--- the four <instance> configs, over twelve actuators ---")
    t.eq(n_act, 12, "nact (was 0 — every <plugin> was skipped)")
    t.eq(len(fmd.plugin_inst_names), 4, "declared plugin instances")

    # MuJoCo's own values, read from `plugin_attr`: J0 kp 2.8 ki 4.0 kd 0.03
    # imax 0.1, J1 2.5/3.0/0.02/0.2, J2 1.1/3.0/0.01/0.2, J3 0.6/3.0/0.008/0.1;
    # slewmax 3.14159 on all four.
    var want_kp: List[Float64] = [2.8, 2.5, 1.1, 0.6]
    var want_ki: List[Float64] = [4.0, 3.0, 3.0, 3.0]
    var want_kd: List[Float64] = [0.03, 0.02, 0.01, 0.008]
    # ⚠ ALREADY DIVIDED BY `ki` — `PidConfig::FromModel` scales the XML's
    # FORCE bound into a bound on the integral. 0.1/4.0, 0.2/3.0, ...
    var want_imax: List[Float64] = [0.025, 0.2 / 3.0, 0.2 / 3.0, 0.1 / 3.0]
    var kinds_ok = True
    var adrs_ok = True
    for i in range(n_act):
        var o = i * MODEL_ACTUATOR_SIZE
        if Int(sf.actuators.data[o + ACT_IDX_KIND]) != ACT_KIND_PID:
            kinds_ok = False
        # ⚠ THE MAPPING, NOT JUST THE PARSE. Actuator i takes instance i % 4:
        # `F0/J0 F0/J1 F0/J2 F0/J3 F1/J0 ...`, three fingers sharing four
        # instances. A per-actuator config table would have passed the value
        # arms below and still got this wrong.
        var j = i % 4
        t.close(Float64(sf.actuators.data[o + ACT_IDX_KP]), want_kp[j],
                0.0, "actuator " + String(i) + " kp")
        t.close(Float64(sf.actuators.data[o + ACT_IDX_PID_KI]), want_ki[j],
                0.0, "actuator " + String(i) + " ki")
        t.close(Float64(sf.actuators.data[o + ACT_IDX_PID_KD]), want_kd[j],
                0.0, "actuator " + String(i) + " kd")
        t.close(Float64(sf.actuators.data[o + ACT_IDX_PID_IMAX]),
                want_imax[j], 1e-15, "actuator " + String(i) + " imax/ki")
        t.close(Float64(sf.actuators.data[o + ACT_IDX_PID_SLEW]), 3.14159,
                0.0, "actuator " + String(i) + " slewmax")
        # Two activation slots each (integral + previous control), packed.
        if Int(sf.actuators.data[o + ACT_IDX_ACT_ADR]) != 2 * i:
            adrs_ok = False
    t.truth(kinds_ok, "all twelve carry ACT_KIND_PID")
    t.truth(adrs_ok, "each owns TWO activation slots at 2*i (MuJoCo na = 24)")
    t.eq(fmd.na, 24, "fmd.na (MuJoCo reports na 24 for this model)")
    # ⚠ NON-VACUITY: four identical instances would satisfy every value arm.
    t.truth(
        want_kp[0] != want_kp[1] and want_kp[1] != want_kp[2]
        and want_kp[2] != want_kp[3],
        "the four instances carry DIFFERENT gains",
    )

    # ── 2. the force at step 0 ───────────────────────────────────────────
    print("--- qfrc_actuator at qpos0, and `previous_ctrl_exists` ---")
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    var act = List[Scalar[DT]](length=n_act, fill=Scalar[DT](0))
    var ctrl0 = _mj_ctrl0()
    apply_actions_fields[DT](sf, d, ctrl0, act, fmd.timestep)
    # ⚠ THE LIST WAS SIZED `nact` AND MUST HAVE GROWN. Every caller in this
    # tree allocates one slot per actuator; a PID needs two, and the loop
    # would otherwise have skipped all twelve on its bounds guard.
    t.truth(len(act) >= 24, "`act` grew from 12 to the model's na")
    var got0 = List[Float64]()
    for i in range(nv):
        got0.append(Float64(d.qfrc.data[i]))
    var want0 = _mj_qfrc0()
    var e0 = _worst(got0, want0)
    print("    worst |d qfrc_actuator| =", e0)
    t.truth(e0 == 0.0, "step 0 matches MuJoCo EXACTLY (0.0)")
    # `previous_ctrl_exists` is false at t == 0: these controls are 0.07 ..
    # 0.40 and `slewmax*dt` is 6.28e-03, so a limiter live on the first step
    # would have shrunk every one of them.
    var big = False
    for i in range(len(want0)):
        if abs(want0[i]) > 0.05:
            big = True
    t.truth(big, "the step-0 golden is far above slewmax*dt (the arm bites)")

    # ── 3/4. the law AT MuJoCo'S OWN STATE, so the integrator is not in it ──
    #
    # ⚠⚠ THE FIRST DRAFT OF THIS GATE STEPPED OUR OWN INTEGRATOR AND FAILED,
    # AND THE PID WAS NOT WHY. dexee carries a SEPARATE, pre-existing defect:
    # driven with ctrl == 0 everywhere — where a PID whose setpoint and
    # `qpos0` are both zero produces no force at all — our trajectory already
    # parts from MuJoCo's by 1.860e-04 after one step and 5.03e-03 after
    # sixty. Its `nefc` is 15 `mjCNSTR_FRICTION_DOF` rows plus one elliptic
    # contact, so it is a `frictionloss` question and not this one. A gate for
    # a FORCE LAW that integrates for sixty steps first is measuring whatever
    # else is wrong with the model.
    #
    # So each arm below LOADS MuJoCo's own `qpos`/`qvel`/`act` and asks for
    # the force at that state. Nothing between the golden and the answer.
    var d2 = Data[DT, DynDims, 1](dims)
    var act2 = List[Scalar[DT]](length=n_act, fill=Scalar[DT](0))

    print("--- the slew limiter, at MuJoCo's states on a 0 -> 1.2 jump ---")
    var jump: List[Float64] = []
    for _ in range(n_act):
        jump.append(1.2)
    var e_j1 = _force_at(sf, d2, act2, _mj_qpos_1(), _mj_qvel_1(),
                         _mj_act_1(), jump, _mj_qfrc_1(), fmd.timestep)
    print("    step 1  worst |d qfrc_actuator| =", e_j1)
    t.truth(e_j1 == 0.0, "step 1 of the jump matches MuJoCo EXACTLY")
    var e_j5 = _force_at(sf, d2, act2, _mj_qpos_5(), _mj_qvel_5(),
                         _mj_act_5(), jump, _mj_qfrc_5(), fmd.timestep)
    print("    step 5  worst |d qfrc_actuator| =", e_j5)
    t.truth(e_j5 == 0.0, "step 5 of the jump matches MuJoCo EXACTLY")
    var e_j30 = _force_at(sf, d2, act2, _mj_qpos_30(), _mj_qvel_30(),
                          _mj_act_30(), jump, _mj_qfrc_30(), fmd.timestep)
    print("    step 30 worst |d qfrc_actuator| =", e_j30)
    t.truth(e_j30 == 0.0, "step 30 of the jump matches MuJoCo EXACTLY")

    # ⚠⚠ AND THE CONTROL FOR ALL THREE. The stored previous control at step 30
    # is 0.18221 while the commanded one is 1.2 — so a build with NO slew
    # limiter answers the same three states with a completely different force.
    # Without this the arms above would pass on one.
    var prev30 = Float64(_mj_act_30()[1])
    print("    stored previous ctrl at step 30 =", prev30, "vs commanded 1.2")
    t.truth(prev30 < 0.3,
            "the slew state is FAR from the command — the limiter is what"
            " these three arms are reading")

    print("--- the `imax` clamp, at a state where the integral is pinned ---")
    # ⚠ A DIFFERENT SCHEDULE ON PURPOSE. Under the jump the integral is still
    # climbing at step 30; under a CONSTANT 1.2 it has saturated, and
    # actuators 0 and 3 of each finger sit exactly on `imax/ki` (0.025 and
    # 0.033333...). A clamp written with the UNDIVIDED `imax` (0.1 / 0.1)
    # would not bind at all here.
    var c30 = _mj_c_act_30()
    t.close(c30[0], 0.025, 1e-15, "MuJoCo's own integral is AT imax/ki")
    t.close(c30[6], 0.1 / 3.0, 1e-15, "and so is actuator 3's")
    var e_c30 = _force_at(sf, d2, act2, _mj_c_qpos_30(), _mj_c_qvel_30(),
                          c30, jump, _mj_c_qfrc_30(), fmd.timestep)
    print("    constant ctrl step 30 worst |d qfrc_actuator| =", e_c30)
    t.truth(e_c30 == 0.0, "the saturated-integral state matches EXACTLY")

    # ── 5. an unresolved `instance=` must raise ──────────────────────────
    print("--- an `instance=` naming nothing ---")
    var bad = String(
        "<mujoco><extension><plugin plugin='mujoco.pid'>"
        "<instance name='real'><config key='kp' value='1'/></instance>"
        "</plugin></extension>"
        "<worldbody><body name='b'><joint name='j' type='hinge'/>"
        "<geom type='sphere' size='0.1'/></body></worldbody>"
        "<actuator><plugin name='a' plugin='mujoco.pid' instance='typo'"
        " joint='j'/></actuator></mujoco>"
    )
    var raised = False
    try:
        var _f = parse_xml_full(expand_mjcf(bad, String("")), String(""))
    except e:
        raised = True
        print("    raised:", e)
    t.truth(raised,
            "an undeclared `instance=` raises (it would be a zero-gain PID"
            " holding a live slot in `nact`)")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_pid_plugin_actuator_vs_mujoco: " + String(t.fails)
            + " failed"
        )
