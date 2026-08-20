"""`studio.validate` calls ERROR exactly what will not load — vs MuJoCo 3.10.0.

WHY THIS EXISTS
===============
A diagnostics panel has two ways to be useless and they pull in opposite
directions. Miss a real defect and the tool hands the user a model that will
not load, with no explanation. Flag something that loads fine and every real
model lights up red, at which point the panel is scrolled past and then turned
off. Only an EXTERNAL verdict can separate the two, so the definition of
`SEV_ERROR` is not a judgement call:

    SEV_ERROR  ==  MuJoCo refuses this text
                   (plus ONE tagged exception: a feature MuJoCo has and this
                    engine does not, where refusing beats building 5% of the
                    model silently)

THREE ARMS, AND THE SECOND IS THE ONE THAT BITES:

  1. NOTHING MuJoCo REFUSES COMES BACK CLEAN. The easy direction.

  2. ⚠⚠ NOTHING MuJoCo ACCEPTS IS CALLED AN ERROR. Thirteen fixtures here load
     in MuJoCo and look wrong — a geom paired with itself, gear="0", an
     inverted range on an UNLIMITED joint, a plane with `size="0 0 1"`, a body
     and a geom sharing a name, a moving body whose mass comes from a jointless
     CHILD. Every one of them was a check I would have written the strict way.
     This arm is what makes the panel usable on real models.

  3. THE CODE, NOT JUST THE SEVERITY. Reporting *an* error for the right
     fixture proves nothing if it is the wrong error — and several fixtures are
     one edit apart from a different check.

⚠ AND THE AT-SCALE CONTROL. Every model in the tree loads in MuJoCo, so every
model in the tree must validate with ZERO errors. That is the arm that catches
a check which is merely *usually* right: 35 fixtures cannot, and a validator
that flagged one body in Menagerie would be a validator nobody could use.

⚠ THE FIXTURES ARE GENERATED (`scripts/dump_mujoco_validity.py`) so the text we
validate is byte-identical to the text MuJoCo judged.

Run: pixi run mojo run -I . tests/physics3d/test_validate_vs_mujoco.mojo
"""

from mojo_rl.physics3d.parser.flat_model import FlatModelDef
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.studio.validate import (
    Diagnostic, validate_document, validate_model, worst_severity, count_at,
    format_diagnostic, SEV_ERROR, SEV_WARN,
)
from tests.physics3d.validity_goldens import (
    validity_case_count, validity_name, validity_xml, validity_expected_code,
    validity_mujoco_refuses,
)


comptime DT = DType.float64

def scale_models() -> List[String]:
    """Real models, one per family, all of which MuJoCo loads.

    ⚠ A FUNCTION, NOT A `comptime` TABLE: an `Array[String, N]` is not
    `ImplicitlyCopyable` and cannot be materialised at runtime — the same
    footgun `scene._prop_mjcf_type` names.
    """
    var v = List[String]()
    v.append(String("mojo_rl/envs/ant/assets/ant.xml"))
    v.append(String("mojo_rl/envs/humanoid/assets/humanoid.xml"))
    v.append(String("mojo_rl/envs/half_cheetah/assets/half_cheetah.xml"))
    v.append(String("mojo_rl/envs/walker2d/assets/walker2d.xml"))
    v.append(String("mojo_rl/envs/hopper/assets/hopper.xml"))
    v.append(String("references/mujoco_menagerie-main/agility_cassie/scene.xml"))
    v.append(String("references/mujoco_menagerie-main/aloha/scene.xml"))
    v.append(String("references/mujoco_menagerie-main/unitree_go2/scene.xml"))
    v.append(String("references/mujoco_menagerie-main/franka_emika_panda/scene.xml"))
    v.append(String("references/mujoco_menagerie-main/iit_softfoot/scene.xml"))
    v.append(String("references/mujoco_menagerie-main/ms_human_700/scene.xml"))
    v.append(String("references/mujoco_menagerie-main/shadow_hand/scene_right.xml"))
    # ⚠⚠ THESE FOUR ARE HERE BECAUSE THEY WERE FALSE ALARMS. A twelve-model
    # control passed while `bitcraze_crazyflie_2` reported four errors (its
    # rotors drive through SITES, which MuJoCo simulates and this engine does
    # not) and `hello_robot_stretch_3` reported three (a real parser bug — a
    # self-closing `<body/>` re-parented half the scene). Both were found by
    # opening every Menagerie scene in the studio, not by this gate; adding
    # them is what stops either coming back.
    v.append(String("references/mujoco_menagerie-main/bitcraze_crazyflie_2/scene.xml"))
    v.append(String("references/mujoco_menagerie-main/hello_robot_stretch_3/scene.xml"))
    v.append(String("references/mujoco_menagerie-main/apptronik_apollo/scene.xml"))
    v.append(String("references/mujoco_menagerie-main/franka_fr3_v2/scene.xml"))
    return v^


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)

    def eq(mut self, got: Int, want: Int, msg: String):
        self.checks += 1
        if got == want:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "got", got, "want", want)


struct Verdict(Copyable, Movable):
    """What the studio would show for one document."""

    var has_error: Bool
    var codes: String
    """Every ERROR code, space separated — so the gate can ask for one."""
    var named_by_validator: Bool
    """False when the LOADER raised and the validator had said nothing."""
    var warn_codes: String

    def __init__(out self, has_error: Bool, codes: String, named: Bool,
                 warn_codes: String = String("")):
        self.has_error = has_error
        self.codes = codes
        self.named_by_validator = named
        self.warn_codes = warn_codes


def _dir_of(p: String) -> String:
    var i = p.rfind("/")
    return String(p[byte=0:i]) if i > 0 else String(".")


def _collect(ds: List[Diagnostic], severity: Int) -> String:
    var s = String("")
    for d in ds:
        if d.severity == severity:
            s += d.code + " "
    return s^


def _verts_budget(fmd: FlatModelDef) raises -> Int:
    """The studio's retry-on-raise mesh budget, so the gate loads what it does.

    ⚠ A FIXED NUMBER HERE WOULD BE A SILENT COVERAGE CAP. The first draft used
    8192 and `agility_cassie` needs 18396 — the gate would have reported a
    load failure for a model that loads fine in the tool. The budget is a
    WORKSPACE size, not a model property; the builder names the number it
    wants, so ask it.
    """
    var verts = 0
    var tries = 0
    while True:
        var dims = dims_from_flat(fmd, nmesh_verts=verts)
        var m = Model[DT, DynDims](dims)
        try:
            build_model_runtime[DT](fmd, dims, m)
            return verts
        except e:
            if String(e).find("mesh vertex capacity") == -1:
                raise e
            tries += 1
            if tries > 24:
                raise e
            verts = 4096 if verts == 0 else verts * 2


def _verdict_of_text(xml: String, base: String) -> Verdict:
    """THE STUDIO'S LOAD PATH, exactly as the tool runs it.

    ⚠⚠ THE LOADER'S RAISE COUNTS AS AN ERROR TOO, and saying so is honest
    rather than convenient: some invariants (a tendon with one waypoint, an
    out-of-range condim) are checked by `full_parser` before a `FlatModelDef`
    exists, so the validator CANNOT be the one that names them. The studio
    shows either as a red marker. `named_by_validator` records which mechanism
    spoke, so the arm below can require that the validator — not the raise —
    is what named the fixtures it is supposed to name.
    """
    var doc = validate_document(xml)
    if worst_severity(doc) >= SEV_ERROR:
        return Verdict(True, _collect(doc, SEV_ERROR), True)

    var loaded = True
    var codes = String("")
    try:
        var fmd = parse_xml_full(expand_mjcf(xml, base), base)
        var dims = dims_from_flat(fmd, nmesh_verts=_verts_budget(fmd))
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        var mdl = validate_model(fmd, m)
        codes = _collect(mdl, SEV_ERROR)
        var warns = _collect(mdl, SEV_WARN)
        if worst_severity(mdl) >= SEV_ERROR:
            return Verdict(True, codes^, True, warns^)
        return Verdict(False, codes^, True, warns^)
    except e:
        loaded = False
        _ = e
    if not loaded:
        return Verdict(True, String("load-raised "), False)
    return Verdict(False, codes^, True)


def _xml_named(name: String) raises -> String:
    for i in range(validity_case_count()):
        if validity_name(i) == name:
            return validity_xml(i)
    raise Error("no such fixture: " + name)


def _warns(mut t: Tally, fixture: String, code: String) raises:
    """This fixture LOADS (no error) and carries exactly this warning."""
    var v = _verdict_of_text(_xml_named(fixture), String("."))
    t.truth(
        (not v.has_error) and v.warn_codes.find(code) != -1,
        fixture + " warns '" + code + "' and is not an error (got: '"
        + v.warn_codes + "')",
    )


def main() raises:
    var t = Tally()
    print("=== studio.validate vs MuJoCo 3.10.0 ===")

    # ── 1-3. the fixture table ────────────────────────────────────────────
    var n = validity_case_count()
    var n_refuse = 0
    var n_accept = 0
    var n_loader_named = 0
    print("--- each fixture, both directions ---")
    for i in range(n):
        var name = validity_name(i)
        var xml = validity_xml(i)
        var want_code = validity_expected_code(i)
        var refuses = validity_mujoco_refuses(i)
        if refuses:
            n_refuse += 1
        else:
            n_accept += 1

        var v = _verdict_of_text(xml, String("."))
        var want_error = want_code.byte_length() > 0

        t.truth(
            v.has_error == want_error,
            String(name) + ": error=" + String(v.has_error)
            + " (MuJoCo refuses=" + String(refuses) + ", want error="
            + String(want_error) + ")",
        )
        if want_error:
            t.truth(
                v.codes.find(want_code) != -1,
                String(name) + ": reported '" + want_code + "' (got: "
                + v.codes + ")",
            )
            if not v.named_by_validator:
                n_loader_named += 1

    # ⚠ NON-VACUITY ON THE TABLE ITSELF. A table of 35 refusals would make arm
    # 2 vacuous, and a table of 35 acceptances would make arm 1 vacuous. Both
    # halves have to be substantial or the whole comparison is decoration.
    print("--- the table contains both verdicts ---")
    t.truth(n_refuse >= 15, String("fixtures MuJoCo REFUSES: ", n_refuse))
    t.truth(n_accept >= 10, String("fixtures MuJoCo ACCEPTS: ", n_accept))

    # ⚠ AND THE VALIDATOR, NOT THE LOADER'S RAISE, MUST BE WHAT SPEAKS. If
    # this drifted upward the gate would still be green while `validate`
    # itself had gone silent — every fixture would be "caught" by the parser
    # aborting, which is the behaviour this whole slice exists to replace.
    t.eq(n_loader_named, 0,
         "fixtures named ONLY by a loader raise (validate said nothing)")

    # ── 4. the WARNINGS are not dead code ─────────────────────────────────
    # ⚠⚠ THE AT-SCALE ARM BELOW REPORTS ZERO WARNINGS, which is the right
    # answer for twelve working models and also means nothing there exercises
    # `_check_soft`. Without this arm the entire WARN half could be
    # unreachable and every other arm would still be green — the vacuity
    # failure this tree meets most often.
    print("--- the WARN checks fire, on models that LOAD ---")
    _warns(t, String("accept_self_pair"), String("self-pair"))
    _warns(t, String("accept_self_exclude"), String("self-exclude"))
    _warns(t, String("accept_zero_gear"), String("zero-gear"))
    _warns(t, String("accept_negative_damping"), String("negative-dissipation"))
    # ⚠ AND THE CONTROL: the clean baseline must warn about NOTHING, or
    # "it warns" would be true of every input.
    var base_v = _verdict_of_text(_xml_named(String("ok_baseline")), String("."))
    t.truth(base_v.warn_codes.byte_length() == 0,
            String("ok_baseline warns about nothing (control): '",
                   base_v.warn_codes, "'"))

    # ── 5. the at-scale control ───────────────────────────────────────────
    # Every one of these loads in MuJoCo, so a single ERROR here is a false
    # alarm — and false alarms are what make a panel unusable.
    print("--- real models validate CLEAN (they all load in MuJoCo) ---")
    var total_warn = 0
    for path in scale_models():
        var f = open(path, "r")
        var raw = f.read()
        f.close()
        var base = _dir_of(path)
        var doc = validate_document(expand_mjcf(raw, base))
        var fmd = parse_model_runtime(path)
        var dims = dims_from_flat(fmd, nmesh_verts=_verts_budget(fmd))
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        var mdl = validate_model(fmd, m)
        var nerr = count_at(doc, SEV_ERROR) + count_at(mdl, SEV_ERROR)
        total_warn += count_at(doc, SEV_WARN) + count_at(mdl, SEV_WARN)
        if nerr != 0:
            for d in doc:
                if d.severity >= SEV_ERROR:
                    print("      ", format_diagnostic(d))
            for d in mdl:
                if d.severity >= SEV_ERROR:
                    print("      ", format_diagnostic(d))
        t.eq(nerr, 0, String(path))

    print("    warnings across the same models:", total_warn)
    # ⚠ AND THE WARNINGS ARE NOT ZERO. `bitcraze_crazyflie_2` MUST warn: its
    # four rotors apply zero force here, and a model that loads and does
    # nothing is the case a silent panel would hide. If this ever reads 0
    # again, either the model changed or the check went quiet.
    t.truth(total_warn >= 4,
            String("real models DO produce warnings where they should: ",
                   total_warn))

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_validate_vs_mujoco: " + String(t.fails) + " failed"
        )
