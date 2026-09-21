"""Every LIBERO family's batched model def and contact budget — GENERATED.

    pixi run gen-libero-envs           # write noeira/tasks/libero_envs/*.mojo
    pixi run gen-libero-envs --check   # CI: fail if one is stale

Writes, from `noeira/tasks/libero/contact_budget.kv` and the `libero*.family`
files:

    libero_envs/budgets.mojo          <FAMILY>_MAX_CONTACTS for all 23 families
    libero_envs/<family>_xml.mojo     the model def, obs width, config and env —
                                      for the 20 scene families

`libero_goal`, `libero_object` and `libero_spatial` keep their hand-written
`tasks/libero_<suite>_xml.mojo` (each carries a header about ITS scene), and
import their budget from `budgets.mojo` like everyone else — one rule.

## ⚠⚠ THE BUDGET RULE: THE MEASURED PEAK + `GRASP_MARGIN`, UP TO A 16

The batched env truncates at `max_contacts` silently
(`libero_contact_budget.mojo`'s header). The measured number is a null-action
peak — the landing of every prop at reset, which is the scene's geometric
ceiling (on `libero_goal` the sampled resets and all fifty frozen inits hit the
same 112, and MuJoCo agrees substep for substep). A grasp adds contacts on top
of a resting scene, and the demonstrations say how many: the batched replay of
`libero_goal`'s demos on the 5090 (256 lanes, 60 steps, PERFORMANCE.md §13.53's
per-step CSV) peaks at 77-89 contacts mid-grasp, steps 25-43, on four tasks,
against a resting 37 — the fingers and a carried bowl or bottle on the
cabinet. `GRASP_MARGIN` covers a grasp while a prop is still landing only on a
family whose peak already sits well above its resting count; on `libero_goal`
the budget, 144, is 55 above that grasp peak.

⚠ AN EARLIER VERSION OF THIS PARAGRAPH SAID THE DEMOS PEAK "7 ABOVE" RESTING.
That came from `libero_demo_success`, which replays only each demo's SUCCESS
WINDOW (its last states, the prop already set down), not the grasp.

⚠ NOT A PROOF. Frozen inits exist on disk for `libero_goal` only; the other 22
budgets are sampled resets, and `test_libero_object` records a LIBERO frozen
state MuJoCo counts 204 contacts in. The drivers therefore REPORT a saturated
lane (`ncon == max_contacts`) instead of trusting this number — re-run
`libero-contact-budget` when a family gains an init table; it reads them.

## WHAT A GENERATED MODULE HOLDS

    comptime <Camel>Model      ModelDefFromXML over the generated dims
    comptime <UPPER>_OBS_DIM   NQ + NV + P.N_FREE + TASK_GOAL_WORDS
    comptime <Camel>OscConfig  LiberoOscConfig[<Camel>Placement]
    comptime <Camel>OscEnv     Phyics3dBatchedEnv[...]

The dims module beside it is `tools/gen_model_dims.py`'s (MuJoCo's counts).
"""

from std.os import listdir
from std.sys import argv

from noeira.tasks.spec import load_family, SLOT_FREE

comptime FAMILY_DIR = "noeira/tasks/families"
comptime OUT_DIR = "noeira/tasks/libero_envs"
comptime BUDGET_KV = "noeira/tasks/libero/contact_budget.kv"
comptime GRASP_MARGIN = 32
comptime ROUND = 16


def _families() raises -> List[String]:
    var out = List[String]()
    for e in listdir(FAMILY_DIR):
        var n = String(e)
        if n.startswith("libero") and n.endswith(".family"):
            out.append(String(n[byte = 0 : n.byte_length() - 7]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def _hand_written(family: String) -> Bool:
    return (
        family == "libero_goal" or family == "libero_object"
        or family == "libero_spatial"
    )


def camel(family: String) -> String:
    """`libero_kitchen_scene10` -> `LiberoKitchenScene10` — the prefix
    `gen_placement_tables.struct_name` puts before `Placement`."""
    var out = String("")
    var up = True
    for cp in family.codepoint_slices():
        var c = String(cp)
        if c == "_":
            up = True
            continue
        out += c.upper() if up else c
        up = False
    return out^


def budget(measured: Int) -> Int:
    """The rule in the header: peak + `GRASP_MARGIN`, rounded up to `ROUND`."""
    var want = measured + GRASP_MARGIN
    return ((want + ROUND - 1) // ROUND) * ROUND


@fieldwise_init
struct Measured(Copyable, Movable):
    var family: String
    var peak: Int
    var at: String


def _read_budgets() raises -> List[Measured]:
    var text: String
    with open(String(BUDGET_KV), "r") as fh:
        text = fh.read()
    var out = List[Measured]()
    var lines = text.split("\n")
    for i in range(len(lines)):
        var l = String(String(lines[i]).strip())
        if l.byte_length() == 0 or l.startswith("#"):
            continue
        var eq = l.find("=")
        var comma = l.find(",")
        if eq < 0 or comma < eq:
            raise Error(BUDGET_KV + ": malformed line '" + l + "'")
        out.append(Measured(
            String(l[byte = 0 : eq]),
            Int(String(l[byte = eq + 1 : comma])),
            String(l[byte = comma + 1 : l.byte_length()]),
        ))
    return out^


def render_budgets(rows: List[Measured]) -> String:
    var o = String(
        '"""Every LIBERO family\'s batched `max_contacts` — GENERATED, DO NOT'
        " EDIT.\n\n"
        "Regenerate with:  pixi run gen-libero-envs\n"
        "CI checks it with: pixi run gen-libero-envs --check\n\n"
        "From `" + BUDGET_KV + "` (`pixi run libero-contact-budget`):\n"
        "the measured null-action peak + " + String(GRASP_MARGIN)
        + ", rounded up to a multiple of " + String(ROUND) + ".\n"
        "`tools/tasks/gen_libero_envs.mojo` says why, and what it does not"
        " prove.\n"
        '"""\n'
    )
    for i in range(len(rows)):
        o += (
            "\ncomptime " + rows[i].family.upper() + "_MAX_CONTACTS: Int = "
            + String(budget(rows[i].peak)) + "\n"
            + '"""Measured peak ' + String(rows[i].peak) + " (" + rows[i].at
            + ')."""\n'
        )
    return o^


def render_env(family: String, n_free: Int, b: Int, peak: Int) raises -> String:
    var up = family.upper()
    var cm = camel(family)
    var o = String(
        '"""`' + family + "` on the batched GPU env — GENERATED, DO NOT EDIT.\n\n"
        "Regenerate with:  pixi run gen-libero-envs\n"
        "CI checks it with: pixi run gen-libero-envs --check\n\n"
        "    families/<family>.family          the slot table\n"
        "    scenes/<family>.xml               composed from it\n"
        "    libero_envs/<family>_dims.mojo    MuJoCo's counts (gen-dims)\n"
        "    placement/<family>.mojo           the device reset's table\n"
        "    THIS FILE                         model def, config, env\n\n"
        "The config is `libero_osc_config.LiberoOscConfig` — OSC_POSE at 20 Hz,"
        " the\n"
        "task layer's hooks over this family's table; its header is the"
        " documentation.\n"
        "`max_contacts` is `budgets." + up + "_MAX_CONTACTS` (" + String(b)
        + ", measured\npeak " + String(peak) + "). " + String(n_free)
        + " free slots. ELLIPTIC cone: robosuite's `base.xml`.\n"
        '"""\n\n'
        "from noeira.physics3d.parser import ModelDefFromXML\n"
        "from noeira.physics3d.types import ConeType\n"
        "from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv\n"
        "from noeira.tasks.task_hooks import TASK_GOAL_WORDS\n"
        "from noeira.tasks.libero_osc_config import LiberoOscConfig\n"
        "from noeira.tasks.placement." + family + " import " + cm
        + "Placement\n"
        "from noeira.tasks.libero_envs.budgets import " + up + "_MAX_CONTACTS\n"
        "from noeira.tasks.libero_envs." + family + "_dims import " + up
        + "_DIMS\n\n"
        "comptime _pm = " + up + "_DIMS\n\n"
        "comptime " + up + "_OBS_DIM: Int = (\n"
        "    _pm.NQ + _pm.NV + " + cm + "Placement.N_FREE + TASK_GOAL_WORDS\n"
        ")\n"
        '"""`task_hooks.write_task_obs`\'s layout: qpos, qvel, one active word'
        ' per\nfree slot, the goal words."""\n\n'
        "comptime " + cm + "Model = ModelDefFromXML[\n"
        '    xml_path="noeira/tasks/scenes/' + family + '.xml",\n'
        "    nbody=_pm.NBODY,\n"
        "    njoint=_pm.NJOINT,\n"
        "    nq=_pm.NQ,\n"
        "    nv=_pm.NV,\n"
        "    ngeom=_pm.NGEOM,\n"
        "    nact=_pm.NACT,\n"
        "    ntex=_pm.NTEX,\n"
        "    nmat=_pm.NMAT,\n"
        "    nlight=_pm.NLIGHT,\n"
        "    ncam=_pm.NCAM,\n"
        "    nsite=_pm.NSITE,\n"
        "    nsensor=_pm.NSENSOR, nsensordata=_pm.NSENSORDATA,\n"
        "    neq=_pm.NEQ,\n"
        "    nexclude=_pm.NEXCLUDE,\n"
        "    npair=_pm.NPAIR,\n"
        "    timestep=_pm.TIMESTEP,\n"
        "    cone_type=ConeType.ELLIPTIC,\n"
        "    max_condim=_pm.MAX_CONDIM,\n"
        "    max_contacts=" + up + "_MAX_CONTACTS,\n"
        "    obs_dim_override=" + up + "_OBS_DIM,\n"
        "    action_dim_override=7,\n"
        "]\n\n"
        "comptime " + cm + "OscConfig = LiberoOscConfig[" + cm + "Placement]\n\n"
        "comptime " + cm + "OscEnv = Phyics3dBatchedEnv[\n"
        "    " + cm + "Model, " + cm + "OscConfig, _, CRBA_TREEWALK=True\n"
        "]\n"
    )
    return o^


def _write_or_check(
    path: String, text: String, check: Bool, mut stale: Int
) raises:
    var old = String("")
    var have = True
    try:
        with open(path, "r") as fh:
            old = fh.read()
    except:
        have = False
    if check:
        if not have or old != text:
            print("  STALE:", path)
            stale += 1
    else:
        if not have or old != text:
            with open(path, "w") as fh:
                fh.write(text)
            print("  wrote", path)


def main() raises:
    var args = argv()
    var check = False
    for i in range(1, len(args)):
        if String(args[i]) == "--check":
            check = True
        else:
            raise Error("gen_libero_envs: unknown argument '" + String(args[i]) + "'")
    var fams = _families()
    var rows = _read_budgets()
    # ⚠ THE KV MUST COVER EXACTLY THE FAMILIES ON DISK, in either direction —
    # a family added after the measurement would otherwise get no budget, and
    # a renamed one would keep a stale line nobody reads.
    if len(rows) != len(fams):
        raise Error(
            BUDGET_KV + " has " + String(len(rows)) + " families, the tree "
            + String(len(fams)) + " — run `pixi run libero-contact-budget`"
        )
    var stale = 0
    _write_or_check(
        String(OUT_DIR) + "/budgets.mojo", render_budgets(rows), check, stale
    )
    var n_env = 0
    for i in range(len(fams)):
        if rows[i].family != fams[i]:
            raise Error(
                BUDGET_KV + ": line " + String(i) + " is '" + rows[i].family
                + "', expected '" + fams[i] + "'"
            )
        if _hand_written(fams[i]):
            continue
        var f = load_family(String(FAMILY_DIR) + "/" + fams[i] + ".family")
        var n_free = 0
        for s in range(len(f.slots)):
            if f.slots[s].kind == SLOT_FREE:
                n_free += 1
        _write_or_check(
            String(OUT_DIR) + "/" + fams[i] + "_xml.mojo",
            render_env(fams[i], n_free, budget(rows[i].peak), rows[i].peak),
            check, stale,
        )
        n_env += 1
    if check and stale > 0:
        raise Error(
            String(stale) + " file(s) stale — run `pixi run gen-libero-envs`"
        )
    print(n_env, "env modules +", len(rows), "budgets",
          "checked" if check else "up to date")
