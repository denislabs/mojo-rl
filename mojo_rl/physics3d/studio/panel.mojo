"""The studio's ImGui sidebar — one NON-GENERIC function over plain data.

## ⚠ WHY THIS IS NOT `viewer_core.build_sidebar`

`run_view` is parameterised on `MODEL: ModelDefLike` + `CONFIG:
Phyics3dEnvConfig` and builds a `Phyics3dEnv[MODEL, CONFIG]`, so a runtime
model cannot call it. But the type signature is the SYMPTOM, not the reason.

**An env is the RL contract, and a scene is not a task.** `Phyics3dEnv` exists
to supply obs / reward / done / action space; a composed scene has none of
those and does not want them. Routing the studio through it would mean
inventing an observation and a reward for a table with two cubes on it — which
is fabricating the very thing the user is supposed to author later. The env
belongs to the BAKE phase, where a scene becomes a task and the missing pieces
first exist. See `docs/PHYSICS3D_STUDIO_PLAN.md` §5.1.

What IS taken from `viewer_core`: the shim entry points, the widget shapes,
the drive modes, and its **KEEP IT NON-GENERIC** rule — which matters more
here, not less, since the studio's whole claim is that its binary does not
grow with the models it can open. Every line in this file compiles once. Take
state as plain arguments and return REQUESTS; never a `Model`, a `Data` or a
dims provider, all of which are generic.
"""

from mojo_rl.render.imgui import (
    ig_begin_panel, ig_end, ig_begin_child, ig_end_child, ig_separator_text,
    ig_text, ig_text_disabled, ig_text_colored, ig_spacing, ig_same_line,
    ig_button, ig_small_button, ig_selectable, ig_checkbox, ig_combo,
    ig_slider_float, ig_tree_node, ig_tree_pop, ig_push_id_int, ig_pop_id,
    ig_set_next_item_width, ig_set_next_item_open, ig_progress_bar,
    ig_is_item_hovered, ig_set_tooltip, ig_input_text, TextBuffer,
)

comptime SIDEBAR_W: Float32 = 320.0

comptime SEL_NONE: Int = 0
comptime SEL_BODY: Int = 1
comptime SEL_GEOM: Int = 2


@fieldwise_init
struct StudioPanel(Movable):
    """Everything the panel OWNS across frames.

    Separate from the studio's simulation locals so the panel can be reasoned
    about (and later, saved) on its own — and so nothing in this file needs a
    generic type.
    """

    var sel_kind: Int
    var sel_index: Int
    """Index into `fmd.bodies`+1 / `fmd.geoms` per `sel_kind`. A GEOM index is
    also an `rf.geom_*` index — `build_render_fields` walks `fmd.geoms` in
    order — which is what lets a ray-pick result be stored here unchanged."""
    var drive: Int
    var scale: Float32
    var paused: Bool
    var show_outline: Bool
    var filter: TextBuffer

    def __init__(out self, drive: Int, scale: Float64):
        self.sel_kind = SEL_NONE
        self.sel_index = -1
        self.drive = drive
        self.scale = Float32(scale)
        self.paused = False
        self.show_outline = True
        self.filter = TextBuffer()


@fieldwise_init
struct PanelOut(Copyable, Movable):
    """What the panel ASKS the studio to do. No side effects on the model.

    ⚠ REQUESTS, NOT MUTATIONS, and not only for the non-generic rule. A panel
    that reset the sim itself would do it MID-FRAME, between the step and the
    draw, and the frame would render a pose that never existed. The studio
    applies these at the top of the next iteration.
    """

    var reset: Bool
    var step_once: Bool
    var reframe: Bool

    def __init__(out self):
        self.reset = False
        self.step_once = False
        self.reframe = False


def _label(names: List[String], i: Int, kind: String) -> String:
    """A name, or a bracketed index when the MJCF gave none.

    ⚠ THE BRACKETS ARE THE TELL. `FlatModelDef` stores "" for an unnamed
    element rather than synthesising `geom3`, so that an export cannot claim a
    name the source never had — MuJoCo itself returns NULL for walker2d's six
    actuators. The invention therefore happens HERE, at display, where it is
    visibly a display choice.
    """
    if i >= 0 and i < len(names) and names[i].byte_length() > 0:
        return names[i].copy()
    return String("<", kind, " ", i, ">")


def _contains(hay: String, needle: String) -> Bool:
    if needle.byte_length() == 0:
        return True
    return hay.find(needle) != -1


def _f2(v: Float64) -> String:
    var scaled = Int(v * 100.0 + (0.5 if v >= 0 else -0.5))
    var whole = scaled // 100
    var frac = scaled % 100
    if frac < 0:
        frac = -frac
    var f = String(frac) if frac >= 10 else "0" + String(frac)
    return String(whole) + "." + f


def _vec3(label: String, x: Float64, y: Float64, z: Float64) raises:
    ig_text(String(label, "  ", _f2(x), "  ", _f2(y), "  ", _f2(z)))


def ui_outline(
    mut p: StudioPanel,
    body_names: List[String],
    geom_names: List[String],
    body_parent: List[Int],
    geom_body: List[Int],
    height: Float32,
) raises:
    """The kinematic tree, selectable. Writes `p.sel_*` directly.

    ⚠ A FLAT LIST WHEN FILTERED, a tree otherwise — the same rule
    `ui_task_tree` uses, and for the same reason: making the user expand
    branches to reach something they just typed the name of defeats the
    search. dog has 62 bodies and 128 geoms, so the filter is not a luxury.

    ⚠ IDs ARE PUSHED PER ROW. ImGui identifies a widget by its LABEL, and a
    model with two unnamed geoms — or, far more common, a geom named after its
    body — would collide, leaving one of the two permanently unclickable. That
    failure looks like "selection is flaky", not like an id bug.
    """
    ig_set_next_item_width(-1.0)
    _ = ig_input_text(String("##filter"), p.filter, String("filter..."))
    var query = p.filter.value()

    if ig_begin_child(String("outline"), 0.0, height, True):
        if query.byte_length() > 0:
            var shown = 0
            for b in range(len(body_names)):
                var nm = _label(body_names, b, "body")
                if _contains(nm, query):
                    shown += 1
                    ig_push_id_int(b)
                    if ig_selectable(
                        nm, p.sel_kind == SEL_BODY and p.sel_index == b
                    ):
                        p.sel_kind = SEL_BODY
                        p.sel_index = b
                    ig_pop_id()
            for g in range(len(geom_names)):
                var nm = _label(geom_names, g, "geom")
                if _contains(nm, query):
                    shown += 1
                    ig_push_id_int(100000 + g)
                    if ig_selectable(
                        nm, p.sel_kind == SEL_GEOM and p.sel_index == g
                    ):
                        p.sel_kind = SEL_GEOM
                        p.sel_index = g
                    ig_pop_id()
            if shown == 0:
                ig_text_disabled(String("no match"))
        else:
            for b in range(len(body_names)):
                ig_push_id_int(b)
                # ⚠ THE ROOT OPENS, THE REST DO NOT. A humanoid expanded to
                # every leaf fills the panel before the inspector is reachable.
                if b <= 1:
                    ig_set_next_item_open(True)
                var is_sel = p.sel_kind == SEL_BODY and p.sel_index == b
                if ig_tree_node(_label(body_names, b, "body")):
                    if ig_small_button(String("select")):
                        p.sel_kind = SEL_BODY
                        p.sel_index = b
                    if is_sel:
                        ig_same_line()
                        ig_text_disabled(String("(selected)"))
                    for g in range(len(geom_body)):
                        if geom_body[g] != b:
                            continue
                        ig_push_id_int(100000 + g)
                        if ig_selectable(
                            _label(geom_names, g, "geom"),
                            p.sel_kind == SEL_GEOM and p.sel_index == g,
                        ):
                            p.sel_kind = SEL_GEOM
                            p.sel_index = g
                        ig_pop_id()
                    ig_tree_pop()
                ig_pop_id()
    ig_end_child()


def ui_inspector(
    p: StudioPanel,
    body_names: List[String],
    geom_names: List[String],
    body_parent: List[Int],
    geom_body: List[Int],
    rec: List[Float64],
) raises:
    """The selected element's record — READ ONLY in S1.

    `rec` is the studio's flattened view of whatever is selected, in the order
    this function reads it. Passing it as a `List[Float64]` rather than a
    `Model`/`Data` is what keeps this file non-generic; the studio owns the
    unpacking because it is the side that knows the dims.

    Layout: body -> [x y z  qw qx qy qz  mass];  geom -> [x y z  qw qx qy qz
    size0 size1 size2  r g b a].
    """
    if p.sel_kind == SEL_NONE or p.sel_index < 0:
        ig_text_disabled(String("nothing selected"))
        ig_text_disabled(String("click a geom, or pick one above"))
        return

    if p.sel_kind == SEL_BODY:
        ig_text(String("body ", p.sel_index, "  ",
                       _label(body_names, p.sel_index, "body")))
        var par = -1
        if p.sel_index > 0 and p.sel_index - 1 < len(body_parent):
            par = body_parent[p.sel_index - 1]
        ig_text_disabled(String("parent: ",
                                _label(body_names, par, "body")))
        if len(rec) >= 8:
            _vec3(String("xpos "), rec[0], rec[1], rec[2])
            ig_text(String("xquat  ", _f2(rec[3]), "  ", _f2(rec[4]),
                           "  ", _f2(rec[5]), "  ", _f2(rec[6])))
            ig_text(String("mass   ", _f2(rec[7])))
    else:
        ig_text(String("geom ", p.sel_index, "  ",
                       _label(geom_names, p.sel_index, "geom")))
        var b = -1
        if p.sel_index < len(geom_body):
            b = geom_body[p.sel_index]
        ig_text_disabled(String("body: ", _label(body_names, b, "body")))
        if len(rec) >= 14:
            _vec3(String("pos  "), rec[0], rec[1], rec[2])
            ig_text(String("quat   ", _f2(rec[3]), "  ", _f2(rec[4]),
                           "  ", _f2(rec[5]), "  ", _f2(rec[6])))
            _vec3(String("size "), rec[7], rec[8], rec[9])
            ig_text_colored(
                String("rgba   ", _f2(rec[10]), " ", _f2(rec[11]),
                       " ", _f2(rec[12]), " ", _f2(rec[13])),
                Float32(rec[10]), Float32(rec[11]), Float32(rec[12]), 1.0,
            )
    ig_spacing()
    ig_text_disabled(String("read-only in S1 — editing is S3"))


def build_panel(
    mut p: StudioPanel,
    path: String,
    nbody: Int,
    ngeom: Int,
    nq: Int,
    nv: Int,
    nact: Int,
    step_i: Int,
    ncon: Int,
    contact_budget: Int,
    step_us: Float64,
    panel_h: Float32,
    body_names: List[String],
    geom_names: List[String],
    body_parent: List[Int],
    geom_body: List[Int],
    rec: List[Float64],
) raises -> PanelOut:
    """The whole sidebar, in one non-generic function. See the module header."""
    var out = PanelOut()
    ig_begin_panel(String("physics3d studio"), 0.0, 0.0, SIDEBAR_W, panel_h)

    ig_text(String(path))
    ig_text_disabled(String("nbody ", nbody, "  ngeom ", ngeom,
                            "  nq ", nq, "  nv ", nv, "  nact ", nact))
    ig_text_disabled(String("step ", step_i, "   ", _f2(step_us), " us"))

    # ⚠⚠ THE CONTACT BUDGET IS A BAR, NOT A LOG LINE. Overflowing
    # `max_contacts` DROPS contacts with no error and no crash — the model
    # just gets quietly softer — and a scene composer raises the count every
    # time a prop lands. §1.2 names this as one of the two silent-failure
    # surfaces the composer creates, so the number that would warn you lives
    # where the eye already is.
    var frac = Float32(0.0)
    if contact_budget > 0:
        frac = Float32(ncon) / Float32(contact_budget)
    ig_progress_bar(frac, -1.0, 0.0,
                    String("contacts ", ncon, "/", contact_budget))
    if ig_is_item_hovered():
        ig_set_tooltip(String(
            "contacts in flight vs the workspace budget.\n"
            "at 100% the solver DROPS contacts silently."
        ))

    ig_separator_text(String("outline"))
    _ = ig_checkbox(String("show tree"), p.show_outline)
    if p.show_outline:
        ui_outline(p, body_names, geom_names, body_parent, geom_body,
                   panel_h - 430.0)

    ig_separator_text(String("inspector"))
    ui_inspector(p, body_names, geom_names, body_parent, geom_body, rec)

    ig_separator_text(String("drive"))
    var modes = List[String]()
    modes.append(String("zero"))
    modes.append(String("random"))
    modes.append(String("sweep"))
    var cur = Int32(p.drive)
    ig_set_next_item_width(-60.0)
    if ig_combo(String("mode"), cur, modes):
        p.drive = Int(cur)
    ig_set_next_item_width(-60.0)
    _ = ig_slider_float(String("scale"), p.scale, 0.0, 1.0)
    _ = ig_checkbox(String("pause"), p.paused)
    ig_same_line()
    if ig_button(String("step")):
        out.step_once = True
    ig_same_line()
    if ig_button(String("reset")):
        out.reset = True

    ig_separator_text(String("view"))
    if ig_button(String("reframe")):
        out.reframe = True
    ig_spacing()
    ig_text_disabled(String("drag: orbit   shift+drag: pan"))
    ig_text_disabled(String("click a geom to select it"))

    ig_end()
    return out^
