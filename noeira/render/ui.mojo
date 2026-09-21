"""Immediate-mode widgets over `Renderer3D`'s screen-space quad pipeline.

IMMEDIATE MODE, meaning a widget is a CALL that both draws and answers: there
is no widget tree, no retained state, no callbacks, no ids to keep in sync
with your data. `if ui.button(r, "reset"): do_reset()` is the whole pattern.
The only state is `UI` itself — pointer position and whether a click is still
unconsumed this frame — which is why it can be built and thrown away every
frame.

WHY THIS EXISTS RATHER THAN A GUI LIBRARY — and the honest current answer.
SDL3 ships no widgets, so something has to provide them, and everything these
need was already here: a screen-space textured-quad path (`draw_text`), a font
atlas, and mouse coordinates the event pump already received.

⚠ AN EARLIER VERSION OF THIS NOTE CLAIMED Dear ImGui WOULD NEED A RENDERING
BACKEND WRITTEN AGAINST THIS PROJECT'S PIPELINE. That was wrong: ImGui ships an
official `imgui_impl_sdlgpu3` backend, and a spike drove it from Mojo through a
152-line C shim, rendering into a render pass this project's own bindings
built. So the reason to stay here is NOT that ImGui is expensive to integrate.

What this layer still buys is no C++ toolchain, no vendored dependency and no
second build step. What it cannot buy, at any reasonable price, is drag
tracking (a slider that streams values mid-gesture needs pointer capture and
per-widget retained state), real text editing (selection, clipboard, IME), or
plots. `text_input` below is the boundary case: it works because task names are
lowercase identifiers, and it would not survive contact with anything else.

⚠ RECTANGLES SHARE THE TEXT BUDGET. `draw_rect` writes into the same vertex
buffer as `draw_text`, capped at `MAX_TEXT_CHARS` quads. A panel costs a few
"characters"; a 39-row list costs about as much as the labels on it (measured:
79 rects + ~624 glyphs, against an engine HUD that already spends ~460). The
budget was raised 512 → 2048 for exactly that list, and overflow now prints
once instead of silently dropping quads — a half-drawn list is a budget that
ran out, and it should not have to be diagnosed as a layout bug first.

⚠ THE CLICK IS CONSUMED BY THE FIRST WIDGET THAT WANTS IT. Widgets are tested
in call order, so overlapping ones resolve to whichever is called first — draw
panels before their contents, and the contents win, which is what you want.

⚠ WIDGETS RECORD, THEY DO NOT DRAW. A widget must paint between
`begin_frame` and `end_frame`, and that whole span lives inside
`Phyics3dEnv.render_frame()` where an application cannot reach. Rather than
thread a callback through the env (Mojo makes capturing closures across that
boundary painful), `UI` accumulates a COMMAND LIST that the env hands to the
renderer at HUD time. It splits cleanly because immediate mode already does
two separable things: hit-testing, which needs only the pointer, and drawing,
which needs the renderer. Hit-testing answers immediately; drawing is deferred.

Usage:

    var ui = UI(env.renderer_mouse_x(), env.renderer_mouse_y(),
                env.renderer_take_click())
    ui.panel(10, 10, 220, 400)
    if ui.button(20, 20, 200, 24, "reset", False):
        _ = env.reset()
    var picked = ui.list_select(         # -1 unless a row was clicked
        10, 40, 198, len(names), names, scroll=0, current=cur,
        text_scale=1, row_h=UI_ROW_H_SMALL,
    )
    env.set_ui(ui.rects, ui.texts)   # drawn during the next render_frame
"""

from .types import Color


@fieldwise_init
struct UIRect(Copyable, Movable):
    """A deferred filled rectangle, screen pixels, top-left origin."""

    var x: Float32
    var y: Float32
    var w: Float32
    var h: Float32
    var color: Color


@fieldwise_init
struct UIText(Copyable, Movable):
    """A deferred text run, screen pixels, top-left origin."""

    var x: Float32
    var y: Float32
    var text: String
    var color: Color
    var scale: Int
    """Glyph scale: 1 = 8 px per character, 2 = 16 px. Long lists need 1."""


comptime UI_ROW_H: Float32 = 22.0
"""Row height that fits 16 px text (scale 2) with 3 px of padding."""

comptime UI_ROW_H_SMALL: Float32 = 14.0
"""Row height for scale-1 text. 39 rows fit a 720 px window; at scale 2 they
do not — 39 * 22 = 858 — which is the whole reason `text_scale` exists."""


def ui_char_w(scale: Int) -> Float32:
    """Advance per character — `draw_text` uses 8 * scale."""
    return Float32(8 * scale)


comptime UI_KEY_BACKSPACE: Int = 8
comptime UI_KEY_ESCAPE: Int = 27
comptime UI_KEY_RETURN: Int = 13


def ui_apply_key(mut text: String, key: Int) -> Bool:
    """Apply one SDL keycode to a text buffer. True if the buffer changed.

    Deliberately tiny: printable ASCII and backspace. That covers a filter over
    identifiers like `manipulator_insert_ball`, which is the only text this UI
    has any business accepting. Anything more — selection, clipboard, IME,
    composed characters — is where a hand-rolled field stops being cheap and a
    real widget library starts paying for itself.

    ⚠ SDL DELIVERS UNSHIFTED KEYCODES here, so this sees lowercase letters and
    the unshifted digit row. That is exactly right for matching task names and
    exactly wrong for anything needing capitals.
    """
    if key == UI_KEY_BACKSPACE:
        if text.byte_length() > 0:
            # ⚠ VIA A TEMPORARY. `text = String(text[...])` aliases the
            # destination with the slice it is built from and the compiler
            # rejects it outright, which is a mercy — that is exactly the shape
            # that silently corrupts in languages without the check.
            var trimmed = String(text[byte = 0 : text.byte_length() - 1])
            text = trimmed
            return True
        return False
    if key >= 32 and key < 127:
        text += chr(key)
        return True
    return False


struct UI(Copyable, Movable):
    """One frame's pointer state. Build it per frame; it holds nothing else."""

    var mx: Float32
    var my: Float32
    var click: Bool
    """Unconsumed click. Goes False as soon as a widget claims it."""
    var rects: List[UIRect]
    var texts: List[UIText]

    def __init__(out self, mx: Float32, my: Float32, click: Bool):
        self.mx = mx
        self.my = my
        self.click = click
        self.rects = List[UIRect]()
        self.texts = List[UIText]()

    def _hit(self, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
        return (
            self.mx >= x
            and self.mx <= x + w
            and self.my >= y
            and self.my <= y + h
        )

    def panel(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        color: Color = Color(18, 20, 28, 225),
    ):
        """Background slab. Record before its contents so they paint on top."""
        self.rects.append(UIRect(x, y, w, h, color))

    def label(
        mut self,
        x: Float32,
        y: Float32,
        text: String,
        color: Color = Color(210, 220, 235, 255),
        text_scale: Int = 2,
    ):
        self.texts.append(UIText(x, y, text, color, text_scale))

    def button(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        text: String,
        active: Bool = False,
        text_scale: Int = 2,
    ) -> Bool:
        """Draw a button; return True on the frame it is clicked.

        `active` renders it as a pressed toggle — enough to build radio groups
        out of buttons without a separate widget.
        """
        var hot = self._hit(x, y, w, h)
        var bg = Color(70, 110, 160, 240) if active else (
            Color(55, 62, 80, 240) if hot else Color(38, 43, 56, 235)
        )
        self.rects.append(UIRect(x, y, w, h, bg))
        # 1 px top highlight so a flat slab reads as raised.
        self.rects.append(
            UIRect(x, y, w, 1.0, Color(120, 135, 160, 200))
        )
        self.texts.append(
            UIText(
                x + 6.0,
                y + (h - Float32(8 * text_scale)) * 0.5,
                text,
                Color(255, 255, 255, 255) if (hot or active)
                else Color(200, 208, 222, 255),
                text_scale,
            )
        )
        if hot and self.click:
            self.click = False  # consume, so nothing behind also fires
            return True
        return False

    def list_select(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        rows: Int,
        items: List[String],
        scroll: Int,
        current: Int,
        text_scale: Int = 2,
        row_h: Float32 = UI_ROW_H,
    ) -> Int:
        """Scrollable single-choice list. Returns the clicked index, else -1.

        `scroll` is the caller's — the widget stays stateless, so paging is
        the caller's business and there is no hidden state to get out of sync
        with the item list. Pass `rows >= len(items)` and it simply draws them
        all, which is what a list short enough to fit should do.
        """
        var picked = -1
        var n = len(items)
        for i in range(rows):
            var idx = scroll + i
            if idx >= n:
                break
            var ry = y + Float32(i) * row_h
            if self.button(
                x, ry, w, row_h - 2.0, items[idx], idx == current, text_scale
            ):
                picked = idx
        return picked

    def tree_header(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        label: String,
        open: Bool,
        count: Int = -1,
        text_scale: Int = 1,
    ) -> Bool:
        """A collapsible group header. Returns True on the frame it is clicked.

        The OPEN STATE IS THE CALLER'S, like `list_select`'s scroll: the widget
        stays stateless, so there is no hidden per-header state to fall out of
        sync with a changing group list.

        Why a tree at all: a flat list costs one row per item, and rows share
        the renderer's quad budget with the HUD. 16 collapsed headers plus one
        expanded group is ~22 rows where a flat list of the same content is 39,
        and at Phase 2 scale (85 tasks) the flat version does not fit at all.
        """
        var hot = self._hit(x, y, w, h)
        self.rects.append(
            UIRect(x, y, w, h, Color(46, 52, 68, 240) if hot
                   else Color(32, 37, 48, 235))
        )
        var arrow = String("- ") if open else String("+ ")
        var txt = arrow + label
        if count >= 0:
            txt += String("  (") + String(count) + String(")")
        self.texts.append(
            UIText(
                x + 5.0,
                y + (h - Float32(8 * text_scale)) * 0.5,
                txt,
                Color(235, 240, 250, 255) if hot
                else Color(185, 196, 214, 255),
                text_scale,
            )
        )
        if hot and self.click:
            self.click = False
            return True
        return False

    def text_input(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        text: String,
        focused: Bool,
        placeholder: String = String(""),
        text_scale: Int = 1,
    ) -> Bool:
        """A single-line field. Returns True on the frame it is CLICKED.

        ⚠ THIS WIDGET DOES NOT EDIT THE STRING. Editing needs keyboard events,
        which arrive through the renderer, so the caller owns the buffer and
        feeds keycodes to `ui_apply_key`. Splitting it this way keeps the whole
        UI layer free of any renderer dependency.

        ⚠ THE CALLER MUST ALSO CALL `set_text_input_mode(focused)`. The
        renderer claims R, S, V, SPACE, 1-9 and ESC for its own bindings, so an
        unfocused-but-typed-into field would fire screenshots and recordings
        instead of inserting characters.
        """
        var hot = self._hit(x, y, w, h)
        self.rects.append(
            UIRect(x, y, w, h, Color(12, 14, 20, 255))
        )
        # A bright 1 px underline is the focus cue; a full border would cost
        # four rects out of the shared quad budget for the same information.
        self.rects.append(
            UIRect(x, y + h - 1.0, w, 1.0,
                   Color(120, 190, 255, 255) if focused
                   else Color(70, 78, 95, 220))
        )
        var shown = text
        var col = Color(225, 232, 245, 255)
        if text.byte_length() == 0 and not focused:
            shown = placeholder
            col = Color(110, 120, 140, 255)
        if focused:
            shown += "_"
        self.texts.append(
            UIText(
                x + 5.0,
                y + (h - Float32(8 * text_scale)) * 0.5,
                shown,
                col,
                text_scale,
            )
        )
        if hot and self.click:
            self.click = False
            return True
        return False

    def scrollbar_hint(
        mut self,
        x: Float32,
        y: Float32,
        h: Float32,
        rows: Int,
        total: Int,
        scroll: Int,
    ):
        """A non-interactive position indicator beside a list.

        Deliberately not draggable: dragging needs press/release tracking that
        `take_click`'s one-shot model does not carry, and a wrong-feeling
        scrollbar is worse than none. Wheel or buttons drive `scroll`.
        """
        if total <= rows:
            return
        self.rects.append(UIRect(x, y, 4.0, h, Color(30, 34, 44, 200)))
        var frac = Float32(rows) / Float32(total)
        var thumb_h = h * frac
        if thumb_h < 12.0:
            thumb_h = 12.0
        var span = h - thumb_h
        var pos = Float32(scroll) / Float32(total - rows)
        self.rects.append(
            UIRect(x, y + span * pos, 4.0, thumb_h, Color(110, 130, 165, 230))
        )
