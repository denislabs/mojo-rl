"""Immediate-mode widgets over `Renderer3D`'s screen-space quad pipeline.

IMMEDIATE MODE, meaning a widget is a CALL that both draws and answers: there
is no widget tree, no retained state, no callbacks, no ids to keep in sync
with your data. `if ui.button(r, "reset"): do_reset()` is the whole pattern.
The only state is `UI` itself — pointer position and whether a click is still
unconsumed this frame — which is why it can be built and thrown away every
frame.

WHY NOT A GUI LIBRARY. SDL3 ships no widgets, so something has to provide
them. Dear ImGui or Nuklear would each still need a rendering backend written
against THIS project's SDL3-GPU/Metal pipeline, and that backend is about as
much code as the widgets below — plus C FFI and vendoring. Everything the
widgets need was already here: a screen-space textured-quad path (`draw_text`),
a font atlas, `SDL_GetMouseState`, `MouseButtonEvent` with coordinates, and
`SDL_StartTextInput` for typing if a filter box is ever wanted. Reconsider
this if the ambition grows to docking panels, property inspectors and plots —
that is where ImGui's depth starts paying for its backend.

⚠ RECTANGLES SHARE THE TEXT BUDGET. `draw_rect` writes into the same vertex
buffer as `draw_text`, capped at `MAX_TEXT_CHARS` quads. A panel costs a few
"characters"; a 39-row list costs about as much as the labels on it. Budget
exhaustion drops quads silently, so a list that renders half its rows is that,
not a logic bug.

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


comptime UI_ROW_H: Float32 = 22.0
"""Row height that fits 16 px text (scale 2) with 3 px of padding."""

comptime _CHAR_W: Float32 = 16.0
"""Advance per character at scale 2 — `draw_text` uses 8 * scale."""


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
    ):
        self.texts.append(UIText(x, y, text, color))

    def button(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        text: String,
        active: Bool = False,
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
                y + (h - 16.0) * 0.5,
                text,
                Color(255, 255, 255, 255) if (hot or active)
                else Color(200, 208, 222, 255),
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
    ) -> Int:
        """Scrollable single-choice list. Returns the clicked index, else -1.

        `scroll` is the caller's — the widget stays stateless, so paging is
        the caller's business and there is no hidden state to get out of sync
        with the item list.
        """
        var picked = -1
        var n = len(items)
        for i in range(rows):
            var idx = scroll + i
            if idx >= n:
                break
            var ry = y + Float32(i) * UI_ROW_H
            if self.button(
                x, ry, w, UI_ROW_H - 2.0, items[idx], idx == current
            ):
                picked = idx
        return picked

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
