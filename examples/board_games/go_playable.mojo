"""Playable Go 9x9 -- two humans alternate placing black and white stones.

Controls:
  Arrow keys: move cursor
  Space / Left click: place stone at cursor
  P / Right click: pass
  R: reset after game ends
  Close window to quit
"""

from std.memory import alloc
from mojo_rl.envs.board_games.go import GoEnv
from mojo_rl.render import Renderer2D, SDL_Color
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode
from mojo_rl.render.sdl.sdl_mouse import get_mouse_state, MouseButtonFlags


def main() raises:
    print("=== Playable Go 9x9 ===")
    print("Controls:")
    print("  Arrow keys to move cursor")
    print("  Space / Left click to place stone")
    print("  P / Right click to pass")
    print("  R to reset after game ends")
    print("  Close window to quit")

    var env = GoEnv[9, DType.float64]()
    _ = env.reset()

    var renderer = Renderer2D(width=500, height=550, fps=30, title="Go 9x9")

    # Colors
    var board_color = SDL_Color(r=0xDE, g=0xB8, b=0x87, a=0xFF)  # tan/wooden
    var line_color = SDL_Color(r=0x00, g=0x00, b=0x00, a=0xFF)
    var black_stone = SDL_Color(r=0x10, g=0x10, b=0x10, a=0xFF)
    var white_stone = SDL_Color(r=0xF0, g=0xF0, b=0xF0, a=0xFF)
    var black_outline = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var black_outline2 = SDL_Color(r=0x00, g=0x00, b=0x00, a=0xFF)
    var cursor_black = SDL_Color(r=0x10, g=0x10, b=0x10, a=0x60)
    var cursor_white = SDL_Color(r=0xF0, g=0xF0, b=0xF0, a=0x60)
    var hoshi_color = SDL_Color(r=0x00, g=0x00, b=0x00, a=0xFF)
    var text_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var status_bg = SDL_Color(r=0x33, g=0x33, b=0x33, a=0xFF)

    # Board geometry
    var margin = 30  # margin from edge to first line
    var cell_size = 50  # distance between lines
    # Grid spans from margin to margin + 8*cell_size = 30 to 430
    # Total board area width: 2*margin + 8*cell_size = 60 + 400 = 460, centered in 500

    # Cursor position (row, col) on the 9x9 grid
    var cursor_row = 4
    var cursor_col = 4

    # Debounce
    var prev_up = False
    var prev_down = False
    var prev_left = False
    var prev_right = False
    var prev_space = False
    var prev_p = False
    var prev_r = False

    var prev_mouse_left = False
    var prev_mouse_right = False
    var mouse_x_ptr = alloc[Float32](1)
    var mouse_y_ptr = alloc[Float32](1)
    mouse_x_ptr[] = Float32(0)
    mouse_y_ptr[] = Float32(0)

    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    while renderer.begin_frame_with_color(board_color):
        var keys = get_keyboard_state(numkeys_ptr)

        # Read current key states
        var cur_up = Bool(keys[Int(Scancode.SCANCODE_UP)])
        var cur_down = Bool(keys[Int(Scancode.SCANCODE_DOWN)])
        var cur_left = Bool(keys[Int(Scancode.SCANCODE_LEFT)])
        var cur_right = Bool(keys[Int(Scancode.SCANCODE_RIGHT)])
        var cur_space = Bool(keys[Int(Scancode.SCANCODE_SPACE)])
        var cur_p = Bool(keys[Int(Scancode.SCANCODE_P)])
        var cur_r = Bool(keys[Int(Scancode.SCANCODE_R)])

        # Read mouse state
        var mouse_buttons = get_mouse_state(
            rebind[Pointer[Float32, MutAnyOrigin]](mouse_x_ptr),
            rebind[Pointer[Float32, MutAnyOrigin]](mouse_y_ptr),
        )
        var cur_mouse_left = (Int(mouse_buttons.value) & 1) != 0
        var cur_mouse_right = (Int(mouse_buttons.value) & 4) != 0
        var mouse_x = Int(mouse_x_ptr[])
        var mouse_y = Int(mouse_y_ptr[])

        # Convert mouse position to nearest grid intersection
        var grid_col = (mouse_x - margin + cell_size // 2) // cell_size
        var grid_row = (mouse_y - margin + cell_size // 2) // cell_size
        var mouse_on_board = (
            grid_row >= 0 and grid_row <= 8 and grid_col >= 0 and grid_col <= 8
        )

        var game_over = env.done

        # Handle reset
        if cur_r and not prev_r and game_over:
            _ = env.reset()
            cursor_row = 4
            cursor_col = 4

        if not game_over:
            # Arrow key cursor movement (debounced)
            if cur_up and not prev_up:
                if cursor_row > 0:
                    cursor_row -= 1
            if cur_down and not prev_down:
                if cursor_row < 8:
                    cursor_row += 1
            if cur_left and not prev_left:
                if cursor_col > 0:
                    cursor_col -= 1
            if cur_right and not prev_right:
                if cursor_col < 8:
                    cursor_col += 1

            # Space to place stone at cursor
            if cur_space and not prev_space:
                var action = cursor_row * 9 + cursor_col
                # Check if legal
                var mask = env.legal_action_mask()
                if mask[action]:
                    _ = env._step_impl(action)

            # P to pass
            if cur_p and not prev_p:
                _ = env._step_impl(81)  # PASS_ACTION = 9*9 = 81

            # Mouse hover updates cursor position
            if mouse_on_board:
                cursor_row = grid_row
                cursor_col = grid_col

            # Left click to place stone
            if cur_mouse_left and not prev_mouse_left and mouse_on_board:
                var m_action = cursor_row * 9 + cursor_col
                var m_mask = env.legal_action_mask()
                if m_mask[m_action]:
                    _ = env._step_impl(m_action)

            # Right click to pass
            if cur_mouse_right and not prev_mouse_right:
                _ = env._step_impl(81)  # PASS_ACTION = 9*9 = 81

        # === Rendering ===

        # Draw grid lines
        for i in range(9):
            # Horizontal lines
            var y = margin + i * cell_size
            renderer.draw_line(
                margin, y, margin + 8 * cell_size, y, line_color, 1
            )
            # Vertical lines
            var x = margin + i * cell_size
            renderer.draw_line(
                x, margin, x, margin + 8 * cell_size, line_color, 1
            )

        # Draw star points (hoshi) at (2,2), (2,6), (6,2), (6,6), (4,4)
        def _hoshi_r(i: Int) -> Int:
            if i == 0:
                return 2
            if i == 1:
                return 2
            if i == 2:
                return 6
            if i == 3:
                return 6
            return 4

        def _hoshi_c(i: Int) -> Int:
            if i == 0:
                return 2
            if i == 1:
                return 6
            if i == 2:
                return 2
            if i == 3:
                return 6
            return 4

        for i in range(5):
            var hx = margin + _hoshi_c(i) * cell_size
            var hy = margin + _hoshi_r(i) * cell_size
            renderer.draw_circle(hx, hy, 4, hoshi_color, filled=True)

        # Draw stones
        for row in range(9):
            for col in range(9):
                var cell_idx = row * 9 + col
                var cell_val = Int(env.state[cell_idx])
                var cx = margin + col * cell_size
                var cy = margin + row * cell_size
                var stone_r = 20

                if cell_val == 1:
                    # Black stone: filled black with white outline
                    renderer.draw_circle(
                        cx, cy, stone_r + 1, black_outline, filled=True
                    )
                    renderer.draw_circle(
                        cx, cy, stone_r, black_stone, filled=True
                    )
                elif cell_val == 2:
                    # White stone: filled white with black outline
                    renderer.draw_circle(
                        cx, cy, stone_r + 1, black_outline2, filled=True
                    )
                    renderer.draw_circle(
                        cx, cy, stone_r, white_stone, filled=True
                    )

        # Draw cursor (translucent stone at cursor position)
        if not game_over:
            var ccx = margin + cursor_col * cell_size
            var ccy = margin + cursor_row * cell_size
            var cell_idx = cursor_row * 9 + cursor_col
            var cell_val = Int(env.state[cell_idx])
            if cell_val == 0:
                var player = env.current_player()
                if player == 0:
                    renderer.draw_circle(
                        ccx, ccy, 18, cursor_black, filled=True
                    )
                else:
                    renderer.draw_circle(
                        ccx, ccy, 18, cursor_white, filled=True
                    )
            # If occupied, show a small yellow indicator
            var indicator_color = SDL_Color(r=0xFF, g=0xFF, b=0x00, a=0xFF)
            renderer.draw_circle(ccx, ccy, 3, indicator_color, filled=True)

        # Status bar at bottom (y=500..550)
        renderer.draw_rect(0, 500, 500, 50, status_bg)

        var game_result = env.game_result()
        if game_result == 0:
            var player = env.current_player()
            if player == 0:
                renderer.draw_text(
                    "Black's turn  (Space=place, P=pass)", 100, 520, text_color
                )
            else:
                renderer.draw_text(
                    "White's turn  (Space=place, P=pass)", 100, 520, text_color
                )
        elif game_result == 1:
            renderer.draw_text(
                "Black Wins!  (R to reset)", 140, 520, text_color
            )
        elif game_result == 2:
            renderer.draw_text(
                "White Wins!  (R to reset)", 140, 520, text_color
            )
        else:
            renderer.draw_text("Draw!  (R to reset)", 160, 520, text_color)

        renderer.flip()

        # Save previous key states
        prev_up = cur_up
        prev_down = cur_down
        prev_left = cur_left
        prev_right = cur_right
        prev_space = cur_space
        prev_p = cur_p
        prev_r = cur_r
        prev_mouse_left = cur_mouse_left
        prev_mouse_right = cur_mouse_right

    numkeys_ptr.free()
    mouse_x_ptr.free()
    mouse_y_ptr.free()
    renderer.close()
    print("=== Done ===")
