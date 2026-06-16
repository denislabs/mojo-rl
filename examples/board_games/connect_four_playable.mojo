"""Playable ConnectFour -- two humans alternate dropping pieces into a 7x6 grid.

Controls:
  Mouse click: drop piece in clicked column
  Left/Right arrows: move column selector
  Space/Return: drop piece in selected column
  R: reset after game ends
  Close window to quit
"""

from std.memory import alloc
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv
from mojo_rl.render import Renderer2D, SDL_Color
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode
from mojo_rl.render.sdl.sdl_mouse import get_mouse_state, MouseButtonFlags


def main() raises:
    print("=== Playable ConnectFour ===")
    print("Controls:")
    print("  Mouse click to drop piece in clicked column")
    print("  Left/Right arrows to move column selector")
    print("  Space/Return to drop piece")
    print("  R to reset after game ends, close window to quit")

    var env = ConnectFourEnv[DType.float64]()
    _ = env.reset()

    # Window: 560 wide (7 cols * 80px), 530 tall (480 board + 50 status)
    var renderer = Renderer2D(
        width=560, height=530, fps=30, title="ConnectFour"
    )

    # Colors (the board itself is drawn by `env.render_board`; these are the
    # play-script chrome: selector + status text).
    var red_color = SDL_Color(r=0xFF, g=0x22, b=0x22, a=0xFF)
    var yellow_color = SDL_Color(r=0xFF, g=0xDD, b=0x00, a=0xFF)
    var bg_color = SDL_Color(r=0x11, g=0x11, b=0x44, a=0xFF)
    var win_text_color = SDL_Color(r=0xFF, g=0xDD, b=0x00, a=0xFF)
    var selector_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var status_bg = SDL_Color(r=0x11, g=0x11, b=0x22, a=0xFF)

    # Board geometry
    var cell_size = 80
    var board_cols = 7
    var board_rows = 6
    var board_width = board_cols * cell_size  # 560
    var board_height = board_rows * cell_size  # 480

    # Column selector
    var selected_col = 3

    # Debounce
    var prev_left = False
    var prev_right = False
    var prev_space = False
    var prev_return = False
    var prev_r = False

    var prev_mouse_left = False
    var mouse_x_ptr = alloc[Float32](1)
    var mouse_y_ptr = alloc[Float32](1)
    mouse_x_ptr[] = Float32(0)
    mouse_y_ptr[] = Float32(0)

    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    while renderer.begin_frame_with_color(bg_color):
        var keys = get_keyboard_state(numkeys_ptr)

        var cur_left = Bool(keys[Int(Scancode.SCANCODE_LEFT)])
        var cur_right = Bool(keys[Int(Scancode.SCANCODE_RIGHT)])
        var cur_space = Bool(keys[Int(Scancode.SCANCODE_SPACE)])
        var cur_return = Bool(keys[Int(Scancode.SCANCODE_RETURN)])
        var cur_r = Bool(keys[Int(Scancode.SCANCODE_R)])

        # Mouse state
        var mouse_buttons = get_mouse_state(
            rebind[UnsafePointer[Float32, MutAnyOrigin]](mouse_x_ptr),
            rebind[UnsafePointer[Float32, MutAnyOrigin]](mouse_y_ptr),
        )
        var cur_mouse_left = (Int(mouse_buttons.value) & 1) != 0
        var mouse_x = Int(mouse_x_ptr[])

        # Convert mouse x to column
        var mouse_col = mouse_x // cell_size
        var mouse_on_board = mouse_col >= 0 and mouse_col < board_cols

        # Update selected column from mouse hover
        if mouse_on_board:
            selected_col = mouse_col

        var game_over = env.done

        # Handle reset
        if cur_r and not prev_r and game_over:
            _ = env.reset()
            selected_col = 3

        if not game_over:
            # Column selector movement (debounced)
            if cur_left and not prev_left:
                if selected_col > 0:
                    selected_col -= 1
            if cur_right and not prev_right:
                if selected_col < board_cols - 1:
                    selected_col += 1

            # Drop piece (keyboard)
            if (cur_space and not prev_space) or (
                cur_return and not prev_return
            ):
                _ = env._step_impl(selected_col)

            # Drop piece (mouse click)
            if cur_mouse_left and not prev_mouse_left and mouse_on_board:
                _ = env._step_impl(mouse_col)

        # === Rendering ===

        # Draw selector indicator at top
        # The board starts at y=50 to leave room for the selector
        var selector_cx = selected_col * cell_size + cell_size // 2
        if not game_over:
            var player = env.current_player()
            var sel_color = red_color if player == 0 else yellow_color
            renderer.draw_circle(selector_cx, 22, 18, sel_color, filled=True)
            # Small triangle/arrow pointing down
            renderer.draw_line(
                selector_cx, 42, selector_cx - 8, 35, selector_color, 2
            )
            renderer.draw_line(
                selector_cx, 42, selector_cx + 8, 35, selector_color, 2
            )

        # Draw the board (background + pieces + grid) via the env.
        env.render_board(renderer)

        # Status bar at bottom (y=530-50..530)
        renderer.draw_rect(0, 50 + board_height, board_width, 50, status_bg)

        var game_result = env.game_result()
        if game_result == 0:
            var player = env.current_player()
            if player == 0:
                renderer.draw_text(
                    "Red's turn", 230, 50 + board_height + 20, red_color
                )
            else:
                renderer.draw_text(
                    "Yellow's turn", 218, 50 + board_height + 20, yellow_color
                )
        elif game_result == 1:
            renderer.draw_text(
                "Red Wins!  (R to reset)",
                180,
                50 + board_height + 20,
                win_text_color,
            )
        elif game_result == 2:
            renderer.draw_text(
                "Yellow Wins!  (R to reset)",
                168,
                50 + board_height + 20,
                win_text_color,
            )
        else:
            renderer.draw_text(
                "Draw!  (R to reset)",
                195,
                50 + board_height + 20,
                win_text_color,
            )

        renderer.flip()

        # Save previous key states
        prev_left = cur_left
        prev_right = cur_right
        prev_space = cur_space
        prev_return = cur_return
        prev_r = cur_r
        prev_mouse_left = cur_mouse_left

    mouse_x_ptr.free()
    mouse_y_ptr.free()
    numkeys_ptr.free()
    renderer.close()
    print("=== Done ===")
