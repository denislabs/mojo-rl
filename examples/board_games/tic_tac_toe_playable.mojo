"""Playable TicTacToe -- two humans alternate placing X and O on a 3x3 board.

Controls:
  Mouse click: place mark at clicked cell
  Arrow keys: move cursor
  Space/Return: place mark at cursor
  Numpad 1-9: place mark directly (1=bottom-left, 9=top-right)
  R: reset after game ends
  Close window to quit
"""

from std.memory import alloc
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv
from mojo_rl.render import Renderer2D, SDL_Color
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode
from mojo_rl.render.sdl.sdl_mouse import get_mouse_state, MouseButtonFlags


def main() raises:
    print("=== Playable TicTacToe ===")
    print("Controls:")
    print("  Mouse click to place mark at clicked cell")
    print("  Arrow keys to move cursor, Space/Return to place mark")
    print("  Numpad 1-9 to place directly (1=bottom-left, 9=top-right)")
    print("  R to reset after game ends, close window to quit")

    var env = TicTacToeEnv[DType.float64]()
    _ = env.reset()

    var renderer = Renderer2D(width=400, height=450, fps=30, title="TicTacToe")

    # Colors
    var bg_color = SDL_Color(r=0x1A, g=0x5C, b=0x2A, a=0xFF)
    var grid_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var x_color = SDL_Color(r=0xFF, g=0x44, b=0x44, a=0xFF)
    var o_color = SDL_Color(r=0x44, g=0x88, b=0xFF, a=0xFF)
    var cursor_color = SDL_Color(r=0xFF, g=0xFF, b=0x00, a=0xFF)
    var text_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var win_color = SDL_Color(r=0xFF, g=0xDD, b=0x00, a=0xFF)

    # Cursor position (row, col) -- row 0 = top visually
    var cursor_row = 1
    var cursor_col = 1

    # Debounce: track previous key states
    var prev_up = False
    var prev_down = False
    var prev_left = False
    var prev_right = False
    var prev_space = False
    var prev_return = False
    var prev_r = False
    var prev_kp1 = False
    var prev_kp2 = False
    var prev_kp3 = False
    var prev_kp4 = False
    var prev_kp5 = False
    var prev_kp6 = False
    var prev_kp7 = False
    var prev_kp8 = False
    var prev_kp9 = False

    var prev_mouse_left = False
    var mouse_x_ptr = alloc[Float32](1)
    var mouse_y_ptr = alloc[Float32](1)
    mouse_x_ptr[] = Float32(0)
    mouse_y_ptr[] = Float32(0)

    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    # Board geometry
    var cell_size = 133  # 400 / 3
    var board_size = 400

    while renderer.begin_frame_with_color(bg_color):
        var keys = get_keyboard_state(numkeys_ptr)

        # Read current key states
        var cur_up = Bool(keys[Int(Scancode.SCANCODE_UP)])
        var cur_down = Bool(keys[Int(Scancode.SCANCODE_DOWN)])
        var cur_left = Bool(keys[Int(Scancode.SCANCODE_LEFT)])
        var cur_right = Bool(keys[Int(Scancode.SCANCODE_RIGHT)])
        var cur_space = Bool(keys[Int(Scancode.SCANCODE_SPACE)])
        var cur_return = Bool(keys[Int(Scancode.SCANCODE_RETURN)])
        var cur_r = Bool(keys[Int(Scancode.SCANCODE_R)])
        var cur_kp1 = Bool(keys[Int(Scancode.SCANCODE_KP_1)])
        var cur_kp2 = Bool(keys[Int(Scancode.SCANCODE_KP_2)])
        var cur_kp3 = Bool(keys[Int(Scancode.SCANCODE_KP_3)])
        var cur_kp4 = Bool(keys[Int(Scancode.SCANCODE_KP_4)])
        var cur_kp5 = Bool(keys[Int(Scancode.SCANCODE_KP_5)])
        var cur_kp6 = Bool(keys[Int(Scancode.SCANCODE_KP_6)])
        var cur_kp7 = Bool(keys[Int(Scancode.SCANCODE_KP_7)])
        var cur_kp8 = Bool(keys[Int(Scancode.SCANCODE_KP_8)])
        var cur_kp9 = Bool(keys[Int(Scancode.SCANCODE_KP_9)])

        # Mouse state
        var mouse_buttons = get_mouse_state(
            rebind[UnsafePointer[Float32, MutAnyOrigin]](mouse_x_ptr),
            rebind[UnsafePointer[Float32, MutAnyOrigin]](mouse_y_ptr),
        )
        var cur_mouse_left = (Int(mouse_buttons.value) & 1) != 0
        var mouse_x = Int(mouse_x_ptr[])
        var mouse_y = Int(mouse_y_ptr[])

        # Convert mouse position to board cell
        var mouse_col = mouse_x // cell_size
        var mouse_row = mouse_y // cell_size
        var mouse_on_board = (
            mouse_col >= 0
            and mouse_col < 3
            and mouse_row >= 0
            and mouse_row < 3
            and mouse_y < board_size
        )

        # Update cursor from mouse hover
        if mouse_on_board:
            cursor_row = mouse_row
            cursor_col = mouse_col

        var game_over = env.done
        var action = -1  # -1 = no action

        # Handle reset
        if cur_r and not prev_r and game_over:
            _ = env.reset()
            cursor_row = 1
            cursor_col = 1

        if not game_over:
            # Arrow key cursor movement (debounced)
            if cur_up and not prev_up:
                if cursor_row > 0:
                    cursor_row -= 1
            if cur_down and not prev_down:
                if cursor_row < 2:
                    cursor_row += 1
            if cur_left and not prev_left:
                if cursor_col > 0:
                    cursor_col -= 1
            if cur_right and not prev_right:
                if cursor_col < 2:
                    cursor_col += 1

            # Space/Return to place mark at cursor
            if (cur_space and not prev_space) or (
                cur_return and not prev_return
            ):
                # Convert visual (row, col) to env cell index (row-major, row 0 = top)
                action = cursor_row * 3 + cursor_col

            # Mouse click to place mark
            if cur_mouse_left and not prev_mouse_left and mouse_on_board:
                action = mouse_row * 3 + mouse_col

            # Numpad direct placement (numpad layout: 1=bottom-left, 9=top-right)
            # Numpad 7=top-left, 8=top-center, 9=top-right
            # Numpad 4=mid-left,  5=mid-center, 6=mid-right
            # Numpad 1=bot-left,  2=bot-center, 3=bot-right
            # Map to row-major index where row 0 = top row
            if cur_kp7 and not prev_kp7:
                action = 0  # row 0, col 0
            if cur_kp8 and not prev_kp8:
                action = 1  # row 0, col 1
            if cur_kp9 and not prev_kp9:
                action = 2  # row 0, col 2
            if cur_kp4 and not prev_kp4:
                action = 3  # row 1, col 0
            if cur_kp5 and not prev_kp5:
                action = 4  # row 1, col 1
            if cur_kp6 and not prev_kp6:
                action = 5  # row 1, col 2
            if cur_kp1 and not prev_kp1:
                action = 6  # row 2, col 0
            if cur_kp2 and not prev_kp2:
                action = 7  # row 2, col 1
            if cur_kp3 and not prev_kp3:
                action = 8  # row 2, col 2

        # Execute action if any (use _step_impl for self-play, no random opponent)
        if action >= 0 and not game_over:
            _ = env._step_impl(action)

        # === Rendering ===

        # Draw grid lines (2 horizontal + 2 vertical)
        for i in range(1, 3):
            # Horizontal lines
            renderer.draw_line(
                0, i * cell_size, board_size, i * cell_size, grid_color, 3
            )
            # Vertical lines
            renderer.draw_line(
                i * cell_size, 0, i * cell_size, board_size, grid_color, 3
            )

        # Draw marks and cursor highlight
        for row in range(3):
            for col in range(3):
                var cx = col * cell_size + cell_size // 2
                var cy = row * cell_size + cell_size // 2
                var cell_idx = row * 3 + col
                var cell_val = Int(env.state[cell_idx])

                # Draw cursor highlight (yellow border)
                if not game_over and row == cursor_row and col == cursor_col:
                    renderer.draw_rect(
                        col * cell_size + 4,
                        row * cell_size + 4,
                        cell_size - 8,
                        cell_size - 8,
                        cursor_color,
                        border_width=3,
                    )

                if cell_val == 1:
                    # Draw X: two crossing lines
                    var margin = 25
                    var x0 = col * cell_size + margin
                    var y0 = row * cell_size + margin
                    var x1 = (col + 1) * cell_size - margin
                    var y1 = (row + 1) * cell_size - margin
                    renderer.draw_line(x0, y0, x1, y1, x_color, 4)
                    renderer.draw_line(x1, y0, x0, y1, x_color, 4)
                elif cell_val == 2:
                    # Draw O: circle
                    renderer.draw_circle(
                        cx, cy, cell_size // 2 - 20, o_color, filled=False
                    )

        # Status bar at bottom (y=400..450)
        renderer.draw_rect(
            0, board_size, 400, 50, SDL_Color(r=0x11, g=0x33, b=0x11, a=0xFF)
        )

        var game_result = env.game_result()
        if game_result == 0:
            # Ongoing
            var player = env.current_player()
            if player == 0:
                renderer.draw_text(
                    "Player X's turn", 140, board_size + 20, text_color
                )
            else:
                renderer.draw_text(
                    "Player O's turn", 140, board_size + 20, text_color
                )
        elif game_result == 1:
            renderer.draw_text(
                "X Wins!  (R to reset)", 120, board_size + 20, win_color
            )
        elif game_result == 2:
            renderer.draw_text(
                "O Wins!  (R to reset)", 120, board_size + 20, win_color
            )
        else:
            renderer.draw_text(
                "Draw!  (R to reset)", 130, board_size + 20, win_color
            )

        renderer.flip()

        # Save previous key states for debouncing
        prev_up = cur_up
        prev_down = cur_down
        prev_left = cur_left
        prev_right = cur_right
        prev_space = cur_space
        prev_return = cur_return
        prev_r = cur_r
        prev_kp1 = cur_kp1
        prev_kp2 = cur_kp2
        prev_kp3 = cur_kp3
        prev_kp4 = cur_kp4
        prev_kp5 = cur_kp5
        prev_kp6 = cur_kp6
        prev_kp7 = cur_kp7
        prev_kp8 = cur_kp8
        prev_kp9 = cur_kp9
        prev_mouse_left = cur_mouse_left

    mouse_x_ptr.free()
    mouse_y_ptr.free()
    numkeys_ptr.free()
    renderer.close()
    print("=== Done ===")
