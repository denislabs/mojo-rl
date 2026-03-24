"""Play TicTacToe against a trained AlphaZero agent.

Load a checkpoint from self-play training and play interactively.
You play as X (first), the AI plays as O using raw policy network.

Controls:
  Mouse click or numpad 1-9 to place mark
  R to reset after game ends
  Close window to quit

Usage:
    # First train the agent:
    pixi run -e nvidia mojo run -I . tests/deep_agents/test_alphazero_final.mojo

    # Then play against it:
    pixi run -e apple mojo run -I . examples/board_games/tictactoe_play_vs_alphazero.mojo
"""

from std.memory import alloc
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroTicTacToeConfig,
    AlphaZeroTicTacToeCNNConfig,
)
from mojo_rl.nn.constants import dtype
from mojo_rl.render import Renderer2D, SDL_Color
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode
from mojo_rl.render.sdl.sdl_mouse import get_mouse_state, MouseButtonFlags


def main() raises:
    print("=== Play vs AlphaZero on TicTacToe ===")
    print("You are X (first player). AI is O.")
    print()

    # ── Load trained agent ───────────────────────────────────────
    # Must match the config used during training!
    comptime Config = AlphaZeroTicTacToeCNNConfig[]
    # comptime Config = AlphaZeroTicTacToeConfig[]

    var agent = GenericAlphaZeroAgent[Config, 64]()

    var ckpt_path = "tictactoe_alphazero.ckpt"
    print("Loading checkpoint:", ckpt_path)
    agent.load_checkpoint(ckpt_path)
    print("Loaded! Train steps:", agent.train_step_count)
    print()
    print("Controls: click cell or numpad 1-9, R to reset")
    print()

    # ── Setup environment + renderer ─────────────────────────────
    var env = TicTacToeEnv[DType.float64]()
    _ = env.reset()

    var renderer = Renderer2D(
        width=400, height=450, fps=30, title="TicTacToe vs AlphaZero"
    )

    var bg_color = SDL_Color(r=0x1A, g=0x5C, b=0x2A, a=0xFF)
    var grid_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var x_color = SDL_Color(r=0xFF, g=0x44, b=0x44, a=0xFF)
    var o_color = SDL_Color(r=0x44, g=0x88, b=0xFF, a=0xFF)
    var cursor_color = SDL_Color(r=0xFF, g=0xFF, b=0x00, a=0xFF)
    var text_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var win_color = SDL_Color(r=0xFF, g=0xDD, b=0x00, a=0xFF)
    var ai_color = SDL_Color(r=0x88, g=0xFF, b=0x88, a=0xFF)

    var cursor_row = 1
    var cursor_col = 1
    var prev_space = False
    var prev_return = False
    var prev_r = False
    var prev_mouse_left = False
    var prev_kp = InlineArray[Bool, 9](fill=False)

    var mouse_x_ptr = alloc[Float32](1)
    var mouse_y_ptr = alloc[Float32](1)
    mouse_x_ptr[] = Float32(0)
    mouse_y_ptr[] = Float32(0)
    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    var cell_size = 133
    var board_size = 400

    while renderer.begin_frame_with_color(bg_color):
        var keys = get_keyboard_state(numkeys_ptr)
        var cur_space = Bool(keys[Int(Scancode.SCANCODE_SPACE)])
        var cur_return = Bool(keys[Int(Scancode.SCANCODE_RETURN)])
        var cur_r = Bool(keys[Int(Scancode.SCANCODE_R)])

        # Mouse
        var mouse_buttons = get_mouse_state(
            rebind[UnsafePointer[Float32, MutAnyOrigin]](mouse_x_ptr),
            rebind[UnsafePointer[Float32, MutAnyOrigin]](mouse_y_ptr),
        )
        var cur_mouse_left = (Int(mouse_buttons.value) & 1) != 0
        var mouse_x = Int(mouse_x_ptr[])
        var mouse_y = Int(mouse_y_ptr[])
        var mouse_col = mouse_x // cell_size
        var mouse_row = mouse_y // cell_size
        var mouse_on_board = (
            mouse_col >= 0
            and mouse_col < 3
            and mouse_row >= 0
            and mouse_row < 3
            and mouse_y < board_size
        )
        if mouse_on_board:
            cursor_row = mouse_row
            cursor_col = mouse_col

        var game_over = env.done
        var action = -1

        # Reset
        if cur_r and not prev_r and game_over:
            _ = env.reset()
            cursor_row = 1
            cursor_col = 1

        # Human turn (player 0 = X)
        if not game_over and env.current_player() == 0:
            if (cur_space and not prev_space) or (
                cur_return and not prev_return
            ):
                action = cursor_row * 3 + cursor_col
            if cur_mouse_left and not prev_mouse_left and mouse_on_board:
                action = mouse_row * 3 + mouse_col

            # Numpad
            var kp_scancodes = InlineArray[Int, 9](uninitialized=True)
            kp_scancodes[0] = Int(Scancode.SCANCODE_KP_7)
            kp_scancodes[1] = Int(Scancode.SCANCODE_KP_8)
            kp_scancodes[2] = Int(Scancode.SCANCODE_KP_9)
            kp_scancodes[3] = Int(Scancode.SCANCODE_KP_4)
            kp_scancodes[4] = Int(Scancode.SCANCODE_KP_5)
            kp_scancodes[5] = Int(Scancode.SCANCODE_KP_6)
            kp_scancodes[6] = Int(Scancode.SCANCODE_KP_1)
            kp_scancodes[7] = Int(Scancode.SCANCODE_KP_2)
            kp_scancodes[8] = Int(Scancode.SCANCODE_KP_3)
            var kp_to_cell = InlineArray[Int, 9](uninitialized=True)
            kp_to_cell[0] = 0
            kp_to_cell[1] = 1
            kp_to_cell[2] = 2
            kp_to_cell[3] = 3
            kp_to_cell[4] = 4
            kp_to_cell[5] = 5
            kp_to_cell[6] = 6
            kp_to_cell[7] = 7
            kp_to_cell[8] = 8
            for k in range(9):
                var cur_kp = Bool(keys[kp_scancodes[k]])
                if cur_kp and not prev_kp[k]:
                    action = kp_to_cell[k]
                prev_kp[k] = cur_kp

        # Execute human action
        if action >= 0 and not game_over and env.current_player() == 0:
            _ = env._step_impl(action)

        # AI turn (player 1 = O) — CPU MCTS with true game rules
        if not game_over and env.current_player() == 1:
            comptime OBS = Config.obs_dim
            comptime TTTCPU = TicTacToeEnv[DType.float64]
            var obs = List[Scalar[dtype]](capacity=OBS)
            var obs_raw = env.get_obs_list()
            for i in range(OBS):
                if i < len(obs_raw):
                    obs.append(Scalar[dtype](obs_raw[i]))
                else:
                    obs.append(Scalar[dtype](0.0))

            var legal = env.legal_action_mask()
            var ai_action = agent.select_action_mcts[TTTCPU](
                obs, legal, env
            )

            if ai_action >= 0 and ai_action < len(legal) and legal[ai_action]:
                _ = env._step_impl(ai_action)
            else:
                for a in range(len(legal)):
                    if legal[a]:
                        _ = env._step_impl(a)
                        break

        # === Rendering ===
        for i in range(1, 3):
            renderer.draw_line(
                0, i * cell_size, board_size, i * cell_size, grid_color, 3
            )
            renderer.draw_line(
                i * cell_size, 0, i * cell_size, board_size, grid_color, 3
            )

        for row in range(3):
            for col in range(3):
                var cx = col * cell_size + cell_size // 2
                var cy = row * cell_size + cell_size // 2
                var cell_idx = row * 3 + col
                var cell_val = Int(env.state[cell_idx])

                if (
                    not game_over
                    and row == cursor_row
                    and col == cursor_col
                    and env.current_player() == 0
                ):
                    renderer.draw_rect(
                        col * cell_size + 4,
                        row * cell_size + 4,
                        cell_size - 8,
                        cell_size - 8,
                        cursor_color,
                        border_width=3,
                    )

                if cell_val == 1:
                    var margin = 25
                    renderer.draw_line(
                        col * cell_size + margin,
                        row * cell_size + margin,
                        (col + 1) * cell_size - margin,
                        (row + 1) * cell_size - margin,
                        x_color,
                        4,
                    )
                    renderer.draw_line(
                        (col + 1) * cell_size - margin,
                        row * cell_size + margin,
                        col * cell_size + margin,
                        (row + 1) * cell_size - margin,
                        x_color,
                        4,
                    )
                elif cell_val == 2:
                    renderer.draw_circle(
                        cx, cy, cell_size // 2 - 20, o_color, filled=False
                    )

        # Status bar
        renderer.draw_rect(
            0, board_size, 400, 50, SDL_Color(r=0x11, g=0x33, b=0x11, a=0xFF)
        )

        var game_result = env.game_result()
        if game_result == 0:
            if env.current_player() == 0:
                renderer.draw_text(
                    "Your turn (X)", 150, board_size + 20, text_color
                )
            else:
                renderer.draw_text(
                    "AI thinking...", 150, board_size + 20, ai_color
                )
        elif game_result == 1:
            renderer.draw_text(
                "You win!  (R to reset)", 115, board_size + 20, win_color
            )
        elif game_result == 2:
            renderer.draw_text(
                "AI wins!  (R to reset)", 115, board_size + 20, win_color
            )
        else:
            renderer.draw_text(
                "Draw!  (R to reset)", 130, board_size + 20, win_color
            )

        renderer.flip()

        prev_space = cur_space
        prev_return = cur_return
        prev_r = cur_r
        prev_mouse_left = cur_mouse_left

    mouse_x_ptr.free()
    mouse_y_ptr.free()
    numkeys_ptr.free()
    renderer.close()
    print("=== Done ===")
