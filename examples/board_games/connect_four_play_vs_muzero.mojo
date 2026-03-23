"""Play Connect Four against a trained MuZero/AlphaZero agent.

You play as Red (first), the AI plays as Yellow using MCTS.

Controls:
  Mouse hover/click: select and drop piece in column
  Left/Right arrows: move column selector
  Space/Return: drop piece
  R: reset after game ends
  Close window to quit

Usage:
    # First train:
    pixi run -e apple mojo run -I . examples/board_games/connect_four_muzero_selfplay.mojo
    # Then play:
    pixi run -e apple mojo run -I . examples/board_games/connect_four_play_vs_muzero.mojo
"""

from std.memory import alloc
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv
from mojo_rl.deep_agents.muzero import GenericMuZeroAgent
from mojo_rl.deep_agents.muzero.configs import AlphaZeroConfig
from mojo_rl.nn.constants import dtype
from mojo_rl.render import Renderer2D, SDL_Color
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode
from mojo_rl.render.sdl.sdl_mouse import get_mouse_state, MouseButtonFlags


def main() raises:
    print("=== Play vs MuZero on Connect Four ===")
    print("You are Red (first player). AI is Yellow.")
    print()

    # ── Load trained agent ───────────────────────────────────────
    comptime C4 = ConnectFourEnv[DType.float64]
    comptime Config = AlphaZeroConfig[
        C4.OBS_DIM,
        C4.NUM_ACTIONS,
        HIDDEN=256,
        LR=5e-4,
        BS=128,
        SIMS=50,
        NODES=128,
    ]

    var agent = GenericMuZeroAgent[Config, 1](
        gamma=1.0,
        v_min=-1.0,
        v_max=1.0,
    )

    var ckpt_path = "connect_four_muzero.ckpt"
    print("Loading checkpoint:", ckpt_path)
    agent.load_checkpoint(ckpt_path)
    print("Loaded! Train steps:", agent.train_step_count)
    print()

    # ── Setup ────────────────────────────────────────────────────
    var env = ConnectFourEnv[DType.float64]()
    _ = env.reset()

    var renderer = Renderer2D(
        width=560, height=530, fps=30, title="Connect Four vs MuZero"
    )

    var board_color = SDL_Color(r=0x00, g=0x00, b=0xAA, a=0xFF)
    var empty_color = SDL_Color(r=0x33, g=0x33, b=0x33, a=0xFF)
    var red_color = SDL_Color(r=0xFF, g=0x22, b=0x22, a=0xFF)
    var yellow_color = SDL_Color(r=0xFF, g=0xDD, b=0x00, a=0xFF)
    var bg_color = SDL_Color(r=0x11, g=0x11, b=0x44, a=0xFF)
    var text_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var win_text_color = SDL_Color(r=0xFF, g=0xDD, b=0x00, a=0xFF)
    var selector_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var status_bg = SDL_Color(r=0x11, g=0x11, b=0x22, a=0xFF)
    var ai_text = SDL_Color(r=0x88, g=0xFF, b=0x88, a=0xFF)

    var cell_size = 80
    var board_cols = 7
    var board_rows = 6
    var board_width = board_cols * cell_size
    var board_height = board_rows * cell_size
    var circle_radius = 32

    var selected_col = 3
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

        var mouse_buttons = get_mouse_state(
            rebind[UnsafePointer[Float32, MutAnyOrigin]](mouse_x_ptr),
            rebind[UnsafePointer[Float32, MutAnyOrigin]](mouse_y_ptr),
        )
        var cur_mouse_left = (Int(mouse_buttons.value) & 1) != 0
        var mouse_x = Int(mouse_x_ptr[])
        var mouse_col = mouse_x // cell_size
        var mouse_on_board = mouse_col >= 0 and mouse_col < board_cols

        if mouse_on_board:
            selected_col = mouse_col

        var game_over = env.done

        # Reset
        if cur_r and not prev_r and game_over:
            _ = env.reset()
            selected_col = 3

        # Human turn (player 0 = Red)
        if not game_over and env.current_player() == 0:
            if cur_left and not prev_left and selected_col > 0:
                selected_col -= 1
            if cur_right and not prev_right and selected_col < board_cols - 1:
                selected_col += 1

            var action = -1
            if (cur_space and not prev_space) or (
                cur_return and not prev_return
            ):
                action = selected_col
            if cur_mouse_left and not prev_mouse_left and mouse_on_board:
                action = mouse_col

            if action >= 0:
                # Check if column is legal
                var legal = env.legal_action_mask()
                if action < len(legal) and legal[action]:
                    _ = env._step_impl(action)

        # AI turn (player 1 = Yellow) — MCTS
        if not game_over and env.current_player() == 1:
            var obs = List[Scalar[dtype]](capacity=Config.obs_dim)
            var obs_raw = env.get_obs_list()
            for i in range(Config.obs_dim):
                if i < len(obs_raw):
                    obs.append(Scalar[dtype](obs_raw[i]))
                else:
                    obs.append(Scalar[dtype](0.0))

            var legal = env.legal_action_mask()
            var ai_action = agent.select_action_policy_only(obs, legal)

            # Verify legal, fallback if needed
            if ai_action >= 0 and ai_action < len(legal) and legal[ai_action]:
                _ = env._step_impl(ai_action)
            else:
                for a in range(len(legal)):
                    if legal[a]:
                        _ = env._step_impl(a)
                        break

        # === Rendering ===

        # Selector at top
        var selector_cx = selected_col * cell_size + cell_size // 2
        if not game_over and env.current_player() == 0:
            renderer.draw_circle(selector_cx, 22, 18, red_color, filled=True)
            renderer.draw_line(
                selector_cx, 42, selector_cx - 8, 35, selector_color, 2
            )
            renderer.draw_line(
                selector_cx, 42, selector_cx + 8, 35, selector_color, 2
            )

        # Board
        renderer.draw_rect(0, 50, board_width, board_height, board_color)

        for col in range(board_cols):
            for row in range(board_rows):
                var visual_row = board_rows - 1 - row
                var cx = col * cell_size + cell_size // 2
                var cy = 50 + visual_row * cell_size + cell_size // 2
                var cell_idx = col * board_rows + row
                var cell_val = Int(env.state[cell_idx])

                if cell_val == 1:
                    renderer.draw_circle(
                        cx, cy, circle_radius, red_color, filled=True
                    )
                elif cell_val == 2:
                    renderer.draw_circle(
                        cx, cy, circle_radius, yellow_color, filled=True
                    )
                else:
                    renderer.draw_circle(
                        cx, cy, circle_radius, empty_color, filled=True
                    )

        # Grid lines
        for i in range(1, board_cols):
            renderer.draw_line(
                i * cell_size,
                50,
                i * cell_size,
                50 + board_height,
                board_color,
                1,
            )
        for i in range(1, board_rows):
            renderer.draw_line(
                0,
                50 + i * cell_size,
                board_width,
                50 + i * cell_size,
                board_color,
                1,
            )

        # Status bar
        renderer.draw_rect(0, 50 + board_height, board_width, 50, status_bg)

        var game_result = env.game_result()
        if game_result == 0:
            if env.current_player() == 0:
                renderer.draw_text(
                    "Your turn (Red)", 200, 50 + board_height + 20, text_color
                )
            else:
                renderer.draw_text(
                    "AI thinking...", 210, 50 + board_height + 20, ai_text
                )
        elif game_result == 1:
            renderer.draw_text(
                "You win!  (R to reset)",
                175,
                50 + board_height + 20,
                win_text_color,
            )
        elif game_result == 2:
            renderer.draw_text(
                "AI wins!  (R to reset)",
                175,
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
