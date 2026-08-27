"""Playable Chess with GIF recording.

Controls:
  Mouse click: select piece / confirm move
  Right click: deselect
  Arrow keys + Space: alternative keyboard control
  Escape: deselect
  R: reset after game ends
  Close window to stop recording & quit

Run with:
    pixi run mojo run -I . examples/board_games/chess_playable_gif.mojo
"""

from std.memory import alloc
from std.ffi import c_int, c_float
from mojo_rl.envs.board_games.chess.chess import (
    ChessEnv,
    _encode_action,
    _decode_action,
    Move,
    EMPTY,
    W_PAWN,
    W_KNIGHT,
    W_BISHOP,
    W_ROOK,
    W_QUEEN,
    W_KING,
    B_PAWN,
    B_KNIGHT,
    B_BISHOP,
    B_ROOK,
    B_QUEEN,
    B_KING,
    S_PLAYER,
    S_RESULT,
    S_WK,
    S_BK,
    RESULT_ONGOING,
    RESULT_WHITE_WINS,
    RESULT_BLACK_WINS,
    RESULT_DRAW,
    _is_friendly,
    _is_enemy,
    _piece_type,
    _row,
    _col,
)
from mojo_rl.envs.board_games.chess.chess_sprites import (
    create_sprite_sheet,
    PIECE_SIZE as SPRITE_SIZE,
    SHEET_WIDTH as SPRITE_SHEET_WIDTH,
    SHEET_HEIGHT as SPRITE_SHEET_HEIGHT,
    BYTES_PER_PIXEL as SPRITE_BPP,
)
from mojo_rl.render import Renderer2D, SDL_Color
from mojo_rl.render.sdl import (
    create_surface_from,
    create_texture_from_surface,
    render_texture,
    destroy_surface,
    set_texture_blend_mode,
    set_texture_scale_mode,
    destroy_texture,
    Surface,
    Texture,
    FRect,
    PixelFormat,
    BlendMode,
    ScaleMode,
)
from mojo_rl.render.sdl.sdl_keyboard import get_keyboard_state
from mojo_rl.render.sdl.sdl_scancode import Scancode
from mojo_rl.render.sdl.sdl_mouse import get_mouse_state, MouseButtonFlags


def _file_letter(c: Int) -> String:
    if c == 0:
        return "a"
    if c == 1:
        return "b"
    if c == 2:
        return "c"
    if c == 3:
        return "d"
    if c == 4:
        return "e"
    if c == 5:
        return "f"
    if c == 6:
        return "g"
    return "h"


def sq_name(sq: Int) -> String:
    return _file_letter(_col(sq)) + String(_row(sq) + 1)


def piece_name(piece: Int) -> String:
    if piece == EMPTY:
        return "empty"
    var color = String("White") if piece <= 6 else String("Black")
    var pt = _piece_type(piece)
    if pt == 1:
        return color + " Pawn"
    if pt == 2:
        return color + " Knight"
    if pt == 3:
        return color + " Bishop"
    if pt == 4:
        return color + " Rook"
    if pt == 5:
        return color + " Queen"
    if pt == 6:
        return color + " King"
    return color + " ???"


def piece_to_sprite_idx(piece: Int) -> Int:
    if piece == W_KING:
        return 0
    if piece == W_QUEEN:
        return 1
    if piece == W_ROOK:
        return 2
    if piece == W_BISHOP:
        return 3
    if piece == W_KNIGHT:
        return 4
    if piece == W_PAWN:
        return 5
    if piece == B_KING:
        return 6
    if piece == B_QUEEN:
        return 7
    if piece == B_ROOK:
        return 8
    if piece == B_BISHOP:
        return 9
    if piece == B_KNIGHT:
        return 10
    if piece == B_PAWN:
        return 11
    return -1


def piece_char(piece: Int) -> String:
    if piece == W_KING:
        return "K"
    if piece == W_QUEEN:
        return "Q"
    if piece == W_ROOK:
        return "R"
    if piece == W_BISHOP:
        return "B"
    if piece == W_KNIGHT:
        return "N"
    if piece == W_PAWN:
        return "P"
    if piece == B_KING:
        return "k"
    if piece == B_QUEEN:
        return "q"
    if piece == B_ROOK:
        return "r"
    if piece == B_BISHOP:
        return "b"
    if piece == B_KNIGHT:
        return "n"
    if piece == B_PAWN:
        return "p"
    return ""


def main() raises:
    print("=== Playable Chess — Recording to GIF ===")
    print("Controls:")
    print("  Mouse click: select piece / confirm move")
    print("  Right click / Escape: deselect")
    print("  R: reset after game ends")
    print("  Close window to stop & save")

    var env = ChessEnv[DType.float64]()
    _ = env.reset()

    var left_margin = 20
    var top_margin = 4
    var sq_size = 64
    var board_px = sq_size * 8
    var win_w = left_margin + board_px + 4
    var win_h = top_margin + board_px + 20 + 50

    var renderer = Renderer2D(width=win_w, height=win_h, fps=30, title="Chess")
    renderer.start_recording("gifs/chess_playable.gif", fps=15, skip=2)

    var sprite_pixels = create_sprite_sheet()
    var sprite_texture = Pointer[Texture, MutAnyOrigin](unsafe_from_address=Int(0))
    var has_sprites = False
    var sprite_draw_size = 48
    var sprite_pad = (sq_size - sprite_draw_size) // 2

    var light_sq = SDL_Color(r=0xF0, g=0xD9, b=0xB5, a=0xFF)
    var dark_sq = SDL_Color(r=0xB5, g=0x88, b=0x63, a=0xFF)
    var cursor_color = SDL_Color(r=0xFF, g=0xFF, b=0x00, a=0xFF)
    var select_color = SDL_Color(r=0x00, g=0xFF, b=0x00, a=0xFF)
    var white_piece_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var black_piece_color = SDL_Color(r=0x20, g=0x20, b=0x20, a=0xFF)
    var text_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var status_bg = SDL_Color(r=0x33, g=0x33, b=0x33, a=0xFF)
    var bg_color = SDL_Color(r=0x22, g=0x22, b=0x22, a=0xFF)
    var legal_highlight = SDL_Color(r=0x00, g=0xAA, b=0x00, a=0x60)

    var cursor_row = 6
    var cursor_col = 4

    var selected_sq = -1

    var prev_up = False
    var prev_down = False
    var prev_left = False
    var prev_right = False
    var prev_space = False
    var prev_esc = False
    var prev_r = False

    var prev_mouse_left = False
    var prev_mouse_right = False
    var mouse_x_ptr = alloc[Float32](1)
    var mouse_y_ptr = alloc[Float32](1)
    mouse_x_ptr[] = Float32(0)
    mouse_y_ptr[] = Float32(0)

    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    while renderer.begin_frame_with_color(bg_color):
        if not has_sprites:
            try:
                var surface = create_surface_from(
                    c_int(SPRITE_SHEET_WIDTH),
                    c_int(SPRITE_SHEET_HEIGHT),
                    PixelFormat.PIXELFORMAT_RGBA32,
                    rebind[Pointer[NoneType, MutAnyOrigin]](
                        sprite_pixels
                    ),
                    c_int(SPRITE_SHEET_WIDTH * SPRITE_BPP),
                )
                sprite_texture = create_texture_from_surface(
                    renderer.sdl_renderer.value(), surface
                )
                set_texture_blend_mode(
                    sprite_texture, BlendMode.BLENDMODE_BLEND
                )
                try:
                    set_texture_scale_mode(
                        sprite_texture, ScaleMode.SCALEMODE_NEAREST
                    )
                except:
                    pass
                destroy_surface(surface)
                has_sprites = True
            except:
                pass

        var keys = get_keyboard_state(numkeys_ptr)

        var cur_up = Bool(keys[Int(Scancode.SCANCODE_UP)])
        var cur_down = Bool(keys[Int(Scancode.SCANCODE_DOWN)])
        var cur_left = Bool(keys[Int(Scancode.SCANCODE_LEFT)])
        var cur_right = Bool(keys[Int(Scancode.SCANCODE_RIGHT)])
        var cur_space = Bool(keys[Int(Scancode.SCANCODE_SPACE)])
        var cur_esc = Bool(keys[Int(Scancode.SCANCODE_ESCAPE)])
        var cur_r = Bool(keys[Int(Scancode.SCANCODE_R)])

        var mouse_buttons = get_mouse_state(
            rebind[Pointer[Float32, MutAnyOrigin]](mouse_x_ptr),
            rebind[Pointer[Float32, MutAnyOrigin]](mouse_y_ptr),
        )
        var cur_mouse_left = (Int(mouse_buttons.value) & 1) != 0
        var cur_mouse_right = (Int(mouse_buttons.value) & 4) != 0
        var mouse_x = Int(mouse_x_ptr[])
        var mouse_y = Int(mouse_y_ptr[])

        var mouse_col = (mouse_x - left_margin) // sq_size
        var mouse_row = (mouse_y - top_margin) // sq_size
        var mouse_on_board = (
            mouse_col >= 0
            and mouse_col < 8
            and mouse_row >= 0
            and mouse_row < 8
        )

        if mouse_on_board:
            cursor_row = mouse_row
            cursor_col = mouse_col

        var game_over = env.done
        var player = Int(env.state[S_PLAYER])

        if (cur_mouse_right and not prev_mouse_right) or (
            cur_esc and not prev_esc
        ):
            selected_sq = -1

        if cur_r and not prev_r and game_over:
            _ = env.reset()
            cursor_row = 6
            cursor_col = 4
            selected_sq = -1

        if not game_over:
            if cur_up and not prev_up:
                if cursor_row > 0:
                    cursor_row -= 1
            if cur_down and not prev_down:
                if cursor_row < 7:
                    cursor_row += 1
            if cur_left and not prev_left:
                if cursor_col > 0:
                    cursor_col -= 1
            if cur_right and not prev_right:
                if cursor_col < 7:
                    cursor_col += 1

            var click_triggered = (cur_space and not prev_space) or (
                cur_mouse_left and not prev_mouse_left and mouse_on_board
            )
            if click_triggered:
                var env_row = 7 - cursor_row
                var env_col = cursor_col
                var env_sq = env_row * 8 + env_col

                if selected_sq < 0:
                    var piece = Int(env.state[env_sq])
                    if piece != EMPTY and _is_friendly(piece, player):
                        selected_sq = env_sq
                else:
                    var from_sq = selected_sq
                    var to_sq = env_sq

                    if from_sq == to_sq:
                        selected_sq = -1
                    else:
                        var promo = 0
                        var from_piece = Int(env.state[from_sq])
                        var to_row = to_sq // 8
                        if from_piece == W_PAWN and to_row == 7:
                            promo = 5
                        elif from_piece == B_PAWN and to_row == 0:
                            promo = 5

                        var move = Move(from_sq, to_sq, promo)
                        var action = _encode_action(move, player)

                        var mask = env.legal_action_mask()
                        if action >= 0 and action < 4672 and mask[action]:
                            _ = env._step_impl(action)
                            selected_sq = -1
                        else:
                            var piece = Int(env.state[env_sq])
                            if piece != EMPTY and _is_friendly(piece, player):
                                selected_sq = env_sq
                            else:
                                selected_sq = -1

        # === Rendering ===

        var legal_mask = env.legal_action_mask()

        for row in range(8):
            for col in range(8):
                var px = left_margin + col * sq_size
                var py = top_margin + row * sq_size

                var is_light = ((row + col) % 2) == 0
                if is_light:
                    renderer.draw_rect(px, py, sq_size, sq_size, light_sq)
                else:
                    renderer.draw_rect(px, py, sq_size, sq_size, dark_sq)

                var env_row = 7 - row
                var env_col = col
                var env_sq = env_row * 8 + env_col

                if env_sq == selected_sq:
                    renderer.draw_rect(
                        px + 1,
                        py + 1,
                        sq_size - 2,
                        sq_size - 2,
                        select_color,
                        border_width=3,
                    )

                if selected_sq >= 0 and env_sq != selected_sq and not game_over:
                    var is_legal_target = False
                    for promo_val in range(6):
                        if promo_val == 1:
                            continue
                        var test_move = Move(selected_sq, env_sq, promo_val)
                        var test_action = _encode_action(test_move, player)
                        if (
                            test_action >= 0
                            and test_action < 4672
                            and legal_mask[test_action]
                        ):
                            is_legal_target = True
                            break
                    if is_legal_target:
                        renderer.draw_circle(
                            px + sq_size // 2,
                            py + sq_size // 2,
                            10,
                            legal_highlight,
                            filled=True,
                        )

                if row == cursor_row and col == cursor_col:
                    renderer.draw_rect(
                        px, py, sq_size, sq_size, cursor_color, border_width=3
                    )

                var piece = Int(env.state[env_sq])
                if piece != EMPTY:
                    var sprite_idx = piece_to_sprite_idx(piece)
                    if has_sprites and sprite_idx >= 0:
                        var src_rect = alloc[FRect](1)
                        src_rect[] = FRect(
                            c_float(sprite_idx * SPRITE_SIZE),
                            c_float(0),
                            c_float(SPRITE_SIZE),
                            c_float(SPRITE_SIZE),
                        )
                        var dst_rect = alloc[FRect](1)
                        dst_rect[] = FRect(
                            c_float(px + sprite_pad),
                            c_float(py + sprite_pad),
                            c_float(sprite_draw_size),
                            c_float(sprite_draw_size),
                        )
                        try:
                            render_texture(
                                renderer.sdl_renderer.value(),
                                sprite_texture,
                                rebind[Pointer[FRect, ImmutAnyOrigin]](
                                    src_rect
                                ),
                                rebind[Pointer[FRect, ImmutAnyOrigin]](
                                    dst_rect
                                ),
                            )
                        except:
                            pass
                        src_rect.free()
                        dst_rect.free()
                    else:
                        var pc = piece_char(piece)
                        var tx = px + sq_size // 2 - 4
                        var ty = py + sq_size // 2 - 4
                        if piece >= 1 and piece <= 6:
                            renderer.draw_text(
                                pc, tx + 1, ty + 1, black_piece_color
                            )
                            renderer.draw_text(pc, tx, ty, white_piece_color)
                        else:
                            renderer.draw_text(
                                pc, tx + 1, ty + 1, white_piece_color
                            )
                            renderer.draw_text(pc, tx, ty, black_piece_color)

        var label_color = SDL_Color(r=0xCC, g=0xCC, b=0xCC, a=0xFF)
        for col in range(8):
            var lx = left_margin + col * sq_size + sq_size // 2 - 4
            var ly = top_margin + 8 * sq_size + 4
            renderer.draw_text(_file_letter(col), lx, ly, label_color)

        for display_row in range(8):
            var rank_num = 8 - display_row
            var lx = left_margin - 12
            var ly = top_margin + display_row * sq_size + sq_size // 2 - 4
            renderer.draw_text(String(rank_num), lx, ly, label_color)

        var status_y = top_margin + board_px + 20
        renderer.draw_rect(0, status_y, win_w, 50, status_bg)

        var result = Int(env.state[S_RESULT])
        if result == RESULT_ONGOING:
            if player == 0:
                renderer.draw_text(
                    "White's turn", 200, status_y + 20, text_color
                )
            else:
                renderer.draw_text(
                    "Black's turn", 200, status_y + 20, text_color
                )
        elif result == RESULT_WHITE_WINS:
            renderer.draw_text(
                "Checkmate! White wins  (R to reset)",
                120,
                status_y + 20,
                text_color,
            )
        elif result == RESULT_BLACK_WINS:
            renderer.draw_text(
                "Checkmate! Black wins  (R to reset)",
                120,
                status_y + 20,
                text_color,
            )
        elif result == RESULT_DRAW:
            renderer.draw_text(
                "Draw!  (R to reset)", 180, status_y + 20, text_color
            )

        renderer.flip()

        prev_up = cur_up
        prev_down = cur_down
        prev_left = cur_left
        prev_right = cur_right
        prev_space = cur_space
        prev_esc = cur_esc
        prev_r = cur_r
        prev_mouse_left = cur_mouse_left
        prev_mouse_right = cur_mouse_right

    renderer.stop_recording()
    mouse_x_ptr.free()
    mouse_y_ptr.free()
    if has_sprites:
        try:
            destroy_texture(sprite_texture)
        except:
            pass
    sprite_pixels.free()
    numkeys_ptr.free()
    renderer.close()
    print("Saved: gifs/chess_playable.gif")
    print("=== Done ===")
