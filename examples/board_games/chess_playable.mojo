"""Playable Chess -- two humans alternate moves on an 8x8 board.

Controls:
  Arrow keys: move cursor
  Space: select piece / confirm move
  Escape: deselect piece
  R: reset after game ends
  Close window to quit
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


fn _file_letter(c: Int) -> String:
    """Return file letter for column index 0-7."""
    if c == 0: return "a"
    if c == 1: return "b"
    if c == 2: return "c"
    if c == 3: return "d"
    if c == 4: return "e"
    if c == 5: return "f"
    if c == 6: return "g"
    return "h"


fn sq_name(sq: Int) -> String:
    """Convert a square index (0-63) to algebraic notation like 'e4'."""
    return _file_letter(_col(sq)) + String(_row(sq) + 1)


fn piece_name(piece: Int) -> String:
    """Return human-readable piece name like 'White Queen'."""
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


fn piece_to_sprite_idx(piece: Int) -> Int:
    """Map piece ID (1-12) to sprite sheet index (0-11)."""
    if piece == W_KING: return 0
    if piece == W_QUEEN: return 1
    if piece == W_ROOK: return 2
    if piece == W_BISHOP: return 3
    if piece == W_KNIGHT: return 4
    if piece == W_PAWN: return 5
    if piece == B_KING: return 6
    if piece == B_QUEEN: return 7
    if piece == B_ROOK: return 8
    if piece == B_BISHOP: return 9
    if piece == B_KNIGHT: return 10
    if piece == B_PAWN: return 11
    return -1


fn piece_char(piece: Int) -> String:
    """Return the character for a given piece ID (fallback)."""
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


fn main() raises:
    print("=== Playable Chess ===")
    print("Controls:")
    print("  Mouse click: select piece / confirm move")
    print("  Right click: deselect")
    print("  Arrow keys + Space: alternative keyboard control")
    print("  Escape: deselect")
    print("  R: reset after game ends")
    print("  Close window to quit")

    var env = ChessEnv[DType.float64]()
    _ = env.reset()

    # Window: 536x586 (20px left margin + 512 board + 4px right + 50px status + 20px bottom labels)
    var left_margin = 20
    var top_margin = 4
    var sq_size = 64
    var board_px = sq_size * 8  # 512
    var win_w = left_margin + board_px + 4  # 536
    var win_h = top_margin + board_px + 20 + 50  # 586

    var renderer = Renderer2D(
        width=win_w, height=win_h, fps=30, title="Chess"
    )

    # Sprite setup (lazy — created after renderer is initialized on first frame)
    var sprite_pixels = create_sprite_sheet()
    var sprite_texture = UnsafePointer[Texture, MutAnyOrigin]()
    var has_sprites = False
    var sprite_draw_size = 48  # Scale 24→48 to fit in 64px cells
    var sprite_pad = (sq_size - sprite_draw_size) // 2

    # Colors
    var light_sq = SDL_Color(r=0xF0, g=0xD9, b=0xB5, a=0xFF)
    var dark_sq = SDL_Color(r=0xB5, g=0x88, b=0x63, a=0xFF)
    var cursor_color = SDL_Color(r=0xFF, g=0xFF, b=0x00, a=0xFF)  # yellow
    var select_color = SDL_Color(r=0x00, g=0xFF, b=0x00, a=0xFF)  # green
    var white_piece_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var black_piece_color = SDL_Color(r=0x20, g=0x20, b=0x20, a=0xFF)
    var text_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var status_bg = SDL_Color(r=0x33, g=0x33, b=0x33, a=0xFF)
    var bg_color = SDL_Color(r=0x22, g=0x22, b=0x22, a=0xFF)
    var legal_highlight = SDL_Color(r=0x00, g=0xAA, b=0x00, a=0x60)

    # Cursor position (row, col) in display coords
    # Display row 0 = rank 8 (top), row 7 = rank 1 (bottom)
    var cursor_row = 6  # start near white pieces
    var cursor_col = 4

    # Selection state: -1 = no selection, else square index in env coords
    var selected_sq = -1

    # Debounce
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
        # Lazy sprite texture creation (renderer must be open first)
        if not has_sprites:
            try:
                var surface = create_surface_from(
                    c_int(SPRITE_SHEET_WIDTH),
                    c_int(SPRITE_SHEET_HEIGHT),
                    PixelFormat.PIXELFORMAT_RGBA32,
                    rebind[UnsafePointer[NoneType, MutAnyOrigin]](sprite_pixels),
                    c_int(SPRITE_SHEET_WIDTH * SPRITE_BPP),
                )
                sprite_texture = create_texture_from_surface(
                    renderer.sdl_renderer, surface
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
                pass  # will retry next frame or fall back to text

        var keys = get_keyboard_state(numkeys_ptr)

        var cur_up = Bool(keys[Int(Scancode.SCANCODE_UP)])
        var cur_down = Bool(keys[Int(Scancode.SCANCODE_DOWN)])
        var cur_left = Bool(keys[Int(Scancode.SCANCODE_LEFT)])
        var cur_right = Bool(keys[Int(Scancode.SCANCODE_RIGHT)])
        var cur_space = Bool(keys[Int(Scancode.SCANCODE_SPACE)])
        var cur_esc = Bool(keys[Int(Scancode.SCANCODE_ESCAPE)])
        var cur_r = Bool(keys[Int(Scancode.SCANCODE_R)])

        # Mouse state
        var mouse_buttons = get_mouse_state(
            rebind[UnsafePointer[Float32, MutAnyOrigin]](mouse_x_ptr),
            rebind[UnsafePointer[Float32, MutAnyOrigin]](mouse_y_ptr),
        )
        var cur_mouse_left = (Int(mouse_buttons.value) & 1) != 0  # SDL_BUTTON_LMASK
        var cur_mouse_right = (Int(mouse_buttons.value) & 4) != 0  # SDL_BUTTON_RMASK
        var mouse_x = Int(mouse_x_ptr[])
        var mouse_y = Int(mouse_y_ptr[])

        # Convert mouse position to board square
        var mouse_col = (mouse_x - left_margin) // sq_size
        var mouse_row = (mouse_y - top_margin) // sq_size
        var mouse_on_board = (
            mouse_col >= 0
            and mouse_col < 8
            and mouse_row >= 0
            and mouse_row < 8
        )

        # Update cursor from mouse position (hover)
        if mouse_on_board:
            cursor_row = mouse_row
            cursor_col = mouse_col

        var game_over = env.done
        var player = Int(env.state[S_PLAYER])

        # Right click or Escape to deselect
        if (cur_mouse_right and not prev_mouse_right) or (
            cur_esc and not prev_esc
        ):
            selected_sq = -1

        # Handle reset
        if cur_r and not prev_r and game_over:
            _ = env.reset()
            cursor_row = 6
            cursor_col = 4
            selected_sq = -1

        if not game_over:
            # Arrow key cursor movement
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

            # Space or left click: select or confirm
            var click_triggered = (cur_space and not prev_space) or (
                cur_mouse_left and not prev_mouse_left and mouse_on_board
            )
            if click_triggered:
                # Convert display (row, col) to env square
                # Display row 0 = rank 8 = env row 7
                var env_row = 7 - cursor_row
                var env_col = cursor_col
                var env_sq = env_row * 8 + env_col

                if selected_sq < 0:
                    # Phase 1: select a piece
                    var piece = Int(env.state[env_sq])
                    if piece != EMPTY and _is_friendly(piece, player):
                        selected_sq = env_sq
                else:
                    # Phase 2: confirm move
                    var from_sq = selected_sq
                    var to_sq = env_sq

                    if from_sq == to_sq:
                        # Clicking same square deselects
                        selected_sq = -1
                    else:
                        # Determine promotion
                        var promo = 0
                        var from_piece = Int(env.state[from_sq])
                        var to_row = to_sq // 8
                        # White pawn reaching rank 8 (row 7) or black pawn reaching rank 1 (row 0)
                        if from_piece == W_PAWN and to_row == 7:
                            promo = 5  # queen
                        elif from_piece == B_PAWN and to_row == 0:
                            promo = 5  # queen

                        # Encode action and check legality
                        var move = Move(from_sq, to_sq, promo)
                        var action = _encode_action(move, player)

                        # Verify action is legal
                        var mask = env.legal_action_mask()
                        if action >= 0 and action < 4672 and mask[action]:
                            _ = env._step_impl(action)
                            selected_sq = -1
                        else:
                            # --- Diagnostic output for rejected move ---
                            var from_p = Int(env.state[from_sq])
                            var to_p = Int(env.state[to_sq])
                            print(
                                "Move rejected:",
                                sq_name(from_sq),
                                "->",
                                sq_name(to_sq),
                            )
                            print(
                                "  From:",
                                piece_name(from_p),
                                "at",
                                sq_name(from_sq),
                            )
                            var to_desc = String("")
                            if to_p == EMPTY:
                                to_desc = " (empty)"
                            elif _is_enemy(to_p, player):
                                to_desc = " (enemy - capturable)"
                            else:
                                to_desc = " (friendly - blocked)"
                            print(
                                "  To:",
                                piece_name(to_p),
                                "at",
                                sq_name(to_sq),
                                to_desc,
                            )

                            # Print path for sliding pieces (bishop, rook, queen)
                            var pt = _piece_type(from_p)
                            if pt == 3 or pt == 4 or pt == 5:
                                var dr = 0
                                var dc = 0
                                var fr = _row(from_sq)
                                var fc = _col(from_sq)
                                var tr = _row(to_sq)
                                var tc = _col(to_sq)
                                var diff_r = tr - fr
                                var diff_c = tc - fc
                                if diff_r != 0:
                                    dr = 1 if diff_r > 0 else -1
                                if diff_c != 0:
                                    dc = 1 if diff_c > 0 else -1
                                var cr = fr + dr
                                var cc = fc + dc
                                while cr != tr or cc != tc:
                                    if cr < 0 or cr > 7 or cc < 0 or cc > 7:
                                        break
                                    var mid_sq = cr * 8 + cc
                                    var mid_p = Int(env.state[mid_sq])
                                    if mid_p == EMPTY:
                                        print(
                                            "  Path:",
                                            sq_name(mid_sq),
                                            "= empty",
                                        )
                                    else:
                                        print(
                                            "  Path:",
                                            sq_name(mid_sq),
                                            "=",
                                            piece_name(mid_p),
                                        )
                                    cr += dr
                                    cc += dc

                            # Check if king would be in check after move
                            var saved_from = env.state[from_sq]
                            var saved_to = env.state[to_sq]
                            var saved_wk = env.state[S_WK]
                            var saved_bk = env.state[S_BK]

                            env.state[to_sq] = env.state[from_sq]
                            env.state[from_sq] = 0.0
                            if _piece_type(Int(saved_from)) == 6:
                                if player == 0:
                                    env.state[S_WK] = Scalar[
                                        env.dtype
                                    ](to_sq)
                                else:
                                    env.state[S_BK] = Scalar[
                                        env.dtype
                                    ](to_sq)

                            var in_check = env._in_check(player)

                            env.state[from_sq] = saved_from
                            env.state[to_sq] = saved_to
                            env.state[S_WK] = saved_wk
                            env.state[S_BK] = saved_bk

                            if in_check:
                                print(
                                    "  After move: King would be in"
                                    " check? YES (pinned!)"
                                )
                            else:
                                print(
                                    "  After move: King would be in"
                                    " check? NO"
                                )

                            var wk = Int(env.state[S_WK])
                            var bk = Int(env.state[S_BK])
                            print(
                                "  White King at",
                                sq_name(wk),
                                "Black King at",
                                sq_name(bk),
                            )

                            # Try clicking a new friendly piece instead
                            var piece = Int(env.state[env_sq])
                            if piece != EMPTY and _is_friendly(piece, player):
                                selected_sq = env_sq
                            else:
                                selected_sq = -1

        # === Rendering ===

        # Get legal action mask for highlighting
        var legal_mask = env.legal_action_mask()

        # Draw board squares and pieces
        for row in range(8):
            for col in range(8):
                var px = left_margin + col * sq_size
                var py = top_margin + row * sq_size

                # Square color
                var is_light = ((row + col) % 2) == 0
                if is_light:
                    renderer.draw_rect(px, py, sq_size, sq_size, light_sq)
                else:
                    renderer.draw_rect(px, py, sq_size, sq_size, dark_sq)

                # Env coords: display row 0 = env row 7
                var env_row = 7 - row
                var env_col = col
                var env_sq = env_row * 8 + env_col

                # Highlight selected square (green border)
                if env_sq == selected_sq:
                    renderer.draw_rect(px + 1, py + 1, sq_size - 2, sq_size - 2, select_color, border_width=3)

                # Highlight legal target squares for selected piece
                if selected_sq >= 0 and env_sq != selected_sq and not game_over:
                    # Check all legal actions for moves from selected_sq to this sq
                    var is_legal_target = False
                    # Try no promo, queen, knight, bishop, rook promotions
                    for promo_val in range(6):
                        # 0=none, 2=knight, 3=bishop, 4=rook, 5=queen; skip 1
                        if promo_val == 1:
                            continue
                        var test_move = Move(selected_sq, env_sq, promo_val)
                        var test_action = _encode_action(test_move, player)
                        if test_action >= 0 and test_action < 4672 and legal_mask[test_action]:
                            is_legal_target = True
                            break
                    if is_legal_target:
                        renderer.draw_circle(
                            px + sq_size // 2, py + sq_size // 2,
                            10, legal_highlight, filled=True
                        )

                # Cursor highlight (yellow border)
                if row == cursor_row and col == cursor_col:
                    renderer.draw_rect(px, py, sq_size, sq_size, cursor_color, border_width=3)

                # Draw piece
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
                                renderer.sdl_renderer,
                                sprite_texture,
                                rebind[UnsafePointer[FRect, ImmutAnyOrigin]](src_rect),
                                rebind[UnsafePointer[FRect, ImmutAnyOrigin]](dst_rect),
                            )
                        except:
                            pass
                        src_rect.free()
                        dst_rect.free()
                    else:
                        # Fallback to text
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

        # File labels (a-h) below the board
        var label_color = SDL_Color(r=0xCC, g=0xCC, b=0xCC, a=0xFF)
        for col in range(8):
            var lx = left_margin + col * sq_size + sq_size // 2 - 4
            var ly = top_margin + 8 * sq_size + 4
            renderer.draw_text(
                _file_letter(col), lx, ly, label_color
            )

        # Rank labels (8 down to 1) to the left of the board
        for display_row in range(8):
            var rank_num = 8 - display_row
            var lx = left_margin - 12
            var ly = top_margin + display_row * sq_size + sq_size // 2 - 4
            renderer.draw_text(String(rank_num), lx, ly, label_color)

        # Status bar
        var status_y = top_margin + board_px + 20
        renderer.draw_rect(0, status_y, win_w, 50, status_bg)

        var result = Int(env.state[S_RESULT])
        if result == RESULT_ONGOING:
            if player == 0:
                renderer.draw_text("White's turn", 200, status_y + 20, text_color)
            else:
                renderer.draw_text("Black's turn", 200, status_y + 20, text_color)
        elif result == RESULT_WHITE_WINS:
            renderer.draw_text("Checkmate! White wins  (R to reset)", 120, status_y + 20, text_color)
        elif result == RESULT_BLACK_WINS:
            renderer.draw_text("Checkmate! Black wins  (R to reset)", 120, status_y + 20, text_color)
        elif result == RESULT_DRAW:
            renderer.draw_text("Draw!  (R to reset)", 180, status_y + 20, text_color)

        renderer.flip()

        # Save previous key/mouse states
        prev_up = cur_up
        prev_down = cur_down
        prev_left = cur_left
        prev_right = cur_right
        prev_space = cur_space
        prev_esc = cur_esc
        prev_r = cur_r
        prev_mouse_left = cur_mouse_left
        prev_mouse_right = cur_mouse_right

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
    print("=== Done ===")
