"""BCD score decoding helpers.

Many Atari games store scores as BCD (Binary Coded Decimal) in RAM.
Each byte holds two decimal digits (upper nibble = tens, lower = ones).

Ported from CuLE (BSD-3): cule/atari/games/detail/utils.hpp
"""

from ..flags import RAM_SIZE


@always_inline
fn get_decimal_score(ram: InlineArray[UInt8, RAM_SIZE], idx: Int) -> Int:
    """Decode a 2-digit BCD score from one RAM byte.

    Returns: 10 * upper_nibble + lower_nibble (0-99)
    """
    var val = Int(ram[idx & 0x7F])
    var right = val & 0x0F
    var left = (val >> 4) & 0x0F
    return 10 * left + right


@always_inline
fn get_decimal_score_2(
    ram: InlineArray[UInt8, RAM_SIZE], lower_idx: Int, higher_idx: Int
) -> Int:
    """Decode a 4-digit BCD score from two RAM bytes.

    lower_idx: ones and tens digits
    higher_idx: hundreds and thousands digits (use -1 to skip)

    Returns: score (0-9999)
    """
    var val_lo = Int(ram[lower_idx & 0x7F])
    var lo_right = val_lo & 0x0F
    var lo_left = (val_lo >> 4) & 0x0F
    var score = 10 * lo_left + lo_right

    if higher_idx < 0:
        return score

    var val_hi = Int(ram[higher_idx & 0x7F])
    var hi_right = val_hi & 0x0F
    var hi_left = (val_hi >> 4) & 0x0F
    score += 1000 * hi_left + 100 * hi_right

    return score


@always_inline
fn get_decimal_score_3(
    ram: InlineArray[UInt8, RAM_SIZE],
    lower_idx: Int,
    middle_idx: Int,
    higher_idx: Int,
) -> Int:
    """Decode a 6-digit BCD score from three RAM bytes.

    Returns: score (0-999999)
    """
    var score = get_decimal_score_2(ram, lower_idx, middle_idx)
    var val_hi = Int(ram[higher_idx & 0x7F])
    var hi_right = val_hi & 0x0F
    var hi_left = (val_hi >> 4) & 0x0F
    score += 100000 * hi_left + 10000 * hi_right
    return score
