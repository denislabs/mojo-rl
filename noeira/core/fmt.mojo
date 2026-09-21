"""Print-formatting helpers for column-aligned console output.

Mojo 1.0 made a contiguous slice with an out-of-range end **abort** instead of
silently clamping it:

    "`List`, `Span`, and `String`/`StringSlice` indexing with a contiguous
     (non-strided) slice now aborts on an invalid slice instead of silently
     clamping it."

That turned the `String(x)[byte=:8]` truncation idiom used throughout the
tests, examples and benchmarks into a runtime landmine — `String(0)` is one
byte long, so any step counter printed that way crashes the process. `fit`
clamps the end explicitly.
"""


def fit(s: String, n: Int) -> String:
    """Truncate `s` to at most `n` bytes.

    Unlike a bare `s[byte=:n]` this never aborts when `s` is shorter than `n`;
    it returns the whole string instead.

    Args:
        s: The rendered value.
        n: Maximum width in bytes.

    Returns:
        `s` truncated to `n` bytes, or `s` unchanged when it is already shorter.
    """
    return String(s[byte = : min(s.byte_length(), n)])
