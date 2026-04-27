"""In-place CLI progress bar — pure CPU, no GPU sync."""


def print_progress_bar(
    current: Int,
    total: Int,
    train_steps: Int,
    algorithm_name: String,
    bar_width: Int = 30,
):
    """Print an in-place progress bar using carriage return (no GPU sync).

    Uses only CPU-side counters so it adds zero overhead to GPU training.

    Args:
        current: Current step count.
        total: Target step count.
        train_steps: Total gradient updates so far.
        algorithm_name: Algorithm name prefix.
        bar_width: Width of the bar in characters (default 30).
    """
    var pct = current * 100 // total
    var filled = current * bar_width // total
    var bar = String("")
    for i in range(bar_width):
        if i < filled:
            bar += "█"
        else:
            bar += "░"
    print(
        "\r"
        + algorithm_name
        + " ["
        + bar
        + "] "
        + String(pct)
        + "% | Step "
        + String(current)
        + "/"
        + String(total)
        + " | Train: "
        + String(train_steps),
        end="",
    )


def clear_progress_bar():
    """Overwrite the current progress bar line with spaces and return to start.

    Call this before printing stats to ensure the progress bar is fully erased.
    """
    # 120 spaces is enough to cover any progress bar output
    print("\r" + String(" ") * 120 + "\r", end="")
