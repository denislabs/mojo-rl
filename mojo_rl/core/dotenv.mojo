"""Minimal .env file parser.

Reads KEY=VALUE lines from a file, skipping comments and blank lines.
Strips surrounding quotes (single or double) from values.
"""

from std.collections import Dict
from std.pathlib import Path


def _split_lines(content: String) -> List[String]:
    """Split content into lines."""
    var lines = List[String]()
    var current = String("")
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == UInt8(ord("\n")):
            lines.append(current)
            current = String("")
        else:
            current += chr(Int(bytes[i]))
    if len(current) > 0:
        lines.append(current)
    return lines^


def load_dotenv(path: String = ".env") raises -> Dict[String, String]:
    """Parse a .env file and return a dict of key-value pairs.

    Args:
        path: Path to the .env file (default: ".env").

    Returns:
        Dict mapping variable names to values. Empty dict if file not found.
    """
    var result = Dict[String, String]()

    var p = Path(path)
    if not p.exists():
        return result^

    var content = p.read_text()
    var lines = _split_lines(content)

    for i in range(len(lines)):
        var line = String(lines[i])

        # Skip empty lines and comments
        if len(line) == 0 or line.startswith("#"):
            continue

        # Skip export prefix
        if line.startswith("export "):
            line = String(line[7:])

        # Find the = separator
        var eq_pos = line.find("=")
        if eq_pos < 0:
            continue

        var key = String(line[:eq_pos])
        var val = String(line[eq_pos + 1 :])

        # Strip surrounding quotes
        if len(val) >= 2:
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = String(val[1 : len(val) - 1])

        result[key] = val

    return result^
