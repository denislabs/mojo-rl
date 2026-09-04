"""Minimal .env file parser.

Reads KEY=VALUE lines from a file, skipping comments and blank lines.
Strips surrounding quotes (single or double) from values.
"""

from mojo_rl.core.bytes import string_from_bytes

from std.collections import Dict
from std.pathlib import Path


def _split_lines(content: String) -> List[String]:
    """Split content into lines."""
    var lines = List[String]()
    # ⚠ BYTES, not a String accumulated with `chr` — see `core/bytes.mojo`.
    # A `.env` value can hold a token or an accented path.
    var cur = List[UInt8]()
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == UInt8(ord("\n")):
            lines.append(string_from_bytes(cur))
            cur = List[UInt8]()
        else:
            cur.append(bytes[i])
    if len(cur) > 0:
        lines.append(string_from_bytes(cur))
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
        if line.byte_length() == 0 or line.startswith("#"):
            continue

        # Skip export prefix
        if line.startswith("export "):
            # Temporary first: nightly rejects building a `String` from a slice
            # of the string being assigned to.
            var rest = String(line[byte=7:])
            line = rest^

        # Find the = separator
        var eq_pos = line.find("=")
        if eq_pos < 0:
            continue

        var key = String(line[byte=:eq_pos])
        var val = String(line[byte=eq_pos + 1 :])

        # Strip surrounding quotes
        if val.byte_length() >= 2:
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                var unquoted = String(val[byte=1 : val.byte_length() - 1])
                val = unquoted^

        result[key] = val

    return result^
