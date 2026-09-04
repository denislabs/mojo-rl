"""`key=value` text — the shared reader for this tree's own file formats.

Several formats here are a flat block of `key=value` lines with one or more
REPEATING keys: `data/manifest.mojo` (`column=`), checkpoint v2, `core/dotenv`,
and now `tasks/spec.mojo` (`slot=`, `region=`, `active=`, `init=`). The parsing
is four lines each and has been written three times.

⚠ THIS IS THE HOME FOR NEW READERS, NOT YET A MIGRATION. `data/manifest._split`
and `core/dotenv._split_lines` are the two older private copies. Moving them
here is a separate change on the DATA path — the manifest lives inside every
`.h5` this tree has written — and is not worth bundling into a task-layer
commit. **Do not add a fourth copy.**

## ⚠ BYTE-WISE ON PURPOSE

Every one of these formats is ASCII by construction. Splitting by codepoint
would make a stray UTF-8 byte a slicing error somewhere far from the line that
carried it; `_a_byte_slice_of_a_string_asserts_a_codepoint_boundary` is the
recorded shape. The one field that is legitimately human text — a task's
`language=` instruction — is carried through as bytes and never split.
"""


def split_on(s: String, sep: String) -> List[String]:

    """Every field of `s` between single-byte `sep`. Empty fields are kept.

    ⚠⚠ BYTES, NOT `chr` PER BYTE. `cur += chr(Int(bytes[i]))` is the obvious
    spelling and it CORRUPTS any value above 127: `chr` yields the CODEPOINT
    of that byte value, which re-encodes as two bytes, so "日本語" comes back
    as mojibake. This reader carried that bug for as long as it has existed —
    every non-ASCII value in every manifest read back wrong — and it surfaced
    only when a byte-exact task string was finally put through it.
    """
    var out = List[String]()
    var cur = List[UInt8]()
    var bytes = s.as_bytes()
    var sb = sep.as_bytes()[0]
    for i in range(len(bytes)):
        if bytes[i] == sb:
            cur.append(0)
            out.append(String(unsafe_from_utf8_ptr=cur.unsafe_ptr()))
            cur = List[UInt8]()
        else:
            cur.append(bytes[i])
    cur.append(0)
    out.append(String(unsafe_from_utf8_ptr=cur.unsafe_ptr()))
    return out^


def split_once(s: String, sep: String) -> List[String]:
    """`[head, tail]` around the FIRST `sep`, or an EMPTY list if absent.

    ⚠ SPLIT-ONCE MATTERS BECAUSE THE VALUE MAY CONTAIN THE SEPARATOR. A task's
    `language=Put the brick in the box, then stop` has no `=`, but a region's
    `region=table:site:table_surface:-0.1,-0.15,0.1,0.15` is split on `:` by a
    caller that must not lose the negative numbers to a greedy split.


    ⚠⚠ BYTES, NOT `chr` PER BYTE. `cur += chr(Int(bytes[i]))` is the obvious
    spelling and it CORRUPTS any value above 127: `chr` yields the CODEPOINT
    of that byte value, which re-encodes as two bytes, so "日本語" comes back
    as mojibake. This reader carried that bug for as long as it has existed —
    every non-ASCII value in every manifest read back wrong — and it surfaced
    only when a byte-exact task string was finally put through it.
    """
    var out = List[String]()
    var bytes = s.as_bytes()
    var sb = sep.as_bytes()[0]
    var cut = -1
    for i in range(len(bytes)):
        if bytes[i] == sb:
            cut = i
            break
    if cut < 0:
        return out^
    var head = List[UInt8]()
    for i in range(cut):
        head.append(bytes[i])
    head.append(0)
    var tail = List[UInt8]()
    for i in range(cut + 1, len(bytes)):
        tail.append(bytes[i])
    tail.append(0)
    out.append(String(unsafe_from_utf8_ptr=head.unsafe_ptr()))
    out.append(String(unsafe_from_utf8_ptr=tail.unsafe_ptr()))
    return out^


struct KvLine(Copyable, ImplicitlyCopyable, Movable):
    """One `key=value` line, plus the 1-based line number it came from.

    ⚠ THE LINE NUMBER IS NOT DECORATION. These files are hand-authored, so a
    diagnostic that cannot say WHERE costs the author a manual scan of a
    file whose whole point is that a human writes it.
    """

    var key: String
    var value: String
    var lineno: Int

    def __init__(out self, key: String, value: String, lineno: Int):
        self.key = key
        self.value = value
        self.lineno = lineno


def kv_lines(text: String, what: String) raises -> List[KvLine]:
    """Every `key=value` line of `text`, in order. Blank and `#` lines skipped.

    `what` names the format in diagnostics ("task spec", "family spec").

    ⚠ RAISES ON A LINE WITH NO `=`. It does NOT decide what to do with an
    unknown KEY — that is the caller's policy, and the two callers here differ:
    `data/manifest.mojo` ignores unknown keys so a store written by a newer
    build stays readable, while `tasks/spec.mojo` REFUSES them, because a
    silently dropped `goal=` is a task that always succeeds. Keeping that
    decision out of here is what lets both be right.
    """
    var out = List[KvLine]()
    var lines = split_on(text, String("\n"))
    for i in range(len(lines)):
        var line = String(lines[i].strip())
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var kv = split_once(line, String("="))
        if len(kv) != 2:
            raise Error(
                what + ": line " + String(i + 1)
                + " has no '=': '" + line + "'"
            )
        var key = String(kv[0].strip())
        if key.byte_length() == 0:
            raise Error(
                what + ": line " + String(i + 1) + " has an empty key: '"
                + line + "'"
            )
        out.append(KvLine(key^, String(kv[1].strip()), i + 1))
    return out^
