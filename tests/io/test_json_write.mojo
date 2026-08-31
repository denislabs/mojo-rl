# +--------------------------------------------------------------------------+ #
# | The JSON writer, gated on its exact output
# +--------------------------------------------------------------------------+ #
"""Gate `JsonWriter` / `json_quote` in `mojo_rl/io/json.mojo`.

    pixi run mojo run -I . tests/io/test_json_write.mojo

⚠ THE EXPECTED TEXT IS PINNED, and it was validated by Python's `json.loads`
before being pinned here. Round-tripping through this repo's own reader is NOT
sufficient on its own: a writer and a reader sharing one wrong assumption
about escaping agree with each other and with nobody else — the
two-parsers-one-wrong-default shape this tree has recorded before. So this
gate checks BOTH: byte-exact output against a literal a third implementation
accepted, and a round trip.

What it covers beyond the happy path:
* the separator rule (`{"a":1,"b":2}`, never `{"a":,1}` or `{"a":1,}`),
* every escape `json_quote` special-cases, plus the `\\u00XX` fallback for the
  control bytes that have no short form,
* NaN and Infinity, which Python's `json.dumps` emits as bare `NaN` /
  `Infinity` — not JSON, and rejected by a strict server,
* an unbalanced document, which must raise at `done()` rather than emit.
"""

from mojo_rl.io.json import JsonWriter, json_quote, parse_json


def _bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(s.byte_length()):
        out.append(s.as_bytes()[i])
    return out^


def _eq(got: String, want: String, what: String) raises:
    if got != want:
        raise Error(what + ":\n     got  " + got + "\n     want " + want)


def main() raises:
    print("=== JsonWriter ===")
    var checks = 0

    # ── the shape `core/logger.mojo` actually posts ──────────────────
    var w = JsonWriter()
    w.begin_object()
    w.member(String("run_id"), String("run_42"))
    w.key(String("metrics"))
    w.begin_array()
    for i in range(2):
        w.begin_object()
        w.member(String("step"), i)
        w.member(String("value"), Float64(i) + 0.5)
        w.end_object()
    w.end_array()
    w.key(String("config"))
    w.begin_object()
    w.end_object()
    w.end_object()
    _eq(
        w.done(),
        String(
            '{"run_id":"run_42","metrics":[{"step":0,"value":0.5},'
            '{"step":1,"value":1.5}],"config":{}}'
        ),
        "the ingest payload",
    )
    checks += 1

    # ── escaping ────────────────────────────────────────────────────
    _eq(json_quote(String('a"b')), String('"a\\"b"'), "quote")
    _eq(json_quote(String("a\\b")), String('"a\\\\b"'), "backslash")
    _eq(json_quote(String("a\nb")), String('"a\\nb"'), "newline")
    _eq(json_quote(String("a\tb")), String('"a\\tb"'), "tab")
    _eq(json_quote(String("a\rb")), String('"a\\rb"'), "carriage return")
    checks += 5

    # ⚠ 0x01 HAS NO SHORT ESCAPE and must come out as a \\u sequence. A raw
    # control byte inside a JSON string is invalid, not merely unusual.
    var ctrl = String("a") + chr(1) + String("b")
    _eq(json_quote(ctrl), String('"a\\u0001b"'), "\\u escape for 0x01")
    var ctrl2 = String("x") + chr(0x1F) + String("y")
    _eq(json_quote(ctrl2), String('"x\\u001fy"'), "\\u escape for 0x1f")
    checks += 2

    # ── UTF-8 passes through unescaped ───────────────────────────────
    var utf8 = String("héllo → ✓")
    var q = json_quote(utf8)
    if q.byte_length() != utf8.byte_length() + 2:
        raise Error("json_quote escaped UTF-8 it should have passed through")
    checks += 1

    # ── non-finite numbers are refused ──────────────────────────────
    for bad in [Float64("nan"), Float64("inf"), -Float64("inf")]:
        var w2 = JsonWriter()
        w2.begin_object()
        w2.key(String("v"))
        var raised = False
        try:
            w2.number(bad)
        except:
            raised = True
        if not raised:
            raise Error(
                "JsonWriter.number accepted " + String(bad) + " — JSON has no"
                " representation for it"
            )
        checks += 1

    # ── an unbalanced document must not be handed out ────────────────
    var w3 = JsonWriter()
    w3.begin_object()
    w3.key(String("a"))
    w3.integer(1)
    var raised_unbalanced = False
    try:
        _ = w3.done()
    except:
        raised_unbalanced = True
    if not raised_unbalanced:
        raise Error("done() returned an object that was never closed")
    checks += 1

    # ── round trip through this repo's reader ───────────────────────
    var w4 = JsonWriter()
    w4.begin_object()
    w4.member(String("name"), String('has "quotes", a \\ and a\ttab'))
    w4.member(String("n"), 7)
    w4.member(String("x"), -3.5)
    w4.key(String("ok"))
    w4.boolean(True)
    w4.key(String("nil"))
    w4.null()
    w4.end_object()
    var text = w4.done()
    var doc = parse_json(_bytes(text))
    var root = doc.root()
    _eq(
        doc.string(doc.field(root, String("name"))),
        String('has "quotes", a \\ and a\ttab'),
        "round-tripped string",
    )
    if doc.integer(doc.field(root, String("n"))) != 7:
        raise Error("round-tripped integer")
    if doc.number(doc.field(root, String("x"))) != -3.5:
        raise Error("round-tripped float")
    if not doc.boolean(doc.field(root, String("ok"))):
        raise Error("round-tripped bool")
    if doc.kind_of(doc.field(root, String("nil"))) != 0:
        raise Error("round-tripped null")
    checks += 5

    print("  " + String(checks) + " checks, 0 failing")
    print("[PASS] JsonWriter")
