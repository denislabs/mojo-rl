# +--------------------------------------------------------------------------+ #
# | safetensors: the round trip, the widenings, and every way a file is wrong
# +--------------------------------------------------------------------------+ #
"""Self-contained gate for `mojo_rl/io/safetensors.mojo` — no dump, no network.

    pixi run mojo run -I . tests/io/test_safetensors.mojo

`tests/io/test_safetensors_reference.mojo` checks that we agree with the
reference implementation. This file checks the half a reference cannot reach:
what happens on files the reference would never write.

## Why the rejection cases are the point

A tensor read is a byte reinterpretation, so a reader that trusts its header
does not crash on a bad one — it returns the right COUNT of plausible floats
from the wrong bytes. Every case below is a header that is well-formed JSON
and internally almost consistent:

  * a header length larger than the file (the 8 attacker-controlled bytes in
    front of an allocation, where safetensors' own CVEs lived),
  * a header length of 2^64-1, which NARROWS to -1 and then looks like a fine
    length to anything that converted before comparing,
  * offsets that overlap, and offsets that leave a gap — each individually
    inside the blob, collectively impossible,
  * a byte span that does not match `shape x itemsize`,
  * a name declared twice, which is legal JSON and would make the file mean
    two things depending on which lookup asked.

Each is asserted to RAISE. A gate that only checks the happy path would pass
on a reader that skips all of this.

## The widenings

f16 and bf16 are checked against bit patterns computed from the IEEE
definitions, not from our own encoder — the point is to catch a widening that
is self-consistently wrong. bf16 in particular is exactly the top half of the
f32, which is easy to state and easy to implement as a rounding conversion by
mistake.
"""

from std.memory import bitcast

from mojo_rl.io.fileio import write_file_atomic
from mojo_rl.io.safetensors import (
    SafeTensors,
    SafeTensorsWriter,
    ST_BF16,
    ST_F16,
    ST_F32,
    dtype_name,
    is_float_dtype,
)


comptime TMP = "/tmp/mojo_rl_st_gate"


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def raises_on(mut fails: Int, name: String, path: String):
    """`SafeTensors(path)` must refuse. Prints the message on a pass so the
    gate shows WHAT was rejected, not just that something was."""
    var msg = String("")
    var refused = False
    try:
        var f = SafeTensors(path)
        _ = f.size()
    except e:
        refused = True
        msg = String(e)
    if refused:
        print("  PASS  rejects " + name)
        print("          " + msg)
    else:
        fails += 1
        print("  FAIL  ACCEPTED " + name)


def put(path: String, var header: String, ref data: List[UInt8]) raises:
    """Write a file with a header length that MATCHES the header."""
    put_raw(path, header^, data, -1)


def put_raw(
    path: String, var header: String, ref data: List[UInt8], claim: Int
) raises:
    """`claim` overrides the declared header length; -1 means "the truth"."""
    var hb = header.as_bytes()
    var n = UInt64(len(hb)) if claim < 0 else UInt64(claim)
    var buf = List[UInt8]()
    for k in range(8):
        buf.append(UInt8(Int((n >> UInt64(8 * k)) & UInt64(0xFF))))
    for i in range(len(hb)):
        buf.append(hb[i])
    for i in range(len(data)):
        buf.append(data[i])
    write_file_atomic(String(path), buf)


def put_u64(path: String, var header: String, ref data: List[UInt8], n: UInt64) raises:
    var hb = header.as_bytes()
    var buf = List[UInt8]()
    for k in range(8):
        buf.append(UInt8(Int((n >> UInt64(8 * k)) & UInt64(0xFF))))
    for i in range(len(hb)):
        buf.append(hb[i])
    for i in range(len(data)):
        buf.append(data[i])
    write_file_atomic(String(path), buf)


def f32_bytes(ref vals: List[Float32]) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(len(vals)):
        var b = bitcast[DType.uint32](vals[i])
        for k in range(4):
            out.append(UInt8(Int((b >> UInt32(8 * k)) & UInt32(0xFF))))
    return out^


def u16_bytes(ref words: List[Int]) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(len(words)):
        out.append(UInt8(words[i] & 0xFF))
        out.append(UInt8((words[i] >> 8) & 0xFF))
    return out^


def main() raises:
    var fails = 0
    print("safetensors: round trip, widening, and malformed headers")
    print("")

    # ══════════════════════════════════════════════════════════════════════
    print("round trip")
    var w = SafeTensorsWriter()
    var big = List[Float32]()
    for i in range(1000):
        big.append(Float32(i) * 0.125 - 60.0)
    var shape_big: List[Int] = [10, 100]
    w.add_f32_list(String("m.big"), shape_big, big)
    var one: List[Float32] = [Float32(-7.25)]
    var shape_scalar = List[Int]()
    w.add_f32_list(String("m.scalar"), shape_scalar, one)
    var none = List[Float32]()
    var shape_zero: List[Int] = [0, 4]
    w.add_f32_list(String("m.empty"), shape_zero, none)
    w.add_metadata(String("producer"), String("mojo-rl"))
    w.save(String(TMP) + "_rt.safetensors")

    var rt = SafeTensors(String(TMP) + "_rt.safetensors")
    check(fails, "3 tensors", rt.size() == 3, String(rt.size()))
    check(
        fails,
        "order is the write order",
        rt.names[0] == "m.big" and rt.names[2] == "m.empty",
    )
    check(fails, "shape survives", rt.shape_str(String("m.big")) == "[10, 100]",
          rt.shape_str(String("m.big")))
    check(fails, "rank 0 survives", rt.shape_str(String("m.scalar")) == "[]",
          rt.shape_str(String("m.scalar")))
    check(fails, "0-element tensor survives",
          rt.numel(String("m.empty")) == 0 and len(rt.read_f32(String("m.empty"))) == 0)
    check(fails, "metadata survives", rt.metadata(String("producer")) == "mojo-rl")
    var back = rt.read_f32(String("m.big"))
    var bad = 0
    for i in range(len(back)):
        if bitcast[DType.uint32](back[i]) != bitcast[DType.uint32](big[i]):
            bad += 1
    check(fails, "1000 values bit-identical", bad == 0 and len(back) == 1000,
          String(len(back)) + " compared, " + String(bad) + " differ")
    check(fails, "8-byte aligned data block", rt.data_start % 8 == 0,
          "data_start = " + String(rt.data_start))

    # ══════════════════════════════════════════════════════════════════════
    print("")
    print("widening (bit patterns from the IEEE definitions, not from us)")
    # 1.5, -2.5, 0.25, -12.5
    var want: List[Float32] = [1.5, -2.5, 0.25, -12.5]
    var bf: List[Int] = [0x3FC0, 0xC020, 0x3E80, 0xC148]
    var hf: List[Int] = [0x3E00, 0xC100, 0x3400, 0xCA40]

    var bfd = u16_bytes(bf)
    put(
        String(TMP) + "_bf16.safetensors",
        String('{"t":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}'),
        bfd,
    )
    var bfile = SafeTensors(String(TMP) + "_bf16.safetensors")
    var bg = bfile.read_f32(String("t"))
    var bbad = 0
    for i in range(4):
        if bitcast[DType.uint32](bg[i]) != bitcast[DType.uint32](want[i]):
            bbad += 1
    check(fails, "BF16 widens exactly (top half of the f32)",
          bbad == 0 and len(bg) == 4,
          String(len(bg)) + " compared, " + String(bbad) + " differ")

    var hfd = u16_bytes(hf)
    put(
        String(TMP) + "_f16.safetensors",
        String('{"t":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}'),
        hfd,
    )
    var hfile = SafeTensors(String(TMP) + "_f16.safetensors")
    var hg = hfile.read_f32(String("t"))
    var hbad = 0
    for i in range(4):
        if bitcast[DType.uint32](hg[i]) != bitcast[DType.uint32](want[i]):
            hbad += 1
    check(fails, "F16 widens exactly", hbad == 0 and len(hg) == 4,
          String(len(hg)) + " compared, " + String(hbad) + " differ")

    # ══════════════════════════════════════════════════════════════════════
    print("")
    print("malformed headers — each of these must RAISE")
    var eight = f32_bytes(want)  # 16 bytes

    # The header length is 8 attacker-controlled bytes in front of an alloc.
    put_raw(
        String(TMP) + "_hlong.safetensors",
        String('{"t":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}'),
        eight,
        1 << 20,
    )
    raises_on(fails, "a header length past the end of the file",
              String(TMP) + "_hlong.safetensors")

    # 2^64-1 narrows to -1. A reader that converts before comparing sees a
    # perfectly reasonable length.
    put_u64(
        String(TMP) + "_hmax.safetensors",
        String('{"t":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}'),
        eight,
        UInt64(0xFFFFFFFFFFFFFFFF),
    )
    raises_on(fails, "a header length of 2^64-1 (narrows to -1)",
              String(TMP) + "_hmax.safetensors")

    var tiny = List[UInt8]()
    tiny.append(UInt8(1))
    write_file_atomic(String(TMP) + "_short.safetensors", tiny)
    raises_on(fails, "a 1-byte file", String(TMP) + "_short.safetensors")

    put(
        String(TMP) + "_overlap.safetensors",
        String(
            '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
            '"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}'
        ),
        eight,
    )
    raises_on(fails, "overlapping tensors", String(TMP) + "_overlap.safetensors")

    var twentyfour = f32_bytes(want)
    for _ in range(8):
        twentyfour.append(UInt8(0))
    put(
        String(TMP) + "_gap.safetensors",
        String(
            '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
            '"b":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}'
        ),
        twentyfour,
    )
    raises_on(fails, "a gap between tensors", String(TMP) + "_gap.safetensors")

    put(
        String(TMP) + "_size.safetensors",
        String('{"t":{"dtype":"F32","shape":[8],"data_offsets":[0,16]}}'),
        eight,
    )
    raises_on(fails, "a span that does not match shape x itemsize",
              String(TMP) + "_size.safetensors")

    put(
        String(TMP) + "_dup.safetensors",
        String(
            '{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
            '"t":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}'
        ),
        eight,
    )
    raises_on(fails, "the same name twice", String(TMP) + "_dup.safetensors")

    put(
        String(TMP) + "_dtype.safetensors",
        String('{"t":{"dtype":"F24","shape":[4],"data_offsets":[0,16]}}'),
        eight,
    )
    raises_on(fails, "a dtype not in the specification",
              String(TMP) + "_dtype.safetensors")

    put(
        String(TMP) + "_negdim.safetensors",
        String('{"t":{"dtype":"F32","shape":[-4],"data_offsets":[0,16]}}'),
        eight,
    )
    raises_on(fails, "a negative dimension", String(TMP) + "_negdim.safetensors")

    put(
        String(TMP) + "_past.safetensors",
        String('{"t":{"dtype":"F32","shape":[4],"data_offsets":[8,24]}}'),
        eight,
    )
    raises_on(fails, "a span reaching past the data block",
              String(TMP) + "_past.safetensors")

    put(
        String(TMP) + "_notobj.safetensors",
        String('[{"dtype":"F32","shape":[4],"data_offsets":[0,16]}]'),
        eight,
    )
    raises_on(fails, "a header that is a list, not an object",
              String(TMP) + "_notobj.safetensors")

    # ══════════════════════════════════════════════════════════════════════
    print("")
    print("the writer refuses too")
    var w2 = SafeTensorsWriter()
    var v2: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var s2: List[Int] = [2, 2]
    w2.add_f32_list(String("x"), s2, v2)

    var dup_refused = False
    try:
        w2.add_f32_list(String("x"), s2, v2)
    except:
        dup_refused = True
    check(fails, "a duplicate name is refused at add time", dup_refused)

    var bad_shape_refused = False
    try:
        var s3: List[Int] = [3, 3]
        w2.add_f32_list(String("y"), s3, v2)
    except:
        bad_shape_refused = True
    check(fails, "a shape that does not match the value count is refused",
          bad_shape_refused)

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("safetensors gate failed")
