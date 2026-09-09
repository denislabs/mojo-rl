"""A reader for joblib pickles of numpy arrays — enough of the pickle VM for
`lafan_29dof.pkl` and its kin, no Python.

`joblib.dump(dict_of_dicts_of_arrays)` writes a protocol-4 pickle in which
every array is a `joblib.numpy_pickle.NumpyArrayWrapper` OBJECT followed by
the array's raw bytes spliced INTO the stream: after the wrapper's BUILD
opcode comes one byte holding a padding length, that many `0xff`, then
`prod(shape) * itemsize` bytes in C order, then the pickle resumes (a new
FRAME). joblib's `NumpyUnpickler.load_build` does exactly that read when
the object on top of the stack is a wrapper, and swaps the array in for
it. This module is that unpickler for the opcodes such a file uses, found
by counting them over the whole file (`docs/BFM_ZERO_G1_REPRODUCTION.md`
§13):

    PROTO FRAME STOP  EMPTY_DICT SETITEMS  MARK TUPLE TUPLE1 TUPLE2 TUPLE3
    EMPTY_TUPLE  SHORT_BINUNICODE BINUNICODE  BININT1 BININT2 BININT
    NEWTRUE NEWFALSE NONE  MEMOIZE BINGET LONG_BINGET  STACK_GLOBAL
    NEWOBJ REDUCE BUILD  BINFLOAT

Values are a tagged arena (`PVal`): none / bool / int / float / str /
tuple / dict / global / object / array. `numpy.dtype('f4', False, True)`
arrives through REDUCE and its state through BUILD (kept, ignored);
`NumpyArrayWrapper()` through NEWOBJ with its fields through BUILD
(`subclass`, `shape`, `order`, `dtype`, `allow_mmap`,
`numpy_array_alignment_bytes`), at which point the array bytes are
consumed. Arrays are returned as `(shape, dtype name, bytes offset)` into
the file image the reader holds, and materialised on request
(`array_f32`) so a 200 MB pickle is one read and no copies until asked.

⚠ NOT A GENERAL UNPICKLER. Any opcode outside the set above raises with
its byte and offset — better than a silent misparse. Object arrays, memmaps
(`allow_mmap` is read and ignored: the bytes are inline either way),
pickles from joblib < 0.10 (`NDArrayWrapper`) and big-endian arrays are
not handled. Only little-endian `f4`, `f8`, `i4`, `i8` dtypes are
materialised; others are kept as bytes.

Gate: `tests/io/test_pickle_joblib.mojo` — a fixture written by joblib with
every array dtype and shape rank this reader claims, plus the LAFAN pickle's
first clip against the reference's own load through the G1 store.
"""

from std.memory import bitcast
from std.os.path import exists


comptime PV_NONE: Int = 0
comptime PV_BOOL: Int = 1
comptime PV_INT: Int = 2
comptime PV_FLOAT: Int = 3
comptime PV_STR: Int = 4
comptime PV_TUPLE: Int = 5
comptime PV_DICT: Int = 6
comptime PV_GLOBAL: Int = 7
comptime PV_OBJ: Int = 8
comptime PV_ARRAY: Int = 9
comptime PV_MARK: Int = 10


struct PVal(Copyable, Movable):
    """One arena node. `kids` holds child node ids: a tuple's items, a dict's
    interleaved key/value ids, an object's [class id, state id]. Arrays keep
    `shape` and the byte offset of their data in the file image."""
    var kind: Int
    var i: Int
    var f: Float64
    var s: String            # str value; a global's "module.name"; an array's dtype
    var kids: List[Int]
    var shape: List[Int]
    var off: Int             # array: byte offset of the data
    var nbytes: Int

    def __init__(out self, kind: Int):
        self.kind = kind
        self.i = 0
        self.f = 0.0
        self.s = String("")
        self.kids = List[Int]()
        self.shape = List[Int]()
        self.off = 0
        self.nbytes = 0

    def __init__(out self, *, copy: Self):
        self.kind = copy.kind
        self.i = copy.i
        self.f = copy.f
        self.s = copy.s.copy()
        self.kids = copy.kids.copy()
        self.shape = copy.shape.copy()
        self.off = copy.off
        self.nbytes = copy.nbytes

    def __init__(out self, *, deinit move: Self):
        self.kind = move.kind
        self.i = move.i
        self.f = move.f
        self.s = move.s^
        self.kids = move.kids^
        self.shape = move.shape^
        self.off = move.off
        self.nbytes = move.nbytes


def _dtype_itemsize(name: String) raises -> Int:
    """numpy dtype strings as they appear in `numpy.dtype(<str>)` REDUCE args."""
    var n = name
    if n.byte_length() > 0 and (n.startswith("<") or n.startswith("=") or n.startswith("|")):
        var rest = String(n[byte=1:])
        n = rest
    if n == "f4" or n == "i4" or n == "u4":
        return 4
    if n == "f8" or n == "i8" or n == "u8":
        return 8
    if n == "f2" or n == "i2" or n == "u2":
        return 2
    if n == "i1" or n == "u1" or n == "b1":
        return 1
    raise Error("pickle: unsupported array dtype '" + name + "'")


struct JoblibPickle(Movable):
    """The parsed file: the arena, the root node, and the file image."""
    var buf: List[UInt8]
    var nodes: List[PVal]
    var root: Int

    def __init__(out self):
        self.buf = List[UInt8]()
        self.nodes = List[PVal]()
        self.root = -1

    def __init__(out self, *, deinit move: Self):
        self.buf = move.buf^
        self.nodes = move.nodes^
        self.root = move.root

    @staticmethod
    def load(path: String) raises -> Self:
        if not exists(path):
            raise Error("pickle: no such file " + path)
        var p = Self()
        var f = open(path, "r")
        p.buf = f.read_bytes()
        f.close()
        p.root = p._run()
        return p^

    # ── the VM ──────────────────────────────────────────────────────────────
    def _push(mut self, var v: PVal) -> Int:
        self.nodes.append(v^)
        return len(self.nodes) - 1

    def _u8(self, pos: Int) -> Int:
        return Int(self.buf[pos])

    def _u16(self, pos: Int) -> Int:
        return Int(self.buf[pos]) | (Int(self.buf[pos + 1]) << 8)

    def _i32(self, pos: Int) -> Int:
        var v = (
            Int(self.buf[pos]) | (Int(self.buf[pos + 1]) << 8)
            | (Int(self.buf[pos + 2]) << 16) | (Int(self.buf[pos + 3]) << 24)
        )
        if v >= (1 << 31):
            v -= 1 << 32
        return v

    def _u32(self, pos: Int) -> Int:
        return (
            Int(self.buf[pos]) | (Int(self.buf[pos + 1]) << 8)
            | (Int(self.buf[pos + 2]) << 16) | (Int(self.buf[pos + 3]) << 24)
        )

    def _str(self, pos: Int, n: Int) -> String:
        var out = String()
        for i in range(n):
            out += chr(Int(self.buf[pos + i]))
        return out

    def _run(mut self) raises -> Int:
        var stack = List[Int]()
        var memo = List[Int]()
        var pos = 0
        var n = len(self.buf)
        while pos < n:
            var op = self._u8(pos)
            pos += 1
            if op == 0x80:  # PROTO
                pos += 1
            elif op == 0x95:  # FRAME: 8-byte length, contents follow inline
                pos += 8
            elif op == 0x2E:  # STOP
                if len(stack) != 1:
                    raise Error("pickle: STOP with " + String(len(stack)) + " values on the stack")
                return stack[0]
            elif op == 0x7D:  # EMPTY_DICT
                stack.append(self._push(PVal(PV_DICT)))
            elif op == 0x29:  # EMPTY_TUPLE
                stack.append(self._push(PVal(PV_TUPLE)))
            elif op == 0x28:  # MARK
                stack.append(self._push(PVal(PV_MARK)))
            elif op == 0x8C:  # SHORT_BINUNICODE
                var ln = self._u8(pos)
                var v = PVal(PV_STR)
                v.s = self._str(pos + 1, ln)
                pos += 1 + ln
                stack.append(self._push(v^))
            elif op == 0x58:  # BINUNICODE
                var ln = self._u32(pos)
                var v = PVal(PV_STR)
                v.s = self._str(pos + 4, ln)
                pos += 4 + ln
                stack.append(self._push(v^))
            elif op == 0x4B:  # BININT1
                var v = PVal(PV_INT)
                v.i = self._u8(pos)
                pos += 1
                stack.append(self._push(v^))
            elif op == 0x4D:  # BININT2
                var v = PVal(PV_INT)
                v.i = self._u16(pos)
                pos += 2
                stack.append(self._push(v^))
            elif op == 0x4A:  # BININT
                var v = PVal(PV_INT)
                v.i = self._i32(pos)
                pos += 4
                stack.append(self._push(v^))
            elif op == 0x47:  # BINFLOAT, big-endian double
                var bits: UInt64 = 0
                for k in range(8):
                    bits = (bits << 8) | UInt64(self.buf[pos + k])
                pos += 8
                var v = PVal(PV_FLOAT)
                v.f = Float64(bitcast[DType.float64](Scalar[DType.uint64](bits)))
                stack.append(self._push(v^))
            elif op == 0x88:  # NEWTRUE
                var v = PVal(PV_BOOL)
                v.i = 1
                stack.append(self._push(v^))
            elif op == 0x89:  # NEWFALSE
                stack.append(self._push(PVal(PV_BOOL)))
            elif op == 0x4E:  # NONE
                stack.append(self._push(PVal(PV_NONE)))
            elif op == 0x94:  # MEMOIZE
                memo.append(stack[len(stack) - 1])
            elif op == 0x68:  # BINGET
                var k = self._u8(pos)
                pos += 1
                stack.append(memo[k])
            elif op == 0x6A:  # LONG_BINGET
                var k = self._u32(pos)
                pos += 4
                stack.append(memo[k])
            elif op == 0x85 or op == 0x86 or op == 0x87:  # TUPLE1/2/3
                var cnt = op - 0x84
                var v = PVal(PV_TUPLE)
                for k in range(cnt):
                    v.kids.append(stack[len(stack) - cnt + k])
                for _ in range(cnt):
                    _ = stack.pop()
                stack.append(self._push(v^))
            elif op == 0x74:  # TUPLE (to MARK)
                var m = self._find_mark(stack)
                var v = PVal(PV_TUPLE)
                for k in range(m + 1, len(stack)):
                    v.kids.append(stack[k])
                while len(stack) > m:
                    _ = stack.pop()
                stack.append(self._push(v^))
            elif op == 0x75:  # SETITEMS (to MARK) into the dict below the mark
                var m = self._find_mark(stack)
                var d = stack[m - 1]
                var k = m + 1
                while k + 1 < len(stack):
                    self.nodes[d].kids.append(stack[k])
                    self.nodes[d].kids.append(stack[k + 1])
                    k += 2
                while len(stack) > m:
                    _ = stack.pop()
            elif op == 0x73:  # SETITEM
                var val = stack.pop()
                var key = stack.pop()
                var d = stack[len(stack) - 1]
                self.nodes[d].kids.append(key)
                self.nodes[d].kids.append(val)
            elif op == 0x93:  # STACK_GLOBAL: name, module on the stack
                var name = stack.pop()
                var mod = stack.pop()
                var v = PVal(PV_GLOBAL)
                v.s = self.nodes[mod].s + "." + self.nodes[name].s
                stack.append(self._push(v^))
            elif op == 0x81:  # NEWOBJ: cls, args -> object
                var args = stack.pop()
                var cls = stack.pop()
                var v = PVal(PV_OBJ)
                v.s = self.nodes[cls].s
                v.kids.append(cls)
                v.kids.append(args)
                stack.append(self._push(v^))
            elif op == 0x52:  # REDUCE: callable(args) -> object (numpy.dtype here)
                var args = stack.pop()
                var callee = stack.pop()
                var v = PVal(PV_OBJ)
                v.s = self.nodes[callee].s
                v.kids.append(callee)
                v.kids.append(args)
                if self.nodes[callee].s == "numpy.dtype":
                    # args = (dtype string, align, copy)
                    var a0 = self.nodes[args].kids[0]
                    v.s = "numpy.dtype:" + self.nodes[a0].s
                stack.append(self._push(v^))
            elif op == 0x62:  # BUILD: state -> the object below
                var state = stack.pop()
                var obj = stack[len(stack) - 1]
                self.nodes[obj].kids.append(state)
                if self.nodes[obj].s == "joblib.numpy_pickle.NumpyArrayWrapper":
                    pos = self._read_array(obj, state, pos)
                    var arr = self.nodes[obj].copy()
                    _ = stack.pop()
                    stack.append(self._push(arr^))
            else:
                raise Error(
                    "pickle: unsupported opcode 0x" + hex(op) + " at byte "
                    + String(pos - 1)
                )
        raise Error("pickle: ran off the end without STOP")

    def _find_mark(self, stack: List[Int]) raises -> Int:
        var k = len(stack) - 1
        while k >= 0:
            if self.nodes[stack[k]].kind == PV_MARK:
                return k
            k -= 1
        raise Error("pickle: no MARK on the stack")

    def _dict_get(self, d: Int, key: String) -> Int:
        var kids = self.nodes[d].kids.copy()
        var k = 0
        while k + 1 < len(kids):
            if self.nodes[kids[k]].kind == PV_STR and self.nodes[kids[k]].s == key:
                return kids[k + 1]
            k += 2
        return -1

    def _read_array(mut self, obj: Int, state: Int, pos_in: Int) raises -> Int:
        """Turn the wrapper into an array node: shape and dtype from the state
        dict, then the spliced bytes (padding byte, padding, data)."""
        var pos = pos_in
        var shape_id = self._dict_get(state, "shape")
        var dtype_id = self._dict_get(state, "dtype")
        var align_id = self._dict_get(state, "numpy_array_alignment_bytes")
        if shape_id < 0 or dtype_id < 0:
            raise Error("pickle: NumpyArrayWrapper without shape/dtype at byte " + String(pos))
        var shape = List[Int]()
        var count = 1
        for k in self.nodes[shape_id].kids:
            var dim = self.nodes[k].i
            shape.append(dim)
            count *= dim
        var dname = self.nodes[dtype_id].s
        if dname.startswith("numpy.dtype:"):
            var tail = String(dname[byte=12:])
            dname = tail
        var itemsize = _dtype_itemsize(dname)
        if align_id >= 0 and self.nodes[align_id].kind == PV_INT:
            var pad = self._u8(pos)
            pos += 1 + pad
        var nbytes = count * itemsize
        if pos + nbytes > len(self.buf):
            raise Error("pickle: array bytes run past the file at byte " + String(pos))
        self.nodes[obj].kind = PV_ARRAY
        self.nodes[obj].s = dname
        self.nodes[obj].shape = shape^
        self.nodes[obj].off = pos
        self.nodes[obj].nbytes = nbytes
        return pos + nbytes

    # ── the reading surface ─────────────────────────────────────────────────
    def dict_keys(self, d: Int) -> List[String]:
        var out = List[String]()
        var kids = self.nodes[d].kids.copy()
        var k = 0
        while k + 1 < len(kids):
            out.append(self.nodes[kids[k]].s.copy())
            k += 2
        return out^

    def get(self, d: Int, key: String) raises -> Int:
        var v = self._dict_get(d, key)
        if v < 0:
            raise Error("pickle: no key '" + key + "'")
        return v

    def kind(self, node: Int) -> Int:
        return self.nodes[node].kind

    def int_of(self, node: Int) raises -> Int:
        if self.nodes[node].kind != PV_INT and self.nodes[node].kind != PV_BOOL:
            raise Error("pickle: not an int")
        return self.nodes[node].i

    def shape_of(self, node: Int) raises -> List[Int]:
        if self.nodes[node].kind != PV_ARRAY:
            raise Error("pickle: not an array")
        return self.nodes[node].shape.copy()

    def dtype_of(self, node: Int) raises -> String:
        if self.nodes[node].kind != PV_ARRAY:
            raise Error("pickle: not an array")
        return self.nodes[node].s.copy()

    def array_f32(self, node: Int) raises -> List[Float32]:
        """Materialise a little-endian `f4` array, C order, flat."""
        if self.nodes[node].kind != PV_ARRAY:
            raise Error("pickle: not an array")
        var dn = self.nodes[node].s
        if not (dn == "f4" or dn == "<f4"):
            raise Error("pickle: array is '" + dn + "', not f4")
        var n = self.nodes[node].nbytes // 4
        var out = List[Float32](unsafe_uninit_length=n)
        var p = self.buf.unsafe_ptr() + self.nodes[node].off
        var fp = p.unsafe_bitcast[Float32]()
        for i in range(n):
            out[i] = fp[unsafe_offset=i]
        return out^

    def array_f64(self, node: Int) raises -> List[Float64]:
        """Materialise a little-endian `f8` array, C order, flat."""
        if self.nodes[node].kind != PV_ARRAY:
            raise Error("pickle: not an array")
        var dn = self.nodes[node].s
        if not (dn == "f8" or dn == "<f8"):
            raise Error("pickle: array is '" + dn + "', not f8")
        var n = self.nodes[node].nbytes // 8
        var out = List[Float64](unsafe_uninit_length=n)
        var fp = (self.buf.unsafe_ptr() + self.nodes[node].off).unsafe_bitcast[Float64]()
        for i in range(n):
            out[i] = fp[unsafe_offset=i]
        return out^

    def array_i64(self, node: Int) raises -> List[Int64]:
        if self.nodes[node].kind != PV_ARRAY:
            raise Error("pickle: not an array")
        var dn = self.nodes[node].s
        if not (dn == "i8" or dn == "<i8"):
            raise Error("pickle: array is '" + dn + "', not i8")
        var n = self.nodes[node].nbytes // 8
        var out = List[Int64](unsafe_uninit_length=n)
        var ip = (self.buf.unsafe_ptr() + self.nodes[node].off).unsafe_bitcast[Int64]()
        for i in range(n):
            out[i] = ip[unsafe_offset=i]
        return out^

    def array_i32(self, node: Int) raises -> List[Int32]:
        if self.nodes[node].kind != PV_ARRAY:
            raise Error("pickle: not an array")
        var dn = self.nodes[node].s
        if not (dn == "i4" or dn == "<i4"):
            raise Error("pickle: array is '" + dn + "', not i4")
        var n = self.nodes[node].nbytes // 4
        var out = List[Int32](unsafe_uninit_length=n)
        var ip = (self.buf.unsafe_ptr() + self.nodes[node].off).unsafe_bitcast[Int32]()
        for i in range(n):
            out[i] = ip[unsafe_offset=i]
        return out^
