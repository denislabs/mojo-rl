# +--------------------------------------------------------------------------+ #
# | Read the stdout of a child process, without Python
# +--------------------------------------------------------------------------+ #
"""A thin `popen` / `fread` / `pclose` wrapper.

Mojo's standard library has no subprocess API, and the alternative — driving a
child through Mojo's embedded CPython — is what `nn/datasets/lewm_pusht.mojo`
documents as leaking every large `PythonObject` until the enclosing function
returns. `popen` is three libc calls with a stable ABI.

    var p = Pipe(quote_arg(path) ... )
    while True:
        var n = p.read_into(buf, count)
        ...
    p.close()          # raises on a non-zero child exit status

⚠ **`popen` RUNS THE STRING THROUGH `/bin/sh`.** Every interpolated path must
go through `quote_arg`, which single-quotes it and REJECTS an embedded single
quote or newline outright. Inside POSIX single quotes every byte is literal
except `'` itself, so rejecting that one character makes the quoting total.
Building a command with bare interpolation would let a filename execute
arbitrary shell — and dataset paths come from a downloaded repo.

⚠ **`pclose` RETURNS A WAIT STATUS, NOT AN EXIT CODE.** A child that exits 1
yields 256. Ignoring it is how a truncated decode reads as a short file rather
than an error, so `close()` decodes the status and raises with the command in
the message.

⚠ **CLOSING EARLY KILLS THE CHILD WITH SIGPIPE, AND THAT IS NOT A FAILURE.**
`pclose` closes the read end; the child's next write gets `SIGPIPE`, whose
default action is to terminate, and the wait status then reports signal 13.
A caller that deliberately stops reading — the video decoder abandoning a file
once it has the frames it needs — passes `allow_sigpipe=True`. Treating that
as an error would make every mid-file rollover raise; treating EVERY signal as
fine would hide an `ffmpeg` that actually crashed, so it is exactly signal 13
that is excused, and only when asked.
"""

from std.ffi import external_call


comptime SIGPIPE = 13


def quote_arg(s: String) raises -> String:
    """Single-quote `s` for `/bin/sh`, refusing what cannot be quoted."""
    for i in range(s.byte_length()):
        var c = Int(s.as_bytes()[i])
        if c == 0x27:
            raise Error(
                "proc: refusing to shell-quote a path containing a single"
                " quote: " + s
            )
        if c == 0x0A or c == 0x0D or c == 0:
            raise Error(
                "proc: refusing to shell-quote a path containing a newline or"
                " NUL byte"
            )
    return "'" + s + "'"


struct Pipe(Movable):
    """A running child process whose stdout is being read.

    Its stderr is inherited, so a failing `ffmpeg` prints where it can be seen
    rather than into a buffer nobody reads.
    """

    var _fp: Int
    """`FILE*` as an address. Zero once closed."""
    var command: String
    var closed: Bool

    def __init__(out self, var command: String) raises:
        var mode = String("r")
        var fp = external_call["popen", Int](
            command.as_c_string_slice().unsafe_ptr(),
            mode.as_c_string_slice().unsafe_ptr(),
        )
        if fp == 0:
            raise Error("proc: popen failed for: " + command)
        self._fp = fp
        self.command = command^
        self.closed = False

    def __init__(out self, *, deinit move: Self):
        self._fp = move._fp
        self.command = move.command^
        self.closed = move.closed

    def __deinit__(deinit self):
        # Best effort: a Pipe dropped without `close()` still must not leak the
        # child. Errors are unreportable here, so `close()` stays the way to
        # learn the exit status.
        if not self.closed and self._fp != 0:
            _ = external_call["pclose", Int32](self._fp)

    def read_into(
        mut self, dst: Pointer[Scalar[DType.uint8], MutAnyOrigin], count: Int
    ) raises -> Int:
        """Read up to `count` bytes. Returns the count read; 0 means EOF.

        ⚠ A PIPE READ IS SHORT WHENEVER IT FEELS LIKE IT. `fread` on a pipe
        already loops internally until it has the full request or hits EOF, so
        a short return here really is the end of the stream — but callers still
        have to treat "fewer bytes than a frame" as EOF rather than as an
        error, which is why this returns the count instead of raising.
        """
        if self.closed:
            raise Error("proc: read from a closed pipe")
        if count <= 0:
            return 0
        return external_call["fread", Int](dst, Int(1), count, self._fp)

    def close(mut self, allow_sigpipe: Bool = False) raises -> Int:
        """Wait for the child and return its exit code. Raises if it failed.

        `allow_sigpipe` excuses signal 13 — see the module docstring. Pass it
        only when this side stopped reading on purpose.
        """
        if self.closed:
            return 0
        var status = Int(external_call["pclose", Int32](self._fp))
        self.closed = True
        self._fp = 0
        if status < 0:
            raise Error("proc: pclose failed for: " + self.command)
        # POSIX wait status: low 7 bits are the signal, next 8 the exit code.
        var sig = status & 0x7F
        if sig != 0 and not (allow_sigpipe and sig == SIGPIPE):
            raise Error(
                "proc: child killed by signal " + String(sig) + ": "
                + self.command
            )
        if sig != 0:
            return 0
        var code = (status >> 8) & 0xFF
        if code != 0:
            raise Error(
                "proc: child exited " + String(code) + ": " + self.command
            )
        return code


def run_capture(var command: String, max_bytes: Int = 1 << 16) raises -> String:
    """Run a command and return its stdout as text. For small outputs only.

    Raises if the output would exceed `max_bytes` rather than returning a
    truncated string — a half-read JSON listing parses as a syntax error at a
    meaningless offset, which is a much worse thing to debug than a cap.
    """
    var p = Pipe(String(command))
    var out = String("")
    var chunk = List[UInt8](unsafe_uninit_length=4096)
    var got = 0
    var overflow = False
    while True:
        var n = p.read_into(
            chunk.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
            .as_unsafe_any_origin(),
            4096,
        )
        if n <= 0:
            break
        if got + n > max_bytes:
            overflow = True
            break
        for i in range(n):
            out += chr(Int(chunk[i]))
        got += n
    _ = p.close(overflow)
    if overflow:
        raise Error(
            "proc: output exceeded " + String(max_bytes) + " bytes: " + command
        )
    return out^
