/* nra_serial.c — the one libc call Mojo's FFI cannot make.
 *
 * `ioctl` is C-variadic. Mojo's `external_call` emits a FIXED prototype, so
 * the third argument lands in a register while the callee's `va_arg` reads
 * the stack (Apple arm64 passes variadic args on the stack). The result is a
 * silent EFAULT, not a compile error. See
 * `docs/SO101_SERIAL_LAYER.md` §"why there is C here at all".
 *
 * Everything else in the serial layer is Mojo. Keep it that way: this file
 * exists for calls that are variadic in C, and for nothing else.
 *
 * Built by scripts/build_serial.sh into libnra_serial.dylib, which is
 * dlopen'd at runtime through `_get_dylib_function` — the same path
 * `render/imgui` uses.
 *
 * ⚠ An earlier design linked nra_serial.o straight into the binary and
 * resolved the symbol with a hand-rolled dlsym, for ONE self-contained
 * executable. That died on two Mojo constraints: `external_call` collides
 * with the stdlib's own declaration of `dlsym` at LLVM lowering, and
 * `mojo run`'s JIT ignores -Xlinker entirely. See native.mojo.
 */

#include <sys/ioctl.h>
#include <stddef.h>
#include <termios.h>
#include <fcntl.h>
#include <errno.h>

#if defined(__APPLE__)
#include <IOKit/serial/ioss.h>
#endif

/* ─── the layout probe ────────────────────────────────────────────────────
 *
 * ⚠⚠ THE HEADERS ARE THE AUTHORITY, NOT A REMEMBERED TABLE.  `port.mojo`
 * hardcodes `struct termios` offsets and a pile of libc constants, and a
 * wrong one is SILENT: a bad OFF_CFLAG writes into c_cc, the port still
 * opens, and the failure surfaces much later as garbled bytes.  Darwin's
 * numbers were verified by an offsetof probe when they were written; this
 * makes that probe a permanent, runnable part of the tree so the Linux side
 * is measured on the machine it runs on rather than recalled.
 *
 * ⚠ THE ORDER IS AN ABI.  `port.mojo` reads these by index and
 * `tests/robot/test_serial_termios_layout.mojo` compares each against its
 * own constant.  APPEND ONLY — inserting a row silently shifts every field
 * after it, which is the exact class of bug this exists to catch.
 */
#define NRA_SERIAL_LAYOUT_N 23

int nra_serial_layout(long *out, int n) {
    if (out == NULL || n < NRA_SERIAL_LAYOUT_N) return -1;
    int i = 0;
    out[i++] = (long)sizeof(struct termios);
    out[i++] = (long)offsetof(struct termios, c_iflag);
    out[i++] = (long)offsetof(struct termios, c_oflag);
    out[i++] = (long)offsetof(struct termios, c_cflag);
    out[i++] = (long)offsetof(struct termios, c_lflag);
    out[i++] = (long)offsetof(struct termios, c_cc);
    out[i++] = (long)sizeof(tcflag_t);
    out[i++] = (long)NCCS;
    out[i++] = (long)VMIN;
    out[i++] = (long)VTIME;
    out[i++] = (long)CSIZE;
    out[i++] = (long)CS8;
    out[i++] = (long)CLOCAL;
    out[i++] = (long)CREAD;
    out[i++] = (long)PARENB;
    out[i++] = (long)CSTOPB;
    out[i++] = (long)CRTSCTS;
    out[i++] = (long)TCSANOW;
    out[i++] = (long)TCIOFLUSH;
    out[i++] = (long)O_NOCTTY;
    out[i++] = (long)O_NONBLOCK;
    out[i++] = (long)EAGAIN;
    out[i++] = (long)AT_FDCWD;
    return i;
}

/* c_ispeed / c_ospeed are NOT in the list above, deliberately: they do not
 * exist on every libc (musl keeps the speed only in c_cflag), so asking for
 * their offset would not compile there.  `port.mojo` cross-checks those two
 * against `cfgetospeed` instead, which is portable and stronger — it is
 * libc reading the same struct its own way. */

/* The Bxxx constant for a numeric line speed, or -1 if this libc has none.
 *
 * ⚠ LINUX Bxxx CONSTANTS ARE SMALL ORDINALS, NOT THE BAUD NUMBERS.
 * B1000000 is 0010010 octal = 4104.  On BSD/Darwin the constant IS the
 * number, so this is the identity there.  `cfsetospeed` REFUSES the numeric
 * value on glibc (EINVAL), and `cfgetospeed` hands the ordinal back — which
 * is why the Mojo side cannot simply compare its readback against the baud
 * it asked for. */
long nra_serial_baud_constant(long baud) {
    switch (baud) {
#define NRA_B(n) case n: return (long)B##n;
    NRA_B(0) NRA_B(50) NRA_B(75) NRA_B(110) NRA_B(134) NRA_B(150)
    NRA_B(200) NRA_B(300) NRA_B(600) NRA_B(1200) NRA_B(1800) NRA_B(2400)
    NRA_B(4800) NRA_B(9600) NRA_B(19200) NRA_B(38400)
#ifdef B57600
    NRA_B(57600)
#endif
#ifdef B115200
    NRA_B(115200)
#endif
#ifdef B230400
    NRA_B(230400)
#endif
#ifdef B460800
    NRA_B(460800)
#endif
#ifdef B500000
    NRA_B(500000)
#endif
#ifdef B576000
    NRA_B(576000)
#endif
#ifdef B921600
    NRA_B(921600)
#endif
#ifdef B1000000
    NRA_B(1000000)
#endif
#ifdef B1152000
    NRA_B(1152000)
#endif
#ifdef B1500000
    NRA_B(1500000)
#endif
#ifdef B2000000
    NRA_B(2000000)
#endif
#ifdef B2500000
    NRA_B(2500000)
#endif
#ifdef B3000000
    NRA_B(3000000)
#endif
#ifdef B3500000
    NRA_B(3500000)
#endif
#ifdef B4000000
    NRA_B(4000000)
#endif
#undef NRA_B
    default: return -1;
    }
}

/* Set a NON-STANDARD line speed on an already-open tty.
 * Returns 0 on success, -1 with errno set on failure. */
int nra_serial_set_speed(int fd, unsigned long baud) {
#if defined(__APPLE__)
    /* Darwin's termios tops out at B230400 and REJECTS a literal 1000000
     * with EINVAL, even though BSD Bxxx constants are the baud numbers
     * themselves. IOSSIOSPEED is the only way up. Measured 2026-08-25. */
    speed_t sp = (speed_t)baud;
    return ioctl(fd, IOSSIOSPEED, &sp);
#else
    /* Linux spells 1 Mbaud B1000000 in termios, so the Mojo side sets it
     * directly and this is a no-op. Rates with no Bxxx constant would need
     * TCSETS2/BOTHER here. */
    (void)fd;
    (void)baud;
    return 0;
#endif
}
