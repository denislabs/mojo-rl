/* mrl_serial.c — the one libc call Mojo's FFI cannot make.
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
 * Built by scripts/build_serial.sh into libmrl_serial.dylib, which is
 * dlopen'd at runtime through `_get_dylib_function` — the same path
 * `render/imgui` uses.
 *
 * ⚠ An earlier design linked mrl_serial.o straight into the binary and
 * resolved the symbol with a hand-rolled dlsym, for ONE self-contained
 * executable. That died on two Mojo constraints: `external_call` collides
 * with the stdlib's own declaration of `dlsym` at LLVM lowering, and
 * `mojo run`'s JIT ignores -Xlinker entirely. See native.mojo.
 */

#include <sys/ioctl.h>

#if defined(__APPLE__)
#include <IOKit/serial/ioss.h>
#endif

/* Set a NON-STANDARD line speed on an already-open tty.
 * Returns 0 on success, -1 with errno set on failure. */
int mrl_serial_set_speed(int fd, unsigned long baud) {
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
