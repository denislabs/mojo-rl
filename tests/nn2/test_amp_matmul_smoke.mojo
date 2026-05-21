"""Smoke test: cast_fp32_to_bf16 + cast_bf16_to_fp32 round-trip on CPU."""

from std.memory import alloc

from mojo_rl.nn2.core.amp_matmul import (
    cast_fp32_to_bf16,
    cast_bf16_to_fp32,
    LinearAMPState,
)
from mojo_rl.nn2.constants import DT


def main() raises:
    comptime N = 17  # not a multiple of CPU_SIMD_W — exercise tail loop
    var src: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var bf:  UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin] = alloc[Scalar[DType.bfloat16]](N)
    var dst: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    for k in range(N):
        src[k] = (Scalar[DT](k) - Scalar[DT](8.0)) * Scalar[DT](0.125)

    cast_fp32_to_bf16[target="cpu", N=N](src, bf)
    cast_bf16_to_fp32[target="cpu", N=N](bf, dst)

    # bf16 round-trip is exact for values that fit in bf16's 8-bit mantissa.
    # Our seed values are powers-of-two scaled — they round-trip exactly.
    var ok = True
    for k in range(N):
        if dst[k] != src[k]:
            ok = False
            print("  k=", k, " src=", src[k], " dst=", dst[k])

    # Smoke LinearAMPState lazy-grow.
    var amp = LinearAMPState[8, 4].make()
    amp.ensure_cpu(3)
    var amp_ok = (
        len(amp.w_bf16_cpu) == 32
        and len(amp.in_bf16_cpu) == 24
        and len(amp.ou_bf16_cpu) == 12
        and amp.batch_cap == 3
        and amp.w_dirty
    )
    # Grow batch; weight buffer stays.
    amp.ensure_cpu(5)
    var grow_ok = (
        len(amp.w_bf16_cpu) == 32
        and len(amp.in_bf16_cpu) == 40
        and len(amp.ou_bf16_cpu) == 20
        and amp.batch_cap == 5
    )

    if ok and amp_ok and grow_ok:
        print("PASS — bf16 round-trip exact + LinearAMPState grow works.")
    else:
        print("FAIL — ok=", ok, " amp_ok=", amp_ok, " grow_ok=", grow_ok)
        raise Error("amp_matmul_smoke failed")

    src.free()
    bf.free()
    dst.free()
