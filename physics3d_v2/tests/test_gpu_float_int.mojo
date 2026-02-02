"""Test GPU Float-to-Int conversion methods.

This diagnostic test checks different approaches for converting
float values to integers on GPU to find a working solution.
"""

from math import floor
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import Layout, LayoutTensor

comptime DTYPE = DType.float32


fn run_conversion_test(
    ctx: DeviceContext,
    mut input_buf: DeviceBuffer[DTYPE],
    mut output_buf: DeviceBuffer[DType.int32],
) raises:
    """Run the conversion test kernel."""

    var input_tensor = LayoutTensor[
        DTYPE, Layout.row_major(16), MutAnyOrigin
    ](input_buf.unsafe_ptr())

    var output_tensor = LayoutTensor[
        DType.int32, Layout.row_major(16, 8), MutAnyOrigin
    ](output_buf.unsafe_ptr())

    @always_inline
    fn kernel_wrapper(
        input: LayoutTensor[DTYPE, Layout.row_major(16), MutAnyOrigin],
        output: LayoutTensor[DType.int32, Layout.row_major(16, 8), MutAnyOrigin],
    ):
        var idx = Int(thread_idx.x)
        if idx >= 16:
            return

        var f = rebind[Scalar[DTYPE]](input[idx])

        # Method 0: Direct Int() conversion
        var m0 = Int(f)
        output[idx, 0] = Int32(m0)

        # Method 1: Int(f + 0.5) for rounding positive values
        var m1 = Int(f + 0.5)
        output[idx, 1] = Int32(m1)

        # Method 2: Conditional range checks
        var m2: Int = -99
        if f > Scalar[DTYPE](-1.5) and f < Scalar[DTYPE](-0.5):
            m2 = -1
        elif f > Scalar[DTYPE](-0.5) and f < Scalar[DTYPE](0.5):
            m2 = 0
        elif f > Scalar[DTYPE](0.5) and f < Scalar[DTYPE](1.5):
            m2 = 1
        elif f > Scalar[DTYPE](1.5) and f < Scalar[DTYPE](2.5):
            m2 = 2
        elif f > Scalar[DTYPE](2.5) and f < Scalar[DTYPE](3.5):
            m2 = 3
        output[idx, 2] = Int32(m2)

        # Method 3: floor() based
        var m3 = Int(floor(f + Scalar[DTYPE](0.5)))
        output[idx, 3] = Int32(m3)

        # Method 4: Sequential threshold (branchless-ish)
        var m4: Int = -1
        if f >= Scalar[DTYPE](-0.5):
            m4 = 0
        if f >= Scalar[DTYPE](0.5):
            m4 = 1
        if f >= Scalar[DTYPE](1.5):
            m4 = 2
        if f >= Scalar[DTYPE](2.5):
            m4 = 3
        output[idx, 4] = Int32(m4)

        # Method 5: Proper rounding for negative and positive
        var rounded = f + Scalar[DTYPE](0.5)
        if f < Scalar[DTYPE](0):
            rounded = f - Scalar[DTYPE](0.5)
        var m5 = Int(rounded)
        output[idx, 5] = Int32(m5)

        # Method 6: Simple > comparison chain
        var m6: Int = -1
        if f > Scalar[DTYPE](-0.5):
            m6 = 0
        if f > Scalar[DTYPE](0.5):
            m6 = 1
        if f > Scalar[DTYPE](1.5):
            m6 = 2
        if f > Scalar[DTYPE](2.5):
            m6 = 3
        output[idx, 6] = Int32(m6)

        # Method 7: Use >= 0 check then add based on thresholds
        var m7: Int = -1
        if f >= Scalar[DTYPE](-0.1):  # Slightly below 0 to handle -0.0
            m7 = 0
            if f >= Scalar[DTYPE](0.9):
                m7 = 1
            if f >= Scalar[DTYPE](1.9):
                m7 = 2
            if f >= Scalar[DTYPE](2.9):
                m7 = 3
        output[idx, 7] = Int32(m7)

    ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
        input_tensor,
        output_tensor,
        grid_dim=(1,),
        block_dim=(16,),
    )


fn main() raises:
    print("=" * 60)
    print("GPU Float-to-Int Conversion Test")
    print("=" * 60)

    var ctx = DeviceContext()

    # Input values to test
    var input_host = List[Float32](capacity=16)
    input_host.append(-1.0)
    input_host.append(-0.5)
    input_host.append(0.0)
    input_host.append(0.5)
    input_host.append(1.0)
    input_host.append(1.5)
    input_host.append(2.0)
    input_host.append(2.5)
    input_host.append(3.0)
    input_host.append(-0.9)
    input_host.append(-0.1)
    input_host.append(0.1)
    input_host.append(0.9)
    input_host.append(1.1)
    input_host.append(1.9)
    input_host.append(2.1)

    # Expected results (nearest integer, with -0.5 -> -1, 0.5 -> 0)
    var expected = List[Int](capacity=16)
    expected.append(-1)  # -1.0 -> -1
    expected.append(-1)  # -0.5 -> -1 (round towards negative)
    expected.append(0)   # 0.0 -> 0
    expected.append(0)   # 0.5 -> 0 (round down for positive .5)
    expected.append(1)   # 1.0 -> 1
    expected.append(1)   # 1.5 -> 1
    expected.append(2)   # 2.0 -> 2
    expected.append(2)   # 2.5 -> 2
    expected.append(3)   # 3.0 -> 3
    expected.append(-1)  # -0.9 -> -1
    expected.append(0)   # -0.1 -> 0
    expected.append(0)   # 0.1 -> 0
    expected.append(1)   # 0.9 -> 1
    expected.append(1)   # 1.1 -> 1
    expected.append(2)   # 1.9 -> 2
    expected.append(2)   # 2.1 -> 2

    # Allocate GPU buffers
    var input_buf = ctx.enqueue_create_buffer[DTYPE](16)
    var output_buf = ctx.enqueue_create_buffer[DType.int32](16 * 8)

    # Initialize output to -99 to detect untouched values
    var output_host = List[Int32](capacity=16 * 8)
    for _ in range(16 * 8):
        output_host.append(-99)

    # Copy to GPU
    ctx.enqueue_copy(input_buf, input_host.unsafe_ptr())
    ctx.enqueue_copy(output_buf, output_host.unsafe_ptr())
    ctx.synchronize()

    # Run kernel
    run_conversion_test(ctx, input_buf, output_buf)
    ctx.synchronize()

    # Copy back
    ctx.enqueue_copy(output_host.unsafe_ptr(), output_buf)
    ctx.synchronize()

    # Print results
    var method_names = List[String](capacity=8)
    method_names.append("Int(f)")
    method_names.append("Int(f+0.5)")
    method_names.append("Range check")
    method_names.append("floor+0.5")
    method_names.append("Sequential>=")
    method_names.append("Round±0.5")
    method_names.append("Sequential>")
    method_names.append("Nested>=")

    print("\nInput values and conversion results:")
    print("-" * 100)
    print("Input    | Exp |", end="")
    for m in range(8):
        print(" M", m, " |", end="")
    print()
    print("-" * 100)

    for i in range(16):
        if input_host[i] >= 0:
            print(" ", input_host[i], " |", expected[i], " |", end="")
        else:
            print("", input_host[i], " |", expected[i], " |", end="")
        for m in range(8):
            var result = output_host[i * 8 + m]
            if result == Int32(expected[i]):
                print("  ", result, " |", end="")  # Correct
            else:
                print(" ", result, "X |", end="")  # Wrong
        print()

    print("-" * 100)
    print("\nMethod legend:")
    for m in range(8):
        print("  M", m, ":", method_names[m])

    # Count correct results per method (for exact integers: -1, 0, 1, 2, 3)
    print("\nCorrect results for exact integers (-1, 0, 1, 2, 3):")
    var test_indices = List[Int](capacity=5)
    test_indices.append(0)   # -1.0
    test_indices.append(2)   # 0.0
    test_indices.append(4)   # 1.0
    test_indices.append(6)   # 2.0
    test_indices.append(8)   # 3.0

    for m in range(8):
        var correct = 0
        for j in range(5):
            var i = test_indices[j]
            if output_host[i * 8 + m] == Int32(expected[i]):
                correct += 1
        var status = "WORKS!" if correct == 5 else ""
        print("  M", m, "(", method_names[m], "):", correct, "/ 5", status)

    # Count correct for ALL values
    print("\nCorrect results for ALL values (including 0.5 boundaries):")
    for m in range(8):
        var correct = 0
        for i in range(16):
            if output_host[i * 8 + m] == Int32(expected[i]):
                correct += 1
        var status = "PERFECT!" if correct == 16 else ("GOOD" if correct >= 14 else "")
        print("  M", m, "(", method_names[m], "):", correct, "/ 16", status)

    print("\n" + "=" * 60)
