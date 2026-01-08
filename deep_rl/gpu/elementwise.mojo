"""GPU elementwise operations for deep RL.

This module provides GPU-accelerated elementwise operations using the
algorithm.functional.elementwise pattern from Mojo GPU puzzles (P23).

Operations:
- gpu_add: output = a + b
- gpu_mul: output = a * b
- gpu_scale: output = input * scalar
- gpu_relu: output = max(input, 0)
- gpu_tanh: output = tanh(input)
- gpu_sigmoid: output = 1 / (1 + exp(-input))
"""

from gpu.host import DeviceContext
from gpu.host.compile import get_gpu_target
from layout import Layout, LayoutTensor
from utils import IndexList
from algorithm.functional import elementwise
from sys import simd_width_of, align_of
from builtin.math import max as builtin_max
from math import tanh, exp


fn gpu_add[
    dtype: DType,
    layout: Layout,
    size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Element-wise addition on GPU: output = a + b.

    Args:
        output: Output tensor (mutable).
        a: First input tensor.
        b: Second input tensor.
        ctx: GPU device context.
    """
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    comptime rank = 1

    @parameter
    @always_inline
    fn add_kernel[
        sw: Int, r: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[r]) capturing -> None:
        idx = indices[0]
        a_vec = a.aligned_load[width=sw](idx, 0)
        b_vec = b.aligned_load[width=sw](idx, 0)
        output.aligned_store[sw](idx, 0, a_vec + b_vec)

    elementwise[add_kernel, simd_width, target="gpu"](size, ctx)


fn gpu_mul[
    dtype: DType,
    layout: Layout,
    size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Element-wise multiplication on GPU: output = a * b.

    Args:
        output: Output tensor (mutable).
        a: First input tensor.
        b: Second input tensor.
        ctx: GPU device context.
    """
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()

    @parameter
    @always_inline
    fn mul_kernel[
        sw: Int, r: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[r]) capturing -> None:
        idx = indices[0]
        a_vec = a.aligned_load[width=sw](idx, 0)
        b_vec = b.aligned_load[width=sw](idx, 0)
        output.aligned_store[sw](idx, 0, a_vec * b_vec)

    elementwise[mul_kernel, simd_width, target="gpu"](size, ctx)


fn gpu_scale[
    dtype: DType,
    layout: Layout,
    size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutAnyOrigin],
    input: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    scale: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    """Element-wise scaling on GPU: output = input * scale.

    Args:
        output: Output tensor (mutable).
        input: Input tensor.
        scale: Scalar multiplier.
        ctx: GPU device context.
    """
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()

    @parameter
    @always_inline
    fn scale_kernel[
        sw: Int, r: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[r]) capturing -> None:
        idx = indices[0]
        x = input.aligned_load[width=sw](idx, 0)
        scale_vec = SIMD[dtype, sw](scale)
        output.aligned_store[sw](idx, 0, x * scale_vec)

    elementwise[scale_kernel, simd_width, target="gpu"](size, ctx)


fn gpu_relu[
    dtype: DType,
    layout: Layout,
    size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutAnyOrigin],
    input: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Element-wise ReLU on GPU: output = max(input, 0).

    Args:
        output: Output tensor (mutable).
        input: Input tensor.
        ctx: GPU device context.
    """
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    comptime rank = 1

    @parameter
    @always_inline
    fn relu_kernel[
        sw: Int, r: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[r]) capturing -> None:
        idx = indices[0]
        x = input.aligned_load[width=sw](idx, 0)
        # ReLU: max(x, 0)
        zero = SIMD[dtype, sw](0)
        result = builtin_max(x, zero)
        output.aligned_store[sw](idx, 0, result)

    elementwise[relu_kernel, simd_width, target="gpu"](size, ctx)


fn gpu_tanh[
    dtype: DType,
    layout: Layout,
    size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutAnyOrigin],
    input: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Element-wise tanh on GPU: output = tanh(input).

    Args:
        output: Output tensor (mutable).
        input: Input tensor.
        ctx: GPU device context.
    """
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()

    @parameter
    @always_inline
    fn tanh_kernel[
        sw: Int, r: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[r]) capturing -> None:
        idx = indices[0]
        x = input.aligned_load[width=sw](idx, 0)
        # tanh using math library
        result = tanh(x)
        output.aligned_store[sw](idx, 0, result)

    elementwise[tanh_kernel, simd_width, target="gpu"](size, ctx)


fn gpu_sigmoid[
    dtype: DType,
    layout: Layout,
    size: Int,
](
    output: LayoutTensor[mut=True, dtype, layout, MutAnyOrigin],
    input: LayoutTensor[mut=False, dtype, layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Element-wise sigmoid on GPU: output = 1 / (1 + exp(-input)).

    Args:
        output: Output tensor (mutable).
        input: Input tensor.
        ctx: GPU device context.
    """
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()

    @parameter
    @always_inline
    fn sigmoid_kernel[
        sw: Int, r: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[r]) capturing -> None:
        idx = indices[0]
        x = input.aligned_load[width=sw](idx, 0)
        # sigmoid: 1 / (1 + exp(-x))
        one = SIMD[dtype, sw](1)
        result = one / (one + exp(-x))
        output.aligned_store[sw](idx, 0, result)

    elementwise[sigmoid_kernel, simd_width, target="gpu"](size, ctx)
