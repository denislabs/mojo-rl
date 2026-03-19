"""Tests for Identity model and CompositeParams helper."""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Linear,
    LinearReLU,
    Identity,
)
from mojo_rl.nn.autodiff.composite_params import CompositeParams
from mojo_rl.nn.autodiff.compute_graph import ComputeGraph, GNode
from mojo_rl.nn.initializer import Xavier
from layout import Layout, LayoutTensor
from std.math import abs


fn test_identity() raises:
    """Identity model passes input through unchanged."""
    print("Test 1: Identity model...")

    comptime BATCH = 2
    comptime DIM = 5
    comptime M = Identity[DIM]

    # Verify compile-time constants
    if M.IN_DIM != DIM or M.OUT_DIM != DIM:
        raise Error("Dimension mismatch")
    if M.PARAM_SIZE != 0 or M.CACHE_SIZE != 0:
        raise Error("Identity should have no params or cache")

    var input_arr = InlineArray[Scalar[dtype], BATCH * DIM](
        uninitialized=True
    )
    for i in range(BATCH * DIM):
        input_arr[i] = Scalar[dtype](Float64(i + 1) * 0.1)
    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](input_arr.unsafe_ptr())

    var output_arr = InlineArray[Scalar[dtype], BATCH * DIM](
        uninitialized=True
    )
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](output_arr.unsafe_ptr())

    var params = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var params_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())

    M.forward[BATCH](input_t, output_t, params_t)

    for i in range(BATCH * DIM):
        if abs(Float64(output_arr[i] - input_arr[i])) > 1e-10:
            raise Error("Identity forward failed")

    # Backward: grad_input should equal grad_output
    var grad_out = InlineArray[Scalar[dtype], BATCH * DIM](
        uninitialized=True
    )
    for i in range(BATCH * DIM):
        grad_out[i] = Scalar[dtype](Float64(i) * 0.2)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_out.unsafe_ptr())

    var grad_in = InlineArray[Scalar[dtype], BATCH * DIM](
        uninitialized=True
    )
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_in.unsafe_ptr())

    var cache = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache.unsafe_ptr())
    var grads = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](grads.unsafe_ptr())

    M.backward[BATCH](grad_out_t, grad_in_t, params_t, cache_t, grads_t)

    for i in range(BATCH * DIM):
        if abs(Float64(grad_in[i] - grad_out[i])) > 1e-10:
            raise Error("Identity backward failed")

    print("  PASSED")


fn test_identity_in_compute_graph() raises:
    """Identity used as concat node in ComputeGraph."""
    print("Test 2: Identity as concat in ComputeGraph...")

    comptime BATCH = 1

    # 2-way fan-out, concat with Identity
    comptime G = ComputeGraph[
        GNode["branch_a", Linear[3, 2]],           # 0: input → a(2)
        GNode["branch_b", Linear[3, 4]],           # 1: input → b(4)  (fan-out)
        GNode["output",   Identity[6], "branch_a", "branch_b"],  # 2: concat(a, b) = (6) ← clean!
    ]

    if G.OUT_DIM != 6:
        raise Error("OUT_DIM should be 6")

    var params = InlineArray[Scalar[dtype], G.PARAM_SIZE](
        uninitialized=True
    )
    var params_t = LayoutTensor[
        dtype, Layout.row_major(G.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    G.initialize_params[Xavier[]](params_t)

    var input_arr = InlineArray[Scalar[dtype], BATCH * 3](
        uninitialized=True
    )
    input_arr[0] = Scalar[dtype](0.5)
    input_arr[1] = Scalar[dtype](-0.3)
    input_arr[2] = Scalar[dtype](0.8)
    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3), MutAnyOrigin
    ](input_arr.unsafe_ptr())

    var output_arr = InlineArray[Scalar[dtype], BATCH * 6](
        uninitialized=True
    )
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 6), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_arr = InlineArray[Scalar[dtype], BATCH * G.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, G.CACHE_SIZE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())

    G.forward[BATCH](input_t, output_t, params_t, cache_t)
    print("  Output:", output_arr[0], output_arr[1], "...", output_arr[5])

    # Gradient check
    var grad_out = InlineArray[Scalar[dtype], BATCH * 6](
        uninitialized=True
    )
    for i in range(6):
        grad_out[i] = Scalar[dtype](1.0)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 6), MutAnyOrigin
    ](grad_out.unsafe_ptr())

    var grad_in = InlineArray[Scalar[dtype], BATCH * 3](
        uninitialized=True
    )
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3), MutAnyOrigin
    ](grad_in.unsafe_ptr())
    var grads_arr = InlineArray[Scalar[dtype], G.PARAM_SIZE](
        uninitialized=True
    )
    for i in range(G.PARAM_SIZE):
        grads_arr[i] = Scalar[dtype](0.0)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(G.PARAM_SIZE), MutAnyOrigin
    ](grads_arr.unsafe_ptr())

    G.backward[BATCH](
        grad_out_t, grad_in_t, params_t, cache_t, grads_t
    )

    # Grad_input should be non-zero (both branches contribute)
    var any_nz = False
    for i in range(3):
        if abs(Float64(grad_in[i])) > 1e-10:
            any_nz = True
    if not any_nz:
        raise Error("grad_input is all zeros")

    print("  PASSED")


fn test_composite_params() raises:
    """CompositeParams assembly and scatter."""
    print("Test 3: CompositeParams...")

    comptime ActorModel = Sequential[LinearReLU[4, 8], Linear[8, 2]]
    comptime CriticModel = Sequential[LinearReLU[6, 8], Linear[8, 1]]

    # Two-model composition (DDPG-like)
    comptime P2 = CompositeParams[ActorModel, CriticModel]
    print(
        "  2-model: TOTAL=", P2.TOTAL_SIZE,
        "offset[0]=", P2.offset[0](),
        "offset[1]=", P2.offset[1](),
    )

    # Three-model composition (SAC-like: actor + critic1 + critic2)
    comptime P3 = CompositeParams[ActorModel, CriticModel, CriticModel]
    print(
        "  3-model: TOTAL=", P3.TOTAL_SIZE,
        "offset[0]=", P3.offset[0](),
        "offset[1]=", P3.offset[1](),
        "offset[2]=", P3.offset[2](),
    )

    # Verify alignment
    if P3.offset[1]() % 4 != 0:
        raise Error("offset[1] not aligned")
    if P3.offset[2]() % 4 != 0:
        raise Error("offset[2] not aligned")

    # Test assemble + scatter roundtrip
    var actor_params = InlineArray[Scalar[dtype], ActorModel.PARAM_SIZE](
        uninitialized=True
    )
    for i in range(ActorModel.PARAM_SIZE):
        actor_params[i] = Scalar[dtype](Float64(i) * 0.01)

    var critic_params = InlineArray[Scalar[dtype], CriticModel.PARAM_SIZE](
        uninitialized=True
    )
    for i in range(CriticModel.PARAM_SIZE):
        critic_params[i] = Scalar[dtype](Float64(i + 100) * 0.01)

    # Assemble
    var combined = InlineArray[Scalar[dtype], P2.TOTAL_SIZE](
        uninitialized=True
    )
    P2.assemble(
        combined.unsafe_ptr(),
        actor_params.unsafe_ptr(),
        critic_params.unsafe_ptr(),
    )

    # Verify assembly
    for i in range(ActorModel.PARAM_SIZE):
        if abs(Float64(combined[i] - actor_params[i])) > 1e-10:
            raise Error("Assembly failed for actor params")

    var c_off = P2.offset[1]()
    for i in range(CriticModel.PARAM_SIZE):
        if abs(Float64(combined[c_off + i] - critic_params[i])) > 1e-10:
            raise Error("Assembly failed for critic params")

    # Scatter
    var actor_out = InlineArray[Scalar[dtype], ActorModel.PARAM_SIZE](
        uninitialized=True
    )
    var critic_out = InlineArray[Scalar[dtype], CriticModel.PARAM_SIZE](
        uninitialized=True
    )
    P2.scatter(
        combined.unsafe_ptr(),
        actor_out.unsafe_ptr(),
        critic_out.unsafe_ptr(),
    )

    for i in range(ActorModel.PARAM_SIZE):
        if abs(Float64(actor_out[i] - actor_params[i])) > 1e-10:
            raise Error("Scatter failed for actor")
    for i in range(CriticModel.PARAM_SIZE):
        if abs(Float64(critic_out[i] - critic_params[i])) > 1e-10:
            raise Error("Scatter failed for critic")

    print("  PASSED")


fn main() raises:
    print("=" * 60)
    print("Identity + CompositeParams Tests")
    print("=" * 60)

    test_identity()
    test_identity_in_compute_graph()
    test_composite_params()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
