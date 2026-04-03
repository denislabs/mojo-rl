"""Test parametric dtype: verify forward pass works with explicit DType.float16."""
from mojo_rl.nn.model import ReLU
from mojo_rl.nn.model.linear_act import LinearReLU
from layout import LayoutTensor, Layout
from std.memory import alloc, memset


def test_relu_float16():
    """Test ReLU forward with explicit float16 dtype."""
    comptime BATCH = 2
    comptime DIM = 4

    var ip = alloc[Scalar[DType.float16]](BATCH * DIM)
    var op = alloc[Scalar[DType.float16]](BATCH * DIM)
    var pp = alloc[Scalar[DType.float16]](0)  # ReLU has no params
    memset(op, 0, BATCH * DIM)

    # Set input: [-1, 2, -3, 4, 5, -6, 7, -8]
    ip[0] = Scalar[DType.float16](-1.0)
    ip[1] = Scalar[DType.float16](2.0)
    ip[2] = Scalar[DType.float16](-3.0)
    ip[3] = Scalar[DType.float16](4.0)
    ip[4] = Scalar[DType.float16](5.0)
    ip[5] = Scalar[DType.float16](-6.0)
    ip[6] = Scalar[DType.float16](7.0)
    ip[7] = Scalar[DType.float16](-8.0)

    var input = rebind[LayoutTensor[DType.float16, Layout.row_major(BATCH, DIM), MutAnyOrigin]](
        LayoutTensor[DType.float16, Layout.row_major(BATCH, DIM)](ip)
    )
    var output = rebind[LayoutTensor[DType.float16, Layout.row_major(BATCH, DIM), MutAnyOrigin]](
        LayoutTensor[DType.float16, Layout.row_major(BATCH, DIM)](op)
    )
    var params = rebind[LayoutTensor[DType.float16, Layout.row_major(0), MutAnyOrigin]](
        LayoutTensor[DType.float16, Layout.row_major(0)](pp)
    )

    # Call forward with explicit float16
    ReLU[DIM].forward[BATCH, DType.float16](input, output, params)

    # Verify: ReLU([-1, 2, -3, 4]) = [0, 2, 0, 4]
    var pass_count = 0
    if rebind[Scalar[DType.float16]](output[0, 0]) == Scalar[DType.float16](0.0):
        pass_count += 1
    if rebind[Scalar[DType.float16]](output[0, 1]) == Scalar[DType.float16](2.0):
        pass_count += 1
    if rebind[Scalar[DType.float16]](output[0, 2]) == Scalar[DType.float16](0.0):
        pass_count += 1
    if rebind[Scalar[DType.float16]](output[0, 3]) == Scalar[DType.float16](4.0):
        pass_count += 1

    if pass_count == 4:
        print("PASS: ReLU float16 forward")
    else:
        print("FAIL: ReLU float16 forward, pass_count =", pass_count)

    ip.free()
    op.free()
    pp.free()


def main():
    test_relu_float16()
    print("Parametric dtype test complete")
