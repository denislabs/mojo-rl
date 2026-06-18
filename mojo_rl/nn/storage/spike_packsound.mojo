"""Is TensorPack's untracked __getitem__ SOUND under store+reload?"""
from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.tensor import Tensor
from mojo_rl.nn.storage.tensor_pack import TensorPack


def main() raises:
    var p = TensorPack[2]()
    p[0].ensure(2)
    p[1].ensure(2)
    p[0].data[0] = Scalar[DT](5)
    p[1].data[0] = Scalar[DT](7)
    print("write-then-read inline:", p[0].data[0], p[1].data[0])  # 5, 7
    var a = p[0].data[0]
    var b = p[1].data[0]
    print("read into vars:        ", a, b)  # 5, 7 ?
