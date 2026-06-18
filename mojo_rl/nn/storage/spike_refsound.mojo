"""Does a Pointer[Tensor, o] pack (origin-threaded) survive a call?"""
from std.memory import Pointer
from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.tensor import Tensor


@fieldwise_init
struct Refs[is_mut: Bool, //, N: Int, o: Origin[mut=is_mut]](Copyable, Movable):
    var ptrs: InlineArray[Pointer[Tensor, Self.o], Self.N]

    @staticmethod
    def of1(ref [Self.o] t0: Tensor) -> Self:
        return Self(
            InlineArray[Pointer[Tensor, Self.o], Self.N](fill=Pointer(to=t0))
        )

    def __getitem__(self, i: Int) -> ref [Self.o] Tensor:
        return self.ptrs[i][]


def thru[o: Origin](r: Refs[1, o]) -> Scalar[DT]:
    ref t = r[0]
    return t.data[0] + t.data[3]   # expect 1 + 4 = 5


def main() raises:
    var x = Tensor.alloc(6)
    for i in range(6):
        x.data[i] = Scalar[DT](i + 1)
    print("thru Pointer pack:", thru(Refs[1].of1(x)))
