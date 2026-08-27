comptime Null[T: AnyType] = Optional[Pointer[T, ImmutExternalOrigin]]()


def foo[T: AnyType, o: Origin, //](p: Optional[Pointer[T, o]]):
    pass


def main():
    foo(Null[Int])  # infers origin!
