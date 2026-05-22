comptime Null[T: AnyType] = Optional[UnsafePointer[T, ImmutExternalOrigin]]()


def foo[T: AnyType, o: Origin, //](p: Optional[UnsafePointer[T, o]]):
    pass


def main():
    foo(Null[Int])  # infers origin!
