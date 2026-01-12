# Try without explicit VariadicPack type

from utils import Variant
from memory import ArcPointer


trait Layer(Movable & Copyable):
    fn forward(self, x: Int) -> Int:
        ...


fn forward_trampoline[T: Layer](ptr: ArcPointer[NoneType], x: Int) -> Int:
    var data = rebind[ArcPointer[T]](ptr)
    return data[].forward(x)


fn clear_trampoline[T: Layer](var ptr: ArcPointer[NoneType]):
    var data = rebind[ArcPointer[T]](ptr^)
    _ = data^


struct DynLayer(Layer & Copyable):
    var data: ArcPointer[NoneType]
    var forward_func: fn (ArcPointer[NoneType], Int) -> Int
    var clear_func: fn (var ArcPointer[NoneType])

    fn __init__[T: Layer](out self, var ptr: ArcPointer[T]):
        self.data = rebind[ArcPointer[NoneType]](ptr^)
        self.forward_func = forward_trampoline[T]
        self.clear_func = clear_trampoline[T]

    fn forward(self, x: Int) -> Int:
        return self.forward_func(self.data, x)

    fn __del__(deinit self):
        self.clear_func(self.data^)


struct Linear(Layer):
    var scale: Int

    fn __init__(out self, scale: Int):
        self.scale = scale

    fn forward(self, x: Int) -> Int:
        return x * self.scale


struct ReLU(Layer):
    fn __init__(out self):
        pass

    fn forward(self, x: Int) -> Int:
        return x if x > 0 else 0


struct Sequential[*Types: Layer]:
    var layers: List[DynLayer]

    fn __init__(out self, *layers: * Self.Types):
        self.layers = []

        @parameter
        for i in range(len(VariadicList(Self.Types))):
            var layer = ArcPointer[Self.Types[i]](layers[i].copy())
            self.layers.append(DynLayer(layer))
        # Forward at init time to test
        var result = 10

        @parameter
        for i in range(len(VariadicList(Self.Types))):
            result = layers[i].forward(result)
        print("Init forward(10):", result)

    # Method that takes layers again (not stored)
    fn forward(self, x: Int) -> Int:
        var result = x

        for i in range(len(self.layers)):
            result = self.layers[i].forward(result)
        return result


fn main():
    print("Test...")

    var l1 = Linear(2)
    var relu = ReLU()
    var l2 = Linear(3)

    var seq = Sequential(l1, relu, l2)

    # Must pass layers again for forward (not ideal)
    var result = seq.forward(5)
    print("forward(5):", result)
