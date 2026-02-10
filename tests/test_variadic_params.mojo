from std.builtin.variadics import Variadic
from testing import *


trait Model(TrivialRegisterPassable):
    comptime PARAMS: Int


@fieldwise_init
struct SimpleModel(Model):
    comptime PARAMS = 2


@fieldwise_init
struct ComplexModel(Model):
    comptime PARAMS = 4


@fieldwise_init
struct VariableModel[params: Int](Model):
    comptime PARAMS = Self.params


@fieldwise_init
struct SequentialModel[*MODELS: Model]:
    comptime model_types = Variadic.types[T=Model, *Self.MODELS]
    comptime N = Variadic.size(Self.model_types)

    @staticmethod
    fn _sum_params() -> Int:
        var total = 0
        @parameter
        for i in range(Self.N):
            total += Self.model_types[i].PARAMS
        return total

    comptime PARAMS: Int = Self._sum_params()


fn main() raises:
    comptime M1 = SequentialModel[SimpleModel, ComplexModel, ComplexModel]
    print("PARAMS:", M1.PARAMS)  # 10

    comptime M2 = SequentialModel[SimpleModel, VariableModel[8]]
    print("PARAMS:", M2.PARAMS)  # 10

    comptime M3 = SequentialModel[SimpleModel]
    print("PARAMS:", M3.PARAMS)  # 2
