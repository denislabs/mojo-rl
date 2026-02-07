@fieldwise_init
struct SimpleModel:
    comptime PARAMS = 2


fn sum_params[*models: SimpleModel]() -> Int:
    comptime list = VariadicList(models)
    var params = 0

    @parameter
    for model in list:
        params += model.PARAMS
    return params


struct SequentialModel[*MODELS: SimpleModel]:
    comptime PARAMS = sum_params[*Self.MODELS]()


fn main() raises:
    comptime MODEL = SequentialModel[SimpleModel(), SimpleModel()]
    print(MODEL.PARAMS)
