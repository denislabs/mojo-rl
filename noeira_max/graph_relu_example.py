import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops

# Build the graph
cpu = CPU()
input_type = TensorType(DType.float32, shape=[5], device=cpu)

with Graph("relu_graph", input_types=[input_type]) as graph:
    x = graph.inputs[0]
    graph.output(ops.relu(x))

# Compile the graph
session = InferenceSession(devices=[cpu])
compiled = session.compile(graph)
model = session.init(compiled)

# Run the computation and inspect results
input_data = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
result = model.execute(input_data)
print(np.from_dlpack(result[0]))