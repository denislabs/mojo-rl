from max import nn
from max.graph import ops, DeviceRef
from max.dtype import DType
from max.graph import Graph, TensorType
from max.engine import InferenceSession
from max.driver import GPU

class FeedForward(nn.Module):
    """Two linear projections with SiLU activation."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, DType.float32, DeviceRef.GPU())
        self.fc2 = nn.Linear(hidden_dim, dim, DType.float32, DeviceRef.GPU())

    def __call__(self, x):
        return self.fc2(ops.silu(self.fc1(x)))

class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.Sequential([
            FeedForward(dim, hidden_dim) for _ in range(n_layers)
        ])

    def __call__(self, x):
        return self.layers(x)

# Instantiate the module
model = FeedForwardBlock(dim=512, hidden_dim=1024, n_layers=4)

# Load weights into the module
model.load_state_dict(my_state_dict)

graph = Graph(
    "my_model",
    forward=model,
    input_types=[TensorType(DType.float32, shape=[1, 512], device=DeviceRef.GPU())],
)

session = InferenceSession(devices=[GPU()])
compiled = session.compile(graph)
runnable_model = session.init(compiled, weights_registry=model.state_dict())
result = runnable_model(input_data)

print(result[0].to_numpy())