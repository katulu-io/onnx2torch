__all__ = [
    'OnnxMatMul',
]

import math

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule, OnnxMapping
from onnx2torch.utils.common import OperationConverterResult

class OnnxMatMul(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    weight: torch.Tensor

    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return torch.matmul(input, self.weight)


@add_converter(operation_type='MatMul', version=1)
@add_converter(operation_type='MatMul', version=9)
@add_converter(operation_type='MatMul', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    a_name = node.input_values[0]
    b_name = node.input_values[1]

    weights = graph.initializers[b_name]
    weights = weights.to_torch()

    torch_module = OnnxMatMul(in_features=weights.shape[1], out_features=weights.shape[0])

    with torch.no_grad():
        torch_module.weight.data = weights

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(inputs=(a_name, ), outputs=node.output_values),
    )
