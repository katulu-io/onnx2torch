import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


@pytest.mark.parametrize(
    'op_type',
    ('Add', 'Sub', 'Mul', 'Div'),
)
def test_math_binary_operation(op_type: str) -> None:  # pylint: disable=missing-function-docstring
    input_shape = [10, 3, 128, 128]
    x = np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32)
    y_variants = [
        np.random.uniform(low=-1.0, high=1.0, size=1).astype(np.float32),
        np.random.uniform(low=-1.0, high=1.0, size=[1] * len(input_shape)).astype(np.float32),
        np.random.uniform(low=-1.0, high=1.0, size=input_shape).astype(np.float32),
        np.array([0.0], dtype=np.float32),
    ]
    for y in y_variants:
        test_inputs = {'x': x, 'y': y}
        initializers = {}
        node = onnx.helper.make_node(
            op_type=op_type,
            inputs=['x', 'y'],
            outputs=['z'],
        )

        model = make_model_from_nodes(nodes=node, initializers=initializers, inputs_example=test_inputs)
        check_onnx_model(model, test_inputs)


@pytest.mark.parametrize(
    'x, y',
    [
        (1, 2),
        (1, 5),
        (5, 30),
        (-1, 2),
        (-1, 5),
        (5, -30),
        (5, 2),
        (-5, 2),
    ],
)
def test_div_operation(x: int, y: int) -> None:  # pylint: disable=missing-function-docstring
    x_ = np.array(x, dtype=np.int64)  # pylint: disable=invalid-name
    y_ = np.array(y, dtype=np.int64)  # pylint: disable=invalid-name
    test_inputs = {'x': x_, 'y': y_}

    node = onnx.helper.make_node(
        op_type='Div',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    model = make_model_from_nodes(nodes=node, initializers={}, inputs_example=test_inputs)
    check_onnx_model(model, test_inputs)
