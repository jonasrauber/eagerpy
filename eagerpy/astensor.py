import sys

from .tensor import PyTorchTensor
from .tensor import TensorFlowTensor
from .tensor import JAXTensor
from .tensor import NumPyTensor

from .tensor.base import AbstractTensor


def _get_module_name(x) -> str:
    # splitting is necessary for TensorFlow tensors
    return type(x).__module__.split(".")[0]


def astensor(x) -> AbstractTensor:
    if isinstance(x, AbstractTensor):
        return x
    # we use the module name instead of isinstance
    # to avoid importing all the frameworks
    module = _get_module_name(x)
    if module == "torch" and isinstance(x, sys.modules[module].Tensor):  # type: ignore
        return PyTorchTensor(x)
    if module == "tensorflow" and isinstance(
        x, sys.modules[module].Tensor
    ):  # type: ignore
        return TensorFlowTensor(x)
    if module == "jax" and isinstance(
        x, sys.modules[module].numpy.ndarray
    ):  # type: ignore
        return JAXTensor(x)
    if module == "numpy" and isinstance(x, sys.modules[module].ndarray):  # type: ignore
        return NumPyTensor(x)
    raise ValueError(f"Unknown type: {type(x)}")
