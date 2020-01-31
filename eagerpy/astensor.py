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
    name = _get_module_name(x)
    m = sys.modules
    if name == "torch" and isinstance(x, m[name].Tensor):  # type: ignore
        return PyTorchTensor(x)
    if name == "tensorflow" and isinstance(x, m[name].Tensor):  # type: ignore
        return TensorFlowTensor(x)
    if name == "jax" and isinstance(x, m[name].numpy.ndarray):  # type: ignore
        return JAXTensor(x)
    if name == "numpy" and isinstance(x, m[name].ndarray):  # type: ignore
        return NumPyTensor(x)
    raise ValueError(f"Unknown type: {type(x)}")
