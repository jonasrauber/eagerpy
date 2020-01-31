from typing import TYPE_CHECKING, Union, overload
import sys

from .tensor import PyTorchTensor
from .tensor import TensorFlowTensor
from .tensor import JAXTensor
from .tensor import NumPyTensor

from .tensor import Tensor

from .tensor.base import AbstractTensor


if TYPE_CHECKING:
    # for static analyzers
    import torch  # noqa: F401
    import tensorflow  # noqa: F401
    import jax  # noqa: F401
    import numpy  # noqa: F401

# tensorflow.Tensor, jax.numpy.ndarray and numpy.ndarray currently evaluate to Any
# we can therefore only provide additional type information for torch.Tensor
NativeTensor = Union[
    "torch.Tensor", "tensorflow.Tensor", "jax.numpy.ndarray", "numpy.ndarray"
]


def _get_module_name(x) -> str:
    # splitting is necessary for TensorFlow tensors
    return type(x).__module__.split(".")[0]


@overload
def astensor(x: Tensor) -> Tensor:
    ...


@overload
def astensor(x: "torch.Tensor") -> PyTorchTensor:
    ...


@overload
def astensor(x: NativeTensor) -> AbstractTensor:
    ...


def astensor(x: Union[NativeTensor, AbstractTensor]) -> AbstractTensor:
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
