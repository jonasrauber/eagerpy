from typing import TYPE_CHECKING, Union, overload
import sys

from .tensor import Tensor
from .tensor import TensorType

from .tensor import PyTorchTensor
from .tensor import TensorFlowTensor
from .tensor import JAXTensor
from .tensor import NumPyTensor

from .types import NativeTensor

if TYPE_CHECKING:
    # for static analyzers
    import torch


def _get_module_name(x) -> str:
    # splitting is necessary for TensorFlow tensors
    return type(x).__module__.split(".")[0]


@overload
def astensor(x: TensorType) -> TensorType:
    ...


@overload
def astensor(x: "torch.Tensor") -> PyTorchTensor:
    ...


@overload
def astensor(x: NativeTensor) -> Tensor:
    ...


def astensor(x: Union[NativeTensor, Tensor]) -> Tensor:
    if isinstance(x, Tensor):
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
