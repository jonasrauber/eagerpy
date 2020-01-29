from typing import overload
from typing_extensions import Literal

from . import PyTorchTensor
from . import TensorFlowTensor
from . import JAXTensor
from . import NumPyTensor

from . import modules


@overload
def get_dummy(framework: Literal["pytorch"]) -> PyTorchTensor:
    ...


@overload
def get_dummy(framework: Literal["tensorflow"]) -> TensorFlowTensor:
    ...


@overload
def get_dummy(framework: Literal["jax"]) -> JAXTensor:
    ...


@overload
def get_dummy(framework: Literal["numpy"]) -> NumPyTensor:
    ...


def get_dummy(framework):
    if framework == "pytorch":
        x = modules.torch.zeros(0)
        assert isinstance(x, PyTorchTensor)
    elif framework == "pytorch-gpu":
        x = modules.torch.zeros(0, device="cuda:0")  # pragma: no cover
        assert isinstance(x, PyTorchTensor)  # pragma: no cover
    elif framework == "tensorflow":
        x = modules.tensorflow.zeros(0)
        assert isinstance(x, TensorFlowTensor)
    elif framework == "jax":
        x = modules.jax.numpy.zeros(0)
        assert isinstance(x, JAXTensor)
    elif framework == "numpy":
        x = modules.numpy.zeros(0)
        assert isinstance(x, NumPyTensor)
    else:
        raise ValueError(f"unknown framework: {framework}")  # pragma: no cover
    return x.float32()
