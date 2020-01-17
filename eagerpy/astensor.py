import sys
from . import PyTorchTensor
from . import TensorFlowTensor
from . import JAXTensor
from . import NumPyTensor
from . import istensor


def _get_module_name(x):
    # splitting is necessary for TensorFlow tensors
    return x.__class__.__module__.split(".")[0]


def astensor(x):
    if istensor(x):
        return x
    # we use the module name instead of isinstance
    # to avoid importing all the frameworks
    module = _get_module_name(x)
    if module == "torch" and isinstance(x, sys.modules[module].Tensor):
        return PyTorchTensor(x)
    if module == "tensorflow" and isinstance(x, sys.modules[module].Tensor):
        return TensorFlowTensor(x)
    if module == "jax" and isinstance(x, sys.modules[module].numpy.ndarray):
        return JAXTensor(x)
    if module == "numpy" and isinstance(x, sys.modules[module].ndarray):
        return NumPyTensor(x)
    raise ValueError(f"Unknown type: {type(x)}")
