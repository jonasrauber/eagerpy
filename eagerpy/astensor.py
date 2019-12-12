from . import PyTorchTensor
from . import TensorFlowTensor
from . import JAXTensor
from . import NumPyTensor


def _get_module_name(x):
    # splitting is necessary for TensorFlow tensors
    return x.__class__.__module__.split(".")[0]


def astensor(x):
    if hasattr(x, "tensor"):
        return x
    # we use the module name instead of isinstance
    # to avoid importing all the frameworks
    module = _get_module_name(x)
    if module == "torch":
        return PyTorchTensor(x)
    if module == "tensorflow":
        return TensorFlowTensor(x)
    if module == "jax":
        return JAXTensor(x)
    if module == "numpy":
        return NumPyTensor(x)
    raise ValueError(f"Unknown type: {type(x)}")
