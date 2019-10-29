import torch
import tensorflow as tf
import jax
import numpy as np

from . import PyTorchTensor
from . import TensorFlowTensor
from . import JAXTensor
from . import NumPyTensor


def astensor(x):
    if hasattr(x, "tensor"):
        return x
    if isinstance(x, torch.Tensor):
        return PyTorchTensor(x)
    if isinstance(x, tf.Tensor):
        return TensorFlowTensor(x)
    if isinstance(x, jax.numpy.ndarray):
        return JAXTensor(x)
    if isinstance(x, np.ndarray):
        return NumPyTensor(x)
    raise ValueError(f"Unknown type: {type(x)}")
