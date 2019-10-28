import torch
import tensorflow as tf
import numpy as np

from . import PyTorchTensor
from . import TensorFlowTensor
from . import NumPyTensor


def astensor(x):
    if hasattr(x, "tensor"):
        return x
    if isinstance(x, torch.Tensor):
        return PyTorchTensor(x)
    if isinstance(x, tf.Tensor):
        return TensorFlowTensor(x)
    if isinstance(x, np.ndarray):
        return NumPyTensor(x)
    raise ValueError(f"Unknown type: {type(x)}")
