from os.path import join, dirname

with open(join(dirname(__file__), "VERSION")) as f:
    __version__ = f.read().strip()

from .tensor import PyTorchTensor  # noqa: F401
from .tensor import TensorFlowTensor  # noqa: F401
from .framework import *  # noqa: F401,F403
