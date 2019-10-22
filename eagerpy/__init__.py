from os.path import join, dirname

with open(join(dirname(__file__), "VERSION")) as f:
    __version__ = f.read().strip()

from .tensor import PyTorchTensor
from .tensor import TensorFlowTensor
from .framework import *
