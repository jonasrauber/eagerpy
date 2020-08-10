from typing import TypeVar
from os.path import join as _join
from os.path import dirname as _dirname

with open(_join(_dirname(__file__), "VERSION")) as _f:
    __version__ = _f.read().strip()

_T = TypeVar("_T")


class _Indexable:
    __slots__ = ()

    def __getitem__(self, index: _T) -> _T:
        return index


index = _Indexable()


from .tensor import Tensor  # noqa: F401,E402
from .tensor import TensorType  # noqa: F401,E402
from .tensor import istensor  # noqa: F401,E402

from .tensor import PyTorchTensor  # noqa: F401,E402
from .tensor import TensorFlowTensor  # noqa: F401,E402
from .tensor import NumPyTensor  # noqa: F401,E402
from .tensor import JAXTensor  # noqa: F401,E402

from . import types  # noqa: F401,E402

from .astensor import astensor  # noqa: F401,E402
from .astensor import astensors  # noqa: F401,E402
from .astensor import astensor_  # noqa: F401,E402
from .astensor import astensors_  # noqa: F401,E402

from .modules import torch  # noqa: F401,E402
from .modules import tensorflow  # noqa: F401,E402
from .modules import jax  # noqa: F401,E402
from .modules import numpy  # noqa: F401,E402

from . import utils  # noqa: F401,E402

from .framework import *  # noqa: F401,E402,F403

from . import norms  # noqa: F401,E402
from .lib import *  # noqa: F401,E402,F403
