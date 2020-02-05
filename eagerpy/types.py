from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # for static analyzers
    import torch  # noqa: F401
    import tensorflow  # noqa: F401
    import jax  # noqa: F401
    import numpy  # noqa: F401

Axes = Tuple[int, ...]
AxisAxes = Union[int, Axes]
Shape = Tuple[int, ...]
ShapeOrScalar = Union[Shape, int]

# tensorflow.Tensor, jax.numpy.ndarray and numpy.ndarray currently evaluate to Any
# we can therefore only provide additional type information for torch.Tensor
NativeTensor = Union[
    "torch.Tensor", "tensorflow.Tensor", "jax.numpy.ndarray", "numpy.ndarray"
]
