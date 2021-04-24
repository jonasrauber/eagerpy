import functools
from typing import (
    TYPE_CHECKING,
    Union,
    overload,
    Tuple,
    TypeVar,
    Generic,
    Any,
    Callable,
)
import sys

from jax import tree_flatten, tree_unflatten

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


def _get_module_name(x: Any) -> str:
    # splitting is necessary for TensorFlow tensors
    return type(x).__module__.split(".")[0]


@overload
def astensor(x: TensorType) -> TensorType:
    ...


@overload
def astensor(x: "torch.Tensor") -> PyTorchTensor:
    ...


@overload
def astensor(x: NativeTensor) -> Tensor:  # type: ignore
    ...


def astensor(x: Union[NativeTensor, Tensor, Any]) -> Union[Tensor, Any]:  # type: ignore
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
    if (name == "jax" or name == "jaxlib") and isinstance(x, m["jax"].numpy.ndarray):  # type: ignore
        return JAXTensor(x)
    if name == "numpy" and isinstance(x, m[name].ndarray):  # type: ignore
        return NumPyTensor(x)

    # non Tensor types are returned unmodified
    return x


def astensors(*xs: Union[NativeTensor, Tensor]) -> Tuple[Tensor, ...]:  # type: ignore
    return tuple(astensor(x) for x in xs)


T = TypeVar("T")


class RestoreTypeFunc(Generic[T]):
    def __init__(self, x: T):
        self.unwrap = not isinstance(x, Tensor)

    @overload
    def __call__(self, x: Tensor) -> T:
        ...

    @overload  # noqa: F811
    def __call__(self, x: Tensor, y: Tensor) -> Tuple[T, T]:
        ...

    @overload  # noqa: F811
    def __call__(self, x: Tensor, y: Tensor, z: Tensor, *args: Tensor) -> Tuple[T, ...]:
        ...

    @overload  # noqa: F811
    def __call__(self, *args: Any) -> Any:
        # catch other types, otherwise we would return type T for input type Any
        ...

    def __call__(self, *args):  # type: ignore  # noqa: F811
        result = tuple(as_raw_tensor(x) for x in args) if self.unwrap else args
        if len(result) == 1:
            (result,) = result
        return result


def astensor_(x: T) -> Tuple[Tensor, RestoreTypeFunc[T]]:
    return astensor(x), RestoreTypeFunc[T](x)


def astensors_(x: T, *xs: T) -> Tuple[Tuple[Tensor, ...], RestoreTypeFunc[T]]:
    return astensors(x, *xs), RestoreTypeFunc[T](x)


def as_tensors(data: Any) -> Any:
    leaf_values, tree_def = tree_flatten(data)
    leaf_values = tuple(astensor(value) for value in leaf_values)
    return tree_unflatten(tree_def, leaf_values)


def has_tensor(tree_def: Any) -> bool:
    return "<class 'eagerpy.tensor" in str(tree_def)


def as_tensors_any(data: Any) -> Tuple[Any, bool]:
    """Convert data structure leaves in Tensor and detect if any of the input data contains a Tensor.

    Parameters
    ----------
    data
        data structure.

    Returns
    -------
    Any
        modified data structure.
    bool
        True if input data contains a Tensor type.
    """
    leaf_values, tree_def = tree_flatten(data)
    transformed_leaf_values = tuple(astensor(value) for value in leaf_values)
    return tree_unflatten(tree_def, transformed_leaf_values), has_tensor(tree_def)


def as_raw_tensor(x: T) -> Any:
    if isinstance(x, Tensor):
        return x.raw
    else:
        return x


def as_raw_tensors(data: Any) -> Any:
    leaf_values, tree_def = tree_flatten(data)

    if not has_tensor(tree_def):
        return data

    leaf_values = tuple(as_raw_tensor(value) for value in leaf_values)
    unwrap_leaf_values = []
    for x in leaf_values:
        name = _get_module_name(x)
        m = sys.modules
        if name == "torch" and isinstance(x, m[name].Tensor):  # type: ignore
            unwrap_leaf_values.append((x, True))
        elif name == "tensorflow" and isinstance(x, m[name].Tensor):  # type: ignore
            unwrap_leaf_values.append((x, True))
        elif (name == "jax" or name == "jaxlib") and isinstance(x, m["jax"].numpy.ndarray):  # type: ignore
            unwrap_leaf_values.append((x, True))
        elif name == "numpy" and isinstance(x, m[name].ndarray):  # type: ignore
            unwrap_leaf_values.append((x, True))
        else:
            unwrap_leaf_values.append(x)
    return tree_unflatten(tree_def, unwrap_leaf_values)


def eager_function(
    func: Callable[..., T], skip_argnums: Tuple = tuple()
) -> Callable[..., T]:
    @functools.wraps(func)
    def eager_func(*args: Any, **kwargs: Any) -> Any:
        sorted_skip_argnums = sorted(skip_argnums)
        skip_args = [arg for i, arg in enumerate(args) if i in sorted_skip_argnums]
        kept_args = [arg for i, arg in enumerate(args) if i not in sorted_skip_argnums]

        (kept_args, kwargs), has_tensor = as_tensors_any((kept_args, kwargs))
        unwrap = not has_tensor

        for i, arg in zip(sorted_skip_argnums, skip_args):
            kept_args.insert(i, arg)

        result = func(*kept_args, **kwargs)

        if unwrap:
            raw_result = as_raw_tensors(result)
            return raw_result
        else:
            return result

    return eager_func
