from typing import overload, Sequence, Callable, Tuple, Any, Optional, cast, Union
from typing_extensions import Literal

from .types import Axes, AxisAxes, Shape, ShapeOrScalar

from .tensor import Tensor
from .tensor import TensorType
from .tensor import TensorOrScalar

newaxis = None
inf = float("inf")
nan = float("nan")


def clip(t: TensorType, min_: float, max_: float) -> TensorType:
    return t.clip(min_, max_)


def abs(t: TensorType) -> TensorType:
    return t.abs()


def sign(t: TensorType) -> TensorType:
    return t.sign()


def sqrt(t: TensorType) -> TensorType:
    return t.sqrt()


def square(t: TensorType) -> TensorType:
    return t.square()


def pow(t: TensorType, exponent: TensorOrScalar) -> TensorType:
    return t.pow(exponent)


def tanh(t: TensorType) -> TensorType:
    return t.tanh()


def arctanh(t: TensorType) -> TensorType:
    return t.arctanh()


def sum(
    t: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return t.sum(axis=axis, keepdims=keepdims)


def prod(
    t: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return t.prod(axis=axis, keepdims=keepdims)


def mean(
    t: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return t.mean(axis=axis, keepdims=keepdims)


def min(
    t: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return t.min(axis=axis, keepdims=keepdims)


def max(
    t: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return t.max(axis=axis, keepdims=keepdims)


@overload
def minimum(x: TensorType, y: TensorOrScalar) -> TensorType:
    ...


@overload
def minimum(x: TensorOrScalar, y: TensorType) -> TensorType:
    ...


def minimum(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    if not isinstance(x, Tensor):
        return cast(Tensor, y).minimum(x)
    return x.minimum(y)


@overload
def maximum(x: TensorType, y: TensorOrScalar) -> TensorType:
    ...


@overload
def maximum(x: TensorOrScalar, y: TensorType) -> TensorType:
    ...


def maximum(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    if not isinstance(x, Tensor):
        return cast(Tensor, y).maximum(x)
    return x.maximum(y)


def argmin(t: TensorType, axis: Optional[int] = None) -> TensorType:
    return t.argmin(axis=axis)


def argmax(t: TensorType, axis: Optional[int] = None) -> TensorType:
    return t.argmax(axis=axis)


def argsort(t: TensorType, axis: int = -1) -> TensorType:
    return t.argsort(axis=axis)


def sort(t: TensorType, axis: int = -1) -> TensorType:
    return t.sort(axis=axis)


def topk(t: TensorType, k: int, sorted: bool = True) -> Tuple[TensorType, TensorType]:
    return t.topk(k, sorted=sorted)


def uniform(
    t: TensorType, shape: ShapeOrScalar, low: float = 0.0, high: float = 1.0
) -> TensorType:
    return t.uniform(shape, low=low, high=high)


def normal(
    t: TensorType, shape: ShapeOrScalar, mean: float = 0.0, stddev: float = 1.0
) -> TensorType:
    return t.normal(shape, mean=mean, stddev=stddev)


def ones(t: TensorType, shape: ShapeOrScalar) -> TensorType:
    return t.ones(shape)


def zeros(t: TensorType, shape: ShapeOrScalar) -> TensorType:
    return t.zeros(shape)


def ones_like(t: TensorType) -> TensorType:
    return t.ones_like()


def zeros_like(t: TensorType) -> TensorType:
    return t.zeros_like()


def full_like(t: TensorType, fill_value: float) -> TensorType:
    return t.full_like(fill_value)


def onehot_like(t: TensorType, indices: TensorType, *, value: float = 1) -> TensorType:
    return t.onehot_like(indices, value=value)


def from_numpy(t: TensorType, a: Any) -> TensorType:
    return t.from_numpy(a)


def concatenate(tensors: Sequence[TensorType], axis: int = 0) -> TensorType:
    t = tensors[0]
    return t._concatenate(tensors, axis=axis)


def transpose(t: TensorType, axes: Optional[Axes] = None) -> TensorType:
    return t.transpose(axes=axes)


@overload
def logical_and(x: TensorType, y: TensorOrScalar) -> TensorType:
    ...


@overload
def logical_and(x: TensorOrScalar, y: TensorType) -> TensorType:
    ...


def logical_and(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    if not isinstance(x, Tensor):
        return cast(Tensor, y).logical_and(x)
    return x.logical_and(y)


@overload
def logical_or(x: TensorType, y: TensorOrScalar) -> TensorType:
    ...


@overload
def logical_or(x: TensorOrScalar, y: TensorType) -> TensorType:
    ...


def logical_or(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    if not isinstance(x, Tensor):
        return cast(Tensor, y).logical_or(x)
    return x.logical_or(y)


def logical_not(t: TensorType) -> TensorType:
    return t.logical_not()


def exp(t: TensorType) -> TensorType:
    return t.exp()


def log(t: TensorType) -> TensorType:
    return t.log()


def log2(t: TensorType) -> TensorType:
    return t.log2()


def log10(t: TensorType) -> TensorType:
    return t.log10()


def log1p(t: TensorType) -> TensorType:
    return t.log1p()


def where(condition: TensorType, x: TensorOrScalar, y: TensorOrScalar) -> TensorType:
    return condition.where(x, y)


def tile(t: TensorType, multiples: Axes) -> TensorType:
    return t.tile(multiples)


def matmul(x: TensorType, y: TensorType) -> TensorType:
    return x.matmul(y)


def softmax(t: TensorType, axis: int = -1) -> TensorType:
    return t.softmax(axis=axis)


def log_softmax(t: TensorType, axis: int = -1) -> TensorType:
    return t.log_softmax(axis=axis)


def stack(tensors: Sequence[TensorType], axis: int = 0) -> TensorType:
    t = tensors[0]
    return t._stack(tensors, axis=axis)


def squeeze(t: TensorType, axis: Optional[AxisAxes] = None) -> TensorType:
    return t.squeeze(axis=axis)


def expand_dims(t: TensorType, axis: int) -> TensorType:
    return t.expand_dims(axis=axis)


def full(t: TensorType, shape: ShapeOrScalar, value: float) -> TensorType:
    return t.full(shape, value)


def index_update(t: TensorType, indices: Any, values: TensorOrScalar) -> TensorType:
    return t.index_update(indices, values)


def arange(
    t: TensorType, start: int, stop: Optional[int] = None, step: Optional[int] = None
) -> TensorType:
    return t.arange(start, stop, step)


def cumsum(t: TensorType, axis: Optional[int] = None) -> TensorType:
    return t.cumsum(axis=axis)


def flip(t: TensorType, axis: Optional[AxisAxes] = None) -> TensorType:
    return t.flip(axis=axis)


def meshgrid(
    t: TensorType, *tensors: TensorType, indexing: str = "xy"
) -> Tuple[TensorType, ...]:
    return t.meshgrid(*tensors, indexing=indexing)


def pad(
    t: TensorType,
    paddings: Tuple[Tuple[int, int], ...],
    mode: str = "constant",
    value: float = 0,
) -> TensorType:
    return t.pad(paddings, mode=mode, value=value)


def isnan(t: TensorType) -> TensorType:
    return t.isnan()


def isinf(t: TensorType) -> TensorType:
    return t.isinf()


def all(
    t: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return t.all(axis=axis, keepdims=keepdims)


def any(
    t: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return t.any(axis=axis, keepdims=keepdims)


def crossentropy(logits: TensorType, labels: TensorType) -> TensorType:
    return logits.crossentropy(labels)


def slogdet(matrix: TensorType) -> Tuple[TensorType, TensorType]:
    return matrix.slogdet()


@overload
def value_and_grad_fn(
    t: TensorType, f: Callable[..., TensorType]
) -> Callable[..., Tuple[TensorType, TensorType]]:
    ...


@overload
def value_and_grad_fn(
    t: TensorType, f: Callable[..., TensorType], has_aux: Literal[False]
) -> Callable[..., Tuple[TensorType, TensorType]]:
    ...


@overload
def value_and_grad_fn(
    t: TensorType, f: Callable[..., Tuple[TensorType, Any]], has_aux: Literal[True]
) -> Callable[..., Tuple[TensorType, Any, TensorType]]:
    ...


def value_and_grad_fn(t: Any, f: Any, has_aux: bool = False) -> Any:
    return t._value_and_grad_fn(f, has_aux=has_aux)


def value_and_grad(
    f: Callable[..., TensorType], t: TensorType, *args: Any, **kwargs: Any
) -> Tuple[TensorType, TensorType]:
    return t.value_and_grad(f, *args, **kwargs)


def value_aux_and_grad(
    f: Callable[..., Tuple[TensorType, Any]], t: TensorType, *args: Any, **kwargs: Any
) -> Tuple[TensorType, Any, TensorType]:
    return t.value_aux_and_grad(f, *args, **kwargs)


def reshape(t: TensorType, shape: Union[Shape, int]) -> TensorType:
    return t.reshape(shape)


def take_along_axis(t: TensorType, indices: TensorType, axis: int) -> TensorType:
    return t.take_along_axis(indices, axis)


def flatten(t: TensorType, start: int = 0, end: int = -1) -> TensorType:
    return t.flatten(start=start, end=end)
