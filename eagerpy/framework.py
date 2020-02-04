from typing import overload, Sequence, Callable, Tuple, Any
from typing_extensions import Literal

from .tensor import TensorType
from .tensor import Tensor

newaxis = None
inf = float("inf")
nan = float("nan")


def clip(t: TensorType, min_, max_) -> TensorType:
    return t.clip(min_, max_)


def abs(t: TensorType) -> TensorType:
    return t.abs()


def sign(t: TensorType) -> TensorType:
    return t.sign()


def sqrt(t: TensorType) -> TensorType:
    return t.sqrt()


def square(t: TensorType) -> TensorType:
    return t.square()


def pow(t: TensorType, exponent) -> TensorType:
    return t.pow(exponent)


def tanh(t: TensorType) -> TensorType:
    return t.tanh()


def arctanh(t: TensorType) -> TensorType:
    return t.arctanh()


def sum(t: TensorType, axis=None, keepdims=False) -> TensorType:
    return t.sum(axis=axis, keepdims=keepdims)


def mean(t: TensorType, axis=None, keepdims=False) -> TensorType:
    return t.mean(axis=axis, keepdims=keepdims)


def min(t: TensorType, axis=None, keepdims=False) -> TensorType:
    return t.min(axis=axis, keepdims=keepdims)


def max(t: TensorType, axis=None, keepdims=False) -> TensorType:
    return t.max(axis=axis, keepdims=keepdims)


@overload
def minimum(x: TensorType, y) -> TensorType:
    ...


@overload
def minimum(x, y: TensorType) -> TensorType:
    ...


def minimum(x, y):
    if not isinstance(x, Tensor):
        return y.minimum(x)
    return x.minimum(y)


@overload
def maximum(x: TensorType, y) -> TensorType:
    ...


@overload
def maximum(x, y: TensorType) -> TensorType:
    ...


def maximum(x, y):
    if not isinstance(x, Tensor):
        return y.maximum(x)
    return x.maximum(y)


def argmin(t: TensorType, axis=None) -> TensorType:
    return t.argmin(axis=axis)


def argmax(t: TensorType, axis=None) -> TensorType:
    return t.argmax(axis=axis)


def argsort(t: TensorType, axis=-1) -> TensorType:
    return t.argsort(axis=axis)


def uniform(t: TensorType, shape, low=0.0, high=1.0) -> TensorType:
    return t.uniform(shape, low=low, high=high)


def normal(t: TensorType, shape, mean=0.0, stddev=1.0) -> TensorType:
    return t.normal(shape, mean=mean, stddev=stddev)


def ones(t: TensorType, shape) -> TensorType:
    return t.ones(shape)


def zeros(t: TensorType, shape) -> TensorType:
    return t.zeros(shape)


def ones_like(t: TensorType) -> TensorType:
    return t.ones_like()


def zeros_like(t: TensorType) -> TensorType:
    return t.zeros_like()


def full_like(t: TensorType, fill_value) -> TensorType:
    return t.full_like(fill_value)


def onehot_like(t: TensorType, indices, *, value=1) -> TensorType:
    return t.onehot_like(indices, value=value)


def from_numpy(t: TensorType, a) -> TensorType:
    return t.from_numpy(a)


def concatenate(tensors: Sequence[TensorType], axis=0) -> TensorType:
    t = tensors[0]
    return t._concatenate(tensors, axis=axis)


def transpose(t: TensorType, axes=None) -> TensorType:
    return t.transpose(axes=axes)


@overload
def logical_and(x: TensorType, y) -> TensorType:
    ...


@overload
def logical_and(x, y: TensorType) -> TensorType:
    ...


def logical_and(x, y):
    if not isinstance(x, Tensor):
        return y.logical_and(x)
    return x.logical_and(y)


@overload
def logical_or(x: TensorType, y) -> TensorType:
    ...


@overload
def logical_or(x, y: TensorType) -> TensorType:
    ...


def logical_or(x, y):
    if not isinstance(x, Tensor):
        return y.logical_or(x)
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


def where(condition: TensorType, x, y) -> TensorType:
    return condition.where(x, y)


def tile(t: TensorType, multiples) -> TensorType:
    return t.tile(multiples)


def matmul(x: TensorType, y: TensorType) -> TensorType:
    return x.matmul(y)


def softmax(t: TensorType, axis=-1) -> TensorType:
    return t.softmax(axis=axis)


def log_softmax(t: TensorType, axis=-1) -> TensorType:
    return t.log_softmax(axis=axis)


def stack(tensors: Sequence[TensorType], axis=0) -> TensorType:
    t = tensors[0]
    return t._stack(tensors, axis=axis)


def squeeze(t: TensorType, axis=None) -> TensorType:
    return t.squeeze(axis=axis)


def expand_dims(t: TensorType, axis=None) -> TensorType:
    return t.expand_dims(axis=axis)


def full(t: TensorType, shape, value) -> TensorType:
    return t.full(shape, value)


def index_update(t: TensorType, indices, values) -> TensorType:
    return t.index_update(indices, values)


def arange(t: TensorType, start, stop=None, step=None) -> TensorType:
    return t.arange(start, stop, step)


def cumsum(t: TensorType, axis=None) -> TensorType:
    return t.cumsum(axis=axis)


def flip(t: TensorType, axis=None) -> TensorType:
    return t.flip(axis=axis)


def meshgrid(t: TensorType, *tensors, indexing="xy") -> Tuple[TensorType, ...]:
    return t.meshgrid(*tensors, indexing=indexing)


def pad(t: TensorType, paddings, mode="constant", value=0) -> TensorType:
    return t.pad(paddings, mode=mode, value=value)


def isnan(t: TensorType) -> TensorType:
    return t.isnan()


def isinf(t: TensorType) -> TensorType:
    return t.isinf()


def all(t: TensorType, axis=None, keepdims=False) -> TensorType:
    return t.all(axis=axis, keepdims=keepdims)


def any(t: TensorType, axis=None, keepdims=False) -> TensorType:
    return t.any(axis=axis, keepdims=keepdims)


def crossentropy(logits: TensorType, labels: TensorType) -> TensorType:
    return logits.crossentropy(labels)


@overload
def value_and_grad_fn(
    t: TensorType, f: Callable
) -> Callable[..., Tuple[TensorType, TensorType]]:
    ...


@overload
def value_and_grad_fn(
    t: TensorType, f: Callable, has_aux: Literal[False]
) -> Callable[..., Tuple[TensorType, TensorType]]:
    ...


@overload
def value_and_grad_fn(
    t: TensorType, f: Callable, has_aux: Literal[True]
) -> Callable[..., Tuple[TensorType, Any, TensorType]]:
    ...


def value_and_grad_fn(t, f, has_aux=False):
    return t._value_and_grad_fn(f, has_aux=has_aux)


def value_and_grad(
    f: Callable, t: TensorType, *args, **kwargs
) -> Tuple[TensorType, TensorType]:
    return t.value_and_grad(f, *args, **kwargs)


def value_aux_and_grad(
    f: Callable, t: TensorType, *args, **kwargs
) -> Tuple[TensorType, Any, TensorType]:
    return t.value_aux_and_grad(f, *args, **kwargs)


def reshape(t: TensorType, shape) -> TensorType:
    return t.reshape(shape)


def take_along_axis(t: TensorType, indices: TensorType, axis: int):
    return t.take_along_axis(indices, axis)
