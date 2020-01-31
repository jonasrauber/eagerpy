from typing import overload, Sequence, Callable, Tuple, Any
from typing_extensions import Literal

from .tensor import TensorType
from .tensor import istensor

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
    if not istensor(x):
        return y.minimum(x)
    return x.minimum(y)


@overload
def maximum(x: TensorType, y) -> TensorType:
    ...


@overload
def maximum(x, y: TensorType) -> TensorType:
    ...


def maximum(x, y):
    if not istensor(x):
        return y.maximum(x)
    return x.maximum(y)


def argmin(t: TensorType, axis=None) -> TensorType:
    return t.argmin(axis=axis)


def argmax(t: TensorType, axis=None) -> TensorType:
    return t.argmax(axis=axis)


def argsort(t: TensorType, *args, **kwargs) -> TensorType:
    return t.argsort(*args, **kwargs)


def uniform(t: TensorType, *args, **kwargs) -> TensorType:
    return t.uniform(*args, **kwargs)


def normal(t: TensorType, *args, **kwargs) -> TensorType:
    return t.normal(*args, **kwargs)


def ones(t: TensorType, *args, **kwargs) -> TensorType:
    return t.ones(*args, **kwargs)


def zeros(t: TensorType, *args, **kwargs) -> TensorType:
    return t.zeros(*args, **kwargs)


def ones_like(t: TensorType) -> TensorType:
    return t.ones_like()


def zeros_like(t: TensorType) -> TensorType:
    return t.zeros_like()


def full_like(t: TensorType, *args, **kwargs) -> TensorType:
    return t.full_like(*args, **kwargs)


def onehot_like(t: TensorType, *args, **kwargs) -> TensorType:
    return t.onehot_like(*args, **kwargs)


def from_numpy(t: TensorType, *args, **kwargs) -> TensorType:
    return t.from_numpy(*args, **kwargs)


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
    if not istensor(x):
        return y.logical_and(x)
    return x.logical_and(y)


@overload
def logical_or(x: TensorType, y) -> TensorType:
    ...


@overload
def logical_or(x, y: TensorType) -> TensorType:
    ...


def logical_or(x, y):
    if not istensor(x):
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


def squeeze(t: TensorType, *args, **kwargs) -> TensorType:
    return t.squeeze(*args, **kwargs)


def expand_dims(t: TensorType, *args, **kwargs) -> TensorType:
    return t.expand_dims(*args, **kwargs)


def full(t: TensorType, shape, value) -> TensorType:
    return t.full(shape, value)


def index_update(t: TensorType, *args, **kwargs) -> TensorType:
    return t.index_update(*args, **kwargs)


def arange(t: TensorType, *args, **kwargs) -> TensorType:
    return t.arange(*args, **kwargs)


def cumsum(t: TensorType, axis=None) -> TensorType:
    return t.cumsum(axis=axis)


def flip(t: TensorType, axis=None) -> TensorType:
    return t.flip(axis=axis)


def meshgrid(t: TensorType, *args, **kwargs) -> Tuple[TensorType, ...]:
    return t.meshgrid(*args, **kwargs)


def pad(t: TensorType, paddings, mode="constant", value=0) -> TensorType:
    return t.pad(paddings, mode=mode, value=value)


def isnan(t: TensorType) -> TensorType:
    return t.isnan()


def isinf(t: TensorType) -> TensorType:
    return t.isinf()


def all(t: TensorType, *args, **kwargs) -> TensorType:
    return t.all(*args, **kwargs)


def any(t: TensorType, *args, **kwargs) -> TensorType:
    return t.any(*args, **kwargs)


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
