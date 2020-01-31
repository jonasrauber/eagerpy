from typing import overload, Sequence, Callable, Tuple, Any
from typing_extensions import Literal

from . import istensor
from . import Tensor

newaxis = None
inf = float("inf")
nan = float("nan")


def clip(t: Tensor, min_, max_) -> Tensor:
    return t.clip(min_, max_)


def abs(t: Tensor) -> Tensor:
    return t.abs()


def sign(t: Tensor) -> Tensor:
    return t.sign()


def sqrt(t: Tensor) -> Tensor:
    return t.sqrt()


def square(t: Tensor) -> Tensor:
    return t.square()


def pow(t: Tensor, exponent) -> Tensor:
    return t.pow(exponent)


def tanh(t: Tensor) -> Tensor:
    return t.tanh()


def arctanh(t: Tensor) -> Tensor:
    return t.arctanh()


def sum(t: Tensor, axis=None, keepdims=False) -> Tensor:
    return t.sum(axis=axis, keepdims=keepdims)


def mean(t: Tensor, axis=None, keepdims=False) -> Tensor:
    return t.mean(axis=axis, keepdims=keepdims)


def min(t: Tensor, axis=None, keepdims=False) -> Tensor:
    return t.min(axis=axis, keepdims=keepdims)


def max(t: Tensor, axis=None, keepdims=False) -> Tensor:
    return t.max(axis=axis, keepdims=keepdims)


@overload
def minimum(x: Tensor, y) -> Tensor:
    ...


@overload
def minimum(x, y: Tensor) -> Tensor:
    ...


def minimum(x, y):
    if not istensor(x):
        return y.minimum(x)
    return x.minimum(y)


@overload
def maximum(x: Tensor, y) -> Tensor:
    ...


@overload
def maximum(x, y: Tensor) -> Tensor:
    ...


def maximum(x, y):
    if not istensor(x):
        return y.maximum(x)
    return x.maximum(y)


def argmin(t: Tensor, axis=None) -> Tensor:
    return t.argmin(axis=axis)


def argmax(t: Tensor, axis=None) -> Tensor:
    return t.argmax(axis=axis)


def argsort(t: Tensor, *args, **kwargs) -> Tensor:
    return t.argsort(*args, **kwargs)


def uniform(t: Tensor, *args, **kwargs) -> Tensor:
    return t.uniform(*args, **kwargs)


def normal(t: Tensor, *args, **kwargs) -> Tensor:
    return t.normal(*args, **kwargs)


def ones(t: Tensor, *args, **kwargs) -> Tensor:
    return t.ones(*args, **kwargs)


def zeros(t: Tensor, *args, **kwargs) -> Tensor:
    return t.zeros(*args, **kwargs)


def ones_like(t: Tensor) -> Tensor:
    return t.ones_like()


def zeros_like(t: Tensor) -> Tensor:
    return t.zeros_like()


def full_like(t: Tensor, *args, **kwargs) -> Tensor:
    return t.full_like(*args, **kwargs)


def onehot_like(t: Tensor, *args, **kwargs) -> Tensor:
    return t.onehot_like(*args, **kwargs)


def from_numpy(t: Tensor, *args, **kwargs) -> Tensor:
    return t.from_numpy(*args, **kwargs)


def concatenate(tensors: Sequence[Tensor], axis=0) -> Tensor:
    t = tensors[0]
    return t._concatenate(tensors, axis=axis)


def transpose(t: Tensor, axes=None) -> Tensor:
    return t.transpose(axes=axes)


@overload
def logical_and(x: Tensor, y) -> Tensor:
    ...


@overload
def logical_and(x, y: Tensor) -> Tensor:
    ...


def logical_and(x, y):
    if not istensor(x):
        return y.logical_and(x)
    return x.logical_and(y)


@overload
def logical_or(x: Tensor, y) -> Tensor:
    ...


@overload
def logical_or(x, y: Tensor) -> Tensor:
    ...


def logical_or(x, y):
    if not istensor(x):
        return y.logical_or(x)
    return x.logical_or(y)


def logical_not(t: Tensor) -> Tensor:
    return t.logical_not()


def exp(t: Tensor) -> Tensor:
    return t.exp()


def log(t: Tensor) -> Tensor:
    return t.log()


def log2(t: Tensor) -> Tensor:
    return t.log2()


def log10(t: Tensor) -> Tensor:
    return t.log10()


def log1p(t: Tensor) -> Tensor:
    return t.log1p()


def where(condition: Tensor, x, y) -> Tensor:
    return condition.where(x, y)


def tile(t: Tensor, multiples) -> Tensor:
    return t.tile(multiples)


def matmul(x: Tensor, y: Tensor) -> Tensor:
    return x.matmul(y)


def softmax(t: Tensor, axis=-1) -> Tensor:
    return t.softmax(axis=axis)


def log_softmax(t: Tensor, axis=-1) -> Tensor:
    return t.log_softmax(axis=axis)


def stack(tensors: Sequence[Tensor], axis=0) -> Tensor:
    t = tensors[0]
    return t._stack(tensors, axis=axis)


def squeeze(t: Tensor, *args, **kwargs) -> Tensor:
    return t.squeeze(*args, **kwargs)


def expand_dims(t: Tensor, *args, **kwargs) -> Tensor:
    return t.expand_dims(*args, **kwargs)


def full(t: Tensor, shape, value) -> Tensor:
    return t.full(shape, value)


def index_update(t: Tensor, *args, **kwargs) -> Tensor:
    return t.index_update(*args, **kwargs)


def arange(t: Tensor, *args, **kwargs) -> Tensor:
    return t.arange(*args, **kwargs)


def cumsum(t: Tensor, axis=None) -> Tensor:
    return t.cumsum(axis=axis)


def flip(t: Tensor, axis=None) -> Tensor:
    return t.flip(axis=axis)


def meshgrid(t: Tensor, *args, **kwargs) -> Tuple[Tensor, ...]:
    return t.meshgrid(*args, **kwargs)


def pad(t: Tensor, paddings, mode="constant", value=0) -> Tensor:
    return t.pad(paddings, mode=mode, value=value)


def isnan(t: Tensor) -> Tensor:
    return t.isnan()


def isinf(t: Tensor) -> Tensor:
    return t.isinf()


def all(t: Tensor, *args, **kwargs) -> Tensor:
    return t.all(*args, **kwargs)


def any(t: Tensor, *args, **kwargs) -> Tensor:
    return t.any(*args, **kwargs)


def crossentropy(logits: Tensor, labels: Tensor) -> Tensor:
    return logits.crossentropy(labels)


@overload
def value_and_grad_fn(t: Tensor, f: Callable) -> Callable[..., Tuple[Tensor, Tensor]]:
    ...


@overload
def value_and_grad_fn(
    t: Tensor, f: Callable, has_aux: Literal[False]
) -> Callable[..., Tuple[Tensor, Tensor]]:
    ...


@overload
def value_and_grad_fn(
    t: Tensor, f: Callable, has_aux: Literal[True]
) -> Callable[..., Tuple[Tensor, Any, Tensor]]:
    ...


def value_and_grad_fn(t, f, has_aux=False):
    return t._value_and_grad_fn(f, has_aux=has_aux)


def value_and_grad(f: Callable, t: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
    return t.value_and_grad(f, *args, **kwargs)


def value_aux_and_grad(
    f: Callable, t: Tensor, *args, **kwargs
) -> Tuple[Tensor, Any, Tensor]:
    return t.value_aux_and_grad(f, *args, **kwargs)


def reshape(t: Tensor, shape) -> Tensor:
    return t.reshape(shape)
