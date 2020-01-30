from .base import AbstractBaseTensor
from .base import unwrap_

from .tensor import istensor
from .tensor import Tensor

from .. import index

import functools
from typing import Tuple, cast, Union, Any, TypeVar, Callable
from importlib import import_module
from collections.abc import Iterable
import numpy as np


FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


if False:
    import tensorflow as tf  # for static analyzers
else:
    tf = None


def samedevice(f: F) -> F:
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        with tf.device(self.raw.device):
            return f(self, *args, **kwargs)

    return cast(F, wrapper)


def common_dtype(f: F) -> F:
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        dtypes = {self.dtype} | {arg.dtype for arg in args if istensor(arg)}
        if len(dtypes) == 1:
            # all dtypes are the same, nothing more to do
            return f(self, *args, **kwargs)
        numpy_dtypes = [np.dtype(dtype.name) for dtype in dtypes]
        common = np.find_common_type(numpy_dtypes, [])
        common = getattr(tf, common.name)
        if self.dtype != common:
            self = self.astype(common)
        args = [
            arg.astype(common) if istensor(arg) and arg.dtype != common else arg
            for arg in args
        ]
        return f(self, *args, **kwargs)

    return cast(F, wrapper)


def assert_bool(x: Tensor) -> None:
    if not istensor(x):
        return
    if x.dtype != tf.bool:
        raise ValueError(f"requires dtype bool, consider t.bool().all()")


class TensorFlowTensor(AbstractBaseTensor):
    def __init__(self, raw: "tf.Tensor"):
        global tf
        if tf is None:
            tf = import_module("tensorflow")
        super().__init__(raw)

    @property
    def raw(self) -> "tf.Tensor":
        return super().raw

    def numpy(self: Tensor) -> Any:
        return self.raw.numpy()

    def item(self: Tensor) -> Union[int, float, bool]:
        return self.numpy().item()  # type: ignore

    @property
    def shape(self: Tensor) -> Tuple:
        return tuple(self.raw.shape.as_list())

    def reshape(self: Tensor, shape) -> Tensor:
        return type(self)(tf.reshape(self.raw, shape))

    def astype(self: Tensor, dtype) -> Tensor:
        return type(self)(tf.cast(self.raw, dtype))

    def clip(self: Tensor, min_, max_) -> Tensor:
        return type(self)(tf.clip_by_value(self.raw, min_, max_))

    def square(self: Tensor) -> Tensor:
        return type(self)(tf.square(self.raw))

    def arctanh(self: Tensor) -> Tensor:
        return type(self)(tf.atanh(self.raw))

    def sum(self: Tensor, axis=None, keepdims=False) -> Tensor:
        if self.raw.dtype == tf.bool:
            return type(self)(self.astype(tf.int64).sum(axis=axis, keepdims=keepdims))
        return type(self)(tf.reduce_sum(self.raw, axis=axis, keepdims=keepdims))

    def mean(self: Tensor, axis=None, keepdims=False) -> Tensor:
        return type(self)(tf.reduce_mean(self.raw, axis=axis, keepdims=keepdims))

    def min(self: Tensor, axis=None, keepdims=False) -> Tensor:
        return type(self)(tf.reduce_min(self.raw, axis=axis, keepdims=keepdims))

    def max(self: Tensor, axis=None, keepdims=False) -> Tensor:
        return type(self)(tf.reduce_max(self.raw, axis=axis, keepdims=keepdims))

    def minimum(self: Tensor, other) -> Tensor:
        return type(self)(tf.minimum(self.raw, unwrap_(other)))

    def maximum(self: Tensor, other) -> Tensor:
        return type(self)(tf.maximum(self.raw, unwrap_(other)))

    def argmin(self: Tensor, axis=None) -> Tensor:
        return type(self)(tf.argmin(self.raw, axis=axis))

    def argmax(self: Tensor, axis=None) -> Tensor:
        return type(self)(tf.argmax(self.raw, axis=axis))

    def argsort(self: Tensor, axis=-1) -> Tensor:
        return type(self)(tf.argsort(self.raw, axis=axis))

    @samedevice
    def uniform(self: Tensor, shape, low=0.0, high=1.0) -> Tensor:
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return type(self)(
            tf.random.uniform(shape, minval=low, maxval=high, dtype=self.raw.dtype)
        )

    @samedevice
    def normal(self: Tensor, shape, mean=0.0, stddev=1.0) -> Tensor:
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return type(self)(
            tf.random.normal(shape, mean=mean, stddev=stddev, dtype=self.raw.dtype)
        )

    @samedevice
    def ones(self: Tensor, shape) -> Tensor:
        return type(self)(tf.ones(shape, dtype=self.raw.dtype))

    @samedevice
    def zeros(self: Tensor, shape) -> Tensor:
        return type(self)(tf.zeros(shape, dtype=self.raw.dtype))

    def ones_like(self: Tensor) -> Tensor:
        return type(self)(tf.ones_like(self.raw))

    def zeros_like(self: Tensor) -> Tensor:
        return type(self)(tf.zeros_like(self.raw))

    def full_like(self: Tensor, fill_value) -> Tensor:
        fill_value = tf.cast(fill_value, self.raw.dtype)
        return type(self)(tf.fill(self.raw.shape, fill_value))

    @samedevice
    def onehot_like(self: Tensor, indices: Tensor, *, value=1) -> Tensor:
        if self.ndim != 2:
            raise ValueError("onehot_like only supported for 2D tensors")
        if indices.ndim != 1:
            raise ValueError("onehot_like requires 1D indices")
        if len(indices) != len(self):
            raise ValueError("length of indices must match length of tensor")
        value = tf.cast(value, self.raw.dtype)
        return type(self)(
            tf.one_hot(
                indices.raw,
                depth=self.raw.shape[-1],
                on_value=value,
                dtype=self.raw.dtype,
            )
        )

    @samedevice
    def from_numpy(self: Tensor, a) -> Tensor:
        return type(self)(tf.convert_to_tensor(a))

    def _concatenate(self: Tensor, tensors, axis=0) -> Tensor:
        # concatenates only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return type(self)(tf.concat(tensors, axis=axis))

    def _stack(self: Tensor, tensors, axis=0) -> Tensor:
        # stacks only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return type(self)(tf.stack(tensors, axis=axis))

    def transpose(self: Tensor, axes=None) -> Tensor:
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return type(self)(tf.transpose(self.raw, perm=axes))

    def bool(self: Tensor) -> Tensor:
        return self.astype(tf.bool)

    def all(self: Tensor, axis=None, keepdims=False) -> Tensor:
        assert_bool(self)
        return type(self)(tf.reduce_all(self.raw, axis=axis, keepdims=keepdims))

    def any(self: Tensor, axis=None, keepdims=False) -> Tensor:
        assert_bool(self)
        return type(self)(tf.reduce_any(self.raw, axis=axis, keepdims=keepdims))

    def logical_and(self: Tensor, other) -> Tensor:
        assert_bool(self)
        assert_bool(other)
        return type(self)(tf.logical_and(self.raw, unwrap_(other)))

    def logical_or(self: Tensor, other) -> Tensor:
        assert_bool(self)
        assert_bool(other)
        return type(self)(tf.logical_or(self.raw, unwrap_(other)))

    def logical_not(self: Tensor) -> Tensor:
        assert_bool(self)
        return type(self)(tf.logical_not(self.raw))

    def exp(self: Tensor) -> Tensor:
        return type(self)(tf.exp(self.raw))

    def log(self: Tensor) -> Tensor:
        return type(self)(tf.math.log(self.raw))

    def log2(self: Tensor) -> Tensor:
        return type(self)(tf.math.log(self.raw) / tf.math.log(2.0))

    def log10(self: Tensor) -> Tensor:
        return type(self)(tf.math.log(self.raw) / tf.math.log(10.0))

    def log1p(self: Tensor) -> Tensor:
        return type(self)(tf.math.log1p(self.raw))

    def tile(self: Tensor, multiples) -> Tensor:
        multiples = unwrap_(multiples)
        if len(multiples) != self.ndim:
            raise ValueError("multiples requires one entry for each dimension")
        return type(self)(tf.tile(self.raw, multiples))

    def softmax(self: Tensor, axis=-1) -> Tensor:
        return type(self)(tf.nn.softmax(self.raw, axis=axis))

    def log_softmax(self: Tensor, axis=-1) -> Tensor:
        return type(self)(tf.nn.log_softmax(self.raw, axis=axis))

    def squeeze(self: Tensor, axis=None) -> Tensor:
        return type(self)(tf.squeeze(self.raw, axis=axis))

    def expand_dims(self: Tensor, axis=None) -> Tensor:
        return type(self)(tf.expand_dims(self.raw, axis=axis))

    @samedevice
    def full(self: Tensor, shape, value) -> Tensor:
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return type(self)(tf.fill(shape, value))

    def index_update(self: Tensor, indices, values) -> Tensor:
        indices, values = unwrap_(indices, values)
        if isinstance(indices, tuple):
            indices = unwrap_(indices)

        x = self.raw
        if isinstance(indices, int):
            return type(self)(tf.tensor_scatter_nd_update(x, [[indices]], values[None]))
        elif isinstance(indices, tuple) and any(
            isinstance(idx, slice) for idx in indices
        ):
            if (
                len(indices) == x.ndim == 2
                and indices[0] == index[:]
                and not isinstance(indices[1], slice)
            ):
                x = tf.transpose(x)
                result = tf.tensor_scatter_nd_update(x, [[indices[-1]]], values[None])
                return type(self)(tf.transpose(result))
            else:
                raise NotImplementedError  # pragma: no cover
        elif isinstance(indices, tuple):
            if all(idx.dtype in [tf.int32, tf.int64] for idx in indices):
                indices = [
                    tf.cast(idx, tf.int64) if idx.dtype == tf.int32 else idx
                    for idx in indices
                ]
            return type(self)(
                tf.tensor_scatter_nd_update(x, tf.stack(indices, axis=-1), values)
            )
        else:
            raise ValueError  # pragma: no cover

    @samedevice
    def arange(self: Tensor, start, stop=None, step=None) -> Tensor:
        if step is None:
            step = 1
        if stop is None:
            stop = start
            start = 0
        return type(self)(tf.range(start, stop, step))

    def cumsum(self: Tensor, axis=None) -> Tensor:
        if axis is None:
            x = tf.reshape(self.raw, (-1,))
            return type(self)(tf.cumsum(x, axis=0))
        return type(self)(tf.cumsum(self.raw, axis=axis))

    def flip(self: Tensor, axis=None) -> Tensor:
        if axis is None:
            axis = tuple(range(self.ndim))
        if not isinstance(axis, Iterable):
            axis = (axis,)
        return type(self)(tf.reverse(self.raw, axis=axis))

    def meshgrid(self: Tensor, *tensors, indexing="xy") -> Tuple[Tensor, ...]:
        tensors = unwrap_(tensors)
        outputs = tf.meshgrid(self.raw, *tensors, indexing=indexing)
        return tuple(type(self)(out) for out in outputs)

    def pad(self: Tensor, paddings, mode="constant", value=0) -> Tensor:
        if len(paddings) != self.ndim:
            raise ValueError("pad requires a tuple for each dimension")
        for p in paddings:
            if len(p) != 2:
                raise ValueError("pad requires a tuple for each dimension")
        if not (mode == "constant" or mode == "reflect"):
            raise ValueError("pad requires mode 'constant' or 'reflect'")
        if mode == "reflect":
            # PyTorch's pad has limited support for 'reflect' padding
            if self.ndim != 3 and self.ndim != 4:
                raise NotImplementedError  # pragma: no cover
            k = self.ndim - 2
            if paddings[:k] != ((0, 0),) * k:
                raise NotImplementedError  # pragma: no cover
        return type(self)(tf.pad(self.raw, paddings, mode=mode, constant_values=value))

    def isnan(self: Tensor) -> Tensor:
        return type(self)(tf.math.is_nan(self.raw))

    def isinf(self: Tensor) -> Tensor:
        return type(self)(tf.math.is_inf(self.raw))

    def crossentropy(self: Tensor, labels: Tensor) -> Tensor:
        if self.ndim != 2:
            raise ValueError("crossentropy only supported for 2D logits tensors")
        if self.shape[:1] != labels.shape:
            raise ValueError("labels must be 1D and must match the length of logits")
        return type(self)(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels.raw, self.raw)
        )

    def _value_and_grad_fn(self: Tensor, f, has_aux=False) -> Any:
        def value_and_grad(x, *args, **kwargs):
            # using tf.identity to make x independent from possible other instances of x in args
            x = x.raw
            x = tf.identity(x)
            x = TensorFlowTensor(x)
            assert isinstance(x, TensorFlowTensor)
            with tf.GradientTape() as tape:
                tape.watch(x.raw)
                if has_aux:
                    loss, aux = f(x, *args, **kwargs)
                else:
                    loss = f(x, *args, **kwargs)
            grad = tape.gradient(loss.raw, x.raw)
            grad = TensorFlowTensor(grad)
            assert grad.shape == x.shape
            if has_aux:
                return loss, aux, grad
            else:
                return loss, grad

        return value_and_grad

    def sign(self: Tensor) -> Tensor:
        return type(self)(tf.sign(self.raw))

    def sqrt(self: Tensor) -> Tensor:
        return type(self)(tf.sqrt(self.raw))

    def tanh(self: Tensor) -> Tensor:
        return type(self)(tf.tanh(self.raw))

    def float32(self: Tensor) -> Tensor:
        return self.astype(tf.float32)

    def where(self: Tensor, x, y) -> Tensor:
        x, y = unwrap_(x, y)
        return type(self)(tf.where(self.raw, x, y))

    def matmul(self: Tensor, other) -> Tensor:
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(
                f"matmul requires both tensors to be 2D, got {self.ndim}D and {other.ndim}D"
            )
        return type(self)(tf.matmul(self.raw, other.raw))

    @common_dtype
    def __lt__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__lt__(unwrap_(other)))

    @common_dtype
    def __le__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__le__(unwrap_(other)))

    @common_dtype
    def __eq__(self: Tensor, other) -> Tensor:  # type: ignore
        return type(self)(self.raw.__eq__(unwrap_(other)))

    @common_dtype
    def __ne__(self: Tensor, other) -> Tensor:  # type: ignore
        return type(self)(self.raw.__ne__(unwrap_(other)))

    @common_dtype
    def __gt__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__gt__(unwrap_(other)))

    @common_dtype
    def __ge__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__ge__(unwrap_(other)))

    def __getitem__(self: Tensor, index) -> Tensor:
        if isinstance(index, tuple):
            index = tuple(x.raw if istensor(x) else x for x in index)
            tensors = any(
                isinstance(x, tf.Tensor) or isinstance(x, np.ndarray) for x in index
            )
            if tensors:
                # workaround for missing support for this in TensorFlow
                index = tf.convert_to_tensor(index)
                index = tf.transpose(index)
                return type(self)(tf.gather_nd(self.raw, index))
        elif istensor(index):
            return type(self)(tf.gather(self.raw, index.raw))
        return type(self)(self.raw.__getitem__(index))
