from .base import AbstractBaseTensor
from .base import unwrapin
from .base import wrapout
from .base import unwrap_

from .tensor import istensor

from .. import index

import functools
import numpy as np
from collections.abc import Iterable


def samedevice(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        import tensorflow as tf

        with tf.device(self.raw.device):
            out = f(self, *args, **kwargs)
        return out

    return wrapper


def common_dtype(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        dtypes = {self.dtype} | {arg.dtype for arg in args if istensor(arg)}
        if len(dtypes) == 1:
            # all dtypes are the same, nothing more to do
            return f(self, *args, **kwargs)
        numpy_dtypes = [np.dtype(dtype.name) for dtype in dtypes]
        common = np.find_common_type(numpy_dtypes, [])
        common = getattr(self.backend, common.name)
        if self.dtype != common:
            self = self.astype(common)
        args = [
            arg.astype(common) if istensor(arg) and arg.dtype != common else arg
            for arg in args
        ]
        return f(self, *args, **kwargs)

    return wrapper


def assert_bool(x):
    if not istensor(x):
        return
    if x.dtype != x.backend.bool:
        raise ValueError(f"all only supports dtype bool, consider t.bool().all()")


class TensorFlowTensor(AbstractBaseTensor):
    def __init__(self, tensor):
        import tensorflow

        super().__init__(tensor)
        self.backend = tensorflow

    @common_dtype
    @unwrapin
    @wrapout
    def __lt__(self, other):
        return self.raw.__lt__(other)

    @common_dtype
    @unwrapin
    @wrapout
    def __le__(self, other):
        return self.raw.__le__(other)

    @common_dtype
    @unwrapin
    @wrapout
    def __eq__(self, other):
        return self.raw.__eq__(other)

    @common_dtype
    @unwrapin
    @wrapout
    def __ne__(self, other):
        return self.raw.__ne__(other)

    @common_dtype
    @unwrapin
    @wrapout
    def __gt__(self, other):
        return self.raw.__gt__(other)

    @common_dtype
    @unwrapin
    @wrapout
    def __ge__(self, other):
        return self.raw.__ge__(other)

    @wrapout
    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = tuple(x.raw if istensor(x) else x for x in index)
            tensors = any(
                isinstance(x, self.backend.Tensor) or isinstance(x, np.ndarray)
                for x in index
            )
            if tensors:
                # workaround for missing support for this in TensorFlow
                index = self.backend.convert_to_tensor(index)
                index = self.backend.transpose(index)
                return self.backend.gather_nd(self.raw, index)
        elif istensor(index):
            return self.backend.gather(self.raw, index.raw)
        return self.raw.__getitem__(index)

    def numpy(self):
        return self.raw.numpy()

    def item(self):
        return self.numpy().item()

    @property
    def shape(self):
        return tuple(self.raw.shape.as_list())

    @wrapout
    def reshape(self, shape):
        return self.backend.reshape(self.raw, shape)

    @wrapout
    def astype(self, dtype):
        return self.backend.cast(self.raw, dtype)

    @wrapout
    def clip(self, min_, max_):
        return self.backend.clip_by_value(self.raw, min_, max_)

    @wrapout
    def square(self):
        return self.backend.square(self.raw)

    @wrapout
    def arctanh(self):
        return self.backend.atanh(self.raw)

    @wrapout
    def sum(self, axis=None, keepdims=False):
        if self.raw.dtype == self.backend.bool:
            return self.astype(self.backend.int64).sum(axis=axis, keepdims=keepdims)
        return self.backend.reduce_sum(self.raw, axis=axis, keepdims=keepdims)

    @wrapout
    def mean(self, axis=None, keepdims=False):
        return self.backend.reduce_mean(self.raw, axis=axis, keepdims=keepdims)

    @wrapout
    def min(self, axis=None, keepdims=False):
        return self.backend.reduce_min(self.raw, axis=axis, keepdims=keepdims)

    @wrapout
    def max(self, axis=None, keepdims=False):
        return self.backend.reduce_max(self.raw, axis=axis, keepdims=keepdims)

    @unwrapin
    @wrapout
    def minimum(self, other):
        return self.backend.minimum(self.raw, other)

    @unwrapin
    @wrapout
    def maximum(self, other):
        return self.backend.maximum(self.raw, other)

    @wrapout
    def argmin(self, axis=None):
        return self.backend.argmin(self.raw, axis=axis)

    @wrapout
    def argmax(self, axis=None):
        return self.backend.argmax(self.raw, axis=axis)

    @wrapout
    def argsort(self, axis=-1):
        return self.backend.argsort(self.raw, axis=axis)

    @samedevice
    @wrapout
    def uniform(self, shape, low=0.0, high=1.0):
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return self.backend.random.uniform(
            shape, minval=low, maxval=high, dtype=self.raw.dtype
        )

    @samedevice
    @wrapout
    def normal(self, shape, mean=0.0, stddev=1.0):
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return self.backend.random.normal(
            shape, mean=mean, stddev=stddev, dtype=self.raw.dtype
        )

    @samedevice
    @wrapout
    def ones(self, shape):
        return self.backend.ones(shape, dtype=self.raw.dtype)

    @samedevice
    @wrapout
    def zeros(self, shape):
        return self.backend.zeros(shape, dtype=self.raw.dtype)

    @wrapout
    def ones_like(self):
        return self.backend.ones_like(self.raw)

    @wrapout
    def zeros_like(self):
        return self.backend.zeros_like(self.raw)

    @wrapout
    def full_like(self, fill_value):
        fill_value = self.backend.cast(fill_value, self.raw.dtype)
        return self.backend.fill(self.raw.shape, fill_value)

    @samedevice
    @wrapout
    def onehot_like(self, indices, *, value=1):
        if self.ndim != 2:
            raise ValueError("onehot_like only supported for 2D tensors")
        if indices.ndim != 1:
            raise ValueError("onehot_like requires 1D indices")
        if len(indices) != len(self.raw):
            raise ValueError("length of indices must match length of tensor")
        value = self.backend.cast(value, self.raw.dtype)
        return self.backend.one_hot(
            indices.raw, depth=self.raw.shape[-1], on_value=value, dtype=self.raw.dtype
        )

    @samedevice
    @wrapout
    def from_numpy(self, a):
        return self.backend.convert_to_tensor(a)

    @wrapout
    def _concatenate(self, tensors, axis=0):
        # concatenates only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return self.backend.concat(tensors, axis=axis)

    @wrapout
    def _stack(self, tensors, axis=0):
        # stacks only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return self.backend.stack(tensors, axis=axis)

    @wrapout
    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return self.backend.transpose(self.raw, perm=axes)

    def bool(self):
        return self.astype(self.backend.bool)

    @wrapout
    def all(self, axis=None, keepdims=False):
        assert_bool(self)
        return self.backend.reduce_all(self.raw, axis=axis, keepdims=keepdims)

    @wrapout
    def any(self, axis=None, keepdims=False):
        assert_bool(self)
        return self.backend.reduce_any(self.raw, axis=axis, keepdims=keepdims)

    @wrapout
    def logical_and(self, other):
        assert_bool(self)
        assert_bool(other)
        return self.backend.logical_and(self.raw, unwrap_(other))

    @wrapout
    def logical_or(self, other):
        assert_bool(self)
        assert_bool(other)
        return self.backend.logical_or(self.raw, unwrap_(other))

    @wrapout
    def logical_not(self):
        assert_bool(self)
        return self.backend.logical_not(self.raw)

    @wrapout
    def exp(self):
        return self.backend.exp(self.raw)

    @wrapout
    def log(self):
        return self.backend.math.log(self.raw)

    @wrapout
    def log2(self):
        return self.backend.math.log(self.raw) / self.backend.math.log(2.0)

    @wrapout
    def log10(self):
        return self.backend.math.log(self.raw) / self.backend.math.log(10.0)

    @wrapout
    def log1p(self):
        return self.backend.math.log1p(self.raw)

    @unwrapin
    @wrapout
    def tile(self, multiples):
        if len(multiples) != self.ndim:
            raise ValueError("multiples requires one entry for each dimension")
        return self.backend.tile(self.raw, multiples)

    @wrapout
    def softmax(self, axis=-1):
        return self.backend.nn.softmax(self.raw, axis=axis)

    @wrapout
    def log_softmax(self, axis=-1):
        return self.backend.nn.log_softmax(self.raw, axis=axis)

    @wrapout
    def squeeze(self, axis=None):
        return self.backend.squeeze(self.raw, axis=axis)

    @wrapout
    def expand_dims(self, axis=None):
        return self.backend.expand_dims(self.raw, axis=axis)

    @samedevice
    @wrapout
    def full(self, shape, value):
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return self.backend.fill(shape, value)

    @unwrapin
    @wrapout
    def index_update(self, indices, values):
        if isinstance(indices, tuple):
            indices = tuple(t.raw if istensor(t) else t for t in indices)

        x = self.raw
        if isinstance(indices, int):
            return self.backend.tensor_scatter_nd_update(x, [[indices]], values[None])
        elif isinstance(indices, tuple) and any(
            isinstance(idx, slice) for idx in indices
        ):
            if (
                len(indices) == x.ndim == 2
                and indices[0] == index[:]
                and not isinstance(indices[1], slice)
            ):
                x = self.backend.transpose(x)
                result = self.backend.tensor_scatter_nd_update(
                    x, [[indices[-1]]], values[None]
                )
                return self.backend.transpose(result)
            else:
                raise NotImplementedError  # pragma: no cover
        elif isinstance(indices, tuple):
            if all(
                idx.dtype in [self.backend.int32, self.backend.int64] for idx in indices
            ):
                indices = [
                    self.backend.cast(idx, self.backend.int64)
                    if idx.dtype == self.backend.int32
                    else idx
                    for idx in indices
                ]
            return self.backend.tensor_scatter_nd_update(
                x, self.backend.stack(indices, axis=-1), values
            )
        else:
            raise ValueError  # pragma: no cover

    @samedevice
    @wrapout
    def arange(self, *args, **kwargs):
        return self.backend.range(*args, **kwargs)

    @wrapout
    def cumsum(self, axis=None):
        if axis is None:
            x = self.backend.reshape(self.raw, (-1,))
            return self.backend.cumsum(x, axis=0)
        return self.backend.cumsum(self.raw, axis=axis)

    @wrapout
    def flip(self, axis=None):
        if axis is None:
            axis = tuple(range(self.ndim))
        if not isinstance(axis, Iterable):
            axis = (axis,)
        return self.backend.reverse(self.raw, axis=axis)

    @unwrapin
    def meshgrid(self, *tensors, indexing="xy"):
        outputs = self.backend.meshgrid(self.raw, *tensors, indexing=indexing)
        outputs = tuple(self.__class__(out) for out in outputs)
        return outputs

    @wrapout
    def pad(self, paddings, mode="constant", value=0):
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
        return self.backend.pad(self.raw, paddings, mode=mode, constant_values=value)

    @wrapout
    def isnan(self):
        return self.backend.math.is_nan(self.raw)

    @wrapout
    def isinf(self):
        return self.backend.math.is_inf(self.raw)

    @unwrapin
    @wrapout
    def crossentropy(self, labels):
        logits = self.raw
        if logits.ndim != 2:
            raise ValueError("crossentropy only supported for 2D logits tensors")
        if logits.shape[:1] != labels.shape:
            raise ValueError("labels must be 1D and must match the length of logits")
        return self.backend.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    def _value_and_grad_fn(self, f, has_aux=False):
        def value_and_grad(x, *args, **kwargs):
            # using tf.identity to make x independent from possible other instances of x in args
            x = x.raw
            x = self.backend.identity(x)
            x = TensorFlowTensor(x)
            assert isinstance(x, TensorFlowTensor)
            with self.backend.GradientTape() as tape:
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

    @wrapout
    def sign(self):
        return self.backend.sign(self.raw)

    @wrapout
    def sqrt(self):
        return self.backend.sqrt(self.raw)

    @wrapout
    def tanh(self):
        return self.backend.tanh(self.raw)

    def float32(self):
        return self.astype(self.backend.float32)

    @unwrapin
    @wrapout
    def where(self, x, y):
        return self.backend.where(self.raw, x, y)

    @wrapout
    def matmul(self, other):
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(
                f"matmul requires both tensors to be 2D, got {self.ndim}D and {other.ndim}D"
            )
        return self.backend.matmul(self.raw, other.raw)
