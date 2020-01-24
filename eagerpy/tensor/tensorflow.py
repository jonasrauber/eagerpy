from .base import AbstractTensor
from .base import unwrapin
from .base import wrapout
from .. import index

import functools
import numpy as np
from collections.abc import Iterable


def samedevice(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        import tensorflow as tf

        with tf.device(self.tensor.device):
            out = f(self, *args, **kwargs)
        return out

    return wrapper


class TensorFlowTensor(AbstractTensor):
    def __init__(self, tensor):
        import tensorflow

        super().__init__(tensor)
        self.backend = tensorflow

    @unwrapin
    @wrapout
    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = tuple(
                x.tensor if isinstance(x, self.__class__) else x for x in index
            )
            tensors = any(
                isinstance(x, self.backend.Tensor) or isinstance(x, np.ndarray)
                for x in index
            )
            if tensors:
                # workaround for missing support for this in TensorFlow
                index = self.backend.convert_to_tensor(index)
                index = self.backend.transpose(index)
                return self.backend.gather_nd(self.tensor, index)
        return self.tensor.__getitem__(index)

    def numpy(self):
        return self.tensor.numpy()

    def item(self):
        return self.numpy().item()

    @property
    def shape(self):
        return tuple(self.tensor.shape.as_list())

    @wrapout
    def reshape(self, shape):
        return self.backend.reshape(self.tensor, shape)

    @wrapout
    def astype(self, dtype):
        return self.backend.cast(self.tensor, dtype)

    @wrapout
    def clip(self, min_, max_):
        return self.backend.clip_by_value(self.tensor, min_, max_)

    @wrapout
    def square(self):
        return self.backend.square(self.tensor)

    @wrapout
    def arctanh(self):
        return self.backend.atanh(self.tensor)

    @wrapout
    def sum(self, axis=None, keepdims=False):
        return self.backend.reduce_sum(self.tensor, axis=axis, keepdims=keepdims)

    @wrapout
    def mean(self, axis=None, keepdims=False):
        return self.backend.reduce_mean(self.tensor, axis=axis, keepdims=keepdims)

    @wrapout
    def min(self, axis=None, keepdims=False):
        return self.backend.reduce_min(self.tensor, axis=axis, keepdims=keepdims)

    @wrapout
    def max(self, axis=None, keepdims=False):
        return self.backend.reduce_max(self.tensor, axis=axis, keepdims=keepdims)

    @unwrapin
    @wrapout
    def minimum(self, other):
        return self.backend.minimum(self.tensor, other)

    @unwrapin
    @wrapout
    def maximum(self, other):
        return self.backend.maximum(self.tensor, other)

    @wrapout
    def argmin(self, axis=None):
        return self.backend.argmin(self.tensor, axis=axis)

    @wrapout
    def argmax(self, axis=None):
        return self.backend.argmax(self.tensor, axis=axis)

    @wrapout
    def argsort(self, axis=-1):
        return self.backend.argsort(self.tensor, axis=axis)

    @samedevice
    @wrapout
    def uniform(self, shape, low=0.0, high=1.0):
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return self.backend.random.uniform(
            shape, minval=low, maxval=high, dtype=self.tensor.dtype
        )

    @samedevice
    @wrapout
    def normal(self, shape, mean=0.0, stddev=1.0):
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return self.backend.random.normal(
            shape, mean=mean, stddev=stddev, dtype=self.tensor.dtype
        )

    @samedevice
    @wrapout
    def ones(self, shape):
        return self.backend.ones(shape, dtype=self.tensor.dtype)

    @samedevice
    @wrapout
    def zeros(self, shape):
        return self.backend.zeros(shape, dtype=self.tensor.dtype)

    @wrapout
    def ones_like(self):
        return self.backend.ones_like(self.tensor)

    @wrapout
    def zeros_like(self):
        return self.backend.zeros_like(self.tensor)

    @wrapout
    def full_like(self, fill_value):
        fill_value = self.backend.cast(fill_value, self.tensor.dtype)
        return self.backend.fill(self.tensor.shape, fill_value)

    @samedevice
    @unwrapin
    @wrapout
    def onehot_like(self, indices, *, value=1):
        assert self.tensor.ndim == 2
        assert indices.ndim == 1
        assert len(indices) == len(self.tensor)
        value = self.backend.cast(value, self.tensor.dtype)
        return self.backend.one_hot(
            indices,
            depth=self.tensor.shape[-1],
            on_value=value,
            dtype=self.tensor.dtype,
        )

    @samedevice
    @wrapout
    def from_numpy(self, a):
        return self.backend.convert_to_tensor(a)

    @wrapout
    def _concatenate(self, tensors, axis=0):
        # concatenates only "tensors", but not "self"
        tensors = [t.tensor if isinstance(t, self.__class__) else t for t in tensors]
        return self.backend.concat(tensors, axis=axis)

    @wrapout
    def _stack(self, tensors, axis=0):
        # stacks only "tensors", but not "self"
        tensors = [t.tensor if isinstance(t, self.__class__) else t for t in tensors]
        return self.backend.stack(tensors, axis=axis)

    @wrapout
    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return self.backend.transpose(self.tensor, perm=axes)

    def bool(self):
        return self.astype(self.backend.bool)

    @wrapout
    def all(self, axis=None, keepdims=False):
        assert self.dtype == self.backend.bool
        return self.backend.reduce_all(self.tensor, axis=axis, keepdims=keepdims)

    @wrapout
    def any(self, axis=None, keepdims=False):
        assert self.dtype == self.backend.bool
        return self.backend.reduce_any(self.tensor, axis=axis, keepdims=keepdims)

    @unwrapin
    @wrapout
    def logical_and(self, other):
        assert self.dtype == self.backend.bool
        return self.backend.logical_and(self.tensor, other)

    @unwrapin
    @wrapout
    def logical_or(self, other):
        assert self.dtype == self.backend.bool
        return self.backend.logical_or(self.tensor, other)

    @wrapout
    def logical_not(self):
        assert self.dtype == self.backend.bool
        return self.backend.logical_not(self.tensor)

    @wrapout
    def exp(self):
        return self.backend.exp(self.tensor)

    @wrapout
    def log(self):
        return self.backend.math.log(self.tensor)

    @wrapout
    def log2(self):
        return self.backend.math.log(self.tensor) / self.backend.math.log(2.0)

    @wrapout
    def log10(self):
        return self.backend.math.log(self.tensor) / self.backend.math.log(10.0)

    @wrapout
    def log1p(self):
        return self.backend.math.log1p(self.tensor)

    @unwrapin
    @wrapout
    def tile(self, multiples):
        assert len(multiples) == self.ndim
        return self.backend.tile(self.tensor, multiples)

    @wrapout
    def softmax(self, axis=-1):
        return self.backend.nn.softmax(self.tensor, axis=axis)

    @wrapout
    def log_softmax(self, axis=-1):
        return self.backend.nn.log_softmax(self.tensor, axis=axis)

    @wrapout
    def squeeze(self, axis=None):
        return self.backend.squeeze(self.tensor, axis=axis)

    @wrapout
    def expand_dims(self, axis=None):
        return self.backend.expand_dims(self.tensor, axis=axis)

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
            indices = tuple(
                t.tensor if isinstance(t, self.__class__) else t for t in indices
            )

        x = self.tensor
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
                raise NotImplementedError
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
            raise ValueError

    @samedevice
    @wrapout
    def arange(self, *args, **kwargs):
        return self.backend.range(*args, **kwargs)

    @wrapout
    def cumsum(self, axis=None):
        if axis is None:
            x = self.backend.reshape(self.tensor, (-1,))
            return self.backend.cumsum(x, axis=0)
        return self.backend.cumsum(self.tensor, axis=axis)

    @wrapout
    def flip(self, axis=None):
        if axis is None:
            axis = tuple(range(self.ndim))
        if not isinstance(axis, Iterable):
            axis = (axis,)
        return self.backend.reverse(self.tensor, axis=axis)

    @unwrapin
    def meshgrid(self, *tensors, indexing="xy"):
        outputs = self.backend.meshgrid(self.tensor, *tensors, indexing=indexing)
        outputs = tuple(self.__class__(out) for out in outputs)
        return outputs

    @wrapout
    def pad(self, paddings, mode="constant", value=0):
        assert len(paddings) == self.ndim
        for p in paddings:
            assert len(p) == 2
        assert mode == "constant" or mode == "reflect"
        if mode == "reflect":
            # PyTorch's pad has limited support for 'reflect' padding
            if self.ndim != 3 and self.ndim != 4:
                raise NotImplementedError
            k = self.ndim - 2
            if paddings[:k] != ((0, 0),) * k:
                raise NotImplementedError
        return self.backend.pad(self.tensor, paddings, mode=mode, constant_values=value)

    @wrapout
    def isnan(self):
        return self.backend.math.is_nan(self.tensor)

    @wrapout
    def isinf(self):
        return self.backend.math.is_inf(self.tensor)

    @unwrapin
    @wrapout
    def crossentropy(self, labels):
        logits = self.tensor
        assert logits.ndim == 2
        assert logits.shape[:1] == labels.shape
        return self.backend.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    def _value_and_grad_fn(self, f, has_aux=False):
        def value_and_grad(x, *args, **kwargs):
            # using tf.identity to make x independent from possible other instances of x in args
            x = x.tensor
            x = self.backend.identity(x)
            x = TensorFlowTensor(x)
            assert isinstance(x, TensorFlowTensor)
            with self.backend.GradientTape() as tape:
                tape.watch(x.tensor)
                if has_aux:
                    loss, aux = f(x, *args, **kwargs)
                else:
                    loss = f(x, *args, **kwargs)
            grad = tape.gradient(loss.tensor, x.tensor)
            grad = TensorFlowTensor(grad)
            assert grad.shape == x.shape
            if has_aux:
                return loss, aux, grad
            else:
                return loss, grad

        return value_and_grad
