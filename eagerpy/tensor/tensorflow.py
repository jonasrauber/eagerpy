from .base import AbstractTensor
from .base import unwrapin
from .base import wrapout

import functools
import numpy as np


def samedevice(f):
    import tensorflow as tf

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
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

    @property
    def ndim(self):
        return self.tensor.ndim

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
        return self.backend.random.uniform(
            shape, minval=low, maxval=high, dtype=self.tensor.dtype
        )

    @samedevice
    @wrapout
    def normal(self, shape, mean=0.0, stddev=1.0):
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
