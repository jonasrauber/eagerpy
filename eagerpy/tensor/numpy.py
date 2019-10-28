from .base import AbstractTensor
from .base import wrapout


class NumPyTensor(AbstractTensor):
    def __init__(self, tensor):
        import numpy

        super().__init__(tensor)
        self.backend = numpy

    def numpy(self):
        return self.tensor

    @property
    def shape(self):
        return self.tensor.shape

    @wrapout
    def reshape(self, shape):
        return self.tensor.reshape(shape)

    def __len__(self):
        return self.tensor.__len__()

    @property
    def ndim(self):
        return self.tensor.ndim

    @wrapout
    def astype(self, dtype):
        return self.tensor.astype(dtype)

    @wrapout
    def clip(self, min_, max_):
        return self.backend.clip(self.tensor, min_, max_)

    @wrapout
    def square(self):
        return self.backend.square(self.tensor)

    @wrapout
    def arctanh(self):
        return self.backend.arctanh(self.tensor)

    @wrapout
    def sum(self, axis=None, keepdims=False):
        return self.tensor.sum(axis=axis, keepdims=keepdims)

    @wrapout
    def mean(self, axis=None, keepdims=False):
        return self.tensor.mean(axis=axis, keepdims=keepdims)

    @wrapout
    def min(self, axis=None, keepdims=False):
        return self.tensor.min(axis=axis, keepdims=keepdims)

    @wrapout
    def max(self, axis=None, keepdims=False):
        return self.tensor.max(axis=axis, keepdims=keepdims)

    @wrapout
    def minimum(self, other):
        if hasattr(other, "tensor"):
            other = other.tensor
        return self.backend.minimum(self.tensor, other)

    @wrapout
    def maximum(self, other):
        if hasattr(other, "tensor"):
            other = other.tensor
        return self.backend.maximum(self.tensor, other)

    @wrapout
    def argmin(self, axis=None):
        return self.tensor.argmin(axis=axis)

    @wrapout
    def argmax(self, axis=None):
        return self.tensor.argmax(axis=axis)

    @wrapout
    def argsort(self, axis=-1):
        return self.tensor.argsort(axis=axis)

    @classmethod
    def uniform(cls, shape, low=0.0, high=1.0):
        return cls(cls.backend.random.rand(*shape) * (high - low) + low)

    @classmethod
    def normal(cls, shape, mean=0.0, stddev=1.0):
        return cls(cls.backend.random.randn(*shape) * stddev + mean)
