from .base import AbstractTensor
from .base import wrapout


class TensorFlowTensor(AbstractTensor):
    def __init__(self, tensor):
        import tensorflow

        super().__init__(tensor)
        self.backend = tensorflow

    def numpy(self):
        return self.tensor.numpy()

    @property
    def shape(self):
        return tuple(self.tensor.shape.as_list())

    @wrapout
    def reshape(self, shape):
        return self.backend.reshape(self.tensor, shape)

    def __len__(self):
        return self.shape[0]

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
        return self.backend.argmin(self.tensor, axis=axis)

    @wrapout
    def argmax(self, axis=None):
        return self.backend.argmax(self.tensor, axis=axis)

    @wrapout
    def argsort(self, axis=-1):
        return self.backend.argsort(self.tensor, axis=axis)

    @classmethod
    def uniform(cls, shape, low=0.0, high=1.0):
        return cls(cls.backend.random.uniform(shape, minval=low, maxval=high))

    @classmethod
    def normal(cls, shape, mean=0.0, stddev=1.0):
        return cls(cls.backend.random.normal(shape, mean=mean, stddev=stddev))
