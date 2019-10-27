from .base import AbstractTensor
from .base import wrapout

from collections.abc import Iterable


class PyTorchTensor(AbstractTensor):
    def __init__(self, tensor):
        import torch

        super().__init__(tensor)
        self.backend = torch

    def numpy(self):
        return self.tensor.cpu().numpy()

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
        return len(self.shape)

    @wrapout
    def astype(self, dtype):
        return self.tensor.to(dtype)

    @wrapout
    def clip(self, min_, max_):
        return self.tensor.clamp(min_, max_)

    @wrapout
    def square(self):
        return self.tensor ** 2

    @wrapout
    def arctanh(self):
        """
        implement once this issue has been fixed:
        https://github.com/pytorch/pytorch/issues/10324
        """
        raise NotImplementedError

    @wrapout
    def sum(self, axis=None, keepdims=False):
        if axis is None:
            assert not keepdims
            return self.tensor.sum()
        return self.tensor.sum(dim=axis, keepdim=keepdims)

    @wrapout
    def mean(self, axis=None, keepdims=False):
        if axis is None:
            assert not keepdims
            return self.tensor.mean()
        return self.tensor.mean(dim=axis, keepdim=keepdims)

    @wrapout
    def min(self, axis=None, keepdims=False):
        """
        simplify once this issue has been fixed:
        https://github.com/pytorch/pytorch/issues/28213
        """
        if axis is None:
            assert not keepdims
            return self.tensor.min()
        if not isinstance(axis, Iterable):
            axis = (axis,)
        axis = reversed(sorted(axis))
        x = self.tensor
        for i in axis:
            x, _ = x.min(i, keepdim=keepdims)
        return x

    @wrapout
    def max(self, axis=None, keepdims=False):
        """
        simplify once this issue has been fixed:
        https://github.com/pytorch/pytorch/issues/28213
        """
        if axis is None:
            assert not keepdims
            return self.tensor.max()
        if not isinstance(axis, Iterable):
            axis = (axis,)
        axis = reversed(sorted(axis))
        x = self.tensor
        for i in axis:
            x, _ = x.max(i, keepdim=keepdims)
        return x

    @wrapout
    def minimum(self, other):
        if hasattr(other, "tensor"):
            other = other.tensor
        else:
            other = self.backend.ones_like(self.tensor) * other
        return self.backend.min(self.tensor, other)

    @wrapout
    def maximum(self, other):
        if hasattr(other, "tensor"):
            other = other.tensor
        else:
            other = self.backend.ones_like(self.tensor) * other
        return self.backend.max(self.tensor, other)

    @wrapout
    def argmin(self, axis=None):
        return self.tensor.argmin(dim=axis)

    @wrapout
    def argmax(self, axis=None):
        return self.tensor.argmax(dim=axis)

    @wrapout
    def argsort(self, axis=-1):
        return self.tensor.argsort(dim=axis)

    @classmethod
    def uniform(cls, shape, low=0.0, high=1.0):
        return cls(cls.backend.rand(*shape) * (high - low) + low)

    @classmethod
    def normal(cls, shape, mean=0.0, stddev=1.0):
        return cls(cls.backend.randn(*shape) * stddev + mean)
