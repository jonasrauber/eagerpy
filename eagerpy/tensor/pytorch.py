from .base import AbstractTensor
from .base import wrapout
from .base import istensor
from .base import unwrapin

import numpy as np
from collections.abc import Iterable


class PyTorchTensor(AbstractTensor):
    def __init__(self, tensor):
        import torch

        super().__init__(tensor)
        self.backend = torch

    def numpy(self):
        return self.tensor.cpu().numpy()

    def item(self):
        return self.tensor.item()

    @property
    def shape(self):
        return self.tensor.shape

    @wrapout
    def reshape(self, shape):
        return self.tensor.reshape(shape)

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
        improve once this issue has been fixed:
        https://github.com/pytorch/pytorch/issues/10324
        """
        return 0.5 * self.backend.log((1 + self.tensor) / (1 - self.tensor))

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
        if istensor(other):
            other = other.tensor
        else:
            other = self.backend.ones_like(self.tensor) * other
        return self.backend.min(self.tensor, other)

    @wrapout
    def maximum(self, other):
        if istensor(other):
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

    @wrapout
    def uniform(self, shape, low=0.0, high=1.0):
        return (
            self.backend.rand(
                *shape, dtype=self.tensor.dtype, device=self.tensor.device
            )
            * (high - low)
            + low
        )

    @wrapout
    def normal(self, shape, mean=0.0, stddev=1.0):
        return (
            self.backend.randn(
                *shape, dtype=self.tensor.dtype, device=self.tensor.device
            )
            * stddev
            + mean
        )

    @wrapout
    def ones(self, shape):
        return self.backend.ones(
            *shape, dtype=self.tensor.dtype, device=self.tensor.device
        )

    @wrapout
    def zeros(self, shape):
        return self.backend.zeros(
            *shape, dtype=self.tensor.dtype, device=self.tensor.device
        )

    @wrapout
    def ones_like(self):
        return self.backend.ones_like(self.tensor)

    @wrapout
    def zeros_like(self):
        return self.backend.zeros_like(self.tensor)

    @unwrapin
    @wrapout
    def onehot_like(self, indices, *, value=1):
        assert self.tensor.ndim == 2
        assert indices.ndim == 1
        x = self.backend.zeros_like(self.tensor)
        rows = np.arange(len(x))
        x[rows, indices] = value
        return x

    @wrapout
    def from_numpy(self, a):
        return self.backend.as_tensor(a, device=self.tensor.device)

    @wrapout
    def _concatenate(self, tensors, axis=0):
        # concatenates only "tensors", but not "self"
        tensors = [t.tensor if isinstance(t, self.__class__) else t for t in tensors]
        return self.backend.cat(tensors, dim=axis)

    @wrapout
    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return self.tensor.permute(*axes)

    def bool(self):
        return self.astype(self.backend.bool)

    @wrapout
    def all(self, axis=None, keepdims=False):
        assert self.dtype == self.backend.bool
        if axis is None:
            assert not keepdims
            return self.tensor.all()
        if not isinstance(axis, Iterable):
            axis = (axis,)
        axis = reversed(sorted(axis))
        x = self.tensor
        for i in axis:
            x = x.all(i, keepdim=keepdims)
        return x

    @wrapout
    def any(self, axis=None, keepdims=False):
        assert self.dtype == self.backend.bool
        if axis is None:
            assert not keepdims
            return self.tensor.any()
        if not isinstance(axis, Iterable):
            axis = (axis,)
        axis = reversed(sorted(axis))
        x = self.tensor
        for i in axis:
            x = x.any(i, keepdim=keepdims)
        return x

    @unwrapin
    @wrapout
    def logical_and(self, other):
        assert self.dtype == self.backend.bool
        return self.tensor & other

    @unwrapin
    @wrapout
    def logical_or(self, other):
        assert self.dtype == self.backend.bool
        return self.tensor | other

    @wrapout
    def logical_not(self):
        assert self.dtype == self.backend.bool
        return ~self.tensor

    @wrapout
    def exp(self):
        return self.backend.exp(self.tensor)

    @wrapout
    def log(self):
        return self.backend.log(self.tensor)

    @wrapout
    def log2(self):
        return self.backend.log2(self.tensor)

    @wrapout
    def log10(self):
        return self.backend.log10(self.tensor)

    @wrapout
    def log1p(self):
        return self.backend.log1p(self.tensor)

    @unwrapin
    @wrapout
    def tile(self, multiples):
        assert len(multiples) == self.ndim
        return self.tensor.repeat(multiples)
