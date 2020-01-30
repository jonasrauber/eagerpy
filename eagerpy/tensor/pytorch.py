from .base import AbstractBaseTensor
from .base import wrapout
from .base import unwrapin
from .base import unwrap_

from .tensor import istensor

# from .tensor import Tensor

import numpy as np
from collections.abc import Iterable

# from typing import TypeVar


# PyTensor = TypeVar("PyTensor", bound="PyTorchTensor")


def assert_bool(x):
    if not istensor(x):
        return
    if x.dtype != x.backend.bool:
        raise ValueError(f"all only supports dtype bool, consider t.bool().all()")


class PyTorchTensor(AbstractBaseTensor):
    def __init__(self, tensor):
        import torch

        super().__init__(tensor)
        self.backend = torch

    def numpy(self):
        return self.raw.detach().cpu().numpy()

    def item(self):
        return self.raw.item()

    @property
    def shape(self):
        return self.raw.shape

    @wrapout
    def reshape(self, shape):
        return self.raw.reshape(shape)

    @wrapout
    def astype(self, dtype):
        return self.raw.to(dtype)

    @wrapout
    def clip(self, min_, max_):
        return self.raw.clamp(min_, max_)

    @wrapout
    def square(self):
        return self.raw ** 2

    @wrapout
    def arctanh(self):
        """
        improve once this issue has been fixed:
        https://github.com/pytorch/pytorch/issues/10324
        """
        return 0.5 * self.backend.log((1 + self.raw) / (1 - self.raw))

    @wrapout
    def sum(self, axis=None, keepdims=False):
        if axis is None and not keepdims:
            return self.raw.sum()
        if axis is None:
            axis = tuple(range(self.ndim))
        return self.raw.sum(dim=axis, keepdim=keepdims)

    @wrapout
    def mean(self, axis=None, keepdims=False):
        if axis is None and not keepdims:
            return self.raw.mean()
        if axis is None:
            axis = tuple(range(self.ndim))
        return self.raw.mean(dim=axis, keepdim=keepdims)

    @wrapout
    def min(self, axis=None, keepdims=False):
        """
        simplify once this issue has been fixed:
        https://github.com/pytorch/pytorch/issues/28213
        """
        if axis is None and not keepdims:
            return self.raw.min()
        if axis is None:
            axis = tuple(range(self.ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)
        axis = reversed(sorted(axis))
        x = self.raw
        for i in axis:
            x, _ = x.min(i, keepdim=keepdims)
        return x

    @wrapout
    def max(self, axis=None, keepdims=False):
        """
        simplify once this issue has been fixed:
        https://github.com/pytorch/pytorch/issues/28213
        """
        if axis is None and not keepdims:
            return self.raw.max()
        if axis is None:
            axis = tuple(range(self.ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)
        axis = reversed(sorted(axis))
        x = self.raw
        for i in axis:
            x, _ = x.max(i, keepdim=keepdims)
        return x

    @wrapout
    def minimum(self, other):
        if istensor(other):
            other = other.raw
        else:
            other = self.backend.ones_like(self.raw) * other
        return self.backend.min(self.raw, other)

    @wrapout
    def maximum(self, other):
        if istensor(other):
            other = other.raw
        else:
            other = self.backend.ones_like(self.raw) * other
        return self.backend.max(self.raw, other)

    @wrapout
    def argmin(self, axis=None):
        return self.raw.argmin(dim=axis)

    @wrapout
    def argmax(self, axis=None):
        return self.raw.argmax(dim=axis)

    @wrapout
    def argsort(self, axis=-1):
        return self.raw.argsort(dim=axis)

    @wrapout
    def uniform(self, shape, low=0.0, high=1.0):
        return (
            self.backend.rand(shape, dtype=self.raw.dtype, device=self.raw.device)
            * (high - low)
            + low
        )

    @wrapout
    def normal(self, shape, mean=0.0, stddev=1.0):
        return (
            self.backend.randn(shape, dtype=self.raw.dtype, device=self.raw.device)
            * stddev
            + mean
        )

    @wrapout
    def ones(self, shape):
        return self.backend.ones(shape, dtype=self.raw.dtype, device=self.raw.device)

    @wrapout
    def zeros(self, shape):
        return self.backend.zeros(shape, dtype=self.raw.dtype, device=self.raw.device)

    @wrapout
    def ones_like(self):
        return self.backend.ones_like(self.raw)

    @wrapout
    def zeros_like(self):
        return self.backend.zeros_like(self.raw)

    @wrapout
    def full_like(self, fill_value):
        return self.backend.full_like(self.raw, fill_value)

    @unwrapin
    @wrapout
    def onehot_like(self, indices, *, value=1):
        if self.ndim != 2:
            raise ValueError("onehot_like only supported for 2D tensors")
        if indices.ndim != 1:
            raise ValueError("onehot_like requires 1D indices")
        if len(indices) != len(self.raw):
            raise ValueError("length of indices must match length of tensor")
        x = self.backend.zeros_like(self.raw)
        rows = np.arange(len(x))
        x[rows, indices] = value
        return x

    @wrapout
    def from_numpy(self, a):
        return self.backend.as_tensor(a, device=self.raw.device)

    @wrapout
    def _concatenate(self, tensors, axis=0):
        # concatenates only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return self.backend.cat(tensors, dim=axis)

    @wrapout
    def _stack(self, tensors, axis=0):
        # stacks only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return self.backend.stack(tensors, dim=axis)

    @wrapout
    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return self.raw.permute(*axes)

    def bool(self):
        return self.astype(self.backend.bool)

    @wrapout
    def all(self, axis=None, keepdims=False):
        assert_bool(self)
        if axis is None and not keepdims:
            return self.raw.all()
        if axis is None:
            axis = tuple(range(self.ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)
        axis = reversed(sorted(axis))
        x = self.raw
        for i in axis:
            x = x.all(i, keepdim=keepdims)
        return x

    @wrapout
    def any(self, axis=None, keepdims=False):
        assert_bool(self)
        if axis is None and not keepdims:
            return self.raw.any()
        if axis is None:
            axis = tuple(range(self.ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)
        axis = reversed(sorted(axis))
        x = self.raw
        for i in axis:
            x = x.any(i, keepdim=keepdims)
        return x

    @wrapout
    def logical_and(self, other):
        assert_bool(self)
        assert_bool(other)
        return self.raw & unwrap_(other)

    @wrapout
    def logical_or(self, other):
        assert_bool(self)
        assert_bool(other)
        return self.raw | unwrap_(other)

    @wrapout
    def logical_not(self):
        assert_bool(self)
        return ~self.raw

    @wrapout
    def exp(self):
        return self.backend.exp(self.raw)

    @wrapout
    def log(self):
        return self.backend.log(self.raw)

    @wrapout
    def log2(self):
        return self.backend.log2(self.raw)

    @wrapout
    def log10(self):
        return self.backend.log10(self.raw)

    @wrapout
    def log1p(self):
        return self.backend.log1p(self.raw)

    @unwrapin
    @wrapout
    def tile(self, multiples):
        if len(multiples) != self.ndim:
            raise ValueError("multiples requires one entry for each dimension")
        return self.raw.repeat(multiples)

    @wrapout
    def softmax(self, axis=-1):
        return self.backend.nn.functional.softmax(self.raw, dim=axis)

    @wrapout
    def log_softmax(self, axis=-1):
        return self.backend.nn.functional.log_softmax(self.raw, dim=axis)

    @wrapout
    def squeeze(self, axis=None):
        if axis is None:
            return self.raw.squeeze()
        if not isinstance(axis, Iterable):
            axis = (axis,)
        axis = reversed(sorted(axis))
        x = self.raw
        for i in axis:
            x = x.squeeze(dim=i)
        return x

    @wrapout
    def expand_dims(self, axis=None):
        return self.raw.unsqueeze(axis=axis)

    @wrapout
    def full(self, shape, value):
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return self.backend.full(
            shape, value, dtype=self.raw.dtype, device=self.raw.device
        )

    @unwrapin
    @wrapout
    def index_update(self, indices, values):
        if isinstance(indices, tuple):
            indices = tuple(t.raw if istensor(t) else t for t in indices)
        x = self.raw.clone()
        x[indices] = values
        return x

    @wrapout
    def arange(self, *args, **kwargs):
        return self.backend.arange(*args, **kwargs, device=self.raw.device)

    @wrapout
    def cumsum(self, axis=None):
        if axis is None:
            return self.raw.reshape(-1).cumsum(dim=0)
        return self.raw.cumsum(dim=axis)

    @wrapout
    def flip(self, axis=None):
        if axis is None:
            axis = tuple(range(self.ndim))
        if not isinstance(axis, Iterable):
            axis = (axis,)
        return self.raw.flip(dims=axis)

    @unwrapin
    def meshgrid(self, *tensors, indexing="xy"):
        if indexing == "ij" or len(tensors) == 0:
            outputs = self.backend.meshgrid(self.raw, *tensors)
        elif indexing == "xy":
            outputs = self.backend.meshgrid(tensors[0], self.raw, *tensors[1:])
        else:
            raise ValueError(  # pragma: no cover
                f"Valid values for indexing are 'xy' and 'ij', got {indexing}"
            )
        outputs = list(self.__class__(out) for out in outputs)
        if indexing == "xy" and len(outputs) >= 2:
            outputs[0], outputs[1] = outputs[1], outputs[0]
        return tuple(outputs)

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
            paddings = paddings[k:]
        paddings = tuple(x for p in reversed(paddings) for x in p)
        return self.backend.nn.functional.pad(
            self.raw, paddings, mode=mode, value=value
        )

    @wrapout
    def isnan(self):
        return self.backend.isnan(self.raw)

    @wrapout
    def isinf(self):
        return self.backend.isinf(self.raw)

    @unwrapin
    @wrapout
    def crossentropy(self, labels):
        logits = self.raw
        if logits.ndim != 2:
            raise ValueError("crossentropy only supported for 2D logits tensors")
        if logits.shape[:1] != labels.shape:
            raise ValueError("labels must be 1D and must match the length of logits")
        return self.backend.nn.functional.cross_entropy(
            logits, labels, reduction="none"
        )

    def _value_and_grad_fn(self, f, has_aux=False):
        def value_and_grad(x, *args, **kwargs):
            assert isinstance(x, PyTorchTensor)
            x = x.raw
            x = x.clone()
            x.requires_grad_()
            x = PyTorchTensor(x)
            if has_aux:
                loss, aux = f(x, *args, **kwargs)
            else:
                loss = f(x, *args, **kwargs)
            loss = loss.raw
            loss.backward()
            grad = PyTorchTensor(x.raw.grad)
            assert grad.shape == x.shape
            loss = loss.detach()
            loss = PyTorchTensor(loss)
            if has_aux:
                if isinstance(aux, PyTorchTensor):
                    aux = PyTorchTensor(aux.raw.detach())
                elif isinstance(aux, tuple):
                    aux = tuple(
                        PyTorchTensor(t.raw.detach())
                        if isinstance(t, PyTorchTensor)
                        else t
                        for t in aux
                    )
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
