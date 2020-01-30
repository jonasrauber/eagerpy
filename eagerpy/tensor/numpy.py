from .base import AbstractBaseTensor
from .base import unwrapin
from .base import wrapout
from .base import unwrap_

from .tensor import istensor


def assert_bool(x):
    if not istensor(x):
        return
    if x.dtype != x.backend.dtype("bool"):
        raise ValueError(f"all only supports dtype bool, consider t.bool().all()")


class NumPyTensor(AbstractBaseTensor):
    def __init__(self, tensor):
        import numpy

        super().__init__(tensor)
        self.backend = numpy

    def numpy(self):
        return self.raw

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
        return self.raw.astype(dtype)

    @wrapout
    def clip(self, min_, max_):
        return self.backend.clip(self.raw, min_, max_)

    @wrapout
    def square(self):
        return self.backend.square(self.raw)

    @wrapout
    def arctanh(self):
        return self.backend.arctanh(self.raw)

    @wrapout
    def sum(self, axis=None, keepdims=False):
        return self.raw.sum(axis=axis, keepdims=keepdims)

    @wrapout
    def mean(self, axis=None, keepdims=False):
        return self.raw.mean(axis=axis, keepdims=keepdims)

    @wrapout
    def min(self, axis=None, keepdims=False):
        return self.raw.min(axis=axis, keepdims=keepdims)

    @wrapout
    def max(self, axis=None, keepdims=False):
        return self.raw.max(axis=axis, keepdims=keepdims)

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
        return self.raw.argmin(axis=axis)

    @wrapout
    def argmax(self, axis=None):
        return self.raw.argmax(axis=axis)

    @wrapout
    def argsort(self, axis=-1):
        return self.raw.argsort(axis=axis)

    @wrapout
    def uniform(self, shape, low=0.0, high=1.0):
        return self.backend.random.uniform(low, high, size=shape)

    @wrapout
    def normal(self, shape, mean=0.0, stddev=1.0):
        return self.backend.random.normal(mean, stddev, size=shape)

    @wrapout
    def ones(self, shape):
        return self.backend.ones(shape, dtype=self.raw.dtype)

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
        rows = self.backend.arange(len(x))
        x[rows, indices] = value
        return x

    @wrapout
    def from_numpy(self, a):
        return self.backend.asarray(a)

    @wrapout
    def _concatenate(self, tensors, axis=0):
        # concatenates only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return self.backend.concatenate(tensors, axis=axis)

    @wrapout
    def _stack(self, tensors, axis=0):
        # stacks only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return self.backend.stack(tensors, axis=axis)

    @wrapout
    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return self.backend.transpose(self.raw, axes=axes)

    def bool(self):
        return self.astype(self.backend.dtype("bool"))

    @wrapout
    def all(self, axis=None, keepdims=False):
        assert_bool(self)
        return self.raw.all(axis=axis, keepdims=keepdims)

    @wrapout
    def any(self, axis=None, keepdims=False):
        assert_bool(self)
        return self.raw.any(axis=axis, keepdims=keepdims)

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
        return self.backend.tile(self.raw, multiples)

    @wrapout
    def softmax(self, axis=-1):
        # for numerical reasons we subtract the max logit
        # (mathematically it doesn't matter!)
        # otherwise exp(logits) might become too large or too small
        logits = self.raw
        logits = logits - logits.max(axis=axis, keepdims=True)
        e = self.backend.exp(logits)
        return e / e.sum(axis=axis, keepdims=True)

    @wrapout
    def log_softmax(self, axis=-1):
        # for numerical reasons we subtract the max logit
        # (mathematically it doesn't matter!)
        # otherwise exp(logits) might become too large or too small
        logits = self.raw
        logits = logits - logits.max(axis=axis, keepdims=True)
        log_sum_exp = self.backend.log(
            self.backend.exp(logits).sum(axis=axis, keepdims=True)
        )
        return logits - log_sum_exp

    @wrapout
    def squeeze(self, axis=None):
        return self.raw.squeeze(axis=axis)

    @wrapout
    def expand_dims(self, axis=None):
        return self.backend.expand_dims(self.raw, axis=axis)

    @wrapout
    def full(self, shape, value):
        return self.backend.full(shape, value, dtype=self.raw.dtype)

    @unwrapin
    @wrapout
    def index_update(self, indices, values):
        if isinstance(indices, tuple):
            indices = tuple(t.raw if istensor(t) else t for t in indices)
        x = self.raw.copy()
        x[indices] = values
        return x

    @wrapout
    def arange(self, *args, **kwargs):
        return self.backend.arange(*args, **kwargs)

    @wrapout
    def cumsum(self, axis=None):
        return self.raw.cumsum(axis=axis)

    @wrapout
    def flip(self, axis=None):
        return self.backend.flip(self.raw, axis=axis)

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
        if mode == "constant":
            return self.backend.pad(
                self.raw, paddings, mode=mode, constant_values=value
            )
        else:
            return self.backend.pad(self.raw, paddings, mode=mode)

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
        # for numerical reasons we subtract the max logit
        # (mathematically it doesn't matter!)
        # otherwise exp(logits) might become too large or too small
        logits = logits - logits.max(axis=1, keepdims=True)
        e = self.backend.exp(logits)
        s = self.backend.sum(e, axis=1)
        ces = self.backend.log(s) - self.backend.take_along_axis(
            logits, labels[:, self.backend.newaxis], axis=1
        ).squeeze(axis=1)
        return ces

    def _value_and_grad_fn(self, f, has_aux=False):
        # TODO: maybe implement this using https://github.com/HIPS/autograd
        raise NotImplementedError  # pragma: no cover

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
