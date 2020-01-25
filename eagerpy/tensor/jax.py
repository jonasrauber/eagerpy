from .base import AbstractTensor
from .base import unwrapin
from .base import wrapout

from collections.abc import Iterable
import numpy as onp


class JAXTensor(AbstractTensor):
    _registered = False
    key = None

    def __new__(cls, *args, **kwargs):
        if not cls._registered:
            import jax

            def flatten(t):
                return ((t.tensor,), None)

            def unflatten(aux_data, children):
                return cls(*children)

            jax.tree_util.register_pytree_node(cls, flatten, unflatten)
            cls._registered = True
        return super().__new__(cls)

    def __init__(self, tensor):
        import jax

        super().__init__(tensor)
        self.jax = jax
        self.backend = jax.numpy

    def numpy(self):
        return onp.asarray(self.tensor)

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
        return self.tensor.argmin(axis=axis)

    @wrapout
    def argmax(self, axis=None):
        return self.tensor.argmax(axis=axis)

    @wrapout
    def argsort(self, axis=-1):
        return self.tensor.argsort(axis=axis)

    @wrapout
    def uniform(self, shape, low=0.0, high=1.0):
        import jax.random as random

        if not isinstance(shape, Iterable):
            shape = (shape,)

        cls = self.__class__
        if cls.key is None:
            cls.key = random.PRNGKey(0)

        cls.key, subkey = random.split(cls.key)
        return random.uniform(subkey, shape, minval=low, maxval=high)

    @wrapout
    def normal(self, shape, mean=0.0, stddev=1.0):
        import jax.random as random

        if not isinstance(shape, Iterable):
            shape = (shape,)

        cls = self.__class__
        if cls.key is None:
            cls.key = random.PRNGKey(0)

        cls.key, subkey = random.split(cls.key)
        return random.normal(subkey, shape) * stddev + mean

    @wrapout
    def ones(self, shape):
        return self.backend.ones(shape, dtype=self.tensor.dtype)

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
        return self.backend.full_like(self.tensor, fill_value)

    @unwrapin
    @wrapout
    def onehot_like(self, indices, *, value=1):
        assert self.tensor.ndim == 2
        assert indices.ndim == 1
        x = self.backend.arange(self.tensor.shape[1]).reshape(1, -1)
        indices = indices.reshape(-1, 1)
        return (x == indices) * value

    @wrapout
    def from_numpy(self, a):
        return self.backend.asarray(a)

    @wrapout
    def _concatenate(self, tensors, axis=0):
        # concatenates only "tensors", but not "self"
        tensors = [t.tensor if isinstance(t, self.__class__) else t for t in tensors]
        return self.backend.concatenate(tensors, axis=axis)

    @wrapout
    def _stack(self, tensors, axis=0):
        # stacks only "tensors", but not "self"
        tensors = [t.tensor if isinstance(t, self.__class__) else t for t in tensors]
        return self.backend.stack(tensors, axis=axis)

    @wrapout
    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return self.backend.transpose(self.tensor, axes=axes)

    def bool(self):
        return self.astype(self.backend.bool_)

    @wrapout
    def all(self, axis=None, keepdims=False):
        assert self.dtype == self.backend.bool_
        return self.tensor.all(axis=axis, keepdims=keepdims)

    @wrapout
    def any(self, axis=None, keepdims=False):
        assert self.dtype == self.backend.bool_
        return self.tensor.any(axis=axis, keepdims=keepdims)

    @unwrapin
    @wrapout
    def logical_and(self, other):
        assert self.dtype == self.backend.bool_
        return self.backend.logical_and(self.tensor, other)

    @unwrapin
    @wrapout
    def logical_or(self, other):
        assert self.dtype == self.backend.bool_
        return self.backend.logical_or(self.tensor, other)

    @wrapout
    def logical_not(self):
        assert self.dtype == self.backend.bool_
        return self.backend.logical_not(self.tensor)

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
        return self.backend.tile(self.tensor, multiples)

    @wrapout
    def softmax(self, axis=-1):
        return self.jax.nn.softmax(self.tensor, axis=axis)

    @wrapout
    def log_softmax(self, axis=-1):
        return self.jax.nn.log_softmax(self.tensor, axis=axis)

    @wrapout
    def squeeze(self, axis=None):
        return self.tensor.squeeze(axis=axis)

    @wrapout
    def expand_dims(self, axis=None):
        return self.backend.expand_dims(self.tensor, axis=axis)

    @wrapout
    def full(self, shape, value):
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return self.backend.full(shape, value, dtype=self.tensor.dtype)

    @unwrapin
    @wrapout
    def index_update(self, indices, values):
        if isinstance(indices, tuple):
            indices = tuple(
                t.tensor if isinstance(t, self.__class__) else t for t in indices
            )
        return self.jax.ops.index_update(self.tensor, indices, values)

    @wrapout
    def arange(self, *args, **kwargs):
        return self.backend.arange(*args, **kwargs)

    @wrapout
    def cumsum(self, axis=None):
        return self.tensor.cumsum(axis=axis)

    @wrapout
    def flip(self, axis=None):
        return self.backend.flip(self.tensor, axis=axis)

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
        if mode == "constant":
            return self.backend.pad(
                self.tensor, paddings, mode=mode, constant_values=value
            )
        else:
            return self.backend.pad(self.tensor, paddings, mode=mode)

    @wrapout
    def isnan(self):
        return self.backend.isnan(self.tensor)

    @wrapout
    def isinf(self):
        return self.backend.isinf(self.tensor)

    @unwrapin
    @wrapout
    def crossentropy(self, labels):
        logits = self.tensor
        assert logits.ndim == 2
        assert logits.shape[:1] == labels.shape
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
        # f takes and returns JAXTensor instances
        # jax.value_and_grad accepts functions that take JAXTensor instances
        # because we registered JAXTensor as JAX type, but it still requires
        # the output to be a scalar (that is not not wrapped as a JAXTensor)

        # f_jax is like f but unwraps loss
        if has_aux:

            def f_jax(*args, **kwargs):
                loss, aux = f(*args, **kwargs)
                return loss.tensor, aux

        else:

            def f_jax(*args, **kwargs):
                loss = f(*args, **kwargs)
                return loss.tensor

        value_and_grad_jax = self.jax.value_and_grad(f_jax, has_aux=has_aux)

        # value_and_grad is like value_and_grad_jax but wraps loss
        if has_aux:

            def value_and_grad(x, *args, **kwargs):
                assert isinstance(x, JAXTensor)
                (loss, aux), grad = value_and_grad_jax(x, *args, **kwargs)
                assert grad.shape == x.shape
                return JAXTensor(loss), aux, grad

        else:

            def value_and_grad(x, *args, **kwargs):
                assert isinstance(x, JAXTensor)
                loss, grad = value_and_grad_jax(x, *args, **kwargs)
                assert grad.shape == x.shape
                return JAXTensor(loss), grad

        return value_and_grad
