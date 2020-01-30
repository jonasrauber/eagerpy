from .base import AbstractBaseTensor
from .base import unwrap_

from .tensor import istensor

# from .tensor import Tensor

from typing import Tuple, cast, Union, Any, TypeVar
from importlib import import_module
from collections.abc import Iterable
import numpy as onp


if False:
    # for static analyzers
    import jax
    import jax.numpy as np
else:
    jax = None
    np = None


Tensor = TypeVar("Tensor", bound="JAXTensor")


def assert_bool(x: Tensor) -> None:
    if not istensor(x):
        return
    if x.dtype != jax.numpy.bool_:
        raise ValueError(f"requires dtype bool, consider t.bool().all()")


class JAXTensor(AbstractBaseTensor):
    _registered = False
    key = None

    def __new__(cls, *args, **kwargs):
        if not cls._registered:
            import jax

            def flatten(t):
                return ((t.raw,), None)

            def unflatten(aux_data, children):
                return cls(*children)

            jax.tree_util.register_pytree_node(cls, flatten, unflatten)
            cls._registered = True
        return super().__new__(cls)

    def __init__(self, raw: "np.ndarray"):
        global jax
        global np
        if jax is None:
            jax = import_module("jax")
            np = import_module("jax.numpy")
        super().__init__(raw)

    @property
    def raw(self) -> "np.ndarray":
        return super().raw

    @classmethod
    def _get_subkey(cls):
        if cls.key is None:
            cls.key = jax.random.PRNGKey(0)
        cls.key, subkey = jax.random.split(cls.key)
        return subkey

    def numpy(self) -> Any:
        return onp.asarray(self.raw)

    def item(self) -> Union[int, float, bool]:
        return self.raw.item()  # type: ignore

    @property
    def shape(self) -> Tuple:
        return cast(Tuple, self.raw.shape)

    def reshape(self: Tensor, shape) -> Tensor:
        return type(self)(self.raw.reshape(shape))

    def astype(self: Tensor, dtype) -> Tensor:
        return type(self)(self.raw.astype(dtype))

    def clip(self: Tensor, min_, max_) -> Tensor:
        return type(self)(np.clip(self.raw, min_, max_))

    def square(self: Tensor) -> Tensor:
        return type(self)(np.square(self.raw))

    def arctanh(self: Tensor) -> Tensor:
        return type(self)(np.arctanh(self.raw))

    def sum(self: Tensor, axis=None, keepdims=False) -> Tensor:
        return type(self)(self.raw.sum(axis=axis, keepdims=keepdims))

    def mean(self: Tensor, axis=None, keepdims=False) -> Tensor:
        return type(self)(self.raw.mean(axis=axis, keepdims=keepdims))

    def min(self: Tensor, axis=None, keepdims=False) -> Tensor:
        return type(self)(self.raw.min(axis=axis, keepdims=keepdims))

    def max(self: Tensor, axis=None, keepdims=False) -> Tensor:
        return type(self)(self.raw.max(axis=axis, keepdims=keepdims))

    def minimum(self: Tensor, other) -> Tensor:
        return type(self)(np.minimum(self.raw, unwrap_(other)))

    def maximum(self: Tensor, other) -> Tensor:
        return type(self)(np.maximum(self.raw, unwrap_(other)))

    def argmin(self: Tensor, axis=None) -> Tensor:
        return type(self)(self.raw.argmin(axis=axis))

    def argmax(self: Tensor, axis=None) -> Tensor:
        return type(self)(self.raw.argmax(axis=axis))

    def argsort(self: Tensor, axis=-1) -> Tensor:
        return type(self)(self.raw.argsort(axis=axis))

    def uniform(self: Tensor, shape, low=0.0, high=1.0) -> Tensor:
        if not isinstance(shape, Iterable):
            shape = (shape,)

        subkey = self._get_subkey()
        return type(self)(jax.random.uniform(subkey, shape, minval=low, maxval=high))

    def normal(self: Tensor, shape, mean=0.0, stddev=1.0) -> Tensor:
        if not isinstance(shape, Iterable):
            shape = (shape,)

        subkey = self._get_subkey()
        return type(self)(jax.random.normal(subkey, shape) * stddev + mean)

    def ones(self: Tensor, shape) -> Tensor:
        return type(self)(np.ones(shape, dtype=self.raw.dtype))

    def zeros(self: Tensor, shape) -> Tensor:
        return type(self)(np.zeros(shape, dtype=self.raw.dtype))

    def ones_like(self: Tensor) -> Tensor:
        return type(self)(np.ones_like(self.raw))

    def zeros_like(self: Tensor) -> Tensor:
        return type(self)(np.zeros_like(self.raw))

    def full_like(self: Tensor, fill_value) -> Tensor:
        return type(self)(np.full_like(self.raw, fill_value))

    def onehot_like(self: Tensor, indices: Tensor, *, value=1) -> Tensor:
        if self.ndim != 2:
            raise ValueError("onehot_like only supported for 2D tensors")
        if indices.ndim != 1:
            raise ValueError("onehot_like requires 1D indices")
        if len(indices) != len(self):
            raise ValueError("length of indices must match length of tensor")
        x = np.arange(self.raw.shape[1]).reshape(1, -1)
        indices = indices.raw.reshape(-1, 1)
        return type(self)((x == indices) * value)

    def from_numpy(self: Tensor, a) -> Tensor:
        return type(self)(np.asarray(a))

    def _concatenate(self: Tensor, tensors, axis=0) -> Tensor:
        # concatenates only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return type(self)(np.concatenate(tensors, axis=axis))

    def _stack(self: Tensor, tensors, axis=0) -> Tensor:
        # stacks only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return type(self)(np.stack(tensors, axis=axis))

    def transpose(self: Tensor, axes=None) -> Tensor:
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return type(self)(np.transpose(self.raw, axes=axes))

    def bool(self: Tensor) -> Tensor:
        return self.astype(np.bool_)

    def all(self: Tensor, axis=None, keepdims=False) -> Tensor:
        assert_bool(self)
        return type(self)(self.raw.all(axis=axis, keepdims=keepdims))

    def any(self: Tensor, axis=None, keepdims=False) -> Tensor:
        assert_bool(self)
        return type(self)(self.raw.any(axis=axis, keepdims=keepdims))

    def logical_and(self: Tensor, other) -> Tensor:
        assert_bool(self)
        assert_bool(other)
        return type(self)(np.logical_and(self.raw, unwrap_(other)))

    def logical_or(self: Tensor, other) -> Tensor:
        assert_bool(self)
        assert_bool(other)
        return type(self)(np.logical_or(self.raw, unwrap_(other)))

    def logical_not(self: Tensor) -> Tensor:
        assert_bool(self)
        return type(self)(np.logical_not(self.raw))

    def exp(self: Tensor) -> Tensor:
        return type(self)(np.exp(self.raw))

    def log(self: Tensor) -> Tensor:
        return type(self)(np.log(self.raw))

    def log2(self: Tensor) -> Tensor:
        return type(self)(np.log2(self.raw))

    def log10(self: Tensor) -> Tensor:
        return type(self)(np.log10(self.raw))

    def log1p(self: Tensor) -> Tensor:
        return type(self)(np.log1p(self.raw))

    def tile(self: Tensor, multiples) -> Tensor:
        multiples = unwrap_(multiples)
        if len(multiples) != self.ndim:
            raise ValueError("multiples requires one entry for each dimension")
        return type(self)(np.tile(self.raw, multiples))

    def softmax(self: Tensor, axis=-1) -> Tensor:
        return type(self)(jax.nn.softmax(self.raw, axis=axis))

    def log_softmax(self: Tensor, axis=-1) -> Tensor:
        return type(self)(jax.nn.log_softmax(self.raw, axis=axis))

    def squeeze(self: Tensor, axis=None) -> Tensor:
        return type(self)(self.raw.squeeze(axis=axis))

    def expand_dims(self: Tensor, axis=None) -> Tensor:
        return type(self)(np.expand_dims(self.raw, axis=axis))

    def full(self: Tensor, shape, value) -> Tensor:
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return type(self)(np.full(shape, value, dtype=self.raw.dtype))

    def index_update(self: Tensor, indices, values) -> Tensor:
        indices, values = unwrap_(indices, values)
        if isinstance(indices, tuple):
            indices = unwrap_(indices)
        return type(self)(jax.ops.index_update(self.raw, indices, values))

    def arange(self: Tensor, start, stop=None, step=None) -> Tensor:
        return type(self)(np.arange(start, stop, step))

    def cumsum(self: Tensor, axis=None) -> Tensor:
        return type(self)(self.raw.cumsum(axis=axis))

    def flip(self: Tensor, axis=None) -> Tensor:
        return type(self)(np.flip(self.raw, axis=axis))

    def meshgrid(self: Tensor, *tensors, indexing="xy") -> Tuple[Tensor, ...]:
        tensors = unwrap_(tensors)
        outputs = np.meshgrid(self.raw, *tensors, indexing=indexing)
        return tuple(type(self)(out) for out in outputs)

    def pad(self: Tensor, paddings, mode="constant", value=0) -> Tensor:
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
            return type(self)(
                np.pad(self.raw, paddings, mode=mode, constant_values=value)
            )
        else:
            return type(self)(np.pad(self.raw, paddings, mode=mode))

    def isnan(self: Tensor) -> Tensor:
        return type(self)(np.isnan(self.raw))

    def isinf(self: Tensor) -> Tensor:
        return type(self)(np.isinf(self.raw))

    def crossentropy(self: Tensor, labels: Tensor) -> Tensor:
        if self.ndim != 2:
            raise ValueError("crossentropy only supported for 2D logits tensors")
        if self.shape[:1] != labels.shape:
            raise ValueError("labels must be 1D and must match the length of logits")
        # for numerical reasons we subtract the max logit
        # (mathematically it doesn't matter!)
        # otherwise exp(logits) might become too large or too small
        logits = self.raw
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        s = np.sum(e, axis=1)
        ces = np.log(s) - np.take_along_axis(
            logits, labels.raw[:, np.newaxis], axis=1
        ).squeeze(axis=1)
        return type(self)(ces)

    def _value_and_grad_fn(self: Tensor, f, has_aux=False) -> Any:
        # f takes and returns JAXTensor instances
        # jax.value_and_grad accepts functions that take JAXTensor instances
        # because we registered JAXTensor as JAX type, but it still requires
        # the output to be a scalar (that is not not wrapped as a JAXTensor)

        # f_jax is like f but unwraps loss
        if has_aux:

            def f_jax(*args, **kwargs):
                loss, aux = f(*args, **kwargs)
                return loss.raw, aux

        else:

            def f_jax(*args, **kwargs):
                loss = f(*args, **kwargs)
                return loss.raw

        value_and_grad_jax = jax.value_and_grad(f_jax, has_aux=has_aux)

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

    def sign(self: Tensor) -> Tensor:
        return type(self)(np.sign(self.raw))

    def sqrt(self: Tensor) -> Tensor:
        return type(self)(np.sqrt(self.raw))

    def tanh(self: Tensor) -> Tensor:
        return type(self)(np.tanh(self.raw))

    def float32(self: Tensor) -> Tensor:
        return self.astype(np.float32)

    def where(self: Tensor, x, y) -> Tensor:
        x, y = unwrap_(x, y)
        return type(self)(np.where(self.raw, x, y))

    def matmul(self: Tensor, other) -> Tensor:
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(
                f"matmul requires both tensors to be 2D, got {self.ndim}D and {other.ndim}D"
            )
        return type(self)(np.matmul(self.raw, other.raw))

    def __lt__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__lt__(unwrap_(other)))

    def __le__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__le__(unwrap_(other)))

    def __eq__(self: Tensor, other) -> Tensor:  # type: ignore
        return type(self)(self.raw.__eq__(unwrap_(other)))

    def __ne__(self: Tensor, other) -> Tensor:  # type: ignore
        return type(self)(self.raw.__ne__(unwrap_(other)))

    def __gt__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__gt__(unwrap_(other)))

    def __ge__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__ge__(unwrap_(other)))

    def __getitem__(self: Tensor, index) -> Tensor:
        if isinstance(index, tuple):
            index = tuple(x.raw if istensor(x) else x for x in index)
        elif istensor(index):
            index = index.raw
        return type(self)(self.raw[index])
