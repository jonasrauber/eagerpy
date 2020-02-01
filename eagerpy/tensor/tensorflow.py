import functools
from typing import Tuple, cast, Union, Any, TypeVar, Callable, TYPE_CHECKING
from importlib import import_module
from collections.abc import Iterable
import numpy as np

from .. import index

from .tensor import istensor
from .tensor import TensorType

from .base import BaseTensor
from .base import unwrap_

if TYPE_CHECKING:
    import tensorflow as tf  # for static analyzers
else:
    # lazy import in TensorFlowTensor
    tf = None

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def samedevice(f: F) -> F:
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        with tf.device(self.raw.device):
            return f(self, *args, **kwargs)

    return cast(F, wrapper)


def common_dtype(f: F) -> F:
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        dtypes = {self.dtype} | {arg.dtype for arg in args if istensor(arg)}
        if len(dtypes) == 1:
            # all dtypes are the same, nothing more to do
            return f(self, *args, **kwargs)
        numpy_dtypes = [np.dtype(dtype.name) for dtype in dtypes]
        common = np.find_common_type(numpy_dtypes, [])
        common = getattr(tf, common.name)
        if self.dtype != common:
            self = self.astype(common)
        args = [
            arg.astype(common) if istensor(arg) and arg.dtype != common else arg
            for arg in args
        ]
        return f(self, *args, **kwargs)

    return cast(F, wrapper)


def assert_bool(x: TensorType) -> None:
    if not istensor(x):
        return
    if x.dtype != tf.bool:
        raise ValueError(f"requires dtype bool, consider t.bool().all()")


class TensorFlowTensor(BaseTensor):
    def __init__(self, raw: "tf.Tensor"):  # type: ignore
        global tf
        if tf is None:
            tf = import_module("tensorflow")
        super().__init__(raw)

    @property
    def raw(self) -> "tf.Tensor":  # type: ignore
        return super().raw

    def numpy(self: TensorType) -> Any:
        return self.raw.numpy()

    def item(self: TensorType) -> Union[int, float, bool]:
        return self.numpy().item()  # type: ignore

    @property
    def shape(self: TensorType) -> Tuple:
        return tuple(self.raw.shape.as_list())

    def reshape(self: TensorType, shape) -> TensorType:
        return type(self)(tf.reshape(self.raw, shape))

    def astype(self: TensorType, dtype) -> TensorType:
        return type(self)(tf.cast(self.raw, dtype))

    def clip(self: TensorType, min_, max_) -> TensorType:
        return type(self)(tf.clip_by_value(self.raw, min_, max_))

    def square(self: TensorType) -> TensorType:
        return type(self)(tf.square(self.raw))

    def arctanh(self: TensorType) -> TensorType:
        return type(self)(tf.atanh(self.raw))

    def sum(self: TensorType, axis=None, keepdims=False) -> TensorType:
        if self.raw.dtype == tf.bool:
            return type(self)(self.astype(tf.int64).sum(axis=axis, keepdims=keepdims))
        return type(self)(tf.reduce_sum(self.raw, axis=axis, keepdims=keepdims))

    def mean(self: TensorType, axis=None, keepdims=False) -> TensorType:
        return type(self)(tf.reduce_mean(self.raw, axis=axis, keepdims=keepdims))

    def min(self: TensorType, axis=None, keepdims=False) -> TensorType:
        return type(self)(tf.reduce_min(self.raw, axis=axis, keepdims=keepdims))

    def max(self: TensorType, axis=None, keepdims=False) -> TensorType:
        return type(self)(tf.reduce_max(self.raw, axis=axis, keepdims=keepdims))

    def minimum(self: TensorType, other) -> TensorType:
        return type(self)(tf.minimum(self.raw, unwrap_(other)))

    def maximum(self: TensorType, other) -> TensorType:
        return type(self)(tf.maximum(self.raw, unwrap_(other)))

    def argmin(self: TensorType, axis=None) -> TensorType:
        return type(self)(tf.argmin(self.raw, axis=axis))

    def argmax(self: TensorType, axis=None) -> TensorType:
        return type(self)(tf.argmax(self.raw, axis=axis))

    def argsort(self: TensorType, axis=-1) -> TensorType:
        return type(self)(tf.argsort(self.raw, axis=axis))

    @samedevice
    def uniform(self: TensorType, shape, low=0.0, high=1.0) -> TensorType:
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return type(self)(
            tf.random.uniform(shape, minval=low, maxval=high, dtype=self.raw.dtype)
        )

    @samedevice
    def normal(self: TensorType, shape, mean=0.0, stddev=1.0) -> TensorType:
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return type(self)(
            tf.random.normal(shape, mean=mean, stddev=stddev, dtype=self.raw.dtype)
        )

    @samedevice
    def ones(self: TensorType, shape) -> TensorType:
        return type(self)(tf.ones(shape, dtype=self.raw.dtype))

    @samedevice
    def zeros(self: TensorType, shape) -> TensorType:
        return type(self)(tf.zeros(shape, dtype=self.raw.dtype))

    def ones_like(self: TensorType) -> TensorType:
        return type(self)(tf.ones_like(self.raw))

    def zeros_like(self: TensorType) -> TensorType:
        return type(self)(tf.zeros_like(self.raw))

    def full_like(self: TensorType, fill_value) -> TensorType:
        fill_value = tf.cast(fill_value, self.raw.dtype)
        return type(self)(tf.fill(self.raw.shape, fill_value))

    @samedevice
    def onehot_like(self: TensorType, indices: TensorType, *, value=1) -> TensorType:
        if self.ndim != 2:
            raise ValueError("onehot_like only supported for 2D tensors")
        if indices.ndim != 1:
            raise ValueError("onehot_like requires 1D indices")
        if len(indices) != len(self):
            raise ValueError("length of indices must match length of tensor")
        value = tf.cast(value, self.raw.dtype)
        return type(self)(
            tf.one_hot(
                indices.raw,
                depth=self.raw.shape[-1],
                on_value=value,
                dtype=self.raw.dtype,
            )
        )

    @samedevice
    def from_numpy(self: TensorType, a) -> TensorType:
        return type(self)(tf.convert_to_tensor(a))

    def _concatenate(self: TensorType, tensors, axis=0) -> TensorType:
        # concatenates only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return type(self)(tf.concat(tensors, axis=axis))

    def _stack(self: TensorType, tensors, axis=0) -> TensorType:
        # stacks only "tensors", but not "self"
        tensors = [t.raw if istensor(t) else t for t in tensors]
        return type(self)(tf.stack(tensors, axis=axis))

    def transpose(self: TensorType, axes=None) -> TensorType:
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return type(self)(tf.transpose(self.raw, perm=axes))

    def bool(self: TensorType) -> TensorType:
        return self.astype(tf.bool)

    def all(self: TensorType, axis=None, keepdims=False) -> TensorType:
        assert_bool(self)
        return type(self)(tf.reduce_all(self.raw, axis=axis, keepdims=keepdims))

    def any(self: TensorType, axis=None, keepdims=False) -> TensorType:
        assert_bool(self)
        return type(self)(tf.reduce_any(self.raw, axis=axis, keepdims=keepdims))

    def logical_and(self: TensorType, other) -> TensorType:
        assert_bool(self)
        assert_bool(other)
        return type(self)(tf.logical_and(self.raw, unwrap_(other)))

    def logical_or(self: TensorType, other) -> TensorType:
        assert_bool(self)
        assert_bool(other)
        return type(self)(tf.logical_or(self.raw, unwrap_(other)))

    def logical_not(self: TensorType) -> TensorType:
        assert_bool(self)
        return type(self)(tf.logical_not(self.raw))

    def exp(self: TensorType) -> TensorType:
        return type(self)(tf.exp(self.raw))

    def log(self: TensorType) -> TensorType:
        return type(self)(tf.math.log(self.raw))

    def log2(self: TensorType) -> TensorType:
        return type(self)(tf.math.log(self.raw) / tf.math.log(2.0))

    def log10(self: TensorType) -> TensorType:
        return type(self)(tf.math.log(self.raw) / tf.math.log(10.0))

    def log1p(self: TensorType) -> TensorType:
        return type(self)(tf.math.log1p(self.raw))

    def tile(self: TensorType, multiples) -> TensorType:
        multiples = unwrap_(multiples)
        if len(multiples) != self.ndim:
            raise ValueError("multiples requires one entry for each dimension")
        return type(self)(tf.tile(self.raw, multiples))

    def softmax(self: TensorType, axis=-1) -> TensorType:
        return type(self)(tf.nn.softmax(self.raw, axis=axis))

    def log_softmax(self: TensorType, axis=-1) -> TensorType:
        return type(self)(tf.nn.log_softmax(self.raw, axis=axis))

    def squeeze(self: TensorType, axis=None) -> TensorType:
        return type(self)(tf.squeeze(self.raw, axis=axis))

    def expand_dims(self: TensorType, axis=None) -> TensorType:
        return type(self)(tf.expand_dims(self.raw, axis=axis))

    @samedevice
    def full(self: TensorType, shape, value) -> TensorType:
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return type(self)(tf.fill(shape, value))

    def index_update(self: TensorType, indices, values) -> TensorType:
        indices, values = unwrap_(indices, values)
        if isinstance(indices, tuple):
            indices = unwrap_(indices)

        x = self.raw
        if isinstance(indices, int):
            return type(self)(tf.tensor_scatter_nd_update(x, [[indices]], values[None]))
        elif isinstance(indices, tuple) and any(
            isinstance(idx, slice) for idx in indices
        ):
            if (
                len(indices) == x.ndim == 2
                and indices[0] == index[:]
                and not isinstance(indices[1], slice)
            ):
                x = tf.transpose(x)
                result = tf.tensor_scatter_nd_update(x, [[indices[-1]]], values[None])
                return type(self)(tf.transpose(result))
            else:
                raise NotImplementedError  # pragma: no cover
        elif isinstance(indices, tuple):
            if all(idx.dtype in [tf.int32, tf.int64] for idx in indices):
                indices = [
                    tf.cast(idx, tf.int64) if idx.dtype == tf.int32 else idx
                    for idx in indices
                ]
            return type(self)(
                tf.tensor_scatter_nd_update(x, tf.stack(indices, axis=-1), values)
            )
        else:
            raise ValueError  # pragma: no cover

    @samedevice
    def arange(self: TensorType, start, stop=None, step=None) -> TensorType:
        if step is None:
            step = 1
        if stop is None:
            stop = start
            start = 0
        return type(self)(tf.range(start, stop, step))

    def cumsum(self: TensorType, axis=None) -> TensorType:
        if axis is None:
            x = tf.reshape(self.raw, (-1,))
            return type(self)(tf.cumsum(x, axis=0))
        return type(self)(tf.cumsum(self.raw, axis=axis))

    def flip(self: TensorType, axis=None) -> TensorType:
        if axis is None:
            axis = tuple(range(self.ndim))
        if not isinstance(axis, Iterable):
            axis = (axis,)
        return type(self)(tf.reverse(self.raw, axis=axis))

    def meshgrid(self: TensorType, *tensors, indexing="xy") -> Tuple[TensorType, ...]:
        tensors = unwrap_(tensors)
        outputs = tf.meshgrid(self.raw, *tensors, indexing=indexing)
        return tuple(type(self)(out) for out in outputs)

    def pad(self: TensorType, paddings, mode="constant", value=0) -> TensorType:
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
        return type(self)(tf.pad(self.raw, paddings, mode=mode, constant_values=value))

    def isnan(self: TensorType) -> TensorType:
        return type(self)(tf.math.is_nan(self.raw))

    def isinf(self: TensorType) -> TensorType:
        return type(self)(tf.math.is_inf(self.raw))

    def crossentropy(self: TensorType, labels: TensorType) -> TensorType:
        if self.ndim != 2:
            raise ValueError("crossentropy only supported for 2D logits tensors")
        if self.shape[:1] != labels.shape:
            raise ValueError("labels must be 1D and must match the length of logits")
        return type(self)(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels.raw, self.raw)
        )

    def _value_and_grad_fn(self: TensorType, f, has_aux=False) -> Any:
        def value_and_grad(x, *args, **kwargs):
            # using tf.identity to make x independent from possible other instances of x in args
            x = x.raw
            x = tf.identity(x)
            x = TensorFlowTensor(x)
            assert isinstance(x, TensorFlowTensor)
            with tf.GradientTape() as tape:
                tape.watch(x.raw)
                if has_aux:
                    loss, aux = f(x, *args, **kwargs)
                else:
                    loss = f(x, *args, **kwargs)
            grad = tape.gradient(loss.raw, x.raw)
            grad = TensorFlowTensor(grad)
            assert grad.shape == x.shape
            if has_aux:
                return loss, aux, grad
            else:
                return loss, grad

        return value_and_grad

    def sign(self: TensorType) -> TensorType:
        return type(self)(tf.sign(self.raw))

    def sqrt(self: TensorType) -> TensorType:
        return type(self)(tf.sqrt(self.raw))

    def tanh(self: TensorType) -> TensorType:
        return type(self)(tf.tanh(self.raw))

    def float32(self: TensorType) -> TensorType:
        return self.astype(tf.float32)

    def where(self: TensorType, x, y) -> TensorType:
        x, y = unwrap_(x, y)
        return type(self)(tf.where(self.raw, x, y))

    def matmul(self: TensorType, other) -> TensorType:
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(
                f"matmul requires both tensors to be 2D, got {self.ndim}D and {other.ndim}D"
            )
        return type(self)(tf.matmul(self.raw, other.raw))

    @common_dtype
    def __lt__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__lt__(unwrap_(other)))

    @common_dtype
    def __le__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__le__(unwrap_(other)))

    @common_dtype
    def __eq__(self: TensorType, other) -> TensorType:  # type: ignore
        return type(self)(self.raw.__eq__(unwrap_(other)))

    @common_dtype
    def __ne__(self: TensorType, other) -> TensorType:  # type: ignore
        return type(self)(self.raw.__ne__(unwrap_(other)))

    @common_dtype
    def __gt__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__gt__(unwrap_(other)))

    @common_dtype
    def __ge__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__ge__(unwrap_(other)))

    def __getitem__(self: TensorType, index) -> TensorType:
        if isinstance(index, tuple):
            index = tuple(x.raw if istensor(x) else x for x in index)
            tensors = any(
                isinstance(x, tf.Tensor) or isinstance(x, np.ndarray) for x in index
            )
            if tensors:
                # workaround for missing support for this in TensorFlow
                index = tf.convert_to_tensor(index)
                index = tf.transpose(index)
                return type(self)(tf.gather_nd(self.raw, index))
        elif istensor(index):
            return type(self)(tf.gather(self.raw, index.raw))
        return type(self)(self.raw.__getitem__(index))
