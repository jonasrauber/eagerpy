from typing import (
    Tuple,
    cast,
    Union,
    Any,
    TypeVar,
    TYPE_CHECKING,
    Iterable,
    Optional,
    overload,
    Callable,
)
from typing_extensions import Literal
import numpy as np
from importlib import import_module
import functools

from ..types import Axes, AxisAxes, Shape, ShapeOrScalar

from .. import index

from .tensor import Tensor
from .tensor import TensorOrScalar
from .tensor import TensorType

from .base import BaseTensor
from .base import unwrap_
from .base import unwrap1

if TYPE_CHECKING:
    import tensorflow as tf  # for static analyzers
    from .extensions import NormsMethods  # noqa: F401
else:
    # lazy import in TensorFlowTensor
    tf = None

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def samedevice(f: F) -> F:
    @functools.wraps(f)
    def wrapper(self: "TensorFlowTensor", *args: Any, **kwargs: Any) -> Any:
        with tf.device(self.raw.device):
            return f(self, *args, **kwargs)

    return cast(F, wrapper)


def common_dtype(f: F) -> F:
    @functools.wraps(f)
    def wrapper(self: "TensorFlowTensor", *args: Any, **kwargs: Any) -> Any:
        dtypes = {self.dtype} | {arg.dtype for arg in args if isinstance(arg, Tensor)}
        if len(dtypes) == 1:
            # all dtypes are the same, nothing more to do
            return f(self, *args, **kwargs)
        numpy_dtypes = [np.dtype(dtype.name) for dtype in dtypes]
        common = np.find_common_type(numpy_dtypes, [])
        common = getattr(tf, common.name)
        if self.dtype != common:
            self = self.astype(common)
        args = tuple(
            arg.astype(common)
            if isinstance(arg, Tensor) and arg.dtype != common
            else arg
            for arg in args
        )
        return f(self, *args, **kwargs)

    return cast(F, wrapper)


def assert_bool(x: Any) -> None:
    if not isinstance(x, Tensor):
        return
    if x.dtype != tf.bool:
        raise ValueError(f"requires dtype bool, got {x.dtype}, consider t.bool().all()")


class TensorFlowTensor(BaseTensor):
    __slots__ = ()

    # more specific types for the extensions
    norms: "NormsMethods[TensorFlowTensor]"

    def __init__(self, raw: "tf.Tensor"):  # type: ignore
        global tf
        if tf is None:
            tf = import_module("tensorflow")
        super().__init__(raw)

    @property
    def raw(self) -> "tf.Tensor":  # type: ignore
        return super().raw

    def numpy(self: TensorType) -> Any:
        a = self.raw.numpy()
        if a.flags.writeable:
            # without the check, we would attempt to set it on array
            # scalars, and that would fail
            a.flags.writeable = False
        return a

    def item(self: TensorType) -> Union[int, float, bool]:
        return self.numpy().item()  # type: ignore

    @property
    def shape(self: TensorType) -> Shape:
        return tuple(self.raw.shape.as_list())

    def reshape(self: TensorType, shape: Union[Shape, int]) -> TensorType:
        if isinstance(shape, int):
            shape = (shape,)
        return type(self)(tf.reshape(self.raw, shape))

    def astype(self: TensorType, dtype: Any) -> TensorType:
        return type(self)(tf.cast(self.raw, dtype))

    def clip(self: TensorType, min_: float, max_: float) -> TensorType:
        return type(self)(tf.clip_by_value(self.raw, min_, max_))

    def square(self: TensorType) -> TensorType:
        return type(self)(tf.square(self.raw))

    def sum(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        if self.raw.dtype == tf.bool:
            return self.astype(tf.int64).sum(axis=axis, keepdims=keepdims)
        return type(self)(tf.reduce_sum(self.raw, axis=axis, keepdims=keepdims))

    def prod(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        if self.raw.dtype == tf.bool:
            return self.astype(tf.int64).prod(axis=axis, keepdims=keepdims)
        return type(self)(tf.reduce_prod(self.raw, axis=axis, keepdims=keepdims))

    def mean(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        if self.raw.dtype not in [tf.float16, tf.float32, tf.float64]:
            raise ValueError(
                f"Can only calculate the mean of floating types. Got {self.raw.dtype} instead."
            )
        return type(self)(tf.reduce_mean(self.raw, axis=axis, keepdims=keepdims))

    def min(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        return type(self)(tf.reduce_min(self.raw, axis=axis, keepdims=keepdims))

    def max(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        return type(self)(tf.reduce_max(self.raw, axis=axis, keepdims=keepdims))

    def minimum(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(tf.minimum(self.raw, unwrap1(other)))

    def maximum(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(tf.maximum(self.raw, unwrap1(other)))

    def argmin(self: TensorType, axis: Optional[int] = None) -> TensorType:
        return type(self)(tf.argmin(self.raw, axis=axis))

    def argmax(self: TensorType, axis: Optional[int] = None) -> TensorType:
        return type(self)(tf.argmax(self.raw, axis=axis))

    def argsort(self: TensorType, axis: Optional[int] = -1) -> TensorType:
        return type(self)(tf.argsort(self.raw, axis=axis))

    def sort(self: TensorType, axis: Optional[int] = -1) -> TensorType:
        return type(self)(tf.sort(self.raw, axis=axis))

    def topk(
        self: TensorType, k: int, sorted: bool = True
    ) -> Tuple[TensorType, TensorType]:
        values, indices = tf.math.top_k(self.raw, k, sorted=sorted)
        return type(self)(values), type(self)(indices)

    @samedevice
    def uniform(
        self: TensorType, shape: ShapeOrScalar, low: float = 0.0, high: float = 1.0
    ) -> TensorType:
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return type(self)(
            tf.random.uniform(shape, minval=low, maxval=high, dtype=self.raw.dtype)
        )

    @samedevice
    def normal(
        self: TensorType, shape: ShapeOrScalar, mean: float = 0.0, stddev: float = 1.0
    ) -> TensorType:
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return type(self)(
            tf.random.normal(shape, mean=mean, stddev=stddev, dtype=self.raw.dtype)
        )

    @samedevice
    def ones(self: TensorType, shape: ShapeOrScalar) -> TensorType:
        return type(self)(tf.ones(shape, dtype=self.raw.dtype))

    @samedevice
    def zeros(self: TensorType, shape: ShapeOrScalar) -> TensorType:
        return type(self)(tf.zeros(shape, dtype=self.raw.dtype))

    def ones_like(self: TensorType) -> TensorType:
        return type(self)(tf.ones_like(self.raw))

    def zeros_like(self: TensorType) -> TensorType:
        return type(self)(tf.zeros_like(self.raw))

    def full_like(self: TensorType, fill_value: float) -> TensorType:
        fill_value = tf.cast(fill_value, self.raw.dtype)
        return type(self)(tf.fill(self.raw.shape, fill_value))

    @samedevice
    def onehot_like(
        self: TensorType, indices: TensorType, *, value: float = 1
    ) -> TensorType:
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
    def from_numpy(self: TensorType, a: Any) -> TensorType:
        return type(self)(tf.convert_to_tensor(a))

    def _concatenate(
        self: TensorType, tensors: Iterable[TensorType], axis: int = 0
    ) -> TensorType:
        # concatenates only "tensors", but not "self"
        tensors_ = unwrap_(*tensors)
        return type(self)(tf.concat(tensors_, axis=axis))

    def _stack(
        self: TensorType, tensors: Iterable[TensorType], axis: int = 0
    ) -> TensorType:
        # stacks only "tensors", but not "self"
        tensors_ = unwrap_(*tensors)
        return type(self)(tf.stack(tensors_, axis=axis))

    def transpose(self: TensorType, axes: Optional[Axes] = None) -> TensorType:
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return type(self)(tf.transpose(self.raw, perm=axes))

    def all(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        assert_bool(self)
        return type(self)(tf.reduce_all(self.raw, axis=axis, keepdims=keepdims))

    def any(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        assert_bool(self)
        return type(self)(tf.reduce_any(self.raw, axis=axis, keepdims=keepdims))

    def logical_and(self: TensorType, other: TensorOrScalar) -> TensorType:
        assert_bool(self)
        assert_bool(other)
        return type(self)(tf.logical_and(self.raw, unwrap1(other)))

    def logical_or(self: TensorType, other: TensorOrScalar) -> TensorType:
        assert_bool(self)
        assert_bool(other)
        return type(self)(tf.logical_or(self.raw, unwrap1(other)))

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

    def tile(self: TensorType, multiples: Axes) -> TensorType:
        multiples = unwrap1(multiples)
        if len(multiples) != self.ndim:
            raise ValueError("multiples requires one entry for each dimension")
        return type(self)(tf.tile(self.raw, multiples))

    def softmax(self: TensorType, axis: int = -1) -> TensorType:
        return type(self)(tf.nn.softmax(self.raw, axis=axis))

    def log_softmax(self: TensorType, axis: int = -1) -> TensorType:
        return type(self)(tf.nn.log_softmax(self.raw, axis=axis))

    def squeeze(self: TensorType, axis: Optional[AxisAxes] = None) -> TensorType:
        return type(self)(tf.squeeze(self.raw, axis=axis))

    def expand_dims(self: TensorType, axis: int) -> TensorType:
        return type(self)(tf.expand_dims(self.raw, axis=axis))

    @samedevice
    def full(self: TensorType, shape: ShapeOrScalar, value: float) -> TensorType:
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return type(self)(tf.fill(shape, value))

    def index_update(
        self: TensorType, indices: Any, values: TensorOrScalar
    ) -> TensorType:
        indices, values_ = unwrap_(indices, values)
        del values
        if isinstance(indices, tuple):
            indices = unwrap_(*indices)

        x = self.raw
        if isinstance(indices, int):
            if isinstance(values_, int) or isinstance(values_, float):
                values_ = tf.fill(x.shape[-1:], values_)
            return type(self)(
                tf.tensor_scatter_nd_update(x, [[indices]], values_[None])
            )
        elif isinstance(indices, tuple) and any(
            isinstance(idx, slice) for idx in indices
        ):
            if (
                len(indices) == x.ndim == 2
                and indices[0] == index[:]
                and not isinstance(indices[1], slice)
            ):
                x = tf.transpose(x)
                if isinstance(values_, int) or isinstance(values_, float):
                    values_ = tf.fill(x.shape[-1:], values_)
                result = tf.tensor_scatter_nd_update(x, [[indices[-1]]], values_[None])
                return type(self)(tf.transpose(result))
            else:
                raise NotImplementedError  # pragma: no cover
        elif isinstance(indices, tuple):
            if all(idx.dtype in [tf.int32, tf.int64] for idx in indices):
                indices = [
                    tf.cast(idx, tf.int64) if idx.dtype == tf.int32 else idx
                    for idx in indices
                ]
            indices = tf.stack(indices, axis=-1)
            if isinstance(values_, int) or isinstance(values_, float):
                values_ = tf.fill((indices.shape[0],), values_)
            return type(self)(tf.tensor_scatter_nd_update(x, indices, values_))
        else:
            raise ValueError  # pragma: no cover

    @samedevice
    def arange(
        self: TensorType,
        start: int,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> TensorType:
        if step is None:
            step = 1
        if stop is None:
            stop = start
            start = 0
        return type(self)(tf.range(start, stop, step))

    def cumsum(self: TensorType, axis: Optional[int] = None) -> TensorType:
        if axis is None:
            x = tf.reshape(self.raw, (-1,))
            return type(self)(tf.cumsum(x, axis=0))
        return type(self)(tf.cumsum(self.raw, axis=axis))

    def flip(self: TensorType, axis: Optional[AxisAxes] = None) -> TensorType:
        if axis is None:
            axis = tuple(range(self.ndim))
        if not isinstance(axis, Iterable):
            axis = (axis,)
        return type(self)(tf.reverse(self.raw, axis=axis))

    def meshgrid(
        self: TensorType, *tensors: TensorType, indexing: str = "xy"
    ) -> Tuple[TensorType, ...]:
        tensors = unwrap_(*tensors)
        outputs = tf.meshgrid(self.raw, *tensors, indexing=indexing)
        return tuple(type(self)(out) for out in outputs)

    def pad(
        self: TensorType,
        paddings: Tuple[Tuple[int, int], ...],
        mode: str = "constant",
        value: float = 0,
    ) -> TensorType:
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

    def slogdet(self: TensorType) -> Tuple[TensorType, TensorType]:
        sign, logabsdet = tf.linalg.slogdet(self.raw)
        return type(self)(sign), type(self)(logabsdet)

    @overload
    def _value_and_grad_fn(
        self: TensorType, f: Callable[..., TensorType]
    ) -> Callable[..., Tuple[TensorType, TensorType]]:
        ...

    @overload  # noqa: F811 (waiting for pyflakes > 2.1.1)
    def _value_and_grad_fn(
        self: TensorType, f: Callable[..., TensorType], has_aux: Literal[False]
    ) -> Callable[..., Tuple[TensorType, TensorType]]:
        ...

    @overload  # noqa: F811 (waiting for pyflakes > 2.1.1)
    def _value_and_grad_fn(
        self: TensorType,
        f: Callable[..., Tuple[TensorType, Any]],
        has_aux: Literal[True],
    ) -> Callable[..., Tuple[TensorType, Any, TensorType]]:
        ...

    def _value_and_grad_fn(  # noqa: F811 (waiting for pyflakes > 2.1.1)
        self: TensorType, f: Callable, has_aux: bool = False
    ) -> Callable[..., Tuple]:
        def value_and_grad(x: TensorType, *args: Any, **kwargs: Any) -> Tuple:
            # using tf.identity to make x independent from possible other instances of x in args
            x_ = TensorFlowTensor(tf.identity(x.raw))
            del x
            with tf.GradientTape() as tape:
                tape.watch(x_.raw)
                if has_aux:
                    loss, aux = f(x_, *args, **kwargs)
                else:
                    loss = f(x_, *args, **kwargs)
            grad = tape.gradient(loss.raw, x_.raw)
            grad = TensorFlowTensor(grad)
            assert grad.shape == x_.shape
            if has_aux:
                return loss, aux, grad
            else:
                return loss, grad

        return value_and_grad

    def sign(self: TensorType) -> TensorType:
        return type(self)(tf.sign(self.raw))

    def sqrt(self: TensorType) -> TensorType:
        return type(self)(tf.sqrt(self.raw))

    def sin(self: TensorType) -> TensorType:
        return type(self)(tf.sin(self.raw))

    def cos(self: TensorType) -> TensorType:
        return type(self)(tf.cos(self.raw))

    def tan(self: TensorType) -> TensorType:
        return type(self)(tf.tan(self.raw))

    def sinh(self: TensorType) -> TensorType:
        return type(self)(tf.sinh(self.raw))

    def cosh(self: TensorType) -> TensorType:
        return type(self)(tf.cosh(self.raw))

    def tanh(self: TensorType) -> TensorType:
        return type(self)(tf.tanh(self.raw))

    def arcsin(self: TensorType) -> TensorType:
        return type(self)(tf.asin(self.raw))

    def arccos(self: TensorType) -> TensorType:
        return type(self)(tf.acos(self.raw))

    def arctan(self: TensorType) -> TensorType:
        return type(self)(tf.atan(self.raw))

    def arcsinh(self: TensorType) -> TensorType:
        return type(self)(tf.asinh(self.raw))

    def arccosh(self: TensorType) -> TensorType:
        return type(self)(tf.acosh(self.raw))

    def arctanh(self: TensorType) -> TensorType:
        return type(self)(tf.atanh(self.raw))

    def inv(self: TensorType) -> TensorType:
        return type(self)(tf.linalg.inv(self.raw))

    def round(self: TensorType) -> TensorType:
        return type(self)(tf.math.round(self.raw))

    def ceil(self: TensorType) -> TensorType:
        return type(self)(tf.math.ceil(self.raw))

    def floor(self: TensorType) -> TensorType:
        return type(self)(tf.math.floor(self.raw))

    def float32(self: TensorType) -> TensorType:
        return self.astype(tf.float32)

    def float64(self: TensorType) -> TensorType:
        return self.astype(tf.float64)

    def where(self: TensorType, x: TensorOrScalar, y: TensorOrScalar) -> TensorType:
        x, y = unwrap_(x, y)
        return type(self)(tf.where(self.raw, x, y))

    @common_dtype
    def __lt__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__lt__(unwrap1(other)))

    @common_dtype
    def __le__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__le__(unwrap1(other)))

    @common_dtype
    def __eq__(self: TensorType, other: TensorOrScalar) -> TensorType:  # type: ignore
        return type(self)(self.raw.__eq__(unwrap1(other)))

    @common_dtype
    def __ne__(self: TensorType, other: TensorOrScalar) -> TensorType:  # type: ignore
        return type(self)(self.raw.__ne__(unwrap1(other)))

    @common_dtype
    def __gt__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__gt__(unwrap1(other)))

    @common_dtype
    def __ge__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__ge__(unwrap1(other)))

    def __getitem__(self: TensorType, index: Any) -> TensorType:
        if isinstance(index, tuple):
            index = tuple(x.raw if isinstance(x, Tensor) else x for x in index)
            basic = all(
                x is None or x is Ellipsis or isinstance(x, int) or isinstance(x, slice)
                for x in index
            )
            if not basic:
                # workaround for missing support for this in TensorFlow
                index = [tf.convert_to_tensor(x) for x in index]
                shapes = [tuple(x.shape) for x in index]
                shape = tuple(max(x) for x in zip(*shapes))
                int64 = any(x.dtype == tf.int64 for x in index)
                for i in range(len(index)):
                    t = index[i]
                    if int64:
                        t = tf.cast(t, tf.int64)  # pragma: no cover
                    assert t.ndim == len(shape)
                    tiling = []
                    for b, k in zip(shape, t.shape):
                        if k == 1:
                            tiling.append(b)
                        elif k == b:
                            tiling.append(1)
                        else:
                            raise ValueError(  # pragma: no cover
                                f"{tuple(t.shape)} cannot be broadcasted to {shape}"
                            )
                    index[i] = tf.tile(t, tiling)
                index = tf.stack(index, axis=-1)
                return type(self)(tf.gather_nd(self.raw, index))
        elif (
            isinstance(index, range)
            or isinstance(index, list)
            or isinstance(index, np.ndarray)
        ):
            return type(self)(tf.gather(self.raw, index))
        elif isinstance(index, Tensor):
            if index.raw.dtype == tf.bool:
                return type(self)(self.raw.__getitem__(index.raw))
            else:
                return type(self)(tf.gather(self.raw, index.raw))
        return type(self)(self.raw.__getitem__(index))

    def take_along_axis(self: TensorType, index: TensorType, axis: int) -> TensorType:
        axis = batch_dims = axis % self.ndim
        if axis != self.ndim - 1:
            raise NotImplementedError(
                "take_along_axis is currently only supported for the last axis"
            )
        return type(self)(
            tf.gather(self.raw, index.raw, axis=axis, batch_dims=batch_dims)
        )

    def bool(self: TensorType) -> TensorType:
        return self.astype(tf.bool)
