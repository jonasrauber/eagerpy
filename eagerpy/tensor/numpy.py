from typing import (
    Tuple,
    cast,
    Union,
    Any,
    Iterable,
    Optional,
    overload,
    Callable,
    TYPE_CHECKING,
)
from typing_extensions import Literal
import numpy as np

from ..types import Axes, AxisAxes, Shape, ShapeOrScalar

from .tensor import TensorType
from .tensor import Tensor
from .tensor import TensorOrScalar

from .base import BaseTensor
from .base import unwrap_
from .base import unwrap1

if TYPE_CHECKING:
    from .extensions import NormsMethods  # noqa: F401


def assert_bool(x: Any) -> None:
    if not isinstance(x, Tensor):
        return
    if x.dtype != np.dtype("bool"):
        raise ValueError(f"requires dtype bool, got {x.dtype}, consider t.bool().all()")


class NumPyTensor(BaseTensor):
    __slots__ = ()

    # more specific types for the extensions
    norms: "NormsMethods[NumPyTensor]"

    def __init__(self, raw: "np.ndarray"):  # type: ignore
        super().__init__(raw)

    @property
    def raw(self) -> "np.ndarray":  # type: ignore
        return super().raw

    def numpy(self: TensorType) -> Any:
        a = self.raw.view()
        if a.flags.writeable:
            # without the check, we would attempt to set it on array
            # scalars, and that would fail
            a.flags.writeable = False
        return a

    def item(self) -> Union[int, float, bool]:
        return self.raw.item()  # type: ignore

    @property
    def shape(self: TensorType) -> Shape:
        return cast(Tuple, self.raw.shape)

    def reshape(self: TensorType, shape: Union[Shape, int]) -> TensorType:
        if isinstance(shape, int):
            shape = (shape,)
        return type(self)(self.raw.reshape(shape))

    def astype(self: TensorType, dtype: Any) -> TensorType:
        return type(self)(self.raw.astype(dtype))

    def clip(self: TensorType, min_: float, max_: float) -> TensorType:
        return type(self)(np.clip(self.raw, min_, max_))

    def square(self: TensorType) -> TensorType:
        return type(self)(np.square(self.raw))

    def arctanh(self: TensorType) -> TensorType:
        return type(self)(np.arctanh(self.raw))

    def sum(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        return type(self)(self.raw.sum(axis=axis, keepdims=keepdims))

    def prod(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        return type(self)(self.raw.prod(axis=axis, keepdims=keepdims))

    def mean(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        if self.raw.dtype not in [np.float16, np.float32, np.float64]:
            raise ValueError(
                f"Can only calculate the mean of floating types. Got {self.raw.dtype} instead."
            )
        return type(self)(self.raw.mean(axis=axis, keepdims=keepdims))

    def min(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        return type(self)(self.raw.min(axis=axis, keepdims=keepdims))

    def max(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        return type(self)(self.raw.max(axis=axis, keepdims=keepdims))

    def minimum(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(np.minimum(self.raw, unwrap1(other)))

    def maximum(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(np.maximum(self.raw, unwrap1(other)))

    def argmin(self: TensorType, axis: Optional[int] = None) -> TensorType:
        return type(self)(self.raw.argmin(axis=axis))

    def argmax(self: TensorType, axis: Optional[int] = None) -> TensorType:
        return type(self)(self.raw.argmax(axis=axis))

    def argsort(self: TensorType, axis: int = -1) -> TensorType:
        return type(self)(self.raw.argsort(axis=axis))

    def sort(self: TensorType, axis: int = -1) -> TensorType:
        return type(self)(np.sort(self.raw, axis=axis))

    def topk(
        self: TensorType, k: int, sorted: bool = True
    ) -> Tuple[TensorType, TensorType]:
        idx = np.take(np.argpartition(self.raw, k - 1), np.arange(-k, 0), axis=-1)
        val = np.take_along_axis(self.raw, idx, axis=-1)
        if sorted:
            perm = np.flip(np.argsort(val, axis=-1), axis=-1)
            idx = np.take_along_axis(idx, perm, axis=-1)
            val = np.take_along_axis(self.raw, idx, axis=-1)
        return type(self)(val), type(self)(idx)

    def uniform(
        self: TensorType, shape: ShapeOrScalar, low: float = 0.0, high: float = 1.0
    ) -> TensorType:
        return type(self)(np.random.uniform(low, high, size=shape))

    def normal(
        self: TensorType, shape: ShapeOrScalar, mean: float = 0.0, stddev: float = 1.0
    ) -> TensorType:
        return type(self)(np.random.normal(mean, stddev, size=shape))

    def ones(self: TensorType, shape: ShapeOrScalar) -> TensorType:
        return type(self)(np.ones(shape, dtype=self.raw.dtype))

    def zeros(self: TensorType, shape: ShapeOrScalar) -> TensorType:
        return type(self)(np.zeros(shape, dtype=self.raw.dtype))

    def ones_like(self: TensorType) -> TensorType:
        return type(self)(np.ones_like(self.raw))

    def zeros_like(self: TensorType) -> TensorType:
        return type(self)(np.zeros_like(self.raw))

    def full_like(self: TensorType, fill_value: float) -> TensorType:
        return type(self)(np.full_like(self.raw, fill_value))

    def onehot_like(
        self: TensorType, indices: TensorType, *, value: float = 1
    ) -> TensorType:
        if self.ndim != 2:
            raise ValueError("onehot_like only supported for 2D tensors")
        if indices.ndim != 1:
            raise ValueError("onehot_like requires 1D indices")
        if len(indices) != len(self):
            raise ValueError("length of indices must match length of tensor")
        x = np.zeros_like(self.raw)
        rows = np.arange(len(x))
        x[rows, indices.raw] = value
        return type(self)(x)

    def from_numpy(self: TensorType, a: Any) -> TensorType:
        return type(self)(np.asarray(a))

    def _concatenate(
        self: TensorType, tensors: Iterable[TensorType], axis: int = 0
    ) -> TensorType:
        # concatenates only "tensors", but not "self"
        tensors_ = unwrap_(*tensors)
        return type(self)(np.concatenate(tensors_, axis=axis))

    def _stack(
        self: TensorType, tensors: Iterable[TensorType], axis: int = 0
    ) -> TensorType:
        # stacks only "tensors", but not "self"
        tensors_ = unwrap_(*tensors)
        return type(self)(np.stack(tensors_, axis=axis))

    def transpose(self: TensorType, axes: Optional[Axes] = None) -> TensorType:
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return type(self)(np.transpose(self.raw, axes=axes))

    def all(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        assert_bool(self)
        return type(self)(self.raw.all(axis=axis, keepdims=keepdims))

    def any(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        assert_bool(self)
        return type(self)(self.raw.any(axis=axis, keepdims=keepdims))

    def logical_and(self: TensorType, other: TensorOrScalar) -> TensorType:
        assert_bool(self)
        assert_bool(other)
        return type(self)(np.logical_and(self.raw, unwrap1(other)))

    def logical_or(self: TensorType, other: TensorOrScalar) -> TensorType:
        assert_bool(self)
        assert_bool(other)
        return type(self)(np.logical_or(self.raw, unwrap1(other)))

    def logical_not(self: TensorType) -> TensorType:
        assert_bool(self)
        return type(self)(np.logical_not(self.raw))

    def exp(self: TensorType) -> TensorType:
        return type(self)(np.exp(self.raw))

    def log(self: TensorType) -> TensorType:
        return type(self)(np.log(self.raw))

    def log2(self: TensorType) -> TensorType:
        return type(self)(np.log2(self.raw))

    def log10(self: TensorType) -> TensorType:
        return type(self)(np.log10(self.raw))

    def log1p(self: TensorType) -> TensorType:
        return type(self)(np.log1p(self.raw))

    def tile(self: TensorType, multiples: Axes) -> TensorType:
        multiples = unwrap1(multiples)
        if len(multiples) != self.ndim:
            raise ValueError("multiples requires one entry for each dimension")
        return type(self)(np.tile(self.raw, multiples))

    def softmax(self: TensorType, axis: int = -1) -> TensorType:
        # for numerical reasons we subtract the max logit
        # (mathematically it doesn't matter!)
        # otherwise exp(logits) might become too large or too small
        logits = self.raw
        logits = logits - logits.max(axis=axis, keepdims=True)
        e = np.exp(logits)
        return type(self)(e / e.sum(axis=axis, keepdims=True))

    def log_softmax(self: TensorType, axis: int = -1) -> TensorType:
        # for numerical reasons we subtract the max logit
        # (mathematically it doesn't matter!)
        # otherwise exp(logits) might become too large or too small
        logits = self.raw
        logits = logits - logits.max(axis=axis, keepdims=True)
        log_sum_exp = np.log(np.exp(logits).sum(axis=axis, keepdims=True))
        return type(self)(logits - log_sum_exp)

    def squeeze(self: TensorType, axis: Optional[AxisAxes] = None) -> TensorType:
        return type(self)(self.raw.squeeze(axis=axis))

    def expand_dims(self: TensorType, axis: int) -> TensorType:
        return type(self)(np.expand_dims(self.raw, axis=axis))

    def full(self: TensorType, shape: ShapeOrScalar, value: float) -> TensorType:
        return type(self)(np.full(shape, value, dtype=self.raw.dtype))

    def index_update(
        self: TensorType, indices: Any, values: TensorOrScalar
    ) -> TensorType:
        indices, values = unwrap_(indices, values)
        if isinstance(indices, tuple):
            indices = unwrap_(*indices)
        x = self.raw.copy()
        x[indices] = values
        return type(self)(x)

    def arange(
        self: TensorType,
        start: int,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> TensorType:
        return type(self)(np.arange(start, stop, step))

    def cumsum(self: TensorType, axis: Optional[int] = None) -> TensorType:
        return type(self)(self.raw.cumsum(axis=axis))

    def flip(self: TensorType, axis: Optional[AxisAxes] = None) -> TensorType:
        return type(self)(np.flip(self.raw, axis=axis))

    def meshgrid(
        self: TensorType, *tensors: TensorType, indexing: str = "xy"
    ) -> Tuple[TensorType, ...]:
        tensors = unwrap_(*tensors)
        outputs = np.meshgrid(self.raw, *tensors, indexing=indexing)
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
        if mode == "constant":
            return type(self)(
                np.pad(self.raw, paddings, mode=mode, constant_values=value)
            )
        else:
            return type(self)(np.pad(self.raw, paddings, mode=mode))

    def isnan(self: TensorType) -> TensorType:
        return type(self)(np.isnan(self.raw))

    def isinf(self: TensorType) -> TensorType:
        return type(self)(np.isinf(self.raw))

    def crossentropy(self: TensorType, labels: TensorType) -> TensorType:
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

    def slogdet(self: TensorType) -> Tuple[TensorType, TensorType]:
        sign, logabsdet = np.linalg.slogdet(self.raw)
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
        # TODO: maybe implement this using https://github.com/HIPS/autograd
        raise NotImplementedError  # pragma: no cover

    def sign(self: TensorType) -> TensorType:
        return type(self)(np.sign(self.raw))

    def sqrt(self: TensorType) -> TensorType:
        return type(self)(np.sqrt(self.raw))

    def tanh(self: TensorType) -> TensorType:
        return type(self)(np.tanh(self.raw))

    def float32(self: TensorType) -> TensorType:
        return self.astype(np.float32)

    def float64(self: TensorType) -> TensorType:
        return self.astype(np.float64)

    def where(self: TensorType, x: TensorOrScalar, y: TensorOrScalar) -> TensorType:
        x, y = unwrap_(x, y)
        return type(self)(np.where(self.raw, x, y))

    def __lt__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__lt__(unwrap1(other)))

    def __le__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__le__(unwrap1(other)))

    def __eq__(self: TensorType, other: TensorOrScalar) -> TensorType:  # type: ignore
        return type(self)(self.raw.__eq__(unwrap1(other)))

    def __ne__(self: TensorType, other: TensorOrScalar) -> TensorType:  # type: ignore
        return type(self)(self.raw.__ne__(unwrap1(other)))

    def __gt__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__gt__(unwrap1(other)))

    def __ge__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__ge__(unwrap1(other)))

    def __getitem__(self: TensorType, index: Any) -> TensorType:
        if isinstance(index, tuple):
            index = tuple(x.raw if isinstance(x, Tensor) else x for x in index)
        elif isinstance(index, Tensor):
            index = index.raw
        return type(self)(self.raw[index])

    def take_along_axis(self: TensorType, index: TensorType, axis: int) -> TensorType:
        if axis % self.ndim != self.ndim - 1:
            raise NotImplementedError(
                "take_along_axis is currently only supported for the last axis"
            )
        return type(self)(np.take_along_axis(self.raw, index.raw, axis=axis))

    def bool(self: TensorType) -> TensorType:
        return self.astype(np.dtype("bool"))
