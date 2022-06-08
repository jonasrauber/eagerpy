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

from ..types import Axes, AxisAxes, Shape, ShapeOrScalar

from .tensor import Tensor
from .tensor import TensorOrScalar

from .base import BaseTensor
from .base import unwrap_
from .base import unwrap1

if TYPE_CHECKING:
    import torch  # for static analyzers
    from .extensions import NormsMethods  # noqa: F401
else:
    # lazy import in PyTorchTensor
    torch = None


# stricter TensorType to get additional type information from the raw method
TensorType = TypeVar("TensorType", bound="PyTorchTensor")


def assert_bool(x: Any) -> None:
    if not isinstance(x, Tensor):
        return
    if x.dtype != torch.bool:
        raise ValueError(f"requires dtype bool, got {x.dtype}, consider t.bool().all()")


class PyTorchTensor(BaseTensor):
    __slots__ = ()

    # more specific types for the extensions
    norms: "NormsMethods[PyTorchTensor]"

    def __init__(self, raw: "torch.Tensor"):
        global torch
        if torch is None:
            torch = import_module("torch")  # type: ignore
        super().__init__(raw)

    @property
    def raw(self) -> "torch.Tensor":
        return cast(torch.Tensor, super().raw)

    def numpy(self: TensorType) -> Any:
        a = self.raw.detach().cpu().numpy()
        if a.flags.writeable:
            # without the check, we would attempt to set it on array
            # scalars, and that would fail
            a.flags.writeable = False
        return a

    def item(self) -> Union[int, float, bool]:
        return self.raw.item()

    @property
    def shape(self) -> Shape:
        return self.raw.shape

    def reshape(self: TensorType, shape: Union[Shape, int]) -> TensorType:
        if isinstance(shape, int):
            shape = (shape,)
        return type(self)(self.raw.reshape(shape))

    def astype(self: TensorType, dtype: Any) -> TensorType:
        return type(self)(self.raw.to(dtype))

    def clip(self: TensorType, min_: float, max_: float) -> TensorType:
        return type(self)(self.raw.clamp(min_, max_))

    def square(self: TensorType) -> TensorType:
        return type(self)(self.raw**2)

    def sin(self: TensorType) -> TensorType:
        return type(self)(torch.sin(self.raw))

    def cos(self: TensorType) -> TensorType:
        return type(self)(torch.cos(self.raw))

    def tan(self: TensorType) -> TensorType:
        return type(self)(torch.tan(self.raw))

    def sinh(self: TensorType) -> TensorType:
        return type(self)(torch.sinh(self.raw))

    def cosh(self: TensorType) -> TensorType:
        return type(self)(torch.cosh(self.raw))

    def tanh(self: TensorType) -> TensorType:
        return type(self)(torch.tanh(self.raw))

    def arcsin(self: TensorType) -> TensorType:
        return type(self)(torch.asin(self.raw))

    def arccos(self: TensorType) -> TensorType:
        return type(self)(torch.acos(self.raw))

    def arctan(self: TensorType) -> TensorType:
        return type(self)(torch.atan(self.raw))

    def arcsinh(self: TensorType) -> TensorType:
        return type(self)(torch.asinh(self.raw))

    def arccosh(self: TensorType) -> TensorType:
        return type(self)(torch.acosh(self.raw))

    def arctanh(self: TensorType) -> TensorType:
        return type(self)(torch.atanh(self.raw))

    def sum(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        if axis is None and not keepdims:
            return type(self)(self.raw.sum())
        if axis is None:
            axis = tuple(range(self.ndim))
        return type(self)(self.raw.sum(dim=axis, keepdim=keepdims))

    def prod(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        if axis is None and not keepdims:
            return type(self)(self.raw.prod())
        if axis is None:
            axis = tuple(range(self.ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)
        x = self.raw
        for i in sorted(axis, reverse=True):
            x = x.prod(i, keepdim=keepdims)
        return type(self)(x)

    def mean(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        if self.raw.dtype not in [torch.float16, torch.float32, torch.float64]:
            raise ValueError(
                f"Can only calculate the mean of floating types. Got {self.raw.dtype} instead."
            )
        if axis is None and not keepdims:
            return type(self)(self.raw.mean())
        if axis is None:
            axis = tuple(range(self.ndim))
        return type(self)(self.raw.mean(dim=axis, keepdim=keepdims))

    def min(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        """
        simplify once this issue has been fixed:
        https://github.com/pytorch/pytorch/issues/28213
        """
        if axis is None and not keepdims:
            return type(self)(self.raw.min())
        if axis is None:
            axis = tuple(range(self.ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)
        x = self.raw
        for i in sorted(axis, reverse=True):
            x, _ = x.min(i, keepdim=keepdims)
        return type(self)(x)

    def max(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        """
        simplify once this issue has been fixed:
        https://github.com/pytorch/pytorch/issues/28213
        """
        if axis is None and not keepdims:
            return type(self)(self.raw.max())
        if axis is None:
            axis = tuple(range(self.ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)
        x = self.raw
        for i in sorted(axis, reverse=True):
            x, _ = x.max(i, keepdim=keepdims)
        return type(self)(x)

    def minimum(self: TensorType, other: TensorOrScalar) -> TensorType:
        if isinstance(other, Tensor):
            other_ = other.raw
        elif isinstance(other, int) or isinstance(other, float):
            other_ = torch.full_like(self.raw, other)
        else:
            raise TypeError(
                "expected x to be a Tensor, int or float"
            )  # pragma: no cover
        return type(self)(torch.min(self.raw, other_))

    def maximum(self: TensorType, other: TensorOrScalar) -> TensorType:
        if isinstance(other, Tensor):
            other_ = other.raw
        elif isinstance(other, int) or isinstance(other, float):
            other_ = torch.full_like(self.raw, other)
        else:
            raise TypeError(
                "expected x to be a Tensor, int or float"
            )  # pragma: no cover
        return type(self)(torch.max(self.raw, other_))

    def argmin(self: TensorType, axis: Optional[int] = None) -> TensorType:
        return type(self)(self.raw.argmin(dim=axis))

    def argmax(self: TensorType, axis: Optional[int] = None) -> TensorType:
        return type(self)(self.raw.argmax(dim=axis))

    def argsort(self: TensorType, axis: int = -1) -> TensorType:
        return type(self)(self.raw.argsort(dim=axis))

    def sort(self: TensorType, axis: int = -1) -> TensorType:
        return type(self)(self.raw.sort(dim=axis).values)

    def topk(
        self: TensorType, k: int, sorted: bool = True
    ) -> Tuple[TensorType, TensorType]:
        values, indices = self.raw.topk(k, sorted=sorted)
        return type(self)(values), type(self)(indices)

    def uniform(
        self: TensorType, shape: ShapeOrScalar, low: float = 0.0, high: float = 1.0
    ) -> TensorType:
        return type(self)(
            torch.rand(shape, dtype=self.raw.dtype, device=self.raw.device)
            * (high - low)
            + low
        )

    def normal(
        self: TensorType, shape: ShapeOrScalar, mean: float = 0.0, stddev: float = 1.0
    ) -> TensorType:
        return type(self)(
            torch.randn(shape, dtype=self.raw.dtype, device=self.raw.device) * stddev
            + mean
        )

    def ones(self: TensorType, shape: ShapeOrScalar) -> TensorType:
        return type(self)(
            torch.ones(shape, dtype=self.raw.dtype, device=self.raw.device)
        )

    def zeros(self: TensorType, shape: ShapeOrScalar) -> TensorType:
        return type(self)(
            torch.zeros(shape, dtype=self.raw.dtype, device=self.raw.device)
        )

    def ones_like(self: TensorType) -> TensorType:
        return type(self)(torch.ones_like(self.raw))

    def zeros_like(self: TensorType) -> TensorType:
        return type(self)(torch.zeros_like(self.raw))

    def full_like(self: TensorType, fill_value: float) -> TensorType:
        return type(self)(torch.full_like(self.raw, fill_value))

    def onehot_like(
        self: TensorType, indices: TensorType, *, value: float = 1
    ) -> TensorType:
        if self.ndim != 2:
            raise ValueError("onehot_like only supported for 2D tensors")
        if indices.ndim != 1:
            raise ValueError("onehot_like requires 1D indices")
        if len(indices) != len(self):
            raise ValueError("length of indices must match length of tensor")
        x = torch.zeros_like(self.raw)
        rows = np.arange(x.shape[0])
        x[rows, indices.raw] = value
        return type(self)(x)

    def from_numpy(self: TensorType, a: Any) -> TensorType:
        return type(self)(torch.as_tensor(a, device=self.raw.device))

    def _concatenate(
        self: TensorType, tensors: Iterable[TensorType], axis: int = 0
    ) -> TensorType:
        # concatenates only "tensors", but not "self"
        tensors_ = unwrap_(*tensors)
        return type(self)(torch.cat(tensors_, dim=axis))

    def _stack(
        self: TensorType, tensors: Iterable[TensorType], axis: int = 0
    ) -> TensorType:
        # stacks only "tensors", but not "self"
        tensors_ = unwrap_(*tensors)
        return type(self)(torch.stack(tensors_, dim=axis))

    def transpose(self: TensorType, axes: Optional[Axes] = None) -> TensorType:
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return type(self)(self.raw.permute(*axes))

    def all(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        assert_bool(self)
        if axis is None and not keepdims:
            return type(self)(self.raw.all())
        if axis is None:
            axis = tuple(range(self.ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)
        x = self.raw
        for i in sorted(axis, reverse=True):
            x = x.all(i, keepdim=keepdims)
        return type(self)(x)

    def any(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        assert_bool(self)
        if axis is None and not keepdims:
            return type(self)(self.raw.any())
        if axis is None:
            axis = tuple(range(self.ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)
        x = self.raw
        for i in sorted(axis, reverse=True):
            x = x.any(i, keepdim=keepdims)
        return type(self)(x)

    def logical_and(self: TensorType, other: TensorOrScalar) -> TensorType:
        assert_bool(self)
        assert_bool(other)
        return type(self)(self.raw & unwrap1(other))

    def logical_or(self: TensorType, other: TensorOrScalar) -> TensorType:
        assert_bool(self)
        assert_bool(other)
        return type(self)(self.raw | unwrap1(other))

    def logical_not(self: TensorType) -> TensorType:
        assert_bool(self)
        return type(self)(~self.raw)

    def exp(self: TensorType) -> TensorType:
        return type(self)(torch.exp(self.raw))

    def log(self: TensorType) -> TensorType:
        return type(self)(torch.log(self.raw))

    def log2(self: TensorType) -> TensorType:
        return type(self)(torch.log2(self.raw))

    def log10(self: TensorType) -> TensorType:
        return type(self)(torch.log10(self.raw))

    def log1p(self: TensorType) -> TensorType:
        return type(self)(torch.log1p(self.raw))

    def tile(self: TensorType, multiples: Axes) -> TensorType:
        if len(multiples) != self.ndim:
            raise ValueError("multiples requires one entry for each dimension")
        return type(self)(self.raw.repeat(multiples))

    def softmax(self: TensorType, axis: int = -1) -> TensorType:
        return type(self)(torch.nn.functional.softmax(self.raw, dim=axis))

    def log_softmax(self: TensorType, axis: int = -1) -> TensorType:
        return type(self)(torch.nn.functional.log_softmax(self.raw, dim=axis))

    def squeeze(self: TensorType, axis: Optional[AxisAxes] = None) -> TensorType:
        if axis is None:
            return type(self)(self.raw.squeeze())
        if not isinstance(axis, Iterable):
            axis = (axis,)
        x = self.raw
        for i in sorted(axis, reverse=True):
            if x.shape[i] != 1:
                raise ValueError(
                    "cannot select an axis to squeeze out which has size not equal to one"
                )
            x = x.squeeze(dim=i)
        return type(self)(x)

    def expand_dims(self: TensorType, axis: int) -> TensorType:
        return type(self)(self.raw.unsqueeze(dim=axis))

    def full(self: TensorType, shape: ShapeOrScalar, value: float) -> TensorType:
        if not isinstance(shape, Iterable):
            shape = (shape,)
        return type(self)(
            torch.full(shape, value, dtype=self.raw.dtype, device=self.raw.device)
        )

    def index_update(
        self: TensorType, indices: Any, values: TensorOrScalar
    ) -> TensorType:
        indices, values_ = unwrap_(indices, values)
        if isinstance(indices, tuple):
            indices = unwrap_(*indices)
        x = self.raw.clone()
        x[indices] = values_
        return type(self)(x)

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
        return type(self)(
            torch.arange(start=start, end=stop, step=step, device=self.raw.device)
        )

    def cumsum(self: TensorType, axis: Optional[int] = None) -> TensorType:
        if axis is None:
            return type(self)(self.raw.reshape(-1).cumsum(dim=0))
        return type(self)(self.raw.cumsum(dim=axis))

    def flip(self: TensorType, axis: Optional[AxisAxes] = None) -> TensorType:
        if axis is None:
            axis = tuple(range(self.ndim))
        if not isinstance(axis, Iterable):
            axis = (axis,)
        return type(self)(self.raw.flip(dims=axis))

    def meshgrid(
        self: TensorType, *tensors: TensorType, indexing: str = "xy"
    ) -> Tuple[TensorType, ...]:
        tensors = unwrap_(*tensors)
        if indexing == "ij" or len(tensors) == 0:
            outputs = torch.meshgrid(self.raw, *tensors)  # type: ignore
        elif indexing == "xy":
            outputs = torch.meshgrid(tensors[0], self.raw, *tensors[1:])  # type: ignore
        else:
            raise ValueError(  # pragma: no cover
                f"Valid values for indexing are 'xy' and 'ij', got {indexing}"
            )
        results = [type(self)(out) for out in outputs]
        if indexing == "xy" and len(results) >= 2:
            results[0], results[1] = results[1], results[0]
        return tuple(results)

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
            paddings = paddings[k:]
        paddings_ = list(x for p in reversed(paddings) for x in p)
        return type(self)(
            torch.nn.functional.pad(self.raw, paddings_, mode=mode, value=value)
        )

    def isnan(self: TensorType) -> TensorType:
        return type(self)(torch.isnan(self.raw))

    def isinf(self: TensorType) -> TensorType:
        return type(self)(torch.isinf(self.raw))

    def crossentropy(self: TensorType, labels: TensorType) -> TensorType:
        if self.ndim != 2:
            raise ValueError("crossentropy only supported for 2D logits tensors")
        if self.shape[:1] != labels.shape:
            raise ValueError("labels must be 1D and must match the length of logits")
        return type(self)(
            torch.nn.functional.cross_entropy(self.raw, labels.raw, reduction="none")
        )

    def slogdet(self: TensorType) -> Tuple[TensorType, TensorType]:
        sign, logabsdet = torch.slogdet(self.raw)
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
            x = type(self)(x.raw.clone().requires_grad_())
            if has_aux:
                loss, aux = f(x, *args, **kwargs)
            else:
                loss = f(x, *args, **kwargs)
            loss = loss.raw
            grad_raw = torch.autograd.grad(loss, x.raw)[0]
            grad = type(self)(grad_raw)
            assert grad.shape == x.shape
            loss = loss.detach()
            loss = type(self)(loss)
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

    def sign(self: TensorType) -> TensorType:
        return type(self)(torch.sign(self.raw))

    def sqrt(self: TensorType) -> TensorType:
        return type(self)(torch.sqrt(self.raw))

    def inv(self: TensorType) -> TensorType:
        return type(self)(torch.linalg.inv(self.raw))

    def round(self: TensorType) -> TensorType:
        return type(self)(torch.round(self.raw))

    def ceil(self: TensorType) -> TensorType:
        return type(self)(torch.ceil(self.raw))

    def floor(self: TensorType) -> TensorType:
        return type(self)(torch.floor(self.raw))

    def float32(self: TensorType) -> TensorType:
        return self.astype(torch.float32)

    def float64(self: TensorType) -> TensorType:
        return self.astype(torch.float64)

    def where(self: TensorType, x: TensorOrScalar, y: TensorOrScalar) -> TensorType:

        if isinstance(x, Tensor):
            x_ = x.raw
        elif isinstance(x, int) or isinstance(x, float):
            if isinstance(y, Tensor):
                dtype = y.raw.dtype
            else:
                dtype = torch.float32
            x_ = torch.full_like(self.raw, x, dtype=dtype)
        else:
            raise TypeError(
                "expected x to be a Tensor, int or float"
            )  # pragma: no cover
        if isinstance(y, Tensor):
            y_ = y.raw
        elif isinstance(y, int) or isinstance(y, float):
            if isinstance(x, Tensor):
                dtype = x.raw.dtype
            else:
                dtype = torch.float32
            y_ = torch.full_like(self.raw, y, dtype=dtype)
        return type(self)(torch.where(self.raw, x_, y_))

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
        return type(self)(torch.gather(self.raw, axis, index.raw))

    def bool(self: TensorType) -> TensorType:
        return self.astype(torch.bool)
