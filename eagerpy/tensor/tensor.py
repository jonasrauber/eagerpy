from abc import ABCMeta, abstractmethod
from typing import (
    TypeVar,
    Callable,
    Tuple,
    Any,
    overload,
    Iterable,
    Iterator,
    Union,
    Optional,
    Type,
    TYPE_CHECKING,
    cast,
)
from typing_extensions import Literal, final

from ..types import Axes, AxisAxes, Shape, ShapeOrScalar

if TYPE_CHECKING:
    from .extensions import NormsMethods  # noqa: F401


TensorType = TypeVar("TensorType", bound="Tensor")

# using Tensor instead of TensorType because of a MyPy bug
# https://github.com/python/mypy/issues/3644
TensorOrScalar = Union["Tensor", int, float]


class LazyCachedAccessor:
    # supports caching under a different name (because Tensor uses __slots__
    # and thus we cannot override the LazyCachedAccessor class var intself)

    # supports lazy extension loading to break cyclic dependencies

    def __init__(self, cache_name: str, extension_name: str):
        self._cache_name = cache_name
        self._extension_name = extension_name

    @property
    def _extension(self) -> Any:  # Type[object]:
        # only imported once needed to break cyclic dependencies
        from . import extensions

        return getattr(extensions, self._extension_name)

    def __get__(
        self, instance: Optional["Tensor"], owner: Optional[Type["Tensor"]] = None
    ) -> Any:
        if instance is None:
            # accessed as a class attribute
            return self._extension

        methods = getattr(instance, self._cache_name, None)
        if methods is not None:
            return methods

        # create the extension for this instance
        methods = self._extension(instance)

        # add it to the instance to avoid recreation
        instance.__setattr__(self._cache_name, methods)
        return methods


class Tensor(metaclass=ABCMeta):
    """Base class defining the common interface of all EagerPy Tensors"""

    # each extension neeeds a slot to cache the instantiated extension
    __slots__ = ("_norms",)

    __array_ufunc__ = None

    # shorten the class name to eagerpy.Tensor (does not help with MyPy)
    __module__ = "eagerpy"

    @abstractmethod
    def __init__(self, raw: Any):
        ...

    @property
    @abstractmethod
    def raw(self) -> Any:
        ...

    @property
    @abstractmethod
    def dtype(self: TensorType) -> Any:
        ...

    @abstractmethod
    def __repr__(self: TensorType) -> str:
        ...

    @abstractmethod
    def __format__(self: TensorType, format_spec: str) -> str:
        ...

    @abstractmethod
    def __getitem__(self: TensorType, index: Any) -> TensorType:
        ...

    @abstractmethod
    def __bool__(self: TensorType) -> bool:
        ...

    @abstractmethod
    def __len__(self: TensorType) -> int:
        ...

    @abstractmethod
    def __abs__(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def __neg__(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def __add__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __radd__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __sub__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __rsub__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __mul__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __rmul__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __truediv__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __rtruediv__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __floordiv__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __rfloordiv__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __mod__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __lt__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __le__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __eq__(self: TensorType, other: TensorOrScalar) -> TensorType:  # type: ignore
        # we ignore the type errors caused by wrong type annotations for object
        # https://github.com/python/typeshed/issues/3685
        ...

    @abstractmethod
    def __ne__(self: TensorType, other: TensorOrScalar) -> TensorType:  # type: ignore
        # we ignore the type errors caused by wrong type annotations for object
        # https://github.com/python/typeshed/issues/3685
        ...

    @abstractmethod
    def __gt__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __ge__(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def __pow__(self: TensorType, exponent: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def sign(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def sqrt(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def sin(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def cos(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def tan(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def sinh(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def cosh(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def tanh(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def arcsin(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def arccos(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def arctan(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def arcsinh(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def arccosh(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def arctanh(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def inv(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def round(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def ceil(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def floor(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def float32(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def float64(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def where(self: TensorType, x: TensorOrScalar, y: TensorOrScalar) -> TensorType:
        ...

    @property
    @abstractmethod
    def ndim(self: TensorType) -> int:
        ...

    @abstractmethod
    def numpy(self: TensorType) -> Any:
        ...

    @abstractmethod
    def item(self: TensorType) -> Union[int, float, bool]:
        ...

    @property
    @abstractmethod
    def shape(self: TensorType) -> Shape:
        ...

    @abstractmethod
    def reshape(self: TensorType, shape: Union[Shape, int]) -> TensorType:
        ...

    @abstractmethod
    def astype(self: TensorType, dtype: Any) -> TensorType:
        ...

    @abstractmethod
    def clip(self: TensorType, min_: float, max_: float) -> TensorType:
        ...

    @abstractmethod
    def square(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def sum(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        ...

    @abstractmethod
    def prod(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        ...

    @abstractmethod
    def mean(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        ...

    @abstractmethod
    def min(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        ...

    @abstractmethod
    def max(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        ...

    @abstractmethod
    def minimum(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def maximum(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def argmin(self: TensorType, axis: Optional[int] = None) -> TensorType:
        ...

    @abstractmethod
    def argmax(self: TensorType, axis: Optional[int] = None) -> TensorType:
        ...

    @abstractmethod
    def argsort(self: TensorType, axis: int = -1) -> TensorType:
        ...

    @abstractmethod
    def sort(self: TensorType, axis: int = -1) -> TensorType:
        ...

    @abstractmethod
    def topk(
        self: TensorType, k: int, sorted: bool = True
    ) -> Tuple[TensorType, TensorType]:
        ...

    @abstractmethod
    def uniform(
        self: TensorType, shape: ShapeOrScalar, low: float = 0.0, high: float = 1.0
    ) -> TensorType:
        ...

    @abstractmethod
    def normal(
        self: TensorType, shape: ShapeOrScalar, mean: float = 0.0, stddev: float = 1.0
    ) -> TensorType:
        ...

    @abstractmethod
    def ones(self: TensorType, shape: ShapeOrScalar) -> TensorType:
        ...

    @abstractmethod
    def zeros(self: TensorType, shape: ShapeOrScalar) -> TensorType:
        ...

    @abstractmethod
    def ones_like(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def zeros_like(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def full_like(self: TensorType, fill_value: float) -> TensorType:
        ...

    @abstractmethod
    def onehot_like(
        self: TensorType, indices: TensorType, *, value: float = 1
    ) -> TensorType:
        ...

    @abstractmethod
    def from_numpy(self: TensorType, a: Any) -> TensorType:
        ...

    @abstractmethod
    def _concatenate(
        self: TensorType, tensors: Iterable[TensorType], axis: int = 0
    ) -> TensorType:
        ...

    @abstractmethod
    def _stack(
        self: TensorType, tensors: Iterable[TensorType], axis: int = 0
    ) -> TensorType:
        ...

    @abstractmethod
    def transpose(self: TensorType, axes: Optional[Axes] = None) -> TensorType:
        ...

    @abstractmethod
    def take_along_axis(self: TensorType, index: TensorType, axis: int) -> TensorType:
        ...

    @abstractmethod
    def all(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        ...

    @abstractmethod
    def any(
        self: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
    ) -> TensorType:
        ...

    @abstractmethod
    def logical_and(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def logical_or(self: TensorType, other: TensorOrScalar) -> TensorType:
        ...

    @abstractmethod
    def logical_not(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def exp(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def log(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def log2(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def log10(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def log1p(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def tile(self: TensorType, multiples: Axes) -> TensorType:
        ...

    @abstractmethod
    def softmax(self: TensorType, axis: int = -1) -> TensorType:
        ...

    @abstractmethod
    def log_softmax(self: TensorType, axis: int = -1) -> TensorType:
        ...

    @abstractmethod
    def squeeze(self: TensorType, axis: Optional[AxisAxes] = None) -> TensorType:
        ...

    @abstractmethod
    def expand_dims(self: TensorType, axis: int) -> TensorType:
        ...

    @abstractmethod
    def full(self: TensorType, shape: ShapeOrScalar, value: float) -> TensorType:
        ...

    @abstractmethod
    def index_update(
        self: TensorType, indices: Any, values: TensorOrScalar
    ) -> TensorType:
        ...

    @abstractmethod
    def arange(
        self: TensorType,
        start: int,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> TensorType:
        ...

    @abstractmethod
    def cumsum(self: TensorType, axis: Optional[int] = None) -> TensorType:
        ...

    @abstractmethod
    def flip(self: TensorType, axis: Optional[AxisAxes] = None) -> TensorType:
        ...

    @abstractmethod
    def meshgrid(
        self: TensorType, *tensors: TensorType, indexing: str = "xy"
    ) -> Tuple[TensorType, ...]:
        ...

    @abstractmethod
    def pad(
        self: TensorType,
        paddings: Tuple[Tuple[int, int], ...],
        mode: str = "constant",
        value: float = 0,
    ) -> TensorType:
        ...

    @abstractmethod
    def isnan(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def isinf(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def crossentropy(self: TensorType, labels: TensorType) -> TensorType:
        ...

    @abstractmethod
    def slogdet(matrix: TensorType) -> Tuple[TensorType, TensorType]:
        ...

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

    @abstractmethod  # noqa: F811 (waiting for pyflakes > 2.1.1)
    def _value_and_grad_fn(
        self: TensorType, f: Callable, has_aux: bool = False
    ) -> Callable[..., Tuple]:
        ...

    @abstractmethod
    def bool(self: TensorType) -> TensorType:
        ...

    # #########################################################################
    # aliases and shared implementations
    # #########################################################################

    @final
    @property
    def T(self: TensorType) -> TensorType:
        return self.transpose()

    @final
    def abs(self: TensorType) -> TensorType:
        return self.__abs__()

    @final
    def pow(self: TensorType, exponent: TensorOrScalar) -> TensorType:
        return self.__pow__(exponent)

    @final
    def value_and_grad(
        self: TensorType, f: Callable[..., TensorType], *args: Any, **kwargs: Any
    ) -> Tuple[TensorType, TensorType]:
        return self._value_and_grad_fn(f, has_aux=False)(self, *args, **kwargs)

    @final
    def value_aux_and_grad(
        self: TensorType,
        f: Callable[..., Tuple[TensorType, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[TensorType, Any, TensorType]:
        return self._value_and_grad_fn(f, has_aux=True)(self, *args, **kwargs)

    def __iter__(self: TensorType) -> Iterator[TensorType]:
        for i in range(len(self)):
            yield self[i]

    @final
    def flatten(self: TensorType, start: int = 0, end: int = -1) -> TensorType:
        start = start % self.ndim
        end = end % self.ndim
        shape = self.shape[:start] + (-1,) + self.shape[end + 1 :]
        return self.reshape(shape)

    def __matmul__(self: TensorType, other: TensorType) -> TensorType:
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(
                f"matmul requires both tensors to be 2D, got {self.ndim}D and {other.ndim}D"
            )
        return type(self)(self.raw.__matmul__(other.raw))

    def matmul(self: TensorType, other: TensorType) -> TensorType:
        return self.__matmul__(other)

    # #########################################################################
    # extensions
    # #########################################################################

    norms = cast("NormsMethods[Tensor]", LazyCachedAccessor("_norms", "NormsMethods"))


def istensor(x: Any) -> bool:
    return isinstance(x, Tensor)
