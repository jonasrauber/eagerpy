from abc import ABC, abstractmethod
from typing import TypeVar, Callable, Tuple, Any, overload, Sequence, Union
from typing_extensions import Literal, final


TensorType = TypeVar("TensorType", bound="Tensor")


class Tensor(ABC):
    """Base class defining the common interface of all EagerPy Tensors"""

    __array_ufunc__ = None

    # shorten the class name to eagerpy.Tensor (does not help with MyPy)
    __module__ = "eagerpy"

    @abstractmethod
    def __init__(self, raw) -> None:
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
    def __format__(self: TensorType, format_spec) -> str:
        ...

    @abstractmethod
    def __getitem__(self: TensorType, index) -> TensorType:
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
    def __add__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __radd__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __sub__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __rsub__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __mul__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __rmul__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __truediv__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __rtruediv__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __floordiv__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __rfloordiv__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __mod__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __lt__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __le__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __eq__(self: TensorType, other) -> TensorType:  # type: ignore
        # we ignore the type errors caused by wrong type annotations for object
        # https://github.com/python/typeshed/issues/3685
        ...

    @abstractmethod
    def __ne__(self: TensorType, other) -> TensorType:  # type: ignore
        # we ignore the type errors caused by wrong type annotations for object
        # https://github.com/python/typeshed/issues/3685
        ...

    @abstractmethod
    def __gt__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __ge__(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def __pow__(self: TensorType, exponent) -> TensorType:
        ...

    @abstractmethod
    def sign(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def sqrt(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def tanh(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def float32(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def where(self: TensorType, x, y) -> TensorType:
        ...

    @abstractmethod
    def matmul(self: TensorType, other) -> TensorType:
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
    def shape(self: TensorType) -> Tuple:
        ...

    @abstractmethod
    def reshape(self: TensorType, shape) -> TensorType:
        ...

    @abstractmethod
    def astype(self: TensorType, dtype) -> TensorType:
        ...

    @abstractmethod
    def clip(self: TensorType, min_, max_) -> TensorType:
        ...

    @abstractmethod
    def square(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def arctanh(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def sum(self: TensorType, axis=None, keepdims=False) -> TensorType:
        ...

    @abstractmethod
    def mean(self: TensorType, axis=None, keepdims=False) -> TensorType:
        ...

    @abstractmethod
    def min(self: TensorType, axis=None, keepdims=False) -> TensorType:
        ...

    @abstractmethod
    def max(self: TensorType, axis=None, keepdims=False) -> TensorType:
        ...

    @abstractmethod
    def minimum(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def maximum(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def argmin(self: TensorType, axis=None) -> TensorType:
        ...

    @abstractmethod
    def argmax(self: TensorType, axis=None) -> TensorType:
        ...

    @abstractmethod
    def argsort(self: TensorType, axis=-1) -> TensorType:
        ...

    @abstractmethod
    def uniform(self: TensorType, shape, low=0.0, high=1.0) -> TensorType:
        ...

    @abstractmethod
    def normal(self: TensorType, shape, mean=0.0, stddev=1.0) -> TensorType:
        ...

    @abstractmethod
    def ones(self: TensorType, shape) -> TensorType:
        ...

    @abstractmethod
    def zeros(self: TensorType, shape) -> TensorType:
        ...

    @abstractmethod
    def ones_like(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def zeros_like(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def full_like(self: TensorType, fill_value) -> TensorType:
        ...

    @abstractmethod
    def onehot_like(self: TensorType, indices, *, value=1) -> TensorType:
        ...

    @abstractmethod
    def from_numpy(self: TensorType, a) -> TensorType:
        ...

    @abstractmethod
    def _concatenate(
        self: TensorType, tensors: Sequence[TensorType], axis=0
    ) -> TensorType:
        ...

    @abstractmethod
    def _stack(self: TensorType, tensors: Sequence[TensorType], axis=0) -> TensorType:
        ...

    @abstractmethod
    def transpose(self: TensorType, axes=None) -> TensorType:
        ...

    @abstractmethod
    def bool(self: TensorType) -> TensorType:
        ...

    @abstractmethod
    def all(self: TensorType, axis=None, keepdims=False) -> TensorType:
        ...

    @abstractmethod
    def any(self: TensorType, axis=None, keepdims=False) -> TensorType:
        ...

    @abstractmethod
    def logical_and(self: TensorType, other) -> TensorType:
        ...

    @abstractmethod
    def logical_or(self: TensorType, other) -> TensorType:
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
    def tile(self: TensorType, multiples) -> TensorType:
        ...

    @abstractmethod
    def softmax(self: TensorType, axis=-1) -> TensorType:
        ...

    @abstractmethod
    def log_softmax(self: TensorType, axis=-1) -> TensorType:
        ...

    @abstractmethod
    def squeeze(self: TensorType, axis=None) -> TensorType:
        ...

    @abstractmethod
    def expand_dims(self: TensorType, axis=None) -> TensorType:
        ...

    @abstractmethod
    def full(self: TensorType, shape, value) -> TensorType:
        ...

    @abstractmethod
    def index_update(self: TensorType, indices, values) -> TensorType:
        ...

    @abstractmethod
    def arange(self: TensorType, start, stop=None, step=None) -> TensorType:
        ...

    @abstractmethod
    def cumsum(self: TensorType, axis=None) -> TensorType:
        ...

    @abstractmethod
    def flip(self: TensorType, axis=None) -> TensorType:
        ...

    @abstractmethod
    def meshgrid(self: TensorType, *tensors, indexing="xy") -> Tuple[TensorType, ...]:
        ...

    @abstractmethod
    def pad(self: TensorType, paddings, mode="constant", value=0) -> TensorType:
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

    @overload
    def _value_and_grad_fn(
        self: TensorType, f: Callable
    ) -> Callable[..., Tuple[TensorType, TensorType]]:
        ...

    @overload  # noqa: F811 (waiting for pyflakes > 2.1.1)
    def _value_and_grad_fn(
        self: TensorType, f: Callable, has_aux: Literal[False]
    ) -> Callable[..., Tuple[TensorType, TensorType]]:
        ...

    @overload  # noqa: F811 (waiting for pyflakes > 2.1.1)
    def _value_and_grad_fn(
        self: TensorType, f: Callable, has_aux: Literal[True]
    ) -> Callable[..., Tuple[TensorType, Any, TensorType]]:
        ...

    @abstractmethod  # noqa: F811 (waiting for pyflakes > 2.1.1)
    def _value_and_grad_fn(self, f, has_aux=False):
        ...

    # #########################################################################
    # aliases
    # #########################################################################

    @final
    @property
    def T(self: TensorType) -> TensorType:
        return self.transpose()

    @final
    def abs(self: TensorType) -> TensorType:
        return self.__abs__()

    @final
    def pow(self: TensorType, exponent) -> TensorType:
        return self.__pow__(exponent)

    @final
    def value_and_grad(
        self: TensorType, f, *args, **kwargs
    ) -> Tuple[TensorType, TensorType]:
        return self._value_and_grad_fn(f, has_aux=False)(self, *args, **kwargs)

    @final
    def value_aux_and_grad(
        self: TensorType, f, *args, **kwargs
    ) -> Tuple[TensorType, Any, TensorType]:
        return self._value_and_grad_fn(f, has_aux=True)(self, *args, **kwargs)


def istensor(x: Any) -> bool:
    return isinstance(x, Tensor)
