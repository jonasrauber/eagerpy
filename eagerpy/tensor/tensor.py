from abc import ABC, abstractmethod
from typing import TypeVar, Callable, Tuple, Any, overload, Sequence, Union
from typing_extensions import Literal, final


Tensor = TypeVar("Tensor", bound="AbstractTensor")


class AbstractTensor(ABC):
    """Base class defining the common interface of all EagerPy Tensors"""

    __array_ufunc__ = None

    @abstractmethod
    def __init__(self, raw):
        ...

    @property
    @abstractmethod
    def raw(self) -> Any:
        ...

    @property
    @abstractmethod
    def dtype(self: Tensor) -> Any:
        ...

    @abstractmethod
    def __repr__(self: Tensor) -> str:
        ...

    @abstractmethod
    def __format__(self: Tensor, format_spec) -> str:
        ...

    @abstractmethod
    def __getitem__(self: Tensor, index) -> Tensor:
        ...

    @abstractmethod
    def __bool__(self: Tensor) -> bool:
        ...

    @abstractmethod
    def __len__(self: Tensor) -> int:
        ...

    @abstractmethod
    def __abs__(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def __neg__(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def __add__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __radd__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __sub__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __rsub__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __mul__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __rmul__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __truediv__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __rtruediv__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __floordiv__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __rfloordiv__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __mod__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __lt__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __le__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __eq__(self: Tensor, other) -> Tensor:  # type: ignore
        # we ignore the type errors caused by wrong type annotations for object
        # https://github.com/python/typeshed/issues/3685
        ...

    @abstractmethod
    def __ne__(self: Tensor, other) -> Tensor:  # type: ignore
        # we ignore the type errors caused by wrong type annotations for object
        # https://github.com/python/typeshed/issues/3685
        ...

    @abstractmethod
    def __gt__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __ge__(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def __pow__(self: Tensor, exponent) -> Tensor:
        ...

    @abstractmethod
    def sign(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def sqrt(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def tanh(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def float32(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def where(self: Tensor, x, y) -> Tensor:
        ...

    @abstractmethod
    def matmul(self: Tensor, other) -> Tensor:
        ...

    @property
    @abstractmethod
    def ndim(self: Tensor) -> int:
        ...

    @abstractmethod
    def numpy(self: Tensor) -> Any:
        ...

    @abstractmethod
    def item(self: Tensor) -> Union[int, float, bool]:
        ...

    @property
    @abstractmethod
    def shape(self: Tensor) -> Tuple:
        ...

    @abstractmethod
    def reshape(self: Tensor, shape) -> Tensor:
        ...

    @abstractmethod
    def astype(self: Tensor, dtype) -> Tensor:
        ...

    @abstractmethod
    def clip(self: Tensor, min_, max_) -> Tensor:
        ...

    @abstractmethod
    def square(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def arctanh(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def sum(self: Tensor, axis=None, keepdims=False) -> Tensor:
        ...

    @abstractmethod
    def mean(self: Tensor, axis=None, keepdims=False) -> Tensor:
        ...

    @abstractmethod
    def min(self: Tensor, axis=None, keepdims=False) -> Tensor:
        ...

    @abstractmethod
    def max(self: Tensor, axis=None, keepdims=False) -> Tensor:
        ...

    @abstractmethod
    def minimum(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def maximum(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def argmin(self: Tensor, axis=None) -> Tensor:
        ...

    @abstractmethod
    def argmax(self: Tensor, axis=None) -> Tensor:
        ...

    @abstractmethod
    def argsort(self: Tensor, axis=-1) -> Tensor:
        ...

    @abstractmethod
    def uniform(self: Tensor, shape, low=0.0, high=1.0) -> Tensor:
        ...

    @abstractmethod
    def normal(self: Tensor, shape, mean=0.0, stddev=1.0) -> Tensor:
        ...

    @abstractmethod
    def ones(self: Tensor, shape) -> Tensor:
        ...

    @abstractmethod
    def zeros(self: Tensor, shape) -> Tensor:
        ...

    @abstractmethod
    def ones_like(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def zeros_like(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def full_like(self: Tensor, fill_value) -> Tensor:
        ...

    @abstractmethod
    def onehot_like(self: Tensor, indices, *, value=1) -> Tensor:
        ...

    @abstractmethod
    def from_numpy(self: Tensor, a) -> Tensor:
        ...

    @abstractmethod
    def _concatenate(self: Tensor, tensors: Sequence[Tensor], axis=0) -> Tensor:
        ...

    @abstractmethod
    def _stack(self: Tensor, tensors: Sequence[Tensor], axis=0) -> Tensor:
        ...

    @abstractmethod
    def transpose(self: Tensor, axes=None) -> Tensor:
        ...

    @abstractmethod
    def bool(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def all(self: Tensor, axis=None, keepdims=False) -> Tensor:
        ...

    @abstractmethod
    def any(self: Tensor, axis=None, keepdims=False) -> Tensor:
        ...

    @abstractmethod
    def logical_and(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def logical_or(self: Tensor, other) -> Tensor:
        ...

    @abstractmethod
    def logical_not(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def exp(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def log(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def log2(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def log10(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def log1p(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def tile(self: Tensor, multiples) -> Tensor:
        ...

    @abstractmethod
    def softmax(self: Tensor, axis=-1) -> Tensor:
        ...

    @abstractmethod
    def log_softmax(self: Tensor, axis=-1) -> Tensor:
        ...

    @abstractmethod
    def squeeze(self: Tensor, axis=None) -> Tensor:
        ...

    @abstractmethod
    def expand_dims(self: Tensor, axis=None) -> Tensor:
        ...

    @abstractmethod
    def full(self: Tensor, shape, value) -> Tensor:
        ...

    @abstractmethod
    def index_update(self: Tensor, indices, values) -> Tensor:
        ...

    @abstractmethod
    def arange(self: Tensor, start, stop=None, step=None) -> Tensor:
        ...

    @abstractmethod
    def cumsum(self: Tensor, axis=None) -> Tensor:
        ...

    @abstractmethod
    def flip(self: Tensor, axis=None) -> Tensor:
        ...

    @abstractmethod
    def meshgrid(self: Tensor, *tensors, indexing="xy") -> Tuple[Tensor, ...]:
        ...

    @abstractmethod
    def pad(self: Tensor, paddings, mode="constant", value=0) -> Tensor:
        ...

    @abstractmethod
    def isnan(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def isinf(self: Tensor) -> Tensor:
        ...

    @abstractmethod
    def crossentropy(self: Tensor, labels: Tensor) -> Tensor:
        ...

    @overload
    def _value_and_grad_fn(
        self: Tensor, f: Callable
    ) -> Callable[..., Tuple[Tensor, Tensor]]:
        ...

    @overload  # noqa: F811 (waiting for pyflakes > 2.1.1)
    def _value_and_grad_fn(
        self: Tensor, f: Callable, has_aux: Literal[False]
    ) -> Callable[..., Tuple[Tensor, Tensor]]:
        ...

    @overload  # noqa: F811 (waiting for pyflakes > 2.1.1)
    def _value_and_grad_fn(
        self: Tensor, f: Callable, has_aux: Literal[True]
    ) -> Callable[..., Tuple[Tensor, Any, Tensor]]:
        ...

    @abstractmethod  # noqa: F811 (waiting for pyflakes > 2.1.1)
    def _value_and_grad_fn(self, f, has_aux=False):
        ...

    # #########################################################################
    # aliases
    # #########################################################################

    @final
    @property
    def T(self: Tensor) -> Tensor:
        return self.transpose()

    @final
    def abs(self: Tensor) -> Tensor:
        return self.__abs__()

    @final
    def pow(self: Tensor, exponent) -> Tensor:
        return self.__pow__(exponent)

    @final
    def value_and_grad(self: Tensor, f, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        return self._value_and_grad_fn(f, has_aux=False)(self, *args, **kwargs)

    @final
    def value_aux_and_grad(
        self: Tensor, f, *args, **kwargs
    ) -> Tuple[Tensor, Any, Tensor]:
        return self._value_and_grad_fn(f, has_aux=True)(self, *args, **kwargs)


def istensor(x) -> bool:
    return isinstance(x, AbstractTensor)
