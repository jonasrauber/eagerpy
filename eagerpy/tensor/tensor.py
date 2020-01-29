from abc import ABC, abstractmethod
from typing import TypeVar


class AbstractTensor(ABC):
    __array_ufunc__ = None

    @property
    def T(self):
        return self.transpose()

    def abs(self):
        return self.__abs__()

    def pow(self, exponent):
        return self.__pow__(exponent)

    def value_and_grad(self, f, *args, **kwargs):
        return self._value_and_grad_fn(f)(self, *args, **kwargs)

    def value_aux_and_grad(self, f, *args, **kwargs):
        return self._value_and_grad_fn(f, has_aux=True)(self, *args, **kwargs)

    @abstractmethod
    def __repr__(self):
        ...

    @abstractmethod
    def __format__(self, *args, **kwargs):
        ...

    @abstractmethod
    def __getitem__(self, index):
        ...

    @property
    @abstractmethod
    def dtype(self):
        ...

    @abstractmethod
    def __bool__(self):
        ...

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __abs__(self):
        ...

    @abstractmethod
    def __neg__(self):
        ...

    @abstractmethod
    def __add__(self, other):
        ...

    @abstractmethod
    def __radd__(self, other):
        ...

    @abstractmethod
    def __sub__(self, other):
        ...

    @abstractmethod
    def __rsub__(self, other):
        ...

    @abstractmethod
    def __mul__(self, other):
        ...

    @abstractmethod
    def __rmul__(self, other):
        ...

    @abstractmethod
    def __truediv__(self, other):
        ...

    @abstractmethod
    def __rtruediv__(self, other):
        ...

    @abstractmethod
    def __floordiv__(self, other):
        ...

    @abstractmethod
    def __rfloordiv__(self, other):
        ...

    @abstractmethod
    def __mod__(self, other):
        ...

    @abstractmethod
    def __lt__(self, other):
        ...

    @abstractmethod
    def __le__(self, other):
        ...

    @abstractmethod
    def __eq__(self, other):
        ...

    @abstractmethod
    def __ne__(self, other):
        ...

    @abstractmethod
    def __gt__(self, other):
        ...

    @abstractmethod
    def __ge__(self, other):
        ...

    @abstractmethod
    def __pow__(self, exponent):
        ...

    @abstractmethod
    def sign(self):
        ...

    @abstractmethod
    def sqrt(self):
        ...

    @abstractmethod
    def tanh(self):
        ...

    @abstractmethod
    def float32(self):
        ...

    @abstractmethod
    def where(self, x, y):
        ...

    @abstractmethod
    def matmul(self, other):
        ...

    @property
    @abstractmethod
    def ndim(self):
        ...

    @abstractmethod
    def numpy(self):
        ...

    @abstractmethod
    def item(self):
        ...

    @property
    @abstractmethod
    def shape(self):
        ...

    @abstractmethod
    def reshape(self, shape):
        ...

    @abstractmethod
    def astype(self, dtype):
        ...

    @abstractmethod
    def clip(self, min_, max_):
        ...

    @abstractmethod
    def square(self):
        ...

    @abstractmethod
    def arctanh(self):
        ...

    @abstractmethod
    def sum(self, axis=None, keepdims=False):
        ...

    @abstractmethod
    def mean(self, axis=None, keepdims=False):
        ...

    @abstractmethod
    def min(self, axis=None, keepdims=False):
        ...

    @abstractmethod
    def max(self, axis=None, keepdims=False):
        ...

    @abstractmethod
    def minimum(self, other):
        ...

    @abstractmethod
    def maximum(self, other):
        ...

    @abstractmethod
    def argmin(self, axis=None):
        ...

    @abstractmethod
    def argmax(self, axis=None):
        ...

    @abstractmethod
    def argsort(self, axis=-1):
        ...

    @abstractmethod
    def uniform(self, shape, low=0.0, high=1.0):
        ...

    @abstractmethod
    def normal(self, shape, mean=0.0, stddev=1.0):
        ...

    @abstractmethod
    def ones(self, shape):
        ...

    @abstractmethod
    def zeros(self, shape):
        ...

    @abstractmethod
    def ones_like(self):
        ...

    @abstractmethod
    def zeros_like(self):
        ...

    @abstractmethod
    def full_like(self, fill_value):
        ...

    @abstractmethod
    def onehot_like(self, indices, *, value=1):
        ...

    @abstractmethod
    def from_numpy(self, a):
        ...

    @abstractmethod
    def _concatenate(self, tensors, axis=0):
        ...

    @abstractmethod
    def _stack(self, tensors, axis=0):
        ...

    @abstractmethod
    def transpose(self, axes=None):
        ...

    @abstractmethod
    def bool(self):
        ...

    @abstractmethod
    def all(self, axis=None, keepdims=False):
        ...

    @abstractmethod
    def any(self, axis=None, keepdims=False):
        ...

    @abstractmethod
    def logical_and(self, other):
        ...

    @abstractmethod
    def logical_or(self, other):
        ...

    @abstractmethod
    def logical_not(self):
        ...

    @abstractmethod
    def exp(self):
        ...

    @abstractmethod
    def log(self):
        ...

    @abstractmethod
    def log2(self):
        ...

    @abstractmethod
    def log10(self):
        ...

    @abstractmethod
    def log1p(self):
        ...

    @abstractmethod
    def tile(self, multiples):
        ...

    @abstractmethod
    def softmax(self, axis=-1):
        ...

    @abstractmethod
    def log_softmax(self, axis=-1):
        ...

    @abstractmethod
    def squeeze(self, axis=None):
        ...

    @abstractmethod
    def expand_dims(self, axis=None):
        ...

    @abstractmethod
    def full(self, shape, value):
        ...

    @abstractmethod
    def index_update(self, indices, values):
        ...

    @abstractmethod
    def arange(self, *args, **kwargs):
        ...

    @abstractmethod
    def cumsum(self, axis=None):
        ...

    @abstractmethod
    def flip(self, axis=None):
        ...

    @abstractmethod
    def meshgrid(self, *tensors, indexing="xy"):
        ...

    @abstractmethod
    def pad(self, paddings, mode="constant", value=0):
        ...

    @abstractmethod
    def isnan(self):
        ...

    @abstractmethod
    def isinf(self):
        ...

    @abstractmethod
    def crossentropy(self, labels):
        ...

    @abstractmethod
    def _value_and_grad_fn(self, f, has_aux=False):
        ...


Tensor = TypeVar("Tensor", bound=AbstractTensor)


def istensor(x):
    return isinstance(x, AbstractTensor)
