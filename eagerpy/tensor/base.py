from abc import ABC
import functools


def istensor(x):
    return isinstance(x, AbstractBaseTensor)


def wrapout(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        out = f(self, *args, **kwargs)
        return self.__class__(out)

    return wrapper


def unwrapin(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        args = [arg.tensor if isinstance(arg, self.__class__) else arg for arg in args]
        return f(self, *args, **kwargs)

    return wrapper


class AbstractBaseTensor(ABC):
    __array_ufunc__ = None

    def __init__(self, tensor):
        self.tensor = tensor


class AbstractTensor(AbstractBaseTensor, ABC):
    def __repr__(self):
        return f"{self.__class__.__name__}({self.tensor.__repr__()})"

    @wrapout
    def __getitem__(self, index):
        return self.tensor.__getitem__(index)

    @property
    def dtype(self):
        return self.tensor.dtype

    def abs(self):
        return abs(self)

    def __bool__(self):
        return self.tensor.__bool__()

    def __len__(self):
        return self.tensor.__len__()

    @wrapout
    def __abs__(self):
        return self.tensor.__abs__()

    @wrapout
    def __neg__(self):
        return self.tensor.__neg__()

    @unwrapin
    @wrapout
    def __add__(self, other):
        return self.tensor.__add__(other)

    @unwrapin
    @wrapout
    def __iadd__(self, other):
        return self.tensor.__iadd__(other)

    @unwrapin
    @wrapout
    def __sub__(self, other):
        return self.tensor.__sub__(other)

    @unwrapin
    @wrapout
    def __mul__(self, other):
        return self.tensor.__mul__(other)

    @unwrapin
    @wrapout
    def __rmul__(self, other):
        return self.tensor.__rmul__(other)

    @unwrapin
    @wrapout
    def __truediv__(self, other):
        return self.tensor.__truediv__(other)

    @unwrapin
    @wrapout
    def __rtruediv__(self, other):
        return self.tensor.__rtruediv__(other)

    @unwrapin
    @wrapout
    def __lt__(self, other):
        return self.tensor.__lt__(other)

    @unwrapin
    @wrapout
    def __le__(self, other):
        return self.tensor.__le__(other)

    @unwrapin
    @wrapout
    def __eq__(self, other):
        return self.tensor.__eq__(other)

    @unwrapin
    @wrapout
    def __ne__(self, other):
        return self.tensor.__ne__(other)

    @unwrapin
    @wrapout
    def __gt__(self, other):
        return self.tensor.__gt__(other)

    @unwrapin
    @wrapout
    def __ge__(self, other):
        return self.tensor.__ge__(other)

    @wrapout
    def __pow__(self, exponent):
        return self.tensor.__pow__(exponent)

    @wrapout
    def sign(self):
        return self.backend.sign(self.tensor)

    @wrapout
    def sqrt(self):
        return self.backend.sqrt(self.tensor)

    @wrapout
    def tanh(self):
        return self.backend.tanh(self.tensor)

    def float32(self):
        return self.astype(self.backend.float32)
