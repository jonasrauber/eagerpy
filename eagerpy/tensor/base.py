from abc import ABC
import functools


def wrapout(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        out = f(self, *args, **kwargs)
        return self.__class__(out)

    return wrapper


class AbstractTensor(ABC):
    def __init__(self, tensor):
        self.tensor = tensor

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

    @wrapout
    def __abs__(self):
        return self.tensor.__abs__()

    @wrapout
    def __neg__(self):
        return self.tensor.__neg__()

    @wrapout
    def __add__(self, other):
        return self.tensor.__add__(other.tensor)

    @wrapout
    def __iadd__(self, other):
        return self.tensor.__iadd__(other.tensor)

    @wrapout
    def __sub__(self, other):
        return self.tensor.__sub__(other.tensor)

    @wrapout
    def __mul__(self, other):
        return self.tensor.__mul__(other.tensor)

    @wrapout
    def __rmul__(self, other):
        if hasattr(other, "tensor"):
            other = other.tensor
        return self.tensor.__rmul__(other)

    @wrapout
    def __truediv__(self, other):
        return self.tensor.__truediv__(other.tensor)

    @wrapout
    def __rtruediv__(self, other):
        if hasattr(other, "tensor"):
            other = other.tensor
        return self.tensor.__rtruediv__(other)

    @wrapout
    def __lt__(self, other):
        return self.tensor.__lt__(other.tensor)

    @wrapout
    def __le__(self, other):
        return self.tensor.__le__(other.tensor)

    @wrapout
    def __eq__(self, other):
        return self.tensor.__eq__(other.tensor)

    @wrapout
    def __ne__(self, other):
        return self.tensor.__ne__(other.tensor)

    @wrapout
    def __gt__(self, other):
        return self.tensor.__gt__(other.tensor)

    @wrapout
    def __ge__(self, other):
        return self.tensor.__ge__(other.tensor)

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
