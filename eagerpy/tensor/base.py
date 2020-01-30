import functools
from typing_extensions import final
from typing import Any

from .tensor import AbstractTensor
from .tensor import istensor
from .tensor import Tensor


def wrapout(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        out = f(self, *args, **kwargs)
        return self.__class__(out)

    return wrapper


def unwrapin(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        args = [arg.raw if istensor(arg) else arg for arg in args]
        return f(self, *args, **kwargs)

    return wrapper


def unwrap_(*args):
    """Unwraps all EagerPy tensors if they are not already unwrapped"""
    result = tuple(t.raw if istensor(t) else t for t in args)
    return result[0] if len(args) == 1 else result


class AbstractBaseTensor(AbstractTensor):
    def __init__(self, raw):
        self._raw = raw

    @final
    @property
    def raw(self) -> Any:
        return self._raw

    @final
    def __repr__(self: Tensor) -> str:
        lines = self.raw.__repr__().split("\n")
        prefix = self.__class__.__name__ + "("
        lines[0] = prefix + lines[0]
        prefix = " " * len(prefix)
        for i in range(1, len(lines)):
            lines[i] = prefix + lines[i]
        lines[-1] = lines[-1] + ")"
        return "\n".join(lines)

    def __format__(self, *args, **kwargs):
        return self.raw.__format__(*args, **kwargs)

    @unwrapin
    @wrapout
    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = tuple(x.raw if istensor(x) else x for x in index)
        return self.raw.__getitem__(index)

    @property
    def dtype(self):
        return self.raw.dtype

    def __bool__(self):
        return self.raw.__bool__()

    def __len__(self):
        return self.raw.__len__()

    def __abs__(self: Tensor) -> Tensor:
        return type(self)(self.raw.__abs__())

    @wrapout
    def __neg__(self):
        return self.raw.__neg__()

    @unwrapin
    @wrapout
    def __add__(self, other):
        return self.raw.__add__(other)

    @unwrapin
    @wrapout
    def __radd__(self, other):
        return self.raw.__radd__(other)

    @unwrapin
    @wrapout
    def __sub__(self, other):
        return self.raw.__sub__(other)

    @unwrapin
    @wrapout
    def __rsub__(self, other):
        return self.raw.__rsub__(other)

    @unwrapin
    @wrapout
    def __mul__(self, other):
        return self.raw.__mul__(other)

    @unwrapin
    @wrapout
    def __rmul__(self, other):
        return self.raw.__rmul__(other)

    @unwrapin
    @wrapout
    def __truediv__(self, other):
        return self.raw.__truediv__(other)

    @unwrapin
    @wrapout
    def __rtruediv__(self, other):
        return self.raw.__rtruediv__(other)

    @unwrapin
    @wrapout
    def __floordiv__(self, other):
        return self.raw.__floordiv__(other)

    @unwrapin
    @wrapout
    def __rfloordiv__(self, other):
        return self.raw.__rfloordiv__(other)

    @unwrapin
    @wrapout
    def __mod__(self, other):
        return self.raw.__mod__(other)

    @unwrapin
    @wrapout
    def __lt__(self, other):
        return self.raw.__lt__(other)

    @unwrapin
    @wrapout
    def __le__(self, other):
        return self.raw.__le__(other)

    @unwrapin
    @wrapout
    def __eq__(self, other):
        return self.raw.__eq__(other)

    @unwrapin
    @wrapout
    def __ne__(self, other):
        return self.raw.__ne__(other)

    @unwrapin
    @wrapout
    def __gt__(self, other):
        return self.raw.__gt__(other)

    @unwrapin
    @wrapout
    def __ge__(self, other):
        return self.raw.__ge__(other)

    def __pow__(self: Tensor, exponent) -> Tensor:
        return type(self)(self.raw.__pow__(exponent))

    @property
    def ndim(self):
        return self.raw.ndim
