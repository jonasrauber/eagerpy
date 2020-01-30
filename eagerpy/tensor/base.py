import functools
from typing_extensions import final
from typing import Any, cast

from .tensor import AbstractTensor
from .tensor import istensor
from .tensor import Tensor

from typing import TypeVar, Callable

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def wrapout(f: F) -> F:
    @functools.wraps(f)
    def wrapper(self: Tensor, *args, **kwargs) -> Tensor:
        out = f(self, *args, **kwargs)
        return self.__class__(out)

    return cast(F, wrapper)


def unwrapin(f: F) -> F:
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs) -> Any:
        args = tuple(arg.raw if istensor(arg) else arg for arg in args)
        return f(self, *args, **kwargs)

    return cast(F, wrapper)


def unwrap_(*args) -> Any:
    """Unwraps all EagerPy tensors if they are not already unwrapped"""
    always_tuple = False
    if len(args) == 1 and isinstance(args[0], tuple) or isinstance(args[0], list):
        args, = args
        always_tuple = True
    result = tuple(t.raw if istensor(t) else t for t in args)
    return result[0] if len(args) == 1 and not always_tuple else result


class AbstractBaseTensor(AbstractTensor):
    def __init__(self: Tensor, raw):
        self._raw = raw

    @property
    def raw(self) -> Any:
        return self._raw

    @final
    def __repr__(self: Tensor) -> str:
        lines = repr(self.raw).split("\n")
        prefix = self.__class__.__name__ + "("
        lines[0] = prefix + lines[0]
        prefix = " " * len(prefix)
        for i in range(1, len(lines)):
            lines[i] = prefix + lines[i]
        lines[-1] = lines[-1] + ")"
        return "\n".join(lines)

    @final
    def __format__(self: Tensor, format_spec) -> str:
        return format(self.raw, format_spec)

    @final
    def __getitem__(self: Tensor, index) -> Tensor:
        if isinstance(index, tuple):
            index = tuple(x.raw if istensor(x) else x for x in index)
        elif istensor(index):
            index = index.raw
        return type(self)(self.raw[index])

    @final
    @property
    def dtype(self: Tensor) -> Any:
        return self.raw.dtype

    @final
    def __bool__(self: Tensor) -> bool:
        return bool(self.raw)

    @final
    def __len__(self: Tensor) -> int:
        return len(self.raw)

    @final
    def __abs__(self: Tensor) -> Tensor:
        return type(self)(abs(self.raw))

    @final
    def __neg__(self: Tensor) -> Tensor:
        return type(self)(-self.raw)

    @final
    def __add__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__add__(unwrap_(other)))

    @final
    def __radd__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__radd__(unwrap_(other)))

    @final
    def __sub__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__sub__(unwrap_(other)))

    @final
    def __rsub__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__rsub__(unwrap_(other)))

    @final
    def __mul__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__mul__(unwrap_(other)))

    @final
    def __rmul__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__rmul__(unwrap_(other)))

    @final
    def __truediv__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__truediv__(unwrap_(other)))

    @final
    def __rtruediv__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__rtruediv__(unwrap_(other)))

    @final
    def __floordiv__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__floordiv__(unwrap_(other)))

    @final
    def __rfloordiv__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__rfloordiv__(unwrap_(other)))

    @final
    def __mod__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__mod__(unwrap_(other)))

    @final
    def __lt__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__lt__(unwrap_(other)))

    @final
    def __le__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__le__(unwrap_(other)))

    @final
    def __eq__(self: Tensor, other) -> Tensor:  # type: ignore
        return type(self)(self.raw.__eq__(unwrap_(other)))

    @final
    def __ne__(self: Tensor, other) -> Tensor:  # type: ignore
        return type(self)(self.raw.__ne__(unwrap_(other)))

    @final
    def __gt__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__gt__(unwrap_(other)))

    @final
    def __ge__(self: Tensor, other) -> Tensor:
        return type(self)(self.raw.__ge__(unwrap_(other)))

    @final
    def __pow__(self: Tensor, exponent) -> Tensor:
        return type(self)(self.raw.__pow__(exponent))

    @final
    @property
    def ndim(self: Tensor) -> int:
        return cast(int, self.raw.ndim)
