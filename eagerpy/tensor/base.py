from typing_extensions import final
from typing import Any, cast

from .tensor import AbstractTensor
from .tensor import istensor
from .tensor import Tensor


def unwrap_(*args) -> Any:
    """Unwraps all EagerPy tensors if they are not already unwrapped"""
    always_tuple = False
    if len(args) == 1 and isinstance(args[0], tuple) or isinstance(args[0], list):
        (args,) = args
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
    def __pow__(self: Tensor, exponent) -> Tensor:
        return type(self)(self.raw.__pow__(exponent))

    @final
    @property
    def ndim(self: Tensor) -> int:
        return cast(int, self.raw.ndim)
