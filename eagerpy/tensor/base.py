from typing_extensions import final
from typing import Any, cast

from .tensor import Tensor
from .tensor import istensor
from .tensor import TensorType


def unwrap_(*args) -> Any:
    """Unwraps all EagerPy tensors if they are not already unwrapped"""
    always_tuple = False
    if len(args) == 1 and isinstance(args[0], tuple) or isinstance(args[0], list):
        (args,) = args
        always_tuple = True
    result = tuple(t.raw if istensor(t) else t for t in args)
    return result[0] if len(args) == 1 and not always_tuple else result


class BaseTensor(Tensor):
    def __init__(self: TensorType, raw) -> None:
        self._raw = raw

    @property
    def raw(self) -> Any:
        return self._raw

    @final
    def __repr__(self: TensorType) -> str:
        lines = repr(self.raw).split("\n")
        prefix = self.__class__.__name__ + "("
        lines[0] = prefix + lines[0]
        prefix = " " * len(prefix)
        for i in range(1, len(lines)):
            lines[i] = prefix + lines[i]
        lines[-1] = lines[-1] + ")"
        return "\n".join(lines)

    @final
    def __format__(self: TensorType, format_spec) -> str:
        return format(self.raw, format_spec)

    @final
    @property
    def dtype(self: TensorType) -> Any:
        return self.raw.dtype

    @final
    def __bool__(self: TensorType) -> bool:
        return bool(self.raw)

    @final
    def __len__(self: TensorType) -> int:
        return len(self.raw)

    @final
    def __abs__(self: TensorType) -> TensorType:
        return type(self)(abs(self.raw))

    @final
    def __neg__(self: TensorType) -> TensorType:
        return type(self)(-self.raw)

    @final
    def __add__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__add__(unwrap_(other)))

    @final
    def __radd__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__radd__(unwrap_(other)))

    @final
    def __sub__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__sub__(unwrap_(other)))

    @final
    def __rsub__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__rsub__(unwrap_(other)))

    @final
    def __mul__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__mul__(unwrap_(other)))

    @final
    def __rmul__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__rmul__(unwrap_(other)))

    @final
    def __truediv__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__truediv__(unwrap_(other)))

    @final
    def __rtruediv__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__rtruediv__(unwrap_(other)))

    @final
    def __floordiv__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__floordiv__(unwrap_(other)))

    @final
    def __rfloordiv__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__rfloordiv__(unwrap_(other)))

    @final
    def __mod__(self: TensorType, other) -> TensorType:
        return type(self)(self.raw.__mod__(unwrap_(other)))

    @final
    def __pow__(self: TensorType, exponent) -> TensorType:
        return type(self)(self.raw.__pow__(exponent))

    @final
    @property
    def ndim(self: TensorType) -> int:
        return cast(int, self.raw.ndim)
