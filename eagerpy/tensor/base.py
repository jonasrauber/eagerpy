from typing_extensions import final
from typing import Any, Type, Union, cast, Tuple

from .tensor import Tensor
from .tensor import TensorType
from .tensor import TensorOrScalar


def unwrap_(*args: Any) -> Any:
    return tuple(t.raw if isinstance(t, Tensor) else t for t in args)


def unwrap1(t: Any) -> Any:
    return t.raw if isinstance(t, Tensor) else t


class BaseTensor(Tensor):
    __slots__ = "_raw"

    _registered = False

    def __new__(cls: Type["BaseTensor"], *args: Any, **kwargs: Any) -> "BaseTensor":
        if not cls._registered:
            import jax

            def flatten(t: Tensor) -> Tuple[Any, None]:
                return ((t.raw,), None)

            def unflatten(aux_data: None, children: Tuple) -> Union[Tensor, Any]:
                assert len(children) == 1
                x = children[0]
                del children

                if isinstance(x, tuple):
                    x, unwrap = x
                    if unwrap:
                        return x

                if isinstance(x, Tensor):
                    return x
                return cls(x)

            jax.tree_util.register_pytree_node(cls, flatten, unflatten)
            cls._registered = True
        return cast("BaseTensor", super().__new__(cls))

    def __init__(self: TensorType, raw: Any):
        assert not isinstance(raw, Tensor)
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
    def __format__(self: TensorType, format_spec: str) -> str:
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
        return cast(int, self.raw.shape[0])

    @final
    def __abs__(self: TensorType) -> TensorType:
        return type(self)(abs(self.raw))

    @final
    def __neg__(self: TensorType) -> TensorType:
        return type(self)(-self.raw)

    @final
    def __add__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__add__(unwrap1(other)))

    @final
    def __radd__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__radd__(unwrap1(other)))

    @final
    def __sub__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__sub__(unwrap1(other)))

    @final
    def __rsub__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__rsub__(unwrap1(other)))

    @final
    def __mul__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__mul__(unwrap1(other)))

    @final
    def __rmul__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__rmul__(unwrap1(other)))

    @final
    def __truediv__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__truediv__(unwrap1(other)))

    @final
    def __rtruediv__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__rtruediv__(unwrap1(other)))

    @final
    def __floordiv__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__floordiv__(unwrap1(other)))

    @final
    def __rfloordiv__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__rfloordiv__(unwrap1(other)))

    @final
    def __mod__(self: TensorType, other: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__mod__(unwrap1(other)))

    @final
    def __pow__(self: TensorType, exponent: TensorOrScalar) -> TensorType:
        return type(self)(self.raw.__pow__(unwrap1(exponent)))

    @final
    @property
    def ndim(self: TensorType) -> int:
        return len(self.raw.shape)
