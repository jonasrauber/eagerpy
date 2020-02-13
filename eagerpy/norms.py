from typing import Union, Optional

from .tensor import TensorType
from .types import AxisAxes
from .framework import inf


def l0(
    x: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return (x != 0).sum(axis=axis, keepdims=keepdims).astype(x.dtype)


def l1(
    x: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return x.abs().sum(axis=axis, keepdims=keepdims)


def l2(
    x: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return x.square().sum(axis=axis, keepdims=keepdims).sqrt()


def linf(
    x: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return x.abs().max(axis=axis, keepdims=keepdims)


def lp(
    x: TensorType,
    p: Union[int, float],
    axis: Optional[AxisAxes] = None,
    keepdims: bool = False,
) -> TensorType:
    if p == 0:
        return l0(x, axis=axis, keepdims=keepdims)
    if p == 1:
        return l1(x, axis=axis, keepdims=keepdims)
    if p == 2:
        return l2(x, axis=axis, keepdims=keepdims)
    if p == inf:
        return linf(x, axis=axis, keepdims=keepdims)
    return x.abs().pow(p).sum(axis=axis, keepdims=keepdims).pow(1.0 / p)
