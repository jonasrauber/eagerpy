from typing import Optional
import pytest
from numpy.testing import assert_allclose
from numpy.linalg import norm
import numpy as np
import eagerpy as ep
from eagerpy import Tensor
from eagerpy.norms import l0, l1, l2, linf, lp

norms = {0: l0, 1: l1, 2: l2, ep.inf: linf}


@pytest.fixture
def x1d(dummy: Tensor) -> Tensor:
    return ep.arange(dummy, 10).float32() / 7.0


@pytest.fixture
def x2d(dummy: Tensor) -> Tensor:
    return ep.arange(dummy, 12).float32().reshape((3, 4)) / 7.0


@pytest.fixture
def x4d(dummy: Tensor) -> Tensor:
    return ep.arange(dummy, 2 * 3 * 4 * 5).float32().reshape((2, 3, 4, 5)) / 7.0


@pytest.mark.parametrize("p", [0, 1, 2, ep.inf])
def test_1d(x1d: Tensor, p: float) -> None:
    assert_allclose(lp(x1d, p).numpy(), norm(x1d.numpy(), ord=p))
    assert_allclose(norms[p](x1d).numpy(), norm(x1d.numpy(), ord=p))


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, ep.inf])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_2d(x2d: Tensor, p: float, axis: int, keepdims: bool) -> None:
    assert isinstance(axis, int)  # see test4d for the more general test
    assert_allclose(
        lp(x2d, p, axis=axis, keepdims=keepdims).numpy(),
        norm(x2d.numpy(), ord=p, axis=axis, keepdims=keepdims),
        rtol=1e-6,
    )
    if p not in norms:
        return
    assert_allclose(
        norms[p](x2d, axis=axis, keepdims=keepdims).numpy(),
        norm(x2d.numpy(), ord=p, axis=axis, keepdims=keepdims),
        rtol=1e-6,
    )


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, ep.inf])
@pytest.mark.parametrize(
    "axis",
    [
        None,
        0,
        1,
        2,
        3,
        -1,
        -2,
        -3,
        -4,
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 2, 3),
        (0, 1, 3),
        (2, 1, 0),
    ],
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_4d(
    x4d: Tensor, p: float, axis: Optional[ep.types.AxisAxes], keepdims: bool
) -> None:
    actual = lp(x4d, p, axis=axis, keepdims=keepdims).numpy()

    # numpy does not support arbitrary axes (limited to vector and matrix norms)
    if axis is None:
        axes = tuple(range(x4d.ndim))
    elif not isinstance(axis, tuple):
        axes = (axis,)
    else:
        axes = axis
    del axis
    axes = tuple(i % x4d.ndim for i in axes)
    x = x4d.numpy()
    other = tuple(i for i in range(x.ndim) if i not in axes)
    x = np.transpose(x, other + axes)
    x = x.reshape(x.shape[: len(other)] + (-1,))
    desired = norm(x, ord=p, axis=-1)
    if keepdims:
        shape = tuple(1 if i in axes else x4d.shape[i] for i in range(x4d.ndim))
        desired = desired.reshape(shape)

    assert_allclose(actual, desired, rtol=1e-6)
