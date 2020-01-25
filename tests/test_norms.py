import pytest
from numpy.testing import assert_allclose
from numpy.linalg import norm
import eagerpy as ep
from eagerpy.norms import l0, l1, l2, linf, lp


norms = {0: l0, 1: l1, 2: l2, ep.inf: linf}


@pytest.fixture
def x1d():
    return ep.numpy.arange(10).float32() / 7.0


@pytest.fixture
def x2d():
    return ep.numpy.arange(12).float32().reshape((3, 4)) / 7.0


@pytest.mark.parametrize("p", [0, 1, 2, ep.inf])
def test_1d(x1d, p):
    assert_allclose(lp(x1d, p).numpy(), norm(x1d.numpy(), ord=p))
    assert_allclose(norms[p](x1d).numpy(), norm(x1d.numpy(), ord=p))


@pytest.mark.parametrize("p", [0, 1, 2, ep.inf])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_2d(x2d, p, axis, keepdims):
    assert_allclose(
        lp(x2d, p, axis=axis, keepdims=keepdims).numpy(),
        norm(x2d.numpy(), ord=p, axis=axis, keepdims=keepdims),
    )
    assert_allclose(
        norms[p](x2d, axis=axis, keepdims=keepdims).numpy(),
        norm(x2d.numpy(), ord=p, axis=axis, keepdims=keepdims),
    )
