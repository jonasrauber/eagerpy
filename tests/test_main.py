from typing import Callable, Dict, Any, Tuple, Union, Optional, cast
import pytest
import functools
import itertools
import numpy as np
import eagerpy as ep
from eagerpy import Tensor
from eagerpy.types import Shape, AxisAxes

# make sure there are no undecorated tests in the "special tests" section below
# -> /\n\ndef test_
# make sure the undecorated tests in the "normal tests" section all contain
# assertions and do not return something
# -> /\n    return


###############################################################################
# normal tests
# - no decorator
# - assertions
###############################################################################


def test_astensor_raw(t: Tensor) -> None:
    assert (ep.astensor(t.raw) == t).all()


def test_astensor_tensor(t: Tensor) -> None:
    assert (ep.astensor(t) == t).all()


def test_astensor_restore_raw(t: Tensor) -> None:
    r = t.raw
    y, restore_type = ep.astensor_(r)
    assert (y == t).all()
    assert type(restore_type(y)) == type(r)
    y = y + 1
    assert type(restore_type(y)) == type(r)


def test_astensor_restore_tensor(t: Tensor) -> None:
    r = t
    y, restore_type = ep.astensor_(r)
    assert (y == t).all()
    assert type(restore_type(y)) == type(r)
    y = y + 1
    assert type(restore_type(y)) == type(r)


def test_astensors_raw(t: Tensor) -> None:
    ts = (t, t + 1, t + 2)
    rs = tuple(t.raw for t in ts)
    ys = ep.astensors(*rs)
    assert isinstance(ys, tuple)
    assert len(ts) == len(ys)
    for ti, yi in zip(ts, ys):
        assert (ti == yi).all()


def test_astensors_tensor(t: Tensor) -> None:
    ts = (t, t + 1, t + 2)
    ys = ep.astensors(*ts)
    assert isinstance(ys, tuple)
    assert len(ts) == len(ys)
    for ti, yi in zip(ts, ys):
        assert (ti == yi).all()


def test_astensors_raw_restore(t: Tensor) -> None:
    ts = (t, t + 1, t + 2)
    rs = tuple(t.raw for t in ts)
    ys, restore_type = ep.astensors_(*rs)
    assert isinstance(ys, tuple)
    assert len(ts) == len(ys)
    for ti, yi in zip(ts, ys):
        assert (ti == yi).all()

    ys = tuple(y + 1 for y in ys)
    xs = restore_type(*ys)
    assert isinstance(xs, tuple)
    assert len(xs) == len(ys)
    for xi, ri in zip(xs, rs):
        assert type(xi) == type(ri)

    x0 = restore_type(ys[0])
    assert not isinstance(x0, tuple)


def test_astensors_tensors_restore(t: Tensor) -> None:
    ts = (t, t + 1, t + 2)
    rs = ts
    ys, restore_type = ep.astensors_(*rs)
    assert isinstance(ys, tuple)
    assert len(ts) == len(ys)
    for ti, yi in zip(ts, ys):
        assert (ti == yi).all()

    ys = tuple(y + 1 for y in ys)
    xs = restore_type(*ys)
    assert isinstance(xs, tuple)
    assert len(xs) == len(ys)
    for xi, ri in zip(xs, rs):
        assert type(xi) == type(ri)

    x0 = restore_type(ys[0])
    assert not isinstance(x0, tuple)  # type: ignore


def test_module() -> None:
    assert ep.istensor(ep.numpy.tanh([3, 5]))
    assert not ep.istensor(ep.numpy.tanh(3))


def test_module_dir() -> None:
    assert "zeros" in dir(ep.numpy)


def test_repr(t: Tensor) -> None:
    assert not repr(t).startswith("<")
    t = ep.zeros(t, (10, 10))
    assert not repr(t).startswith("<")
    assert len(repr(t).split("\n")) > 1


def test_logical_or_manual(t: Tensor) -> None:
    assert (ep.logical_or(t < 3, ep.zeros_like(t).bool()) == (t < 3)).all()


def test_logical_not_manual(t: Tensor) -> None:
    assert (ep.logical_not(t > 3) == (t <= 3)).all()


def test_softmax_manual(t: Tensor) -> None:
    s = ep.softmax(t)
    assert (s >= 0).all()
    assert (s <= 1).all()
    np.testing.assert_allclose(s.sum().numpy(), 1.0, rtol=1e-6)


def test_log_softmax_manual(t: Tensor) -> None:
    np.testing.assert_allclose(
        ep.log_softmax(t).exp().numpy(), ep.softmax(t).numpy(), rtol=1e-6
    )


def test_value_and_grad_fn(dummy: Tensor) -> None:
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x: ep.Tensor) -> ep.Tensor:
        return x.square().sum()

    vgf = ep.value_and_grad_fn(dummy, f)
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, g = vgf(t)
    assert v.item() == 140
    assert (g == 2 * t).all()


def test_value_and_grad_fn_with_aux(dummy: Tensor) -> None:
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.square()
        return x.sum(), x

    vgf = ep.value_and_grad_fn(dummy, f, has_aux=True)
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, aux, g = vgf(t)
    assert v.item() == 140
    assert (aux == t.square()).all()
    assert (g == 2 * t).all()


def test_value_and_grad(dummy: Tensor) -> None:
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x: Tensor) -> Tensor:
        return x.square().sum()

    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, g = ep.value_and_grad(f, t)
    assert v.item() == 140
    assert (g == 2 * t).all()


def test_value_and_grad_repeated_calls(dummy: Tensor) -> None:
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x: Tensor) -> Tensor:
        return x.square().sum()

    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v1, g1 = ep.value_and_grad(f, t)
    v2, g2 = ep.value_and_grad(f, t)
    assert v1.item() == 140 == v2.item()
    assert (g1 == 2 * t).all()
    assert (g2 == 2 * t).all()


def test_value_aux_and_grad(dummy: Tensor) -> None:
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.square()
        return x.sum(), x

    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, aux, g = ep.value_aux_and_grad(f, t)
    assert v.item() == 140
    assert (aux == t.square()).all()
    assert (g == 2 * t).all()


def test_value_aux_and_grad_multiple_aux(dummy: Tensor) -> None:
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x = x.square()
        return x.sum(), (x, x + 1)

    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, (aux0, aux1), g = ep.value_aux_and_grad(f, t)
    assert v.item() == 140
    assert (aux0 == t.square()).all()
    assert (aux1 == t.square() + 1).all()
    assert (g == 2 * t).all()


def test_value_and_grad_multiple_args(dummy: Tensor) -> None:
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x: Tensor, y: Tensor) -> Tensor:
        return (x * y).sum()

    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, g = ep.value_and_grad(f, t, t)
    assert v.item() == 140
    assert (g == t).all()


def test_logical_and_manual(t: Tensor) -> None:
    assert (ep.logical_and(t < 3, ep.ones_like(t).bool()) == (t < 3)).all()


def test_transpose_1d(dummy: Tensor) -> None:
    t = ep.arange(dummy, 8).float32()
    assert (ep.transpose(t) == t).all()


def test_inv(dummy: Tensor) -> None:
    x_list = [[1, -1, 0], [2, -3, 1], [-2, 0, 1]]
    x = ep.from_numpy(dummy, np.array(x_list, dtype=float))
    n_list = [[-3, 1, -1], [-4, 1, -1], [-6, 2, -1]]
    n = ep.from_numpy(dummy, np.array(n_list, dtype=float))
    t = ep.inv(x)

    t = t.numpy()
    n = n.numpy()
    assert t.shape == n.shape
    np.testing.assert_allclose(t, n, rtol=1e-6)


def test_round(dummy: Tensor) -> None:
    x = ep.from_numpy(dummy, np.linspace(0, 5, 13))
    n = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5], dtype=float)

    t = ep.round(x)
    t = t.numpy()
    assert t.shape == n.shape
    np.testing.assert_allclose(t, n, rtol=1e-6)


def test_ceil(dummy: Tensor) -> None:
    x = ep.from_numpy(dummy, np.linspace(0, 5, 13))
    n = np.array([0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5], dtype=float)

    t = ep.ceil(x)
    t = t.numpy()
    assert t.shape == n.shape
    np.testing.assert_allclose(t, n, rtol=1e-6)


def test_floor(dummy: Tensor) -> None:
    x = ep.from_numpy(dummy, np.linspace(0, 5, 13))
    n = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5], dtype=float)

    t = ep.floor(x)
    t = t.numpy()
    assert t.shape == n.shape
    np.testing.assert_allclose(t, n, rtol=1e-6)


def test_onehot_like_raises(dummy: Tensor) -> None:
    t = ep.arange(dummy, 18).float32().reshape((6, 3))
    indices = ep.arange(t, 6) // 2
    ep.onehot_like(t, indices)

    t = ep.arange(dummy, 90).float32().reshape((6, 3, 5))
    indices = ep.arange(t, 6) // 2
    with pytest.raises(ValueError):
        ep.onehot_like(t, indices)

    t = ep.arange(dummy, 18).float32().reshape((6, 3))
    indices = ep.arange(t, 6).reshape((6, 1)) // 2
    with pytest.raises(ValueError):
        ep.onehot_like(t, indices)

    t = ep.arange(dummy, 18).float32().reshape((6, 3))
    indices = ep.arange(t, 5) // 2
    with pytest.raises(ValueError):
        ep.onehot_like(t, indices)


def test_tile_raises(t: Tensor) -> None:
    ep.tile(t, (3,) * t.ndim)
    with pytest.raises(ValueError):
        ep.tile(t, (3,) * (t.ndim - 1))


def test_pad_raises(dummy: Tensor) -> None:
    t = ep.arange(dummy, 120).reshape((2, 3, 4, 5)).float32()
    ep.pad(t, ((0, 0), (0, 0), (2, 3), (1, 2)), mode="constant")
    with pytest.raises(ValueError):
        ep.pad(t, ((0, 0), (2, 3), (1, 2)), mode="constant")
    with pytest.raises(ValueError):
        ep.pad(
            t, ((0, 0), (0, 0, 1, 2), (2, 3), (1, 2)), mode="constant"  # type: ignore
        )
    with pytest.raises(ValueError):
        ep.pad(t, ((0, 0), (0, 0), (2, 3), (1, 2)), mode="foo")


def test_mean_bool(t: Tensor) -> None:
    with pytest.raises(ValueError):
        ep.mean(t != 0)


def test_mean_int(t: Tensor) -> None:
    with pytest.raises(ValueError):
        ep.mean(ep.arange(t, 5))


@pytest.mark.parametrize("f", [ep.logical_and, ep.logical_or])
def test_logical_and_nonboolean(
    t: Tensor, f: Callable[[Tensor, Tensor], Tensor]
) -> None:
    t = t.float32()
    f(t > 1, t > 1)
    with pytest.raises(ValueError):
        f(t, t > 1)
    with pytest.raises(ValueError):
        f(t > 1, t)
    with pytest.raises(ValueError):
        f(t, t)


def test_crossentropy_raises(dummy: Tensor) -> None:
    t = ep.arange(dummy, 50).reshape((10, 5)).float32()
    t = t / t.max()
    ep.crossentropy(t, t.argmax(axis=-1))

    t = ep.arange(dummy, 150).reshape((10, 5, 3)).float32()
    t = t / t.max()
    with pytest.raises(ValueError):
        ep.crossentropy(t, t.argmax(axis=-1))

    t = ep.arange(dummy, 50).reshape((10, 5)).float32()
    t = t / t.max()
    with pytest.raises(ValueError):
        ep.crossentropy(t, t.argmax(axis=-1)[:8])


def test_matmul_raise(dummy: Tensor) -> None:
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    ep.matmul(t, t.T)
    with pytest.raises(ValueError):
        ep.matmul(t, t[0])
    with pytest.raises(ValueError):
        ep.matmul(t[0], t)
    with pytest.raises(ValueError):
        ep.matmul(t[0], t[0])


def test_take_along_axis_2d_first_raises(dummy: Tensor) -> None:
    t = ep.arange(dummy, 32).float32().reshape((8, 4))
    indices = ep.arange(t, t.shape[-1]) % t.shape[0]
    with pytest.raises(NotImplementedError):
        ep.take_along_axis(t, indices[ep.newaxis], axis=0)


def test_norms_class() -> None:
    assert ep.Tensor.norms is not None


def test_numpy_readonly(t: Tensor) -> None:
    a = t.numpy()
    assert a.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        a[:] += 1


def test_numpy_inplace(t: Tensor) -> None:
    copy = t + 0
    a = t.numpy().copy()
    a[:] += 1
    assert (t == copy).all()


def test_iter_list_stack(t: Tensor) -> None:
    t2 = ep.stack(list(iter(t)))
    assert t.shape == t2.shape
    assert (t == t2).all()


def test_list_stack(t: Tensor) -> None:
    t2 = ep.stack(list(t))
    assert t.shape == t2.shape
    assert (t == t2).all()


def test_iter_next(t: Tensor) -> None:
    assert isinstance(next(iter(t)), Tensor)


def test_flatten(dummy: Tensor) -> None:
    t = ep.ones(dummy, (16, 3, 32, 32))
    assert ep.flatten(t).shape == (16 * 3 * 32 * 32,)
    assert ep.flatten(t, start=1).shape == (16, 3 * 32 * 32)
    assert ep.flatten(t, start=2).shape == (16, 3, 32 * 32)
    assert ep.flatten(t, start=3).shape == (16, 3, 32, 32)
    assert ep.flatten(t, end=-2).shape == (16 * 3 * 32, 32)
    assert ep.flatten(t, end=-3).shape == (16 * 3, 32, 32)
    assert ep.flatten(t, end=-4).shape == (16, 3, 32, 32)
    assert ep.flatten(t, start=1, end=-2).shape == (16, 3 * 32, 32)


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
def test_squeeze_not_one(dummy: Tensor, axis: Optional[AxisAxes]) -> None:
    t = ep.zeros(dummy, (3, 4, 5))
    if axis is None:
        t.squeeze(axis=axis)
    else:
        with pytest.raises(Exception):
            # squeezing specifc axis should fail if they are not 1
            t.squeeze(axis=axis)


###############################################################################
# special tests
# - decorated with compare_*
# - return values
###############################################################################


def get_numpy_kwargs(kwargs: Any) -> Dict:
    return {
        k: ep.astensor(t.numpy()) if ep.istensor(t) else t for k, t in kwargs.items()
    }


def compare_all(f: Callable[..., Tensor]) -> Callable[..., None]:
    """A decorator to simplify writing test functions"""

    @functools.wraps(f)
    def test_fn(*args: Any, **kwargs: Any) -> None:
        assert len(args) == 0
        nkwargs = get_numpy_kwargs(kwargs)
        t = f(*args, **kwargs)
        n = f(*args, **nkwargs)
        t = t.numpy()
        n = n.numpy()
        assert t.shape == n.shape
        assert (t == n).all()

    return test_fn


def compare_allclose(*args: Any, rtol: float = 1e-07, atol: float = 0) -> Callable:
    """A decorator to simplify writing test functions"""

    def compare_allclose_inner(f: Callable[..., Tensor]) -> Callable[..., None]:
        @functools.wraps(f)
        def test_fn(*args: Any, **kwargs: Any) -> None:
            assert len(args) == 0
            nkwargs = get_numpy_kwargs(kwargs)
            t = f(*args, **kwargs)
            n = f(*args, **nkwargs)
            t = t.numpy()
            n = n.numpy()
            assert t.shape == n.shape
            np.testing.assert_allclose(t, n, rtol=rtol, atol=atol)

        return test_fn

    if len(args) == 1 and callable(args[0]):
        # decorator applied without parenthesis
        return compare_allclose_inner(args[0])
    return compare_allclose_inner


def compare_equal(
    f: Callable[..., Union[Tensor, int, float, bool, Shape]]
) -> Callable[..., None]:
    """A decorator to simplify writing test functions"""

    @functools.wraps(f)
    def test_fn(*args: Any, **kwargs: Any) -> None:
        assert len(args) == 0
        nkwargs = get_numpy_kwargs(kwargs)
        t = f(*args, **kwargs)
        n = f(*args, **nkwargs)
        assert isinstance(t, type(n))
        assert t == n

    return test_fn


@compare_equal
def test_format(dummy: Tensor) -> bool:
    t = ep.arange(dummy, 5).sum()
    return f"{t:.1f}" == "10.0"


@compare_equal
def test_item(t: Tensor) -> float:
    t = t.sum()
    return t.item()


@compare_equal
def test_len(t: Tensor) -> int:
    return len(t)


@compare_equal
def test_scalar_bool(t: Tensor) -> bool:
    return bool(ep.sum(t) == 0)


@compare_all
def test_neg(t: Tensor) -> Tensor:
    return -t


@compare_all
def test_square(t: Tensor) -> Tensor:
    return ep.square(t)


@compare_allclose(rtol=1e-6)
def test_pow(t: Tensor) -> Tensor:
    return ep.pow(t, 3)


@compare_allclose(rtol=1e-6)
def test_pow_float(t: Tensor) -> Tensor:
    return ep.pow(t, 2.5)


@compare_allclose(rtol=1e-6)
def test_pow_op(t: Tensor) -> Tensor:
    return t**3


@compare_allclose(rtol=1e-5)
def test_pow_tensor(t: Tensor) -> Tensor:
    return ep.pow(t, (t + 0.5))


@compare_allclose(rtol=1e-5)
def test_pow_op_tensor(t: Tensor) -> Tensor:
    return t ** (t + 0.5)


@compare_all
def test_add(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + t2


@compare_all
def test_add_scalar(t: Tensor) -> Tensor:
    return t + 3


@compare_all
def test_radd_scalar(t: Tensor) -> Tensor:
    return 3 + t


@compare_all
def test_sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 - t2


@compare_all
def test_sub_scalar(t: Tensor) -> Tensor:
    return t - 3


@compare_all
def test_rsub_scalar(t: Tensor) -> Tensor:
    return 3 - t


@compare_all
def test_mul(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 * t2


@compare_all
def test_mul_scalar(t: Tensor) -> Tensor:
    return t * 3


@compare_all
def test_rmul_scalar(t: Tensor) -> Tensor:
    return 3 * t


@compare_allclose
def test_truediv(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 / t2


@compare_allclose(rtol=1e-6)
def test_truediv_scalar(t: Tensor) -> Tensor:
    return t / 3


@compare_allclose
def test_rtruediv_scalar(t: Tensor) -> Tensor:
    return 3 / (abs(t) + 3e-8)


@compare_allclose
def test_floordiv(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 // t2


@compare_allclose(rtol=1e-6)
def test_floordiv_scalar(t: Tensor) -> Tensor:
    return t // 3


@compare_allclose
def test_rfloordiv_scalar(t: Tensor) -> Tensor:
    return 3 // (abs(t) + 1e-8)


@compare_all
def test_mod(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 % (abs(t2) + 1)


@compare_all
def test_mod_scalar(t: Tensor) -> Tensor:
    return t % 3


@compare_all
def test_getitem(t: Tensor) -> Tensor:
    return t[2]


@compare_all
def test_getitem_tuple(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    return t[1, 3]


@compare_all
def test_getitem_newaxis(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 8).float32()
    return t[ep.newaxis]


@compare_all
def test_getitem_ellipsis_newaxis(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 8).float32()
    return t[..., ep.newaxis]


@compare_all
def test_getitem_tensor(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32()
    indices = ep.arange(t, 3, 10, 2)
    return t[indices]


@compare_all
def test_getitem_range(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32()
    indices = range(3, 10, 2)
    return t[indices]


@compare_all
def test_getitem_list(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32()
    indices = list(range(3, 10, 2))
    return t[indices]


@compare_all
def test_getitem_ndarray(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32()
    indices = np.arange(3, 10, 2)
    return t[indices]


@compare_all
def test_getitem_tuple_tensors(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32().reshape((8, 4))
    rows = ep.arange(t, len(t))
    indices = ep.arange(t, len(t)) % t.shape[1]
    return t[rows, indices]


@compare_all
def test_getitem_tuple_tensors_full(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32().reshape((8, 4))
    rows = ep.arange(t, len(t))[:, np.newaxis].tile((1, t.shape[-1]))
    cols = t.argsort(axis=-1)
    return t[rows, cols]


@compare_all
def test_getitem_tuple_tensors_full_broadcast(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32().reshape((8, 4))
    rows = ep.arange(t, len(t))[:, np.newaxis]
    cols = t.argsort(axis=-1)
    return t[rows, cols]


@compare_all
def test_getitem_tuple_range_tensor(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32().reshape((8, 4))
    rows = range(len(t))
    indices = ep.arange(t, len(t)) % t.shape[1]
    return t[rows, indices]


@compare_all
def test_getitem_tuple_range_range(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 36).float32().reshape((6, 6))
    rows = cols = range(len(t))
    return t[rows, cols]


@compare_all
def test_getitem_tuple_list_tensor(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32().reshape((8, 4))
    rows = list(range(len(t)))
    indices = ep.arange(t, len(t)) % t.shape[1]
    return t[rows, indices]


@compare_all
def test_getitem_slice(t: Tensor) -> Tensor:
    return t[1:3]


@compare_all
def test_getitem_slice_slice(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32().reshape((4, 8))
    return t[:, :3]


@compare_all
def test_getitem_boolean_tensor(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32().reshape((4, 8))
    indices = ep.arange(t, 4) <= 2
    return t[indices]


@compare_all
def test_take_along_axis_2d(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 32).float32().reshape((8, 4))
    indices = ep.arange(t, len(t)) % t.shape[-1]
    return ep.take_along_axis(t, indices[..., ep.newaxis], axis=-1)


@compare_all
def test_take_along_axis_3d(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 64).float32().reshape((2, 8, 4))
    indices = ep.arange(t, 2 * 8).reshape((2, 8, 1)) % t.shape[-1]
    return ep.take_along_axis(t, indices, axis=-1)


@compare_allclose
def test_sqrt(t: Tensor) -> Tensor:
    return ep.sqrt(t)


@compare_equal
def test_shape(t: Tensor) -> Shape:
    return t.shape


@compare_all
def test_reshape(t: Tensor) -> Tensor:
    shape = (1,) + t.shape + (1,)
    return ep.reshape(t, shape)


@compare_all
def test_reshape_minus_1(t: Tensor) -> Tensor:
    return ep.reshape(t, -1)


@compare_all
def test_reshape_int(t: Tensor) -> Tensor:
    n = 1
    for k in t.shape:
        n *= k
    return ep.reshape(t, n)


@compare_all
def test_clip(t: Tensor) -> Tensor:
    return ep.clip(t, 2, 3.5)


@compare_all
def test_sign(t: Tensor) -> Tensor:
    return ep.sign(t)


@compare_all
def test_sum(t: Tensor) -> Tensor:
    return ep.sum(t)


@compare_all
def test_sum_axis(t: Tensor) -> Tensor:
    return ep.sum(t, axis=0)


@compare_all
def test_sum_axes(dummy: Tensor) -> Tensor:
    t = ep.ones(dummy, 30).float32().reshape((3, 5, 2))
    return ep.sum(t, axis=(0, 1))


@compare_all
def test_sum_keepdims(t: Tensor) -> Tensor:
    return ep.sum(t, axis=0, keepdims=True)


@compare_all
def test_sum_none_keepdims(t: Tensor) -> Tensor:
    return ep.sum(t, axis=None, keepdims=True)


@compare_all
def test_sum_bool(t: Tensor) -> Tensor:
    return ep.sum(t != 0)


@compare_all
def test_sum_int(t: Tensor) -> Tensor:
    return ep.sum(ep.arange(t, 5))


@compare_all
def test_prod(t: Tensor) -> Tensor:
    return ep.prod(t)


@compare_all
def test_prod_axis(t: Tensor) -> Tensor:
    return ep.prod(t, axis=0)


@compare_all
def test_prod_axes(dummy: Tensor) -> Tensor:
    t = ep.ones(dummy, 30).float32().reshape((3, 5, 2))
    return ep.prod(t, axis=(0, 1))


@compare_all
def test_prod_keepdims(t: Tensor) -> Tensor:
    return ep.prod(t, axis=0, keepdims=True)


@compare_all
def test_prod_none_keepdims(t: Tensor) -> Tensor:
    return ep.prod(t, axis=None, keepdims=True)


@compare_all
def test_prod_bool(t: Tensor) -> Tensor:
    return ep.prod(t != 0)


@compare_all
def test_prod_int(t: Tensor) -> Tensor:
    return ep.prod(ep.arange(t, 5))


@compare_all
def test_mean(t: Tensor) -> Tensor:
    return ep.mean(t)


@compare_all
def test_mean_axis(t: Tensor) -> Tensor:
    return ep.mean(t, axis=0)


@compare_all
def test_mean_axes(dummy: Tensor) -> Tensor:
    t = ep.ones(dummy, 30).float32().reshape((3, 5, 2))
    return ep.mean(t, axis=(0, 1))


@compare_all
def test_mean_keepdims(t: Tensor) -> Tensor:
    return ep.mean(t, axis=0, keepdims=True)


@compare_all
def test_mean_none_keepdims(t: Tensor) -> Tensor:
    return ep.mean(t, axis=None, keepdims=True)


@compare_all
def test_all(t: Tensor) -> Tensor:
    return ep.all(t > 3)


@compare_all
def test_all_axis(t: Tensor) -> Tensor:
    return ep.all(t > 3, axis=0)


@compare_all
def test_all_axes(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 30).float32().reshape((3, 5, 2))
    return ep.all(t > 3, axis=(0, 1))


@compare_all
def test_all_keepdims(t: Tensor) -> Tensor:
    return ep.all(t > 3, axis=0, keepdims=True)


@compare_all
def test_all_none_keepdims(t: Tensor) -> Tensor:
    return ep.all(t > 3, axis=None, keepdims=True)


@compare_all
def test_any(t: Tensor) -> Tensor:
    return ep.any(t > 3)


@compare_all
def test_any_axis(t: Tensor) -> Tensor:
    return ep.any(t > 3, axis=0)


@compare_all
def test_any_axes(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 30).float32().reshape((3, 5, 2))
    return ep.any(t > 3, axis=(0, 1))


@compare_all
def test_any_keepdims(t: Tensor) -> Tensor:
    return ep.any(t > 3, axis=0, keepdims=True)


@compare_all
def test_any_none_keepdims(t: Tensor) -> Tensor:
    return ep.any(t > 3, axis=None, keepdims=True)


@compare_all
def test_min(t: Tensor) -> Tensor:
    return ep.min(t)


@compare_all
def test_min_axis(t: Tensor) -> Tensor:
    return ep.min(t, axis=0)


@compare_all
def test_min_axes(dummy: Tensor) -> Tensor:
    t = ep.ones(dummy, 30).float32().reshape((3, 5, 2))
    return ep.min(t, axis=(0, 1))


@compare_all
def test_min_keepdims(t: Tensor) -> Tensor:
    return ep.min(t, axis=0, keepdims=True)


@compare_all
def test_min_none_keepdims(t: Tensor) -> Tensor:
    return ep.min(t, axis=None, keepdims=True)


@compare_all
def test_max(t: Tensor) -> Tensor:
    return ep.max(t)


@compare_all
def test_max_axis(t: Tensor) -> Tensor:
    return ep.max(t, axis=0)


@compare_all
def test_max_axes(dummy: Tensor) -> Tensor:
    t = ep.ones(dummy, 30).float32().reshape((3, 5, 2))
    return ep.max(t, axis=(0, 1))


@compare_all
def test_max_keepdims(t: Tensor) -> Tensor:
    return ep.max(t, axis=0, keepdims=True)


@compare_all
def test_max_none_keepdims(t: Tensor) -> Tensor:
    return ep.max(t, axis=None, keepdims=True)


@compare_allclose(rtol=1e-6)
def test_exp(t: Tensor) -> Tensor:
    return ep.exp(t)


@compare_allclose
def test_log(t: Tensor) -> Tensor:
    return ep.log(t.maximum(1e-8))


@compare_allclose
def test_log2(t: Tensor) -> Tensor:
    return ep.log2(t.maximum(1e-8))


@compare_allclose
def test_log10(t: Tensor) -> Tensor:
    return ep.log10(t.maximum(1e-8))


@compare_allclose
def test_log1p(t: Tensor) -> Tensor:
    return ep.log1p(t)


@compare_allclose(rtol=1e-6)
def test_sin(t: Tensor) -> Tensor:
    return ep.sin(t)


@compare_allclose(rtol=1e-6)
def test_cos(t: Tensor) -> Tensor:
    return ep.cos(t)


@compare_allclose(rtol=1e-6)
def test_tan(t: Tensor) -> Tensor:
    return ep.tan(t)


@compare_allclose(rtol=1e-6)
def test_sinh(t: Tensor) -> Tensor:
    return ep.sinh(t)


@compare_allclose(rtol=1e-6)
def test_cosh(t: Tensor) -> Tensor:
    return ep.cosh(t)


@compare_allclose(rtol=1e-6)
def test_tanh(t: Tensor) -> Tensor:
    return ep.tanh(t)


@compare_allclose(rtol=1e-6)
def test_arcsin(t: Tensor) -> Tensor:
    # domain is [-1, 1]
    return ep.arcsin((t - t.mean()) / t.max())


@compare_allclose(rtol=1e-6)
def test_arccos(t: Tensor) -> Tensor:
    # domain is [-1, 1]
    return ep.arccos((t - t.mean()) / t.max())


@compare_allclose(rtol=1e-6)
def test_arctan(t: Tensor) -> Tensor:
    return ep.arctan(t)


@compare_allclose(rtol=1e-6)
def test_arcsinh(t: Tensor) -> Tensor:
    return ep.arcsinh(t)


@compare_allclose(rtol=1e-6)
def test_arccosh(t: Tensor) -> Tensor:
    # domain is [1, inf)
    return ep.arccosh(1 + t - t.min())


@compare_allclose(rtol=1e-6)
def test_arctanh(t: Tensor) -> Tensor:
    # domain is [-1, 1]
    return ep.arctanh((t - t.mean()) / t.max())


@compare_all
def test_abs_op(t: Tensor) -> Tensor:
    return abs(t)


@compare_all
def test_abs(t: Tensor) -> Tensor:
    return ep.abs(t)


@compare_all
def test_minimum(t1: Tensor, t2: Tensor) -> Tensor:
    return ep.minimum(t1, t2)


@compare_all
def test_minimum_scalar(t: Tensor) -> Tensor:
    return ep.minimum(t, 3)


@compare_all
def test_rminimum_scalar(t: Tensor) -> Tensor:
    return ep.minimum(3, t)


@compare_all
def test_maximum(t1: Tensor, t2: Tensor) -> Tensor:
    return ep.maximum(t1, t2)


@compare_all
def test_maximum_scalar(t: Tensor) -> Tensor:
    return ep.maximum(t, 3)


@compare_all
def test_rmaximum_scalar(t: Tensor) -> Tensor:
    return ep.maximum(3, t)


@compare_all
def test_argmin(t: Tensor) -> Tensor:
    return ep.argmin(t)


@compare_all
def test_argmin_axis(t: Tensor) -> Tensor:
    return ep.argmin(t, axis=0)


@compare_all
def test_argmax(t: Tensor) -> Tensor:
    return ep.argmax(t)


@compare_all
def test_argmax_axis(t: Tensor) -> Tensor:
    return ep.argmax(t, axis=0)


@compare_all
def test_logical_and(t: Tensor) -> Tensor:
    return ep.logical_and(t < 3, t > 1)


@compare_all
def test_logical_and_scalar(t: Tensor) -> Tensor:
    return ep.logical_and(True, t < 3)


@compare_all
def test_logical_or(t: Tensor) -> Tensor:
    return ep.logical_or(t > 3, t < 1)


@compare_all
def test_logical_or_scalar(t: Tensor) -> Tensor:
    return ep.logical_or(True, t < 1)


@compare_all
def test_logical_not(t: Tensor) -> Tensor:
    return ep.logical_not(t > 3)


@compare_all
def test_isnan_false(t: Tensor) -> Tensor:
    return ep.isnan(t)


@compare_all
def test_isnan_true(t: Tensor) -> Tensor:
    return ep.isnan(t + ep.nan)


@compare_all
def test_isinf(t: Tensor) -> Tensor:
    return ep.isinf(t)


@compare_all
def test_isinf_posinf(t: Tensor) -> Tensor:
    return ep.isinf(t + ep.inf)


@compare_all
def test_isinf_neginf(t: Tensor) -> Tensor:
    return ep.isinf(t - ep.inf)


@compare_all
def test_zeros_like(t: Tensor) -> Tensor:
    return ep.zeros_like(t)


@compare_all
def test_ones_like(t: Tensor) -> Tensor:
    return ep.ones_like(t)


@compare_all
def test_full_like(t: Tensor) -> Tensor:
    return ep.full_like(t, 5)


@pytest.mark.parametrize("value", [1, -1, 2])
@compare_all
def test_onehot_like(dummy: Tensor, value: float) -> Tensor:
    t = ep.arange(dummy, 18).float32().reshape((6, 3))
    indices = ep.arange(t, 6) // 2
    return ep.onehot_like(t, indices, value=value)


@compare_all
def test_zeros_scalar(t: Tensor) -> Tensor:
    return ep.zeros(t, 5)


@compare_all
def test_zeros_tuple(t: Tensor) -> Tensor:
    return ep.zeros(t, (2, 3))


@compare_all
def test_ones_scalar(t: Tensor) -> Tensor:
    return ep.ones(t, 5)


@compare_all
def test_ones_tuple(t: Tensor) -> Tensor:
    return ep.ones(t, (2, 3))


@compare_all
def test_full_scalar(t: Tensor) -> Tensor:
    return ep.full(t, 5, 4.0)


@compare_all
def test_full_tuple(t: Tensor) -> Tensor:
    return ep.full(t, (2, 3), 4.0)


@compare_equal
def test_uniform_scalar(t: Tensor) -> Shape:
    return ep.uniform(t, 5).shape


@compare_equal
def test_uniform_tuple(t: Tensor) -> Shape:
    return ep.uniform(t, (2, 3)).shape


@compare_equal
def test_normal_scalar(t: Tensor) -> Shape:
    return ep.normal(t, 5).shape


@compare_equal
def test_normal_tuple(t: Tensor) -> Shape:
    return ep.normal(t, (2, 3)).shape


@compare_all
def test_argsort(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 6).float32().reshape((2, 3))
    return ep.argsort(t)


@compare_all
def test_sort(dummy: Tensor) -> Tensor:
    t = -ep.arange(dummy, 6).float32().reshape((2, 3))
    return ep.sort(t)


@compare_all
def test_topk_values(dummy: Tensor) -> Tensor:
    t = (ep.arange(dummy, 27).reshape((3, 3, 3)) ** 2 * 10000 % 1234).float32()
    values, _ = ep.topk(t, 2)
    return values


@compare_all
def test_topk_indices(dummy: Tensor) -> Tensor:
    t = -(ep.arange(dummy, 27).reshape((3, 3, 3)) ** 2 * 10000 % 1234).float32()
    _, indices = ep.topk(t, 2)
    return indices


@compare_all
def test_transpose(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    return ep.transpose(t)


@compare_all
def test_transpose_axes(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 60).float32().reshape((3, 4, 5))
    return ep.transpose(t, axes=(1, 2, 0))


@compare_all
def test_where(t: Tensor) -> Tensor:
    return ep.where(t >= 3, t, -t)


@compare_all
def test_where_first_scalar(t: Tensor) -> Tensor:
    return ep.where(t >= 3, 2, -t)


@compare_all
def test_where_second_scalar(t: Tensor) -> Tensor:
    return ep.where(t >= 3, t, 2)


@compare_all
def test_where_first_scalar64(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 60).float64().reshape((3, 4, 5))
    return ep.where(t >= 3, 2, -t)


@compare_all
def test_where_second_scalar64(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 60).float64().reshape((3, 4, 5))
    return ep.where(t >= 3, t, 2)


@compare_all
def test_where_both_scalars(t: Tensor) -> Tensor:
    return ep.where(t >= 3, 2, 5)


@compare_all
def test_tile(t: Tensor) -> Tensor:
    return ep.tile(t, (3,) * t.ndim)


@compare_all
def test_matmul(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    return ep.matmul(t, t.T)


@compare_all
def test_matmul_operator(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    return t @ t.T


@compare_allclose(rtol=1e-6)
def test_softmax(t: Tensor) -> Tensor:
    return ep.softmax(t)


@compare_allclose(rtol=1e-5)
def test_log_softmax(t: Tensor) -> Tensor:
    return ep.log_softmax(t)


@compare_allclose
def test_crossentropy(dummy: Tensor) -> Tensor:
    t = ep.arange(dummy, 50).reshape((10, 5)).float32()
    t = t / t.max()
    return ep.crossentropy(t, t.argmax(axis=-1))


@pytest.mark.parametrize(
    "array, output",
    itertools.product(
        [
            np.array([[1, 2], [3, 4]]),
            np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]),
            np.arange(100).reshape((10, 10)),
        ],
        ["sign", "logdet"],
    ),
    ids=map(
        lambda *l: "_".join(*l),
        itertools.product(
            ["matrix_finite", "stack_of_matrices", "matrix_infinite"],
            ["sign", "logdet"],
        ),
    ),
)
@compare_allclose
def test_slogdet(dummy: Tensor, array: Tensor, output: str) -> Tensor:
    a = ep.from_numpy(dummy, array).float32()
    outputs = dict()
    outputs["sign"], outputs["logdet"] = ep.slogdet(a)
    return outputs[output]


@pytest.mark.parametrize("axis", [0, 1, -1])
@compare_all
def test_stack(t1: Tensor, t2: Tensor, axis: int) -> Tensor:
    return ep.stack([t1, t2], axis=axis)


@compare_all
def test_concatenate_axis0(dummy: Tensor) -> Tensor:
    t1 = ep.arange(dummy, 12).float32().reshape((4, 3))
    t2 = ep.arange(dummy, 20, 32, 2).float32().reshape((2, 3))
    return ep.concatenate([t1, t2], axis=0)


@compare_all
def test_concatenate_axis1(dummy: Tensor) -> Tensor:
    t1 = ep.arange(dummy, 12).float32().reshape((3, 4))
    t2 = ep.arange(dummy, 20, 32, 2).float32().reshape((3, 2))
    return ep.concatenate([t1, t2], axis=1)


@pytest.mark.parametrize("axis", [0, 1, -1])
@compare_all
def test_expand_dims(t: Tensor, axis: int) -> Tensor:
    return ep.expand_dims(t, axis)


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
@compare_all
def test_squeeze(t: Tensor, axis: Optional[AxisAxes]) -> Tensor:
    t = t.expand_dims(axis=0).expand_dims(axis=1)
    return ep.squeeze(t, axis=axis)


@compare_all
def test_arange(dummy: Tensor) -> Tensor:
    return ep.arange(dummy, 6)


@compare_all
def test_arange_start(dummy: Tensor) -> Tensor:
    return ep.arange(dummy, 5, 10)


@compare_all
def test_arange_step(dummy: Tensor) -> Tensor:
    return ep.arange(dummy, 4, 8, 2)


@compare_all
def test_cumsum(t: Tensor) -> Tensor:
    return ep.cumsum(t)


@compare_all
def test_cumsum_axis(t: Tensor) -> Tensor:
    return ep.cumsum(t, axis=0)


@compare_all
def test_flip(t: Tensor) -> Tensor:
    return ep.flip(t)


@compare_all
def test_flip_axis(t: Tensor) -> Tensor:
    return ep.flip(t, axis=0)


@pytest.mark.parametrize("indexing", ["ij", "xy"])
@pytest.mark.parametrize("i", [0, 1])
@compare_all
def test_meshgrid_a(dummy: Tensor, indexing: str, i: int) -> Tensor:
    t1 = ep.arange(dummy, 5)
    t2 = ep.arange(dummy, 3)
    results = ep.meshgrid(t1, t2, indexing=indexing)
    assert len(results) == 2
    return results[i]


@pytest.mark.parametrize(
    "mode,value", [("constant", 0), ("constant", -2), ("reflect", 0)]
)
@compare_all
def test_pad(dummy: Tensor, mode: str, value: float) -> Tensor:
    t = ep.arange(dummy, 120).reshape((2, 3, 4, 5)).float32()
    return ep.pad(t, ((0, 0), (0, 0), (2, 3), (1, 2)), mode=mode, value=value)


@compare_all
def test_index_update_row(dummy: Tensor) -> Tensor:
    x = ep.ones(dummy, (3, 4))
    return ep.index_update(x, ep.index[1], ep.ones(x, 4) * 66.0)


@compare_all
def test_index_update_row_scalar(dummy: Tensor) -> Tensor:
    x = ep.ones(dummy, (3, 4))
    return ep.index_update(x, ep.index[1], 66.0)


@compare_all
def test_index_update_column(dummy: Tensor) -> Tensor:
    x = ep.ones(dummy, (3, 4))
    return ep.index_update(x, ep.index[:, 1], ep.ones(x, 3) * 66.0)


@compare_all
def test_index_update_column_scalar(dummy: Tensor) -> Tensor:
    x = ep.ones(dummy, (3, 4))
    return ep.index_update(x, ep.index[:, 1], 66.0)


@compare_all
def test_index_update_indices(dummy: Tensor) -> Tensor:
    x = ep.ones(dummy, (3, 4))
    ind = ep.from_numpy(dummy, np.array([0, 1, 2, 1]))
    return ep.index_update(x, ep.index[ind, ep.arange(x, 4)], ep.ones(x, 4) * 33.0)


@compare_all
def test_index_update_indices_scalar(dummy: Tensor) -> Tensor:
    x = ep.ones(dummy, (3, 4))
    ind = ep.from_numpy(dummy, np.array([0, 1, 2, 1]))
    return ep.index_update(x, ep.index[ind, ep.arange(x, 4)], 33.0)


@compare_all
def test_lt(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 < t2


@compare_all
def test_lt_scalar(t1: Tensor, t2: Tensor) -> Tensor:
    return 3 < t2


@compare_all
def test_le(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 <= t2


@compare_all
def test_le_scalar(t1: Tensor, t2: Tensor) -> Tensor:
    return 3 <= t2


@compare_all
def test_gt(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 > t2


@compare_all
def test_gt_scalar(t1: Tensor, t2: Tensor) -> Tensor:
    return 3 > t2


@compare_all
def test_ge(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 >= t2


@compare_all
def test_ge_scalar(t1: Tensor, t2: Tensor) -> Tensor:
    return 3 >= t2


@compare_all
def test_eq(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 == t2


@compare_all
def test_eq_scalar(t1: Tensor, t2: Tensor) -> Tensor:
    return cast(Tensor, 3 == t2)


@compare_all
def test_ne(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 != t2


@compare_all
def test_ne_scalar(t1: Tensor, t2: Tensor) -> Tensor:
    return cast(Tensor, 3 != t2)


@compare_all
def test_float_int_lt(t1: Tensor, t2int: Tensor) -> Tensor:
    return t1 < t2int


@compare_all
def test_float_int_le(t1: Tensor, t2int: Tensor) -> Tensor:
    return t1 <= t2int


@compare_all
def test_float_int_gt(t1: Tensor, t2int: Tensor) -> Tensor:
    return t1 > t2int


@compare_all
def test_float_int_ge(t1: Tensor, t2int: Tensor) -> Tensor:
    return t1 >= t2int


@compare_all
def test_float_int_eq(t1: Tensor, t2int: Tensor) -> Tensor:
    return t1 == t2int


@compare_all
def test_float_int_ne(t1: Tensor, t2int: Tensor) -> Tensor:
    return t1 != t2int


@compare_all
def test_int_float_lt(t1int: Tensor, t2: Tensor) -> Tensor:
    return t1int < t2


@compare_all
def test_int_float_le(t1int: Tensor, t2: Tensor) -> Tensor:
    return t1int <= t2


@compare_all
def test_int_float_gt(t1int: Tensor, t2: Tensor) -> Tensor:
    return t1int > t2


@compare_all
def test_int_float_ge(t1int: Tensor, t2: Tensor) -> Tensor:
    return t1int >= t2


@compare_all
def test_int_float_eq(t1int: Tensor, t2: Tensor) -> Tensor:
    return t1int == t2


@compare_all
def test_int_float_ne(t1int: Tensor, t2: Tensor) -> Tensor:
    return t1int != t2


@compare_all
def test_norms_l0(t: Tensor) -> Tensor:
    return t.norms.l0()


@compare_allclose
def test_norms_l1(t: Tensor) -> Tensor:
    return t.norms.l1()


@compare_allclose
def test_norms_l2(t: Tensor) -> Tensor:
    return t.norms.l2()


@compare_all
def test_norms_linf(t: Tensor) -> Tensor:
    return t.norms.linf()


@compare_allclose
def test_norms_lp(t: Tensor) -> Tensor:
    return t.norms.lp(2)


@compare_allclose
def test_norms_cache(t: Tensor) -> Tensor:
    return t.norms.l1() + t.norms.l2()
