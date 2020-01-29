import pytest
import functools
import numpy as np
import eagerpy as ep
from eagerpy import Tensor


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


def test_astensor_tensor(t: Tensor):
    assert ep.istensor(t)
    assert (ep.astensor(t) == t).all()


def test_module():
    assert ep.istensor(ep.numpy.tanh([3, 5]))
    assert not ep.istensor(ep.numpy.tanh(3))


def test_module_dir():
    assert "zeros" in dir(ep.numpy)


def test_repr(t: Tensor):
    assert not repr(t).startswith("<")
    t = ep.zeros(t, (10, 10))
    assert not repr(t).startswith("<")
    assert len(repr(t).split("\n")) > 1


def test_logical_or_manual(t: Tensor):
    assert (ep.logical_or(t < 3, ep.zeros_like(t).bool()) == (t < 3)).all()


def test_logical_not_manual(t: Tensor):
    assert (ep.logical_not(t > 3) == (t <= 3)).all()


def test_softmax_manual(t: Tensor):
    s = ep.softmax(t)
    assert (s >= 0).all()
    assert (s <= 1).all()
    np.testing.assert_allclose(s.sum().numpy(), 1.0, rtol=1e-6)


def test_log_softmax_manual(t: Tensor):
    np.testing.assert_allclose(
        ep.log_softmax(t).exp().numpy(), ep.softmax(t).numpy(), rtol=1e-6
    )


def test_value_and_grad_fn(dummy: Tensor):
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x):
        return x.square().sum()

    vgf = ep.value_and_grad_fn(dummy, f)
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, g = vgf(t)
    assert v.item() == 140
    assert (g == 2 * t).all()


def test_value_and_grad_fn_with_aux(dummy: Tensor):
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x):
        x = x.square()
        return x.sum(), x

    vgf = ep.value_and_grad_fn(dummy, f, has_aux=True)
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, aux, g = vgf(t)
    assert v.item() == 140
    assert (aux == t.square()).all()
    assert (g == 2 * t).all()


def test_value_and_grad(dummy: Tensor):
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x):
        return x.square().sum()

    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, g = ep.value_and_grad(f, t)
    assert v.item() == 140
    assert (g == 2 * t).all()


def test_value_aux_and_grad(dummy: Tensor):
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x):
        x = x.square()
        return x.sum(), x

    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, aux, g = ep.value_aux_and_grad(f, t)
    assert v.item() == 140
    assert (aux == t.square()).all()
    assert (g == 2 * t).all()


def test_value_aux_and_grad_multiple_aux(dummy: Tensor):
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x):
        x = x.square()
        return x.sum(), (x, x + 1)

    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, (aux0, aux1), g = ep.value_aux_and_grad(f, t)
    assert v.item() == 140
    assert (aux0 == t.square()).all()
    assert (aux1 == t.square() + 1).all()
    assert (g == 2 * t).all()


def test_value_and_grad_multiple_args(dummy: Tensor):
    if isinstance(dummy, ep.NumPyTensor):
        pytest.skip()

    def f(x, y):
        return (x * y).sum()

    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    v, g = ep.value_and_grad(f, t, t)
    assert v.item() == 140
    assert (g == t).all()


def test_logical_and_manual(t: Tensor):
    assert (ep.logical_and(t < 3, ep.ones_like(t).bool()) == (t < 3)).all()


def test_transpose_1d(dummy: Tensor):
    t = ep.arange(dummy, 8).float32()
    assert (ep.transpose(t) == t).all()


###############################################################################
# special tests
# - decorated with compare_*
# - return values
###############################################################################


def get_numpy_kwargs(kwargs):
    return {
        k: ep.astensor(t.numpy()) if ep.istensor(t) else t for k, t in kwargs.items()
    }


def compare_all(f):
    """A decorator to simplify writing test functions"""

    @functools.wraps(f)
    def test_fn(*args, **kwargs):
        assert len(args) == 0
        nkwargs = get_numpy_kwargs(kwargs)
        t = f(*args, **kwargs)
        n = f(*args, **nkwargs)
        t = t.numpy()
        n = n.numpy()
        assert t.shape == n.shape
        assert (t == n).all()

    return test_fn


def compare_allclose(*args, rtol=1e-07, atol=0):
    """A decorator to simplify writing test functions"""

    def compare_allclose_inner(f):
        @functools.wraps(f)
        def test_fn(*args, **kwargs):
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


def compare_equal(f):
    """A decorator to simplify writing test functions"""

    @functools.wraps(f)
    def test_fn(*args, **kwargs):
        assert len(args) == 0
        nkwargs = get_numpy_kwargs(kwargs)
        t = f(*args, **kwargs)
        n = f(*args, **nkwargs)
        assert isinstance(t, type(n))
        assert t == n

    return test_fn


@compare_equal
def test_format(dummy: Tensor):
    t = ep.arange(dummy, 5).sum()
    return f"{t:.1f}" == "10.0"


@compare_equal
def test_item(t: Tensor):
    t = t.sum()
    return t.item()


@compare_equal
def test_len(t: Tensor):
    return len(t)


@compare_equal
def test_scalar_bool(t: Tensor):
    return bool(ep.sum(t) == 0)


@compare_all
def test_neg(t: Tensor):
    return -t


@compare_all
def test_square(t: Tensor):
    return ep.square(t)


@compare_allclose
def test_pow(t: Tensor):
    return ep.pow(t, 3)


@compare_allclose
def test_pow_op(t: Tensor):
    return t ** 3


@compare_all
def test_add(t1: Tensor, t2: Tensor):
    return t1 + t2


@compare_all
def test_add_scalar(t: Tensor):
    return t + 3


@compare_all
def test_radd_scalar(t: Tensor):
    return 3 + t


@compare_all
def test_sub(t1: Tensor, t2: Tensor):
    return t1 - t2


@compare_all
def test_sub_scalar(t: Tensor):
    return t - 3


@compare_all
def test_rsub_scalar(t: Tensor):
    return 3 - t


@compare_all
def test_mul(t1: Tensor, t2: Tensor):
    return t1 * t2


@compare_all
def test_mul_scalar(t: Tensor):
    return t * 3


@compare_all
def test_rmul_scalar(t: Tensor):
    return 3 * t


@compare_allclose
def test_truediv(t1: Tensor, t2: Tensor):
    return t1 / t2


@compare_allclose(rtol=1e-6)
def test_truediv_scalar(t: Tensor):
    return t / 3


@compare_allclose
def test_rtruediv_scalar(t: Tensor):
    return 3 / (abs(t) + 3e-8)


@compare_allclose
def test_floordiv(t1: Tensor, t2: Tensor):
    return t1 // t2


@compare_allclose(rtol=1e-6)
def test_floordiv_scalar(t: Tensor):
    return t // 3


@compare_allclose
def test_rfloordiv_scalar(t: Tensor):
    return 3 // (abs(t) + 1e-8)


@compare_all
def test_mod(t1: Tensor, t2: Tensor):
    return t1 % (abs(t2) + 1)


@compare_all
def test_mod_scalar(t: Tensor):
    return t % 3


@compare_all
def test_getitem(t: Tensor):
    return t[2]


@compare_all
def test_getitem_tuple(dummy: Tensor):
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    return t[1, 3]


@compare_all
def test_getitem_tuple_tensors(dummy: Tensor):
    t = ep.arange(dummy, 32).float32().reshape((8, 4))
    rows = ep.arange(t, len(t))
    indices = ep.arange(t, len(t)) % t.shape[1]
    return t[rows, indices]


@compare_all
def test_getitem_slice(t: Tensor):
    return t[1:3]


@compare_all
def test_sqrt(t: Tensor):
    return ep.sqrt(t)


@compare_equal
def test_shape(t: Tensor):
    return t.shape


@compare_all
def test_reshape(t: Tensor):
    shape = (1,) + t.shape + (1,)
    return ep.reshape(t, shape)


@compare_all
def test_clip(t: Tensor):
    return ep.clip(t, 2, 3.5)


@compare_all
def test_sign(t: Tensor):
    return ep.sign(t)


@compare_all
def test_sum(t: Tensor):
    return ep.sum(t)


@compare_all
def test_sum_axis(t: Tensor):
    return ep.sum(t, axis=0)


@compare_all
def test_sum_keepdims(t: Tensor):
    return ep.sum(t, axis=0, keepdims=True)


@compare_all
def test_sum_none_keepdims(t: Tensor):
    return ep.sum(t, axis=None, keepdims=True)


@compare_all
def test_mean(t: Tensor):
    return ep.mean(t)


@compare_all
def test_mean_axis(t: Tensor):
    return ep.mean(t, axis=0)


@compare_all
def test_mean_keepdims(t: Tensor):
    return ep.mean(t, axis=0, keepdims=True)


@compare_all
def test_mean_none_keepdims(t: Tensor):
    return ep.mean(t, axis=None, keepdims=True)


@compare_all
def test_all(t: Tensor):
    return ep.all(t > 3)


@compare_all
def test_all_axis(t: Tensor):
    return ep.all(t > 3, axis=0)


@compare_all
def test_all_keepdims(t: Tensor):
    return ep.all(t > 3, axis=0, keepdims=True)


@compare_all
def test_any(t: Tensor):
    return ep.any(t > 3)


@compare_all
def test_any_axis(t: Tensor):
    return ep.any(t > 3, axis=0)


@compare_all
def test_any_keepdims(t: Tensor):
    return ep.any(t > 3, axis=0, keepdims=True)


@compare_all
def test_min(t: Tensor):
    return ep.min(t)


@compare_all
def test_min_axis(t: Tensor):
    return ep.min(t, axis=0)


@compare_all
def test_min_keepdims(t: Tensor):
    return ep.min(t, axis=0, keepdims=True)


@compare_all
def test_max(t: Tensor):
    return ep.max(t)


@compare_all
def test_max_axis(t: Tensor):
    return ep.max(t, axis=0)


@compare_all
def test_max_keepdims(t: Tensor):
    return ep.max(t, axis=0, keepdims=True)


@compare_allclose
def test_exp(t: Tensor):
    return ep.exp(t)


@compare_allclose
def test_log(t: Tensor):
    return ep.log(t.maximum(1e-8))


@compare_allclose
def test_log2(t: Tensor):
    return ep.log2(t.maximum(1e-8))


@compare_allclose
def test_log10(t: Tensor):
    return ep.log10(t.maximum(1e-8))


@compare_allclose
def test_log1p(t: Tensor):
    return ep.log1p(t)


@compare_allclose(rtol=1e-6)
def test_tanh(t: Tensor):
    return ep.tanh(t)


@compare_allclose(rtol=1e-6)
def test_arctanh(t: Tensor):
    return ep.arctanh((t - t.mean()) / t.max())


@compare_all
def test_abs_op(t: Tensor):
    return abs(t)


@compare_all
def test_abs(t: Tensor):
    return ep.abs(t)


@compare_all
def test_minimum(t1: Tensor, t2: Tensor):
    return ep.minimum(t1, t2)


@compare_all
def test_minimum_scalar(t: Tensor):
    return ep.minimum(t, 3)


@compare_all
def test_rminimum_scalar(t: Tensor):
    return ep.minimum(3, t)


@compare_all
def test_maximum(t1: Tensor, t2: Tensor):
    return ep.maximum(t1, t2)


@compare_all
def test_maximum_scalar(t: Tensor):
    return ep.maximum(t, 3)


@compare_all
def test_rmaximum_scalar(t: Tensor):
    return ep.maximum(3, t)


@compare_all
def test_argmin(t: Tensor):
    return ep.argmin(t)


@compare_all
def test_argmin_axis(t: Tensor):
    return ep.argmin(t, axis=0)


@compare_all
def test_argmax(t: Tensor):
    return ep.argmax(t)


@compare_all
def test_argmax_axis(t: Tensor):
    return ep.argmax(t, axis=0)


@compare_all
def test_logical_and(t: Tensor):
    return ep.logical_and(t < 3, t > 1)


@compare_all
def test_logical_and_scalar(t: Tensor):
    return ep.logical_and(True, t < 3)


@compare_all
def test_logical_or(t: Tensor):
    return ep.logical_or(t > 3, t < 1)


@compare_all
def test_logical_or_scalar(t: Tensor):
    return ep.logical_or(True, t < 1)


@compare_all
def test_logical_not(t: Tensor):
    return ep.logical_not(t > 3)


@compare_all
def test_isnan_false(t: Tensor):
    return ep.isnan(t)


@compare_all
def test_isnan_true(t: Tensor):
    return ep.isnan(t + ep.nan)


@compare_all
def test_isinf(t: Tensor):
    return ep.isinf(t)


@compare_all
def test_isinf_posinf(t: Tensor):
    return ep.isinf(t + ep.inf)


@compare_all
def test_isinf_neginf(t: Tensor):
    return ep.isinf(t - ep.inf)


@compare_all
def test_zeros_like(t: Tensor):
    return ep.zeros_like(t)


@compare_all
def test_ones_like(t: Tensor):
    return ep.ones_like(t)


@compare_all
def test_full_like(t: Tensor):
    return ep.full_like(t, 5)


@pytest.mark.parametrize("value", [1, -1, 2])
@compare_all
def test_onehot_like(dummy: Tensor, value):
    t = ep.arange(dummy, 18).float32().reshape((6, 3))
    indices = ep.arange(t, 6) // 2
    return ep.onehot_like(t, indices, value=value)


@compare_all
def test_zeros_scalar(t: Tensor):
    return ep.zeros(t, 5)


@compare_all
def test_zeros_tuple(t: Tensor):
    return ep.zeros(t, (2, 3))


@compare_all
def test_ones_scalar(t: Tensor):
    return ep.ones(t, 5)


@compare_all
def test_ones_tuple(t: Tensor):
    return ep.ones(t, (2, 3))


@compare_all
def test_full_scalar(t: Tensor):
    return ep.full(t, 5, 4.0)


@compare_all
def test_full_tuple(t: Tensor):
    return ep.full(t, (2, 3), 4.0)


@compare_equal
def test_uniform_scalar(t: Tensor):
    return ep.uniform(t, 5).shape


@compare_equal
def test_uniform_tuple(t: Tensor):
    return ep.uniform(t, (2, 3)).shape


@compare_equal
def test_normal_scalar(t: Tensor):
    return ep.normal(t, 5).shape


@compare_equal
def test_normal_tuple(t: Tensor):
    return ep.normal(t, (2, 3)).shape


@compare_all
def test_argsort(t: Tensor):
    return ep.argsort(t)


@compare_all
def test_transpose(dummy: Tensor):
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    return ep.transpose(t)


@compare_all
def test_transpose_axes(dummy: Tensor):
    t = ep.arange(dummy, 60).float32().reshape((3, 4, 5))
    return ep.transpose(t, axes=(1, 2, 0))


@compare_all
def test_where(t: Tensor):
    return ep.where(t >= 3, t, -t)


@compare_all
def test_tile(t: Tensor):
    return ep.tile(t, (3,) * t.ndim)


@compare_all
def test_matmul(dummy: Tensor):
    t = ep.arange(dummy, 8).float32().reshape((2, 4))
    return ep.matmul(t, t.T)


@compare_allclose(rtol=1e-6)
def test_softmax(t: Tensor):
    return ep.softmax(t)


@compare_allclose(rtol=1e-5)
def test_log_softmax(t: Tensor):
    return ep.log_softmax(t)


@compare_allclose
def test_crossentropy(dummy: Tensor):
    t = ep.arange(dummy, 50).reshape((10, 5)).float32()
    t = t / t.max()
    return ep.crossentropy(t, t.argmax(axis=-1))


@pytest.mark.parametrize("axis", [0, 1, -1])
@compare_all
def test_stack(t1: Tensor, t2, axis):
    return ep.stack([t1, t2], axis=axis)


@compare_all
def test_concatenate_axis0(dummy: Tensor):
    t1 = ep.arange(dummy, 12).float32().reshape((4, 3))
    t2 = ep.arange(dummy, 20, 32, 2).float32().reshape((2, 3))
    return ep.concatenate([t1, t2], axis=0)


@compare_all
def test_concatenate_axis1(dummy: Tensor):
    t1 = ep.arange(dummy, 12).float32().reshape((3, 4))
    t2 = ep.arange(dummy, 20, 32, 2).float32().reshape((3, 2))
    return ep.concatenate([t1, t2], axis=1)


@pytest.mark.parametrize("axis", [0, 1, -1])
@compare_all
def test_expand_dims(t: Tensor, axis):
    return ep.expand_dims(t, axis)


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
@compare_all
def test_squeeze(t: Tensor, axis):
    t = t.expand_dims(axis=0).expand_dims(axis=1)
    return ep.squeeze(t, axis=axis)


@compare_all
def test_arange(dummy: Tensor):
    return ep.arange(dummy, 6)


@compare_all
def test_arange_start(dummy: Tensor):
    return ep.arange(dummy, 5, 10)


@compare_all
def test_arange_step(dummy: Tensor):
    return ep.arange(dummy, 4, 8, 2)


@compare_all
def test_cumsum(t: Tensor):
    return ep.cumsum(t)


@compare_all
def test_cumsum_axis(t: Tensor):
    return ep.cumsum(t, axis=0)


@compare_all
def test_flip(t: Tensor):
    return ep.flip(t)


@compare_all
def test_flip_axis(t: Tensor):
    return ep.flip(t, axis=0)


@pytest.mark.parametrize("indexing", ["ij", "xy"])
@pytest.mark.parametrize("i", [0, 1])
@compare_all
def test_meshgrid_a(dummy: Tensor, indexing, i):
    t1 = ep.arange(dummy, 5)
    t2 = ep.arange(dummy, 3)
    results = ep.meshgrid(t1, t2, indexing=indexing)
    assert len(results) == 2
    return results[i]


@pytest.mark.parametrize(
    "mode,value", [("constant", 0), ("constant", -2), ("reflect", 0)]
)
@compare_all
def test_pad(dummy: Tensor, mode, value):
    t = ep.arange(dummy, 120).reshape((2, 3, 4, 5)).float32()
    return ep.pad(t, ((0, 0), (0, 0), (2, 3), (1, 2)), mode=mode, value=value)


@compare_all
def test_index_update_row(dummy: Tensor):
    x = ep.ones(dummy, (3, 4))
    return ep.index_update(x, ep.index[1], ep.ones(x, 4) * 66.0)


@compare_all
def test_index_update_column(dummy: Tensor):
    x = ep.ones(dummy, (3, 4))
    return ep.index_update(x, ep.index[:, 1], ep.ones(x, 3) * 66.0)


@compare_all
def test_index_update_indices(dummy: Tensor):
    x = ep.ones(dummy, (3, 4))
    ind = ep.from_numpy(dummy, np.array([0, 1, 2, 1]))
    return ep.index_update(x, ep.index[ind, ep.arange(x, 4)], ep.ones(x, 4) * 33.0)


@compare_all
def test_lt(t1: Tensor, t2: Tensor):
    return t1 < t2


@compare_all
def test_le(t1: Tensor, t2: Tensor):
    return t1 <= t2


@compare_all
def test_gt(t1: Tensor, t2: Tensor):
    return t1 > t2


@compare_all
def test_ge(t1: Tensor, t2: Tensor):
    return t1 >= t2


@compare_all
def test_eq(t1: Tensor, t2: Tensor):
    return t1 == t2


@compare_all
def test_ne(t1: Tensor, t2: Tensor):
    return t1 != t2


@compare_all
def test_float_int_lt(t1: Tensor, t2int: Tensor):
    return t1 < t2int


@compare_all
def test_float_int_le(t1: Tensor, t2int: Tensor):
    return t1 <= t2int


@compare_all
def test_float_int_gt(t1: Tensor, t2int: Tensor):
    return t1 > t2int


@compare_all
def test_float_int_ge(t1: Tensor, t2int: Tensor):
    return t1 >= t2int


@compare_all
def test_float_int_eq(t1: Tensor, t2int: Tensor):
    return t1 == t2int


@compare_all
def test_float_int_ne(t1: Tensor, t2int: Tensor):
    return t1 != t2int


@compare_all
def test_int_float_lt(t1int: Tensor, t2: Tensor):
    return t1int < t2


@compare_all
def test_int_float_le(t1int: Tensor, t2: Tensor):
    return t1int <= t2


@compare_all
def test_int_float_gt(t1int: Tensor, t2: Tensor):
    return t1int > t2


@compare_all
def test_int_float_ge(t1int: Tensor, t2: Tensor):
    return t1int >= t2


@compare_all
def test_int_float_eq(t1int: Tensor, t2: Tensor):
    return t1int == t2


@compare_all
def test_int_float_ne(t1int: Tensor, t2: Tensor):
    return t1int != t2
